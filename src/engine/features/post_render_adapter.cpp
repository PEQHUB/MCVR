#include "post_render_adapter.hpp"
#include "color_to_depth_adapter.hpp"
#include "app/engine_services.hpp"
#include "frame/frame_scheduler.hpp"
#include "platform/vulkan/vk2_device.hpp"
#include "scene/scene_resource_service.hpp"
#include "scene/texture_service.hpp"
#include "rendergraph/graph_builder.hpp"
#include "diagnostics/log.hpp"

#include "common/shared.hpp"

#include <volk.h>
#include <cstring>

namespace engine {

namespace {

// Insert a pipeline barrier for an image layout transition.
void imageBarrier(VkCommandBuffer cmd, VkImage image,
                  VkImageLayout oldLayout, VkImageLayout newLayout,
                  VkAccessFlags2 srcAccess, VkPipelineStageFlags2 srcStage,
                  VkAccessFlags2 dstAccess, VkPipelineStageFlags2 dstStage,
                  VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT) {
    VkImageMemoryBarrier2 barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    barrier.srcStageMask = srcStage;
    barrier.srcAccessMask = srcAccess;
    barrier.dstStageMask = dstStage;
    barrier.dstAccessMask = dstAccess;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.image = image;
    barrier.subresourceRange = {aspect, 0, 1, 0, 1};

    VkDependencyInfo dep{};
    dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.imageMemoryBarrierCount = 1;
    dep.pImageMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(cmd, &dep);
}

} // namespace

void PostRenderAdapter::registerPass(GraphBuilder& builder, ResourceHandle colorInputHandle,
                                      ResourceHandle c2dOrderingHandle) {
    inputHandle_ = colorInputHandle;

    // The post-render pass reads the color image (tone-mapped + sharpened) and
    // the depth image (from ColorToDepthAdapter). It outputs the composited color
    // with entities rasterized on top.
    //
    // The color input is read-modify-write: entities are blended over it.
    // We declare the same image as both input and output. Since the graph tracks
    // this as an in-place modification, downstream passes see the updated image.
    //
    // The c2dOrderingHandle is a dummy resource from ColorToDepthAdapter that
    // ensures C2D executes before PostRender (the actual depth image is adapter-owned).
    outputHandle_ = colorInputHandle;  // in-place modification

    PassDescriptor pass;
    pass.name = "post_render";
    pass.queue = QueueAffinity::Graphics;
    pass.inputs.push_back(colorInputHandle);
    pass.inputs.push_back(c2dOrderingHandle);  // ordering: C2D before PostRender
    pass.outputs.push_back(colorInputHandle);  // in-place
    pass.execute = [this](const FrameContext& ctx, const PassResources& res) {
        execute(ctx, res);
    };
    builder.addPass(std::move(pass));
}

void PostRenderAdapter::init(EngineServices& services) {
    services_ = &services;
    device_ = services.device().device();

    std::string shaderBase = services.resourceDir() + "/shaders/";

    // Load vertex shader
    {
        auto r = vk2::ShaderModule::fromFile(device_, shaderBase + "world/v2_post/v2_world_post_vert.spv");
        if (!r) {
            log::error("post-render", "Failed to load v2_world_post_vert.spv: " + r.error().message);
            return;
        }
        vertShader_ = std::move(r.value());
    }

    // Load fragment shader
    {
        auto r = vk2::ShaderModule::fromFile(device_, shaderBase + "world/v2_post/v2_world_post_frag.spv");
        if (!r) {
            log::error("post-render", "Failed to load v2_world_post_frag.spv: " + r.error().message);
            return;
        }
        fragShader_ = std::move(r.value());
    }

    // Descriptor set layout: binding 0 = combined image sampler (texture atlas)
    {
        VkDescriptorSetLayoutBinding binding{};
        binding.binding = 0;
        binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        binding.descriptorCount = 1;
        binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        ci.bindingCount = 1;
        ci.pBindings = &binding;
        if (vkCreateDescriptorSetLayout(device_, &ci, nullptr, &texDescSetLayout_) != VK_SUCCESS) {
            log::error("post-render", "Descriptor set layout creation failed");
            return;
        }
    }

    // Pipeline layout: 1 descriptor set (texture atlas) + push constants (viewProj, view, cameraPos)
    {
        VkPushConstantRange pcRange{};
        pcRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        pcRange.offset = 0;
        pcRange.size = sizeof(PushConstants);

        vk2::PipelineLayoutDesc pld;
        pld.setLayouts = {texDescSetLayout_};
        pld.pushConstants = {pcRange};
        auto r = vk2::PipelineLayout::create(device_, pld);
        if (!r) {
            log::error("post-render", "Pipeline layout creation failed: " + r.error().message);
            return;
        }
        pipelineLayout_ = std::move(r.value());
    }

    // Graphics pipeline: PBRTriangle vertex input, alpha blending, depth test (no write)
    {
        VkPipelineShaderStageCreateInfo stages[2]{};
        stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        stages[0].module = vertShader_.handle();
        stages[0].pName = "main";
        stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[1].module = fragShader_.handle();
        stages[1].pName = "main";

        // --- PBRTriangle vertex input (96-byte stride, 12 attributes) ---
        VkVertexInputBindingDescription vertexBinding{};
        vertexBinding.binding = 0;
        vertexBinding.stride = 96;  // sizeof(PBRTriangle)
        vertexBinding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        // All attribute offsets match PBRTriangle layout from common/shared.hpp:
        //   pos(0..11), flags(12..15), norm(16..27), albedoEmission(28..31),
        //   colorLayer(32..47), postBase(48..59), emissiveBlockType(60..63),
        //   textureUV(64..71), glintUV(72..79), textureID(80..83),
        //   glintTexture(84..87), overlayPacked(88..91), lightPacked(92..95)
        VkVertexInputAttributeDescription attrs[12]{};
        // location 0: pos (vec3, offset 0)
        attrs[0]  = {0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0};
        // location 1: flags (uint, offset 12)
        attrs[1]  = {1, 0, VK_FORMAT_R32_UINT, 12};
        // location 2: norm (vec3, offset 16)
        attrs[2]  = {2, 0, VK_FORMAT_R32G32B32_SFLOAT, 16};
        // location 3: albedoEmission (float, offset 28)
        attrs[3]  = {3, 0, VK_FORMAT_R32_SFLOAT, 28};
        // location 4: colorLayer (vec4, offset 32)
        attrs[4]  = {4, 0, VK_FORMAT_R32G32B32A32_SFLOAT, 32};
        // location 5: postBase (vec4, offset 48 — postBase.xyz + emissiveBlockType packed as .w)
        attrs[5]  = {5, 0, VK_FORMAT_R32G32B32A32_SFLOAT, 48};
        // location 6: textureUV (vec2, offset 64)
        attrs[6]  = {6, 0, VK_FORMAT_R32G32_SFLOAT, 64};
        // location 7: glintUV (vec2, offset 72)
        attrs[7]  = {7, 0, VK_FORMAT_R32G32_SFLOAT, 72};
        // location 8: textureID (uint, offset 80)
        attrs[8]  = {8, 0, VK_FORMAT_R32_UINT, 80};
        // location 9: glintTexture (uint, offset 84)
        attrs[9]  = {9, 0, VK_FORMAT_R32_UINT, 84};
        // location 10: overlayPacked (uint, offset 88)
        attrs[10] = {10, 0, VK_FORMAT_R32_UINT, 88};
        // location 11: lightPacked (uint, offset 92)
        attrs[11] = {11, 0, VK_FORMAT_R32_UINT, 92};

        VkPipelineVertexInputStateCreateInfo vertexInput{};
        vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInput.vertexBindingDescriptionCount = 1;
        vertexInput.pVertexBindingDescriptions = &vertexBinding;
        vertexInput.vertexAttributeDescriptionCount = 12;
        vertexInput.pVertexAttributeDescriptions = attrs;

        VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkPipelineViewportStateCreateInfo viewportState{};
        viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.scissorCount = 1;

        VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.cullMode = VK_CULL_MODE_NONE;  // Entities have arbitrary winding
        rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rasterizer.lineWidth = 1.0f;

        VkPipelineMultisampleStateCreateInfo multisample{};
        multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        // Depth test LESS_OR_EQUAL against RT depth, but NO depth writes.
        // Entities are tested against the world depth but don't occlude each other.
        VkPipelineDepthStencilStateCreateInfo depthStencil{};
        depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depthStencil.depthTestEnable = VK_TRUE;
        depthStencil.depthWriteEnable = VK_FALSE;
        depthStencil.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

        // Alpha blending: srcAlpha / oneMinusSrcAlpha
        VkPipelineColorBlendAttachmentState blendAttachment{};
        blendAttachment.blendEnable = VK_TRUE;
        blendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        blendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        blendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
        blendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        blendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        blendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;
        blendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                                       | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

        VkPipelineColorBlendStateCreateInfo colorBlend{};
        colorBlend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlend.attachmentCount = 1;
        colorBlend.pAttachments = &blendAttachment;

        VkDynamicState dynamicStates[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dynamicState{};
        dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamicState.dynamicStateCount = 2;
        dynamicState.pDynamicStates = dynamicStates;

        // Dynamic rendering: color R8G8B8A8_UNORM + depth D32_SFLOAT
        VkFormat colorFormat = VK_FORMAT_R8G8B8A8_UNORM;
        VkPipelineRenderingCreateInfo renderingInfo{};
        renderingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
        renderingInfo.colorAttachmentCount = 1;
        renderingInfo.pColorAttachmentFormats = &colorFormat;
        renderingInfo.depthAttachmentFormat = VK_FORMAT_D32_SFLOAT;

        VkGraphicsPipelineCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        ci.pNext = &renderingInfo;
        ci.stageCount = 2;
        ci.pStages = stages;
        ci.pVertexInputState = &vertexInput;
        ci.pInputAssemblyState = &inputAssembly;
        ci.pViewportState = &viewportState;
        ci.pRasterizationState = &rasterizer;
        ci.pMultisampleState = &multisample;
        ci.pDepthStencilState = &depthStencil;
        ci.pColorBlendState = &colorBlend;
        ci.pDynamicState = &dynamicState;
        ci.layout = pipelineLayout_.handle();

        VkResult vr = vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &ci, nullptr, &pipeline_);
        if (vr != VK_SUCCESS) {
            log::error("post-render", "Graphics pipeline creation failed");
            return;
        }
    }

    // Descriptor allocator (1 set per frame: texture atlas sampler)
    {
        std::vector<VkDescriptorPoolSize> sizes = {
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1},
        };
        uint32_t fif = services.frame().framesInFlight();
        auto r = descAllocator_.init(device_, fif, sizes, 1);
        if (!r) {
            log::error("post-render", "Descriptor allocator init failed: " + r.error().message);
            return;
        }
    }

    initialized_ = true;
    log::info("post-render", "PostRenderAdapter initialized");
}

void PostRenderAdapter::configure(const ConfigSnapshot& /*config*/) {
    // No configurable parameters yet
}

void PostRenderAdapter::setPendingDrawData(CmdEntityPostDraw data) {
    pendingDraw_ = std::move(data);
    hasPendingDraw_ = true;
}

void PostRenderAdapter::ensureBuffers(VkDeviceSize vertexBytes, VkDeviceSize indexBytes) {
    // Round up to avoid frequent reallocations
    auto roundUp = [](VkDeviceSize size, VkDeviceSize granularity) {
        return ((size + granularity - 1) / granularity) * granularity;
    };

    if (vertexBytes > vertexBufferCapacity_) {
        vertexBuffer_ = {};  // destroy old
        VkDeviceSize newCap = roundUp(vertexBytes, 256 * 1024);  // 256 KB granularity
        if (newCap < 512 * 1024) newCap = 512 * 1024;  // minimum 512 KB

        vk2::Buffer::Desc desc{};
        desc.size = newCap;
        desc.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
        desc.vmaFlags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
                      | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        desc.memoryUsage = VMA_MEMORY_USAGE_AUTO;

        auto r = vk2::Buffer::create(device_, services_->device().vma(), desc);
        if (!r) {
            log::error("post-render", "Vertex buffer creation failed: " + r.error().message);
            return;
        }
        vertexBuffer_ = std::move(r.value());
        vertexBufferCapacity_ = newCap;
    }

    if (indexBytes > indexBufferCapacity_) {
        indexBuffer_ = {};
        VkDeviceSize newCap = roundUp(indexBytes, 64 * 1024);  // 64 KB granularity
        if (newCap < 128 * 1024) newCap = 128 * 1024;  // minimum 128 KB

        vk2::Buffer::Desc desc{};
        desc.size = newCap;
        desc.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
        desc.vmaFlags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
                      | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        desc.memoryUsage = VMA_MEMORY_USAGE_AUTO;

        auto r = vk2::Buffer::create(device_, services_->device().vma(), desc);
        if (!r) {
            log::error("post-render", "Index buffer creation failed: " + r.error().message);
            return;
        }
        indexBuffer_ = std::move(r.value());
        indexBufferCapacity_ = newCap;
    }
}

void PostRenderAdapter::execute(const FrameContext& ctx, const PassResources& resources) {
    if (!initialized_ || !resources.resolved || resources.cmd == VK_NULL_HANDLE) return;

    // Skip if no pending draw data or no draw calls
    if (!hasPendingDraw_ || pendingDraw_.drawCalls.empty()
        || pendingDraw_.vertexData.empty() || pendingDraw_.indexData.empty()) {
        hasPendingDraw_ = false;
        return;
    }

    // Need color output from graph + depth from C2D adapter
    if (resources.outputs.empty() || !resources.outputs[0].valid()) return;
    if (!c2dAdapter_ || c2dAdapter_->depthImage().handle() == VK_NULL_HANDLE) return;

    // Need texture atlas for sampling
    auto& texSvc = services_->texture();
    if (!texSvc.isFinalized()) return;

    VkCommandBuffer cmd = resources.cmd;
    uint32_t outIdx = resources.outputs[0].index;

    auto& pool = services_->frame().resourcePool();
    uint32_t width = pool.image(outIdx).width();
    uint32_t height = pool.image(outIdx).height();
    if (width == 0 || height == 0) return;

    // Upload vertex and index data to host-visible GPU buffers
    VkDeviceSize vertexBytes = static_cast<VkDeviceSize>(pendingDraw_.vertexData.size());
    VkDeviceSize indexBytes = static_cast<VkDeviceSize>(pendingDraw_.indexData.size() * sizeof(uint32_t));
    ensureBuffers(vertexBytes, indexBytes);
    if (vertexBuffer_.handle() == VK_NULL_HANDLE || indexBuffer_.handle() == VK_NULL_HANDLE) {
        hasPendingDraw_ = false;
        return;
    }

    std::memcpy(vertexBuffer_.mappedPtr(), pendingDraw_.vertexData.data(), vertexBytes);
    vertexBuffer_.flush();
    std::memcpy(indexBuffer_.mappedPtr(), pendingDraw_.indexData.data(), indexBytes);
    indexBuffer_.flush();

    // Allocate and write descriptor set for texture atlas
    descAllocator_.resetFrame(ctx.frameIndex);
    auto setResult = descAllocator_.allocate(texDescSetLayout_, ctx.frameIndex);
    if (!setResult) {
        hasPendingDraw_ = false;
        return;
    }
    VkDescriptorSet texDescSet = setResult.value();

    // Write texture atlas sampler (albedo array from TextureService)
    vk2::descriptor::writeCombinedImageSampler(
        device_, texDescSet, 0,
        texSvc.albedoArrayView(),
        texSvc.arrayNearestSampler(),
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    // Transition depth image to DEPTH_READ_ONLY_OPTIMAL for depth testing
    // (C2D wrote it as DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
    imageBarrier(cmd, c2dAdapter_->depthImage().handle(),
                 VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                 VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL,
                 VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                 VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                 VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT,
                 VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
                 VK_IMAGE_ASPECT_DEPTH_BIT);
    c2dAdapter_->depthImage().setLayout(VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL);

    // Transition color image GENERAL → COLOR_ATTACHMENT_OPTIMAL before dynamic rendering.
    // BarrierPlanner sets the output image to VK_IMAGE_LAYOUT_GENERAL (layoutForOutput for
    // a Graphics pass uses GENERAL for storage writes). vkCmdBeginRendering requires
    // the attachment to be in COLOR_ATTACHMENT_OPTIMAL — using GENERAL in the
    // VkRenderingAttachmentInfo without this transition triggers VUID-VkRenderingAttachmentInfo
    // -imageLayout and causes undefined (black) rendering output.
    imageBarrier(cmd, resources.resolved->images[outIdx],
                 VK_IMAGE_LAYOUT_GENERAL,
                 VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                 VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                 VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                 VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                 VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT);

    // Begin dynamic rendering: color LOAD + depth LOAD (read-only)
    VkRenderingAttachmentInfo colorAttachment{};
    colorAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    colorAttachment.imageView = resources.resolved->views[outIdx];
    colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

    VkRenderingAttachmentInfo depthAttachment{};
    depthAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    depthAttachment.imageView = c2dAdapter_->depthImage().defaultView();
    depthAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL;
    depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_NONE;

    VkRenderingInfo renderInfo{};
    renderInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    renderInfo.renderArea = {{0, 0}, {width, height}};
    renderInfo.layerCount = 1;
    renderInfo.colorAttachmentCount = 1;
    renderInfo.pColorAttachments = &colorAttachment;
    renderInfo.pDepthAttachment = &depthAttachment;

    vkCmdBeginRendering(cmd, &renderInfo);

    // Set viewport and scissor
    VkViewport viewport{};
    viewport.width = static_cast<float>(width);
    viewport.height = static_cast<float>(height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &viewport);

    VkRect2D scissor{{0, 0}, {width, height}};
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    // Bind pipeline
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);

    // Bind texture descriptor set (set 0)
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout_.handle(),
                            0, 1, &texDescSet, 0, nullptr);

    // Push constants: viewProj and view matrices from camera
    PushConstants pc{};
    if (ctx.camera.valid) {
        // Compute viewProj = projection * view
        // Both are column-major 4x4. Multiply manually.
        const float* V = ctx.camera.view;
        const float* P = ctx.camera.projection;
        for (int c = 0; c < 4; ++c) {
            for (int r = 0; r < 4; ++r) {
                float sum = 0.0f;
                for (int k = 0; k < 4; ++k) {
                    sum += P[k * 4 + r] * V[c * 4 + k];
                }
                pc.viewProj[c * 4 + r] = sum;
            }
        }
        std::memcpy(pc.view, ctx.camera.view, sizeof(pc.view));
        pc.cameraPos[0] = ctx.camera.posX;
        pc.cameraPos[1] = ctx.camera.posY;
        pc.cameraPos[2] = ctx.camera.posZ;
        pc.cameraPos[3] = 0.0f;
    }
    vkCmdPushConstants(cmd, pipelineLayout_.handle(), VK_SHADER_STAGE_VERTEX_BIT,
                       0, sizeof(PushConstants), &pc);

    // Bind vertex and index buffers
    VkBuffer vb = vertexBuffer_.handle();
    VkDeviceSize vbOffset = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &vb, &vbOffset);
    vkCmdBindIndexBuffer(cmd, indexBuffer_.handle(), 0, VK_INDEX_TYPE_UINT32);

    // Issue draw calls
    for (const auto& dc : pendingDraw_.drawCalls) {
        if (dc.indexCount == 0) continue;
        // vertexByteOffset is in bytes; divide by stride for firstVertex
        uint32_t firstVertex = dc.vertexByteOffset / 96;
        vkCmdDrawIndexed(cmd, dc.indexCount, 1, dc.indexElementOffset, static_cast<int32_t>(firstVertex), 0);
    }

    vkCmdEndRendering(cmd);

    // Transition color image back to GENERAL so downstream passes (StarField, composite)
    // see the layout that BarrierPlanner expects (GENERAL for a Graphics-queue output).
    // Without this the StarField adapter's GENERAL→COLOR_ATTACHMENT_OPTIMAL barrier would
    // declare the wrong oldLayout, causing a validation error and a potential black screen.
    imageBarrier(cmd, resources.resolved->images[outIdx],
                 VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                 VK_IMAGE_LAYOUT_GENERAL,
                 VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                 VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                 VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                 VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT);

    // Diagnostic: log periodically
    static uint64_t lastLogFrame = 0;
    if (ctx.frameNumber - lastLogFrame >= 300) {
        uint32_t totalIndices = 0;
        for (const auto& dc : pendingDraw_.drawCalls) totalIndices += dc.indexCount;
        log::info("post-render", "frame=" + std::to_string(ctx.frameNumber)
            + " drawCalls=" + std::to_string(pendingDraw_.drawCalls.size())
            + " indices=" + std::to_string(totalIndices)
            + " vbytes=" + std::to_string(vertexBytes)
            + " ibytes=" + std::to_string(indexBytes));
        lastLogFrame = ctx.frameNumber;
    }

    // Consume the pending data
    hasPendingDraw_ = false;
}

void PostRenderAdapter::destroyPipeline() {
    if (pipeline_ != VK_NULL_HANDLE && device_ != VK_NULL_HANDLE) {
        vkDestroyPipeline(device_, pipeline_, nullptr);
        pipeline_ = VK_NULL_HANDLE;
    }
}

void PostRenderAdapter::shutdown() {
    if (device_ == VK_NULL_HANDLE) return;

    vertexBuffer_ = {};
    indexBuffer_ = {};
    vertexBufferCapacity_ = 0;
    indexBufferCapacity_ = 0;

    destroyPipeline();
    pipelineLayout_ = {};
    vertShader_ = {};
    fragShader_ = {};

    descAllocator_.shutdown();
    if (texDescSetLayout_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device_, texDescSetLayout_, nullptr);
        texDescSetLayout_ = VK_NULL_HANDLE;
    }

    hasPendingDraw_ = false;

    initialized_ = false;
    log::info("post-render", "PostRenderAdapter shut down");
}

} // namespace engine
