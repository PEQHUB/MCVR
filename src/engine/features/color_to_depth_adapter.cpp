#include "color_to_depth_adapter.hpp"
#include "app/engine_services.hpp"
#include "frame/frame_scheduler.hpp"
#include "platform/vulkan/vk2_device.hpp"
#include "rendergraph/graph_builder.hpp"
#include "diagnostics/log.hpp"

#include <volk.h>

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

void ColorToDepthAdapter::registerPass(GraphBuilder& builder, ResourceHandle firstHitDepthHandle) {
    inputHandle_ = firstHitDepthHandle;

    // Create a dummy ordering resource so downstream passes (PostRender, StarField)
    // can declare a dependency on C2D completing first.
    ImageResourceDesc orderDesc;
    orderDesc.name = "c2d_ordering_token";
    orderDesc.format = VK_FORMAT_R8_UINT;  // dummy, never allocated
    orderDesc.fixedWidth = 1;
    orderDesc.fixedHeight = 1;
    orderingHandle_ = builder.addResource(orderDesc);

    PassDescriptor pass;
    pass.name = "color_to_depth";
    pass.queue = QueueAffinity::Graphics;
    pass.inputs.push_back(firstHitDepthHandle);
    pass.outputs.push_back(orderingHandle_);  // ordering token for downstream passes
    pass.execute = [this](const FrameContext& ctx, const PassResources& res) {
        execute(ctx, res);
    };
    builder.addPass(std::move(pass));
}

void ColorToDepthAdapter::init(EngineServices& services) {
    services_ = &services;
    device_ = services.device().device();

    std::string shaderBase = services.resourceDir() + "/shaders/";

    // Load vertex shader
    {
        auto r = vk2::ShaderModule::fromFile(device_, shaderBase + "world/v2_post/v2_color_to_depth_vert.spv");
        if (!r) {
            log::error("c2d", "Failed to load v2_color_to_depth_vert.spv: " + r.error().message);
            return;
        }
        vertShader_ = std::move(r.value());
    }

    // Load fragment shader
    {
        auto r = vk2::ShaderModule::fromFile(device_, shaderBase + "world/v2_post/v2_color_to_depth_frag.spv");
        if (!r) {
            log::error("c2d", "Failed to load v2_color_to_depth_frag.spv: " + r.error().message);
            return;
        }
        fragShader_ = std::move(r.value());
    }

    // Descriptor set layout: binding 0 = storage image (firstHitDepth, r32f)
    {
        VkDescriptorSetLayoutBinding binding{};
        binding.binding = 0;
        binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        binding.descriptorCount = 1;
        binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        ci.bindingCount = 1;
        ci.pBindings = &binding;
        if (vkCreateDescriptorSetLayout(device_, &ci, nullptr, &descSetLayout_) != VK_SUCCESS) {
            log::error("c2d", "Descriptor set layout creation failed");
            return;
        }
    }

    // Pipeline layout (1 descriptor set, no push constants)
    {
        vk2::PipelineLayoutDesc pld;
        pld.setLayouts = {descSetLayout_};
        auto r = vk2::PipelineLayout::create(device_, pld);
        if (!r) {
            log::error("c2d", "Pipeline layout creation failed: " + r.error().message);
            return;
        }
        pipelineLayout_ = std::move(r.value());
    }

    // Graphics pipeline: depth-only (no color attachment), depth write enabled.
    // We can't use vk2::GraphicsPipeline because it requires a color attachment.
    // Build the pipeline manually using dynamic rendering (VkPipelineRenderingCreateInfo).
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

        VkPipelineVertexInputStateCreateInfo vertexInput{};
        vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

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
        rasterizer.cullMode = VK_CULL_MODE_NONE;
        rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rasterizer.lineWidth = 1.0f;

        VkPipelineMultisampleStateCreateInfo multisample{};
        multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineDepthStencilStateCreateInfo depthStencil{};
        depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depthStencil.depthTestEnable = VK_TRUE;
        depthStencil.depthWriteEnable = VK_TRUE;
        depthStencil.depthCompareOp = VK_COMPARE_OP_ALWAYS;  // Always write (fullscreen depth fill)

        // No color attachment — depth-only pass
        VkPipelineColorBlendStateCreateInfo colorBlend{};
        colorBlend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlend.attachmentCount = 0;
        colorBlend.pAttachments = nullptr;

        VkDynamicState dynamicStates[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dynamicState{};
        dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamicState.dynamicStateCount = 2;
        dynamicState.pDynamicStates = dynamicStates;

        // Dynamic rendering: depth-only, no color
        VkPipelineRenderingCreateInfo renderingInfo{};
        renderingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
        renderingInfo.colorAttachmentCount = 0;
        renderingInfo.pColorAttachmentFormats = nullptr;
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
            log::error("c2d", "Graphics pipeline creation failed");
            return;
        }
    }

    // Descriptor allocator (1 set per frame: the storage image binding)
    {
        std::vector<VkDescriptorPoolSize> sizes = {
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1},
        };
        uint32_t fif = services.frame().framesInFlight();
        auto r = descAllocator_.init(device_, fif, sizes, 1);
        if (!r) {
            log::error("c2d", "Descriptor allocator init failed: " + r.error().message);
            return;
        }
    }

    initialized_ = true;
    log::info("c2d", "ColorToDepthAdapter initialized");
}

void ColorToDepthAdapter::configure(const ConfigSnapshot& /*config*/) {
    // No configurable parameters
}

void ColorToDepthAdapter::ensureDepthImage(uint32_t width, uint32_t height) {
    if (lastWidth_ == width && lastHeight_ == height && depthImage_.handle() != VK_NULL_HANDLE) return;

    depthImage_ = {};

    vk2::Image::Desc desc{};
    desc.width = width;
    desc.height = height;
    desc.format = VK_FORMAT_D32_SFLOAT;
    desc.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
               | VK_IMAGE_USAGE_SAMPLED_BIT;

    auto r = vk2::Image::create(device_, services_->device().vma(), desc);
    if (!r) {
        log::error("c2d", "Failed to create depth image: " + r.error().message);
        return;
    }
    depthImage_ = std::move(r.value());
    lastWidth_ = width;
    lastHeight_ = height;

    log::info("c2d", "Depth image created: " + std::to_string(width) + "x" + std::to_string(height));
}

void ColorToDepthAdapter::execute(const FrameContext& ctx, const PassResources& resources) {
    if (!initialized_ || !resources.resolved || resources.cmd == VK_NULL_HANDLE) return;
    if (resources.inputs.empty() || !resources.inputs[0].valid()) return;

    VkCommandBuffer cmd = resources.cmd;
    uint32_t inIdx = resources.inputs[0].index;

    auto& pool = services_->frame().resourcePool();
    uint32_t width = pool.image(inIdx).width();
    uint32_t height = pool.image(inIdx).height();
    if (width == 0 || height == 0) return;

    // Ensure depth image exists at the right resolution
    ensureDepthImage(width, height);
    if (depthImage_.handle() == VK_NULL_HANDLE) return;

    descAllocator_.resetFrame(ctx.frameIndex);

    // Allocate and write descriptor set
    auto setResult = descAllocator_.allocate(descSetLayout_, ctx.frameIndex);
    if (!setResult) return;
    VkDescriptorSet descSet = setResult.value();

    VkDescriptorImageInfo imageInfo{};
    imageInfo.imageView = resources.resolved->views[inIdx];
    imageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = descSet;
    write.dstBinding = 0;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    write.pImageInfo = &imageInfo;
    vkUpdateDescriptorSets(device_, 1, &write, 0, nullptr);

    // Transition depth image: UNDEFINED -> DEPTH_STENCIL_ATTACHMENT_OPTIMAL
    imageBarrier(cmd, depthImage_.handle(),
                 VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                 0, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                 VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                 VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                 VK_IMAGE_ASPECT_DEPTH_BIT);
    depthImage_.setLayout(VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

    // Begin dynamic rendering (depth-only, no color)
    VkRenderingAttachmentInfo depthAttachment{};
    depthAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    depthAttachment.imageView = depthImage_.defaultView();
    depthAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    depthAttachment.clearValue.depthStencil = {1.0f, 0};

    VkRenderingInfo renderInfo{};
    renderInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    renderInfo.renderArea = {{0, 0}, {width, height}};
    renderInfo.layerCount = 1;
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

    // Bind pipeline and descriptors, draw fullscreen triangle
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout_.handle(),
                            0, 1, &descSet, 0, nullptr);
    vkCmdDraw(cmd, 3, 1, 0, 0);

    vkCmdEndRendering(cmd);
}

void ColorToDepthAdapter::destroyPipeline() {
    if (pipeline_ != VK_NULL_HANDLE && device_ != VK_NULL_HANDLE) {
        vkDestroyPipeline(device_, pipeline_, nullptr);
        pipeline_ = VK_NULL_HANDLE;
    }
}

void ColorToDepthAdapter::shutdown() {
    if (device_ == VK_NULL_HANDLE) return;

    depthImage_ = {};
    lastWidth_ = 0;
    lastHeight_ = 0;

    destroyPipeline();
    pipelineLayout_ = {};
    vertShader_ = {};
    fragShader_ = {};

    descAllocator_.shutdown();
    if (descSetLayout_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device_, descSetLayout_, nullptr);
        descSetLayout_ = VK_NULL_HANDLE;
    }

    initialized_ = false;
    log::info("c2d", "ColorToDepthAdapter shut down");
}

} // namespace engine
