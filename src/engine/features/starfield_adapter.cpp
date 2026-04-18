#include "starfield_adapter.hpp"
#include "color_to_depth_adapter.hpp"
#include "app/engine_services.hpp"
#include "frame/frame_scheduler.hpp"
#include "platform/vulkan/vk2_device.hpp"
#include "rendergraph/graph_builder.hpp"
#include "scene/scene_resource_service.hpp"
#include "diagnostics/log.hpp"

#include <volk.h>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>

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

void StarFieldAdapter::registerPass(GraphBuilder& builder, ResourceHandle colorInputHandle) {
    inputHandle_ = colorInputHandle;

    // Stars are additively blended over the color image in-place.
    // We declare the color image as input ONLY (not as output) to avoid
    // creating a graph cycle with PostRenderAdapter, which also reads/writes
    // the same resource. The graph ensures PostRender executes before StarField
    // because PostRender declares the resource as output, creating the edge
    // PostRender -> StarField via the writer->reader relationship.
    outputHandle_ = colorInputHandle;

    PassDescriptor pass;
    pass.name = "starfield";
    pass.queue = QueueAffinity::Graphics;
    pass.inputs.push_back(colorInputHandle);
    // NO output declaration — avoids bidirectional cycle with PostRender.
    // The pass still writes to the color image via dynamic rendering LOAD,
    // but the graph doesn't track it as a formal output.
    pass.execute = [this](const FrameContext& ctx, const PassResources& res) {
        execute(ctx, res);
    };
    builder.addPass(std::move(pass));
}

void StarFieldAdapter::generateStars() {
    // Procedural star generation: 3000 stars on a sphere of radius 400.
    // Each star is a billboard quad (2 triangles, 6 vertices).
    std::mt19937 rng(42);  // Fixed seed for deterministic star positions
    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

    std::vector<StarVertex> vertices;
    vertices.reserve(TOTAL_VERTICES);

    const float RADIUS = 400.0f;
    const float SUN_EXCLUSION_COS = std::cos(0.05f);   // No stars within ~2.9 degrees of +X (sun)
    const float MOON_EXCLUSION_COS = std::cos(0.08f);  // No stars within ~4.6 degrees of -X (moon)

    uint32_t generated = 0;
    while (generated < STAR_COUNT) {
        // Uniform sphere sampling via rejection
        float u = dist01(rng) * 2.0f - 1.0f;
        float v = dist01(rng) * 2.0f - 1.0f;
        float w = dist01(rng) * 2.0f - 1.0f;
        float len2 = u * u + v * v + w * w;
        if (len2 < 0.0001f || len2 > 1.0f) continue;

        float invLen = 1.0f / std::sqrt(len2);
        float nx = u * invLen;
        float ny = v * invLen;
        float nz = w * invLen;

        // Reject stars too close to sun (+X axis) or moon (-X axis)
        if (nx > SUN_EXCLUSION_COS) continue;
        if (nx < -MOON_EXCLUSION_COS) continue;  // -nx > MOON_EXCLUSION_COS

        // Star brightness
        float brightness;
        if (dist01(rng) < 0.35f) {
            // 35% bright stars: uniform [0.75, 1.0]
            brightness = 0.75f + dist01(rng) * 0.25f;
        } else {
            // 65% dim stars: cubic falloff [0.02, 0.37]
            float t = dist01(rng);
            brightness = 0.02f + (0.37f - 0.02f) * (t * t * t);
        }

        // Star color tinting
        float cr = 1.0f, cg = 1.0f, cb = 1.0f;
        float colorRoll = dist01(rng);
        if (colorRoll < 0.75f) {
            // 75% white — no tint
        } else if (colorRoll < 0.90f) {
            // 15% warm
            cr = 1.0f; cg = 0.95f; cb = 0.85f;
        } else {
            // 10% cool
            cr = 0.85f; cg = 0.9f; cb = 1.0f;
        }

        float r = cr * brightness;
        float g = cg * brightness;
        float b = cb * brightness;
        float a = brightness;  // Alpha encodes per-star brightness for visibility modulation

        // Billboard quad size: random [0.5, 0.7]
        float halfSize = (0.5f + dist01(rng) * 0.2f) * 0.5f;

        // Star center on the sphere
        float cx = nx * RADIUS;
        float cy = ny * RADIUS;
        float cz = nz * RADIUS;

        // Build billboard axes perpendicular to the direction from origin
        // Use the "up" vector trick for billboard orientation
        float ax, ay, az;  // right axis
        float bx, by, bz;  // up axis

        // Cross product with world up (0,1,0) to get right axis
        // right = normalize(cross(normal, up))
        float rx = nz;  // ny*0 - nz*0 ... cross(n, (0,1,0)) = (nz, 0, -nx)
        float ry = 0.0f;
        float rz = -nx;
        float rLen = std::sqrt(rx * rx + rz * rz);

        if (rLen < 0.001f) {
            // Normal is nearly parallel to up — use (1,0,0) as alternate up
            rx = 0.0f; ry = -nz; rz = ny;
            rLen = std::sqrt(ry * ry + rz * rz);
            if (rLen < 0.001f) { rx = 1.0f; ry = 0.0f; rz = 0.0f; rLen = 1.0f; }
        }
        float rInv = 1.0f / rLen;
        ax = rx * rInv * halfSize;
        ay = ry * rInv * halfSize;
        az = rz * rInv * halfSize;

        // up = normalize(cross(right_normalized, normal)) * halfSize
        float ux = (ry * rInv * nz - rz * rInv * ny) * halfSize;
        float uy = (rz * rInv * nx - rx * rInv * nz) * halfSize;
        float uz = (rx * rInv * ny - ry * rInv * nx) * halfSize;
        bx = ux; by = uy; bz = uz;

        // Quad corners: center +/- right +/- up
        // v0 = center - right - up  (bottom-left)
        // v1 = center + right - up  (bottom-right)
        // v2 = center + right + up  (top-right)
        // v3 = center - right + up  (top-left)
        // Two triangles: (v0, v1, v2), (v0, v2, v3)

        StarVertex v0 = {{cx - ax - bx, cy - ay - by, cz - az - bz}, {r, g, b, a}};
        StarVertex v1 = {{cx + ax - bx, cy + ay - by, cz + az - bz}, {r, g, b, a}};
        StarVertex v2 = {{cx + ax + bx, cy + ay + by, cz + az + bz}, {r, g, b, a}};
        StarVertex v3 = {{cx - ax + bx, cy - ay + by, cz - az + bz}, {r, g, b, a}};

        vertices.push_back(v0);
        vertices.push_back(v1);
        vertices.push_back(v2);
        vertices.push_back(v0);
        vertices.push_back(v2);
        vertices.push_back(v3);

        ++generated;
    }

    // Upload to device-local buffer via staging
    VkDeviceSize bufferSize = vertices.size() * sizeof(StarVertex);

    // Create staging buffer (host-visible)
    vk2::Buffer::Desc stagingDesc{};
    stagingDesc.size = bufferSize;
    stagingDesc.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    stagingDesc.vmaFlags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    stagingDesc.memoryUsage = VMA_MEMORY_USAGE_AUTO;

    auto stagingResult = vk2::Buffer::create(device_, services_->device().vma(), stagingDesc);
    if (!stagingResult) {
        log::error("starfield", "Failed to create staging buffer: " + stagingResult.error().message);
        return;
    }
    auto staging = std::move(stagingResult.value());
    std::memcpy(staging.mappedPtr(), vertices.data(), bufferSize);
    staging.flush(); // Ensure GPU-visible on non-coherent heaps before copy command

    // Create device-local vertex buffer
    vk2::Buffer::Desc vbDesc{};
    vbDesc.size = bufferSize;
    vbDesc.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    vbDesc.memoryUsage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

    auto vbResult = vk2::Buffer::create(device_, services_->device().vma(), vbDesc);
    if (!vbResult) {
        log::error("starfield", "Failed to create vertex buffer: " + vbResult.error().message);
        return;
    }
    vertexBuffer_ = std::move(vbResult.value());

    // One-shot command buffer for the staging copy
    VkCommandBuffer cmd = services_->frame().beginOneShotCommandBuffer();
    if (cmd == VK_NULL_HANDLE) {
        log::error("starfield", "Failed to begin one-shot command buffer for VB upload");
        vertexBuffer_ = {};
        return;
    }

    VkBufferCopy region{};
    region.size = bufferSize;
    vkCmdCopyBuffer(cmd, staging.handle(), vertexBuffer_.handle(), 1, &region);

    services_->frame().endOneShotCommandBuffer(cmd);

    log::info("starfield", "Star vertex buffer created: " + std::to_string(STAR_COUNT)
              + " stars, " + std::to_string(vertices.size()) + " vertices, "
              + std::to_string(bufferSize) + " bytes");
}

void StarFieldAdapter::init(EngineServices& services) {
    services_ = &services;
    device_ = services.device().device();

    std::string shaderBase = services.resourceDir() + "/shaders/";

    // Load vertex shader
    {
        auto r = vk2::ShaderModule::fromFile(device_, shaderBase + "world/v2_post/v2_starfield_vert.spv");
        if (!r) {
            log::error("starfield", "Failed to load v2_starfield_vert.spv: " + r.error().message);
            return;
        }
        vertShader_ = std::move(r.value());
    }

    // Load fragment shader
    {
        auto r = vk2::ShaderModule::fromFile(device_, shaderBase + "world/v2_post/v2_starfield_frag.spv");
        if (!r) {
            log::error("starfield", "Failed to load v2_starfield_frag.spv: " + r.error().message);
            return;
        }
        fragShader_ = std::move(r.value());
    }

    // Pipeline layout: no descriptor sets, push constants only
    {
        vk2::PipelineLayoutDesc pld;
        VkPushConstantRange pushRange{};
        pushRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        pushRange.offset = 0;
        pushRange.size = sizeof(PushConstants);
        pld.pushConstants = {pushRange};

        auto r = vk2::PipelineLayout::create(device_, pld);
        if (!r) {
            log::error("starfield", "Pipeline layout creation failed: " + r.error().message);
            return;
        }
        pipelineLayout_ = std::move(r.value());
    }

    // Graphics pipeline: additive blending, depth test, custom vertex input
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

        // Vertex input: StarVertex { vec3 pos; vec4 color; } = 28 bytes
        VkVertexInputBindingDescription bindingDesc{};
        bindingDesc.binding = 0;
        bindingDesc.stride = sizeof(StarVertex);
        bindingDesc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        VkVertexInputAttributeDescription attrDescs[2]{};
        // location 0: position (vec3, offset 0)
        attrDescs[0].location = 0;
        attrDescs[0].binding = 0;
        attrDescs[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        attrDescs[0].offset = offsetof(StarVertex, pos);
        // location 1: color (vec4, offset 12)
        attrDescs[1].location = 1;
        attrDescs[1].binding = 0;
        attrDescs[1].format = VK_FORMAT_R32G32B32A32_SFLOAT;
        attrDescs[1].offset = offsetof(StarVertex, color);

        VkPipelineVertexInputStateCreateInfo vertexInput{};
        vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInput.vertexBindingDescriptionCount = 1;
        vertexInput.pVertexBindingDescriptions = &bindingDesc;
        vertexInput.vertexAttributeDescriptionCount = 2;
        vertexInput.pVertexAttributeDescriptions = attrDescs;

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
        depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;

        // Additive blending: src=ONE, dst=ONE for color; alpha: src=ZERO, dst=ONE
        VkPipelineColorBlendAttachmentState blendAttachment{};
        blendAttachment.blendEnable = VK_TRUE;
        blendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
        blendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
        blendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
        blendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        blendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
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

        // Dynamic rendering: color (R8G8B8A8_UNORM) + depth (D32_SFLOAT)
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
            log::error("starfield", "Graphics pipeline creation failed");
            return;
        }
    }

    // Generate star vertex buffer
    generateStars();
    if (vertexBuffer_.handle() == VK_NULL_HANDLE) {
        log::error("starfield", "Star generation failed — adapter will be non-functional");
        // Still mark initialized so shutdown() cleans up pipeline resources
    }

    initialized_ = true;
    log::info("starfield", "StarFieldAdapter initialized");
}

void StarFieldAdapter::configure(const ConfigSnapshot& /*config*/) {
    // No configurable parameters yet
}

void StarFieldAdapter::execute(const FrameContext& ctx, const PassResources& resources) {
    if (!initialized_ || !resources.resolved || resources.cmd == VK_NULL_HANDLE) return;
    if (vertexBuffer_.handle() == VK_NULL_HANDLE) return;
    if (resources.inputs.empty() || !resources.inputs[0].valid()) return;
    if (!c2dAdapter_) return;

    VkCommandBuffer cmd = resources.cmd;
    uint32_t colorIdx = resources.inputs[0].index;

    auto& pool = services_->frame().resourcePool();
    uint32_t width = pool.image(colorIdx).width();
    uint32_t height = pool.image(colorIdx).height();
    if (width == 0 || height == 0) return;

    // Get the depth image from ColorToDepthAdapter
    auto& depthImage = c2dAdapter_->depthImage();
    if (depthImage.handle() == VK_NULL_HANDLE) return;

    // Build viewProj from camera view and projection matrices (column-major)
    const float* view = ctx.camera.view;
    const float* proj = ctx.camera.projection;

    PushConstants pc{};
    // viewProj = proj * view (column-major matrix multiply)
    for (int col = 0; col < 4; ++col) {
        for (int row = 0; row < 4; ++row) {
            float sum = 0.0f;
            for (int k = 0; k < 4; ++k) {
                sum += proj[k * 4 + row] * view[col * 4 + k];
            }
            pc.viewProj[col * 4 + row] = sum;
        }
    }

    // Read sky state from SceneResourceService (CPU-side copy of latest SkyUBO).
    // sunDirection drives the star sphere rotation (exclusion zone tracks the sun).
    // starVisibility = smoothstep(0.0, -0.2, sunDirection.y): 0 during day, 1 at night.
    const auto& sky = services_->sceneRes().latestSkyData();
    pc.sunDirection[0] = sky.sunDirection[0];
    pc.sunDirection[1] = sky.sunDirection[1];
    pc.sunDirection[2] = sky.sunDirection[2];
    pc.skyType = sky.skyType;
    pc.rainGradient = sky.rainGradient;

    // Compute star visibility from sun elevation: smoothstep(0.0, -0.2, sunDir.y)
    // When sun is above horizon (sunDir.y > 0): visibility = 0 (daytime)
    // When sun is below -0.2 (deep night): visibility = 1
    {
        float y = sky.sunDirection[1];
        float t = (y - 0.0f) / (-0.2f - 0.0f);  // Remap [0, -0.2] -> [0, 1]
        t = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);  // clamp
        pc.starVisibility = t * t * (3.0f - 2.0f * t);  // smoothstep
    }

    // Early out if stars are invisible (daytime)
    if (pc.starVisibility < 0.001f) return;

    // Transition depth image to DEPTH_STENCIL_ATTACHMENT_OPTIMAL for read+write
    // (C2D leaves it in this layout after its pass, so this should already be correct,
    // but we transition explicitly for safety)
    if (depthImage.layout() != VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
        imageBarrier(cmd, depthImage.handle(),
                     depthImage.layout(), VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                     VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                     VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                     VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                     VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                     VK_IMAGE_ASPECT_DEPTH_BIT);
    }

    // Transition color image to COLOR_ATTACHMENT_OPTIMAL for rendering
    VkImage colorImage = resources.resolved->images[colorIdx];
    imageBarrier(cmd, colorImage,
                 VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                 VK_ACCESS_2_SHADER_WRITE_BIT,
                 VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                 VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                 VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT);

    // Begin dynamic rendering with color (LOAD) + depth (LOAD)
    VkRenderingAttachmentInfo colorAttachment{};
    colorAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    colorAttachment.imageView = resources.resolved->views[colorIdx];
    colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

    VkRenderingAttachmentInfo depthAttachment{};
    depthAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    depthAttachment.imageView = depthImage.defaultView();
    depthAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

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

    // Bind pipeline and push constants
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);
    vkCmdPushConstants(cmd, pipelineLayout_.handle(),
                       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof(PushConstants), &pc);

    // Bind vertex buffer
    VkBuffer vb = vertexBuffer_.handle();
    VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &vb, &offset);

    // Draw all stars
    vkCmdDraw(cmd, TOTAL_VERTICES, 1, 0, 0);

    vkCmdEndRendering(cmd);

    // Transition color image back to GENERAL for downstream passes (composite)
    imageBarrier(cmd, colorImage,
                 VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                 VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                 VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                 VK_ACCESS_2_SHADER_READ_BIT,
                 VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
}

void StarFieldAdapter::destroyPipeline() {
    if (pipeline_ != VK_NULL_HANDLE && device_ != VK_NULL_HANDLE) {
        vkDestroyPipeline(device_, pipeline_, nullptr);
        pipeline_ = VK_NULL_HANDLE;
    }
}

void StarFieldAdapter::shutdown() {
    if (device_ == VK_NULL_HANDLE) return;

    vertexBuffer_ = {};
    destroyPipeline();
    pipelineLayout_ = {};
    vertShader_ = {};
    fragShader_ = {};

    if (descSetLayout_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device_, descSetLayout_, nullptr);
        descSetLayout_ = VK_NULL_HANDLE;
    }

    initialized_ = false;
    log::info("starfield", "StarFieldAdapter shut down");
}

} // namespace engine
