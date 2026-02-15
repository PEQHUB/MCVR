#include "core/render/hdr_composite_pass.hpp"

#include "core/render/render_framework.hpp"
#include "core/render/renderer.hpp"

#include <filesystem>
#include <iostream>

static std::ostream &hdrCompositeCout() {
    return std::cout << "[HdrCompositePass] ";
}

// Push constant matching hdr_ui_composite.frag
struct HdrCompositePushConstant {
    float uiBrightnessNits;
};

HdrCompositePass::~HdrCompositePass() {
    destroy();
}

void HdrCompositePass::init(std::shared_ptr<Framework> framework) {
    framework_ = framework;

    auto swapchain = framework->swapchain();
    width_ = swapchain->vkExtent().width;
    height_ = swapchain->vkExtent().height;

    hdrCompositeCout() << "init " << width_ << "x" << height_ << std::endl;

    initShaders();
    initSampler();
    initDescriptorSets();
    initRenderPass();
    initFramebuffers();
    initPipeline();
}

void HdrCompositePass::recreate(std::shared_ptr<Framework> framework) {
    destroy();
    init(framework);
}

void HdrCompositePass::destroy() {
    pipelineHdr_.reset();
    pipelineSdr_.reset();
    framebuffers_.clear();
    renderPass_.reset();
    descriptorTables_.clear();
    sampler_.reset();
    fragShaderSdr_.reset();
    fragShaderHdr_.reset();
    vertShader_.reset();
}

// ─── Shader loading ────────────────────────────────────────────────────────

void HdrCompositePass::initShaders() {
    auto framework = framework_.lock();
    std::filesystem::path shaderPath = Renderer::folderPath / "shaders";

    vertShader_ = vk::Shader::create(
        framework->device(),
        (shaderPath / "full_screen_vert.spv").string());

    fragShaderHdr_ = vk::Shader::create(
        framework->device(),
        (shaderPath / "hdr_ui_composite_frag.spv").string());

    fragShaderSdr_ = vk::Shader::create(
        framework->device(),
        (shaderPath / "sdr_ui_composite_frag.spv").string());
}

// ─── Sampler ───────────────────────────────────────────────────────────────

void HdrCompositePass::initSampler() {
    auto framework = framework_.lock();

    // Nearest sampling — source and destination dimensions match exactly.
    sampler_ = vk::Sampler::create(
        framework->device(),
        VK_FILTER_NEAREST,
        VK_SAMPLER_MIPMAP_MODE_NEAREST,
        VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
}

// ─── Descriptor sets (one per swapchain image) ────────────────────────────

void HdrCompositePass::initDescriptorSets() {
    auto framework = framework_.lock();
    uint32_t imageCount = framework->swapchain()->imageCount();

    descriptorTables_.resize(imageCount);

    for (uint32_t i = 0; i < imageCount; i++) {
        descriptorTables_[i] = vk::DescriptorTableBuilder{}
            .beginDescriptorLayoutSet()  // set 0
            .beginDescriptorLayoutSetBinding()
            .defineDescriptorLayoutSetBinding({
                .binding = 0,
                .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
            })
            .defineDescriptorLayoutSetBinding({
                .binding = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
            })
            .endDescriptorLayoutSetBinding()
            .endDescriptorLayoutSet()
            .definePushConstant(VkPushConstantRange{
                .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
                .offset = 0,
                .size = sizeof(HdrCompositePushConstant),
            })
            .build(framework->device());
    }
}

// ─── Render pass ───────────────────────────────────────────────────────────

void HdrCompositePass::initRenderPass() {
    auto framework = framework_.lock();
    auto swapchainFormat = framework->swapchain()->vkSurfaceFormat().format;

    renderPass_ = vk::RenderPassBuilder{}
        .beginAttachmentDescription()
        .defineAttachmentDescription({
            .format = swapchainFormat,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,   // shader writes every pixel
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
            .initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        })
        .endAttachmentDescription()
        .beginAttachmentReference()
        .defineAttachmentReference({
            .attachment = 0,
            .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        })
        .endAttachmentReference()
        .beginSubpassDescription()
        .defineSubpassDescription({
            .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
            .colorAttachmentIndices = {0},
        })
        .endSubpassDescription()
        .build(framework->device());
}

// ─── Framebuffers (one per swapchain image) ────────────────────────────────

void HdrCompositePass::initFramebuffers() {
    auto framework = framework_.lock();
    auto &swapchainImages = framework->swapchain()->swapchainImages();
    uint32_t imageCount = framework->swapchain()->imageCount();

    framebuffers_.resize(imageCount);

    for (uint32_t i = 0; i < imageCount; i++) {
        framebuffers_[i] = vk::FramebufferBuilder{}
            .beginAttachment()
            .defineAttachment(swapchainImages[i])
            .endAttachment()
            .build(framework->device(), renderPass_);
    }
}

// ─── Graphics pipeline ─────────────────────────────────────────────────────

void HdrCompositePass::initPipeline() {
    auto framework = framework_.lock();
    auto device = framework->device();

    auto makePipeline = [&](std::shared_ptr<vk::Shader> fragShader) {
        return vk::GraphicsPipelineBuilder{}
            .defineRenderPass(renderPass_, 0)
            .beginShaderStage()
            .defineShaderStage(vertShader_, VK_SHADER_STAGE_VERTEX_BIT)
            .defineShaderStage(fragShader, VK_SHADER_STAGE_FRAGMENT_BIT)
            .endShaderStage()
            .defineVertexInputState<void>()
            .defineViewportScissorState({
                .viewport = {
                    .x = 0,
                    .y = 0,
                    .width = static_cast<float>(width_),
                    .height = static_cast<float>(height_),
                    .minDepth = 0.0,
                    .maxDepth = 1.0,
                },
                .scissor = {
                    .offset = {.x = 0, .y = 0},
                    .extent = {width_, height_},
                },
            })
            .defineDepthStencilState({
                .depthTestEnable = VK_FALSE,
                .depthWriteEnable = VK_FALSE,
                .depthCompareOp = VK_COMPARE_OP_ALWAYS,
                .depthBoundsTestEnable = VK_FALSE,
                .stencilTestEnable = VK_FALSE,
            })
            .beginColorBlendAttachmentState()
            .defineDefaultColorBlendAttachmentState()
            .endColorBlendAttachmentState()
            .definePipelineLayout(descriptorTables_[0])
            .build(device);
    };

    pipelineHdr_ = makePipeline(fragShaderHdr_);
    pipelineSdr_ = makePipeline(fragShaderSdr_);
}

// ─── Record composite pass into command buffer ─────────────────────────────

void HdrCompositePass::record(std::shared_ptr<vk::CommandBuffer> cmd,
                              uint32_t frameIndex,
                              OutputMode mode,
                              float uiBrightnessNits,
                              std::shared_ptr<vk::DeviceLocalImage> worldImage,
                              std::shared_ptr<vk::DeviceLocalImage> overlayImage,
                              std::shared_ptr<vk::SwapchainImage> swapchainImage,
                              uint32_t mainQueueIndex) {

     auto pipeline = (mode == OutputMode::Hdr10) ? pipelineHdr_ : pipelineSdr_;

    // ── Bind textures to this frame's descriptor set ──
    auto descriptorTable = descriptorTables_[frameIndex];
    descriptorTable->bindSamplerImageForShader(sampler_, worldImage, 0, 0);    // set 0, binding 0
    descriptorTable->bindSamplerImageForShader(sampler_, overlayImage, 0, 1);  // set 0, binding 1

    // ── Transition images for shader read / render target ──
    cmd->barriersBufferImage(
        {}, {
            // World output → shader read
            {
                .srcStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
                .oldLayout = worldImage->imageLayout(),
                .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                .srcQueueFamilyIndex = mainQueueIndex,
                .dstQueueFamilyIndex = mainQueueIndex,
                .image = worldImage,
                .subresourceRange = vk::wholeColorSubresourceRange,
            },
            // Overlay → shader read
            {
                .srcStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
                .oldLayout = overlayImage->imageLayout(),
                .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                .srcQueueFamilyIndex = mainQueueIndex,
                .dstQueueFamilyIndex = mainQueueIndex,
                .image = overlayImage,
                .subresourceRange = vk::wholeColorSubresourceRange,
            },
            // Swapchain → color attachment
            {
                .srcStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                .dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                .dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                .oldLayout = swapchainImage->imageLayout(),
                .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                .srcQueueFamilyIndex = mainQueueIndex,
                .dstQueueFamilyIndex = mainQueueIndex,
                .image = swapchainImage,
                .subresourceRange = vk::wholeColorSubresourceRange,
            },
        });

    worldImage->imageLayout() = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    overlayImage->imageLayout() = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    swapchainImage->imageLayout() = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    // ── Push constant ──
    HdrCompositePushConstant pc{};
    pc.uiBrightnessNits = uiBrightnessNits;

    vkCmdPushConstants(
        cmd->vkCommandBuffer(),
        descriptorTable->vkPipelineLayout(),
        VK_SHADER_STAGE_FRAGMENT_BIT,
        0,
        sizeof(HdrCompositePushConstant),
        &pc);

    // ── Begin render pass (targeting swapchain) ──
    cmd->beginRenderPass({
        .renderPass = renderPass_,
        .framebuffer = framebuffers_[frameIndex],
        .renderAreaExtent = {width_, height_},
        .clearValues = {},  // LOAD_OP_DONT_CARE — no clear needed
    });

    // ── Draw fullscreen triangle ──
    cmd->bindGraphicsPipeline(pipeline)
        ->bindDescriptorTable(descriptorTable, VK_PIPELINE_BIND_POINT_GRAPHICS)
        ->draw(3, 1)
        ->endRenderPass();

    // Render pass finalLayout = PRESENT_SRC_KHR handles the final transition.
    swapchainImage->imageLayout() = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    // Transition world and overlay images back to usable layouts for next frame.
#ifdef USE_AMD
    cmd->barriersBufferImage(
        {}, {
            {
                .srcStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                .srcAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
                .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                .oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                .srcQueueFamilyIndex = mainQueueIndex,
                .dstQueueFamilyIndex = mainQueueIndex,
                .image = worldImage,
                .subresourceRange = vk::wholeColorSubresourceRange,
            },
            {
                .srcStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                .srcAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
                .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                .oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                .srcQueueFamilyIndex = mainQueueIndex,
                .dstQueueFamilyIndex = mainQueueIndex,
                .image = overlayImage,
                .subresourceRange = vk::wholeColorSubresourceRange,
            },
        });
    worldImage->imageLayout() = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    overlayImage->imageLayout() = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
#else
    cmd->barriersBufferImage(
        {}, {
            {
                .srcStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                .srcAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
                .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                .oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                .newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                .srcQueueFamilyIndex = mainQueueIndex,
                .dstQueueFamilyIndex = mainQueueIndex,
                .image = worldImage,
                .subresourceRange = vk::wholeColorSubresourceRange,
            },
            {
                .srcStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                .srcAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
                .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                .oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                .newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                .srcQueueFamilyIndex = mainQueueIndex,
                .dstQueueFamilyIndex = mainQueueIndex,
                .image = overlayImage,
                .subresourceRange = vk::wholeColorSubresourceRange,
            },
        });
    worldImage->imageLayout() = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    overlayImage->imageLayout() = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
#endif
}
