#include "core/render/modules/world/tone_mapping/tone_mapping_module.hpp"

#include "core/render/pipeline.hpp"
#include "core/render/render_framework.hpp"
#include "core/render/renderer.hpp"

ToneMappingModule::ToneMappingModule() {}

void ToneMappingModule::init(std::shared_ptr<Framework> framework, std::shared_ptr<WorldPipeline> worldPipeline) {
    WorldModule::init(framework, worldPipeline);

    uint32_t size = framework->swapchain()->imageCount();

    hdrImages_.resize(size);
    ldrImages_.resize(size);
}

bool ToneMappingModule::setOrCreateInputImages(std::vector<std::shared_ptr<vk::DeviceLocalImage>> &images,
                                               std::vector<VkFormat> &formats,
                                               uint32_t frameIndex) {
    if (images.size() == 0) return false;

    auto framework = framework_.lock();
    if (images[0] == nullptr) {
        hdrImages_[frameIndex] = images[0] = vk::DeviceLocalImage::create(
            framework->device(), framework->vma(), false, width_, height_, 1, formats[0],
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    } else {
        if (images[0]->width() != width_ || images[0]->height() != height_) return false;
        hdrImages_[frameIndex] = images[0];
    }

    return true;
}

bool ToneMappingModule::setOrCreateOutputImages(std::vector<std::shared_ptr<vk::DeviceLocalImage>> &images,
                                                std::vector<VkFormat> &formats,
                                                uint32_t frameIndex) {
    if (images.size() == 0 || images[0] == nullptr) return false;

    width_ = images[0]->width();
    height_ = images[0]->height();

    ldrImages_[frameIndex] = images[0];

    return true;
}

void ToneMappingModule::setAttributes(int attributeCount, std::vector<std::string> &attributeKVs) {
    for (int i = 0; i < attributeCount; i++) {
        if (attributeKVs[2 * i] == "render_pipeline.module.tone_mapping.attribute.middle_grey") {
            middleGrey_ = std::stof(attributeKVs[2 * i + 1]);
        } else if (attributeKVs[2 * i] == "render_pipeline.module.tone_mapping.attribute.exposure_up_speed") {
            speedUp_ = std::stof(attributeKVs[2 * i + 1]);
        } else if (attributeKVs[2 * i] == "render_pipeline.module.tone_mapping.attribute.exposure_down_speed") {
            speedDown_ = std::stof(attributeKVs[2 * i + 1]);
        } else if (attributeKVs[2 * i] == "render_pipeline.module.tone_mapping.attribute.tonemap_mode") {
            if (attributeKVs[2 * i + 1] == "pbr_neutral") {
                tonemapMode_ = 0.0f;
            } else if (attributeKVs[2 * i + 1] == "reinhard_extended") {
                tonemapMode_ = 1.0f;
            } else if (attributeKVs[2 * i + 1] == "aces") {
                tonemapMode_ = 2.0f;
            } else if (attributeKVs[2 * i + 1] == "agx") {
                tonemapMode_ = 3.0f;
            } else if (attributeKVs[2 * i + 1] == "lottes") {
                tonemapMode_ = 4.0f;
            } else if (attributeKVs[2 * i + 1] == "frostbite") {
                tonemapMode_ = 5.0f;
            } else if (attributeKVs[2 * i + 1] == "uncharted2") {
                tonemapMode_ = 6.0f;
            } else if (attributeKVs[2 * i + 1] == "gt") {
                tonemapMode_ = 7.0f;
            }
        } else if (attributeKVs[2 * i] == "render_pipeline.module.tone_mapping.attribute.lwhite") {
            Lwhite_ = std::stof(attributeKVs[2 * i + 1]);
        } else if (attributeKVs[2 * i] == "render_pipeline.module.tone_mapping.attribute.max_exposure") {
            maxExposure_ = std::stof(attributeKVs[2 * i + 1]);
        } else if (attributeKVs[2 * i] == "render_pipeline.module.tone_mapping.attribute.exposure_compensation") {
            Renderer::options.exposureCompensation = std::stof(attributeKVs[2 * i + 1]);
        }
    }
}

void ToneMappingModule::build() {
    auto framework = framework_.lock();
    auto worldPipeline = worldPipeline_.lock();
    uint32_t size = framework->swapchain()->imageCount();

    initDescriptorTables();
    initImages();
    initBuffers();
    initRenderPass();
    initFrameBuffers();
    initPipeline();

    contexts_.resize(size);

    for (int i = 0; i < size; i++) {
        contexts_[i] = ToneMappingModuleContext::create(framework->contexts()[i], worldPipeline->contexts()[i],
                                                        shared_from_this());
    }
}

std::vector<std::shared_ptr<WorldModuleContext>> &ToneMappingModule::contexts() {
    return contexts_;
}

void ToneMappingModule::bindTexture(std::shared_ptr<vk::Sampler> sampler,
                                    std::shared_ptr<vk::DeviceLocalImage> image,
                                    int index) {}

void ToneMappingModule::preClose() {}

void ToneMappingModule::initDescriptorTables() {
    auto framework = framework_.lock();
    uint32_t size = framework->swapchain()->imageCount();

    descriptorTables_.resize(size);
    samplers_.resize(size);

    for (int i = 0; i < size; i++) {
        descriptorTables_[i] = vk::DescriptorTableBuilder{}
                                   .beginDescriptorLayoutSet() // set 0
                                   .beginDescriptorLayoutSetBinding()
                                   .defineDescriptorLayoutSetBinding({
                                       .binding = 0,
                                       .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                       .descriptorCount = 1,
                                       .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT,
                                   })
                                   .defineDescriptorLayoutSetBinding({
                                       .binding = 1,
                                       .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                       .descriptorCount = 1,
                                       .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT,
                                   })
                                   .defineDescriptorLayoutSetBinding({
                                       .binding = 2,
                                       .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                       .descriptorCount = 1,
                                       .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT,
                                   })
                                   .endDescriptorLayoutSetBinding()
                                   .endDescriptorLayoutSet()
                                   .definePushConstant(VkPushConstantRange{
                                       .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                                       .offset = 0,
                                       .size = sizeof(ToneMappingModulePushConstant),
                                   })
                                   .build(framework->device());

        samplers_[i] = vk::Sampler::create(framework->device(), VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_LINEAR,
                                           VK_SAMPLER_ADDRESS_MODE_REPEAT);
    }
}

void ToneMappingModule::initImages() {
    auto framework = framework_.lock();
    uint32_t size = framework->swapchain()->imageCount();

    for (int i = 0; i < size; i++) {
        descriptorTables_[i]->bindSamplerImageForShader(samplers_[i], hdrImages_[i], 0, 0);
    }
}

void ToneMappingModule::initBuffers() {
    auto framework = framework_.lock();
    auto vma = framework->vma();
    auto device = framework->device();
    uint32_t size = framework->swapchain()->imageCount();

    histBuffers_.resize(size);

    exposureData_ =
        vk::DeviceLocalBuffer::create(vma, device, sizeof(ToneMappingModuleExposureData),
                                      VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    for (int i = 0; i < size; i++) {
        histBuffers_[i] =
            vk::DeviceLocalBuffer::create(vma, device, histSize * sizeof(uint32_t),
                                          VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        descriptorTables_[i]->bindBuffer(histBuffers_[i], 0, 1);

        descriptorTables_[i]->bindBuffer(exposureData_, 0, 2);
    }
}

void ToneMappingModule::initRenderPass() {
    renderPass_ = vk::RenderPassBuilder{}
                      .beginAttachmentDescription()
                      .defineAttachmentDescription({
                          // color
                          .format = ldrImages_[0]->vkFormat(),
                      .samples = VK_SAMPLE_COUNT_1_BIT,
                      .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
                      .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
                      .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                      .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
#ifdef USE_AMD
                      .initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                      .finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
#else
                      .initialLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                      .finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
#endif
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
                      .build(framework_.lock()->device());
}

void ToneMappingModule::initFrameBuffers() {
    auto framework = framework_.lock();
    uint32_t size = framework->swapchain()->imageCount();

    framebuffers_.resize(size);

    for (int i = 0; i < size; i++) {
        framebuffers_[i] = vk::FramebufferBuilder{}
                               .beginAttachment()
                               .defineAttachment(ldrImages_[i])
                               .endAttachment()
                               .build(framework->device(), renderPass_);
    }
}

void ToneMappingModule::initPipeline() {
    auto framework = framework_.lock();
    auto device = framework->device();
    std::filesystem::path shaderPath = Renderer::folderPath / "shaders";

    histShader_ = vk::Shader::create(framework->device(), (shaderPath / "world/tone_mapping/hist_comp.spv").string());
    histPipeline_ =
        vk::ComputePipelineBuilder{}.defineShader(histShader_).definePipelineLayout(descriptorTables_[0]).build(device);

    exposureShader_ =
        vk::Shader::create(framework->device(), (shaderPath / "world/tone_mapping/exposure_comp.spv").string());
    exposurePipeline_ = vk::ComputePipelineBuilder{}
                            .defineShader(exposureShader_)
                            .definePipelineLayout(descriptorTables_[0])
                            .build(device);

    vertShader_ =
        vk::Shader::create(framework->device(), (shaderPath / "world/tone_mapping/tone_mapping_vert.spv").string());
    fragShader_ =
        vk::Shader::create(framework->device(), (shaderPath / "world/tone_mapping/tone_mapping_frag.spv").string());

    pipeline_ = vk::GraphicsPipelineBuilder{}
                    .defineRenderPass(renderPass_, 0)
                    .beginShaderStage()
                    .defineShaderStage(vertShader_, VK_SHADER_STAGE_VERTEX_BIT)
                    .defineShaderStage(fragShader_, VK_SHADER_STAGE_FRAGMENT_BIT)
                    .endShaderStage()
                    .defineVertexInputState<void>()
                    .defineViewportScissorState({
                        .viewport =
                            {
                                .x = 0,
                                .y = 0,
                                .width = static_cast<float>(framework->swapchain()->vkExtent().width),
                                .height = static_cast<float>(framework->swapchain()->vkExtent().height),
                                .minDepth = 0.0,
                                .maxDepth = 1.0,
                            },
                        .scissor =
                            {
                                .offset = {.x = 0, .y = 0},
                                .extent = framework->swapchain()->vkExtent(),
                            },
                    })
                    .defineDepthStencilState({
                        .depthTestEnable = VK_TRUE,
                        .depthWriteEnable = VK_TRUE,
                        .depthCompareOp = VK_COMPARE_OP_LESS,
                        .depthBoundsTestEnable = VK_FALSE,
                        .stencilTestEnable = VK_FALSE,
                    })
                    .beginColorBlendAttachmentState()
                    .defineDefaultColorBlendAttachmentState() // color
                    .endColorBlendAttachmentState()
                    .definePipelineLayout(descriptorTables_[0])
                    .build(device);
}

ToneMappingModuleContext::ToneMappingModuleContext(std::shared_ptr<FrameworkContext> frameworkContext,
                                                   std::shared_ptr<WorldPipelineContext> worldPipelineContext,
                                                   std::shared_ptr<ToneMappingModule> toneMappingModule)
    : WorldModuleContext(frameworkContext, worldPipelineContext),
      toneMappingModule(toneMappingModule),
      hdrImage(toneMappingModule->hdrImages_[frameworkContext->frameIndex]),
      descriptorTable(toneMappingModule->descriptorTables_[frameworkContext->frameIndex]),
      framebuffer(toneMappingModule->framebuffers_[frameworkContext->frameIndex]),
      histBuffer(toneMappingModule->histBuffers_[frameworkContext->frameIndex]),
      ldrImage(toneMappingModule->ldrImages_[frameworkContext->frameIndex]) {}

void ToneMappingModuleContext::render() {
    auto context = frameworkContext.lock();
    auto framework = context->framework.lock();
    auto worldCommandBuffer = context->worldCommandBuffer;
    auto mainQueueIndex = framework->physicalDevice()->mainQueueIndex();

    auto module = toneMappingModule.lock();

    auto chooseSrc = [](VkImageLayout oldLayout,
                        VkPipelineStageFlags2 fallbackStage,
                        VkAccessFlags2 fallbackAccess,
                        VkPipelineStageFlags2 &outStage,
                        VkAccessFlags2 &outAccess) {
        if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED) {
            outStage = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
            outAccess = 0;
        } else {
            outStage = fallbackStage;
            outAccess = fallbackAccess;
        }
    };

    VkPipelineStageFlags2 hdrSrcStage = 0;
    VkAccessFlags2 hdrSrcAccess = 0;
    chooseSrc(hdrImage->imageLayout(),
              VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                  VK_PIPELINE_STAGE_2_TRANSFER_BIT,
              VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
              hdrSrcStage, hdrSrcAccess);

    VkPipelineStageFlags2 ldrSrcStage = 0;
    VkAccessFlags2 ldrSrcAccess = 0;
    chooseSrc(ldrImage->imageLayout(),
              VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
              VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
              ldrSrcStage, ldrSrcAccess);

    worldCommandBuffer->barriersBufferImage(
        {{
            .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                            VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            .srcQueueFamilyIndex = mainQueueIndex,
            .dstQueueFamilyIndex = mainQueueIndex,
            .buffer = histBuffer,
        }},
        {{
             .srcStageMask = hdrSrcStage,
             .srcAccessMask = hdrSrcAccess,
             .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                             VK_PIPELINE_STAGE_2_TRANSFER_BIT,
             .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
             .oldLayout = hdrImage->imageLayout(),
             .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
             .srcQueueFamilyIndex = mainQueueIndex,
             .dstQueueFamilyIndex = mainQueueIndex,
             .image = hdrImage,
             .subresourceRange = vk::wholeColorSubresourceRange,
         },
         {
             .srcStageMask = ldrSrcStage,
             .srcAccessMask = ldrSrcAccess,
             .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
             .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
             .oldLayout = ldrImage->imageLayout(),
#ifdef USE_AMD
                 .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
#else
                 .newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
#endif
             .srcQueueFamilyIndex = mainQueueIndex,
             .dstQueueFamilyIndex = mainQueueIndex,
             .image = ldrImage,
             .subresourceRange = vk::wholeColorSubresourceRange,
         }});
    hdrImage->imageLayout() = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
#ifdef USE_AMD
    ldrImage->imageLayout() = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
#else
    ldrImage->imageLayout() = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
#endif

    vkCmdFillBuffer(worldCommandBuffer->vkCommandBuffer(), histBuffer->vkBuffer(), 0, VK_WHOLE_SIZE, 0);

    worldCommandBuffer->barriersBufferImage(
        {{
            .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                            VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
            .srcQueueFamilyIndex = mainQueueIndex,
            .dstQueueFamilyIndex = mainQueueIndex,
            .buffer = histBuffer,
        }},
        {});

    std::chrono::time_point<std::chrono::high_resolution_clock> currentTimePoint =
        std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedTime = currentTimePoint - module->lastTimePoint_;
    module->lastTimePoint_ = currentTimePoint;

    // HDR pipeline is always enabled for visual parity.
    // HDR10 output encoding depends on swapchain HDR + user HDR output toggle.
    bool hdrPipelineEnabled = true;
    bool hdr10OutputEnabled = Renderer::options.hdrEnabled && framework->swapchain()->isHDR();

    ToneMappingModulePushConstant pc{};
    pc.log2Min = -12.0f;
    pc.log2Max = (Renderer::options.legacyExposure) ? +8.0f : Renderer::options.exposureLog2MaxImproved;
    pc.epsilon = 1e-6f;
    pc.lowPercent = 0.005f;
    // In improved mode, include more of the bright tail in the meter so sunlit ground isn't ignored.
    // Legacy mode keeps the original 99% trim (legacy failure mode).
    pc.highPercent = (Renderer::options.legacyExposure) ? 0.99f : 0.999f;
    pc.middleGrey = Renderer::options.middleGrey;
    pc.dt = elapsedTime.count();
    pc.speedUp = Renderer::options.exposureUpSpeed;
    pc.speedDown = Renderer::options.exposureDownSpeed;
    pc.brightAdaptBoost = Renderer::options.exposureBrightAdaptBoost;
    pc.minExposure = Renderer::options.minExposure;
    pc.maxExposure = Renderer::options.maxExposure;
    pc.tonemapMode = static_cast<float>(Renderer::options.tonemappingMode);
    pc.Lwhite = Renderer::options.Lwhite;
    pc.exposureCompensation = Renderer::options.exposureCompensation;
    pc.legacyExposure = Renderer::options.legacyExposure ? 1.0f : 0.0f;
    // Improved auto-exposure highlight protection (legacy mode ignores these).
    pc.highlightPercent = Renderer::options.exposureHighlightPercentile;
    pc.highlightProtection = Renderer::options.exposureHighlightProtection;
    pc.highlightSmoothingSpeed = Renderer::options.exposureHighlightSmoothingSpeed;
    // HDR fields
    pc.hdrPipelineEnabled = hdrPipelineEnabled ? 1.0f : 0.0f;
    pc.hdr10OutputEnabled = hdr10OutputEnabled ? 1.0f : 0.0f;
    pc.peakNits = Renderer::options.hdrPeakNits;
    pc.paperWhiteNits = Renderer::options.hdrPaperWhiteNits;
    pc.saturation = Renderer::options.saturation;
    pc.sdrTransferFunction = static_cast<float>(Renderer::options.sdrTransferFunction);
    pc.manualExposureEnabled = Renderer::options.manualExposureEnabled ? 1.0f : 0.0f;
    pc.manualExposure = Renderer::options.manualExposure;

    vkCmdPushConstants(worldCommandBuffer->vkCommandBuffer(), descriptorTable->vkPipelineLayout(),
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ToneMappingModulePushConstant), &pc);

    worldCommandBuffer->bindDescriptorTable(descriptorTable, VK_PIPELINE_BIND_POINT_COMPUTE)
        ->bindComputePipeline(module->histPipeline_);

    uint32_t groupX = (module->width_ + 16 - 1) / 16;
    uint32_t groupY = (module->height_ + 16 - 1) / 16;
    vkCmdDispatch(worldCommandBuffer->vkCommandBuffer(), groupX, groupY, 1);

    worldCommandBuffer->barriersBufferImage(
        {{
            .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                            VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
            .srcQueueFamilyIndex = mainQueueIndex,
            .dstQueueFamilyIndex = mainQueueIndex,
            .buffer = histBuffer,
        }},
        {});

    worldCommandBuffer->bindComputePipeline(module->exposurePipeline_);
    vkCmdDispatch(worldCommandBuffer->vkCommandBuffer(), 1, 1, 1);

    worldCommandBuffer->barriersBufferImage(
        {{
            .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                            VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
            .srcQueueFamilyIndex = mainQueueIndex,
            .dstQueueFamilyIndex = mainQueueIndex,
            .buffer = module->exposureData_,
        }},
        {});

    worldCommandBuffer->beginRenderPass({
        .renderPass = module->renderPass_,
        .framebuffer = framebuffer,
        .renderAreaExtent = {ldrImage->width(), ldrImage->height()},
        .clearValues = {{.color = {0.1f, 0.1f, 0.1f, 1.0f}}, {.depthStencil = {.depth = 1.0f}}},
    });
    ldrImage->imageLayout() = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    worldCommandBuffer->bindGraphicsPipeline(module->pipeline_)
        ->bindDescriptorTable(descriptorTable, VK_PIPELINE_BIND_POINT_GRAPHICS)
        ->draw(3, 1)
        ->endRenderPass();
#ifdef USE_AMD
    ldrImage->imageLayout() = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
#else
    ldrImage->imageLayout() = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
#endif
}
