#include "core/render/modules/world/dlss/dlss_module.hpp"

#include "core/render/buffers.hpp"
#include "core/render/pipeline.hpp"
#include "core/render/render_framework.hpp"
#include "core/render/renderer.hpp"

std::shared_ptr<NgxContext> DLSSModule::ngxContext_ = nullptr;

bool DLSSModule::initNGXContext() {
    std::filesystem::path dlssPath = Renderer::folderPath / "dlss";
    std::error_code ec;
    if (!std::filesystem::create_directories(dlssPath, ec)) {
        if (ec) {
            std::cerr << "Failed to create directory: " << ec.message() << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    auto framework = Renderer::instance().framework();
    ngxContext_ = NgxContext::create();

    NgxContext::NgxInitInfo ngxInitInfo{};
    ngxInitInfo.instance = framework->instance();
    ngxInitInfo.physicalDevice = framework->physicalDevice();
    ngxInitInfo.device = framework->device();
    ngxInitInfo.applicationPath = dlssPath.string();
    if (ngxContext_->init(ngxInitInfo) != NVSDK_NGX_Result_Success) {
        // init() may have partially succeeded (e.g. VULKAN_Init ok but GetCapabilityParameters
        // failed). NgxContext::init() calls deinit() internally in that case, but call it here
        // defensively to ensure no NGX process-level state is left behind.
        ngxContext_->deinit();
        ngxContext_ = nullptr;
        return false;
    }

    if (ngxContext_->queryDlssRRAvailable() != NVSDK_NGX_Result_Success) {
        // init() succeeded (NVSDK_NGX_VULKAN_Init was called and device_ is set).
        // We MUST call deinit() → NVSDK_NGX_VULKAN_Shutdown1() before clearing ngxContext_,
        // otherwise the NGX process-level state stays initialized and the next
        // NVSDK_NGX_VULKAN_Init() call (e.g. after user drops in DLSS DLLs) will fail.
        ngxContext_->deinit();
        ngxContext_ = nullptr;
        return false;
    }

    return true;
}

void DLSSModule::deinitNGXContext() {
    if (ngxContext_ != nullptr) {
        ngxContext_->deinit();
        ngxContext_ = nullptr;
    }
}

DLSSModule::DLSSModule() {}

void DLSSModule::init(std::shared_ptr<Framework> framework, std::shared_ptr<WorldPipeline> worldPipeline) {
    WorldModule::init(framework, worldPipeline);

    uint32_t size = framework->swapchain()->imageCount();

    hdrImages_.resize(size);
    diffuseAlbedoImages_.resize(size);
    specularAlbedoImages_.resize(size);
    normalRoughnessImages_.resize(size);
    motionVectorImages_.resize(size);
    linearDepthImages_.resize(size);
    specularHitDepthImages_.resize(size);
    firstHitDepthImages_.resize(size);
    processedImages_.resize(size);
    upscaledFirstHitDepthImages_.resize(size);
    upscaled2xImages_.resize(size);

    dlss_ = DlssRR::create();
}

bool DLSSModule::setOrCreateInputImages(std::vector<std::shared_ptr<vk::DeviceLocalImage>> &images,
                                        std::vector<VkFormat> &formats,
                                        uint32_t frameIndex) {
    auto framework = framework_.lock();
    if (ngxContext_ == nullptr) return false;

    if (images.size() != inputImageNum) return false;

    // Apply runtime quality override from Java settings BEFORE querying sizes
    switch (Renderer::options.upscalerMode) {
        case 0: mode_ = NVSDK_NGX_PerfQuality_Value_MaxPerf; break;
        case 1: mode_ = NVSDK_NGX_PerfQuality_Value_Balanced; break;
        case 2: mode_ = NVSDK_NGX_PerfQuality_Value_MaxQuality; break;
        case 3: mode_ = NVSDK_NGX_PerfQuality_Value_DLAA; break;
        case 4: mode_ = NVSDK_NGX_PerfQuality_Value_Balanced; break; // Custom: use Balanced as NGX base
    }

    if (Renderer::options.upscalerMode == 4) {
        // Custom mode: use resolution override percentage instead of NGX presets
        float scale = static_cast<float>(Renderer::options.upscalerResOverride) / 100.0f;
        inputWidth_ = std::max(1u, static_cast<uint32_t>(outputWidth_ * scale));
        inputHeight_ = std::max(1u, static_cast<uint32_t>(outputHeight_ * scale));
    } else {
        // Preset mode: query NGX for optimal input sizes
        NgxContext::QuerySizeInfo querySizeInfo{};
        querySizeInfo.outputSize.width = outputWidth_;
        querySizeInfo.outputSize.height = outputHeight_;
        querySizeInfo.quality = mode_;
        ngxContext_->querySupportedDlssInputSizes(querySizeInfo, supportedSizes_);
#ifdef DEBUG
        std::cout << "DLSS sizes:" << std::endl;
        std::cout << "\tminSize: [" << supportedSizes_.minSize.width << ", " << supportedSizes_.minSize.height << "]"
                  << std::endl;
        std::cout << "\tmaxSize: [" << supportedSizes_.maxSize.width << ", " << supportedSizes_.maxSize.height << "]"
                  << std::endl;
        std::cout << "\toptimalSize: [" << supportedSizes_.optimalSize.width << ", " << supportedSizes_.optimalSize.height
                  << "]" << std::endl;
#endif
        inputWidth_ = supportedSizes_.optimalSize.width;
        inputHeight_ = supportedSizes_.optimalSize.height;
    }

    // For each input image: create if null, or recreate if size doesn't match (e.g. after DLSS quality change)
    auto createOrResize = [&](int idx) {
        if (images[idx] == nullptr || images[idx]->width() != inputWidth_ || images[idx]->height() != inputHeight_) {
            images[idx] = vk::DeviceLocalImage::create(
                framework->device(), framework->vma(), false, inputWidth_, inputHeight_, 1, formats[idx],
                VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
        }
    };

    createOrResize(0); hdrImages_[frameIndex] = images[0];
    createOrResize(1); diffuseAlbedoImages_[frameIndex] = images[1];
    createOrResize(2); specularAlbedoImages_[frameIndex] = images[2];
    createOrResize(3); normalRoughnessImages_[frameIndex] = images[3];
    createOrResize(4); motionVectorImages_[frameIndex] = images[4];
    createOrResize(5); linearDepthImages_[frameIndex] = images[5];
    createOrResize(6); specularHitDepthImages_[frameIndex] = images[6];
    createOrResize(7); firstHitDepthImages_[frameIndex] = images[7];

    return true;
}

bool DLSSModule::setOrCreateOutputImages(std::vector<std::shared_ptr<vk::DeviceLocalImage>> &images,
                                         std::vector<VkFormat> &formats,
                                         uint32_t frameIndex) {
    auto framework = framework_.lock();
    if (ngxContext_ == nullptr) return false;

    if (images.size() != outputImageNum || images[0] == nullptr) return false;

    outputWidth_ = images[0]->width();
    outputHeight_ = images[0]->height();

    processedImages_[frameIndex] = images[0];

    if (images[1] == nullptr) {
        upscaledFirstHitDepthImages_[frameIndex] = images[1] = vk::DeviceLocalImage::create(
            framework->device(), framework->vma(), false, outputWidth_, outputHeight_, 1, formats[1],
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    } else {
        if (images[1]->width() != outputWidth_ || images[1]->height() != outputHeight_) return false;
        upscaledFirstHitDepthImages_[frameIndex] = images[1];
    }

    return true;
}

void DLSSModule::setAttributes(int attributeCount, std::vector<std::string> &attributeKVs) {
    for (int i = 0; i < attributeCount; i++) {
        if (attributeKVs[2 * i] == "render_pipeline.module.dlss.attribute.mode") {
            if (attributeKVs[2 * i + 1] == "render_pipeline.module.dlss.attribute.mode.performance") {
                mode_ = NVSDK_NGX_PerfQuality_Value_MaxPerf;
            } else if (attributeKVs[2 * i + 1] == "render_pipeline.module.dlss.attribute.mode.balanced") {
                mode_ = NVSDK_NGX_PerfQuality_Value_Balanced;
            } else if (attributeKVs[2 * i + 1] == "render_pipeline.module.dlss.attribute.mode.quality") {
                mode_ = NVSDK_NGX_PerfQuality_Value_MaxQuality;
            } else if (attributeKVs[2 * i + 1] == "render_pipeline.module.dlss.attribute.mode.dlaa") {
                mode_ = NVSDK_NGX_PerfQuality_Value_DLAA;
            }
        }
    }

    // Runtime override from Java settings (takes priority over pipeline YAML)
    switch (Renderer::options.upscalerMode) {
        case 0: mode_ = NVSDK_NGX_PerfQuality_Value_MaxPerf; break;
        case 1: mode_ = NVSDK_NGX_PerfQuality_Value_Balanced; break;
        case 2: mode_ = NVSDK_NGX_PerfQuality_Value_MaxQuality; break;
        case 3: mode_ = NVSDK_NGX_PerfQuality_Value_DLAA; break;
        case 4: mode_ = NVSDK_NGX_PerfQuality_Value_Balanced; break; // Custom: use Balanced as NGX base
    }
}

void DLSSModule::build() {
    if (ngxContext_ == nullptr) return;

    auto framework = framework_.lock();
    auto worldPipeline = worldPipeline_.lock();
    if (!framework || !worldPipeline) return;
    uint32_t size = framework->swapchain()->imageCount();

    // Output Scale 2x: DLSS targets 2x, Lanczos downscales to 1x
    dlssOutputWidth_ = Renderer::options.outputScale2x ? outputWidth_ * 2 : outputWidth_;
    dlssOutputHeight_ = Renderer::options.outputScale2x ? outputHeight_ * 2 : outputHeight_;

    NgxContext::DlssRRInitInfo dlssRRInitInfo{};
    dlssRRInitInfo.inputSize = {inputWidth_, inputHeight_};
    dlssRRInitInfo.outputSize = {dlssOutputWidth_, dlssOutputHeight_};
    dlssRRInitInfo.quality = mode_;
    dlssRRInitInfo.preset = static_cast<NVSDK_NGX_RayReconstruction_Hint_Render_Preset>(Renderer::options.upscalerPreset);
    ngxContext_->initDlssRR(dlssRRInitInfo, framework->mainCommandPool(), dlss_);

    if (Renderer::options.outputScale2x) {
        initLanczosResources();
    }

    contexts_.resize(size);

    for (int i = 0; i < size; i++) {
        auto ctx = DLSSModuleContext::create(framework->contexts()[i], worldPipeline->contexts()[i], shared_from_this());

        // Output Scale 2x: redirect DLSS output to 2x intermediate
        if (Renderer::options.outputScale2x && upscaled2xImages_[i]) {
            ctx->processedImage = upscaled2xImages_[i];          // DLSS writes to 2x
            ctx->finalOutputImage = processedImages_[i];          // Lanczos target (shared 1x)
        } else {
            ctx->finalOutputImage = nullptr;
        }

        contexts_[i] = ctx;
    }
}

std::vector<std::shared_ptr<WorldModuleContext>> &DLSSModule::contexts() {
    return contexts_;
}

void DLSSModule::bindTexture(std::shared_ptr<vk::Sampler> sampler,
                             std::shared_ptr<vk::DeviceLocalImage> image,
                             int index) {}

void DLSSModule::preClose() {
    if (dlss_) dlss_->deinit();

    // Lanczos resources cleanup
    lanczosPipeline_.reset();
    lanczosDescriptorTables_.clear();
    lanczosSampler_.reset();
    lanczosShader_.reset();
    upscaled2xImages_.clear();
}

void DLSSModule::initLanczosResources() {
    auto fw = framework_.lock();
    uint32_t size = fw->swapchain()->imageCount();

    // Create 2x intermediate images for DLSS output
    for (uint32_t i = 0; i < size; i++) {
        upscaled2xImages_[i] = vk::DeviceLocalImage::create(
            fw->device(), fw->vma(), false, dlssOutputWidth_, dlssOutputHeight_, 1,
            VK_FORMAT_R16G16B16A16_SFLOAT,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    }

    // Load Lanczos compute shader
    lanczosShader_ = vk::Shader::create(
        fw->device(),
        (Renderer::folderPath / "shaders/world/post_render/lanczos_downscale_comp.spv").string());

    // Sampler (texelFetch bypasses it, but needed for combined image sampler binding)
    lanczosSampler_ = vk::Sampler::create(
        fw->device(), VK_FILTER_NEAREST, VK_SAMPLER_MIPMAP_MODE_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);

    // Descriptor tables
    lanczosDescriptorTables_.resize(size);
    for (uint32_t i = 0; i < size; i++) {
        lanczosDescriptorTables_[i] = vk::DescriptorTableBuilder{}
            .beginDescriptorLayoutSet()
            .beginDescriptorLayoutSetBinding()
            .defineDescriptorLayoutSetBinding({
                .binding = 0,
                .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            })
            .defineDescriptorLayoutSetBinding({
                .binding = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            })
            .endDescriptorLayoutSetBinding()
            .endDescriptorLayoutSet()
            .definePushConstant(VkPushConstantRange{
                .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                .offset = 0,
                .size = sizeof(int32_t) * 4,
            })
            .build(fw->device());
    }

    // Compute pipeline
    lanczosPipeline_ = vk::ComputePipelineBuilder{}
        .defineShader(lanczosShader_)
        .definePipelineLayout(lanczosDescriptorTables_[0])
        .build(fw->device());
}

DLSSModuleContext::DLSSModuleContext(std::shared_ptr<FrameworkContext> frameworkContext,
                                     std::shared_ptr<WorldPipelineContext> worldPipelineContext,
                                     std::shared_ptr<DLSSModule> dlssModule)
    : WorldModuleContext(frameworkContext, worldPipelineContext),
      dLSSModule(dlssModule),
      hdrImage(dlssModule->hdrImages_[frameworkContext->frameIndex]),
      diffuseAlbedoImage(dlssModule->diffuseAlbedoImages_[frameworkContext->frameIndex]),
      specularAlbedoImage(dlssModule->specularAlbedoImages_[frameworkContext->frameIndex]),
      normalRoughnessImage(dlssModule->normalRoughnessImages_[frameworkContext->frameIndex]),
      motionVectorImage(dlssModule->motionVectorImages_[frameworkContext->frameIndex]),
      linearDepthImage(dlssModule->linearDepthImages_[frameworkContext->frameIndex]),
      specularHitDepthImage(dlssModule->specularHitDepthImages_[frameworkContext->frameIndex]),
      firstHitDepthImage(dlssModule->firstHitDepthImages_[frameworkContext->frameIndex]),
      processedImage(dlssModule->processedImages_[frameworkContext->frameIndex]),
      upscaledFirstHitDepthImage(dlssModule->upscaledFirstHitDepthImages_[frameworkContext->frameIndex]) {}

void DLSSModuleContext::render() {
    auto context = frameworkContext.lock();
    auto framework = context->framework.lock();
    auto worldCommandBuffer = context->worldCommandBuffer;
    auto mainQueueIndex = framework->physicalDevice()->mainQueueIndex();

    auto module = dLSSModule.lock();
    if (!module) return;

    {
        worldCommandBuffer->barriersBufferImage(
            {}, {{
                     .srcStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR |
                                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                     .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                     .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                                     VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                     .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                     .oldLayout = hdrImage->imageLayout(),
                     .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                     .srcQueueFamilyIndex = mainQueueIndex,
                     .dstQueueFamilyIndex = mainQueueIndex,
                     .image = hdrImage,
                     .subresourceRange = vk::wholeColorSubresourceRange,
                 },
                 {
                     .srcStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                     .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                     .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                     .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                     .oldLayout = diffuseAlbedoImage->imageLayout(),
                     .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                     .srcQueueFamilyIndex = mainQueueIndex,
                     .dstQueueFamilyIndex = mainQueueIndex,
                     .image = diffuseAlbedoImage,
                     .subresourceRange = vk::wholeColorSubresourceRange,
                 },
                 {
                     .srcStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                     .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                     .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                     .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                     .oldLayout = specularAlbedoImage->imageLayout(),
                     .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                     .srcQueueFamilyIndex = mainQueueIndex,
                     .dstQueueFamilyIndex = mainQueueIndex,
                     .image = specularAlbedoImage,
                     .subresourceRange = vk::wholeColorSubresourceRange,
                 },
                 {
                     .srcStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                     .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                     .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                     .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                     .oldLayout = normalRoughnessImage->imageLayout(),
                     .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                     .srcQueueFamilyIndex = mainQueueIndex,
                     .dstQueueFamilyIndex = mainQueueIndex,
                     .image = normalRoughnessImage,
                     .subresourceRange = vk::wholeColorSubresourceRange,
                 },
                 {
                     .srcStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                     .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                     .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                     .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                     .oldLayout = motionVectorImage->imageLayout(),
                     .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                     .srcQueueFamilyIndex = mainQueueIndex,
                     .dstQueueFamilyIndex = mainQueueIndex,
                     .image = motionVectorImage,
                     .subresourceRange = vk::wholeColorSubresourceRange,
                 },
                 {
                     .srcStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                     .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                     .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                     .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                     .oldLayout = linearDepthImage->imageLayout(),
                     .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                     .srcQueueFamilyIndex = mainQueueIndex,
                     .dstQueueFamilyIndex = mainQueueIndex,
                     .image = linearDepthImage,
                     .subresourceRange = vk::wholeColorSubresourceRange,
                 },
                 {
                     .srcStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR |
                                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                     .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                     .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                                     VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                     .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                     .oldLayout = processedImage->imageLayout(),
                     .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                     .srcQueueFamilyIndex = mainQueueIndex,
                     .dstQueueFamilyIndex = mainQueueIndex,
                     .image = processedImage,
                     .subresourceRange = vk::wholeColorSubresourceRange,
                 }});
        hdrImage->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;
        diffuseAlbedoImage->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;
        specularAlbedoImage->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;
        normalRoughnessImage->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;
        motionVectorImage->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;
        linearDepthImage->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;
        processedImage->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;

        module->dlss_->setResource(DlssRR::RESOURCE_COLOR_IN, hdrImage);
        module->dlss_->setResource(DlssRR::RESOURCE_COLOR_OUT, processedImage);
        module->dlss_->setResource(DlssRR::RESOURCE_DIFFUSE_ALBEDO, diffuseAlbedoImage);
        module->dlss_->setResource(DlssRR::RESOURCE_SPECULAR_ALBEDO, specularAlbedoImage);
        module->dlss_->setResource(DlssRR::RESOURCE_NORMALROUGHNESS, normalRoughnessImage);
        module->dlss_->setResource(DlssRR::RESOURCE_MOTIONVECTOR, motionVectorImage);
        module->dlss_->setResource(DlssRR::RESOURCE_LINEARDEPTH, linearDepthImage);
        module->dlss_->setResource(DlssRR::RESOURCE_SPECULAR_HITDISTANCE, specularHitDepthImage);

        auto worldUBOBuffer = Renderer::instance().buffers()->worldUniformBuffer();
        auto worldUBO = static_cast<vk::Data::WorldUBO *>(worldUBOBuffer->mappedPtr());
        if (worldUBO != nullptr) {
            glm::vec2 jitter = worldUBO->cameraJitter;
            module->dlss_->denoise(worldCommandBuffer, glm::uvec2{module->inputWidth_, module->inputHeight_}, jitter,
                                   worldUBO->cameraViewMat, worldUBO->cameraProjMat);
        }
    }

    // Output Scale 2x: Lanczos downscale from 2x intermediate to 1x shared output
    if (finalOutputImage && module->lanczosPipeline_) {
        struct LanczosPushConstant {
            int32_t srcWidth, srcHeight;
            int32_t dstWidth, dstHeight;
        };

        // Barrier: 2x output → shader read, 1x final → general (for storage write)
        worldCommandBuffer->barriersBufferImage(
            {}, {
                {.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                 .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                 .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                 .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
                 .oldLayout = processedImage->imageLayout(),
                 .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                 .srcQueueFamilyIndex = mainQueueIndex,
                 .dstQueueFamilyIndex = mainQueueIndex,
                 .image = processedImage,
                 .subresourceRange = vk::wholeColorSubresourceRange},
                {.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                 .srcAccessMask = 0,
                 .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                 .dstAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
                 .oldLayout = finalOutputImage->imageLayout(),
                 .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                 .srcQueueFamilyIndex = mainQueueIndex,
                 .dstQueueFamilyIndex = mainQueueIndex,
                 .image = finalOutputImage,
                 .subresourceRange = vk::wholeColorSubresourceRange}});
        processedImage->imageLayout() = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        finalOutputImage->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;

        auto frameIndex = context->frameIndex;
        auto lanczosTable = module->lanczosDescriptorTables_[frameIndex];
        lanczosTable->bindSamplerImageForShader(module->lanczosSampler_, processedImage, 0, 0);
        lanczosTable->bindImage(finalOutputImage, VK_IMAGE_LAYOUT_GENERAL, 0, 1);

        LanczosPushConstant pc{};
        pc.srcWidth = static_cast<int32_t>(module->dlssOutputWidth_);
        pc.srcHeight = static_cast<int32_t>(module->dlssOutputHeight_);
        pc.dstWidth = static_cast<int32_t>(module->outputWidth_);
        pc.dstHeight = static_cast<int32_t>(module->outputHeight_);

        worldCommandBuffer->bindDescriptorTable(lanczosTable, VK_PIPELINE_BIND_POINT_COMPUTE)
            ->bindComputePipeline(module->lanczosPipeline_);

        vkCmdPushConstants(worldCommandBuffer->vkCommandBuffer(), lanczosTable->vkPipelineLayout(),
                           VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(LanczosPushConstant), &pc);

        vkCmdDispatch(worldCommandBuffer->vkCommandBuffer(),
                      (module->outputWidth_ + 15) / 16, (module->outputHeight_ + 15) / 16, 1);

        // Post-dispatch barrier: Lanczos output visible to downstream
        worldCommandBuffer->barriersBufferImage(
            {}, {
                {.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                 .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
                 .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                 .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_MEMORY_READ_BIT,
                 .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
                 .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                 .srcQueueFamilyIndex = mainQueueIndex,
                 .dstQueueFamilyIndex = mainQueueIndex,
                 .image = finalOutputImage,
                 .subresourceRange = vk::wholeColorSubresourceRange}});
    }

    worldCommandBuffer->barriersBufferImage(
        {}, {{
                 .srcStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR |
                                 VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                 .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                 .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                                 VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                 .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                 .oldLayout = firstHitDepthImage->imageLayout(),
                 .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                 .srcQueueFamilyIndex = mainQueueIndex,
                 .dstQueueFamilyIndex = mainQueueIndex,
                 .image = firstHitDepthImage,
                 .subresourceRange = vk::wholeColorSubresourceRange,
             },
             {
                 .srcStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                 .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                 .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                 .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                 .oldLayout = upscaledFirstHitDepthImage->imageLayout(),
                 .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                 .srcQueueFamilyIndex = mainQueueIndex,
                 .dstQueueFamilyIndex = mainQueueIndex,
                 .image = upscaledFirstHitDepthImage,
                 .subresourceRange = vk::wholeColorSubresourceRange,
             }});
    firstHitDepthImage->imageLayout() = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    upscaledFirstHitDepthImage->imageLayout() = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;

    // TODO: add to command buffer
    VkImageBlit imageBlit{};
    imageBlit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    imageBlit.srcSubresource.mipLevel = 0;
    imageBlit.srcSubresource.baseArrayLayer = 0;
    imageBlit.srcSubresource.layerCount = 1;
    imageBlit.srcOffsets[0] = {0, 0, 0};
    imageBlit.srcOffsets[1] = {static_cast<int>(firstHitDepthImage->width()),
                               static_cast<int>(firstHitDepthImage->height()), 1};
    imageBlit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    imageBlit.dstSubresource.mipLevel = 0;
    imageBlit.dstSubresource.baseArrayLayer = 0;
    imageBlit.dstSubresource.layerCount = 1;
    imageBlit.dstOffsets[0] = {0, 0, 0};
    imageBlit.dstOffsets[1] = {static_cast<int>(upscaledFirstHitDepthImage->width()),
                               static_cast<int>(upscaledFirstHitDepthImage->height()), 1};

    vkCmdBlitImage(worldCommandBuffer->vkCommandBuffer(), firstHitDepthImage->vkImage(),
                   firstHitDepthImage->imageLayout(), upscaledFirstHitDepthImage->vkImage(),
                   upscaledFirstHitDepthImage->imageLayout(), 1, &imageBlit, VK_FILTER_LINEAR);
}