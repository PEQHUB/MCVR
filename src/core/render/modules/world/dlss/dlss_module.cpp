#include "core/render/modules/world/dlss/dlss_module.hpp"

#include "core/render/buffers.hpp"
#include "core/render/gpu_diagnostics.hpp"
#include "core/render/pipeline.hpp"
#include "core/render/render_framework.hpp"
#include "core/render/renderer.hpp"

#include <algorithm>

std::shared_ptr<NgxContext> DLSSModule::ngxContext_ = nullptr;

bool DLSSModule::initNGXContext() {
    auto framework = Renderer::instance().framework();

    // DLSS requires an NVIDIA GPU — skip entirely on AMD/Intel to avoid
    // crashes in NVSDK_NGX_VULKAN_Init (DLL loader fails before returning).
    uint32_t vendorID = framework->physicalDevice()->properties().vendorID;
    if (vendorID != 0x10de) {
        std::cerr << "[DLSS] Skipping — GPU vendor 0x" << std::hex << vendorID
                  << std::dec << " is not NVIDIA (0x10de)" << std::endl;
        return false;
    }

    std::filesystem::path dlssPath = Renderer::folderPath / "dlss";
    std::error_code ec;
    if (!std::filesystem::create_directories(dlssPath, ec)) {
        if (ec) {
            std::cerr << "[DLSS] Failed to create directory: " << ec.message() << std::endl;
            return false;
        }
    }

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
    diffuseRayDirHitDistImages_.resize(size);
    specularRayDirHitDistImages_.resize(size);
    reflectionMvImages_.resize(size);
    animatedTexMaskImages_.resize(size);
    particleMaskImages_.resize(size);
    firstHitBaseEmissionImages_.resize(size);
    biasMaskImages_.resize(size);
    rtHitDistImages_.resize(size);
    motionVectors3DImages_.resize(size);
    gbufferMetallicImages_.resize(size);
    gbufferShadingModelIdImages_.resize(size);
    gbufferMaterialIdImages_.resize(size);
    positionViewSpaceImages_.resize(size);
    transparencyLayerImages_.resize(size);
    transparencyLayerOpacityImages_.resize(size);
    transparencyLayerMvecsImages_.resize(size);
    processedImages_.resize(size);
    upscaledFirstHitDepthImages_.resize(size);
    upscaled2xImages_.resize(size);

    dlss_ = DlssRR::create();
}

bool DLSSModule::setOrCreateInputImages(std::vector<std::shared_ptr<vk::DeviceLocalImage>> &images,
                                        std::vector<VkFormat> &formats,
                                        uint32_t frameIndex) {
    auto framework = framework_.lock();
    if (!framework) return false;
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
        float scale = static_cast<float>(std::clamp(Renderer::options.upscalerResOverride, 1u, 100u)) / 100.0f;
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
    createOrResize(8); diffuseRayDirHitDistImages_[frameIndex] = images[8];
    createOrResize(9); specularRayDirHitDistImages_[frameIndex] = images[9];
    createOrResize(10); reflectionMvImages_[frameIndex] = images[10];
    createOrResize(11); animatedTexMaskImages_[frameIndex] = images[11];
    createOrResize(12); particleMaskImages_[frameIndex] = images[12];
    createOrResize(13); firstHitBaseEmissionImages_[frameIndex] = images[13];
    createOrResize(14); biasMaskImages_[frameIndex] = images[14];
    createOrResize(15); rtHitDistImages_[frameIndex] = images[15];
    createOrResize(16); motionVectors3DImages_[frameIndex] = images[16];
    createOrResize(17); gbufferMetallicImages_[frameIndex] = images[17];
    createOrResize(18); gbufferShadingModelIdImages_[frameIndex] = images[18];
    createOrResize(19); gbufferMaterialIdImages_[frameIndex] = images[19];
    createOrResize(20); positionViewSpaceImages_[frameIndex] = images[20];
    createOrResize(21); transparencyLayerImages_[frameIndex] = images[21];
    createOrResize(22); transparencyLayerOpacityImages_[frameIndex] = images[22];
    createOrResize(23); transparencyLayerMvecsImages_[frameIndex] = images[23];

    return true;
}

bool DLSSModule::setOrCreateOutputImages(std::vector<std::shared_ptr<vk::DeviceLocalImage>> &images,
                                         std::vector<VkFormat> &formats,
                                         uint32_t frameIndex) {
    auto framework = framework_.lock();
    if (!framework) return false;
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

    // Output Scale 2x: DLSS targets 2x, EASU downscales to 1x
    dlssOutputWidth_ = Renderer::options.outputScale2x ? outputWidth_ * 2 : outputWidth_;
    dlssOutputHeight_ = Renderer::options.outputScale2x ? outputHeight_ * 2 : outputHeight_;

    NgxContext::DlssRRInitInfo dlssRRInitInfo{};
    dlssRRInitInfo.inputSize = {inputWidth_, inputHeight_};
    dlssRRInitInfo.outputSize = {dlssOutputWidth_, dlssOutputHeight_};
    dlssRRInitInfo.quality = mode_;
    dlssRRInitInfo.preset = static_cast<NVSDK_NGX_RayReconstruction_Hint_Render_Preset>(Renderer::options.upscalerPreset);
    ngxContext_->initDlssRR(dlssRRInitInfo, framework->mainCommandPool(), dlss_);

    if (Renderer::options.outputScale2x) {
        initEasuResources();
    }

    contexts_.resize(size);

    for (int i = 0; i < size; i++) {
        auto ctx = DLSSModuleContext::create(framework->contexts()[i], worldPipeline->contexts()[i], shared_from_this());

        // Output Scale 2x: redirect DLSS output to 2x intermediate
        if (Renderer::options.outputScale2x && upscaled2xImages_[i]) {
            ctx->processedImage = upscaled2xImages_[i];          // DLSS writes to 2x
            ctx->finalOutputImage = processedImages_[i];          // EASU target (shared 1x)
        } else {
            ctx->finalOutputImage = nullptr;
        }

        contexts_[i] = ctx;
    }

    // Publish render-res HDR (DLSS input) for tone mapping histogram metering.
    // The histogram must read pre-DLSS data to get accurate scene brightness,
    // because DLSS-RR may attenuate extreme HDR values during neural upscaling.
    Renderer::renderResHdrImages = hdrImages_;
}

std::vector<std::shared_ptr<WorldModuleContext>> &DLSSModule::contexts() {
    return contexts_;
}

void DLSSModule::bindTexture(std::shared_ptr<vk::Sampler> sampler,
                             std::shared_ptr<vk::DeviceLocalImage> image,
                             int index) {}

void DLSSModule::preClose() {
    if (dlss_) dlss_->deinit();

    // EASU resources cleanup
    easuPipeline_.reset();
    easuDescriptorTables_.clear();
    easuSampler_.reset();
    easuShader_.reset();
    upscaled2xImages_.clear();
}

// Compute EASU constants (mirrors ffxFsrPopulateEasuConstants from ffx_fsr1.h)
static void computeEasuConstants(uint32_t con0[4], uint32_t con1[4], uint32_t con2[4], uint32_t con3[4],
                                  float srcW, float srcH, float dstW, float dstH) {
    auto asUint = [](float f) -> uint32_t { uint32_t u; memcpy(&u, &f, 4); return u; };
    con0[0] = asUint(srcW / dstW);
    con0[1] = asUint(srcH / dstH);
    con0[2] = asUint(0.5f * srcW / dstW - 0.5f);
    con0[3] = asUint(0.5f * srcH / dstH - 0.5f);
    con1[0] = asUint(1.0f / srcW);
    con1[1] = asUint(1.0f / srcH);
    con1[2] = asUint(1.0f / srcW);
    con1[3] = asUint(-1.0f / srcH);
    con2[0] = asUint(-1.0f / srcW);
    con2[1] = asUint(2.0f / srcH);
    con2[2] = asUint(1.0f / srcW);
    con2[3] = asUint(2.0f / srcH);
    con3[0] = asUint(0.0f / srcW);
    con3[1] = asUint(4.0f / srcH);
    con3[2] = 0; con3[3] = 0;
}

void DLSSModule::initEasuResources() {
    auto fw = framework_.lock();
    if (!fw) return;
    uint32_t size = fw->swapchain()->imageCount();

    // Create 2x intermediate images for DLSS output
    for (uint32_t i = 0; i < size; i++) {
        upscaled2xImages_[i] = vk::DeviceLocalImage::create(
            fw->device(), fw->vma(), false, dlssOutputWidth_, dlssOutputHeight_, 1,
            VK_FORMAT_R16G16B16A16_SFLOAT,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    }

    // Load FSR1 EASU compute shader
    easuShader_ = vk::Shader::create(
        fw->device(),
        (Renderer::folderPath / "shaders/world/post_render/fsr1_easu_downscale_comp.spv").string());

    // EASU uses textureGather which needs LINEAR sampler
    easuSampler_ = vk::Sampler::create(
        fw->device(), VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);

    // Descriptor tables
    easuDescriptorTables_.resize(size);
    for (uint32_t i = 0; i < size; i++) {
        easuDescriptorTables_[i] = vk::DescriptorTableBuilder{}
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
                .size = sizeof(uint32_t) * 16,  // 4 x uvec4 = 64 bytes
            })
            .build(fw->device());
    }

    // Compute pipeline
    easuPipeline_ = vk::ComputePipelineBuilder{}
        .defineShader(easuShader_)
        .definePipelineLayout(easuDescriptorTables_[0])
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
      diffuseRayDirHitDistImage(dlssModule->diffuseRayDirHitDistImages_[frameworkContext->frameIndex]),
      specularRayDirHitDistImage(dlssModule->specularRayDirHitDistImages_[frameworkContext->frameIndex]),
      reflectionMvImage(dlssModule->reflectionMvImages_[frameworkContext->frameIndex]),
      animatedTexMaskImage(dlssModule->animatedTexMaskImages_[frameworkContext->frameIndex]),
      particleMaskImage(dlssModule->particleMaskImages_[frameworkContext->frameIndex]),
      firstHitBaseEmissionImage(dlssModule->firstHitBaseEmissionImages_[frameworkContext->frameIndex]),
      biasMaskImage(dlssModule->biasMaskImages_[frameworkContext->frameIndex]),
      rtHitDistImage(dlssModule->rtHitDistImages_[frameworkContext->frameIndex]),
      motionVectors3DImage(dlssModule->motionVectors3DImages_[frameworkContext->frameIndex]),
      gbufferMetallicImage(dlssModule->gbufferMetallicImages_[frameworkContext->frameIndex]),
      gbufferShadingModelIdImage(dlssModule->gbufferShadingModelIdImages_[frameworkContext->frameIndex]),
      gbufferMaterialIdImage(dlssModule->gbufferMaterialIdImages_[frameworkContext->frameIndex]),
      positionViewSpaceImage(dlssModule->positionViewSpaceImages_[frameworkContext->frameIndex]),
      transparencyLayerImage(dlssModule->transparencyLayerImages_[frameworkContext->frameIndex]),
      transparencyLayerOpacityImage(dlssModule->transparencyLayerOpacityImages_[frameworkContext->frameIndex]),
      transparencyLayerMvecsImage(dlssModule->transparencyLayerMvecsImages_[frameworkContext->frameIndex]),
      processedImage(dlssModule->processedImages_[frameworkContext->frameIndex]),
      upscaledFirstHitDepthImage(dlssModule->upscaledFirstHitDepthImages_[frameworkContext->frameIndex]) {}

void DLSSModuleContext::render() {
    auto context = frameworkContext.lock();
    if (!context) return;
    auto framework = context->framework.lock();
    if (!framework) return;
    auto worldCommandBuffer = context->worldCommandBuffer;
    auto mainQueueIndex = framework->physicalDevice()->mainQueueIndex();

    auto module = dLSSModule.lock();
    if (!module) return;

    // Raw Fast / Raw Accurate: bypass DLSS, blit accumulated RT output to DLSS output
    if (Renderer::options.offlineState == 2 && Renderer::options.offlineDenoised != 2) {
        auto cmd = worldCommandBuffer->vkCommandBuffer();
        // Transition images for blit
        worldCommandBuffer->barriersBufferImage({}, {
            {.srcStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
             .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
             .dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
             .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT,
             .oldLayout = hdrImage->imageLayout(), .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
             .srcQueueFamilyIndex = mainQueueIndex, .dstQueueFamilyIndex = mainQueueIndex,
             .image = hdrImage, .subresourceRange = vk::wholeColorSubresourceRange},
            {.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, .srcAccessMask = 0,
             .dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
             .dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
             .oldLayout = processedImage->imageLayout(), .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
             .srcQueueFamilyIndex = mainQueueIndex, .dstQueueFamilyIndex = mainQueueIndex,
             .image = processedImage, .subresourceRange = vk::wholeColorSubresourceRange}});

        VkImageBlit region{};
        region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.srcOffsets[1] = {(int)hdrImage->width(), (int)hdrImage->height(), 1};
        region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.dstOffsets[1] = {(int)processedImage->width(), (int)processedImage->height(), 1};
        vkCmdBlitImage(cmd, hdrImage->vkImage(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       processedImage->vkImage(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                       1, &region, VK_FILTER_LINEAR);

        hdrImage->imageLayout() = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        processedImage->imageLayout() = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;

        // Also blit first_hit_depth -> upscaledFirstHitDepth (post-render needs it)
        if (firstHitDepthImage && upscaledFirstHitDepthImage) {
            worldCommandBuffer->barriersBufferImage({}, {
                {.srcStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                 .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
                 .dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                 .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT,
                 .oldLayout = firstHitDepthImage->imageLayout(), .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                 .srcQueueFamilyIndex = mainQueueIndex, .dstQueueFamilyIndex = mainQueueIndex,
                 .image = firstHitDepthImage, .subresourceRange = vk::wholeColorSubresourceRange},
                {.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, .srcAccessMask = 0,
                 .dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                 .dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                 .oldLayout = upscaledFirstHitDepthImage->imageLayout(), .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                 .srcQueueFamilyIndex = mainQueueIndex, .dstQueueFamilyIndex = mainQueueIndex,
                 .image = upscaledFirstHitDepthImage, .subresourceRange = vk::wholeColorSubresourceRange}});
            VkImageBlit depthRegion{};
            depthRegion.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            depthRegion.srcOffsets[1] = {(int)firstHitDepthImage->width(), (int)firstHitDepthImage->height(), 1};
            depthRegion.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            depthRegion.dstOffsets[1] = {(int)upscaledFirstHitDepthImage->width(), (int)upscaledFirstHitDepthImage->height(), 1};
            vkCmdBlitImage(cmd, firstHitDepthImage->vkImage(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           upscaledFirstHitDepthImage->vkImage(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                           1, &depthRegion, VK_FILTER_LINEAR);
            firstHitDepthImage->imageLayout() = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            upscaledFirstHitDepthImage->imageLayout() = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        }
        return;
    }

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

        // DLSS-RR guide buffers: barriers
        worldCommandBuffer->barriersBufferImage(
            {}, {{
                     .srcStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                     .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                     .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                     .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                     .oldLayout = diffuseRayDirHitDistImage->imageLayout(),
                     .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                     .srcQueueFamilyIndex = mainQueueIndex,
                     .dstQueueFamilyIndex = mainQueueIndex,
                     .image = diffuseRayDirHitDistImage,
                     .subresourceRange = vk::wholeColorSubresourceRange,
                 },
                 {
                     .srcStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                     .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                     .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                     .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                     .oldLayout = specularRayDirHitDistImage->imageLayout(),
                     .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                     .srcQueueFamilyIndex = mainQueueIndex,
                     .dstQueueFamilyIndex = mainQueueIndex,
                     .image = specularRayDirHitDistImage,
                     .subresourceRange = vk::wholeColorSubresourceRange,
                 }});
        diffuseRayDirHitDistImage->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;
        specularRayDirHitDistImage->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;

        // Reflection MV is optional — guard against null if pipeline connection is missing
        if (reflectionMvImage) {
            worldCommandBuffer->barriersBufferImage(
                {}, {{
                         .srcStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                         .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                         .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                         .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                         .oldLayout = reflectionMvImage->imageLayout(),
                         .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                         .srcQueueFamilyIndex = mainQueueIndex,
                         .dstQueueFamilyIndex = mainQueueIndex,
                         .image = reflectionMvImage,
                         .subresourceRange = vk::wholeColorSubresourceRange,
                     }});
            reflectionMvImage->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;
        }

        // Animated texture mask is optional — guard against null if format creation failed
        if (animatedTexMaskImage) {
            worldCommandBuffer->barriersBufferImage(
                {}, {{
                         .srcStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                         .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                         .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                         .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                         .oldLayout = animatedTexMaskImage->imageLayout(),
                         .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                         .srcQueueFamilyIndex = mainQueueIndex,
                         .dstQueueFamilyIndex = mainQueueIndex,
                         .image = animatedTexMaskImage,
                         .subresourceRange = vk::wholeColorSubresourceRange,
                     }});
            animatedTexMaskImage->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;
        }

        module->dlss_->setResource(DlssRR::RESOURCE_COLOR_IN, hdrImage);
        module->dlss_->setResource(DlssRR::RESOURCE_COLOR_OUT, processedImage);
        module->dlss_->setResource(DlssRR::RESOURCE_DIFFUSE_ALBEDO, diffuseAlbedoImage);
        module->dlss_->setResource(DlssRR::RESOURCE_SPECULAR_ALBEDO, specularAlbedoImage);
        module->dlss_->setResource(DlssRR::RESOURCE_NORMALROUGHNESS, normalRoughnessImage);
        module->dlss_->setResource(DlssRR::RESOURCE_MOTIONVECTOR, motionVectorImage);
        module->dlss_->setResource(DlssRR::RESOURCE_LINEARDEPTH, linearDepthImage);
        module->dlss_->setResource(DlssRR::RESOURCE_SPECULAR_HITDISTANCE, specularHitDepthImage);
        module->dlss_->setResource(DlssRR::RESOURCE_DIFFUSE_RAY_DIR_HIT_DIST, diffuseRayDirHitDistImage);
        module->dlss_->setResource(DlssRR::RESOURCE_SPECULAR_RAY_DIR_HIT_DIST, specularRayDirHitDistImage);
        if (reflectionMvImage)
            module->dlss_->setResource(DlssRR::RESOURCE_REFLECTION_MV, reflectionMvImage);
        if (animatedTexMaskImage)
            module->dlss_->setResource(DlssRR::RESOURCE_ANIMATED_TEX_MASK, animatedTexMaskImage);

        // Extended DLSS-RR guide buffers — all optional
        if (particleMaskImage) {
            worldCommandBuffer->barriersBufferImage(
                {}, {{
                         .srcStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                         .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                         .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                         .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                         .oldLayout = particleMaskImage->imageLayout(),
                         .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                         .srcQueueFamilyIndex = mainQueueIndex,
                         .dstQueueFamilyIndex = mainQueueIndex,
                         .image = particleMaskImage,
                         .subresourceRange = vk::wholeColorSubresourceRange,
                     }});
            particleMaskImage->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;
            module->dlss_->setResource(DlssRR::RESOURCE_PARTICLE_MASK, particleMaskImage);
        }
        if (firstHitBaseEmissionImage) {
            worldCommandBuffer->barriersBufferImage(
                {}, {{
                         .srcStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                         .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                         .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                         .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                         .oldLayout = firstHitBaseEmissionImage->imageLayout(),
                         .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                         .srcQueueFamilyIndex = mainQueueIndex,
                         .dstQueueFamilyIndex = mainQueueIndex,
                         .image = firstHitBaseEmissionImage,
                         .subresourceRange = vk::wholeColorSubresourceRange,
                     }});
            firstHitBaseEmissionImage->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;
            module->dlss_->setResource(DlssRR::RESOURCE_EMISSIVE, firstHitBaseEmissionImage);
        }
        if (biasMaskImage) {
            worldCommandBuffer->barriersBufferImage(
                {}, {{
                         .srcStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                         .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                         .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                         .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                         .oldLayout = biasMaskImage->imageLayout(),
                         .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                         .srcQueueFamilyIndex = mainQueueIndex,
                         .dstQueueFamilyIndex = mainQueueIndex,
                         .image = biasMaskImage,
                         .subresourceRange = vk::wholeColorSubresourceRange,
                     }});
            biasMaskImage->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;
            module->dlss_->setResource(DlssRR::RESOURCE_BIAS_MASK, biasMaskImage);
        }
        if (rtHitDistImage) {
            worldCommandBuffer->barriersBufferImage(
                {}, {{
                         .srcStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                         .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                         .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                         .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                         .oldLayout = rtHitDistImage->imageLayout(),
                         .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                         .srcQueueFamilyIndex = mainQueueIndex,
                         .dstQueueFamilyIndex = mainQueueIndex,
                         .image = rtHitDistImage,
                         .subresourceRange = vk::wholeColorSubresourceRange,
                     }});
            rtHitDistImage->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;
            module->dlss_->setResource(DlssRR::RESOURCE_RT_HIT_DIST, rtHitDistImage);
        }
        if (motionVectors3DImage) {
            worldCommandBuffer->barriersBufferImage(
                {}, {{
                         .srcStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                         .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                         .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                         .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                         .oldLayout = motionVectors3DImage->imageLayout(),
                         .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                         .srcQueueFamilyIndex = mainQueueIndex,
                         .dstQueueFamilyIndex = mainQueueIndex,
                         .image = motionVectors3DImage,
                         .subresourceRange = vk::wholeColorSubresourceRange,
                     }});
            motionVectors3DImage->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;
            module->dlss_->setResource(DlssRR::RESOURCE_MOTION_VECTORS_3D, motionVectors3DImage);
        }
        if (gbufferMetallicImage) {
            worldCommandBuffer->barriersBufferImage(
                {}, {{
                         .srcStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                         .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                         .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                         .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                         .oldLayout = gbufferMetallicImage->imageLayout(),
                         .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                         .srcQueueFamilyIndex = mainQueueIndex,
                         .dstQueueFamilyIndex = mainQueueIndex,
                         .image = gbufferMetallicImage,
                         .subresourceRange = vk::wholeColorSubresourceRange,
                     }});
            gbufferMetallicImage->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;
            module->dlss_->setResource(DlssRR::RESOURCE_GBUFFER_METALLIC, gbufferMetallicImage);
        }
        if (gbufferShadingModelIdImage) {
            worldCommandBuffer->barriersBufferImage(
                {}, {{
                         .srcStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                         .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                         .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                         .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                         .oldLayout = gbufferShadingModelIdImage->imageLayout(),
                         .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                         .srcQueueFamilyIndex = mainQueueIndex,
                         .dstQueueFamilyIndex = mainQueueIndex,
                         .image = gbufferShadingModelIdImage,
                         .subresourceRange = vk::wholeColorSubresourceRange,
                     }});
            gbufferShadingModelIdImage->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;
            module->dlss_->setResource(DlssRR::RESOURCE_GBUFFER_SHADING_MODEL_ID, gbufferShadingModelIdImage);
        }
        if (gbufferMaterialIdImage) {
            worldCommandBuffer->barriersBufferImage(
                {}, {{
                         .srcStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                         .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                         .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                         .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                         .oldLayout = gbufferMaterialIdImage->imageLayout(),
                         .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                         .srcQueueFamilyIndex = mainQueueIndex,
                         .dstQueueFamilyIndex = mainQueueIndex,
                         .image = gbufferMaterialIdImage,
                         .subresourceRange = vk::wholeColorSubresourceRange,
                     }});
            gbufferMaterialIdImage->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;
            module->dlss_->setResource(DlssRR::RESOURCE_GBUFFER_MATERIAL_ID, gbufferMaterialIdImage);
        }
        if (positionViewSpaceImage) {
            worldCommandBuffer->barriersBufferImage(
                {}, {{
                         .srcStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                         .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                         .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                         .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                         .oldLayout = positionViewSpaceImage->imageLayout(),
                         .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                         .srcQueueFamilyIndex = mainQueueIndex,
                         .dstQueueFamilyIndex = mainQueueIndex,
                         .image = positionViewSpaceImage,
                         .subresourceRange = vk::wholeColorSubresourceRange,
                     }});
            positionViewSpaceImage->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;
            module->dlss_->setResource(DlssRR::RESOURCE_POSITION_VIEW_SPACE, positionViewSpaceImage);
        }
        // Transparency layer (stable planes for glass/water)
        if (transparencyLayerImage) {
            worldCommandBuffer->barriersBufferImage(
                {}, {{
                         .srcStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                         .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                         .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                         .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                         .oldLayout = transparencyLayerImage->imageLayout(),
                         .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                         .srcQueueFamilyIndex = mainQueueIndex,
                         .dstQueueFamilyIndex = mainQueueIndex,
                         .image = transparencyLayerImage,
                         .subresourceRange = vk::wholeColorSubresourceRange,
                     }});
            transparencyLayerImage->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;
            module->dlss_->setResource(DlssRR::RESOURCE_TRANSPARENCY_LAYER, transparencyLayerImage);
        }
        if (transparencyLayerOpacityImage) {
            worldCommandBuffer->barriersBufferImage(
                {}, {{
                         .srcStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                         .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                         .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                         .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                         .oldLayout = transparencyLayerOpacityImage->imageLayout(),
                         .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                         .srcQueueFamilyIndex = mainQueueIndex,
                         .dstQueueFamilyIndex = mainQueueIndex,
                         .image = transparencyLayerOpacityImage,
                         .subresourceRange = vk::wholeColorSubresourceRange,
                     }});
            transparencyLayerOpacityImage->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;
            module->dlss_->setResource(DlssRR::RESOURCE_TRANSPARENCY_LAYER_OPACITY, transparencyLayerOpacityImage);
        }
        if (transparencyLayerMvecsImage) {
            worldCommandBuffer->barriersBufferImage(
                {}, {{
                         .srcStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                         .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                         .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                         .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                         .oldLayout = transparencyLayerMvecsImage->imageLayout(),
                         .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                         .srcQueueFamilyIndex = mainQueueIndex,
                         .dstQueueFamilyIndex = mainQueueIndex,
                         .image = transparencyLayerMvecsImage,
                         .subresourceRange = vk::wholeColorSubresourceRange,
                     }});
            transparencyLayerMvecsImage->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;
            module->dlss_->setResource(DlssRR::RESOURCE_TRANSPARENCY_LAYER_MVECS, transparencyLayerMvecsImage);
        }

        auto worldUBOBuffer = Renderer::instance().buffers()->worldUniformBuffer();
        auto worldUBO = static_cast<vk::Data::WorldUBO *>(worldUBOBuffer->mappedPtr());
        if (worldUBO != nullptr) {
            glm::vec2 jitter = worldUBO->cameraJitter;
            // Per-context frame time delta for DLSS temporal motion estimation
            auto now = std::chrono::steady_clock::now();
            float frameTimeDeltaMs = std::chrono::duration<float, std::milli>(now - lastRenderTime_).count();
            lastRenderTime_ = now;
            // Keep DLSS-D/RR scene-referred. RT no longer pre-exposes by 0.1, so
            // NGX sees the same scale that the histogram and tone mapper use.
            float preExposure = 1.0f;
            // Use cameraEffectedViewMat — must match the matrix used to compute
            // linearDepthImage and motionVectorImage in world.rgen. Using cameraViewMat
            // (without view bob/camera effects) causes a depth↔matrix mismatch that
            // breaks DLSS-RR temporal reprojection.
            //
            // Denoised mode (epoch-based): reset DLSS temporal history only on the first
            // frame of each epoch. Within an epoch, DLSS-RR builds temporal history normally.
            // At epoch end, the converged output is Welford-averaged across epochs.
            bool dlssReset = (Renderer::options.offlineState == 2
                              && Renderer::options.offlineDenoised == 2
                              && Renderer::dlssEpochFrame == 0);
            GpuDiag::checkpoint(worldCommandBuffer->vkCommandBuffer(), GpuDiag::DLSS_EVALUATE);
            module->dlss_->denoise(worldCommandBuffer, glm::uvec2{module->inputWidth_, module->inputHeight_}, jitter,
                                   worldUBO->cameraEffectedViewMat, worldUBO->cameraProjMat, preExposure, dlssReset,
                                   frameTimeDeltaMs);
        }
    }

    // Denoised mode: epoch-based post-DLSS Welford accumulation
    // Each epoch = dlssEpochLength frames of DLSS-RR temporal convergence.
    // At epoch end, snapshot the converged DLSS output into Welford accumulator.
    // Between epochs, display the Welford result (stable, converging).
    if (Renderer::options.offlineState == 2 && Renderer::options.offlineDenoised == 2) {
        bool epochEnd = (Renderer::dlssEpochFrame >= Renderer::options.dlssEpochLength - 1);

        if (epochEnd && Renderer::accumPipelineReady) {
            auto cmd = worldCommandBuffer->vkCommandBuffer();

            // Barrier: processedImage → GENERAL for compute read
            worldCommandBuffer->barriersBufferImage({}, {{
                .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
                .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
                .oldLayout = processedImage->imageLayout(),
                .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                .srcQueueFamilyIndex = mainQueueIndex,
                .dstQueueFamilyIndex = mainQueueIndex,
                .image = processedImage,
                .subresourceRange = vk::wholeColorSubresourceRange,
            }});
            processedImage->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;

            // Ensure accumBufferImage is in GENERAL
            if (Renderer::accumBufferImage->imageLayout() != VK_IMAGE_LAYOUT_GENERAL) {
                worldCommandBuffer->barriersBufferImage({}, {{
                    .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                    .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
                    .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
                    .oldLayout = Renderer::accumBufferImage->imageLayout(),
                    .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                    .srcQueueFamilyIndex = mainQueueIndex,
                    .dstQueueFamilyIndex = mainQueueIndex,
                    .image = Renderer::accumBufferImage,
                    .subresourceRange = vk::wholeColorSubresourceRange,
                }});
                Renderer::accumBufferImage->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;
            }

            // Update descriptor set: bind accumBuffer and processedImage (DLSS output)
            uint32_t frameIdx = context->frameIndex % Renderer::accumDescSets.size();
            VkDescriptorImageInfo accumInfo{
                .sampler = VK_NULL_HANDLE,
                .imageView = Renderer::accumBufferImage->vkImageView(),
                .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
            };
            VkDescriptorImageInfo inputInfo{
                .sampler = VK_NULL_HANDLE,
                .imageView = processedImage->vkImageView(),
                .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
            };
            VkWriteDescriptorSet writes[2] = {
                {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                 .dstSet = Renderer::accumDescSets[frameIdx],
                 .dstBinding = 0, .dstArrayElement = 0, .descriptorCount = 1,
                 .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                 .pImageInfo = &accumInfo},
                {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                 .dstSet = Renderer::accumDescSets[frameIdx],
                 .dstBinding = 1, .dstArrayElement = 0, .descriptorCount = 1,
                 .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                 .pImageInfo = &inputInfo},
            };
            vkUpdateDescriptorSets(framework->device()->vkDevice(), 2, writes, 0, nullptr);

            // Dispatch accumulate.comp: Welford average with epoch count as N
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, Renderer::accumPipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                Renderer::accumPipelineLayout, 0, 1, &Renderer::accumDescSets[frameIdx], 0, nullptr);

            int32_t ec = static_cast<int32_t>(Renderer::dlssEpochCount);
            vkCmdPushConstants(cmd, Renderer::accumPipelineLayout,
                VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(int32_t), &ec);

            uint32_t gx = (processedImage->width() + 7) / 8;
            uint32_t gy = (processedImage->height() + 7) / 8;
            vkCmdDispatch(cmd, gx, gy, 1);

            Renderer::dlssEpochCount++;
            Renderer::dlssEpochFrame = 0;  // reset for next epoch
        } else {
            Renderer::dlssEpochFrame++;
        }

        // Display: show Welford-averaged result after first epoch completes.
        // During first epoch (no snapshots yet), live DLSS-RR output passes through.
        if (Renderer::dlssEpochCount > 0 && Renderer::accumPipelineReady) {
            auto cmd = worldCommandBuffer->vkCommandBuffer();

            // Barrier: accumBuffer → transfer src, processedImage → transfer dst
            worldCommandBuffer->barriersBufferImage({}, {
                {.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                 .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
                 .dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                 .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT,
                 .oldLayout = Renderer::accumBufferImage->imageLayout(),
                 .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                 .srcQueueFamilyIndex = mainQueueIndex,
                 .dstQueueFamilyIndex = mainQueueIndex,
                 .image = Renderer::accumBufferImage,
                 .subresourceRange = vk::wholeColorSubresourceRange},
                {.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                 .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                 .dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                 .dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                 .oldLayout = processedImage->imageLayout(),
                 .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                 .srcQueueFamilyIndex = mainQueueIndex,
                 .dstQueueFamilyIndex = mainQueueIndex,
                 .image = processedImage,
                 .subresourceRange = vk::wholeColorSubresourceRange}
            });

            // Blit Welford average → processedImage (downstream sees converged result)
            VkImageBlit blitRegion{};
            blitRegion.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            blitRegion.srcOffsets[1] = {(int)Renderer::accumBufferImage->width(),
                                        (int)Renderer::accumBufferImage->height(), 1};
            blitRegion.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            blitRegion.dstOffsets[1] = {(int)processedImage->width(),
                                        (int)processedImage->height(), 1};
            vkCmdBlitImage(cmd, Renderer::accumBufferImage->vkImage(),
                           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           processedImage->vkImage(),
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                           1, &blitRegion, VK_FILTER_LINEAR);

            Renderer::accumBufferImage->imageLayout() = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            processedImage->imageLayout() = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        }

        Renderer::accumFrameCount++;  // total frames for HUD
    }

    // Output Scale 2x: FSR1 EASU downscale from 2x intermediate to 1x shared output
    if (finalOutputImage && module->easuPipeline_) {
        struct EasuPushConstant {
            uint32_t con0[4], con1[4], con2[4], con3[4];
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
        auto easuTable = module->easuDescriptorTables_[frameIndex];
        easuTable->bindSamplerImageForShader(module->easuSampler_, processedImage, 0, 0);
        easuTable->bindImage(finalOutputImage, VK_IMAGE_LAYOUT_GENERAL, 0, 1);

        EasuPushConstant pc{};
        computeEasuConstants(pc.con0, pc.con1, pc.con2, pc.con3,
                             static_cast<float>(module->dlssOutputWidth_),
                             static_cast<float>(module->dlssOutputHeight_),
                             static_cast<float>(module->outputWidth_),
                             static_cast<float>(module->outputHeight_));

        worldCommandBuffer->bindDescriptorTable(easuTable, VK_PIPELINE_BIND_POINT_COMPUTE)
            ->bindComputePipeline(module->easuPipeline_);

        vkCmdPushConstants(worldCommandBuffer->vkCommandBuffer(), easuTable->vkPipelineLayout(),
                           VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(EasuPushConstant), &pc);

        vkCmdDispatch(worldCommandBuffer->vkCommandBuffer(),
                      (module->outputWidth_ + 15) / 16, (module->outputHeight_ + 15) / 16, 1);

        // Post-dispatch barrier: EASU output visible to downstream
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
