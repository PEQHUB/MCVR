#include "core/render/modules/world/cloud/cloud_module.hpp"

#include "core/render/buffers.hpp"
#include "core/render/pipeline.hpp"
#include "core/render/render_framework.hpp"
#include "core/render/renderer.hpp"
#include "core/render/world.hpp"

CloudModule::CloudModule() {}

void CloudModule::init(std::shared_ptr<Framework> framework, std::shared_ptr<WorldPipeline> worldPipeline) {
    WorldModule::init(framework, worldPipeline);

    uint32_t size = framework->swapchain()->imageCount();

    radianceImages_.resize(size);
    linearDepthImages_.resize(size);
    cloudRadianceImages_.resize(size);
    cloudColorImages_.resize(size);
}

bool CloudModule::setOrCreateInputImages(std::vector<std::shared_ptr<vk::DeviceLocalImage>> &images,
                                          std::vector<VkFormat> &formats,
                                          uint32_t frameIndex) {
    if (images.size() != inputImageNum) return false;

    auto framework = framework_.lock();
    if (!framework) return false;

    for (uint32_t i = 0; i < inputImageNum; i++) {
        if (images[i] == nullptr) {
            images[i] = vk::DeviceLocalImage::create(
                framework->device(), framework->vma(), false, width_, height_, 1, formats[i],
                VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
        }
    }

    radianceImages_[frameIndex] = images[0];
    linearDepthImages_[frameIndex] = images[1];

    return true;
}

bool CloudModule::setOrCreateOutputImages(std::vector<std::shared_ptr<vk::DeviceLocalImage>> &images,
                                           std::vector<VkFormat> &formats,
                                           uint32_t frameIndex) {
    if (images.size() != outputImageNum) return false;

    if (images[0] != nullptr) {
        width_ = images[0]->width();
        height_ = images[0]->height();
        cloudRadianceImages_[frameIndex] = images[0];
    } else {
        return false;
    }

    // Output 1: cloud_shadow_map (R16F, shared across frames).
    // Pipeline pre-creates this at render resolution, but we need shadowMapSize_ resolution.
    // Create once (frame 0) at correct size, then reuse for all frames.
    if (!cloudShadowImage_) {
        auto framework = framework_.lock();
        if (framework) {
            // Derive shadow map size from quality (same logic as build())
            uint32_t smSize;
            switch (Renderer::options.cloudQuality) {
                case 1:  smSize = 128; break;
                case 5:  smSize = 512; break;
                case 6:  smSize = 512; break;
                default: smSize = 256; break;
            }
            cloudShadowImage_ = vk::DeviceLocalImage::create(
                framework->device(), framework->vma(), false, smSize, smSize, 1,
                VK_FORMAT_R16_SFLOAT,
                VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
        }
    }
    // Replace the pipeline's pre-created render-res image with our correctly-sized one
    images[1] = cloudShadowImage_;

    return true;
}

void CloudModule::setAttributes(int attributeCount, std::vector<std::string> &attributeKVs) {
    for (int i = 0; i < attributeCount; i++) {
        const auto &key = attributeKVs[2 * i];
        const auto &val = attributeKVs[2 * i + 1];

        if (key == "render_pipeline.module.cloud.attribute.quality") {
            if (val == "off") {
                enabled_ = false;
            } else if (val == "low") {
                enabled_ = true;
                resolutionDivisor_ = 4; marchSteps_ = 48; lightSteps_ = 3;
                temporalBlend_ = 0.97f; shadowMapSize_ = 128;
            } else if (val == "medium") {
                enabled_ = true;
                resolutionDivisor_ = 4; marchSteps_ = 64; lightSteps_ = 4;
                temporalBlend_ = 0.95f; shadowMapSize_ = 256;
            } else if (val == "high") {
                enabled_ = true;
                resolutionDivisor_ = 2; marchSteps_ = 64; lightSteps_ = 4;
                temporalBlend_ = 0.95f; shadowMapSize_ = 256;
            } else if (val == "ultra") {
                enabled_ = true;
                resolutionDivisor_ = 2; marchSteps_ = 96; lightSteps_ = 6;
                temporalBlend_ = 0.93f; shadowMapSize_ = 256;
            } else if (val == "extreme") {
                enabled_ = true;
                resolutionDivisor_ = 1; marchSteps_ = 96; lightSteps_ = 6;
                temporalBlend_ = 0.93f; shadowMapSize_ = 512;
            }
        } else if (key == "render_pipeline.module.cloud.attribute.density") {
            densityMultiplier_ = std::stof(val);
        } else if (key == "render_pipeline.module.cloud.attribute.coverage") {
            coverage_ = std::stof(val);
        } else if (key == "render_pipeline.module.cloud.attribute.altitude") {
            cloudBase_ = std::stof(val);
        } else if (key == "render_pipeline.module.cloud.attribute.thickness") {
            cloudThickness_ = std::stof(val);
        } else if (key == "render_pipeline.module.cloud.attribute.speed") {
            windSpeed_ = std::stof(val);
        }
    }
}

void CloudModule::build() {
    auto framework = framework_.lock();
    if (!framework) return;
    auto worldPipeline = worldPipeline_.lock();
    if (!worldPipeline) return;
    uint32_t size = framework->swapchain()->imageCount();

    // Apply quality from Renderer::options (set via JNI, triggers needRecreate)
    switch (Renderer::options.cloudQuality) {
        case 0: enabled_ = false; resolutionDivisor_ = 2; marchSteps_ = 64; lightSteps_ = 4; temporalBlend_ = 0.95f; shadowMapSize_ = 256; break;
        case 1: enabled_ = true;  resolutionDivisor_ = 4; marchSteps_ = 48; lightSteps_ = 3; temporalBlend_ = 0.97f; shadowMapSize_ = 128; break;
        case 2: enabled_ = true;  resolutionDivisor_ = 4; marchSteps_ = 64; lightSteps_ = 4; temporalBlend_ = 0.95f; shadowMapSize_ = 256; break;
        case 3: enabled_ = true;  resolutionDivisor_ = 2; marchSteps_ = 64; lightSteps_ = 4; temporalBlend_ = 0.95f; shadowMapSize_ = 256; break;
        case 4: enabled_ = true;  resolutionDivisor_ = 2; marchSteps_ = 96; lightSteps_ = 6; temporalBlend_ = 0.93f; shadowMapSize_ = 256; break;
        case 5: enabled_ = true;  resolutionDivisor_ = 1; marchSteps_ = 96; lightSteps_ = 6; temporalBlend_ = 0.93f; shadowMapSize_ = 512; break;
        case 6: enabled_ = true;  resolutionDivisor_ = 1; marchSteps_ = 128; lightSteps_ = 8; temporalBlend_ = 0.90f; shadowMapSize_ = 512; break;
        default: enabled_ = true; resolutionDivisor_ = 2; marchSteps_ = 64; lightSteps_ = 4; temporalBlend_ = 0.95f; shadowMapSize_ = 256; break;
    }

    if (Renderer::options.cloudResDivisorOverride > 0) {
        resolutionDivisor_ = std::clamp(Renderer::options.cloudResDivisorOverride, 1u, 4u);
    }
    if (Renderer::options.cloudMarchStepsOverride > 0) {
        marchSteps_ = std::clamp(Renderer::options.cloudMarchStepsOverride, 1u, 256u);
    }
    if (Renderer::options.cloudLightStepsOverride > 0) {
        lightSteps_ = std::clamp(Renderer::options.cloudLightStepsOverride, 1u, 12u);
    }

    cloudWidth_ = (width_ + resolutionDivisor_ - 1) / resolutionDivisor_;
    cloudHeight_ = (height_ + resolutionDivisor_ - 1) / resolutionDivisor_;

    // Apply runtime options
    cloudBase_ = Renderer::options.cloudAltitude;
    cloudThickness_ = Renderer::options.cloudThickness;
    coverage_ = Renderer::options.cloudCoverage;
    cloudType_ = Renderer::options.cloudType;
    densityMultiplier_ = Renderer::options.cloudDensity;
    windSpeed_ = Renderer::options.cloudSpeed;

    noiseSampler_ = vk::Sampler::create(framework->device(), VK_FILTER_LINEAR,
                                         VK_SAMPLER_MIPMAP_MODE_LINEAR, VK_SAMPLER_ADDRESS_MODE_REPEAT);
    weatherSampler_ = vk::Sampler::create(framework->device(), VK_FILTER_LINEAR,
                                          VK_SAMPLER_MIPMAP_MODE_LINEAR, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);

    initDescriptorTables();
    initImages();
    initPipeline();

    contexts_.resize(size);
    for (uint32_t i = 0; i < size; i++) {
        contexts_[i] = CloudModuleContext::create(
            framework->contexts()[i], worldPipeline->contexts()[i], shared_from_this());
    }
}

std::vector<std::shared_ptr<WorldModuleContext>> &CloudModule::contexts() {
    return contexts_;
}

void CloudModule::bindTexture(std::shared_ptr<vk::Sampler> sampler,
                               std::shared_ptr<vk::DeviceLocalImage> image,
                               int index) {}

void CloudModule::preClose() {
    contexts_.clear();
    shadowPipeline_.reset();
    shadowShader_.reset();
    compositePipeline_.reset();
    compositeShader_.reset();
    temporalPipeline_.reset();
    temporalShader_.reset();
    raymarchPipeline_.reset();
    raymarchShader_.reset();
    weatherPipeline_.reset();
    weatherShader_.reset();
    noiseGenPipeline_.reset();
    noiseGenShader_.reset();
    descriptorTables_.clear();
    noiseSampler_.reset();
    weatherSampler_.reset();
    noiseTexture3D_.reset();
    weatherMapImage_.reset();
    cloudShadowImage_.reset();
    cloudHistoryImage_.reset();
    cloudColorImages_.clear();
    radianceImages_.clear();
    linearDepthImages_.clear();
    cloudRadianceImages_.clear();
    noiseGenerated_ = false;
}

void CloudModule::initDescriptorTables() {
    auto framework = framework_.lock();
    if (!framework) return;
    uint32_t size = framework->swapchain()->imageCount();

    descriptorTables_.resize(size);

    for (uint32_t i = 0; i < size; i++) {
        descriptorTables_[i] = vk::DescriptorTableBuilder{}
                                   .beginDescriptorLayoutSet() // set 0: images
                                   .beginDescriptorLayoutSetBinding()
                                   // binding 0: radiance input (storage image)
                                   .defineDescriptorLayoutSetBinding({
                                       .binding = 0,
                                       .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                       .descriptorCount = 1,
                                       .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                                   })
                                   // binding 1: linear depth (storage image)
                                   .defineDescriptorLayoutSetBinding({
                                       .binding = 1,
                                       .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                       .descriptorCount = 1,
                                       .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                                   })
                                   // binding 2: cloud radiance output (storage image)
                                   .defineDescriptorLayoutSetBinding({
                                       .binding = 2,
                                       .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                       .descriptorCount = 1,
                                       .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                                   })
                                   // binding 3: cloud color (storage image, cloud res)
                                   .defineDescriptorLayoutSetBinding({
                                       .binding = 3,
                                       .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                       .descriptorCount = 1,
                                       .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                                   })
                                   // binding 4: cloud history (storage image)
                                   .defineDescriptorLayoutSetBinding({
                                       .binding = 4,
                                       .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                       .descriptorCount = 1,
                                       .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                                   })
                                   // binding 5: weather map (storage image)
                                   .defineDescriptorLayoutSetBinding({
                                       .binding = 5,
                                       .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                       .descriptorCount = 1,
                                       .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                                   })
                                   // binding 6: 3D noise (combined image sampler, repeat)
                                   .defineDescriptorLayoutSetBinding({
                                       .binding = 6,
                                       .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                       .descriptorCount = 1,
                                       .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                                   })
                                   // binding 7: cloud shadow map (storage image)
                                   .defineDescriptorLayoutSetBinding({
                                       .binding = 7,
                                       .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                       .descriptorCount = 1,
                                       .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                                   })
                                   // binding 8: weather map (combined image sampler, for raymarch read)
                                   .defineDescriptorLayoutSetBinding({
                                       .binding = 8,
                                       .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                       .descriptorCount = 1,
                                       .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                                   })
                                   // binding 9: 3D noise (storage image, for noise gen write)
                                   .defineDescriptorLayoutSetBinding({
                                       .binding = 9,
                                       .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                       .descriptorCount = 1,
                                       .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                                   })
                                   .endDescriptorLayoutSetBinding()
                                   .endDescriptorLayoutSet()
                                   .beginDescriptorLayoutSet() // set 1: uniform buffers
                                   .beginDescriptorLayoutSetBinding()
                                   // binding 0: WorldUBO
                                   .defineDescriptorLayoutSetBinding({
                                       .binding = 0,
                                       .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                       .descriptorCount = 1,
                                       .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                                   })
                                   // binding 1: SkyUBO
                                   .defineDescriptorLayoutSetBinding({
                                       .binding = 1,
                                       .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                       .descriptorCount = 1,
                                       .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                                   })
                                   // binding 2: LastWorldUBO (previous frame camera matrices)
                                   .defineDescriptorLayoutSetBinding({
                                       .binding = 2,
                                       .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                       .descriptorCount = 1,
                                       .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                                   })
                                   .endDescriptorLayoutSetBinding()
                                   .endDescriptorLayoutSet()
                                   .definePushConstant(VkPushConstantRange{
                                       .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                                       .offset = 0,
                                       .size = sizeof(CloudPushConstant),
                                   })
                                   .build(framework->device());
    }
}

void CloudModule::initImages() {
    auto framework = framework_.lock();
    if (!framework) return;
    uint32_t size = framework->swapchain()->imageCount();

    // 3D noise texture — resolution configurable: 128 (8MB), 256 (64MB), 512 (512MB)
    uint32_t noiseRes = std::clamp(Renderer::options.cloudNoiseRes, 128u, 512u);
    noiseTexture3D_ = vk::DeviceLocalImage::create3D(
        framework->device(), framework->vma(), noiseRes, noiseRes, noiseRes,
        VK_FORMAT_R8G8B8A8_UNORM,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);

    // 512x512 RGBA8 weather map (Nyquist-safe for FBM up to cellFreq=16)
    weatherMapImage_ = vk::DeviceLocalImage::create(
        framework->device(), framework->vma(), false, 512, 512, 1,
        VK_FORMAT_R8G8B8A8_UNORM,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);

    // Cloud shadow map (R16F) — already created in setOrCreateOutputImages at correct size.
    // Only create here as fallback (shouldn't happen in normal flow).
    if (!cloudShadowImage_) {
        cloudShadowImage_ = vk::DeviceLocalImage::create(
            framework->device(), framework->vma(), false, shadowMapSize_, shadowMapSize_, 1,
            VK_FORMAT_R16_SFLOAT,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    }

    // Cloud history (single shared, cloud resolution)
    cloudHistoryImage_ = vk::DeviceLocalImage::create(
        framework->device(), framework->vma(), false, cloudWidth_, cloudHeight_, 1,
        VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);

    for (uint32_t i = 0; i < size; i++) {
        // Cloud color buffer (cloud resolution RGBA16F)
        cloudColorImages_[i] = vk::DeviceLocalImage::create(
            framework->device(), framework->vma(), false, cloudWidth_, cloudHeight_, 1,
            VK_FORMAT_R16G16B16A16_SFLOAT,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);

        // Set 0: bind all images
        descriptorTables_[i]->bindImage(radianceImages_[i], VK_IMAGE_LAYOUT_GENERAL, 0, 0);
        descriptorTables_[i]->bindImage(linearDepthImages_[i], VK_IMAGE_LAYOUT_GENERAL, 0, 1);
        descriptorTables_[i]->bindImage(cloudRadianceImages_[i], VK_IMAGE_LAYOUT_GENERAL, 0, 2);
        descriptorTables_[i]->bindImage(cloudColorImages_[i], VK_IMAGE_LAYOUT_GENERAL, 0, 3);
        descriptorTables_[i]->bindImage(cloudHistoryImage_, VK_IMAGE_LAYOUT_GENERAL, 0, 4);
        descriptorTables_[i]->bindImage(weatherMapImage_, VK_IMAGE_LAYOUT_GENERAL, 0, 5);
        descriptorTables_[i]->bindSamplerImage(noiseSampler_, noiseTexture3D_,
                                                VK_IMAGE_LAYOUT_GENERAL, 0, 6, 0);
        descriptorTables_[i]->bindImage(cloudShadowImage_, VK_IMAGE_LAYOUT_GENERAL, 0, 7);
        descriptorTables_[i]->bindSamplerImage(weatherSampler_, weatherMapImage_,
                                                VK_IMAGE_LAYOUT_GENERAL, 0, 8, 0);
        descriptorTables_[i]->bindImage(noiseTexture3D_, VK_IMAGE_LAYOUT_GENERAL, 0, 9);

        // UBO binding deferred to render() — during build(), the per-frame UBO buffers
        // may not be available yet (pipeline recreation happens before first frame upload).
        // Each frame's render() binds its own UBO via the deferred path.
    }
}

void CloudModule::initPipeline() {
    auto framework = framework_.lock();
    if (!framework) return;
    auto device = framework->device();
    std::filesystem::path shaderPath = Renderer::folderPath / "shaders";

    noiseGenShader_ = vk::Shader::create(device,
        (shaderPath / "world/cloud/cloud_noise_gen_comp.spv").string());
    noiseGenPipeline_ = vk::ComputePipelineBuilder{}
                            .defineShader(noiseGenShader_)
                            .definePipelineLayout(descriptorTables_[0])
                            .build(device);

    weatherShader_ = vk::Shader::create(device,
        (shaderPath / "world/cloud/cloud_weather_comp.spv").string());
    weatherPipeline_ = vk::ComputePipelineBuilder{}
                           .defineShader(weatherShader_)
                           .definePipelineLayout(descriptorTables_[0])
                           .build(device);

    raymarchShader_ = vk::Shader::create(device,
        (shaderPath / "world/cloud/cloud_raymarch_comp.spv").string());
    raymarchPipeline_ = vk::ComputePipelineBuilder{}
                            .defineShader(raymarchShader_)
                            .definePipelineLayout(descriptorTables_[0])
                            .build(device);

    temporalShader_ = vk::Shader::create(device,
        (shaderPath / "world/cloud/cloud_temporal_comp.spv").string());
    temporalPipeline_ = vk::ComputePipelineBuilder{}
                            .defineShader(temporalShader_)
                            .definePipelineLayout(descriptorTables_[0])
                            .build(device);

    compositeShader_ = vk::Shader::create(device,
        (shaderPath / "world/cloud/cloud_composite_comp.spv").string());
    compositePipeline_ = vk::ComputePipelineBuilder{}
                             .defineShader(compositeShader_)
                             .definePipelineLayout(descriptorTables_[0])
                             .build(device);

    shadowShader_ = vk::Shader::create(device,
        (shaderPath / "world/cloud/cloud_shadow_comp.spv").string());
    shadowPipeline_ = vk::ComputePipelineBuilder{}
                          .defineShader(shadowShader_)
                          .definePipelineLayout(descriptorTables_[0])
                          .build(device);
}

// --- Context ---

CloudModuleContext::CloudModuleContext(
    std::shared_ptr<FrameworkContext> frameworkContext,
    std::shared_ptr<WorldPipelineContext> worldPipelineContext,
    std::shared_ptr<CloudModule> cloudModule)
    : WorldModuleContext(frameworkContext, worldPipelineContext),
      cloudModule(cloudModule),
      radianceImage(cloudModule->radianceImages_[frameworkContext->frameIndex]),
      linearDepthImage(cloudModule->linearDepthImages_[frameworkContext->frameIndex]),
      cloudColorImage(cloudModule->cloudColorImages_[frameworkContext->frameIndex]),
      cloudHistoryImage(cloudModule->cloudHistoryImage_),
      weatherMapImage(cloudModule->weatherMapImage_),
      noiseTexture3D(cloudModule->noiseTexture3D_),
      cloudShadowImage(cloudModule->cloudShadowImage_),
      cloudRadianceImage(cloudModule->cloudRadianceImages_[frameworkContext->frameIndex]),
      descriptorTable(cloudModule->descriptorTables_[frameworkContext->frameIndex]) {}

void CloudModuleContext::render() {
    auto context = frameworkContext.lock();
    if (!context) return;
    auto framework = context->framework.lock();
    if (!framework) return;
    auto worldCommandBuffer = context->worldCommandBuffer;
    auto mainQueueIndex = framework->physicalDevice()->mainQueueIndex();

    auto module = cloudModule.lock();
    if (!module) return;

    // Bind UBOs every frame — vkUpdateDescriptorSets is cheap, and deferring
    // until all frame indices have rendered is fragile (frame 0 may never run).
    {
        auto buffers = Renderer::instance().buffers();
        if (!buffers) return;

        auto worldUBO = buffers->worldUniformBuffer();
        auto skyUBO = buffers->skyUniformBuffer();
        if (!worldUBO || !skyUBO) return;

        uint32_t fi = context->frameIndex;
        module->descriptorTables_[fi]->bindBuffer(worldUBO, 1, 0);
        module->descriptorTables_[fi]->bindBuffer(skyUBO, 1, 1);
        auto lastWorldUBO = buffers->lastWorldUniformBuffer();
        module->descriptorTables_[fi]->bindBuffer(lastWorldUBO ? lastWorldUBO : worldUBO, 1, 2);
    }

    // --- Helper lambdas (same pattern as VolumetricModule) ---
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

    auto subresourceFor = [](std::shared_ptr<vk::DeviceLocalImage> &img) -> VkImageSubresourceRange {
        bool is3D = (img->imageType() == VK_IMAGE_TYPE_3D);
        return {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = VK_REMAINING_MIP_LEVELS,
            .baseArrayLayer = 0,
            .layerCount = is3D ? 1u : VK_REMAINING_ARRAY_LAYERS,
        };
    };

    auto makeBarrier = [&](std::shared_ptr<vk::DeviceLocalImage> &img,
                           VkPipelineStageFlags2 fallbackStage,
                           VkAccessFlags2 dstAccess) -> vk::CommandBuffer::ImageMemoryBarrier {
        VkPipelineStageFlags2 srcStage = 0;
        VkAccessFlags2 srcAccess = 0;
        chooseSrc(img->imageLayout(), fallbackStage, VK_ACCESS_2_MEMORY_WRITE_BIT, srcStage, srcAccess);
        return {
            .srcStageMask = srcStage,
            .srcAccessMask = srcAccess,
            .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            .dstAccessMask = dstAccess,
            .oldLayout = img->imageLayout(),
            .newLayout = VK_IMAGE_LAYOUT_GENERAL,
            .srcQueueFamilyIndex = mainQueueIndex,
            .dstQueueFamilyIndex = mainQueueIndex,
            .image = img,
            .subresourceRange = subresourceFor(img),
        };
    };

    auto rtStage = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    auto compStage = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;

    // --- Barrier: transition all images to GENERAL for compute ---
    std::vector<vk::CommandBuffer::ImageMemoryBarrier> imageBarriers = {
        makeBarrier(radianceImage, rtStage, VK_ACCESS_2_SHADER_READ_BIT),
        makeBarrier(linearDepthImage, rtStage, VK_ACCESS_2_SHADER_READ_BIT),
        makeBarrier(cloudRadianceImage, compStage, VK_ACCESS_2_SHADER_WRITE_BIT),
        makeBarrier(cloudColorImage, compStage, VK_ACCESS_2_SHADER_WRITE_BIT),
        makeBarrier(cloudHistoryImage, compStage, VK_ACCESS_2_SHADER_READ_BIT),
        makeBarrier(weatherMapImage, compStage, VK_ACCESS_2_SHADER_WRITE_BIT),
        makeBarrier(noiseTexture3D, compStage, VK_ACCESS_2_SHADER_READ_BIT),
        makeBarrier(cloudShadowImage, compStage, VK_ACCESS_2_SHADER_WRITE_BIT),
    };

    worldCommandBuffer->barriersBufferImage({}, imageBarriers);
    radianceImage->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;
    linearDepthImage->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;
    cloudRadianceImage->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;
    cloudColorImage->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;
    cloudHistoryImage->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;
    weatherMapImage->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;
    noiseTexture3D->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;
    cloudShadowImage->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;

    // --- Push constants ---
    CloudPushConstant pc{};
    pc.renderWidth = module->width_;
    pc.renderHeight = module->height_;
    pc.cloudWidth = module->cloudWidth_;
    pc.cloudHeight = module->cloudHeight_;
    pc.cloudBase = module->enabled_ ? std::clamp(Renderer::options.cloudAltitude, 64.0f, 320.0f) : 99999.0f;
    pc.cloudThickness = std::clamp(Renderer::options.cloudThickness, 16.0f, 256.0f);
    pc.coverage = std::clamp(Renderer::options.cloudCoverage, 0.0f, 1.0f);
    pc.cloudType = std::clamp(Renderer::options.cloudType, 0.0f, 2.0f);
    pc.densityMultiplier = module->enabled_ ? std::clamp(Renderer::options.cloudDensity, 0.0f, 5.0f) : 0.0f;
    pc.windTime = module->windTime_;
    pc.frameIndex = module->frameCounter_++;
    pc.marchSteps = Renderer::options.cloudMarchStepsOverride > 0
        ? Renderer::options.cloudMarchStepsOverride : module->marchSteps_;
    pc.lightSteps = Renderer::options.cloudLightStepsOverride > 0
        ? Renderer::options.cloudLightStepsOverride : module->lightSteps_;
    pc.temporalBlend = Renderer::options.cloudTemporalBlend >= 0.0f
        ? std::clamp(Renderer::options.cloudTemporalBlend, 0.0f, 1.0f)
        : std::clamp(module->temporalBlend_, 0.0f, 1.0f);
    pc.shadowMapSize = module->shadowMapSize_;

    // Camera world position from World (same source as WorldUBO.cameraPos)
    auto world = Renderer::instance().world();
    glm::dvec3 camPos = world->getCameraPos();
    pc.eyePosX = static_cast<float>(camPos.x);
    pc.eyePosY = static_cast<float>(camPos.y);
    pc.eyePosZ = static_cast<float>(camPos.z);
    pc.detailStrength = std::clamp(Renderer::options.cloudDetailStrength, 0.0f, 3.0f);
    pc.scatterOctaves = std::clamp(Renderer::options.cloudScatterOctaves, 1u, 8u);
    pc.ambientStrength = std::clamp(Renderer::options.cloudAmbientStrength, 0.0f, 3.0f);
    pc.noiseScale = std::clamp(Renderer::options.cloudNoiseScale, 16.0f, 4096.0f);
    pc.cellFrequency = std::clamp(Renderer::options.cloudCellFrequency, 1.0f, 32.0f);
    pc.atmosphereFadeDist = std::clamp(Renderer::options.cloudAtmosphereFadeDist, 100.0f, 4000.0f);
    pc.windAngle = Renderer::options.cloudWindAngle;
    pc.debugMode = std::clamp(Renderer::options.cloudDebugMode, 0u, 8u);

    // Advance wind time using actual frame delta
    auto now = std::chrono::steady_clock::now();
    float deltaSeconds = 1.0f / 60.0f; // fallback for first frame
    if (module->lastFrameTime_.time_since_epoch().count() > 0) {
        deltaSeconds = std::chrono::duration<float>(now - module->lastFrameTime_).count();
        deltaSeconds = std::clamp(deltaSeconds, 0.001f, 0.1f);
    }
    module->lastFrameTime_ = now;
    module->windTime_ += Renderer::options.cloudSpeed * deltaSeconds;
    // Wrap at 24-hour boundary to prevent FP32 precision loss.
    // At windSpeed=6.0 and 60fps, windTime reaches ~518400 after 24h.
    // FP32 has ~7 significant digits, so 518400 + 0.001 = 518400 (frozen wind).
    // fmod preserves the fractional part that drives noise UV offsets.
    module->windTime_ = std::fmod(module->windTime_, 86400.0f);

    vkCmdPushConstants(worldCommandBuffer->vkCommandBuffer(), descriptorTable->vkPipelineLayout(),
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(CloudPushConstant), &pc);

    worldCommandBuffer->bindDescriptorTable(descriptorTable, VK_PIPELINE_BIND_POINT_COMPUTE);

    if (!module->enabled_) {
        worldCommandBuffer->bindComputePipeline(module->compositePipeline_);
        uint32_t compX = (module->width_ + 7) / 8;
        uint32_t compY = (module->height_ + 7) / 8;
        vkCmdDispatch(worldCommandBuffer->vkCommandBuffer(), compX, compY, 1);

        worldCommandBuffer->barriersBufferImage({}, {{
            .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
            .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
            .newLayout = VK_IMAGE_LAYOUT_GENERAL,
            .srcQueueFamilyIndex = mainQueueIndex,
            .dstQueueFamilyIndex = mainQueueIndex,
            .image = cloudRadianceImage,
            .subresourceRange = subresourceFor(cloudRadianceImage),
        }});
        return;
    }

    // --- Pass 0: Noise generation (first frame only) ---
    if (!module->noiseGenerated_) {
        worldCommandBuffer->bindComputePipeline(module->noiseGenPipeline_);
        uint32_t noiseRes = module->noiseTexture3D_->width();
        vkCmdDispatch(worldCommandBuffer->vkCommandBuffer(), noiseRes / 4, noiseRes / 4, noiseRes / 4);

        // Barrier: noise texture written
        worldCommandBuffer->barriersBufferImage({}, {{
            .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
            .newLayout = VK_IMAGE_LAYOUT_GENERAL,
            .srcQueueFamilyIndex = mainQueueIndex,
            .dstQueueFamilyIndex = mainQueueIndex,
            .image = noiseTexture3D,
            .subresourceRange = subresourceFor(noiseTexture3D),
        }});
        module->noiseGenerated_ = true;
    }

    // --- Pass 1: Weather map update (512x512) ---
    worldCommandBuffer->bindComputePipeline(module->weatherPipeline_);
    vkCmdDispatch(worldCommandBuffer->vkCommandBuffer(), 512 / 8, 512 / 8, 1);

    // Barrier: weather map written, raymarch will sample it
    worldCommandBuffer->barriersBufferImage({}, {{
        .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
        .newLayout = VK_IMAGE_LAYOUT_GENERAL,
        .srcQueueFamilyIndex = mainQueueIndex,
        .dstQueueFamilyIndex = mainQueueIndex,
        .image = weatherMapImage,
        .subresourceRange = subresourceFor(weatherMapImage),
    }});

    // --- Pass 2: Cloud ray march (cloud resolution) ---
    worldCommandBuffer->bindComputePipeline(module->raymarchPipeline_);
    uint32_t marchX = (module->cloudWidth_ + 7) / 8;
    uint32_t marchY = (module->cloudHeight_ + 7) / 8;
    vkCmdDispatch(worldCommandBuffer->vkCommandBuffer(), marchX, marchY, 1);

    // Barrier: cloudColor written, temporal reads it
    worldCommandBuffer->barriersBufferImage({}, {{
        .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
        .newLayout = VK_IMAGE_LAYOUT_GENERAL,
        .srcQueueFamilyIndex = mainQueueIndex,
        .dstQueueFamilyIndex = mainQueueIndex,
        .image = cloudColorImage,
        .subresourceRange = subresourceFor(cloudColorImage),
    }});

    // --- Pass 3: Temporal reprojection (cloud resolution) ---
    worldCommandBuffer->bindComputePipeline(module->temporalPipeline_);
    vkCmdDispatch(worldCommandBuffer->vkCommandBuffer(), marchX, marchY, 1);

    // Barrier: temporal wrote cloudColor → copy to history + composite reads
    worldCommandBuffer->barriersBufferImage({}, {{
        .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
        .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_TRANSFER_READ_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
        .newLayout = VK_IMAGE_LAYOUT_GENERAL,
        .srcQueueFamilyIndex = mainQueueIndex,
        .dstQueueFamilyIndex = mainQueueIndex,
        .image = cloudColorImage,
        .subresourceRange = subresourceFor(cloudColorImage),
    }});

    // Copy current cloudColor → history for next frame's temporal blend
    {
        VkImageCopy copyRegion = {};
        copyRegion.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        copyRegion.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        copyRegion.extent = {module->cloudWidth_, module->cloudHeight_, 1};
        vkCmdCopyImage(
            worldCommandBuffer->vkCommandBuffer(),
            cloudColorImage->vkImage(), VK_IMAGE_LAYOUT_GENERAL,
            cloudHistoryImage->vkImage(), VK_IMAGE_LAYOUT_GENERAL,
            1, &copyRegion);

        worldCommandBuffer->barriersBufferImage({}, {{
            .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
            .newLayout = VK_IMAGE_LAYOUT_GENERAL,
            .srcQueueFamilyIndex = mainQueueIndex,
            .dstQueueFamilyIndex = mainQueueIndex,
            .image = cloudHistoryImage,
            .subresourceRange = subresourceFor(cloudHistoryImage),
        }});
    }

    // --- Pass 4: Composite clouds onto radiance (full resolution) ---
    worldCommandBuffer->bindComputePipeline(module->compositePipeline_);
    uint32_t compX = (module->width_ + 7) / 8;
    uint32_t compY = (module->height_ + 7) / 8;
    vkCmdDispatch(worldCommandBuffer->vkCommandBuffer(), compX, compY, 1);

    // Barrier: cloudRadianceImage written by composite → downstream modules read it
    worldCommandBuffer->barriersBufferImage({}, {{
        .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
        .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
        .newLayout = VK_IMAGE_LAYOUT_GENERAL,
        .srcQueueFamilyIndex = mainQueueIndex,
        .dstQueueFamilyIndex = mainQueueIndex,
        .image = cloudRadianceImage,
        .subresourceRange = subresourceFor(cloudRadianceImage),
    }});

    // --- Pass 5: Cloud shadow map (independent of composite — parallel-safe) ---
    worldCommandBuffer->bindComputePipeline(module->shadowPipeline_);
    uint32_t shadowGroups = (module->shadowMapSize_ + 7) / 8;
    vkCmdDispatch(worldCommandBuffer->vkCommandBuffer(), shadowGroups, shadowGroups, 1);

    // Barrier: cloudShadowImage written → VML or downstream modules read it
    worldCommandBuffer->barriersBufferImage({}, {{
        .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
        .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
        .newLayout = VK_IMAGE_LAYOUT_GENERAL,
        .srcQueueFamilyIndex = mainQueueIndex,
        .dstQueueFamilyIndex = mainQueueIndex,
        .image = cloudShadowImage,
        .subresourceRange = subresourceFor(cloudShadowImage),
    }});
}
