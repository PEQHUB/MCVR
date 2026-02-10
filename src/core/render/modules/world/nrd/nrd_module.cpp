#include "nrd_module.hpp"
#include "core/render/buffers.hpp"
#include "core/render/pipeline.hpp"
#include "core/render/render_framework.hpp"
#include "core/render/renderer.hpp"
#include <iostream>

NrdModule::NrdModule() {}

void NrdModule::init(std::shared_ptr<Framework> framework, std::shared_ptr<WorldPipeline> worldPipeline) {
    WorldModule::init(framework, worldPipeline);
    m_device = framework->device();
    m_vma = framework->vma();

    uint32_t size = framework->swapchain()->imageCount();

    diffuseRadianceImages_.resize(size);
    specularRadianceImages_.resize(size);
    directRadianceImages_.resize(size);
    diffuseAlbedoImages_.resize(size);
    specularAlbedoImages_.resize(size);
    normalRoughnessImages_.resize(size);
    motionVectorImages_.resize(size);
    linearDepthImages_.resize(size);
    clearRadianceImages_.resize(size);
    baseEmissionImages_.resize(size);
    diffuseHitDepthImages_.resize(size);
    specularHitDepthImages_.resize(size);
    denoisedRadianceImages_.resize(size);
    denoisedDiffuseRadianceImages_.resize(size);
    denoisedSpecularRadianceImages_.resize(size);
}

bool NrdModule::setOrCreateInputImages(std::vector<std::shared_ptr<vk::DeviceLocalImage>> &images,
                                       std::vector<VkFormat> &formats,
                                       uint32_t frameIndex) {
    if (images.size() != inputImageNum) return false;

    auto createImage = [&](uint32_t index) {
        images[index] = vk::DeviceLocalImage::create(m_device, m_vma, false, width_, height_, 1, formats[index],
                                                     VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                                                         VK_IMAGE_USAGE_SAMPLED_BIT);
    };

    for (uint32_t i = 0; i < images.size(); i++) {
        if (images[i] != nullptr) {
            if (width_ == 0 && height_ == 0) {
                width_ = images[i]->width();
                height_ = images[i]->height();
            } else if (images[i]->width() != width_ || images[i]->height() != height_) {
                std::cerr << "[NrdModule] Error: Input image " << i << " size mismatch. Expected " << width_ << "x"
                          << height_ << ", got " << images[i]->width() << "x" << images[i]->height() << std::endl;
                return false;
            }
        }
    }

    for (uint32_t i = 0; i < images.size(); i++) {
        if (images[i] == nullptr) {
            if (width_ == 0 || height_ == 0) {
                std::cerr << "[NrdModule] Error: Cannot create input image " << i << " because dimensions are unknown."
                          << std::endl;
                return false;
            }
            createImage(i);
        }
    }

    diffuseRadianceImages_[frameIndex] = images[0];
    specularRadianceImages_[frameIndex] = images[1];
    directRadianceImages_[frameIndex] = images[2];
    diffuseAlbedoImages_[frameIndex] = images[3];
    specularAlbedoImages_[frameIndex] = images[4];
    normalRoughnessImages_[frameIndex] = images[5];
    motionVectorImages_[frameIndex] = images[6];
    linearDepthImages_[frameIndex] = images[7];
    clearRadianceImages_[frameIndex] = images[8];
    baseEmissionImages_[frameIndex] = images[9];
    diffuseHitDepthImages_[frameIndex] = images[10];
    specularHitDepthImages_[frameIndex] = images[11];

    return true;
}

bool NrdModule::setOrCreateOutputImages(std::vector<std::shared_ptr<vk::DeviceLocalImage>> &images,
                                        std::vector<VkFormat> &formats,
                                        uint32_t frameIndex) {
    if (images.size() != outputImageNum) return false;
    if (!images[0]) return false;

    width_ = images[0]->width();
    height_ = images[0]->height();

    denoisedRadianceImages_[frameIndex] = images[0];

    return true;
}

void NrdModule::build() {
    auto framework = framework_.lock();
    auto worldPipeline = worldPipeline_.lock();
    uint32_t size = framework->swapchain()->imageCount();

    m_wrapper = std::make_shared<NrdWrapper>();

    auto createInternal = [&](std::vector<std::shared_ptr<vk::DeviceLocalImage>> &images) {
        for (uint32_t i = 0; i < size; i++) {
            if (images[i] != nullptr) continue;
            images[i] =
                vk::DeviceLocalImage::create(m_device, m_vma, false, width_, height_, 1, VK_FORMAT_R16G16B16A16_SFLOAT,
                                             VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
        }
    };
    createInternal(denoisedDiffuseRadianceImages_);
    createInternal(denoisedSpecularRadianceImages_);

    bool ok = m_wrapper->init(m_device, m_vma, framework->physicalDevice(), width_, height_, size);
    if (!ok) {
        std::cerr << "[NrdModule] init failed." << std::endl;
        m_wrapper.reset();
    }

    createCompositionPipeline(m_device, size);
    createPreparePipeline(m_device, size);

    contexts_.resize(size);
    for (int i = 0; i < size; i++) {
        contexts_[i] =
            NrdModuleContext::create(framework->contexts()[i], worldPipeline->contexts()[i], shared_from_this());
    }
}

void NrdModule::setAttributes(int attributeCount, std::vector<std::string> &attributeKVs) {
    auto parseBool = [](const std::string &value) {
        return value == "1" || value == "true" || value == "True" || value == "TRUE";
    };

    for (int i = 0; i < attributeCount; i++) {
        const std::string &key = attributeKVs[2 * i];
        const std::string &value = attributeKVs[2 * i + 1];

        if (key == "render_pipeline.module.nrd.attribute.max_accumulated_frame_num") {
            maxAccumulatedFrameNum_ = std::stoi(value);
        } else if (key == "render_pipeline.module.nrd.attribute.max_fast_accumulated_frame_num") {
            maxFastAccumulatedFrameNum_ = std::stoi(value);
        } else if (key == "render_pipeline.module.nrd.attribute.max_blur_radius") {
            maxBlurRadius_ = std::stof(value);
        } else if (key == "render_pipeline.module.nrd.attribute.enable_anti_firefly") {
            enableAntiFirefly_ = parseBool(value);
        } else if (key == "render_pipeline.module.nrd.attribute.hit_distance_reconstruction_mode") {
            std::cout << "Setting hit_distance_reconstruction_mode to: " << value << std::endl;
            if (value == "3x3")
                hitDistanceReconstructionMode_ = nrd::HitDistanceReconstructionMode::AREA_3X3;
            else if (value == "5x5")
                hitDistanceReconstructionMode_ = nrd::HitDistanceReconstructionMode::AREA_5X5;
        }
    }
}

std::vector<std::shared_ptr<WorldModuleContext>> &NrdModule::contexts() {
    return contexts_;
}

void NrdModule::bindTexture(std::shared_ptr<vk::Sampler> sampler,
                            std::shared_ptr<vk::DeviceLocalImage> image,
                            int index) {}

void NrdModule::preClose() {
    m_wrapper.reset();
    composeDescriptorTables_.clear();
    prepareDescriptorTables_.clear();
    composeSamplers_ = {};
    composePipeline_.reset();
    preparePipeline_.reset();
}

NrdModuleContext::NrdModuleContext(std::shared_ptr<FrameworkContext> frameworkContext,
                                   std::shared_ptr<WorldPipelineContext> worldPipelineContext,
                                   std::shared_ptr<NrdModule> nrdModule)
    : WorldModuleContext(frameworkContext, worldPipelineContext),
      nrdModule(nrdModule),
      diffuseRadianceImage(nrdModule->diffuseRadianceImages_[frameworkContext->frameIndex]),
      specularRadianceImage(nrdModule->specularRadianceImages_[frameworkContext->frameIndex]),
      directRadianceImage(nrdModule->directRadianceImages_[frameworkContext->frameIndex]),
      diffuseAlbedoImage(nrdModule->diffuseAlbedoImages_[frameworkContext->frameIndex]),
      specularAlbedoImage(nrdModule->specularAlbedoImages_[frameworkContext->frameIndex]),
      normalRoughnessImage(nrdModule->normalRoughnessImages_[frameworkContext->frameIndex]),
      motionVectorImage(nrdModule->motionVectorImages_[frameworkContext->frameIndex]),
      linearDepthImage(nrdModule->linearDepthImages_[frameworkContext->frameIndex]),
      clearRadianceImage(nrdModule->clearRadianceImages_[frameworkContext->frameIndex]),
      baseEmissionImage(nrdModule->baseEmissionImages_[frameworkContext->frameIndex]),
      diffuseHitDepthImage(nrdModule->diffuseHitDepthImages_[frameworkContext->frameIndex]),
      specularHitDepthImage(nrdModule->specularHitDepthImages_[frameworkContext->frameIndex]),
      denoisedRadianceImage(nrdModule->denoisedRadianceImages_[frameworkContext->frameIndex]),
      denoisedDiffuseRadianceImage(nrdModule->denoisedDiffuseRadianceImages_[frameworkContext->frameIndex]),
      denoisedSpecularRadianceImage(nrdModule->denoisedSpecularRadianceImages_[frameworkContext->frameIndex]) {}

void NrdModuleContext::render() {
    auto module = nrdModule.lock();
    if (!module || !module->wrapper()) return;

    auto context = frameworkContext.lock();
    auto worldCommandBuffer = context->worldCommandBuffer;
    auto framework = context->framework.lock();
    auto mainQueueIndex = framework->physicalDevice()->mainQueueIndex();
    auto buffers = Renderer::instance().buffers();
    auto worldUBO = static_cast<vk::Data::WorldUBO *>(buffers->worldUniformBuffer()->mappedPtr());
    auto lastWorldUBO = static_cast<vk::Data::WorldUBO *>(buffers->lastWorldUniformBuffer()->mappedPtr());

    if (module->width_ == 0 || module->height_ == 0) return;

    nrd::CommonSettings commonSettings = {};
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            commonSettings.viewToClipMatrix[i * 4 + j] = worldUBO->cameraProjMat[i][j];
            commonSettings.viewToClipMatrixPrev[i * 4 + j] = lastWorldUBO->cameraProjMat[i][j];
            commonSettings.worldToViewMatrix[i * 4 + j] = worldUBO->cameraViewMat[i][j];
            commonSettings.worldToViewMatrixPrev[i * 4 + j] = lastWorldUBO->cameraViewMat[i][j];
        }
    }

    commonSettings.resourceSize[0] = static_cast<uint16_t>(module->width_);
    commonSettings.resourceSize[1] = static_cast<uint16_t>(module->height_);
    commonSettings.rectSize[0] = static_cast<uint16_t>(module->width_);
    commonSettings.rectSize[1] = static_cast<uint16_t>(module->height_);
    commonSettings.resourceSizePrev[0] = static_cast<uint16_t>(module->width_);
    commonSettings.resourceSizePrev[1] = static_cast<uint16_t>(module->height_);
    commonSettings.rectSizePrev[0] = static_cast<uint16_t>(module->width_);
    commonSettings.rectSizePrev[1] = static_cast<uint16_t>(module->height_);

    commonSettings.cameraJitter[0] = std::max(-0.5f, std::min(0.5f, static_cast<float>(worldUBO->cameraJitter.x)));
    commonSettings.cameraJitter[1] = std::max(-0.5f, std::min(0.5f, static_cast<float>(worldUBO->cameraJitter.y)));
    commonSettings.cameraJitterPrev[0] =
        std::max(-0.5f, std::min(0.5f, static_cast<float>(lastWorldUBO->cameraJitter.x)));
    commonSettings.cameraJitterPrev[1] =
        std::max(-0.5f, std::min(0.5f, static_cast<float>(lastWorldUBO->cameraJitter.y)));

    commonSettings.motionVectorScale[0] = 1.0f / module->width_;
    commonSettings.motionVectorScale[1] = 1.0f / module->height_;
    commonSettings.motionVectorScale[2] = 0.0f;
    commonSettings.isMotionVectorInWorldSpace = false;

    commonSettings.worldPrevToWorldMatrix[0] = 1.0f;
    commonSettings.worldPrevToWorldMatrix[5] = 1.0f;
    commonSettings.worldPrevToWorldMatrix[10] = 1.0f;
    commonSettings.worldPrevToWorldMatrix[15] = 1.0f;

    commonSettings.isBaseColorMetalnessAvailable = false;
    commonSettings.isDisocclusionThresholdMixAvailable = false;
    commonSettings.enableValidation = false;

    nrd::ReblurSettings reblurSettings = {};
    reblurSettings.hitDistanceParameters = {};
    reblurSettings.diffusePrepassBlurRadius = 25.0f;
    reblurSettings.specularPrepassBlurRadius = 30.0f;
    reblurSettings.maxAccumulatedFrameNum = static_cast<uint16_t>(module->maxAccumulatedFrameNum_);
    reblurSettings.maxFastAccumulatedFrameNum = static_cast<uint16_t>(module->maxFastAccumulatedFrameNum_);
    reblurSettings.enableAntiFirefly = module->enableAntiFirefly_;
    reblurSettings.antilagSettings.luminanceSigmaScale = 0.8f;
    reblurSettings.antilagSettings.luminanceSensitivity = 2.0f;
    reblurSettings.minBlurRadius = 20.0f;
    reblurSettings.maxBlurRadius = module->maxBlurRadius_;
    reblurSettings.lobeAngleFraction = 0.4f;
    reblurSettings.roughnessFraction = 0.2f;
    reblurSettings.historyFixFrameNum = 3;
    reblurSettings.hitDistanceReconstructionMode = nrd::HitDistanceReconstructionMode::AREA_5X5;

    {
        auto prepareTable = module->prepareDescriptorTables_[context->frameIndex];
        prepareTable->bindImage(motionVectorImage, VK_IMAGE_LAYOUT_GENERAL, 0, 0);
        prepareTable->bindImage(normalRoughnessImage, VK_IMAGE_LAYOUT_GENERAL, 0, 1);
        prepareTable->bindImage(linearDepthImage, VK_IMAGE_LAYOUT_GENERAL, 0, 2);
        prepareTable->bindImage(diffuseRadianceImage, VK_IMAGE_LAYOUT_GENERAL, 0, 3);
        prepareTable->bindImage(specularRadianceImage, VK_IMAGE_LAYOUT_GENERAL, 0, 4);
        prepareTable->bindImage(module->m_nrdMotionVectorImages[context->frameIndex], VK_IMAGE_LAYOUT_GENERAL, 0, 5);
        prepareTable->bindImage(module->m_nrdNormalRoughnessImages[context->frameIndex], VK_IMAGE_LAYOUT_GENERAL, 0, 6);
        prepareTable->bindImage(module->m_nrdDiffuseRadianceImages[context->frameIndex], VK_IMAGE_LAYOUT_GENERAL, 0, 7);
        prepareTable->bindImage(module->m_nrdSpecularRadianceImages[context->frameIndex], VK_IMAGE_LAYOUT_GENERAL, 0,
                                8);
        prepareTable->bindImage(diffuseAlbedoImage, VK_IMAGE_LAYOUT_GENERAL, 0, 9);
        prepareTable->bindImage(specularAlbedoImage, VK_IMAGE_LAYOUT_GENERAL, 0, 10);
        prepareTable->bindImage(directRadianceImage, VK_IMAGE_LAYOUT_GENERAL, 0, 11);
        prepareTable->bindImage(clearRadianceImage, VK_IMAGE_LAYOUT_GENERAL, 0, 12);
        prepareTable->bindBuffer(Renderer::instance().buffers()->worldUniformBuffer(), 0, 13);
        prepareTable->bindImage(diffuseHitDepthImage, VK_IMAGE_LAYOUT_GENERAL, 0, 14);
        prepareTable->bindImage(specularHitDepthImage, VK_IMAGE_LAYOUT_GENERAL, 0, 15);

        worldCommandBuffer->bindDescriptorTable(prepareTable, VK_PIPELINE_BIND_POINT_COMPUTE)
            ->bindComputePipeline(module->preparePipeline_);
        vkCmdDispatch(worldCommandBuffer->vkCommandBuffer(), (module->width_ + 15) / 16, (module->height_ + 15) / 16,
                      1);

        worldCommandBuffer->barriersBufferImage(
            {}, {{
                     .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                     .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
                     .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                     .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
                     .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
                     .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                     .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                     .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                     .image = module->m_nrdMotionVectorImages[context->frameIndex],
                     .subresourceRange = vk::wholeColorSubresourceRange,
                 },
                 {
                     .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                     .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
                     .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                     .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
                     .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
                     .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                     .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                     .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                     .image = module->m_nrdNormalRoughnessImages[context->frameIndex],
                     .subresourceRange = vk::wholeColorSubresourceRange,
                 },
                 {
                     .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                     .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
                     .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                     .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
                     .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
                     .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                     .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                     .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                     .image = module->m_nrdDiffuseRadianceImages[context->frameIndex],
                     .subresourceRange = vk::wholeColorSubresourceRange,
                 },
                 {
                     .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                     .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
                     .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                     .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
                     .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
                     .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                     .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                     .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                     .image = module->m_nrdSpecularRadianceImages[context->frameIndex],
                     .subresourceRange = vk::wholeColorSubresourceRange,
                 }});
    }

    module->wrapper()->updateSettings(commonSettings, reblurSettings);

    std::map<nrd::ResourceType, std::shared_ptr<vk::DeviceLocalImage>> userTextures;
    userTextures[nrd::ResourceType::IN_MV] = module->m_nrdMotionVectorImages[context->frameIndex];
    userTextures[nrd::ResourceType::IN_NORMAL_ROUGHNESS] = module->m_nrdNormalRoughnessImages[context->frameIndex];
    userTextures[nrd::ResourceType::IN_VIEWZ] = linearDepthImage;
    userTextures[nrd::ResourceType::IN_DIFF_RADIANCE_HITDIST] = module->m_nrdDiffuseRadianceImages[context->frameIndex];
    userTextures[nrd::ResourceType::IN_SPEC_RADIANCE_HITDIST] =
        module->m_nrdSpecularRadianceImages[context->frameIndex];
    userTextures[nrd::ResourceType::OUT_DIFF_RADIANCE_HITDIST] = denoisedDiffuseRadianceImage;
    userTextures[nrd::ResourceType::OUT_SPEC_RADIANCE_HITDIST] = denoisedSpecularRadianceImage;

    module->wrapper()->denoise(worldCommandBuffer->vkCommandBuffer(), context->frameIndex, userTextures);

    std::vector<vk::CommandBuffer::ImageMemoryBarrier> barriers;
    auto ensureGeneral = [&](std::shared_ptr<vk::DeviceLocalImage> img) {
        if (!img) return;
        if (img->imageLayout() == VK_IMAGE_LAYOUT_GENERAL) return;

        VkPipelineStageFlags2 srcStage = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
        VkAccessFlags2 srcAccess = 0;
        if (img->imageLayout() == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
            srcStage = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            srcAccess = VK_ACCESS_2_SHADER_READ_BIT;
        }

        barriers.push_back({
            .srcStageMask = srcStage,
            .srcAccessMask = srcAccess,
            .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            .dstAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
            .oldLayout = img->imageLayout(),
            .newLayout = VK_IMAGE_LAYOUT_GENERAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = img,
            .subresourceRange = vk::wholeColorSubresourceRange,
        });
        img->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;
    };

    ensureGeneral(denoisedRadianceImage);

    auto addBarrier = [&](std::shared_ptr<vk::DeviceLocalImage> img) {
        barriers.push_back({
            .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
            .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = img,
            .subresourceRange = vk::wholeColorSubresourceRange,
        });
        img->imageLayout() = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    };
    addBarrier(diffuseAlbedoImage);
    addBarrier(specularAlbedoImage);
    addBarrier(normalRoughnessImage);
    addBarrier(linearDepthImage);
    addBarrier(denoisedDiffuseRadianceImage);
    addBarrier(denoisedSpecularRadianceImage);
    addBarrier(clearRadianceImage);
    addBarrier(baseEmissionImage);

    if (directRadianceImage->imageLayout() == VK_IMAGE_LAYOUT_GENERAL) { addBarrier(directRadianceImage); }

    worldCommandBuffer->barriersBufferImage({}, barriers);

    std::map<std::string, std::shared_ptr<vk::DeviceLocalImage>> composeInputs;
    composeInputs["direct"] = directRadianceImage;
    composeInputs["diffuse"] = denoisedDiffuseRadianceImage;
    composeInputs["specular"] = denoisedSpecularRadianceImage;
    composeInputs["albedo"] = diffuseAlbedoImage;
    composeInputs["specularAlbedo"] = specularAlbedoImage;
    composeInputs["normal"] = normalRoughnessImage;
    composeInputs["depth"] = linearDepthImage;
    composeInputs["clearRadiance"] = clearRadianceImage;
    composeInputs["baseEmission"] = baseEmissionImage;

    module->dispatchComposition(worldCommandBuffer, context->frameIndex, composeInputs, buffers->worldUniformBuffer(),
                                denoisedRadianceImage);

    denoisedRadianceImage->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;
}

void NrdModule::createCompositionPipeline(std::shared_ptr<vk::Device> device, uint32_t contextCount) {
    auto framework = framework_.lock();
    composeDescriptorTables_.resize(contextCount);

    composeSamplers_[0] = vk::Sampler::create(device, VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_LINEAR,
                                              VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
    composeSamplers_[1] = vk::Sampler::create(device, VK_FILTER_NEAREST, VK_SAMPLER_MIPMAP_MODE_NEAREST,
                                              VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);

    for (uint32_t i = 0; i < contextCount; ++i) {
        composeDescriptorTables_[i] =
            vk::DescriptorTableBuilder{}
                .beginDescriptorLayoutSet() // set 0
                .beginDescriptorLayoutSetBinding()
                .defineDescriptorLayoutSetBinding({.binding = 0,
                                                   .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                                   .descriptorCount = 1,
                                                   .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT})
                .defineDescriptorLayoutSetBinding({.binding = 1,
                                                   .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                                   .descriptorCount = 1,
                                                   .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT})
                .defineDescriptorLayoutSetBinding({.binding = 2,
                                                   .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                                   .descriptorCount = 1,
                                                   .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT})
                .defineDescriptorLayoutSetBinding({.binding = 3,
                                                   .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                                   .descriptorCount = 1,
                                                   .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT})
                .defineDescriptorLayoutSetBinding({.binding = 4,
                                                   .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                                   .descriptorCount = 1,
                                                   .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT})
                .defineDescriptorLayoutSetBinding({.binding = 5,
                                                   .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                                   .descriptorCount = 1,
                                                   .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT})
                .defineDescriptorLayoutSetBinding({.binding = 6,
                                                   .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                                   .descriptorCount = 1,
                                                   .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT})
                .defineDescriptorLayoutSetBinding({.binding = 7,
                                                   .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                                   .descriptorCount = 1,
                                                   .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT})
                .defineDescriptorLayoutSetBinding({.binding = 8,
                                                   .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                                   .descriptorCount = 1,
                                                   .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT})
                .defineDescriptorLayoutSetBinding({.binding = 9,
                                                   .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                                   .descriptorCount = 1,
                                                   .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT})
                .defineDescriptorLayoutSetBinding({.binding = 10,
                                                   .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                                   .descriptorCount = 1,
                                                   .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT})
                .endDescriptorLayoutSetBinding()
                .endDescriptorLayoutSet()
                .build(framework->device());
    }

    auto shader =
        vk::Shader::create(device, (Renderer::folderPath / "shaders/world/nrd/nrd_compose_comp.spv").string());
    composePipeline_ = vk::ComputePipelineBuilder{}
                           .defineShader(shader)
                           .definePipelineLayout(composeDescriptorTables_[0])
                           .build(device);
}

void NrdModule::dispatchComposition(std::shared_ptr<vk::CommandBuffer> cmd,
                                    uint32_t frameIndex,
                                    const std::map<std::string, std::shared_ptr<vk::DeviceLocalImage>> &images,
                                    std::shared_ptr<vk::HostVisibleBuffer> worldUBO,
                                    std::shared_ptr<vk::DeviceLocalImage> outputImage) {
    if (!composePipeline_) return;
    if (!outputImage) {
        std::cerr << "[NrdModule] Error: Output image is NULL in dispatchComposition!" << std::endl;
        return;
    }

    auto descriptorTable = composeDescriptorTables_[frameIndex];

    descriptorTable->bindImage(outputImage, VK_IMAGE_LAYOUT_GENERAL, 0, 0);
    descriptorTable->bindSamplerImage(composeSamplers_[0], images.at("direct"),
                                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 0, 1, 0);
    descriptorTable->bindSamplerImage(composeSamplers_[0], images.at("diffuse"),
                                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 0, 2, 0);
    descriptorTable->bindSamplerImage(composeSamplers_[0], images.at("specular"),
                                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 0, 3, 0);
    descriptorTable->bindSamplerImage(composeSamplers_[0], images.at("albedo"),
                                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 0, 4, 0);
    descriptorTable->bindSamplerImage(composeSamplers_[1], images.at("normal"),
                                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 0, 5, 0);
    descriptorTable->bindSamplerImage(composeSamplers_[1], images.at("depth"), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                      0, 6, 0);
    descriptorTable->bindBuffer(worldUBO, 0, 7);
    descriptorTable->bindSamplerImage(composeSamplers_[0], images.at("specularAlbedo"),
                                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 0, 8, 0);
    descriptorTable->bindSamplerImage(composeSamplers_[0], images.at("clearRadiance"),
                                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 0, 9, 0);
    descriptorTable->bindSamplerImage(composeSamplers_[0], images.at("baseEmission"),
                                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 0, 10, 0);

    cmd->bindDescriptorTable(descriptorTable, VK_PIPELINE_BIND_POINT_COMPUTE)->bindComputePipeline(composePipeline_);

    vkCmdDispatch(cmd->vkCommandBuffer(), (width_ + 15) / 16, (height_ + 15) / 16, 1);
}

void NrdModule::createPreparePipeline(std::shared_ptr<vk::Device> device, uint32_t contextCount) {
    m_nrdMotionVectorImages.resize(contextCount);
    m_nrdNormalRoughnessImages.resize(contextCount);
    m_nrdDiffuseRadianceImages.resize(contextCount);
    m_nrdSpecularRadianceImages.resize(contextCount);
    for (uint32_t i = 0; i < contextCount; ++i) {
        m_nrdMotionVectorImages[i] =
            vk::DeviceLocalImage::create(device, m_vma, false, width_, height_, 1, VK_FORMAT_R16G16_SFLOAT,
                                         VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
        m_nrdNormalRoughnessImages[i] =
            vk::DeviceLocalImage::create(device, m_vma, false, width_, height_, 1, VK_FORMAT_R16G16B16A16_SFLOAT,
                                         VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
        m_nrdDiffuseRadianceImages[i] =
            vk::DeviceLocalImage::create(device, m_vma, false, width_, height_, 1, VK_FORMAT_R16G16B16A16_SFLOAT,
                                         VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
        m_nrdSpecularRadianceImages[i] =
            vk::DeviceLocalImage::create(device, m_vma, false, width_, height_, 1, VK_FORMAT_R16G16B16A16_SFLOAT,
                                         VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    }

    auto framework = framework_.lock();
    prepareDescriptorTables_.resize(contextCount);
    for (uint32_t i = 0; i < contextCount; ++i) {
        prepareDescriptorTables_[i] =
            vk::DescriptorTableBuilder{}
                .beginDescriptorLayoutSet() // set 0
                .beginDescriptorLayoutSetBinding()
                .defineDescriptorLayoutSetBinding({.binding = 0,
                                                   .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                                   .descriptorCount = 1,
                                                   .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT})
                .defineDescriptorLayoutSetBinding({.binding = 1,
                                                   .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                                   .descriptorCount = 1,
                                                   .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT})
                .defineDescriptorLayoutSetBinding({.binding = 2,
                                                   .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                                   .descriptorCount = 1,
                                                   .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT})
                .defineDescriptorLayoutSetBinding({.binding = 3,
                                                   .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                                   .descriptorCount = 1,
                                                   .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT})
                .defineDescriptorLayoutSetBinding({.binding = 4,
                                                   .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                                   .descriptorCount = 1,
                                                   .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT})
                .defineDescriptorLayoutSetBinding({.binding = 5,
                                                   .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                                   .descriptorCount = 1,
                                                   .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT})
                .defineDescriptorLayoutSetBinding({.binding = 6,
                                                   .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                                   .descriptorCount = 1,
                                                   .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT})
                .defineDescriptorLayoutSetBinding({.binding = 7,
                                                   .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                                   .descriptorCount = 1,
                                                   .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT})
                .defineDescriptorLayoutSetBinding({.binding = 8,
                                                   .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                                   .descriptorCount = 1,
                                                   .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT})
                .defineDescriptorLayoutSetBinding({.binding = 9,
                                                   .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                                   .descriptorCount = 1,
                                                   .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT})
                .defineDescriptorLayoutSetBinding({.binding = 10,
                                                   .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                                   .descriptorCount = 1,
                                                   .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT})
                .defineDescriptorLayoutSetBinding({.binding = 11,
                                                   .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                                   .descriptorCount = 1,
                                                   .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT})
                .defineDescriptorLayoutSetBinding({.binding = 12,
                                                   .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                                   .descriptorCount = 1,
                                                   .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT})
                .defineDescriptorLayoutSetBinding({.binding = 13,
                                                   .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                                   .descriptorCount = 1,
                                                   .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT})
                .defineDescriptorLayoutSetBinding({.binding = 14,
                                                   .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                                   .descriptorCount = 1,
                                                   .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT})
                .defineDescriptorLayoutSetBinding({.binding = 15,
                                                   .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                                   .descriptorCount = 1,
                                                   .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT})
                .endDescriptorLayoutSetBinding()
                .endDescriptorLayoutSet()
                .build(framework->device());
    }

    auto shader =
        vk::Shader::create(device, (Renderer::folderPath / "shaders/world/nrd/nrd_prepare_comp.spv").string());
    preparePipeline_ = vk::ComputePipelineBuilder{}
                           .defineShader(shader)
                           .definePipelineLayout(prepareDescriptorTables_[0])
                           .build(device);
}
