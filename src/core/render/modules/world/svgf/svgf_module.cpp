#include "svgf_module.hpp"
#include "core/render/buffers.hpp"
#include "core/render/pipeline.hpp"
#include "core/render/render_framework.hpp"
#include "core/render/renderer.hpp"
#include <iostream>

SvgfModule::SvgfModule() {}

void SvgfModule::init(std::shared_ptr<Framework> framework, std::shared_ptr<WorldPipeline> worldPipeline) {
    WorldModule::init(framework, worldPipeline);

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
    denoisedRadianceImages_.resize(size);
    denoisedDiffuseRadianceImages_.resize(size);
    denoisedSpecularRadianceImages_.resize(size);
}

bool SvgfModule::setOrCreateInputImages(std::vector<std::shared_ptr<vk::DeviceLocalImage>> &images,
                                        std::vector<VkFormat> &formats,
                                        uint32_t frameIndex) {
    if (images.size() != inputImageNum) return false;

    auto framework = framework_.lock();
    auto createImage = [&](uint32_t index) {
        images[index] = vk::DeviceLocalImage::create(
            framework->device(), framework->vma(), false, width_, height_, 1, formats[index],
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    };

    for (uint32_t i = 0; i < images.size(); i++) {
        if (images[i] == nullptr) {
            createImage(i);
        } else if (images[i]->width() != width_ || images[i]->height() != height_) {
            return false;
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

    return true;
}

bool SvgfModule::setOrCreateOutputImages(std::vector<std::shared_ptr<vk::DeviceLocalImage>> &images,
                                         std::vector<VkFormat> &formats,
                                         uint32_t frameIndex) {
    if (images.size() != outputImageNum) return false;
    if (images[0] == nullptr) return false;

    width_ = images[0]->width();
    height_ = images[0]->height();

    denoisedRadianceImages_[frameIndex] = images[0];

    return true;
}

void SvgfModule::build() {
    auto framework = framework_.lock();
    auto worldPipeline = worldPipeline_.lock();
    uint32_t size = framework->swapchain()->imageCount();

    m_denoiser = std::make_shared<SvgfDenoiser>();

    auto createInternal = [&](std::vector<std::shared_ptr<vk::DeviceLocalImage>> &images) {
        for (uint32_t i = 0; i < size; i++) {
            if (images[i] != nullptr) continue;
            images[i] = vk::DeviceLocalImage::create(framework->device(), framework->vma(), false, width_, height_, 1,
                                                     VK_FORMAT_R16G16B16A16_SFLOAT,
                                                     VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
        }
    };
    createInternal(denoisedDiffuseRadianceImages_);
    createInternal(denoisedSpecularRadianceImages_);

    bool ok = m_denoiser->init(framework->instance(), framework->physicalDevice(), framework->device(),
                               framework->vma(), width_, height_, size);
    if (!ok) {
        std::cerr << "[SvgfModule] init failed." << std::endl;
        m_denoiser.reset();
    }

    contexts_.resize(size);
    for (int i = 0; i < size; i++) {
        contexts_[i] =
            SvgfModuleContext::create(framework->contexts()[i], worldPipeline->contexts()[i], shared_from_this());
    }
}

void SvgfModule::setAttributes(int attributeCount, std::vector<std::string> &attributeKVs) {}

std::vector<std::shared_ptr<WorldModuleContext>> &SvgfModule::contexts() {
    return contexts_;
}

void SvgfModule::bindTexture(std::shared_ptr<vk::Sampler> sampler,
                             std::shared_ptr<vk::DeviceLocalImage> image,
                             int index) {}

void SvgfModule::preClose() {
    m_denoiser.reset();
}

// --- Render Logic ---

SvgfModuleContext::SvgfModuleContext(std::shared_ptr<FrameworkContext> frameworkContext,
                                     std::shared_ptr<WorldPipelineContext> worldPipelineContext,
                                     std::shared_ptr<SvgfModule> svgfModule)
    : WorldModuleContext(frameworkContext, worldPipelineContext),
      svgfModule(svgfModule),
      diffuseRadianceImage(svgfModule->diffuseRadianceImages_[frameworkContext->frameIndex]),
      specularRadianceImage(svgfModule->specularRadianceImages_[frameworkContext->frameIndex]),
      directRadianceImage(svgfModule->directRadianceImages_[frameworkContext->frameIndex]),
      diffuseAlbedoImage(svgfModule->diffuseAlbedoImages_[frameworkContext->frameIndex]),
      specularAlbedoImage(svgfModule->specularAlbedoImages_[frameworkContext->frameIndex]),
      normalRoughnessImage(svgfModule->normalRoughnessImages_[frameworkContext->frameIndex]),
      motionVectorImage(svgfModule->motionVectorImages_[frameworkContext->frameIndex]),
      linearDepthImage(svgfModule->linearDepthImages_[frameworkContext->frameIndex]),
      clearRadianceImage(svgfModule->clearRadianceImages_[frameworkContext->frameIndex]),
      baseEmissionImage(svgfModule->baseEmissionImages_[frameworkContext->frameIndex]),
      denoisedRadianceImage(svgfModule->denoisedRadianceImages_[frameworkContext->frameIndex]),
      denoisedDiffuseRadianceImage(svgfModule->denoisedDiffuseRadianceImages_[frameworkContext->frameIndex]),
      denoisedSpecularRadianceImage(svgfModule->denoisedSpecularRadianceImages_[frameworkContext->frameIndex]) {}

void SvgfModuleContext::render() {
    auto module = svgfModule.lock();
    if (!module || !module->denoiser()) return;

    auto context = frameworkContext.lock();
    auto worldCommandBuffer = context->worldCommandBuffer;

    // Transition intermediate outputs to General layout before compute
    std::vector<vk::CommandBuffer::ImageMemoryBarrier> initBarriers;
    auto initImage = [&](std::shared_ptr<vk::DeviceLocalImage> image) {
        if (!image || image->imageLayout() != VK_IMAGE_LAYOUT_UNDEFINED) return;
        initBarriers.push_back({
            .srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
            .srcAccessMask = 0,
            .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = VK_IMAGE_LAYOUT_GENERAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = image,
            .subresourceRange = vk::wholeColorSubresourceRange,
        });
        image->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;
    };
    initImage(denoisedDiffuseRadianceImage);
    initImage(denoisedSpecularRadianceImage);
    if (!initBarriers.empty()) { worldCommandBuffer->barriersBufferImage({}, initBarriers); }

    SvgfInputs inputs{};
    inputs.noisyDiffuseRadiance = diffuseRadianceImage;
    inputs.noisySpecularRadiance = specularRadianceImage;
    inputs.directRadiance = directRadianceImage;
    inputs.motionVectors = motionVectorImage;
    inputs.linearDepth = linearDepthImage;
    inputs.normalRoughness = normalRoughnessImage;
    inputs.diffuseAlbedo = diffuseAlbedoImage;
    inputs.specularAlbedo = specularAlbedoImage;
    inputs.clearRadiance = clearRadianceImage;
    inputs.baseEmission = baseEmissionImage;
    inputs.worldUniformBuffer = Renderer::instance().buffers()->worldUniformBuffer();

    SvgfOutputs outputs{};
    outputs.hdrOutput = denoisedRadianceImage;

    module->denoiser()->denoise(worldCommandBuffer, inputs, outputs, context->frameIndex);

    denoisedRadianceImage->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;
}
