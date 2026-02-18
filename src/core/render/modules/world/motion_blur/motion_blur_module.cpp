#include "core/render/modules/world/motion_blur/motion_blur_module.hpp"

#include "core/render/pipeline.hpp"
#include "core/render/render_framework.hpp"
#include "core/render/renderer.hpp"

MotionBlurModule::MotionBlurModule() {}

void MotionBlurModule::init(std::shared_ptr<Framework> framework, std::shared_ptr<WorldPipeline> worldPipeline) {
    WorldModule::init(framework, worldPipeline);

    uint32_t size = framework->swapchain()->imageCount();
    hdrImages_.resize(size);
    motionVectorImages_.resize(size);
    outputImages_.resize(size);
}

bool MotionBlurModule::setOrCreateInputImages(std::vector<std::shared_ptr<vk::DeviceLocalImage>> &images,
                                              std::vector<VkFormat> &formats,
                                              uint32_t frameIndex) {
    if (images.size() < 2) return false;

    auto framework = framework_.lock();

    // images[0]: HDR colour
    if (images[0] == nullptr) {
        hdrImages_[frameIndex] = images[0] = vk::DeviceLocalImage::create(
            framework->device(), framework->vma(), false, width_, height_, 1, formats[0],
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    } else {
        if (images[0]->width() != width_ || images[0]->height() != height_) return false;
        hdrImages_[frameIndex] = images[0];
    }

    // images[1]: motion vectors (must be provided by a prior module — e.g. TemporalAccumulation)
    if (images[1] == nullptr) return false;
    motionVectorImages_[frameIndex] = images[1];

    return true;
}

bool MotionBlurModule::setOrCreateOutputImages(std::vector<std::shared_ptr<vk::DeviceLocalImage>> &images,
                                               std::vector<VkFormat> &formats,
                                               uint32_t frameIndex) {
    if (images.empty() || images[0] == nullptr) return false;

    width_  = images[0]->width();
    height_ = images[0]->height();

    outputImages_[frameIndex] = images[0];
    return true;
}

void MotionBlurModule::setAttributes(int attributeCount, std::vector<std::string> &attributeKVs) {}

void MotionBlurModule::build() {
    auto framework     = framework_.lock();
    auto worldPipeline = worldPipeline_.lock();
    uint32_t size      = framework->swapchain()->imageCount();

    initDescriptorTables();
    initImages();
    initPipeline();

    contexts_.resize(size);
    for (int i = 0; i < (int)size; i++) {
        contexts_[i] = MotionBlurModuleContext::create(framework->contexts()[i], worldPipeline->contexts()[i],
                                                       shared_from_this());
    }
}

std::vector<std::shared_ptr<WorldModuleContext>> &MotionBlurModule::contexts() {
    return contexts_;
}

void MotionBlurModule::bindTexture(std::shared_ptr<vk::Sampler> sampler,
                                   std::shared_ptr<vk::DeviceLocalImage> image,
                                   int index) {}

void MotionBlurModule::preClose() {}

void MotionBlurModule::initDescriptorTables() {
    auto framework = framework_.lock();
    uint32_t size  = framework->swapchain()->imageCount();

    descriptorTables_.resize(size);
    hdrSamplers_.resize(size);
    mvSamplers_.resize(size);

    for (int i = 0; i < (int)size; i++) {
        descriptorTables_[i] =
            vk::DescriptorTableBuilder{}
                .beginDescriptorLayoutSet()  // set 0
                .beginDescriptorLayoutSetBinding()
                .defineDescriptorLayoutSetBinding({
                    .binding        = 0,
                    .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    .descriptorCount = 1,
                    .stageFlags     = VK_SHADER_STAGE_COMPUTE_BIT,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding        = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    .descriptorCount = 1,
                    .stageFlags     = VK_SHADER_STAGE_COMPUTE_BIT,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding        = 2,
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .descriptorCount = 1,
                    .stageFlags     = VK_SHADER_STAGE_COMPUTE_BIT,
                })
                .endDescriptorLayoutSetBinding()
                .endDescriptorLayoutSet()
                .definePushConstant(VkPushConstantRange{
                    .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                    .offset     = 0,
                    .size       = sizeof(MotionBlurModulePushConstant),
                })
                .build(framework->device());

        hdrSamplers_[i] = vk::Sampler::create(framework->device(), VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_LINEAR,
                                               VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
        mvSamplers_[i]  = vk::Sampler::create(framework->device(), VK_FILTER_NEAREST, VK_SAMPLER_MIPMAP_MODE_NEAREST,
                                               VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
    }
}

void MotionBlurModule::initImages() {
    auto framework = framework_.lock();
    uint32_t size  = framework->swapchain()->imageCount();

    for (int i = 0; i < (int)size; i++) {
        descriptorTables_[i]->bindSamplerImageForShader(hdrSamplers_[i], hdrImages_[i], 0, 0);
        descriptorTables_[i]->bindSamplerImageForShader(mvSamplers_[i],  motionVectorImages_[i], 0, 1);
        descriptorTables_[i]->bindImage(outputImages_[i], VK_IMAGE_LAYOUT_GENERAL, 0, 2);
    }
}

void MotionBlurModule::initPipeline() {
    auto framework   = framework_.lock();
    auto device      = framework->device();
    std::filesystem::path shaderPath = Renderer::folderPath / "shaders";

    motionBlurShader_ = vk::Shader::create(device, (shaderPath / "world/post_render/motion_blur_comp.spv").string());
    motionBlurPipeline_ = vk::ComputePipelineBuilder{}
                              .defineShader(motionBlurShader_)
                              .definePipelineLayout(descriptorTables_[0])
                              .build(device);
}

// ---------------------------------------------------------------------------
// MotionBlurModuleContext
// ---------------------------------------------------------------------------

MotionBlurModuleContext::MotionBlurModuleContext(std::shared_ptr<FrameworkContext>     frameworkContext,
                                                  std::shared_ptr<WorldPipelineContext>  worldPipelineContext,
                                                  std::shared_ptr<MotionBlurModule>      motionBlurModule)
    : WorldModuleContext(frameworkContext, worldPipelineContext),
      motionBlurModule(motionBlurModule),
      hdrImage(motionBlurModule->hdrImages_[frameworkContext->frameIndex]),
      motionVectorImage(motionBlurModule->motionVectorImages_[frameworkContext->frameIndex]),
      outputImage(motionBlurModule->outputImages_[frameworkContext->frameIndex]),
      descriptorTable(motionBlurModule->descriptorTables_[frameworkContext->frameIndex]) {}

void MotionBlurModuleContext::render() {
    auto context            = frameworkContext.lock();
    auto framework          = context->framework.lock();
    auto worldCommandBuffer = context->worldCommandBuffer;
    auto mainQueueIndex     = framework->physicalDevice()->mainQueueIndex();

    auto module = motionBlurModule.lock();

    worldCommandBuffer->barriersBufferImage(
        {},
        {
            {
                .srcStageMask   = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                .srcAccessMask  = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                .dstStageMask   = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                .dstAccessMask  = VK_ACCESS_2_SHADER_READ_BIT,
                .oldLayout      = hdrImage->imageLayout(),
                .newLayout      = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                .srcQueueFamilyIndex = mainQueueIndex,
                .dstQueueFamilyIndex = mainQueueIndex,
                .image          = hdrImage,
                .subresourceRange = vk::wholeColorSubresourceRange,
            },
            {
                .srcStageMask   = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                .srcAccessMask  = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                .dstStageMask   = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                .dstAccessMask  = VK_ACCESS_2_SHADER_READ_BIT,
                .oldLayout      = motionVectorImage->imageLayout(),
                .newLayout      = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                .srcQueueFamilyIndex = mainQueueIndex,
                .dstQueueFamilyIndex = mainQueueIndex,
                .image          = motionVectorImage,
                .subresourceRange = vk::wholeColorSubresourceRange,
            },
            {
                .srcStageMask   = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                .srcAccessMask  = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                .dstStageMask   = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                .dstAccessMask  = VK_ACCESS_2_SHADER_WRITE_BIT,
                .oldLayout      = outputImage->imageLayout(),
                .newLayout      = VK_IMAGE_LAYOUT_GENERAL,
                .srcQueueFamilyIndex = mainQueueIndex,
                .dstQueueFamilyIndex = mainQueueIndex,
                .image          = outputImage,
                .subresourceRange = vk::wholeColorSubresourceRange,
            },
        });

    hdrImage->imageLayout()          = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    motionVectorImage->imageLayout() = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    outputImage->imageLayout()       = VK_IMAGE_LAYOUT_GENERAL;

    MotionBlurModulePushConstant pc{};
    pc.enabled  = Renderer::options.motionBlurEnabled ? 1.0f : 0.0f;
    pc.strength = Renderer::options.motionBlurStrength;
    pc.samples  = static_cast<float>(Renderer::options.motionBlurSamples);

    vkCmdPushConstants(worldCommandBuffer->vkCommandBuffer(),
                       descriptorTable->vkPipelineLayout(),
                       VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(MotionBlurModulePushConstant), &pc);

    worldCommandBuffer->bindDescriptorTable(descriptorTable, VK_PIPELINE_BIND_POINT_COMPUTE)
        ->bindComputePipeline(module->motionBlurPipeline_);

    uint32_t groupX = (module->width_  + 15) / 16;
    uint32_t groupY = (module->height_ + 15) / 16;
    vkCmdDispatch(worldCommandBuffer->vkCommandBuffer(), groupX, groupY, 1);

    worldCommandBuffer->barriersBufferImage(
        {},
        {{
            .srcStageMask   = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            .srcAccessMask  = VK_ACCESS_2_SHADER_WRITE_BIT,
            .dstStageMask   = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            .dstAccessMask  = VK_ACCESS_2_SHADER_READ_BIT,
            .oldLayout      = VK_IMAGE_LAYOUT_GENERAL,
            .newLayout      = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            .srcQueueFamilyIndex = mainQueueIndex,
            .dstQueueFamilyIndex = mainQueueIndex,
            .image          = outputImage,
            .subresourceRange = vk::wholeColorSubresourceRange,
        }});

    outputImage->imageLayout() = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
}
