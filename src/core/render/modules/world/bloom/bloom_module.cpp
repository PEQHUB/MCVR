#include "core/render/modules/world/bloom/bloom_module.hpp"

#include "core/render/pipeline.hpp"
#include "core/render/render_framework.hpp"
#include "core/render/renderer.hpp"

BloomModule::BloomModule() {}

void BloomModule::init(std::shared_ptr<Framework> framework, std::shared_ptr<WorldPipeline> worldPipeline) {
    WorldModule::init(framework, worldPipeline);

    uint32_t size = framework->swapchain()->imageCount();
    inputImages_.resize(size);
    outputImages_.resize(size);
}

bool BloomModule::setOrCreateInputImages(std::vector<std::shared_ptr<vk::DeviceLocalImage>> &images,
                                         std::vector<VkFormat> &formats,
                                         uint32_t frameIndex) {
    if (images.empty()) return false;

    auto framework = framework_.lock();
    if (images[0] == nullptr) {
        inputImages_[frameIndex] = images[0] = vk::DeviceLocalImage::create(
            framework->device(), framework->vma(), false, width_, height_, 1, formats[0],
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    } else {
        if (images[0]->width() != width_ || images[0]->height() != height_) return false;
        inputImages_[frameIndex] = images[0];
    }

    return true;
}

bool BloomModule::setOrCreateOutputImages(std::vector<std::shared_ptr<vk::DeviceLocalImage>> &images,
                                          std::vector<VkFormat> &formats,
                                          uint32_t frameIndex) {
    if (images.empty() || images[0] == nullptr) return false;

    width_  = images[0]->width();
    height_ = images[0]->height();

    outputImages_[frameIndex] = images[0];
    return true;
}

void BloomModule::setAttributes(int attributeCount, std::vector<std::string> &attributeKVs) {
    // No pipeline-configuration attributes — all driven by Renderer::options at runtime.
}

void BloomModule::build() {
    auto framework    = framework_.lock();
    auto worldPipeline = worldPipeline_.lock();
    uint32_t size     = framework->swapchain()->imageCount();

    initDescriptorTables();
    initImages();
    initPipeline();

    contexts_.resize(size);
    for (int i = 0; i < (int)size; i++) {
        contexts_[i] = BloomModuleContext::create(framework->contexts()[i], worldPipeline->contexts()[i],
                                                   shared_from_this());
    }
}

std::vector<std::shared_ptr<WorldModuleContext>> &BloomModule::contexts() {
    return contexts_;
}

void BloomModule::bindTexture(std::shared_ptr<vk::Sampler> sampler,
                               std::shared_ptr<vk::DeviceLocalImage> image,
                               int index) {}

void BloomModule::preClose() {}

void BloomModule::initDescriptorTables() {
    auto framework = framework_.lock();
    uint32_t size  = framework->swapchain()->imageCount();

    descriptorTables_.resize(size);
    samplers_.resize(size);

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
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .descriptorCount = 1,
                    .stageFlags     = VK_SHADER_STAGE_COMPUTE_BIT,
                })
                .endDescriptorLayoutSetBinding()
                .endDescriptorLayoutSet()
                .definePushConstant(VkPushConstantRange{
                    .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                    .offset     = 0,
                    .size       = sizeof(BloomModulePushConstant),
                })
                .build(framework->device());

        samplers_[i] = vk::Sampler::create(framework->device(), VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_LINEAR,
                                            VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
    }
}

void BloomModule::initImages() {
    auto framework = framework_.lock();
    uint32_t size  = framework->swapchain()->imageCount();

    for (int i = 0; i < (int)size; i++) {
        descriptorTables_[i]->bindSamplerImageForShader(samplers_[i], inputImages_[i], 0, 0);
        descriptorTables_[i]->bindImage(outputImages_[i], VK_IMAGE_LAYOUT_GENERAL, 0, 1);
    }
}

void BloomModule::initPipeline() {
    auto framework   = framework_.lock();
    auto device      = framework->device();
    std::filesystem::path shaderPath = Renderer::folderPath / "shaders";

    bloomShader_ = vk::Shader::create(device, (shaderPath / "world/post_render/bloom_comp.spv").string());
    bloomPipeline_ = vk::ComputePipelineBuilder{}
                         .defineShader(bloomShader_)
                         .definePipelineLayout(descriptorTables_[0])
                         .build(device);
}

// ---------------------------------------------------------------------------
// BloomModuleContext
// ---------------------------------------------------------------------------

BloomModuleContext::BloomModuleContext(std::shared_ptr<FrameworkContext>     frameworkContext,
                                       std::shared_ptr<WorldPipelineContext>  worldPipelineContext,
                                       std::shared_ptr<BloomModule>           bloomModule)
    : WorldModuleContext(frameworkContext, worldPipelineContext),
      bloomModule(bloomModule),
      inputImage(bloomModule->inputImages_[frameworkContext->frameIndex]),
      outputImage(bloomModule->outputImages_[frameworkContext->frameIndex]),
      descriptorTable(bloomModule->descriptorTables_[frameworkContext->frameIndex]) {}

void BloomModuleContext::render() {
    auto context         = frameworkContext.lock();
    auto framework       = context->framework.lock();
    auto worldCommandBuffer = context->worldCommandBuffer;
    auto mainQueueIndex  = framework->physicalDevice()->mainQueueIndex();

    auto module = bloomModule.lock();

    // Transition input to shader-read, output to general (storage write)
    worldCommandBuffer->barriersBufferImage(
        {},
        {
            {
                .srcStageMask   = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                .srcAccessMask  = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                .dstStageMask   = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                .dstAccessMask  = VK_ACCESS_2_SHADER_READ_BIT,
                .oldLayout      = inputImage->imageLayout(),
                .newLayout      = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                .srcQueueFamilyIndex = mainQueueIndex,
                .dstQueueFamilyIndex = mainQueueIndex,
                .image          = inputImage,
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

    inputImage->imageLayout()  = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    outputImage->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;

    BloomModulePushConstant pc{};
    pc.enabled   = Renderer::options.bloomEnabled ? 1.0f : 0.0f;
    pc.threshold = Renderer::options.bloomThreshold;
    pc.strength  = Renderer::options.bloomStrength;
    pc.radius    = Renderer::options.bloomRadius;

    vkCmdPushConstants(worldCommandBuffer->vkCommandBuffer(),
                       descriptorTable->vkPipelineLayout(),
                       VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(BloomModulePushConstant), &pc);

    worldCommandBuffer->bindDescriptorTable(descriptorTable, VK_PIPELINE_BIND_POINT_COMPUTE)
        ->bindComputePipeline(module->bloomPipeline_);

    uint32_t groupX = (module->width_  + 15) / 16;
    uint32_t groupY = (module->height_ + 15) / 16;
    vkCmdDispatch(worldCommandBuffer->vkCommandBuffer(), groupX, groupY, 1);

    // Transition output to shader-read for subsequent passes
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
