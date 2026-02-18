#include "core/render/modules/world/dof/dof_module.hpp"

#include "core/render/pipeline.hpp"
#include "core/render/render_framework.hpp"
#include "core/render/renderer.hpp"

DofModule::DofModule() {}

void DofModule::init(std::shared_ptr<Framework> framework, std::shared_ptr<WorldPipeline> worldPipeline) {
    WorldModule::init(framework, worldPipeline);

    uint32_t size = framework->swapchain()->imageCount();
    hdrImages_.resize(size);
    depthImages_.resize(size);
    outputImages_.resize(size);
}

bool DofModule::setOrCreateInputImages(std::vector<std::shared_ptr<vk::DeviceLocalImage>> &images,
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

    // images[1]: linear depth (must be provided by a prior module — e.g. NRD)
    if (images[1] == nullptr) return false;
    depthImages_[frameIndex] = images[1];

    return true;
}

bool DofModule::setOrCreateOutputImages(std::vector<std::shared_ptr<vk::DeviceLocalImage>> &images,
                                        std::vector<VkFormat> &formats,
                                        uint32_t frameIndex) {
    if (images.empty() || images[0] == nullptr) return false;

    width_  = images[0]->width();
    height_ = images[0]->height();

    outputImages_[frameIndex] = images[0];
    return true;
}

void DofModule::setAttributes(int attributeCount, std::vector<std::string> &attributeKVs) {}

void DofModule::build() {
    auto framework     = framework_.lock();
    auto worldPipeline = worldPipeline_.lock();
    uint32_t size      = framework->swapchain()->imageCount();

    initDescriptorTables();
    initImages();
    initPipeline();

    contexts_.resize(size);
    for (int i = 0; i < (int)size; i++) {
        contexts_[i] = DofModuleContext::create(framework->contexts()[i], worldPipeline->contexts()[i],
                                                 shared_from_this());
    }
}

std::vector<std::shared_ptr<WorldModuleContext>> &DofModule::contexts() {
    return contexts_;
}

void DofModule::bindTexture(std::shared_ptr<vk::Sampler> sampler,
                             std::shared_ptr<vk::DeviceLocalImage> image,
                             int index) {}

void DofModule::preClose() {}

void DofModule::initDescriptorTables() {
    auto framework = framework_.lock();
    uint32_t size  = framework->swapchain()->imageCount();

    descriptorTables_.resize(size);
    hdrSamplers_.resize(size);
    depthSamplers_.resize(size);

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
                    .size       = sizeof(DofModulePushConstant),
                })
                .build(framework->device());

        hdrSamplers_[i]   = vk::Sampler::create(framework->device(), VK_FILTER_LINEAR,  VK_SAMPLER_MIPMAP_MODE_LINEAR,
                                                  VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
        depthSamplers_[i] = vk::Sampler::create(framework->device(), VK_FILTER_NEAREST, VK_SAMPLER_MIPMAP_MODE_NEAREST,
                                                  VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
    }
}

void DofModule::initImages() {
    auto framework = framework_.lock();
    uint32_t size  = framework->swapchain()->imageCount();

    for (int i = 0; i < (int)size; i++) {
        descriptorTables_[i]->bindSamplerImageForShader(hdrSamplers_[i],   hdrImages_[i],   0, 0);
        descriptorTables_[i]->bindSamplerImageForShader(depthSamplers_[i], depthImages_[i], 0, 1);
        descriptorTables_[i]->bindImage(outputImages_[i], VK_IMAGE_LAYOUT_GENERAL, 0, 2);
    }
}

void DofModule::initPipeline() {
    auto framework   = framework_.lock();
    auto device      = framework->device();
    std::filesystem::path shaderPath = Renderer::folderPath / "shaders";

    dofShader_ = vk::Shader::create(device, (shaderPath / "world/post_render/dof_comp.spv").string());
    dofPipeline_ = vk::ComputePipelineBuilder{}
                       .defineShader(dofShader_)
                       .definePipelineLayout(descriptorTables_[0])
                       .build(device);
}

// ---------------------------------------------------------------------------
// DofModuleContext
// ---------------------------------------------------------------------------

DofModuleContext::DofModuleContext(std::shared_ptr<FrameworkContext>     frameworkContext,
                                   std::shared_ptr<WorldPipelineContext>  worldPipelineContext,
                                   std::shared_ptr<DofModule>             dofModule)
    : WorldModuleContext(frameworkContext, worldPipelineContext),
      dofModule(dofModule),
      hdrImage(dofModule->hdrImages_[frameworkContext->frameIndex]),
      depthImage(dofModule->depthImages_[frameworkContext->frameIndex]),
      outputImage(dofModule->outputImages_[frameworkContext->frameIndex]),
      descriptorTable(dofModule->descriptorTables_[frameworkContext->frameIndex]) {}

void DofModuleContext::render() {
    auto context            = frameworkContext.lock();
    auto framework          = context->framework.lock();
    auto worldCommandBuffer = context->worldCommandBuffer;
    auto mainQueueIndex     = framework->physicalDevice()->mainQueueIndex();

    auto module = dofModule.lock();

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
                .oldLayout      = depthImage->imageLayout(),
                .newLayout      = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                .srcQueueFamilyIndex = mainQueueIndex,
                .dstQueueFamilyIndex = mainQueueIndex,
                .image          = depthImage,
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

    hdrImage->imageLayout()   = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    depthImage->imageLayout() = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    outputImage->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;

    DofModulePushConstant pc{};
    pc.enabled       = Renderer::options.dofEnabled ? 1.0f : 0.0f;
    pc.focalDistance = Renderer::options.dofFocalDistance;
    pc.aperture      = Renderer::options.dofAperture;
    pc.maxRadius     = static_cast<float>(Renderer::options.dofMaxRadius);

    vkCmdPushConstants(worldCommandBuffer->vkCommandBuffer(),
                       descriptorTable->vkPipelineLayout(),
                       VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(DofModulePushConstant), &pc);

    worldCommandBuffer->bindDescriptorTable(descriptorTable, VK_PIPELINE_BIND_POINT_COMPUTE)
        ->bindComputePipeline(module->dofPipeline_);

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
