#include "core/render/modules/world/ray_tracing/submodules/atmosphere.hpp"

#include "core/render/buffers.hpp"
#include "core/render/modules/world/ray_tracing/ray_tracing_module.hpp"
#include "core/render/render_framework.hpp"
#include "core/render/renderer.hpp"

#include <filesystem>

Atmosphere::Atmosphere() {}

void Atmosphere::init(std::shared_ptr<Framework> framework, std::shared_ptr<RayTracingModule> rayTracingModule) {
    framework_ = framework;
    rayTracingModule_ = rayTracingModule;
}

void Atmosphere::build() {
    auto framework = framework_.lock();
    auto rayTracingModule = rayTracingModule_.lock();
    uint32_t size = framework->swapchain()->imageCount();

    contexts_.resize(size);

    initDescriptorTables();
    initImages();
    initAtmLUTRenderPass();
    initAtmCubeMapRenderPass();
    initFrameBuffers();
    initAtmLUTPipeline();
    initAtmCubeMapPipeline();

    for (int i = 0; i < size; i++) {
        contexts_[i] = AtmosphereContext::create(framework->contexts()[i], shared_from_this());
    }
}

void Atmosphere::initDescriptorTables() {
    auto framework = framework_.lock();

    atmLUTImageSampler_ = vk::Sampler::create(framework->device(), VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_LINEAR,
                                              VK_SAMPLER_ADDRESS_MODE_REPEAT);

    uint32_t size = framework->swapchain()->imageCount();
    atmDescriptorTables_.resize(size);
    atmCubeMapImageSamplers_.resize(size);

    for (int i = 0; i < size; i++) {
        atmDescriptorTables_[i] =
            vk::DescriptorTableBuilder{}
                .beginDescriptorLayoutSet() // set 0
                .beginDescriptorLayoutSetBinding()
                .defineDescriptorLayoutSetBinding({
                    .binding = 0, // world atmosphere LUT
                    .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                                  VK_SHADER_STAGE_FRAGMENT_BIT,
                })
                .endDescriptorLayoutSetBinding()
                .endDescriptorLayoutSet()
                .beginDescriptorLayoutSet() // set 1
                .beginDescriptorLayoutSetBinding()
                .defineDescriptorLayoutSetBinding({
                    .binding = 0, // binding 0: world ubo
                    .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                                  VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 1, // binding 2: sky ubo
                    .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                                  VK_SHADER_STAGE_ANY_HIT_BIT_KHR | VK_SHADER_STAGE_VERTEX_BIT |
                                  VK_SHADER_STAGE_FRAGMENT_BIT,
                })
                .endDescriptorLayoutSetBinding()
                .endDescriptorLayoutSet()
                .definePushConstant(VkPushConstantRange{
                    .stageFlags = VK_SHADER_STAGE_ALL,
                    .offset = 0,
                    .size = sizeof(int),
                })
                .build(framework->device());

        atmCubeMapImageSamplers_[i] = vk::Sampler::create(
            framework->device(), VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_LINEAR, VK_SAMPLER_ADDRESS_MODE_REPEAT);
    }
}

void Atmosphere::initImages() {
    auto framework = framework_.lock();

    atmLUTImage_ = vk::DeviceLocalImage::create(framework->device(), framework->vma(), false, 512, 128, 1,
                                                VK_FORMAT_R16G16B16A16_SFLOAT,
                                                VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);

    uint32_t size = framework->swapchain()->imageCount();
    atmCubeMapImages_.resize(size);

    for (int i = 0; i < size; i++) {
        atmDescriptorTables_[i]->bindSamplerImageForShader(atmLUTImageSampler_, atmLUTImage_, 0, 0);

        {
            atmCubeMapImages_[i] = vk::DeviceLocalImage::create(
                framework->device(), framework->vma(), false, 512, 512, 6, VK_FORMAT_R16G16B16A16_SFLOAT,
                VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, 0,
                VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT);

            // 1...6
            for (int faceIndex = 0; faceIndex < 6; faceIndex++) {
                atmCubeMapImages_[i]->addImageView(VkImageViewCreateInfo{
                    .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                    .image = atmCubeMapImages_[i]->vkImage(),
                    .viewType = VK_IMAGE_VIEW_TYPE_2D,
                    .format = atmCubeMapImages_[i]->vkFormat(),
                    .components =
                        {
                            .r = VK_COMPONENT_SWIZZLE_IDENTITY,
                            .g = VK_COMPONENT_SWIZZLE_IDENTITY,
                            .b = VK_COMPONENT_SWIZZLE_IDENTITY,
                            .a = VK_COMPONENT_SWIZZLE_IDENTITY,
                        },
                    .subresourceRange =
                        {
                            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                            .baseMipLevel = 0,
                            .levelCount = 1,
                            .baseArrayLayer = static_cast<uint32_t>(faceIndex),
                            .layerCount = 1,
                        },
                });
            }

            // 7
            atmCubeMapImages_[i]->addImageView(VkImageViewCreateInfo{
                .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                .image = atmCubeMapImages_[i]->vkImage(),
                .viewType = VK_IMAGE_VIEW_TYPE_CUBE,
                .format = atmCubeMapImages_[i]->vkFormat(),
                .components =
                    {
                        .r = VK_COMPONENT_SWIZZLE_IDENTITY,
                        .g = VK_COMPONENT_SWIZZLE_IDENTITY,
                        .b = VK_COMPONENT_SWIZZLE_IDENTITY,
                        .a = VK_COMPONENT_SWIZZLE_IDENTITY,
                    },
                .subresourceRange =
                    {
                        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                        .baseMipLevel = 0,
                        .levelCount = 1,
                        .baseArrayLayer = 0,
                        .layerCount = 6,
                    },
            });
        }
    }
}

void Atmosphere::initAtmLUTRenderPass() {
    atmLUTRenderPass_ = vk::RenderPassBuilder{}
                            .beginAttachmentDescription()
                            .defineAttachmentDescription({
                                // color
                                .format = atmLUTImage_->vkFormat(),
                                .samples = VK_SAMPLE_COUNT_1_BIT,
                                .loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                                .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
                                .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                                .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
                                .initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                .finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
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

void Atmosphere::initAtmCubeMapRenderPass() {
    atmCubeMapRenderPass_ = vk::RenderPassBuilder{}
                                .beginAttachmentDescription()
                                .defineAttachmentDescription({
                                    // color
                                    .format = atmCubeMapImages_[0]->vkFormat(),
                                    .samples = VK_SAMPLE_COUNT_1_BIT,
                                    .loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                                    .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
                                    .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                                    .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
                                    .initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                    .finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
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

void Atmosphere::initFrameBuffers() {
    auto framework = framework_.lock();
    auto rayTracingModule = rayTracingModule_.lock();

    atmLUTFramebuffer_ = vk::FramebufferBuilder{}
                             .beginAttachment()
                             .defineAttachment(atmLUTImage_)
                             .endAttachment()
                             .build(framework->device(), atmLUTRenderPass_);

    uint32_t size = framework->swapchain()->imageCount();
    atmCubeMapFramebuffers_.resize(size);

    for (int i = 0; i < size; i++) {
        for (int faceIndex = 0; faceIndex < 6; faceIndex++) {
            atmCubeMapFramebuffers_[i][faceIndex] = vk::FramebufferBuilder{}
                                                        .beginAttachment()
                                                        .defineAttachment(atmCubeMapImages_[i], faceIndex + 1)
                                                        .endAttachment()
                                                        .build(framework->device(), atmCubeMapRenderPass_);
        }
    }
}

void Atmosphere::initAtmLUTPipeline() {
    auto framework = framework_.lock();
    std::filesystem::path shaderPath = Renderer::folderPath / "shaders";
    atmLUTVertShader_ = vk::Shader::create(framework->device(),
                                           (shaderPath / "world/ray_tracing/atmosphere/trans_lut_vert.spv").string());
    atmLUTFragShader_ = vk::Shader::create(framework->device(),
                                           (shaderPath / "world/ray_tracing/atmosphere/trans_lut_frag.spv").string());

    atmLUTPipeline_ = vk::GraphicsPipelineBuilder{}
                          .defineRenderPass(atmLUTRenderPass_, 0)
                          .beginShaderStage()
                          .defineShaderStage(atmLUTVertShader_, VK_SHADER_STAGE_VERTEX_BIT)
                          .defineShaderStage(atmLUTFragShader_, VK_SHADER_STAGE_FRAGMENT_BIT)
                          .endShaderStage()
                          .defineVertexInputState<void>()
                          .defineViewportScissorState({
                              .viewport =
                                  {
                                      .x = 0,
                                      .y = 0,
                                      .width = static_cast<float>(atmLUTImage_->width()),
                                      .height = static_cast<float>(atmLUTImage_->height()),
                                      .minDepth = 0.0,
                                      .maxDepth = 1.0,
                                  },
                              .scissor =
                                  {
                                      .offset = {.x = 0, .y = 0},
                                      .extent =
                                          {
                                              .width = atmLUTImage_->width(),
                                              .height = atmLUTImage_->height(),
                                          },
                                  },
                          })
                          .beginColorBlendAttachmentState()
                          .defineDefaultColorBlendAttachmentState() // color
                          .endColorBlendAttachmentState()
                          .definePipelineLayout(atmDescriptorTables_[0])
                          .build(framework->device());
}

void Atmosphere::initAtmCubeMapPipeline() {
    auto framework = framework_.lock();
    std::filesystem::path shaderPath = Renderer::folderPath / "shaders";
    atmCubeMapVertShader_ = vk::Shader::create(framework->device(),
                                               (shaderPath / "world/ray_tracing/atmosphere/skycube_vert.spv").string());
    atmCubeMapFragShader_ = vk::Shader::create(framework->device(),
                                               (shaderPath / "world/ray_tracing/atmosphere/skycube_frag.spv").string());

    atmCubeMapPipeline_ = vk::GraphicsPipelineBuilder{}
                              .defineRenderPass(atmCubeMapRenderPass_, 0)
                              .beginShaderStage()
                              .defineShaderStage(atmCubeMapVertShader_, VK_SHADER_STAGE_VERTEX_BIT)
                              .defineShaderStage(atmCubeMapFragShader_, VK_SHADER_STAGE_FRAGMENT_BIT)
                              .endShaderStage()
                              .defineVertexInputState<void>()
                              .defineViewportScissorState({
                                  .viewport =
                                      {
                                          .x = 0,
                                          .y = 0,
                                          .width = static_cast<float>(atmCubeMapImages_[0]->width()),
                                          .height = static_cast<float>(atmCubeMapImages_[0]->height()),
                                          .minDepth = 0.0,
                                          .maxDepth = 1.0,
                                      },
                                  .scissor =
                                      {
                                          .offset = {.x = 0, .y = 0},
                                          .extent =
                                              {
                                                  .width = atmCubeMapImages_[0]->width(),
                                                  .height = atmCubeMapImages_[0]->height(),
                                              },
                                      },
                              })
                              .beginColorBlendAttachmentState()
                              .defineDefaultColorBlendAttachmentState() // color
                              .endColorBlendAttachmentState()
                              .definePipelineLayout(atmDescriptorTables_[0])
                              .build(framework->device());
}

AtmosphereContext::AtmosphereContext(std::shared_ptr<FrameworkContext> frameworkContext,
                                     std::shared_ptr<Atmosphere> atmosphere)
    : frameworkContext(frameworkContext),
      atmosphere(atmosphere),
      atmDescriptorTable(atmosphere->atmDescriptorTables_[frameworkContext->frameIndex]),
      atmLUTImage(atmosphere->atmLUTImage_),
      atmLUTFramebuffer(atmosphere->atmLUTFramebuffer_),
      atmCubeMapImage(atmosphere->atmCubeMapImages_[frameworkContext->frameIndex]),
      atmCubeMapFramebuffer(atmosphere->atmCubeMapFramebuffers_[frameworkContext->frameIndex]) {}

void AtmosphereContext::render() {
    auto buffers = Renderer::instance().buffers();

    atmDescriptorTable->bindBuffer(buffers->worldUniformBuffer(), 1, 0);
    atmDescriptorTable->bindBuffer(buffers->skyUniformBuffer(), 1, 1);

    auto frameworkContextPtr = frameworkContext.lock();
    auto worldCommandBuffer = frameworkContextPtr->worldCommandBuffer;
    auto physicalDevice = frameworkContextPtr->physicalDevice;
    auto mainQueueIndex = physicalDevice->mainQueueIndex();

    auto module = atmosphere.lock();

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

    // render atmosphere transmit LUT only once
    if (!module->lutRendered_) {
        VkPipelineStageFlags2 lutSrcStage = 0;
        VkAccessFlags2 lutSrcAccess = 0;
        chooseSrc(atmLUTImage->imageLayout(),
                  VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                  VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                  lutSrcStage, lutSrcAccess);
        worldCommandBuffer->barriersBufferImage(
            {}, {{
                    .srcStageMask = lutSrcStage,
                    .srcAccessMask = lutSrcAccess,
                    .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                    .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                    .oldLayout = atmLUTImage->imageLayout(),
                    .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                    .srcQueueFamilyIndex = mainQueueIndex,
                    .dstQueueFamilyIndex = mainQueueIndex,
                    .image = atmLUTImage,
                    .subresourceRange = vk::wholeColorSubresourceRange,
                }});
        atmLUTImage->imageLayout() = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        worldCommandBuffer->beginRenderPass({
            .renderPass = module->atmLUTRenderPass_,
            .framebuffer = atmLUTFramebuffer,
            .renderAreaExtent = {atmLUTImage->width(), atmLUTImage->height()},
            .clearValues = {},
        });
        atmLUTImage->imageLayout() = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        worldCommandBuffer->bindGraphicsPipeline(module->atmLUTPipeline_)
            ->bindDescriptorTable(atmDescriptorTable, VK_PIPELINE_BIND_POINT_GRAPHICS)
            ->draw(3, 1)
            ->endRenderPass();
        atmLUTImage->imageLayout() = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        module->lutRendered_ = true;
    }

    // render atmosphere cube map
    {
        VkPipelineStageFlags2 cubeSrcStage = 0;
        VkAccessFlags2 cubeSrcAccess = 0;
        chooseSrc(atmCubeMapImage->imageLayout(),
                  VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                  VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                  cubeSrcStage, cubeSrcAccess);

        VkPipelineStageFlags2 lutSrcStage = 0;
        VkAccessFlags2 lutSrcAccess = 0;
        chooseSrc(module->atmLUTImage_->imageLayout(),
                  VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                  VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                  lutSrcStage, lutSrcAccess);
        worldCommandBuffer->barriersBufferImage(
            {}, {{
                     .srcStageMask = cubeSrcStage,
                     .srcAccessMask = cubeSrcAccess,
                     .dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                     .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                     .oldLayout = atmCubeMapImage->imageLayout(),
                     .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                     .srcQueueFamilyIndex = mainQueueIndex,
                     .dstQueueFamilyIndex = mainQueueIndex,
                     .image = atmCubeMapImage,
                     .subresourceRange = vk::wholeColorSubresourceRange,
                 },
                 {
                     .srcStageMask = lutSrcStage,
                     .srcAccessMask = lutSrcAccess,
                     .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                     .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                     .oldLayout = module->atmLUTImage_->imageLayout(),
                    .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    .srcQueueFamilyIndex = mainQueueIndex,
                     .dstQueueFamilyIndex = mainQueueIndex,
                     .image = module->atmLUTImage_,
                     .subresourceRange = vk::wholeColorSubresourceRange,
                 }});
        atmCubeMapImage->imageLayout() = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        module->atmLUTImage_->imageLayout() = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        for (int faceIndex = 0; faceIndex < 6; faceIndex++) {
            worldCommandBuffer->beginRenderPass({
                .renderPass = module->atmCubeMapRenderPass_,
                .framebuffer = atmCubeMapFramebuffer[faceIndex],
                .renderAreaExtent = {atmCubeMapImage->width(), atmCubeMapImage->height()},
                .clearValues = {},
            });
            atmCubeMapImage->imageLayout() = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

            worldCommandBuffer->bindGraphicsPipeline(module->atmCubeMapPipeline_)
                ->bindDescriptorTable(atmDescriptorTable, VK_PIPELINE_BIND_POINT_GRAPHICS);

            int pushConst = faceIndex;
            vkCmdPushConstants(worldCommandBuffer->vkCommandBuffer(), atmDescriptorTable->vkPipelineLayout(),
                               VK_SHADER_STAGE_ALL, 0, sizeof(int), &pushConst);

            worldCommandBuffer->draw(3, 1)->endRenderPass();
            atmCubeMapImage->imageLayout() = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        }
    }
}
