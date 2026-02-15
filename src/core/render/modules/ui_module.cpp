#include "core/render/modules/ui_module.hpp"

#include "core/render/buffers.hpp"
#include "core/render/pipeline.hpp"
#include "core/render/render_framework.hpp"
#include "core/render/renderer.hpp"
#include "core/render/world.hpp"

UIModule::UIModule() {}

void UIModule::init(std::shared_ptr<Framework> framework) {
    framework_ = framework;

    initOverlayDescriptorTablesAndFrameSamplers();

    initOverlayDrawImages();
    initOverlayDrawRenderPass();
    initOverlayDrawFrameBuffers();
    initOverlayDrawPipelineTypes();
    initOverlayDrawPipelines();

    initOverlayPostImages();
    initOverlayPostRenderPass();
    initOverlayPostFrameBuffers();
    initOverlayPostPipelineTypes();
    initOverlayPostPipelines();

    uint32_t size = framework->swapchain()->imageCount();
    contexts_.resize(size);
    for (int i = 0; i < size; i++) {
        contexts_[i] = UIModuleContext::create(framework->contexts()[i], shared_from_this());
    }
}

std::vector<std::shared_ptr<UIModuleContext>> &UIModule::contexts() {
    return contexts_;
}

std::vector<std::shared_ptr<vk::DescriptorTable>> &UIModule::overlayDescriptorTables() {
    return overlayDescriptorTables_;
}

void UIModule::bindTexture(std::shared_ptr<vk::Sampler> sampler,
                           std::shared_ptr<vk::DeviceLocalImage> image,
                           int index) {
    auto framework = framework_.lock();

    uint32_t size = framework->swapchain()->imageCount();
    for (int i = 0; i < size; i++) {
        overlayDescriptorTables_[i]->bindSamplerImage(sampler, image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 0, 0,
                                                      index);
    }
}

void UIModule::initOverlayDescriptorTablesAndFrameSamplers() {
    auto framework = framework_.lock();

    uint32_t size = framework->swapchain()->imageCount();
    overlayDescriptorTables_.resize(size);
    overlayDrawColorImageSamplers_.resize(size);

    for (int i = 0; i < size; i++) {
        overlayDescriptorTables_[i] = vk::DescriptorTableBuilder{}
                                          .beginDescriptorLayoutSet() // set 0
                                          .beginDescriptorLayoutSetBinding()
                                          .defineDescriptorLayoutSetBinding({
                                              .binding = 0,
                                              .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                              .descriptorCount = 4096, // a very big number
                                              .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                          })
                                          .defineDescriptorLayoutSetBinding({
                                              .binding = 1,
                                              .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                              .descriptorCount = 1,
                                              .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                          })
                                          .endDescriptorLayoutSetBinding()
                                          .endDescriptorLayoutSet()
                                          .beginDescriptorLayoutSet() // set 1
                                          .beginDescriptorLayoutSetBinding()
                                          .defineDescriptorLayoutSetBinding({
                                              .binding = 0,
                                              .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                              .descriptorCount = 1,
                                              .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                          })
                                          .defineDescriptorLayoutSetBinding({
                                              .binding = 1,
                                              .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                              .descriptorCount = 1,
                                              .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                          })
                                          .endDescriptorLayoutSetBinding()
                                          .endDescriptorLayoutSet()
                                          .definePushConstant(VkPushConstantRange{
                                              .stageFlags = VK_SHADER_STAGE_ALL,
                                              .offset = 0,
                                              .size = sizeof(int),
                                          })
                                          .build(framework->device());

        overlayDrawColorImageSamplers_[i] = vk::Sampler::create(
            framework->device(), VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_LINEAR, VK_SAMPLER_ADDRESS_MODE_REPEAT);
    }
}

void UIModule::initOverlayDrawImages() {
    auto framework = framework_.lock();

    uint32_t size = framework->swapchain()->imageCount();
    overlayDrawColorImages_.resize(size);
    overlayDrawDepthStencilImages_.resize(size);

    for (int i = 0; i < size; i++) {
        overlayDrawColorImages_[i] = vk::DeviceLocalImage::create(
            framework->device(), framework->vma(), false, framework->swapchain()->vkExtent().width,
            framework->swapchain()->vkExtent().height, 1, VK_FORMAT_R8G8B8A8_SRGB,
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
                | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT
        );
        overlayDrawDepthStencilImages_[i] = vk::DeviceLocalImage::create(
            framework->device(), framework->vma(), false, framework->swapchain()->vkExtent().width,
            framework->swapchain()->vkExtent().height, 1, VK_FORMAT_D32_SFLOAT,
            VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);

        overlayDescriptorTables_[i]->bindSamplerImage(overlayDrawColorImageSamplers_[i], overlayDrawColorImages_[i],
                                                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 0, 1, 0);
    }
}

void UIModule::initOverlayDrawRenderPass() {
    auto framework = framework_.lock();

    overlayDrawRenderPass_ = vk::RenderPassBuilder{}
                                 .beginAttachmentDescription()
                                 .defineAttachmentDescription(VkAttachmentDescription{
                                     // color
                                     .format = overlayDrawColorImages_[0]->vkFormat(),
                                     .samples = VK_SAMPLE_COUNT_1_BIT,
                                     .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
                                     .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
                                     .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
                                     .stencilStoreOp = VK_ATTACHMENT_STORE_OP_STORE,
#ifdef USE_AMD
                                     .initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                     .finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
#else
                                     .initialLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                                     .finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
#endif
                                 })
                                 .defineAttachmentDescription(VkAttachmentDescription{
                                     // depth
                                     .format = overlayDrawDepthStencilImages_[0]->vkFormat(),
                                     .samples = VK_SAMPLE_COUNT_1_BIT,
                                     .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
                                     .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
                                     .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
                                     .stencilStoreOp = VK_ATTACHMENT_STORE_OP_STORE,
                                     .initialLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                                     .finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                                 })
                                 .endAttachmentDescription()
                                 .beginAttachmentReference()
                                 .defineAttachmentReference({
                                     .attachment = 0,
                                     .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                 })
                                 .defineAttachmentReference({
                                     .attachment = 1,
                                     .layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                                 })
                                 .endAttachmentReference()
                                 .beginSubpassDescription()
                                 .defineSubpassDescription({
                                     .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
                                     .colorAttachmentIndices = {0},
                                     .depthStencilAttachmentIndex = 1,
                                 })
                                 .endSubpassDescription()
                                 .build(framework->device());
}

void UIModule::initOverlayDrawFrameBuffers() {
    auto framework = framework_.lock();

    uint32_t size = framework->swapchain()->imageCount();
    overlayDrawFramebuffers_.resize(size);

    for (int i = 0; i < size; i++) {
        overlayDrawFramebuffers_[i] = vk::FramebufferBuilder{}
                                          .beginAttachment()
                                          .defineAttachment(overlayDrawColorImages_[i])
                                          .defineAttachment(overlayDrawDepthStencilImages_[i])
                                          .endAttachment()
                                          .build(framework->device(), overlayDrawRenderPass_);
    }
}

void UIModule::initOverlayDrawPipelineTypes() {
    std::filesystem::path shaderPath = Renderer::folderPath / "shaders";
    overlayDrawPipelineInfos_[POSITION_TEX] = {
        .vertexShaderFile = (shaderPath / "overlay/core/position_tex_glint_vert.spv").string(),
        .fragmentShaderFile = (shaderPath / "overlay/core/position_tex_glint_frag.spv").string(),
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
    };

    overlayDrawPipelineInfos_[POSITION_COLOR] = {
        .vertexShaderFile = (shaderPath / "overlay/core/position_color_vert.spv").string(),
        .fragmentShaderFile = (shaderPath / "overlay/core/position_color_frag.spv").string(),
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
    };

    overlayDrawPipelineInfos_[POSITION_TEX_COLOR] = {
        .vertexShaderFile = (shaderPath / "overlay/core/position_tex_color_vert.spv").string(),
        .fragmentShaderFile = (shaderPath / "overlay/core/position_tex_color_frag.spv").string(),
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
    };

    overlayDrawPipelineInfos_[POSITION_COLOR_TEX_LIGHT] = {
        .vertexShaderFile = (shaderPath / "overlay/core/position_color_tex_light_vert.spv").string(),
        .fragmentShaderFile = (shaderPath / "overlay/core/position_color_tex_light_frag.spv").string(),
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
    };

    overlayDrawPipelineInfos_[POSITION_COLOR_TEXTURE_OVERLAY_LIGHT_NORMAL] = {
        .vertexShaderFile =
            (shaderPath / "overlay/core/position_color_tex_overlay_light_normal_entity_cull_vert.spv").string(),
        .fragmentShaderFile =
            (shaderPath / "overlay/core/position_color_tex_overlay_light_normal_entity_cull_frag.spv").string(),
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
    };

    overlayDrawPipelineInfos_[POSITION_COLOR_TEXTURE_OVERLAY_LIGHT_NORMAL_NO_OUTLINE] = {
        .vertexShaderFile =
            (shaderPath / "overlay/core/position_color_tex_overlay_light_normal_entity_no_outline_vert.spv").string(),
        .fragmentShaderFile =
            (shaderPath / "overlay/core/position_color_tex_overlay_light_normal_entity_no_outline_frag.spv").string(),
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
    };

    overlayDrawPipelineInfos_[POSITION_END_PORTAL] = {
        .vertexShaderFile = (shaderPath / "overlay/core/position_end_portal_vert.spv").string(),
        .fragmentShaderFile = (shaderPath / "overlay/core/position_end_portal_frag.spv").string(),
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
    };

    overlayDrawPipelineInfos_[POSITION] = {
        .vertexShaderFile = (shaderPath / "overlay/core/position_vert.spv").string(),
        .fragmentShaderFile = (shaderPath / "overlay/core/position_frag.spv").string(),
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
    };
}

void UIModule::initOverlayDrawPipelines() {
    auto framework = framework_.lock();

    for (auto overlayPipelineInfo : overlayDrawPipelineInfos_) {
        auto [type, info] = overlayPipelineInfo;
        auto [vertexShaderFile, fragmentShaderFile, topology] = info;

        overlayDrawPipelineShaders_[type] = {
            .vertexShader = vk::Shader::create(framework->device(), vertexShaderFile),
            .fragmentShader = vk::Shader::create(framework->device(), fragmentShaderFile),
        };

        vk::DynamicGraphicsPipelineBuilder builder{1};
        builder.defineRenderPass(overlayDrawRenderPass_, 0)
            .beginShaderStage()
            .defineShaderStage(overlayDrawPipelineShaders_[type].vertexShader, VK_SHADER_STAGE_VERTEX_BIT)
            .defineShaderStage(overlayDrawPipelineShaders_[type].fragmentShader, VK_SHADER_STAGE_FRAGMENT_BIT)
            .endShaderStage();

        switch (type) {
            case POSITION_TEX: {
                builder.defineVertexInputState<vk::VertexFormat::PositionTex>();
                break;
            }
            case POSITION_TEX_COLOR: {
                builder.defineVertexInputState<vk::VertexFormat::PositionTexColor>();
                break;
            }
            case POSITION_COLOR: {
                builder.defineVertexInputState<vk::VertexFormat::PositionColor>();
                break;
            }
            case POSITION_COLOR_TEX_LIGHT: {
                builder.defineVertexInputState<vk::VertexFormat::PositionColorTexLight>();
                break;
            }
            case POSITION_COLOR_TEXTURE_OVERLAY_LIGHT_NORMAL_NO_OUTLINE:
            case POSITION_COLOR_TEXTURE_OVERLAY_LIGHT_NORMAL: {
                builder.defineVertexInputState<vk::VertexFormat::PositionColorTexOverlayLightNormal>();
                break;
            }
            case POSITION_END_PORTAL:
            case POSITION: {
                builder.defineVertexInputState<vk::VertexFormat::PositionOnly>();
                break;
            }

            default: throw std::runtime_error("A vertex formet should be defined!");
        }

        overlayDrawPipelines_[type] = builder.defineInputAssemblyState(topology)
                                          .definePipelineLayout(overlayDescriptorTables_[0])
                                          .build(framework->device());
    }
}

void UIModule::initOverlayPostImages() {
    auto framework = framework_.lock();

    uint32_t size = framework->swapchain()->imageCount();
    overlayPostColorImages_.resize(size);

    for (int i = 0; i < size; i++) {
        overlayPostColorImages_[i] = vk::DeviceLocalImage::create(
            framework->device(), framework->vma(), false, framework->swapchain()->vkExtent().width,
            framework->swapchain()->vkExtent().height, 1, VK_FORMAT_R8G8B8A8_SRGB,
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
#ifdef USE_AMD
                | VK_IMAGE_USAGE_TRANSFER_SRC_BIT
#endif
        );
    }
}

void UIModule::initOverlayPostRenderPass() {
    auto framework = framework_.lock();

    overlayPostRenderPass_ = vk::RenderPassBuilder{}
                                 .beginAttachmentDescription()
                                 .defineAttachmentDescription({
                                     // color
                                     .format = overlayPostColorImages_[0]->vkFormat(),
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
                                 .build(framework->device());
}

void UIModule::initOverlayPostFrameBuffers() {
    auto framework = framework_.lock();

    uint32_t size = framework->swapchain()->imageCount();
    overlayPostFramebuffers_.resize(size);

    for (int i = 0; i < size; i++) {
        overlayPostFramebuffers_[i] = vk::FramebufferBuilder{}
                                          .beginAttachment()
                                          .defineAttachment(overlayPostColorImages_[i])
                                          .endAttachment()
                                          .build(framework->device(), overlayPostRenderPass_);
    }
}

void UIModule::initOverlayPostPipelineTypes() {
    std::filesystem::path shaderPath = Renderer::folderPath / "shaders";
    overlayPostPipelineInfos_[BLUR] = {
        .vertexShaderFile = (shaderPath / "overlay/post/blur_vert.spv").string(),
        .fragmentShaderFile = (shaderPath / "overlay/post/blur_frag.spv").string(),
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
    };
}

void UIModule::initOverlayPostPipelines() {
    auto framework = framework_.lock();

    for (auto overlayPipelineInfo : overlayPostPipelineInfos_) {
        auto [type, info] = overlayPipelineInfo;
        auto [vertexShaderFile, fragmentShaderFile, topology] = info;

        overlayPostPipelineShaders_[type] = {
            .vertexShader = vk::Shader::create(framework->device(), vertexShaderFile),
            .fragmentShader = vk::Shader::create(framework->device(), fragmentShaderFile),
        };

        overlayPostPipelines_[type] =
            vk::GraphicsPipelineBuilder{}
                .defineRenderPass(overlayPostRenderPass_, 0)
                .beginShaderStage()
                .defineShaderStage(overlayPostPipelineShaders_[type].vertexShader, VK_SHADER_STAGE_VERTEX_BIT)
                .defineShaderStage(overlayPostPipelineShaders_[type].fragmentShader, VK_SHADER_STAGE_FRAGMENT_BIT)
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
                .definePipelineLayout(overlayDescriptorTables_[0])
                .build(framework->device());
    }
}

UIModuleContext::UIModuleContext(std::shared_ptr<FrameworkContext> context, std::shared_ptr<UIModule> uiModule)
    : frameworkContext(context),
      uiModule(uiModule),
      overlayDescriptorTable(uiModule->overlayDescriptorTables_[context->frameIndex]),
      overlayDrawColorImage(uiModule->overlayDrawColorImages_[context->frameIndex]),
      overlayDrawDepthStencilImage(uiModule->overlayDrawDepthStencilImages_[context->frameIndex]),
      overlayDrawFramebuffer(uiModule->overlayDrawFramebuffers_[context->frameIndex]),
      overlayPostColorImage(uiModule->overlayPostColorImages_[context->frameIndex]),
      overlayDrawColorImageSampler(uiModule->overlayDrawColorImageSamplers_[context->frameIndex]),
      overlayPostFramebuffer(uiModule->overlayPostFramebuffers_[context->frameIndex]) {
    overlayScissorEnabled = VK_FALSE;
    overlayScissor = {
        .offset = {0, 0},
        .extent = context->swapchain->vkExtent(),
    };
    overlayViewport = {
        .x = 0,
        .y = 0,
        .width = static_cast<float>(context->swapchain->vkExtent().width),
        .height = static_cast<float>(context->swapchain->vkExtent().height),
        .minDepth = 0.0,
        .maxDepth = 1.0,
    };

    overlayBlendEnabled = VK_FALSE;
    overlayColorBlendEquation = {
        .srcColorBlendFactor = VK_BLEND_FACTOR_ONE,
        .dstColorBlendFactor = VK_BLEND_FACTOR_ZERO,
        .colorBlendOp = VK_BLEND_OP_ADD,
        .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
        .dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
        .alphaBlendOp = VK_BLEND_OP_ADD,
    };
    overlayColorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    overlayColorLogicOpEnable = VK_FALSE;
    overlayColorLogicOp = VK_LOGIC_OP_COPY;
    overlayBlendConstants = {0.f, 0.f, 0.f, 0.f};

    overlayDepthTestEnable = VK_FALSE;
    overlayDepthWriteEnable = VK_TRUE;
    overlayDepthCompareOp = VK_COMPARE_OP_LESS;
    overlayStencilTestEnable = VK_FALSE;
    overlayFailOp = {VK_STENCIL_OP_KEEP, VK_STENCIL_OP_KEEP};
    overlayPassOp = {VK_STENCIL_OP_KEEP, VK_STENCIL_OP_KEEP};
    overlayDepthFailOp = {VK_STENCIL_OP_KEEP, VK_STENCIL_OP_KEEP};
    overlayCompareOp = {VK_COMPARE_OP_ALWAYS, VK_COMPARE_OP_ALWAYS};
    overlayReference = {0, 0};
    overlayCompareMask = {0xffffffff, 0xffffffff};
    overlayWriteMask = {0xffffffff, 0xffffffff};

    overlayCullMode = VK_CULL_MODE_NONE;
    overlayFrontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    overlayPolygonMode = VK_POLYGON_MODE_FILL;
    overlayDepthBiasEnable = VK_FALSE;
    overlayDepthBiasConstantFactor = {0.0, 0.0, 0.0};
    overlayDepthBiasClamp = {0.0, 0.0, 0.0};
    overlayDepthBiasSlopeFactor = {0.0, 0.0, 0.0};
    overlayLineWidth = 1.0;

    bool hdrActive = Renderer::options.hdrEnabled && context->swapchain->isHDR();

    // HDR: transparent black so composite shader's alpha test works correctly.
    // SDR: opaque white (original behavior — world is copied underneath).
    if (hdrActive) {
        overlayClearColors = {0.0f, 0.0f, 0.0f, 0.0f};
    } else {
        overlayClearColors = {1.0f, 1.0f, 1.0f, 1.0f};
    }
    overlayClearDepth = 1.0;
    overlayClearStencil = 0xffffffff;
}

void UIModuleContext::syncToCommandBuffer() {
    auto context = frameworkContext.lock();
    auto framework = context->framework.lock();

    if (!framework->isRunning()) return;

    auto commandBuffer = context->overlayCommandBuffer->vkCommandBuffer();

    // ------------ VkPipelineViewportStateCreateInfo ------------
    /* VK_DYNAMIC_STATE_VIEWPORT */
    vkCmdSetViewport(commandBuffer, 0, 1, &overlayViewport);

    /* VK_DYNAMIC_STATE_SCISSOR */
    if (overlayScissorEnabled)
        vkCmdSetScissor(commandBuffer, 0, 1, &overlayScissor);
    else {
        VkRect2D full_scissor = {
            .offset = {0, 0},
            .extent = framework->swapchain()->vkExtent(),
        };
        vkCmdSetScissor(commandBuffer, 0, 1, &full_scissor);
    }

    // ------------ VkPipelineDepthStencilStateCreateInfo ------------
    /* VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE */
    vkCmdSetDepthTestEnable(commandBuffer, overlayDepthTestEnable);

    /* VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE */
    vkCmdSetDepthWriteEnable(commandBuffer, overlayDepthWriteEnable);

    /* VK_DYNAMIC_STATE_DEPTH_COMPARE_OP */
    vkCmdSetDepthCompareOp(commandBuffer, overlayDepthCompareOp);

    /* VK_DYNAMIC_STATE_STENCIL_TEST_ENABLE */
    vkCmdSetStencilTestEnable(commandBuffer, overlayStencilTestEnable);

    /* VK_DYNAMIC_STATE_STENCIL_OP */
    vkCmdSetStencilOp(commandBuffer, VK_STENCIL_FACE_FRONT_BIT, overlayFailOp[0], overlayPassOp[0],
                      overlayDepthFailOp[0], overlayCompareOp[0]);
    vkCmdSetStencilOp(commandBuffer, VK_STENCIL_FACE_BACK_BIT, overlayFailOp[1], overlayPassOp[1],
                      overlayDepthFailOp[1], overlayCompareOp[1]);

    /* VK_DYNAMIC_STATE_STENCIL_REFERENCE */
    vkCmdSetStencilReference(commandBuffer, VK_STENCIL_FACE_FRONT_BIT, overlayReference[0]);
    vkCmdSetStencilReference(commandBuffer, VK_STENCIL_FACE_BACK_BIT, overlayReference[1]);

    /* VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK */
    vkCmdSetStencilCompareMask(commandBuffer, VK_STENCIL_FACE_FRONT_BIT, overlayCompareMask[0]);
    vkCmdSetStencilCompareMask(commandBuffer, VK_STENCIL_FACE_BACK_BIT, overlayCompareMask[1]);

    /* VK_DYNAMIC_STATE_STENCIL_WRITE_MASK */
    vkCmdSetStencilWriteMask(commandBuffer, VK_STENCIL_FACE_FRONT_BIT, overlayWriteMask[0]);
    vkCmdSetStencilWriteMask(commandBuffer, VK_STENCIL_FACE_BACK_BIT, overlayWriteMask[1]);

    // ------------ VkPipelineRasterizationStateCreateInfo ------------
    /* VK_DYNAMIC_STATE_CULL_MODE */
    vkCmdSetCullMode(commandBuffer, overlayCullMode);

    /* VK_DYNAMIC_STATE_FRONT_FACE */
    vkCmdSetFrontFace(commandBuffer, overlayFrontFace);

    /* VK_DYNAMIC_STATE_POLYGON_MODE_EXT */
    vkCmdSetPolygonModeEXT(commandBuffer, overlayPolygonMode);

    /* VK_DYNAMIC_STATE_DEPTH_BIAS_ENABLE */
    vkCmdSetDepthBiasEnable(commandBuffer, overlayDepthBiasEnable);

    /* VK_DYNAMIC_STATE_DEPTH_BIAS */
    vkCmdSetDepthBias(commandBuffer, overlayDepthBiasConstantFactor[overlayPolygonMode],
                      overlayDepthBiasClamp[overlayPolygonMode], overlayDepthBiasSlopeFactor[overlayPolygonMode]);

    /* VK_DYNAMIC_STATE_LINE_WIDTH */
    vkCmdSetLineWidth(commandBuffer, overlayLineWidth);

    // ------------ VkPipelineColorBlendAttachmentState / StateCreateInfo ------------
    /* VK_DYNAMIC_STATE_COLOR_BLEND_ENABLE_EXT */
    vkCmdSetColorBlendEnableEXT(commandBuffer, 0, 1, &overlayBlendEnabled);

    /* VK_DYNAMIC_STATE_COLOR_BLEND_EQUATION_EXT */
    vkCmdSetColorBlendEquationEXT(commandBuffer, 0, 1, &overlayColorBlendEquation);

    /* VK_DYNAMIC_STATE_COLOR_WRITE_MASK_EXT */
    vkCmdSetColorWriteMaskEXT(commandBuffer, 0, 1, &overlayColorWriteMask);

    // ------------ VkPipelineColorBlendStateCreateInfo ------------
    /* VK_DYNAMIC_STATE_LOGIC_OP_EXT and VK_DYNAMIC_STATE_LOGIC_OP_ENABLE_EXT */
    // Only call these if extendedDynamicState2LogicOp feature is enabled
    if (context->device->hasExtendedDynamicState2LogicOp()) {
        vkCmdSetLogicOpEXT(commandBuffer, overlayColorLogicOp);
        vkCmdSetLogicOpEnableEXT(commandBuffer, overlayColorLogicOpEnable);
    }

    /* VK_DYNAMIC_STATE_BLEND_CONSTANTS */
    vkCmdSetBlendConstants(commandBuffer, overlayBlendConstants.data());
}

void UIModuleContext::syncFromContext(std::shared_ptr<UIModuleContext> other) {
    auto context = frameworkContext.lock();
    auto framework = context->framework.lock();

    if (!framework->isRunning()) return;

    overlayScissorEnabled = other->overlayScissorEnabled;
    overlayScissor = other->overlayScissor;
    overlayViewport = other->overlayViewport;

    overlayBlendEnabled = other->overlayBlendEnabled;
    overlayColorBlendEquation = other->overlayColorBlendEquation;
    overlayColorWriteMask = other->overlayColorWriteMask;
    overlayColorLogicOpEnable = other->overlayColorLogicOpEnable;
    overlayColorLogicOp = other->overlayColorLogicOp;
    overlayBlendConstants = other->overlayBlendConstants;

    overlayDepthTestEnable = other->overlayDepthTestEnable;
    overlayDepthWriteEnable = other->overlayDepthWriteEnable;
    overlayDepthCompareOp = other->overlayDepthCompareOp;
    overlayStencilTestEnable = other->overlayStencilTestEnable;
    overlayFailOp = other->overlayFailOp;
    overlayPassOp = other->overlayPassOp;
    overlayDepthFailOp = other->overlayDepthFailOp;
    overlayCompareOp = other->overlayCompareOp;
    overlayReference = other->overlayReference;
    overlayCompareMask = other->overlayCompareMask;
    overlayWriteMask = other->overlayWriteMask;

    overlayCullMode = other->overlayCullMode;
    overlayFrontFace = other->overlayFrontFace;
    overlayPolygonMode = other->overlayPolygonMode;
    overlayDepthBiasEnable = other->overlayDepthBiasEnable;
    overlayDepthBiasConstantFactor = other->overlayDepthBiasConstantFactor;
    overlayDepthBiasClamp = other->overlayDepthBiasClamp;
    overlayDepthBiasSlopeFactor = other->overlayDepthBiasSlopeFactor;
    overlayLineWidth = other->overlayLineWidth;

    overlayClearColors = other->overlayClearColors;
    overlayClearDepth = other->overlayClearDepth;
    overlayClearStencil = other->overlayClearStencil;

    syncToCommandBuffer();
}

void UIModuleContext::setOverlayScissorEnabled(bool enabled) {
    auto context = frameworkContext.lock();
    auto framework = context->framework.lock();

    if (!framework->isRunning()) return;

    overlayScissorEnabled = enabled;
    if (overlayScissorEnabled) {
        vkCmdSetScissor(context->overlayCommandBuffer->vkCommandBuffer(), 0, 1, &overlayScissor);
    } else {
        VkRect2D full_scissor = {
            .offset = {0, 0},
            .extent = framework->swapchain()->vkExtent(),
        };
        vkCmdSetScissor(context->overlayCommandBuffer->vkCommandBuffer(), 0, 1, &full_scissor);
    }
}

void UIModuleContext::setOverlayScissor(int x, int y, int width, int height) {
    auto context = frameworkContext.lock();
    auto framework = context->framework.lock();

    if (!framework->isRunning()) return;

    // opengl scissor box with left bottom as origin
    // vulkan scissor box with left top as origin
    y = context->swapchainImage->height() - (y + height);

    x = std::max(x, 0);
    y = std::max(y, 0);
    if (x + width > context->swapchainImage->width()) width = context->swapchainImage->width() - x;
    if (y + height > context->swapchainImage->height()) height = context->swapchainImage->height() - y;

    overlayScissor.offset.x = x;
    overlayScissor.offset.y = y;
    overlayScissor.extent.width = width;
    overlayScissor.extent.height = height;
    setOverlayScissorEnabled(overlayScissorEnabled);
}

void UIModuleContext::setOverlayViewport(int x, int y, int width, int height) {
    auto context = frameworkContext.lock();
    auto framework = context->framework.lock();

    if (!framework->isRunning()) return;

    overlayViewport.x = x;
    overlayViewport.y = y;
    overlayViewport.width = width;
    overlayViewport.height = height;
    vkCmdSetViewport(context->overlayCommandBuffer->vkCommandBuffer(), 0, 1, &overlayViewport);
}

void UIModuleContext::setOverlayBlendEnable(bool enable) {
    auto context = frameworkContext.lock();
    auto framework = context->framework.lock();

    if (!framework->isRunning()) return;

    overlayBlendEnabled = enable;
    vkCmdSetColorBlendEnableEXT(context->overlayCommandBuffer->vkCommandBuffer(), 0, 1, &overlayBlendEnabled);
}

void UIModuleContext::setOverlayColorBlendConstants(float const1, float const2, float const3, float const4) {
    auto context = frameworkContext.lock();
    auto framework = context->framework.lock();

    if (!framework->isRunning()) return;

    overlayBlendConstants[0] = const1;
    overlayBlendConstants[1] = const2;
    overlayBlendConstants[2] = const3;
    overlayBlendConstants[3] = const4;
    vkCmdSetBlendConstants(context->overlayCommandBuffer->vkCommandBuffer(), overlayBlendConstants.data());
}

void UIModuleContext::setOverlayColorLogicOpEnable(bool enable) {
    auto context = frameworkContext.lock();
    auto framework = context->framework.lock();

    if (!framework->isRunning()) return;

    overlayColorLogicOpEnable = enable;
    if (context->device->hasExtendedDynamicState2LogicOp()) {
        vkCmdSetLogicOpEnableEXT(context->overlayCommandBuffer->vkCommandBuffer(), overlayColorLogicOpEnable);
    }
}

void UIModuleContext::setOverlayBlendFuncSeparate(int srcColorBlendFactor,
                                                  int srcAlphaBlendFactor,
                                                  int dstColorBlendFactor,
                                                  int dstAlphaBlendFactor) {
    auto context = frameworkContext.lock();
    auto framework = context->framework.lock();

    if (!framework->isRunning()) return;

    overlayColorBlendEquation.srcColorBlendFactor = static_cast<VkBlendFactor>(srcColorBlendFactor);
    overlayColorBlendEquation.srcAlphaBlendFactor = static_cast<VkBlendFactor>(srcAlphaBlendFactor);
    overlayColorBlendEquation.dstColorBlendFactor = static_cast<VkBlendFactor>(dstColorBlendFactor);
    overlayColorBlendEquation.dstAlphaBlendFactor = static_cast<VkBlendFactor>(dstAlphaBlendFactor);
    vkCmdSetColorBlendEquationEXT(context->overlayCommandBuffer->vkCommandBuffer(), 0, 1, &overlayColorBlendEquation);
}

void UIModuleContext::setOverlayBlendOpSeparate(int colorBlendOp, int alphaBlendOp) {
    auto context = frameworkContext.lock();
    auto framework = context->framework.lock();

    if (!framework->isRunning()) return;

    overlayColorBlendEquation.colorBlendOp = static_cast<VkBlendOp>(colorBlendOp);
    overlayColorBlendEquation.alphaBlendOp = static_cast<VkBlendOp>(alphaBlendOp);
    vkCmdSetColorBlendEquationEXT(context->overlayCommandBuffer->vkCommandBuffer(), 0, 1, &overlayColorBlendEquation);
}

void UIModuleContext::setOverlayColorWriteMask(int colorWriteMask) {
    auto context = frameworkContext.lock();
    auto framework = context->framework.lock();

    if (!framework->isRunning()) return;

    // Keep alpha writes enabled in SDR as well.
    // The final composite pass samples overlay alpha to blend UI over the world.
    overlayColorWriteMask = colorWriteMask;
    vkCmdSetColorWriteMaskEXT(context->overlayCommandBuffer->vkCommandBuffer(), 0, 1, &overlayColorWriteMask);
}

void UIModuleContext::setOverlayColorLogicOp(int colorLogicOp) {
    auto context = frameworkContext.lock();
    auto framework = context->framework.lock();

    if (!framework->isRunning()) return;

    overlayColorLogicOp = static_cast<VkLogicOp>(colorLogicOp);
    if (context->device->hasExtendedDynamicState2LogicOp()) {
        vkCmdSetLogicOpEXT(context->overlayCommandBuffer->vkCommandBuffer(), overlayColorLogicOp);
    }
}

void UIModuleContext::setOverlayDepthTestEnable(bool enable) {
    auto context = frameworkContext.lock();
    auto framework = context->framework.lock();

    if (!framework->isRunning()) return;

    overlayDepthTestEnable = enable;
    vkCmdSetDepthTestEnable(context->overlayCommandBuffer->vkCommandBuffer(), overlayDepthTestEnable);
}

void UIModuleContext::setOverlayDepthWriteEnable(bool enable) {
    auto context = frameworkContext.lock();
    auto framework = context->framework.lock();

    if (!framework->isRunning()) return;

    overlayDepthWriteEnable = enable;
    vkCmdSetDepthWriteEnable(context->overlayCommandBuffer->vkCommandBuffer(), overlayDepthWriteEnable);
}

void UIModuleContext::setOverlayStencilTestEnable(bool enable) {
    auto context = frameworkContext.lock();
    auto framework = context->framework.lock();

    if (!framework->isRunning()) return;

    overlayStencilTestEnable = enable;
    vkCmdSetStencilTestEnable(context->overlayCommandBuffer->vkCommandBuffer(), overlayStencilTestEnable);
}

void UIModuleContext::setOverlayDepthCompareOp(int depthCompareOp) {
    auto context = frameworkContext.lock();
    auto framework = context->framework.lock();

    if (!framework->isRunning()) return;

    overlayDepthCompareOp = static_cast<VkCompareOp>(depthCompareOp);
    vkCmdSetDepthCompareOp(context->overlayCommandBuffer->vkCommandBuffer(), overlayDepthCompareOp);
}

void UIModuleContext::setOverlayStencilFrontFunc(int compareOp, int reference, int compareMask) {
    auto context = frameworkContext.lock();
    auto framework = context->framework.lock();

    if (!framework->isRunning()) return;

    overlayCompareOp[0] = static_cast<VkCompareOp>(compareOp);
    overlayReference[0] = reference;
    overlayCompareMask[0] = compareMask;
    vkCmdSetStencilOp(context->overlayCommandBuffer->vkCommandBuffer(), VK_STENCIL_FACE_FRONT_BIT, overlayFailOp[0],
                      overlayPassOp[0], overlayDepthFailOp[0], overlayCompareOp[0]);
    vkCmdSetStencilReference(context->overlayCommandBuffer->vkCommandBuffer(), VK_STENCIL_FACE_FRONT_BIT,
                             overlayReference[0]);
    vkCmdSetStencilCompareMask(context->overlayCommandBuffer->vkCommandBuffer(), VK_STENCIL_FACE_FRONT_BIT,
                               overlayCompareMask[0]);
}

void UIModuleContext::setOverlayStencilBackFunc(int compareOp, int reference, int compareMask) {
    auto context = frameworkContext.lock();
    auto framework = context->framework.lock();

    if (!framework->isRunning()) return;

    overlayCompareOp[1] = static_cast<VkCompareOp>(compareOp);
    overlayReference[1] = reference;
    overlayCompareMask[1] = compareMask;
    vkCmdSetStencilOp(context->overlayCommandBuffer->vkCommandBuffer(), VK_STENCIL_FACE_BACK_BIT, overlayFailOp[1],
                      overlayPassOp[1], overlayDepthFailOp[1], overlayCompareOp[1]);
    vkCmdSetStencilReference(context->overlayCommandBuffer->vkCommandBuffer(), VK_STENCIL_FACE_BACK_BIT,
                             overlayReference[1]);
    vkCmdSetStencilCompareMask(context->overlayCommandBuffer->vkCommandBuffer(), VK_STENCIL_FACE_BACK_BIT,
                               overlayCompareMask[1]);
}

void UIModuleContext::setOverlayStencilFrontOp(int failOp, int depthFailOp, int passOp) {
    auto context = frameworkContext.lock();
    auto framework = context->framework.lock();

    if (!framework->isRunning()) return;

    overlayFailOp[0] = static_cast<VkStencilOp>(failOp);
    overlayDepthFailOp[0] = static_cast<VkStencilOp>(depthFailOp);
    overlayPassOp[0] = static_cast<VkStencilOp>(passOp);
    vkCmdSetStencilOp(context->overlayCommandBuffer->vkCommandBuffer(), VK_STENCIL_FACE_FRONT_BIT, overlayFailOp[0],
                      overlayPassOp[0], overlayDepthFailOp[0], overlayCompareOp[0]);
}

void UIModuleContext::setOverlayStencilBackOp(int failOp, int depthFailOp, int passOp) {
    auto context = frameworkContext.lock();
    auto framework = context->framework.lock();

    if (!framework->isRunning()) return;

    overlayFailOp[1] = static_cast<VkStencilOp>(failOp);
    overlayDepthFailOp[1] = static_cast<VkStencilOp>(depthFailOp);
    overlayPassOp[1] = static_cast<VkStencilOp>(passOp);
    vkCmdSetStencilOp(context->overlayCommandBuffer->vkCommandBuffer(), VK_STENCIL_FACE_BACK_BIT, overlayFailOp[1],
                      overlayPassOp[1], overlayDepthFailOp[1], overlayCompareOp[1]);
}

void UIModuleContext::setOverlayStencilFrontWriteMask(int writeMask) {
    auto context = frameworkContext.lock();
    auto framework = context->framework.lock();

    if (!framework->isRunning()) return;

    overlayWriteMask[0] = writeMask;
    vkCmdSetStencilWriteMask(context->overlayCommandBuffer->vkCommandBuffer(), VK_STENCIL_FACE_FRONT_BIT,
                             overlayWriteMask[0]);
}

void UIModuleContext::setOverlayStencilBackWriteMask(int writeMask) {
    auto context = frameworkContext.lock();
    auto framework = context->framework.lock();

    if (!framework->isRunning()) return;

    overlayWriteMask[1] = writeMask;
    vkCmdSetStencilWriteMask(context->overlayCommandBuffer->vkCommandBuffer(), VK_STENCIL_FACE_BACK_BIT,
                             overlayWriteMask[1]);
}

void UIModuleContext::setOverlayLineWidth(float lineWidth) {
    auto context = frameworkContext.lock();
    auto framework = context->framework.lock();

    if (!framework->isRunning()) return;

    overlayLineWidth = lineWidth;
    vkCmdSetLineWidth(context->overlayCommandBuffer->vkCommandBuffer(), overlayLineWidth);
}

void UIModuleContext::setOverlayPolygonMode(int polygonMode) {
    auto context = frameworkContext.lock();
    auto framework = context->framework.lock();

    if (!framework->isRunning()) return;

    overlayPolygonMode = static_cast<VkPolygonMode>(polygonMode);
    vkCmdSetPolygonModeEXT(context->overlayCommandBuffer->vkCommandBuffer(), overlayPolygonMode);
    setOverlayDepthBiasEnable(overlayPolygonMode, overlayDepthBiasEnable);
}

void UIModuleContext::setOverlayCullMode(int cullMode) {
    auto context = frameworkContext.lock();
    auto framework = context->framework.lock();

    if (!framework->isRunning()) return;

    overlayCullMode = cullMode;
    vkCmdSetCullMode(context->overlayCommandBuffer->vkCommandBuffer(), overlayCullMode);
}

void UIModuleContext::setOverlayFrontFace(int frontFace) {
    auto context = frameworkContext.lock();
    auto framework = context->framework.lock();

    if (!framework->isRunning()) return;

    overlayFrontFace = static_cast<VkFrontFace>(frontFace);
    vkCmdSetFrontFace(context->overlayCommandBuffer->vkCommandBuffer(), overlayFrontFace);
}

void UIModuleContext::setOverlayDepthBiasEnable(int polygonMode, bool enable) {
    auto context = frameworkContext.lock();
    auto framework = context->framework.lock();

    if (!framework->isRunning()) return;

    overlayDepthBiasEnable = enable;
    if (overlayDepthBiasEnable)
        vkCmdSetDepthBias(context->overlayCommandBuffer->vkCommandBuffer(),
                          overlayDepthBiasConstantFactor[overlayPolygonMode], overlayDepthBiasClamp[overlayPolygonMode],
                          overlayDepthBiasSlopeFactor[overlayPolygonMode]);
    vkCmdSetDepthBiasEnable(context->overlayCommandBuffer->vkCommandBuffer(), overlayDepthBiasEnable);
}

void UIModuleContext::setOverlayDepthBias(float depthBiasSlopeFactor, float depthBiasConstantFactor) {
    auto context = frameworkContext.lock();
    auto framework = context->framework.lock();

    if (!framework->isRunning()) return;

    overlayDepthBiasSlopeFactor[overlayPolygonMode] = depthBiasSlopeFactor;
    overlayDepthBiasConstantFactor[overlayPolygonMode] = depthBiasConstantFactor;
    setOverlayDepthBiasEnable(overlayPolygonMode, overlayDepthBiasEnable);
}

void UIModuleContext::setOverlayClearColor(float red, float green, float blue, float alpha) {
    auto context = frameworkContext.lock();
    auto framework = context->framework.lock();

    if (!framework->isRunning()) return;

    bool hdrActive = Renderer::options.hdrEnabled && framework->swapchain()->isHDR();

    // In HDR mode, force fully transparent overlay to prevent tinted compositing.
    if (hdrActive) {
        overlayClearColors[0] = 0.0f;
        overlayClearColors[1] = 0.0f;
        overlayClearColors[2] = 0.0f;
        overlayClearColors[3] = 0.0f;
        return;
    }

    overlayClearColors[0] = red;
    overlayClearColors[1] = green;
    overlayClearColors[2] = blue;
    overlayClearColors[3] = 1.0f;
}

void UIModuleContext::setOverlayClearDepth(double depth) {
    auto context = frameworkContext.lock();
    auto framework = context->framework.lock();

    if (!framework->isRunning()) return;

    overlayClearDepth = depth;
}

void UIModuleContext::setOverlayClearStencil(int stencil) {
    auto context = frameworkContext.lock();
    auto framework = context->framework.lock();

    if (!framework->isRunning()) return;

    overlayClearStencil = stencil;
}

void UIModuleContext::switchOverlayDraw() {
    auto context = frameworkContext.lock();
    auto framework = context->framework.lock();
    auto module = uiModule.lock();

    if (!framework->isRunning()) return;

    auto mainQueueIndex = context->physicalDevice->mainQueueIndex();

    if (overlayMode == POST) {
        context->overlayCommandBuffer->endRenderPass();
#ifdef USE_AMD
        overlayPostColorImage->imageLayout() = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
#else
        overlayPostColorImage->imageLayout() = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
#endif
    }

    if (overlayMode == NONE || overlayMode == POST) {
        context->overlayCommandBuffer->barriersBufferImage(
            {}, {{
                     .srcStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                     .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                     .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                     .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                     .oldLayout = overlayDrawColorImage->imageLayout(),
#ifdef USE_AMD
                     .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
#else
                     .newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
#endif
                     .srcQueueFamilyIndex = mainQueueIndex,
                     .dstQueueFamilyIndex = mainQueueIndex,
                     .image = overlayDrawColorImage,
                     .subresourceRange = vk::wholeColorSubresourceRange,
                 },
                 {
                     .srcStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT |
                                     VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                     .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                     .dstStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                     .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                     .oldLayout = overlayDrawDepthStencilImage->imageLayout(),
                     .newLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL,
                     .srcQueueFamilyIndex = mainQueueIndex,
                     .dstQueueFamilyIndex = mainQueueIndex,
                     .image = overlayDrawDepthStencilImage,
                     .subresourceRange = vk::wholeDepthSubresourceRange,
                 }});
#ifdef USE_AMD
        overlayDrawColorImage->imageLayout() = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
#else
        overlayDrawColorImage->imageLayout() = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
#endif
        overlayDrawDepthStencilImage->imageLayout() = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL;

        context->overlayCommandBuffer->beginRenderPass({
            .renderPass = module->overlayDrawRenderPass_,
            .framebuffer = overlayDrawFramebuffer,
            .renderAreaExtent = {overlayDrawColorImage->width(), overlayDrawColorImage->height()},
            .clearValues = {{.color = {0.1f, 0.1f, 0.1f, 1.0f}}, {.depthStencil = {.depth = 1.0f}}},
        });

        overlayDrawColorImage->imageLayout() = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        overlayDrawDepthStencilImage->imageLayout() = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        syncToCommandBuffer();
    }

    overlayMode = DRAW;
}

void UIModuleContext::switchOverlayPost() {
    auto context = frameworkContext.lock();
    auto framework = context->framework.lock();
    auto module = uiModule.lock();

    if (!framework->isRunning()) return;

    auto mainQueueIndex = context->physicalDevice->mainQueueIndex();

    if (overlayMode == DRAW) {
        context->overlayCommandBuffer->endRenderPass();
#ifdef USE_AMD
        overlayDrawColorImage->imageLayout() = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

#else
        overlayDrawColorImage->imageLayout() = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
#endif
        overlayDrawDepthStencilImage->imageLayout() = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    }

    if (overlayMode == NONE || overlayMode == DRAW) {
        context->overlayCommandBuffer->barriersBufferImage(
            {}, {{
                     .srcStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                     .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                     .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                     .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                     .oldLayout = overlayPostColorImage->imageLayout(),
#ifdef USE_AMD
                     .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
#else
                     .newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
#endif
                     .srcQueueFamilyIndex = mainQueueIndex,
                     .dstQueueFamilyIndex = mainQueueIndex,
                     .image = overlayPostColorImage,
                     .subresourceRange = vk::wholeColorSubresourceRange,
                 },
                 {
                     .srcStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                     .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                     .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                     .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                     .oldLayout = overlayDrawColorImage->imageLayout(),
                     .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     .srcQueueFamilyIndex = mainQueueIndex,
                     .dstQueueFamilyIndex = mainQueueIndex,
                     .image = overlayDrawColorImage,
                     .subresourceRange = vk::wholeColorSubresourceRange,
                 }});
#ifdef USE_AMD
        overlayPostColorImage->imageLayout() = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
#else
        overlayPostColorImage->imageLayout() = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
#endif
        overlayDrawColorImage->imageLayout() = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        context->overlayCommandBuffer->beginRenderPass({
            .renderPass = module->overlayPostRenderPass_,
            .framebuffer = overlayPostFramebuffer,
            .renderAreaExtent = {overlayPostColorImage->width(), overlayPostColorImage->height()},
            .clearValues = {{.color = {0.1f, 0.1f, 0.1f, 1.0f}}, {.depthStencil = {.depth = 1.0f}}},
        });
        overlayPostColorImage->imageLayout() = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    }

    overlayMode = POST;
}

void UIModuleContext::clearOverlayEntireColorAttachment() {
    auto context = frameworkContext.lock();
    auto framework = context->framework.lock();

    if (!framework->isRunning()) return;

    switchOverlayDraw();

    VkClearAttachment clearAttachment{};
    clearAttachment.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    clearAttachment.colorAttachment = 0;
    // Always clear to transparent black.
    // World (SDR/HDR) is composited later in FrameworkContext::fuseFinal().
    clearAttachment.clearValue.color.float32[0] = 0.0f;
    clearAttachment.clearValue.color.float32[1] = 0.0f;
    clearAttachment.clearValue.color.float32[2] = 0.0f;
    clearAttachment.clearValue.color.float32[3] = 0.0f;

    VkClearRect clearRect{};
    clearRect.rect.offset = {0, 0};
    clearRect.rect.extent = framework->swapchain()->vkExtent();
    clearRect.baseArrayLayer = 0;
    clearRect.layerCount = 1;

    vkCmdClearAttachments(context->overlayCommandBuffer->vkCommandBuffer(), 1, &clearAttachment, 1, &clearRect);
}

void UIModuleContext::clearOverlayEntireDepthStencilAttachment(int aspectMask) {
    auto context = frameworkContext.lock();
    auto framework = context->framework.lock();

    if (!framework->isRunning()) return;

    switchOverlayDraw();

    VkClearAttachment clearAttachment{};
    clearAttachment.aspectMask = aspectMask;
    clearAttachment.clearValue.depthStencil.depth = overlayClearDepth;
    clearAttachment.clearValue.depthStencil.stencil = overlayClearStencil;

    VkClearRect clearRect{};
    clearRect.rect.offset = {0, 0};
    clearRect.rect.extent = framework->swapchain()->vkExtent();
    clearRect.baseArrayLayer = 0;
    clearRect.layerCount = 1;

    vkCmdClearAttachments(context->overlayCommandBuffer->vkCommandBuffer(), 1, &clearAttachment, 1, &clearRect);
}

void UIModuleContext::drawIndexed(std::shared_ptr<vk::DeviceLocalBuffer> vertexBuffer,
                                  std::shared_ptr<vk::DeviceLocalBuffer> indexBuffer,
                                  OverlayDrawPipelineType pipelineType,
                                  uint32_t indexCount,
                                  VkIndexType indexType) {
    auto context = frameworkContext.lock();
    auto framework = context->framework.lock();
    auto module = uiModule.lock();

    if (!framework->isRunning()) return;

    switchOverlayDraw();

    vkCmdBindPipeline(context->overlayCommandBuffer->vkCommandBuffer(), VK_PIPELINE_BIND_POINT_GRAPHICS,
                      module->overlayDrawPipelines_[pipelineType]->vkPipeline());

    auto pipelineLayout = overlayDescriptorTable->vkPipelineLayout();
    int drawID = Renderer::instance().buffers()->getDrawID();
    vkCmdPushConstants(context->overlayCommandBuffer->vkCommandBuffer(), pipelineLayout, VK_SHADER_STAGE_ALL, 0,
                       sizeof(int), &drawID);

    context->overlayCommandBuffer->bindVertexBuffers(vertexBuffer)
        ->bindIndexBuffer(indexBuffer, indexType)
        ->drawIndexed(indexCount, 1);
}

void UIModuleContext::postBlur(int times) {
    auto context = frameworkContext.lock();
    auto framework = context->framework.lock();
    auto module = uiModule.lock();

    if (!framework->isRunning()) return;

    int postID = Renderer::instance().buffers()->getPostID();

    for (int i = postID - times + 1; i <= postID; i++) {
        switchOverlayPost();

        vkCmdBindPipeline(context->overlayCommandBuffer->vkCommandBuffer(), VK_PIPELINE_BIND_POINT_GRAPHICS,
                          module->overlayPostPipelines_[BLUR]->vkPipeline());

        auto pipelineLayout = overlayDescriptorTable->vkPipelineLayout();

        vkCmdPushConstants(context->overlayCommandBuffer->vkCommandBuffer(), pipelineLayout, VK_SHADER_STAGE_ALL, 0,
                           sizeof(int), &i);

        context->overlayCommandBuffer->draw(3, 1);

        context->overlayCommandBuffer->endRenderPass();
#ifdef USE_AMD
        overlayPostColorImage->imageLayout() = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
#else
        overlayPostColorImage->imageLayout() = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
#endif
        overlayMode = NONE;

        auto mainQueueIndex = context->physicalDevice->mainQueueIndex();
        context->overlayCommandBuffer->barriersBufferImage(
            {}, {
                    {
                        .srcStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                        .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                        .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                        .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                        .oldLayout = overlayPostColorImage->imageLayout(),
                        .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                        .srcQueueFamilyIndex = mainQueueIndex,
                        .dstQueueFamilyIndex = mainQueueIndex,
                        .image = overlayPostColorImage,
                        .subresourceRange = vk::wholeColorSubresourceRange,
                    },
                    {
                        .srcStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                        .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                        .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                        .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                        .oldLayout = overlayDrawColorImage->imageLayout(),
                        .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                        .srcQueueFamilyIndex = mainQueueIndex,
                        .dstQueueFamilyIndex = mainQueueIndex,
                        .image = overlayDrawColorImage,
                        .subresourceRange = vk::wholeColorSubresourceRange,
                    },
                });

        overlayPostColorImage->imageLayout() = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        overlayDrawColorImage->imageLayout() = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;

        // TODO: add to command buffer
        VkImageBlit imageBlit{};
        imageBlit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        imageBlit.srcSubresource.mipLevel = 0;
        imageBlit.srcSubresource.baseArrayLayer = 0;
        imageBlit.srcSubresource.layerCount = 1;
        imageBlit.srcOffsets[0] = {0, 0, 0};
        imageBlit.srcOffsets[1] = {static_cast<int>(overlayPostColorImage->width()),
                                   static_cast<int>(overlayPostColorImage->height()), 1};
        imageBlit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        imageBlit.dstSubresource.mipLevel = 0;
        imageBlit.dstSubresource.baseArrayLayer = 0;
        imageBlit.dstSubresource.layerCount = 1;
        imageBlit.dstOffsets[0] = {0, 0, 0};
        imageBlit.dstOffsets[1] = {static_cast<int>(overlayDrawColorImage->width()),
                                   static_cast<int>(overlayDrawColorImage->height()), 1};

        vkCmdBlitImage(context->overlayCommandBuffer->vkCommandBuffer(), overlayPostColorImage->vkImage(),
                       overlayPostColorImage->imageLayout(), overlayDrawColorImage->vkImage(),
                       overlayDrawColorImage->imageLayout(), 1, &imageBlit, VK_FILTER_LINEAR);

        context->overlayCommandBuffer->barriersBufferImage(
            {}, {{
                     .srcStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                     .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                     .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                     .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                     .oldLayout = overlayPostColorImage->imageLayout(),
#ifdef USE_AMD
                     .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
#else
                     .newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
#endif
                     .srcQueueFamilyIndex = mainQueueIndex,
                     .dstQueueFamilyIndex = mainQueueIndex,
                     .image = overlayPostColorImage,
                     .subresourceRange = vk::wholeColorSubresourceRange,
                 },
                 {
                     .srcStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                     .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                     .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                     .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                     .oldLayout = overlayDrawColorImage->imageLayout(),
                     .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     .srcQueueFamilyIndex = mainQueueIndex,
                     .dstQueueFamilyIndex = mainQueueIndex,
                     .image = overlayDrawColorImage,
                     .subresourceRange = vk::wholeColorSubresourceRange,
                 }});
#ifdef USE_AMD
        overlayPostColorImage->imageLayout() = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
#else
        overlayPostColorImage->imageLayout() = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
#endif
        overlayDrawColorImage->imageLayout() = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }
}

void UIModuleContext::begin(std::shared_ptr<UIModuleContext> lastContext) {
    auto context = frameworkContext.lock();
    auto framework = context->framework.lock();

    if (!framework->isRunning()) return;

    overlayMode = NONE;

    context->overlayCommandBuffer->bindDescriptorTable(overlayDescriptorTable, VK_PIPELINE_BIND_POINT_GRAPHICS);

    if (lastContext != nullptr)
        syncFromContext(lastContext);
    else
        syncToCommandBuffer();
}

void UIModuleContext::end() {
    auto context = frameworkContext.lock();
    auto framework = context->framework.lock();

    if (!framework->isRunning()) return;

    if (overlayMode == DRAW) {
        context->overlayCommandBuffer->endRenderPass();
#ifdef USE_AMD
        overlayDrawColorImage->imageLayout() = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
#else
        overlayDrawColorImage->imageLayout() = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
#endif
    } else if (overlayMode == POST) {
        context->overlayCommandBuffer->endRenderPass();

        auto mainQueueIndex = context->physicalDevice->mainQueueIndex();
        context->overlayCommandBuffer->barriersBufferImage(
            {}, {{
                    .srcStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                    .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                    .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                    .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                    .oldLayout = overlayDrawColorImage->imageLayout(),
#ifdef USE_AMD
                    .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
#else
                    .newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
#endif
                    .srcQueueFamilyIndex = mainQueueIndex,
                    .dstQueueFamilyIndex = mainQueueIndex,
                    .image = overlayDrawColorImage,
                    .subresourceRange = vk::wholeColorSubresourceRange,
                }});
#ifdef USE_AMD
        overlayDrawColorImage->imageLayout() = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
#else
        overlayDrawColorImage->imageLayout() = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
#endif
    }

    overlayMode = NONE;
}
