#pragma once

#include "core/all_extern.hpp"
#include "core/vulkan/vertex.hpp"

#include <iostream>
#include <string>
#include <vector>

namespace vk {
class Device;
class RenderPass;
class DescriptorTable;
class GraphicsPipelineBuilder;
class Shader;

class GraphicsPipeline : public SharedObject<GraphicsPipeline> {
    friend GraphicsPipelineBuilder;

  public:
    GraphicsPipeline(std::shared_ptr<Device> device, VkPipeline pipeline);
    ~GraphicsPipeline();

    VkPipeline &vkPipeline();

  private:
    std::shared_ptr<Device> device_;

    VkPipeline pipeline_ = VK_NULL_HANDLE;
};

class RayTracingPipeline : public SharedObject<RayTracingPipeline> {
  public:
    RayTracingPipeline(std::shared_ptr<Device> device, VkPipeline pipeline);
    ~RayTracingPipeline();

    VkPipeline &vkPipeline();

  private:
    std::shared_ptr<Device> device_;

    VkPipeline pipeline_ = VK_NULL_HANDLE;
};

class ComputePipeline : public SharedObject<ComputePipeline> {
    friend GraphicsPipelineBuilder;

  public:
    ComputePipeline(std::shared_ptr<Device> device, VkPipeline pipeline);
    ~ComputePipeline();

    VkPipeline &vkPipeline();

  private:
    std::shared_ptr<Device> device_;

    VkPipeline pipeline_ = VK_NULL_HANDLE;
};

class GraphicsPipelineBuilder {
  public:
    struct ShaderStageBuilder {
        GraphicsPipelineBuilder &parent;
        std::vector<VkPipelineShaderStageCreateInfo> shaderStageCreateInfos;

        ShaderStageBuilder(GraphicsPipelineBuilder &parent);

        ShaderStageBuilder &defineShaderStage(VkPipelineShaderStageCreateInfo shaderStageCreateInfo);
        ShaderStageBuilder &defineShaderStage(VkShaderModule shaderModule, VkShaderStageFlagBits shaderStage);
        ShaderStageBuilder &defineShaderStage(std::shared_ptr<Shader> shader, VkShaderStageFlagBits shaderStage);
        GraphicsPipelineBuilder &endShaderStage();
    };

    struct ColorBlendAttachmentStateBuilder {
        GraphicsPipelineBuilder &parent;
        std::vector<VkPipelineColorBlendAttachmentState> colorBlendAttachmentStates;

        ColorBlendAttachmentStateBuilder(GraphicsPipelineBuilder &parent);

        ColorBlendAttachmentStateBuilder &
        defineColorBlendAttachmentState(VkPipelineColorBlendAttachmentState colorBlendAttachmentState);
        ColorBlendAttachmentStateBuilder &defineDefaultColorBlendAttachmentState();

        GraphicsPipelineBuilder &endColorBlendAttachmentState();
    };

    struct ViewPortAndScissor {
        VkViewport viewport;
        VkRect2D scissor;
    };

    struct DepthStencilState {
        VkPipelineDepthStencilStateCreateFlags flags;
        VkBool32 depthTestEnable;
        VkBool32 depthWriteEnable;
        VkCompareOp depthCompareOp;
        VkBool32 depthBoundsTestEnable;
        VkBool32 stencilTestEnable;
        VkStencilOpState front;
        VkStencilOpState back;
        float minDepthBounds;
        float maxDepthBounds;
    };

  public:
    GraphicsPipelineBuilder();

    GraphicsPipelineBuilder &defineRenderPass(std::shared_ptr<RenderPass> renderPass, uint32_t subpassIndex);

    ShaderStageBuilder &beginShaderStage();

    GraphicsPipelineBuilder &defineVertexInputState(VkPipelineVertexInputStateCreateInfo vertexInputStateCreateInfo);
    template <typename T>
    GraphicsPipelineBuilder &defineVertexInputState();

    GraphicsPipelineBuilder &
    defineInputAssemblyState(VkPipelineInputAssemblyStateCreateInfo inputAssemblyStateCreateInfo);

    GraphicsPipelineBuilder &defineViewportState(VkPipelineViewportStateCreateInfo viewportStateCreateInfo);

    GraphicsPipelineBuilder &defineViewportScissorState(ViewPortAndScissor viewPortAndScissor);

    GraphicsPipelineBuilder &
    defineRasterizationState(VkPipelineRasterizationStateCreateInfo rasterizationStateCreateInfo);

    GraphicsPipelineBuilder &defineMultisampleState(VkPipelineMultisampleStateCreateInfo multisampleStateCreateInfo);

    ColorBlendAttachmentStateBuilder &beginColorBlendAttachmentState();

    GraphicsPipelineBuilder &defineColorBlendState(VkPipelineColorBlendStateCreateInfo colorBlendState);

    GraphicsPipelineBuilder &definePipelineLayout(std::shared_ptr<DescriptorTable> descriptorTable);
    GraphicsPipelineBuilder &definePipelineLayout(VkPipelineLayout pipelineLayout);

    // PipelineBuilder &defineDepthStencilState(VkPipelineDepthStencilStateCreateInfo depthStencilStateCreateInfo);
    GraphicsPipelineBuilder &defineDepthStencilState(DepthStencilState depthStencilState);

    std::shared_ptr<vk::GraphicsPipeline> build(std::shared_ptr<Device> device);

  private:
    ShaderStageBuilder shaderStageBuilder_;
    ColorBlendAttachmentStateBuilder colorBlendAttachmentStateBuilder_;

    VkPipelineVertexInputStateCreateInfo vertexInputStateCreateInfo_{};

    VkPipelineInputAssemblyStateCreateInfo inputAssemblyStateCreateInfo_ = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        .primitiveRestartEnable = VK_FALSE,
    };

    ViewPortAndScissor viewPortAndScissor_{};
    VkPipelineViewportStateCreateInfo viewportStateCreateInfo_{};

    // Default rasterization
    // Note: depth bias and using polygon modes other than fill require changes to logical device creation
    // (device features)
    VkPipelineRasterizationStateCreateInfo rasterizationStateCreateInfo_ = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .depthClampEnable = VK_FALSE,
        .rasterizerDiscardEnable = VK_FALSE,
        .polygonMode = VK_POLYGON_MODE_FILL,
        .cullMode = VK_CULL_MODE_BACK_BIT,
        .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
        .depthBiasEnable = VK_FALSE,
        .depthBiasConstantFactor = 0.0f,
        .depthBiasClamp = 0.0f,
        .depthBiasSlopeFactor = 0.0f,
        .lineWidth = 1.0f,
    };

    // Default multisampling
    // Note: using multisampling also requires turning on device features
    VkPipelineMultisampleStateCreateInfo multisampleStateCreateInfo_ = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
        .sampleShadingEnable = VK_FALSE,
        .minSampleShading = 1.0f,
        .alphaToCoverageEnable = VK_FALSE,
        .alphaToOneEnable = VK_FALSE,
    };

    // Default color blend state
    // attachments can be filled with ColorBlendAttachmentStateBuilder
    VkPipelineColorBlendStateCreateInfo colorBlendState_ = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .logicOpEnable = VK_FALSE,
        .logicOp = VK_LOGIC_OP_COPY,
        .attachmentCount = 0,
        .pAttachments = nullptr,
        .blendConstants = {0.0f},
    };

    // Default depth stencil state
    VkPipelineDepthStencilStateCreateInfo depthStencilState_ = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .depthTestEnable = VK_FALSE,
        .depthWriteEnable = VK_FALSE,
        .depthCompareOp = VK_COMPARE_OP_LESS,
        .depthBoundsTestEnable = VK_FALSE,
        .stencilTestEnable = VK_FALSE,
        .front = VkStencilOpState{},
        .back = VkStencilOpState{},
        .minDepthBounds = 0.0f,
        .maxDepthBounds = 1.0f,
    };

    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
    VkRenderPass renderPass_ = VK_NULL_HANDLE;
    uint32_t subpassIndex_ = 0;
};

class RayTracingPipelineBuilder {
  public:
    struct ShaderStageBuilder {
        RayTracingPipelineBuilder &parent;
        std::vector<VkPipelineShaderStageCreateInfo> shaderStageCreateInfos;

        ShaderStageBuilder(RayTracingPipelineBuilder &parent);

        ShaderStageBuilder &defineShaderStage(std::shared_ptr<Shader> shader, VkShaderStageFlagBits shaderStage);
        RayTracingPipelineBuilder &endShaderStage();
    };

    struct ShaderGroupBuilder {
        RayTracingPipelineBuilder &parent;
        std::vector<VkRayTracingShaderGroupCreateInfoKHR> shaderGroupCreateInfos;

        ShaderGroupBuilder(RayTracingPipelineBuilder &parent);

        ShaderGroupBuilder &defineShaderGroup(VkRayTracingShaderGroupTypeKHR type,
                                              uint32_t generalShader,
                                              uint32_t closestHitShader,
                                              uint32_t anyHitShader,
                                              uint32_t intersectionShader);
        RayTracingPipelineBuilder &endShaderGroup();
    };

  public:
    RayTracingPipelineBuilder();

    ShaderStageBuilder &beginShaderStage();
    ShaderGroupBuilder &beginShaderGroup();
    RayTracingPipelineBuilder &definePipelineLayout(std::shared_ptr<DescriptorTable> descriptorTable);
    std::shared_ptr<RayTracingPipeline> build(std::shared_ptr<Device> device);

  private:
    ShaderStageBuilder shaderStageBuilder_;
    ShaderGroupBuilder shaderGroupBuilder_;

    VkPipelineLayout pipelineLayout_;
};

template <typename T>
GraphicsPipelineBuilder &GraphicsPipelineBuilder::defineVertexInputState() {
    VertexLayoutInfo &vertexLayoutInfo = Vertex::vertexLayoutInfo<T>();

    vertexInputStateCreateInfo_.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputStateCreateInfo_.vertexBindingDescriptionCount = 1;
    vertexInputStateCreateInfo_.pVertexBindingDescriptions = &vertexLayoutInfo.bindingDescription;
    vertexInputStateCreateInfo_.vertexAttributeDescriptionCount = vertexLayoutInfo.attributeDescriptions.size();
    vertexInputStateCreateInfo_.pVertexAttributeDescriptions = vertexLayoutInfo.attributeDescriptions.data();

    return *this;
}

template <>
GraphicsPipelineBuilder &GraphicsPipelineBuilder::defineVertexInputState<void>();

class ComputePipelineBuilder {
  public:
    ComputePipelineBuilder &defineShader(std::shared_ptr<Shader> shader);
    ComputePipelineBuilder &definePipelineLayout(std::shared_ptr<DescriptorTable> descriptorTable);

    std::shared_ptr<ComputePipeline> build(std::shared_ptr<Device> device);

  private:
    VkShaderModule shaderModule_ = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
};
}; // namespace vk