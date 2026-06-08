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

class DynamicGraphicsPipelineBuilder;

class DynamicGraphicsPipeline : public SharedObject<DynamicGraphicsPipeline> {
    friend DynamicGraphicsPipelineBuilder;

  public:
    DynamicGraphicsPipeline(std::shared_ptr<Device> device, VkPipeline pipeline);
    ~DynamicGraphicsPipeline();

    VkPipeline vkPipeline();

  private:
  std::shared_ptr<Device> device_;

    VkPipeline pipeline_ = VK_NULL_HANDLE;
};

class DynamicGraphicsPipelineBuilder {
  public:
    struct ShaderStageBuilder {
        DynamicGraphicsPipelineBuilder &parent;
        std::vector<VkPipelineShaderStageCreateInfo> shaderStageCreateInfos;

        ShaderStageBuilder(DynamicGraphicsPipelineBuilder &parent);

        ShaderStageBuilder &defineShaderStage(VkPipelineShaderStageCreateInfo shaderStageCreateInfo);
        ShaderStageBuilder &defineShaderStage(VkShaderModule shaderModule, VkShaderStageFlagBits shaderStage);
        ShaderStageBuilder &defineShaderStage(std::shared_ptr<Shader> shader, VkShaderStageFlagBits shaderStage);
        DynamicGraphicsPipelineBuilder &endShaderStage();
    };

  public:
    DynamicGraphicsPipelineBuilder(int numColorAttachments);

    DynamicGraphicsPipelineBuilder &defineRenderPass(std::shared_ptr<RenderPass> renderPass, uint32_t subpassIndex);

    ShaderStageBuilder &beginShaderStage();

    template <typename T>
    DynamicGraphicsPipelineBuilder &defineVertexInputState();

    DynamicGraphicsPipelineBuilder &definePipelineLayout(std::shared_ptr<DescriptorTable> descriptorTable);

    DynamicGraphicsPipelineBuilder &defineInputAssemblyState(VkPrimitiveTopology topology);

    std::shared_ptr<vk::DynamicGraphicsPipeline> build(std::shared_ptr<Device> device);

  private:
    constexpr static VkDynamicState dynamicState_[] = {
        // VkPipelineViewportStateCreateInfo
        VK_DYNAMIC_STATE_VIEWPORT, // pViewports
        VK_DYNAMIC_STATE_SCISSOR,  // pScissors
        // uint32_t viewportCount;
        // uint32_t scissorCount;

        // VkPipelineDepthStencilStateCreateInfo
        VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE,    // depthTestEnable
        VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE,   // depthWriteEnable
        VK_DYNAMIC_STATE_DEPTH_COMPARE_OP,     // depthCompareOp
        VK_DYNAMIC_STATE_STENCIL_TEST_ENABLE,  // stencilTestEnable
        VK_DYNAMIC_STATE_STENCIL_OP,           // {front, back}::{failOp, passOp, depthFailOp, compareOp}
        VK_DYNAMIC_STATE_STENCIL_REFERENCE,    // {front, back}::reference
        VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK, // {front, back}::compareMask
        VK_DYNAMIC_STATE_STENCIL_WRITE_MASK,   // {front, back}::writeMask
        // VkBool32 depthBoundsTestEnable;
        // float minDepthBounds;
        // float maxDepthBounds;

        // VkPipelineInputAssemblyStateCreateInfo
        // VK_DYNAMIC_STATE_PRIMITIVE_TOPOLOGY, // topology, not used, topology should paired with shader
        // VkBool32 primitiveRestartEnable;

        // VkPipelineRasterizationStateCreateInfo
        VK_DYNAMIC_STATE_CULL_MODE,         // cullMode
        VK_DYNAMIC_STATE_FRONT_FACE,        // frontFace
        VK_DYNAMIC_STATE_POLYGON_MODE_EXT,  // polygonMode
        VK_DYNAMIC_STATE_DEPTH_BIAS_ENABLE, // depthBiasEnable
        VK_DYNAMIC_STATE_DEPTH_BIAS,        // {depthBiasConstantFactor, depthBiasClamp, depthBiasSlopeFactor}
        VK_DYNAMIC_STATE_LINE_WIDTH,        // lineWidth
        // VkBool32 depthClampEnable;
        // VkBool32 rasterizerDiscardEnable;

        // VkPipelineColorBlendAttachmentState contained in VkPipelineColorBlendStateCreateInfo for every attachment
        VK_DYNAMIC_STATE_COLOR_BLEND_ENABLE_EXT,   // blendEnable
        VK_DYNAMIC_STATE_COLOR_BLEND_EQUATION_EXT, // {srcColorBlendFactor, dstColorBlendFactor, colorBlendOp,
                                                   // srcAlphaBlendFactor, dstAlphaBlendFactor, alphaBlendOp}
        VK_DYNAMIC_STATE_COLOR_WRITE_MASK_EXT,     // colorWriteMask

        // VkPipelineColorBlendStateCreateInfo
        VK_DYNAMIC_STATE_LOGIC_OP_EXT,        // logicOp
        VK_DYNAMIC_STATE_LOGIC_OP_ENABLE_EXT, // logicOpEnable
        VK_DYNAMIC_STATE_BLEND_CONSTANTS,     // blendConstants

        // VK_DYNAMIC_STATE_VERTEX_INPUT_EXT // not used
    };

    VkPipelineDynamicStateCreateInfo dynamicStateCreateInfo_{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = sizeof(dynamicState_) / sizeof(*dynamicState_),
        .pDynamicStates = dynamicState_,
    };

    VkPipelineVertexInputStateCreateInfo vertexInputStateCreateInfo_{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
    };

    VkPipelineInputAssemblyStateCreateInfo inputAssemblyStateCreateInfo_{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .primitiveRestartEnable = VK_FALSE,
    };

    VkPipelineViewportStateCreateInfo viewportStateCreateInfo_{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1,
        .scissorCount = 1,
    };

    VkPipelineRasterizationStateCreateInfo rasterizationStateCreateInfo_{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .depthClampEnable = VK_FALSE,
        .rasterizerDiscardEnable = VK_FALSE,
    };

    VkPipelineMultisampleStateCreateInfo multisampleStateCreateInfo_{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
        .sampleShadingEnable = VK_FALSE,
        .minSampleShading = 1.0f,
        .alphaToCoverageEnable = VK_FALSE,
        .alphaToOneEnable = VK_FALSE,
    };

    std::vector<VkPipelineColorBlendAttachmentState> colorBlendAttachmentStates_;

    VkPipelineColorBlendStateCreateInfo colorBlendState_{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .attachmentCount = 0,
        .pAttachments = nullptr,
    };

    VkPipelineDepthStencilStateCreateInfo depthStencilState_{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .depthBoundsTestEnable = VK_FALSE,
        .minDepthBounds = 0.0f,
        .maxDepthBounds = 1.0f,
    };

    VkPipelineVertexInputStateCreateInfo vertexInputState{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
    };

    ShaderStageBuilder shaderStageBuilder_;

    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
    VkRenderPass renderPass_ = VK_NULL_HANDLE;
    uint32_t subpassIndex_ = 0;
};

template <typename T>
DynamicGraphicsPipelineBuilder &DynamicGraphicsPipelineBuilder::defineVertexInputState() {
    VertexLayoutInfo &vertexLayoutInfo = Vertex::vertexLayoutInfo<T>();

    vertexInputStateCreateInfo_.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputStateCreateInfo_.vertexBindingDescriptionCount = 1;
    vertexInputStateCreateInfo_.pVertexBindingDescriptions = &vertexLayoutInfo.bindingDescription;
    vertexInputStateCreateInfo_.vertexAttributeDescriptionCount = vertexLayoutInfo.attributeDescriptions.size();
    vertexInputStateCreateInfo_.pVertexAttributeDescriptions = vertexLayoutInfo.attributeDescriptions.data();

    return *this;
}

template <>
DynamicGraphicsPipelineBuilder &DynamicGraphicsPipelineBuilder::defineVertexInputState<void>();
} // namespace vk
