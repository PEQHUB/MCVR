#include "vk2_pipeline.hpp"

#include <volk.h>

namespace engine::vk2 {

// --- PipelineLayout ---

PipelineLayout::~PipelineLayout() { destroy(); }

PipelineLayout::PipelineLayout(PipelineLayout&& other) noexcept
    : device_(other.device_), layout_(other.layout_) {
    other.device_ = VK_NULL_HANDLE;
    other.layout_ = VK_NULL_HANDLE;
}

PipelineLayout& PipelineLayout::operator=(PipelineLayout&& other) noexcept {
    if (this != &other) {
        destroy();
        device_ = other.device_;
        layout_ = other.layout_;
        other.device_ = VK_NULL_HANDLE;
        other.layout_ = VK_NULL_HANDLE;
    }
    return *this;
}

Result<PipelineLayout> PipelineLayout::create(VkDevice device, const PipelineLayoutDesc& desc) {
    VkPipelineLayoutCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    ci.setLayoutCount = static_cast<uint32_t>(desc.setLayouts.size());
    ci.pSetLayouts = desc.setLayouts.empty() ? nullptr : desc.setLayouts.data();
    ci.pushConstantRangeCount = static_cast<uint32_t>(desc.pushConstants.size());
    ci.pPushConstantRanges = desc.pushConstants.empty() ? nullptr : desc.pushConstants.data();

    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkResult r = vkCreatePipelineLayout(device, &ci, nullptr, &layout);
    if (r != VK_SUCCESS) {
        return makeError(r, "vkCreatePipelineLayout failed");
    }

    PipelineLayout pl;
    pl.device_ = device;
    pl.layout_ = layout;
    return pl;
}

void PipelineLayout::destroy() {
    if (layout_ != VK_NULL_HANDLE && device_ != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device_, layout_, nullptr);
    }
    layout_ = VK_NULL_HANDLE;
    device_ = VK_NULL_HANDLE;
}

// --- ComputePipeline ---

ComputePipeline::~ComputePipeline() { destroy(); }

ComputePipeline::ComputePipeline(ComputePipeline&& other) noexcept
    : device_(other.device_), pipeline_(other.pipeline_) {
    other.device_ = VK_NULL_HANDLE;
    other.pipeline_ = VK_NULL_HANDLE;
}

ComputePipeline& ComputePipeline::operator=(ComputePipeline&& other) noexcept {
    if (this != &other) {
        destroy();
        device_ = other.device_;
        pipeline_ = other.pipeline_;
        other.device_ = VK_NULL_HANDLE;
        other.pipeline_ = VK_NULL_HANDLE;
    }
    return *this;
}

Result<ComputePipeline> ComputePipeline::create(VkDevice device, const ComputePipelineDesc& desc) {
    VkComputePipelineCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    ci.layout = desc.layout;
    ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    ci.stage.module = desc.shaderModule;
    ci.stage.pName = desc.entryPoint;

    VkPipeline pipeline = VK_NULL_HANDLE;
    VkResult r = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &ci, nullptr, &pipeline);
    if (r != VK_SUCCESS) {
        return makeError(r, "vkCreateComputePipelines failed");
    }

    ComputePipeline cp;
    cp.device_ = device;
    cp.pipeline_ = pipeline;
    return cp;
}

void ComputePipeline::destroy() {
    if (pipeline_ != VK_NULL_HANDLE && device_ != VK_NULL_HANDLE) {
        vkDestroyPipeline(device_, pipeline_, nullptr);
    }
    pipeline_ = VK_NULL_HANDLE;
    device_ = VK_NULL_HANDLE;
}

// --- GraphicsPipeline ---

GraphicsPipeline::~GraphicsPipeline() { destroy(); }

GraphicsPipeline::GraphicsPipeline(GraphicsPipeline&& other) noexcept
    : device_(other.device_), pipeline_(other.pipeline_) {
    other.device_ = VK_NULL_HANDLE;
    other.pipeline_ = VK_NULL_HANDLE;
}

GraphicsPipeline& GraphicsPipeline::operator=(GraphicsPipeline&& other) noexcept {
    if (this != &other) {
        destroy();
        device_ = other.device_;
        pipeline_ = other.pipeline_;
        other.device_ = VK_NULL_HANDLE;
        other.pipeline_ = VK_NULL_HANDLE;
    }
    return *this;
}

Result<GraphicsPipeline> GraphicsPipeline::create(VkDevice device, const GraphicsPipelineDesc& desc) {
    // Shader stages
    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = desc.vertexShader;
    stages[0].pName = desc.vertexEntryPoint;
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = desc.fragmentShader;
    stages[1].pName = desc.fragmentEntryPoint;

    // No vertex input (fullscreen triangle from gl_VertexIndex)
    VkPipelineVertexInputStateCreateInfo vertexInput{};
    vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    // Dynamic viewport + scissor
    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.cullMode = VK_CULL_MODE_NONE;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo multisample{};
    multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = desc.depthTestEnable ? VK_TRUE : VK_FALSE;
    depthStencil.depthWriteEnable = desc.depthWriteEnable ? VK_TRUE : VK_FALSE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

    VkPipelineColorBlendAttachmentState blendAttachment{};
    blendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                                   | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    blendAttachment.blendEnable = desc.blendEnable ? VK_TRUE : VK_FALSE;

    VkPipelineColorBlendStateCreateInfo colorBlend{};
    colorBlend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlend.attachmentCount = 1;
    colorBlend.pAttachments = &blendAttachment;

    VkDynamicState dynamicStates[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = 2;
    dynamicState.pDynamicStates = dynamicStates;

    // Dynamic rendering (no VkRenderPass)
    VkPipelineRenderingCreateInfo renderingInfo{};
    renderingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachmentFormats = &desc.colorAttachmentFormat;
    renderingInfo.depthAttachmentFormat = desc.depthAttachmentFormat;

    VkGraphicsPipelineCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    ci.pNext = &renderingInfo;
    ci.stageCount = 2;
    ci.pStages = stages;
    ci.pVertexInputState = &vertexInput;
    ci.pInputAssemblyState = &inputAssembly;
    ci.pViewportState = &viewportState;
    ci.pRasterizationState = &rasterizer;
    ci.pMultisampleState = &multisample;
    ci.pDepthStencilState = &depthStencil;
    ci.pColorBlendState = &colorBlend;
    ci.pDynamicState = &dynamicState;
    ci.layout = desc.layout;

    VkPipeline pipeline = VK_NULL_HANDLE;
    VkResult r = vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &ci, nullptr, &pipeline);
    if (r != VK_SUCCESS) {
        return makeError(r, "vkCreateGraphicsPipelines failed");
    }

    GraphicsPipeline gp;
    gp.device_ = device;
    gp.pipeline_ = pipeline;
    return gp;
}

void GraphicsPipeline::destroy() {
    if (pipeline_ != VK_NULL_HANDLE && device_ != VK_NULL_HANDLE) {
        vkDestroyPipeline(device_, pipeline_, nullptr);
    }
    pipeline_ = VK_NULL_HANDLE;
    device_ = VK_NULL_HANDLE;
}

} // namespace engine::vk2
