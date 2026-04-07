#pragma once

// Ownership: Caller (unique ownership via RAII, movable).
// Thread: Create on main thread. Handles are immutable after creation.
// Dependencies: VkDevice, ShaderModule.

#include "vk2_result.hpp"

#include <cstdint>
#include <vector>

namespace engine::vk2 {

// --- Pipeline Layout ---

struct PipelineLayoutDesc {
    std::vector<VkDescriptorSetLayout> setLayouts;
    std::vector<VkPushConstantRange> pushConstants;
};

class PipelineLayout {
public:
    ~PipelineLayout();

    PipelineLayout(const PipelineLayout&) = delete;
    PipelineLayout& operator=(const PipelineLayout&) = delete;
    PipelineLayout(PipelineLayout&& other) noexcept;
    PipelineLayout& operator=(PipelineLayout&& other) noexcept;

    PipelineLayout() = default;

    static Result<PipelineLayout> create(VkDevice device, const PipelineLayoutDesc& desc);

    VkPipelineLayout handle() const { return layout_; }

private:

    VkDevice device_ = VK_NULL_HANDLE;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;

    void destroy();
};

// --- Compute Pipeline ---

struct ComputePipelineDesc {
    VkShaderModule shaderModule = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    const char* entryPoint = "main";
};

class ComputePipeline {
public:
    ~ComputePipeline();

    ComputePipeline(const ComputePipeline&) = delete;
    ComputePipeline& operator=(const ComputePipeline&) = delete;
    ComputePipeline(ComputePipeline&& other) noexcept;
    ComputePipeline& operator=(ComputePipeline&& other) noexcept;

    ComputePipeline() = default;

    static Result<ComputePipeline> create(VkDevice device, const ComputePipelineDesc& desc);

    VkPipeline handle() const { return pipeline_; }

private:

    VkDevice device_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;

    void destroy();
};

// --- Graphics Pipeline (dynamic rendering) ---

struct GraphicsPipelineDesc {
    VkShaderModule vertexShader = VK_NULL_HANDLE;
    VkShaderModule fragmentShader = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkFormat colorAttachmentFormat = VK_FORMAT_UNDEFINED;
    VkFormat depthAttachmentFormat = VK_FORMAT_UNDEFINED;
    bool depthTestEnable = false;
    bool depthWriteEnable = false;
    bool blendEnable = false;
    const char* vertexEntryPoint = "main";
    const char* fragmentEntryPoint = "main";
};

class GraphicsPipeline {
public:
    ~GraphicsPipeline();

    GraphicsPipeline(const GraphicsPipeline&) = delete;
    GraphicsPipeline& operator=(const GraphicsPipeline&) = delete;
    GraphicsPipeline(GraphicsPipeline&& other) noexcept;
    GraphicsPipeline& operator=(GraphicsPipeline&& other) noexcept;

    GraphicsPipeline() = default;

    static Result<GraphicsPipeline> create(VkDevice device, const GraphicsPipelineDesc& desc);

    VkPipeline handle() const { return pipeline_; }

private:

    VkDevice device_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;

    void destroy();
};

} // namespace engine::vk2
