#pragma once

// Ownership: Caller (RAII, movable).
// Thread: Create on main thread.
// Dependencies: VkDevice, ShaderModules.

#include "vk2_buffer.hpp"
#include "vk2_result.hpp"

#include <cstdint>
#include <vector>

namespace engine::vk2 {

// --- RT Pipeline ---

struct RtShaderGroup {
    VkRayTracingShaderGroupTypeKHR type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    uint32_t generalShader = VK_SHADER_UNUSED_KHR;
    uint32_t closestHitShader = VK_SHADER_UNUSED_KHR;
    uint32_t anyHitShader = VK_SHADER_UNUSED_KHR;
    uint32_t intersectionShader = VK_SHADER_UNUSED_KHR;
};

struct RtPipelineDesc {
    std::vector<VkPipelineShaderStageCreateInfo> stages;
    std::vector<RtShaderGroup> groups;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    uint32_t maxRecursionDepth = 1;
};

class RtPipeline {
public:
    RtPipeline() = default;
    ~RtPipeline();

    RtPipeline(const RtPipeline&) = delete;
    RtPipeline& operator=(const RtPipeline&) = delete;
    RtPipeline(RtPipeline&& other) noexcept;
    RtPipeline& operator=(RtPipeline&& other) noexcept;

    static Result<RtPipeline> create(VkDevice device, VkPhysicalDevice physicalDevice,
                                     const RtPipelineDesc& desc);

    VkPipeline handle() const { return pipeline_; }

    // Get shader group handles (for SBT construction).
    const std::vector<uint8_t>& groupHandles() const { return groupHandles_; }
    uint32_t groupCount() const { return groupCount_; }

private:
    VkDevice device_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    std::vector<uint8_t> groupHandles_;
    uint32_t groupCount_ = 0;

    void destroy();
};

// --- Shader Binding Table ---

class ShaderBindingTable {
public:
    ShaderBindingTable() = default;

    // Build SBT regions from an RT pipeline's group handles.
    // groupCounts: [rgenCount, missCount, hitCount, callableCount]
    Result<void> build(VkDevice device, VmaAllocator vma,
                       VkPhysicalDevice physicalDevice,
                       const RtPipeline& pipeline,
                       uint32_t rgenCount, uint32_t missCount,
                       uint32_t hitCount, uint32_t callableCount = 0);

    const VkStridedDeviceAddressRegionKHR& rgenRegion() const { return rgen_; }
    const VkStridedDeviceAddressRegionKHR& missRegion() const { return miss_; }
    const VkStridedDeviceAddressRegionKHR& hitRegion() const { return hit_; }
    const VkStridedDeviceAddressRegionKHR& callableRegion() const { return callable_; }

private:
    Buffer rgenBuffer_;
    Buffer missBuffer_;
    Buffer hitBuffer_;

    VkStridedDeviceAddressRegionKHR rgen_{};
    VkStridedDeviceAddressRegionKHR miss_{};
    VkStridedDeviceAddressRegionKHR hit_{};
    VkStridedDeviceAddressRegionKHR callable_{};
};

} // namespace engine::vk2
