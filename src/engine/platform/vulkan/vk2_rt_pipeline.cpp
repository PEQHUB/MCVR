#include "vk2_rt_pipeline.hpp"

#include <volk.h>
#include <cstring>

namespace engine::vk2 {

// --- RtPipeline ---

RtPipeline::~RtPipeline() { destroy(); }

RtPipeline::RtPipeline(RtPipeline&& o) noexcept
    : device_(o.device_), pipeline_(o.pipeline_),
      groupHandles_(std::move(o.groupHandles_)), groupCount_(o.groupCount_) {
    o.device_ = VK_NULL_HANDLE;
    o.pipeline_ = VK_NULL_HANDLE;
    o.groupCount_ = 0;
}

RtPipeline& RtPipeline::operator=(RtPipeline&& o) noexcept {
    if (this != &o) {
        destroy();
        device_ = o.device_; pipeline_ = o.pipeline_;
        groupHandles_ = std::move(o.groupHandles_); groupCount_ = o.groupCount_;
        o.device_ = VK_NULL_HANDLE; o.pipeline_ = VK_NULL_HANDLE; o.groupCount_ = 0;
    }
    return *this;
}

Result<RtPipeline> RtPipeline::create(VkDevice device, VkPhysicalDevice physicalDevice,
                                      const RtPipelineDesc& desc) {
    // Convert groups
    std::vector<VkRayTracingShaderGroupCreateInfoKHR> groups(desc.groups.size());
    for (size_t i = 0; i < desc.groups.size(); ++i) {
        auto& g = groups[i];
        g.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
        g.type = desc.groups[i].type;
        g.generalShader = desc.groups[i].generalShader;
        g.closestHitShader = desc.groups[i].closestHitShader;
        g.anyHitShader = desc.groups[i].anyHitShader;
        g.intersectionShader = desc.groups[i].intersectionShader;
    }

    VkRayTracingPipelineCreateInfoKHR ci{};
    ci.sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR;
    ci.stageCount = static_cast<uint32_t>(desc.stages.size());
    ci.pStages = desc.stages.data();
    ci.groupCount = static_cast<uint32_t>(groups.size());
    ci.pGroups = groups.data();
    ci.maxPipelineRayRecursionDepth = desc.maxRecursionDepth;
    ci.layout = desc.layout;

    VkPipeline pipeline = VK_NULL_HANDLE;
    VkResult r = vkCreateRayTracingPipelinesKHR(device, VK_NULL_HANDLE, VK_NULL_HANDLE,
                                                 1, &ci, nullptr, &pipeline);
    if (r != VK_SUCCESS) return makeError(r, "vkCreateRayTracingPipelinesKHR failed");

    // Get shader group handles — query actual handle size from device
    VkPhysicalDeviceRayTracingPipelinePropertiesKHR rtProps{};
    rtProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR;
    VkPhysicalDeviceProperties2 props2{};
    props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    props2.pNext = &rtProps;
    vkGetPhysicalDeviceProperties2(physicalDevice, &props2);
    uint32_t handleSize = rtProps.shaderGroupHandleSize;

    uint32_t groupCount = static_cast<uint32_t>(groups.size());
    std::vector<uint8_t> handles(groupCount * handleSize);
    r = vkGetRayTracingShaderGroupHandlesKHR(device, pipeline, 0, groupCount,
                                              handles.size(), handles.data());
    if (r != VK_SUCCESS) {
        vkDestroyPipeline(device, pipeline, nullptr);
        return makeError(r, "vkGetRayTracingShaderGroupHandlesKHR failed");
    }

    RtPipeline rtp;
    rtp.device_ = device;
    rtp.pipeline_ = pipeline;
    rtp.groupHandles_ = std::move(handles);
    rtp.groupCount_ = groupCount;
    return rtp;
}

void RtPipeline::destroy() {
    if (pipeline_ != VK_NULL_HANDLE && device_ != VK_NULL_HANDLE) {
        vkDestroyPipeline(device_, pipeline_, nullptr);
    }
    pipeline_ = VK_NULL_HANDLE;
    device_ = VK_NULL_HANDLE;
}

// --- ShaderBindingTable ---

static uint32_t alignUp(uint32_t value, uint32_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

Result<void> ShaderBindingTable::build(VkDevice device, VmaAllocator vma,
                                       VkPhysicalDevice physicalDevice,
                                       const RtPipeline& pipeline,
                                       uint32_t rgenCount, uint32_t missCount,
                                       uint32_t hitCount, uint32_t callableCount) {
    // Query RT pipeline properties for handle size + alignment
    VkPhysicalDeviceRayTracingPipelinePropertiesKHR rtProps{};
    rtProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR;
    VkPhysicalDeviceProperties2 props2{};
    props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    props2.pNext = &rtProps;
    vkGetPhysicalDeviceProperties2(physicalDevice, &props2);

    uint32_t handleSize = rtProps.shaderGroupHandleSize;
    uint32_t handleAlignment = rtProps.shaderGroupHandleAlignment;
    uint32_t baseAlignment = rtProps.shaderGroupBaseAlignment;
    uint32_t alignedHandleSize = alignUp(handleSize, handleAlignment);

    const auto& handles = pipeline.groupHandles();
    uint32_t offset = 0;

    auto createRegionBuffer = [&](uint32_t count, VkStridedDeviceAddressRegionKHR& region,
                                   Buffer& buffer) -> Result<void> {
        if (count == 0) { region = {}; return {}; }

        uint32_t stride = alignUp(handleSize, handleAlignment);
        // Region size (for VkStridedDeviceAddressRegionKHR): tight-packed.
        // VUID-vkCmdTraceRaysKHR-pRayGenShaderBindingTable-04023 requires raygen size == stride.
        // For count=1 (raygen, miss, hit with single entries), regionSize naturally equals stride.
        // For count>1, regionSize = count * stride (still valid for miss/hit/callable).
        VkDeviceSize regionSize = static_cast<VkDeviceSize>(count) * stride;
        // Buffer allocation size: padded to baseAlignment for size hygiene.
        // bd.minAlignment = baseAlignment ensures the base address is also aligned,
        // satisfying the spec requirement on VkStridedDeviceAddressRegionKHR.deviceAddress.
        VkDeviceSize bufferSize = alignUp(static_cast<uint32_t>(regionSize), baseAlignment);

        Buffer::Desc bd{};
        bd.size = bufferSize;
        bd.minAlignment = baseAlignment;  // spec: VkStridedDeviceAddressRegionKHR.deviceAddress must be a multiple of shaderGroupBaseAlignment
        bd.usage = VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR
                 | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        bd.vmaFlags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
                    | VMA_ALLOCATION_CREATE_MAPPED_BIT;

        auto r = Buffer::create(device, vma, bd);
        if (!r) return makeError(r.error().vkResult, "SBT buffer: " + r.error().message);
        buffer = std::move(r.value());

        // Copy handles
        auto* dst = static_cast<uint8_t*>(buffer.mappedPtr());
        for (uint32_t i = 0; i < count; ++i) {
            std::memcpy(dst + i * stride, handles.data() + (offset + i) * handleSize, handleSize);
        }
        buffer.flush(); // Ensure SBT is GPU-visible on non-coherent heaps
        offset += count;

        region.deviceAddress = buffer.deviceAddress();
        region.stride = stride;
        region.size = regionSize;
        return {};
    };

    auto r1 = createRegionBuffer(rgenCount, rgen_, rgenBuffer_);
    if (!r1) return r1;
    auto r2 = createRegionBuffer(missCount, miss_, missBuffer_);
    if (!r2) return r2;
    auto r3 = createRegionBuffer(hitCount, hit_, hitBuffer_);
    if (!r3) return r3;

    callable_ = {}; // Not used yet

    return {};
}

} // namespace engine::vk2
