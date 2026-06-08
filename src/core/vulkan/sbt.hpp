#pragma once

#include "core/all_extern.hpp"

#include <vector>

namespace vk {
class HostVisibleBuffer;
class PhysicalDevice;
class Device;
class VMA;
class RayTracingPipeline;

class SBT : public SharedObject<SBT> {
  public:
    SBT(std::shared_ptr<PhysicalDevice> physicalDevice,
        std::shared_ptr<Device> device,
        std::shared_ptr<VMA> vma,
        std::shared_ptr<RayTracingPipeline> pipeline,
        uint32_t missCount,
        uint32_t hitCount);
    ~SBT();

    void setupHitSBT(std::vector<uint32_t> &hitGroupIndices);

    VkStridedDeviceAddressRegionKHR &raygenRegion();
    VkStridedDeviceAddressRegionKHR &missRegion();
    VkStridedDeviceAddressRegionKHR &hitRegion();
    VkStridedDeviceAddressRegionKHR &callableRegion();

  private:
    std::shared_ptr<vk::VMA> vma_;
    std::shared_ptr<vk::Device> device_;

    uint32_t handleSize_;
    uint32_t alignedHandleSize_;
    uint32_t baseAlignment_;
    uint32_t missCount_;

    std::vector<uint8_t> shaderHandleStorage_;

    VkStridedDeviceAddressRegionKHR raygenRegion_{};
    VkStridedDeviceAddressRegionKHR missRegion_{};
    VkStridedDeviceAddressRegionKHR hitRegion_{};
    VkStridedDeviceAddressRegionKHR callableRegion_{};

    std::shared_ptr<HostVisibleBuffer> rgenSBT_;
    std::shared_ptr<HostVisibleBuffer> rmissSBT_;
    std::shared_ptr<HostVisibleBuffer> rhitSBT_;
};
}; // namespace vk