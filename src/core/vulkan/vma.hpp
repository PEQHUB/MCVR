#pragma once

#include <functional>
#include <deque>

#include "core/all_extern.hpp"

namespace vk {
class PhysicalDevice;
class Device;
class Instance;

class VMA : public SharedObject<VMA> {
  public:
    struct VmaStatsSnapshot {
        uint64_t totalAllocBytes;
        uint64_t totalUsedBytes;
        uint64_t budgetBytes;
        uint64_t budgetUsageBytes;
        uint32_t allocationCount;
        uint32_t blockCount;
    };

    VMA(std::shared_ptr<Instance> instance,
        std::shared_ptr<PhysicalDevice> physicalDevice,
        std::shared_ptr<Device> device);
    ~VMA();

    VmaAllocator &allocator();
    VmaStatsSnapshot getStats() const;

  private:
    VmaAllocator allocator_ = VK_NULL_HANDLE;
    VmaVulkanFunctions vulkanFunctions_{};
};
}; // namespace vk