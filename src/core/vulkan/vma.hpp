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
    VMA(std::shared_ptr<Instance> instance,
        std::shared_ptr<PhysicalDevice> physicalDevice,
        std::shared_ptr<Device> device);
    ~VMA();

    VmaAllocator &allocator();

  private:
    VmaAllocator allocator_ = VK_NULL_HANDLE;
    VmaVulkanFunctions vulkanFunctions_{};
};
}; // namespace vk