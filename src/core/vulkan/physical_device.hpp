#pragma once

#include "core/all_extern.hpp"

namespace vk {
class Instance;
class Window;

class PhysicalDevice : public SharedObject<PhysicalDevice> {
  public:
    PhysicalDevice(std::shared_ptr<Instance> instance, std::shared_ptr<Window> window);
    ~PhysicalDevice();

    VkPhysicalDevice &vkPhysicalDevice();
    uint32_t mainQueueIndex();
    uint32_t secondaryQueueIndex();

    void findPhysicalDevice();
    void findQueueFamilies();

    VkPhysicalDeviceProperties properties();
    VkPhysicalDeviceRayTracingPipelinePropertiesKHR rayTracingProperties();
    VkPhysicalDeviceAccelerationStructurePropertiesKHR accelerationStructProperties();

    uint32_t vendorID() const { return properties_.vendorID; }
    bool isAMD() const { return properties_.vendorID == 0x1002; }
    bool isNVIDIA() const { return properties_.vendorID == 0x10de; }
    bool isIntel() const { return properties_.vendorID == 0x8086; }

  private:
    std::shared_ptr<Instance> instance_;
    std::shared_ptr<Window> window_;

    VkPhysicalDevice physicalDevice_ = VK_NULL_HANDLE;
    uint32_t mainQueueIndex_ = -1;
    uint32_t secondaryQueueIndex_ = -1;

    VkPhysicalDeviceProperties properties_;
    VkPhysicalDeviceRayTracingPipelinePropertiesKHR rayTracingProperties_;
    VkPhysicalDeviceAccelerationStructurePropertiesKHR accelerationStructProperties_;
};
} // namespace vk