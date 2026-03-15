#pragma once

#include "core/all_extern.hpp"

namespace vk {
class Instance;
class Window;
class PhysicalDevice;

class Device : public SharedObject<Device> {
  public:
    Device(std::shared_ptr<Instance> instance,
           std::shared_ptr<Window> window,
           std::shared_ptr<PhysicalDevice> physicalDevice);
    ~Device();

    VkDevice &vkDevice();
    VkQueue &mainVkQueue();
    VkQueue &secondaryQueue();

    bool hasExtendedDynamicState2LogicOp() const { return extendedDynamicState2LogicOp_; }
    bool hasOMM() const { return ommSupported_; }
    bool has64BitAtomics() const { return shaderBufferInt64Atomics_; }

  private:
    std::shared_ptr<Instance> instance_;
    std::shared_ptr<Window> window_;
    std::shared_ptr<PhysicalDevice> physicalDevice_;

    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue mainQueue_ = VK_NULL_HANDLE;
    VkQueue secondaryQueue_ = VK_NULL_HANDLE;

    bool extendedDynamicState2LogicOp_ = false;
    bool ommSupported_ = false;
    bool shaderBufferInt64Atomics_ = false;
};
}; // namespace vk