#pragma once

#include "core/all_extern.hpp"

#include <mutex>

namespace vk {
class Instance;
class Window;
class PhysicalDevice;
class TimelineSemaphore;

class Device : public SharedObject<Device> {
  public:
    Device(std::shared_ptr<Instance> instance,
           std::shared_ptr<Window> window,
           std::shared_ptr<PhysicalDevice> physicalDevice);
    ~Device();

    VkDevice &vkDevice();
    VkQueue &mainVkQueue();
    VkQueue &secondaryQueue();
    std::mutex &queueMutex() { return queueMtx_; }
    VkPipelineCache pipelineCache() const { return pipelineCache_; }

    void savePipelineCache();

    /// Create Device-level timeline semaphores. Must be called after Device::create() returns
    /// (shared_from_this() is not valid inside the constructor).
    void createTimelineSemaphores();
    std::shared_ptr<TimelineSemaphore> blasSemaphore() { return blasSemaphore_; }

    bool hasExtendedDynamicState2LogicOp() const { return extendedDynamicState2LogicOp_; }
    bool hasOMM() const { return ommSupported_; }
    bool hasSER() const { return serSupported_; }
    bool hasShaderClock() const { return shaderClockSupported_; }
    bool hasCheckpoints() const { return checkpointsSupported_; }
    bool hasDeviceFault() const { return deviceFaultSupported_; }

  private:
    std::shared_ptr<Instance> instance_;
    std::shared_ptr<Window> window_;
    std::shared_ptr<PhysicalDevice> physicalDevice_;

    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue mainQueue_ = VK_NULL_HANDLE;
    VkQueue secondaryQueue_ = VK_NULL_HANDLE;
    VkPipelineCache pipelineCache_ = VK_NULL_HANDLE;

    std::mutex queueMtx_;  // protects mainVkQueue submits from multiple threads
    bool extendedDynamicState2LogicOp_ = false;
    bool ommSupported_ = false;
    bool serSupported_ = false;
    bool shaderClockSupported_ = false;
    bool checkpointsSupported_ = false;
    bool deviceFaultSupported_ = false;

    std::shared_ptr<TimelineSemaphore> blasSemaphore_;

    void loadPipelineCache();
    static std::string pipelineCachePath();
};
}; // namespace vk