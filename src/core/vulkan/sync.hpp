#pragma once

#include "core/all_extern.hpp"

namespace vk {
class Device;

class Semaphore : public SharedObject<Semaphore> {
  public:
    Semaphore(std::shared_ptr<Device> device);
    ~Semaphore();

    VkSemaphore& vkSemaphore();

  private:
    std::shared_ptr<Device> device_;

    VkSemaphore semaphore_;
};

class Fence : public SharedObject<Fence> {
  public:
    Fence(std::shared_ptr<Device> device);
    Fence(std::shared_ptr<Device> device, bool signaled);
    ~Fence();

    VkFence& vkFence();

  private:
    std::shared_ptr<Device> device_;

    VkFence fence_;
};

class TimelineSemaphore : public SharedObject<TimelineSemaphore> {
  public:
    TimelineSemaphore(std::shared_ptr<Device> device, uint64_t initialValue = 0);
    ~TimelineSemaphore();

    VkSemaphore& vkSemaphore();

    /// Query current counter value (may lag behind GPU — use waitValue for guaranteed sync)
    uint64_t getValue() const;

    /// CPU-side blocking wait until counter >= value. Returns VK_SUCCESS or VK_ERROR_DEVICE_LOST.
    VkResult waitValue(uint64_t value, uint64_t timeout = UINT64_MAX) const;

    /// CPU-side signal: set counter to value (must be > current value). Used for shutdown unblocking.
    void signal(uint64_t value);

  private:
    std::shared_ptr<Device> device_;
    VkSemaphore semaphore_;
};
}; // namespace vk