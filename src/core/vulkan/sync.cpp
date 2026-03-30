#include "core/vulkan/sync.hpp"

#include "core/vulkan/device.hpp"

vk::Semaphore::Semaphore(std::shared_ptr<Device> device) : device_(device) {
    VkSemaphoreCreateInfo semaphoreInfo{};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    vkCreateSemaphore(device_->vkDevice(), &semaphoreInfo, nullptr, &semaphore_);
}

vk::Semaphore::~Semaphore() {
    vkDestroySemaphore(device_->vkDevice(), semaphore_, nullptr);
}

VkSemaphore &vk::Semaphore::vkSemaphore() {
    return semaphore_;
}

vk::Fence::Fence(std::shared_ptr<Device> device) : Fence(device, false) {}

vk::Fence::Fence(std::shared_ptr<Device> device, bool signaled) : device_(device) {
    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    if (signaled) { fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT; }

    vkCreateFence(device_->vkDevice(), &fenceInfo, nullptr, &fence_);
}

vk::Fence::~Fence() {
    vkDestroyFence(device_->vkDevice(), fence_, nullptr);
}

VkFence &vk::Fence::vkFence() {
    return fence_;
}

// --- TimelineSemaphore ---

vk::TimelineSemaphore::TimelineSemaphore(std::shared_ptr<Device> device, uint64_t initialValue) : device_(device) {
    VkSemaphoreTypeCreateInfo typeInfo{};
    typeInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    typeInfo.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    typeInfo.initialValue = initialValue;

    VkSemaphoreCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    createInfo.pNext = &typeInfo;

    vkCreateSemaphore(device_->vkDevice(), &createInfo, nullptr, &semaphore_);
}

vk::TimelineSemaphore::~TimelineSemaphore() {
    vkDestroySemaphore(device_->vkDevice(), semaphore_, nullptr);
}

VkSemaphore &vk::TimelineSemaphore::vkSemaphore() {
    return semaphore_;
}

uint64_t vk::TimelineSemaphore::getValue() const {
    uint64_t value = 0;
    vkGetSemaphoreCounterValue(device_->vkDevice(), semaphore_, &value);
    return value;
}

VkResult vk::TimelineSemaphore::waitValue(uint64_t value, uint64_t timeout) const {
    VkSemaphoreWaitInfo waitInfo{};
    waitInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
    waitInfo.semaphoreCount = 1;
    waitInfo.pSemaphores = &semaphore_;
    waitInfo.pValues = &value;
    return vkWaitSemaphores(device_->vkDevice(), &waitInfo, timeout);
}

void vk::TimelineSemaphore::signal(uint64_t value) {
    VkSemaphoreSignalInfo signalInfo{};
    signalInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO;
    signalInfo.semaphore = semaphore_;
    signalInfo.value = value;
    vkSignalSemaphore(device_->vkDevice(), &signalInfo);
}