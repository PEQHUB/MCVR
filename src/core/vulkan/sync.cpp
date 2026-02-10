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