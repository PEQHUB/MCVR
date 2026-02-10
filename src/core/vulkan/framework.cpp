#include "core/vulkan/framework.hpp"

#include <iostream>

vk::Framework::Context::Context(Framework &framework,
                                uint32_t frame_index,
                                std::shared_ptr<Instance> instance,
                                std::shared_ptr<Window> window,
                                std::shared_ptr<PhysicalDevice> physicalDevice,
                                std::shared_ptr<Device> device,
                                std::shared_ptr<VMA> vma,
                                std::shared_ptr<CommandBuffer> commandBuffer,
                                std::shared_ptr<SwapchainImage> swapchainImage,
                                std::shared_ptr<Semaphore> commandProcessedSemaphore,
                                std::shared_ptr<Fence> commandFinishedFence)
    : framework(framework),
      frameIndex(frame_index),
      instance(instance),
      window(window),
      physicalDevice(physicalDevice),
      device(device),
      vma(vma),
      commandBuffer(commandBuffer),
      swapchainImage(swapchainImage),
      commandProcessedSemaphore(commandProcessedSemaphore),
      commandFinishedFence(commandFinishedFence) {}

vk::Framework::Context::~Context() {
#ifdef DEBUG
    std::cout << "[Context] context deconstructed" << std::endl;
#endif
}

vk::Framework::Framework(GLFWwindow *window)
    : instance_(Instance::create()),
      window_(Window::create(instance_, window)),
      physicalDevice_(PhysicalDevice::create(instance_, window_)),
      device_(Device::create(instance_, window_, physicalDevice_)),
      vma_(VMA::create(instance_, physicalDevice_, device_)),
      swapchain_(Swapchain::create(physicalDevice_, device_, window_)),
      commandPool_(CommandPool::create(physicalDevice_, device_)) {
    uint32_t imageCount = swapchain_->imageCount();

    // create command buffer for each context
    for (int i = 0; i < imageCount; i++) { commandBuffers_.emplace_back(CommandBuffer::create(device_, commandPool_)); }

    // create fence for each context
    for (int i = 0; i < imageCount; i++) { commandFinishedFences_.push_back(Fence::create(device_, true)); }

    // create semaphore for each context for command procssed
    for (int i = 0; i < imageCount; i++) { commandProcessedSemaphores_.push_back(Semaphore::create(device_)); }

    for (int i = 0; i < imageCount; i++) {
        contexts_.push_back(Context::create(*this, i, instance_, window_, physicalDevice_, device_, vma_,
                                            commandBuffers_[i], swapchain_->swapchainImages()[i],
                                            commandProcessedSemaphores_[i], commandFinishedFences_[i]));
    }
}

vk::Framework::Framework(uint32_t width, uint32_t height)
    : instance_(Instance::create()),
      window_(Window::create(instance_, width, height)),
      physicalDevice_(PhysicalDevice::create(instance_, window_)),
      device_(Device::create(instance_, window_, physicalDevice_)),
      vma_(VMA::create(instance_, physicalDevice_, device_)),
      swapchain_(Swapchain::create(physicalDevice_, device_, window_)),
      commandPool_(CommandPool::create(physicalDevice_, device_)) {
    uint32_t imageCount = swapchain_->imageCount();

    // create command buffer for each context
    for (int i = 0; i < imageCount; i++) { commandBuffers_.emplace_back(CommandBuffer::create(device_, commandPool_)); }

    // create fence for each context
    for (int i = 0; i < imageCount; i++) { commandFinishedFences_.push_back(Fence::create(device_, true)); }

    // create semaphore for each context for command procssed
    for (int i = 0; i < imageCount; i++) { commandProcessedSemaphores_.push_back(Semaphore::create(device_)); }

    for (int i = 0; i < imageCount; i++) {
        contexts_.push_back(Context::create(*this, i, instance_, window_, physicalDevice_, device_, vma_,
                                            commandBuffers_[i], swapchain_->swapchainImages()[i],
                                            commandProcessedSemaphores_[i], commandFinishedFences_[i]));
    }
}

vk::Framework::~Framework() {
#ifdef DEBUG
    std::cout << "[Framework] framework deconstructed" << std::endl;
#endif
}

std::shared_ptr<vk::Framework::Context> vk::Framework::acquireContext() {
    std::shared_ptr<Semaphore> imageAcquiredSemaphore = acquireSemaphore();

    uint32_t imageIndex;
    if (vkAcquireNextImageKHR(device_->vkDevice(), swapchain_->vkSwapchain(), UINT64_MAX,
                              imageAcquiredSemaphore->vkSemaphore(), VK_NULL_HANDLE, &imageIndex) != VK_SUCCESS) {
        std::cerr << "Cannot acquire images from swapchain" << std::endl;
        exit(EXIT_FAILURE);
    }

    std::shared_ptr<Fence> fence = contexts_[imageIndex]->commandFinishedFence;
    vkWaitForFences(device_->vkDevice(), 1, &fence->vkFence(), true, UINT64_MAX);
    vkResetFences(device_->vkDevice(), 1, &fence->vkFence());

    if (contexts_[imageIndex]->imageAcquiredSemaphore != VK_NULL_HANDLE) {
        recycleSemaphore(contexts_[imageIndex]->imageAcquiredSemaphore);
        contexts_[imageIndex]->imageAcquiredSemaphore = VK_NULL_HANDLE;
    }
    contexts_[imageIndex]->imageAcquiredSemaphore = imageAcquiredSemaphore;

    return contexts_[imageIndex];
}

void vk::Framework::submitCommand(std::shared_ptr<Context> context) {
    context->commandBuffer->submitMainQueue(
        device_, {
                     .waitSemaphoresAndStageMasks = {{context->imageAcquiredSemaphore->vkSemaphore(),
                                                      VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT}},
                     .signalSemaphores = {context->commandProcessedSemaphore->vkSemaphore()},
                     .signalFence = context->commandFinishedFence->vkFence(),
                 });
}

void vk::Framework::present(std::shared_ptr<Context> context) {
    VkPresentInfoKHR presentInfo = {};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = &context->commandProcessedSemaphore->vkSemaphore();

    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = &swapchain_->vkSwapchain();
    presentInfo.pImageIndices = &context->frameIndex;

    VkResult res = vkQueuePresentKHR(device_->mainVkQueue(), &presentInfo);

    // TODO: recreate swapchain
    if (res != VK_SUCCESS) {
        std::cerr << "failed to submit present command buffer" << std::endl;
        exit(1);
    }
}

std::shared_ptr<vk::Instance> vk::Framework::instance() {
    return instance_;
}

std::shared_ptr<vk::Window> vk::Framework::window() {
    return window_;
}

std::shared_ptr<vk::PhysicalDevice> vk::Framework::physicalDevice() {
    return physicalDevice_;
}

std::shared_ptr<vk::Device> vk::Framework::device() {
    return device_;
}

std::shared_ptr<vk::VMA> vk::Framework::vma() {
    return vma_;
}

std::shared_ptr<vk::Swapchain> vk::Framework::swapchain() {
    return swapchain_;
}

std::shared_ptr<vk::CommandPool> vk::Framework::commandPool() {
    return commandPool_;
}

std::shared_ptr<std::vector<std::shared_ptr<vk::CommandBuffer>>> vk::Framework::commandBuffers() {
    return std::shared_ptr<std::vector<std::shared_ptr<vk::CommandBuffer>>>(shared_from_this(), &commandBuffers_);
}

std::queue<std::shared_ptr<vk::Semaphore>> &vk::Framework::recycledSemaphores() {
    return recycledImageAcquiredSemaphores_;
}

std::vector<std::shared_ptr<vk::Semaphore>> &vk::Framework::commandProcessedSemaphores() {
    return commandProcessedSemaphores_;
}

std::vector<std::shared_ptr<vk::Fence>> &vk::Framework::commandFinishedFences() {
    return commandFinishedFences_;
}

std::vector<std::shared_ptr<vk::Framework::Context>> &vk::Framework::contexts() {
    return contexts_;
}

std::shared_ptr<vk::Semaphore> vk::Framework::acquireSemaphore() {
    std::shared_ptr<Semaphore> semaphore;
    if (recycledImageAcquiredSemaphores_.empty()) {
        // std::cout << "no recycledSemaphores left, alloc new one: " << std::hex << semaphore << std::endl;
        semaphore = Semaphore::create(device_);
    } else {
        // std::cout << "recycledSemaphores left, use existing one: " << std::hex << semaphore << std::endl;
        semaphore = recycledImageAcquiredSemaphores_.front();
        recycledImageAcquiredSemaphores_.pop();
    }
    return semaphore;
}

void vk::Framework::recycleSemaphore(std::shared_ptr<Semaphore> semaphore) {
    recycledImageAcquiredSemaphores_.push(semaphore);
}