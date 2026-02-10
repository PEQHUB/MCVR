#pragma once

#include "core/all_extern.hpp"

#include "core/vulkan/command.hpp"
#include "core/vulkan/device.hpp"
#include "core/vulkan/image.hpp"
#include "core/vulkan/instance.hpp"
#include "core/vulkan/physical_device.hpp"
#include "core/vulkan/swapchain.hpp"
#include "core/vulkan/sync.hpp"
#include "core/vulkan/vma.hpp"
#include "core/vulkan/window.hpp"

#include <queue>
#include <vector>

namespace vk {

class Framework : public SharedObject<Framework> {
  public:
    struct Context : public SharedObject<Context> {
        Framework &framework;

        uint32_t frameIndex;

        std::shared_ptr<Instance> instance;
        std::shared_ptr<Window> window;
        std::shared_ptr<PhysicalDevice> physicalDevice;
        std::shared_ptr<Device> device;
        std::shared_ptr<VMA> vma;
        std::shared_ptr<CommandBuffer> commandBuffer;
        std::shared_ptr<SwapchainImage> swapchainImage;

        std::shared_ptr<Semaphore> imageAcquiredSemaphore = nullptr;
        std::shared_ptr<Semaphore> commandProcessedSemaphore;
        std::shared_ptr<Fence> commandFinishedFence;

        Context(Framework &framework,
                uint32_t frame_index,
                std::shared_ptr<Instance> instance,
                std::shared_ptr<Window> window,
                std::shared_ptr<PhysicalDevice> physicalDevice,
                std::shared_ptr<Device> device,
                std::shared_ptr<VMA> vma,
                std::shared_ptr<CommandBuffer> commandBuffer,
                std::shared_ptr<SwapchainImage> swapchainImage,
                std::shared_ptr<Semaphore> commandProcessedSemaphore,
                std::shared_ptr<Fence> commandFinishedFence);
        ~Context();
    };

  public:
    Framework(GLFWwindow *window_);
    Framework(uint32_t width, uint32_t height);
    ~Framework();

    std::shared_ptr<Context> acquireContext();
    void submitCommand(std::shared_ptr<Context> context);
    void present(std::shared_ptr<Context> context);

    std::shared_ptr<Instance> instance();
    std::shared_ptr<Window> window();
    std::shared_ptr<PhysicalDevice> physicalDevice();
    std::shared_ptr<Device> device();
    std::shared_ptr<VMA> vma();
    std::shared_ptr<Swapchain> swapchain();
    std::shared_ptr<CommandPool> commandPool();
    std::shared_ptr<std::vector<std::shared_ptr<CommandBuffer>>> commandBuffers();

    std::queue<std::shared_ptr<Semaphore>> &recycledSemaphores();
    std::vector<std::shared_ptr<Semaphore>> &commandProcessedSemaphores();
    std::vector<std::shared_ptr<Fence>> &commandFinishedFences();
    std::vector<std::shared_ptr<Context>> &contexts();

  private:
    std::shared_ptr<Semaphore> acquireSemaphore();
    void recycleSemaphore(std::shared_ptr<Semaphore> semaphore);

  private:
    std::shared_ptr<Instance> instance_;
    std::shared_ptr<Window> window_;
    std::shared_ptr<PhysicalDevice> physicalDevice_;
    std::shared_ptr<Device> device_;
    std::shared_ptr<VMA> vma_;
    std::shared_ptr<Swapchain> swapchain_;
    std::shared_ptr<CommandPool> commandPool_;
    std::vector<std::shared_ptr<CommandBuffer>> commandBuffers_;

    std::queue<std::shared_ptr<Semaphore>> recycledImageAcquiredSemaphores_;
    std::vector<std::shared_ptr<Semaphore>> commandProcessedSemaphores_;
    std::vector<std::shared_ptr<Fence>> commandFinishedFences_;

    std::vector<std::shared_ptr<Context>> contexts_;
};
}; // namespace vk