#pragma once

// Ownership: EngineServices (1:1).
// Thread: All methods main-thread only.
// Dependencies: DeviceService (device, surface, queues).

#include "vk2_result.hpp"

#include <cstdint>
#include <vector>

namespace engine::vk2 {

class DeviceService;

class SwapchainService {
public:
    SwapchainService() = default;
    ~SwapchainService();

    SwapchainService(const SwapchainService&) = delete;
    SwapchainService& operator=(const SwapchainService&) = delete;

    // Create swapchain from device service.
    Result<void> init(DeviceService& device);

    // Destroy and recreate (e.g., after window resize).
    Result<void> recreate(uint32_t width, uint32_t height);

    void shutdown();
    bool isInitialized() const { return swapchain_ != VK_NULL_HANDLE; }

    // --- Frame operations ---

    // Acquire next image. Returns image index. Signals acquireSemaphore.
    Result<uint32_t> acquireNextImage(VkSemaphore signalSemaphore);

    // Present the given image. Waits on waitSemaphore.
    Result<void> present(uint32_t imageIndex, VkSemaphore waitSemaphore);

    // --- Accessors ---

    VkSwapchainKHR handle() const { return swapchain_; }
    uint32_t imageCount() const { return static_cast<uint32_t>(images_.size()); }
    VkImage image(uint32_t index) const { return images_[index]; }
    VkImageView imageView(uint32_t index) const { return imageViews_[index]; }
    VkFormat format() const { return format_; }
    VkExtent2D extent() const { return extent_; }

private:
    DeviceService* deviceService_ = nullptr;
    VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
    std::vector<VkImage> images_;
    std::vector<VkImageView> imageViews_;
    VkFormat format_ = VK_FORMAT_UNDEFINED;
    VkExtent2D extent_{0, 0};

    Result<void> createSwapchain(VkSwapchainKHR oldSwapchain);
    void destroyImageViews();
};

} // namespace engine::vk2
