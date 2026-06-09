#pragma once

// Ownership: EngineServices (1:1).
// Thread: All methods main-thread only.
// Dependencies: DeviceService (device, surface, queues).
//
// State machine: recreateNeeded_ is set by acquire/present on SUBOPTIMAL/OUT_OF_DATE.
// zeroExtent_ is set when the window is minimized (0x0 surface). Caller must
// skip rendering when zeroExtent() is true and recreate when recreateNeeded() is true.

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

    Result<void> init(DeviceService& device);
    Result<void> recreate(uint32_t width, uint32_t height);
    void shutdown();

    bool isInitialized() const { return swapchain_ != VK_NULL_HANDLE; }

    // --- State queries ---
    bool isRecreateNeeded() const { return recreateNeeded_; }
    void markRecreateNeeded() { recreateNeeded_ = true; }
    void clearRecreateNeeded() { recreateNeeded_ = false; }
    bool isZeroExtent() const { return zeroExtent_; }
    bool supportsTransferSrc() const { return transferSrcEnabled_; }
    VkPresentModeKHR presentMode() const { return presentMode_; }

    // --- Frame operations ---
    Result<uint32_t> acquireNextImage(VkSemaphore signalSemaphore);
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
    VkPresentModeKHR presentMode_ = VK_PRESENT_MODE_FIFO_KHR;
    bool recreateNeeded_ = false;
    bool zeroExtent_ = false;
    bool transferSrcEnabled_ = false;

    Result<void> createSwapchain(VkSwapchainKHR oldSwapchain);
    void destroyImageViews();
};

} // namespace engine::vk2
