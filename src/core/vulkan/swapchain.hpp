#pragma once

#include "core/all_extern.hpp"

#include <vector>

namespace vk {
class PhysicalDevice;
class Device;
class Window;
class SwapchainImage;

class Swapchain : public SharedObject<Swapchain> {
    friend class SwapchainImage;
    friend class std::vector<SwapchainImage>;

  public:
    Swapchain(std::shared_ptr<PhysicalDevice> physicalDevice,
              std::shared_ptr<Device> device,
              std::shared_ptr<Window> window);
    ~Swapchain();

    void reconstruct();
    VkSwapchainKHR &vkSwapchain();
    VkExtent2D &vkExtent();
    VkExtent2D &vkMaxExtent();
    VkExtent2D &vkMinExtent();
    VkSurfaceFormatKHR &vkSurfaceFormat();
    std::vector<std::shared_ptr<SwapchainImage>> &swapchainImages();
    uint32_t imageCount();

    enum class HdrMode { None, HDR10, ScRGB };

    /// Returns true if any HDR format is active (HDR10 or scRGB)
    bool isHDR() const { return hdrMode_ != HdrMode::None; }

    /// Returns true if HDR10 (A2B10G10R10 + ST.2084) is active
    bool isHDR10() const { return hdrMode_ == HdrMode::HDR10; }

    /// Returns true if scRGB (R16G16B16A16_SFLOAT + EXTENDED_SRGB_LINEAR) is active
    bool isScRGB() const { return hdrMode_ == HdrMode::ScRGB; }

    /// Returns true if the swapchain images were created with TRANSFER_SRC usage.
    bool supportsTransferSrc() const { return transferSrcEnabled_; }

    /// Returns true if the current surface supports any HDR swapchain format.
    bool isHDRSupported() const;

    /// Returns true if scRGB format is available on this surface.
    bool isScRGBSupported() const;

  private:
    std::shared_ptr<PhysicalDevice> physicalDevice_;
    std::shared_ptr<Device> device_;
    std::shared_ptr<Window> window_;

    uint32_t imageCount_;
    VkSurfaceFormatKHR surfaceFormat_;
    VkPresentModeKHR presentMode_;
    VkExtent2D extent_;
    VkExtent2D maxExtent_;
    VkExtent2D minExtent_;
    VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
    HdrMode hdrMode_ = HdrMode::None;
    bool hdrActive_ = false;  // legacy compat — mirrors hdrMode_ != None
    bool transferSrcEnabled_ = false;

    std::vector<std::shared_ptr<SwapchainImage>> swapchainImages_;
};
}; // namespace vk
