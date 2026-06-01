#include "core/vulkan/swapchain.hpp"

#include "core/vulkan/device.hpp"
#include "core/vulkan/debug_utils.hpp"
#include "core/vulkan/image.hpp"
#include "core/vulkan/physical_device.hpp"
#include "core/vulkan/window.hpp"

#include "core/render/renderer.hpp"
#include "core/render/streamline_context.hpp"

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

std::ostream &swapchainCout() {
    return std::cout << "[Swapchain] ";
}

std::ostream &swapchainCerr() {
    return std::cerr << "[Swapchain] ";
}

vk::Swapchain::Swapchain(std::shared_ptr<PhysicalDevice> physicalDevice,
                         std::shared_ptr<Device> device,
                         std::shared_ptr<Window> window)
    : physicalDevice_(physicalDevice), device_(device), window_(window) {
    reconstruct();
}

// SDR surface format selection
VkSurfaceFormatKHR chooseSurfaceFormat(const std::vector<VkSurfaceFormatKHR> &availableFormats) {
    // We can either choose any format
    if (availableFormats.size() == 1 && availableFormats[0].format == VK_FORMAT_UNDEFINED) {
        return {VK_FORMAT_R8G8B8A8_UNORM, VK_COLORSPACE_SRGB_NONLINEAR_KHR};
    }

    // Prefer SDR sRGB colorspace first to avoid washed-out output.
    for (const auto &availableSurfaceFormat : availableFormats) {
        if (availableSurfaceFormat.format == VK_FORMAT_R8G8B8A8_UNORM
            && availableSurfaceFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            return availableSurfaceFormat;
        }
    }

    for (const auto &availableSurfaceFormat : availableFormats) {
        if (availableSurfaceFormat.format == VK_FORMAT_B8G8R8A8_UNORM
            && availableSurfaceFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            return availableSurfaceFormat;
        }
    }

    // Fallback to same formats with any colorspace.
    for (const auto &availableSurfaceFormat : availableFormats) {
        if (availableSurfaceFormat.format == VK_FORMAT_R8G8B8A8_UNORM) { return availableSurfaceFormat; }
    }

    for (const auto &availableSurfaceFormat : availableFormats) {
        if (availableSurfaceFormat.format == VK_FORMAT_B8G8R8A8_UNORM) { return availableSurfaceFormat; }
    }

    // Or fall back to the first available one
    return availableFormats[0];
}

// HDR surface format selection — returns {format, hdrMode} pair
// HDR10 is default (compatible with DLSS-FG). scRGB is optional (breaks DLSS-FG).
// Only called when Renderer::options.hdrEnabled is true
std::tuple<VkSurfaceFormatKHR, vk::Swapchain::HdrMode> chooseHDRSurfaceFormat(
    const std::vector<VkSurfaceFormatKHR> &availableFormats, bool preferScRGB) {

    // If user explicitly requested scRGB and it's available, use it
    if (preferScRGB) {
        for (const auto &fmt : availableFormats) {
            if (fmt.format == VK_FORMAT_R16G16B16A16_SFLOAT &&
                fmt.colorSpace == VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT) {
                swapchainCout() << "scRGB format found: R16G16B16A16_SFLOAT + EXTENDED_SRGB_LINEAR" << std::endl;
                return {fmt, vk::Swapchain::HdrMode::ScRGB};
            }
        }
        swapchainCerr() << "scRGB not available, trying HDR10" << std::endl;
    }

    // Default: HDR10 (A2B10G10R10_UNORM_PACK32 + HDR10_ST2084) — DLSS-FG compatible
    for (const auto &fmt : availableFormats) {
        if (fmt.format == VK_FORMAT_A2B10G10R10_UNORM_PACK32 &&
            fmt.colorSpace == VK_COLOR_SPACE_HDR10_ST2084_EXT) {
            swapchainCout() << "HDR10 format found: A2B10G10R10_UNORM + HDR10_ST2084" << std::endl;
            return {fmt, vk::Swapchain::HdrMode::HDR10};
        }
    }

    // Fallback: try A2R10G10B10 variant (some drivers report this instead)
    for (const auto &fmt : availableFormats) {
        if (fmt.format == VK_FORMAT_A2R10G10B10_UNORM_PACK32 &&
            fmt.colorSpace == VK_COLOR_SPACE_HDR10_ST2084_EXT) {
            swapchainCout() << "HDR10 format found: A2R10G10B10_UNORM + HDR10_ST2084" << std::endl;
            return {fmt, vk::Swapchain::HdrMode::HDR10};
        }
    }

    swapchainCerr() << "No HDR format available, falling back to SDR" << std::endl;
    return {{}, vk::Swapchain::HdrMode::None};
}

VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR &surfaceCapabilities, uint32_t width, uint32_t height) {
    if (surfaceCapabilities.currentExtent.width == -1) {
        VkExtent2D swapChainExtent = {};

        swapChainExtent.width = std::min(std::max(width, surfaceCapabilities.minImageExtent.width),
                                         surfaceCapabilities.maxImageExtent.width);
        swapChainExtent.height = std::min(std::max(height, surfaceCapabilities.minImageExtent.height),
                                          surfaceCapabilities.maxImageExtent.height);
        return swapChainExtent;
    } else {
        return surfaceCapabilities.currentExtent;
    }
}

VkPresentModeKHR choosePresentMode(const std::vector<VkPresentModeKHR> presentModes) {
    if (Renderer::options.vsync) {
        // When Streamline Reflex is available, prefer MAILBOX over FIFO.
        // FIFO blocks the CPU thread at vkQueuePresentKHR until vblank, which:
        //   1. Prevents slReflexSleep from controlling frame pacing
        //   2. Adds unavoidable latency that Reflex can't compensate for
        //   3. Defeats GSYNC's variable refresh rate (display always waits for vblank)
        // MAILBOX doesn't block at present (still no tearing — display flips at vblank)
        // so Reflex can pace frames via sleep and GSYNC can adapt the refresh rate.
        if (StreamlineContext::isReflexAvailable()) {
            for (const auto &presentMode : presentModes) {
                if (presentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
                    swapchainCout() << "vsync+Reflex: using MAILBOX for Reflex frame pacing" << std::endl;
                    return presentMode;
                }
            }
            swapchainCout() << "vsync+Reflex: MAILBOX unavailable, falling back to FIFO" << std::endl;
        }
        return VK_PRESENT_MODE_FIFO_KHR;
    }

    for (const auto &presentMode : presentModes) {
        if (presentMode == VK_PRESENT_MODE_IMMEDIATE_KHR) { return presentMode; }
    }

    // If immediate is unavailable, fall back to FIFO (guaranteed to be available)
    return VK_PRESENT_MODE_FIFO_KHR;
}

void vk::Swapchain::reconstruct() {
    // Find surface capabilities
    VkSurfaceCapabilitiesKHR surfaceCapabilities;
    if (vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice_->vkPhysicalDevice(), window_->vkSurface(),
                                                  &surfaceCapabilities) != VK_SUCCESS) {
        swapchainCerr() << "failed to acquire presentation surface capabilities" << std::endl;
        exit(EXIT_FAILURE);
    }

    maxExtent_ = surfaceCapabilities.maxImageExtent;
    minExtent_ = surfaceCapabilities.minImageExtent;

    // Find supported surface formats
    uint32_t formatCount;
    if (vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice_->vkPhysicalDevice(), window_->vkSurface(), &formatCount,
                                             nullptr) != VK_SUCCESS ||
        formatCount == 0) {
        swapchainCerr() << "failed to get number of supported surface formats" << std::endl;
        exit(EXIT_FAILURE);
    }

    std::vector<VkSurfaceFormatKHR> surfaceFormats(formatCount);
    if (vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice_->vkPhysicalDevice(), window_->vkSurface(), &formatCount,
                                             surfaceFormats.data()) != VK_SUCCESS) {
        swapchainCerr() << "failed to get supported surface formats" << std::endl;
        exit(EXIT_FAILURE);
    }

// Select a surface format
#ifdef DEBUG
    for (int i = 0; i < formatCount; i++) {
        swapchainCout() << "Supported Format: " << surfaceFormats[i].format
                        << " ColorSpace: " << surfaceFormats[i].colorSpace << std::endl;
    }
#endif
    // HDR format selection (when enabled), else SDR (existing path)
    hdrMode_ = HdrMode::None;
    hdrActive_ = false;
    if (Renderer::options.hdrEnabled) {
        bool preferScRGB = false;
        if (Renderer::options.hdrScrgbMode) {
            swapchainCerr() << "scRGB requested but currently disabled; falling back to HDR10" << std::endl;
        }
        auto [hdrFormat, mode] = chooseHDRSurfaceFormat(surfaceFormats, preferScRGB);
        if (mode != HdrMode::None) {
            surfaceFormat_ = hdrFormat;
            hdrMode_ = mode;
            hdrActive_ = true;
        } else {
            surfaceFormat_ = chooseSurfaceFormat(surfaceFormats);
        }
    } else {
        surfaceFormat_ = chooseSurfaceFormat(surfaceFormats);
    }

    swapchainCout() << "Selected surface format=" << surfaceFormat_.format
                    << " colorSpace=" << surfaceFormat_.colorSpace
                    << " hdrActive=" << (hdrActive_ ? 1 : 0) << std::endl;

    // Find supported present modes
    uint32_t presentModeCount;
    if (vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice_->vkPhysicalDevice(), window_->vkSurface(),
                                                  &presentModeCount, nullptr) != VK_SUCCESS ||
        presentModeCount == 0) {
        swapchainCerr() << "failed to get number of supported presentation modes" << std::endl;
        exit(EXIT_FAILURE);
    }

    std::vector<VkPresentModeKHR> presentModes(presentModeCount);
    if (vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice_->vkPhysicalDevice(), window_->vkSurface(),
                                                  &presentModeCount, presentModes.data()) != VK_SUCCESS) {
        swapchainCerr() << "failed to get supported presentation modes" << std::endl;
        exit(EXIT_FAILURE);
    }

    // Choose presentation mode (preferring MAILBOX ~= triple buffering)
    presentMode_ = choosePresentMode(presentModes);

    // Determine number of images for swap chain.
    // Default: 3 images (minImageCount+1, clamped to 3).
    // 4 images can help MAILBOX pacing on some GPUs but introduces extra latency
    // and is not needed when a native FPS limiter provides frame pacing.
    // Keep 4-image mode behind an explicit override for testing only.
    imageCount_ = surfaceCapabilities.minImageCount + 1;
    uint32_t maxImages = 3;  // Default to 3 - native limiter handles pacing
    imageCount_ = std::clamp(imageCount_, (uint32_t)2, maxImages);
    if (surfaceCapabilities.maxImageCount != 0 && imageCount_ > surfaceCapabilities.maxImageCount) {
        imageCount_ = surfaceCapabilities.maxImageCount;
    }

    swapchainCout() << "using " << imageCount_ << " images for swap chain (present mode "
                    << (presentMode_ == VK_PRESENT_MODE_MAILBOX_KHR ? "MAILBOX" :
                        presentMode_ == VK_PRESENT_MODE_FIFO_KHR ? "FIFO" : "other") << ")" << std::endl;

    // Select swap chain size
    extent_ = chooseSwapExtent(surfaceCapabilities, window_->width(), window_->height());

    // Determine transformation to use (preferring no transform)
    VkSurfaceTransformFlagBitsKHR surfaceTransform;
    if (surfaceCapabilities.supportedTransforms & VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR) {
        surfaceTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
    } else {
        surfaceTransform = surfaceCapabilities.currentTransform;
    }

    // Finally, create the swap chain
    VkSwapchainKHR oldSwapchain = swapchain_;

    VkSwapchainCreateInfoKHR createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    createInfo.surface = window_->vkSurface();
    createInfo.minImageCount = imageCount_;
    createInfo.imageFormat = surfaceFormat_.format;
    createInfo.imageColorSpace = surfaceFormat_.colorSpace;
    createInfo.imageExtent = extent_;
    createInfo.imageArrayLayers = 1;

    VkImageUsageFlags desiredUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                                    VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                    VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                    VK_IMAGE_USAGE_SAMPLED_BIT;
    VkImageUsageFlags supportedUsage = surfaceCapabilities.supportedUsageFlags;
    VkImageUsageFlags imageUsage = desiredUsage & supportedUsage;

    // Color attachment is non-negotiable for the final composite path.
    if ((imageUsage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) == 0) {
        swapchainCerr() << "surface does not support COLOR_ATTACHMENT usage for swapchain" << std::endl;
        exit(EXIT_FAILURE);
    }

    if ((desiredUsage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) != 0 &&
        (imageUsage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) == 0) {
        swapchainCerr() << "swapchain does not support TRANSFER_SRC; with-UI screenshots will be disabled" << std::endl;
    }

    transferSrcEnabled_ = (imageUsage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) != 0;

    createInfo.imageUsage = imageUsage;
    createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    createInfo.queueFamilyIndexCount = 0;
    createInfo.pQueueFamilyIndices = nullptr;
    createInfo.preTransform = surfaceTransform;
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    createInfo.presentMode = presentMode_;
    createInfo.clipped = VK_TRUE;
    // Normal Vulkan swapchain transition: pass old swapchain for atomic replacement.
    // DLSS-G feature is always unloaded in beforeSwapchainRecreate(), so Streamline's
    // tracker is clean — no "only one swap-chain" rejection.
    createInfo.oldSwapchain = oldSwapchain;

    VkResult swapResult = vkCreateSwapchainKHR(device_->vkDevice(), &createInfo, nullptr, &swapchain_);
    if (swapResult != VK_SUCCESS) {
        // DLSS-G hooks may reject non-FIFO modes during recreation. Retry with FIFO (guaranteed).
        if (presentMode_ != VK_PRESENT_MODE_FIFO_KHR) {
            swapchainCerr() << "swap chain creation failed with mode " << presentMode_
                            << " (result=" << swapResult << "), retrying with FIFO" << std::endl;
            presentMode_ = VK_PRESENT_MODE_FIFO_KHR;
            createInfo.presentMode = VK_PRESENT_MODE_FIFO_KHR;
            swapResult = vkCreateSwapchainKHR(device_->vkDevice(), &createInfo, nullptr, &swapchain_);
        }
        if (swapResult != VK_SUCCESS) {
            swapchainCerr() << "failed to create swap chain (result=" << swapResult << ")" << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    if (oldSwapchain != VK_NULL_HANDLE) { vkDestroySwapchainKHR(device_->vkDevice(), oldSwapchain, nullptr); }

    vk::DebugUtils::setObjectName(device_->vkDevice(), VK_OBJECT_TYPE_SWAPCHAIN_KHR,
                                  swapchain_, "Radiance Swapchain");

    // Set HDR metadata when HDR10 is active (SMPTE ST.2086 mastering display + CTA 861.3 content light)
    if (hdrActive_ && vkSetHdrMetadataEXT) {
        VkHdrMetadataEXT hdrMetadata = {};
        hdrMetadata.sType = VK_STRUCTURE_TYPE_HDR_METADATA_EXT;
        // BT.2020 display primaries
        hdrMetadata.displayPrimaryRed   = {0.708f, 0.292f};
        hdrMetadata.displayPrimaryGreen = {0.170f, 0.797f};
        hdrMetadata.displayPrimaryBlue  = {0.131f, 0.046f};
        hdrMetadata.whitePoint          = {0.3127f, 0.3290f};  // D65
        hdrMetadata.maxLuminance        = Renderer::options.hdrPeakNits;
        hdrMetadata.minLuminance        = 0.0001f;
        hdrMetadata.maxContentLightLevel     = static_cast<float>(Renderer::options.hdrPeakNits);
        hdrMetadata.maxFrameAverageLightLevel = 200.0f;

        vkSetHdrMetadataEXT(device_->vkDevice(), 1, &swapchain_, &hdrMetadata);
        swapchainCout() << "HDR metadata set: peak=" << Renderer::options.hdrPeakNits
                        << " nits, paper white=" << Renderer::options.hdrPaperWhiteNits << " nits" << std::endl;
    }

    // Store the images used by the swap chain
    // Note: these are the images that swap chain image indices refer to
    // Note: actual number of images may differ from requested number, since it's a lower bound
    uint32_t actualImageCount = 0;
    if (vkGetSwapchainImagesKHR(device_->vkDevice(), swapchain_, &actualImageCount, nullptr) != VK_SUCCESS ||
        actualImageCount == 0) {
        swapchainCerr() << "failed to acquire number of swap chain images" << std::endl;
        exit(EXIT_FAILURE);
    }
#ifdef DEBUG
    std::cout << "actualImageCount: " << actualImageCount << std::endl;
#endif
    imageCount_ = actualImageCount;

    std::vector<VkImage> images(actualImageCount);
    if (vkGetSwapchainImagesKHR(device_->vkDevice(), swapchain_, &actualImageCount, images.data()) != VK_SUCCESS) {
        swapchainCerr() << "failed to acquire swap chain images" << std::endl;
        exit(EXIT_FAILURE);
    }
    swapchainImages_.clear();
    for (int i = 0; i < actualImageCount; i++) {
        swapchainImages_.push_back(
            SwapchainImage::create(device_, images[i], extent_.width, extent_.height, surfaceFormat_.format));
        vk::DebugUtils::setObjectName(device_->vkDevice(), VK_OBJECT_TYPE_IMAGE,
                                      images[i], "Radiance SwapchainImage[" + std::to_string(i) + "]");
        vk::DebugUtils::setObjectName(device_->vkDevice(), VK_OBJECT_TYPE_IMAGE_VIEW,
                                      swapchainImages_.back()->vkImageView(),
                                      "Radiance SwapchainImageView[" + std::to_string(i) + "]");
    }

#ifdef DEBUG
    swapchainCout() << "acquired swap chain images" << std::endl;
#endif

}

vk::Swapchain::~Swapchain() {
    vkDestroySwapchainKHR(device_->vkDevice(), swapchain_, nullptr);

#ifdef DEBUG
    swapchainCout() << "swapchain deconstructed" << std::endl;
#endif
}

VkSwapchainKHR &vk::Swapchain::vkSwapchain() {
    return swapchain_;
}

VkExtent2D &vk::Swapchain::vkExtent() {
    return extent_;
}

VkExtent2D &vk::Swapchain::vkMaxExtent() {
    return maxExtent_;
}

VkExtent2D &vk::Swapchain::vkMinExtent() {
    return minExtent_;
}

VkSurfaceFormatKHR &vk::Swapchain::vkSurfaceFormat() {
    return surfaceFormat_;
}

std::vector<std::shared_ptr<vk::SwapchainImage>> &vk::Swapchain::swapchainImages() {
    return swapchainImages_;
}

uint32_t vk::Swapchain::imageCount() {
    return imageCount_;
}

bool vk::Swapchain::isHDRSupported() const {
    uint32_t formatCount = 0;
    if (vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice_->vkPhysicalDevice(), window_->vkSurface(),
                                             &formatCount, nullptr) != VK_SUCCESS
        || formatCount == 0) {
        return false;
    }

    std::vector<VkSurfaceFormatKHR> surfaceFormats(formatCount);
    if (vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice_->vkPhysicalDevice(), window_->vkSurface(),
                                             &formatCount, surfaceFormats.data()) != VK_SUCCESS) {
        return false;
    }

    for (const auto &fmt : surfaceFormats) {
        // HDR10
        if ((fmt.format == VK_FORMAT_A2B10G10R10_UNORM_PACK32 || fmt.format == VK_FORMAT_A2R10G10B10_UNORM_PACK32)
            && fmt.colorSpace == VK_COLOR_SPACE_HDR10_ST2084_EXT) {
            return true;
        }
        // scRGB
        if (fmt.format == VK_FORMAT_R16G16B16A16_SFLOAT
            && fmt.colorSpace == VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT) {
            return true;
        }
    }

    return false;
}

bool vk::Swapchain::isScRGBSupported() const {
    uint32_t formatCount = 0;
    if (vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice_->vkPhysicalDevice(), window_->vkSurface(),
                                             &formatCount, nullptr) != VK_SUCCESS
        || formatCount == 0) {
        return false;
    }

    std::vector<VkSurfaceFormatKHR> surfaceFormats(formatCount);
    if (vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice_->vkPhysicalDevice(), window_->vkSurface(),
                                             &formatCount, surfaceFormats.data()) != VK_SUCCESS) {
        return false;
    }

    for (const auto &fmt : surfaceFormats) {
        if (fmt.format == VK_FORMAT_R16G16B16A16_SFLOAT
            && fmt.colorSpace == VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT) {
            return true;
        }
    }
    return false;
}
