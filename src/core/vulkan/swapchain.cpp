#include "core/vulkan/swapchain.hpp"

#include "core/vulkan/device.hpp"
#include "core/vulkan/image.hpp"
#include "core/vulkan/physical_device.hpp"
#include "core/vulkan/window.hpp"

#include "core/render/renderer.hpp"

#include <algorithm>
#include <iostream>
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

// SDR surface format selection (existing logic, untouched)
VkSurfaceFormatKHR chooseSurfaceFormat(const std::vector<VkSurfaceFormatKHR> &availableFormats) {
    // We can either choose any format
    if (availableFormats.size() == 1 && availableFormats[0].format == VK_FORMAT_UNDEFINED) {
        return {VK_FORMAT_R8G8B8A8_UNORM, VK_COLORSPACE_SRGB_NONLINEAR_KHR};
    }

    // Or go with the standard format - if available
    for (const auto &availableSurfaceFormat : availableFormats) {
        if (availableSurfaceFormat.format == VK_FORMAT_R8G8B8A8_UNORM) { return availableSurfaceFormat; }
    }

    // Or fall back to the first available one
    return availableFormats[0];
}

// HDR10 surface format selection — returns {format, found} pair
// Only called when Renderer::options.hdrEnabled is true
std::pair<VkSurfaceFormatKHR, bool> chooseHDRSurfaceFormat(const std::vector<VkSurfaceFormatKHR> &availableFormats) {
    // Look for A2B10G10R10_UNORM_PACK32 + HDR10_ST2084
    for (const auto &fmt : availableFormats) {
        if (fmt.format == VK_FORMAT_A2B10G10R10_UNORM_PACK32 &&
            fmt.colorSpace == VK_COLOR_SPACE_HDR10_ST2084_EXT) {
            swapchainCout() << "HDR10 format found: A2B10G10R10_UNORM + HDR10_ST2084" << std::endl;
            return {fmt, true};
        }
    }

    // Fallback: try A2R10G10B10 variant (some drivers report this instead)
    for (const auto &fmt : availableFormats) {
        if (fmt.format == VK_FORMAT_A2R10G10B10_UNORM_PACK32 &&
            fmt.colorSpace == VK_COLOR_SPACE_HDR10_ST2084_EXT) {
            swapchainCout() << "HDR10 format found: A2R10G10B10_UNORM + HDR10_ST2084" << std::endl;
            return {fmt, true};
        }
    }

    swapchainCerr() << "HDR10 format not available, falling back to SDR" << std::endl;
    return {{}, false};
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
    if (Renderer::options.vsync) { return VK_PRESENT_MODE_FIFO_KHR; }

    for (const auto &presentMode : presentModes) {
        if (presentMode == VK_PRESENT_MODE_IMMEDIATE_KHR) { return presentMode; }
    }

    // If mailbox is unavailable, fall back to FIFO (guaranteed to be available)
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

    // Determine number of images for swap chain
    imageCount_ = surfaceCapabilities.minImageCount + 1;
    imageCount_ = std::clamp(imageCount_, (uint32_t)2, (uint32_t)3);
    if (surfaceCapabilities.maxImageCount != 0 && imageCount_ > surfaceCapabilities.maxImageCount) {
        imageCount_ = surfaceCapabilities.maxImageCount;
    }

#ifdef DEBUG
    swapchainCout() << "using " << imageCount_ << " images for swap chain" << std::endl;
#endif

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
    // HDR10 format selection (when enabled and supported), else SDR (existing path)
    hdrActive_ = false;
    if (Renderer::options.hdrEnabled) {
        auto [hdrFormat, found] = chooseHDRSurfaceFormat(surfaceFormats);
        if (found) {
            surfaceFormat_ = hdrFormat;
            hdrActive_ = true;
        } else {
            surfaceFormat_ = chooseSurfaceFormat(surfaceFormats);
        }
    } else {
        surfaceFormat_ = chooseSurfaceFormat(surfaceFormats);
    }

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
    createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                            VK_IMAGE_USAGE_SAMPLED_BIT; // TODO: cancel VK_IMAGE_USAGE_SAMPLED_BIT
    createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    createInfo.queueFamilyIndexCount = 0;
    createInfo.pQueueFamilyIndices = nullptr;
    createInfo.preTransform = surfaceTransform;
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    createInfo.presentMode = presentMode_;
    createInfo.clipped = VK_TRUE;
    createInfo.oldSwapchain = oldSwapchain;

    if (vkCreateSwapchainKHR(device_->vkDevice(), &createInfo, nullptr, &swapchain_) != VK_SUCCESS) {
        swapchainCerr() << "failed to create swap chain" << std::endl;
        exit(EXIT_FAILURE);
    } else {
#ifdef DEBUG
        swapchainCout() << "created swap chain" << std::endl;
#endif
    }

    if (oldSwapchain != VK_NULL_HANDLE) { vkDestroySwapchainKHR(device_->vkDevice(), oldSwapchain, nullptr); }

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