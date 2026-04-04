#include "vk2_swapchain.hpp"
#include "vk2_device.hpp"
#include "diagnostics/log.hpp"

#include <volk.h>
#include <algorithm>

namespace engine::vk2 {

SwapchainService::~SwapchainService() {
    shutdown();
}

Result<void> SwapchainService::init(DeviceService& device) {
    deviceService_ = &device;
    return createSwapchain(VK_NULL_HANDLE);
}

Result<void> SwapchainService::recreate(uint32_t width, uint32_t height) {
    if (!deviceService_) return makeError(VK_ERROR_INITIALIZATION_FAILED, "Not initialized");

    deviceService_->waitIdle();
    if (width > 0 && height > 0) {
        extent_ = {width, height};
    }

    // Query surface caps to check zero-extent BEFORE destroying anything
    VkSurfaceCapabilitiesKHR surfCaps;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
        deviceService_->physicalDevice(), deviceService_->surface(), &surfCaps);

    VkExtent2D checkExtent = surfCaps.currentExtent;
    if (checkExtent.width == 0 || checkExtent.height == 0) {
        // Still minimized — keep old swapchain alive, just mark zero-extent
        zeroExtent_ = true;
        recreateNeeded_ = true; // Retry next time
        log::debug("vulkan", "Recreate deferred: window still minimized");
        return {};
    }

    auto oldSwapchain = swapchain_;
    auto oldImages = std::move(images_);
    auto oldViews = std::move(imageViews_);

    swapchain_ = VK_NULL_HANDLE;
    images_.clear();
    imageViews_.clear();

    auto r = createSwapchain(oldSwapchain);

    if (r && !zeroExtent_) {
        // Real success — new swapchain created, destroy old
        for (auto v : oldViews) {
            if (v != VK_NULL_HANDLE) vkDestroyImageView(deviceService_->device(), v, nullptr);
        }
        if (oldSwapchain != VK_NULL_HANDLE) {
            vkDestroySwapchainKHR(deviceService_->device(), oldSwapchain, nullptr);
        }
        recreateNeeded_ = false;
    } else if (!r) {
        // Failed — restore old state
        swapchain_ = oldSwapchain;
        images_ = std::move(oldImages);
        imageViews_ = std::move(oldViews);
        log::warn("vulkan", "Swapchain recreate failed, keeping old: " + r.error().message);
    } else {
        // createSwapchain returned success but set zeroExtent (race) — restore old
        swapchain_ = oldSwapchain;
        images_ = std::move(oldImages);
        imageViews_ = std::move(oldViews);
        recreateNeeded_ = true;
    }
    return r;
}

void SwapchainService::shutdown() {
    if (!deviceService_ || !deviceService_->isInitialized()) {
        swapchain_ = VK_NULL_HANDLE;
        return;
    }
    destroyImageViews();
    if (swapchain_ != VK_NULL_HANDLE) {
        vkDestroySwapchainKHR(deviceService_->device(), swapchain_, nullptr);
        swapchain_ = VK_NULL_HANDLE;
    }
}

Result<uint32_t> SwapchainService::acquireNextImage(VkSemaphore signalSemaphore) {
    if (zeroExtent_) return makeError(VK_NOT_READY, "Window minimized (zero extent)");

    uint32_t imageIndex;
    VkResult result = vkAcquireNextImageKHR(
        deviceService_->device(), swapchain_, UINT64_MAX,
        signalSemaphore, VK_NULL_HANDLE, &imageIndex);

    if (result == VK_ERROR_OUT_OF_DATE_KHR) {
        recreateNeeded_ = true;
        return makeError(result, "Swapchain out of date");
    }
    if (result == VK_SUBOPTIMAL_KHR) {
        recreateNeeded_ = true;
        // Still usable this frame — fall through
    }
    if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
        return makeError(result, "vkAcquireNextImageKHR failed");
    }
    return imageIndex;
}

Result<void> SwapchainService::present(uint32_t imageIndex, VkSemaphore waitSemaphore) {
    VkPresentInfoKHR info{};
    info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    info.waitSemaphoreCount = 1;
    info.pWaitSemaphores = &waitSemaphore;
    info.swapchainCount = 1;
    info.pSwapchains = &swapchain_;
    info.pImageIndices = &imageIndex;

    VkResult result = vkQueuePresentKHR(deviceService_->mainQueue().queue, &info);
    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
        recreateNeeded_ = true;
        // Not a fatal error — frame was still presented (or will be on next acquire)
        return {};
    }
    if (result != VK_SUCCESS) {
        return makeError(result, "vkQueuePresentKHR failed");
    }
    return {};
}

Result<void> SwapchainService::createSwapchain(VkSwapchainKHR oldSwapchain) {
    auto device = deviceService_->device();
    auto physDevice = deviceService_->physicalDevice();
    auto surface = deviceService_->surface();

    // Query surface capabilities
    VkSurfaceCapabilitiesKHR caps;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physDevice, surface, &caps);

    // Choose extent
    if (caps.currentExtent.width != UINT32_MAX) {
        extent_ = caps.currentExtent;
    } else {
        extent_.width = std::clamp(extent_.width, caps.minImageExtent.width, caps.maxImageExtent.width);
        extent_.height = std::clamp(extent_.height, caps.minImageExtent.height, caps.maxImageExtent.height);
    }

    // Zero-extent (minimized window) — not an error, just defer
    if (extent_.width == 0 || extent_.height == 0) {
        zeroExtent_ = true;
        log::debug("vulkan", "Zero-extent swapchain (window minimized) — deferring");
        return {};
    }
    zeroExtent_ = false;

    // --- Format selection (preference chain) ---
    uint32_t fmtCount = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(physDevice, surface, &fmtCount, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(fmtCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(physDevice, surface, &fmtCount, formats.data());

    VkSurfaceFormatKHR chosen = formats[0];
    bool found = false;
    // Pref 1: R8G8B8A8_UNORM + sRGB
    for (const auto& f : formats) {
        if (f.format == VK_FORMAT_R8G8B8A8_UNORM && f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            chosen = f; found = true; break;
        }
    }
    // Pref 2: B8G8R8A8_UNORM + sRGB
    if (!found) for (const auto& f : formats) {
        if (f.format == VK_FORMAT_B8G8R8A8_UNORM && f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            chosen = f; found = true; break;
        }
    }
    // Pref 3: Any R8G8B8A8_UNORM
    if (!found) for (const auto& f : formats) {
        if (f.format == VK_FORMAT_R8G8B8A8_UNORM) { chosen = f; found = true; break; }
    }
    // Pref 4: Any B8G8R8A8_UNORM
    if (!found) for (const auto& f : formats) {
        if (f.format == VK_FORMAT_B8G8R8A8_UNORM) { chosen = f; found = true; break; }
    }
    format_ = chosen.format;

    // --- Present mode (FIFO guaranteed available) ---
    presentMode_ = VK_PRESENT_MODE_FIFO_KHR;

    // --- Image count (prefer triple buffering) ---
    uint32_t imageCount = caps.minImageCount + 1;
    imageCount = std::clamp(imageCount, 2u, 3u);
    if (caps.maxImageCount > 0 && imageCount > caps.maxImageCount) {
        imageCount = caps.maxImageCount;
    }

    // --- preTransform: prefer identity, fall back to current ---
    VkSurfaceTransformFlagBitsKHR preTransform;
    if (caps.supportedTransforms & VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR) {
        preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
    } else {
        preTransform = caps.currentTransform;
    }

    // --- compositeAlpha: prefer opaque, fall back to supported ---
    VkCompositeAlphaFlagBitsKHR compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    if (!(caps.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR)) {
        const VkCompositeAlphaFlagBitsKHR fallbacks[] = {
            VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR,
            VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR,
            VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR,
        };
        for (auto fb : fallbacks) {
            if (caps.supportedCompositeAlpha & fb) { compositeAlpha = fb; break; }
        }
    }

    // --- Image usage: intersect desired with supported ---
    VkImageUsageFlags desiredUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                                     VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                     VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    VkImageUsageFlags actualUsage = desiredUsage & caps.supportedUsageFlags;
    if (!(actualUsage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)) {
        return makeError(VK_ERROR_INITIALIZATION_FAILED, "Surface does not support COLOR_ATTACHMENT");
    }
    transferSrcEnabled_ = (actualUsage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) != 0;

    // --- Create swapchain ---
    VkSwapchainCreateInfoKHR createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    createInfo.surface = surface;
    createInfo.minImageCount = imageCount;
    createInfo.imageFormat = chosen.format;
    createInfo.imageColorSpace = chosen.colorSpace;
    createInfo.imageExtent = extent_;
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage = actualUsage;
    createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    createInfo.preTransform = preTransform;
    createInfo.compositeAlpha = compositeAlpha;
    createInfo.presentMode = presentMode_;
    createInfo.clipped = VK_TRUE;
    createInfo.oldSwapchain = oldSwapchain;

    VkResult result = vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapchain_);
    if (result != VK_SUCCESS) {
        return makeError(result, "vkCreateSwapchainKHR failed");
    }

    // Get swapchain images
    uint32_t imgCount = 0;
    vkGetSwapchainImagesKHR(device, swapchain_, &imgCount, nullptr);
    images_.resize(imgCount);
    vkGetSwapchainImagesKHR(device, swapchain_, &imgCount, images_.data());

    // Create image views
    imageViews_.resize(imgCount);
    for (uint32_t i = 0; i < imgCount; ++i) {
        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = images_[i];
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = format_;
        viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = 1;

        result = vkCreateImageView(device, &viewInfo, nullptr, &imageViews_[i]);
        if (result != VK_SUCCESS) {
            return makeError(result, "vkCreateImageView for swapchain image failed");
        }
    }

    log::info("vulkan", "Swapchain created: " + std::to_string(extent_.width) + "x" +
              std::to_string(extent_.height) + " (" + std::to_string(imgCount) + " images)");
    return {};
}

void SwapchainService::destroyImageViews() {
    if (!deviceService_ || !deviceService_->isInitialized()) return;
    for (auto view : imageViews_) {
        if (view != VK_NULL_HANDLE) {
            vkDestroyImageView(deviceService_->device(), view, nullptr);
        }
    }
    imageViews_.clear();
    images_.clear();
}

} // namespace engine::vk2
