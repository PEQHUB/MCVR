#include "screenshot_capture.hpp"
#include "platform/vulkan/vk2_device.hpp"
#include "diagnostics/log.hpp"

#include <volk.h>
#include <fstream>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

namespace engine {

void ScreenshotCapture::requestCapture(const std::string& outputPath) {
    outputPath_ = outputPath;
    pending_ = true;
    readbackReady_ = false;
    log::info("screenshot", "Capture requested: " + outputPath);
}

void ScreenshotCapture::recordCopy(VkCommandBuffer cmd, VkImage swapImage, VkFormat format,
                                    VkExtent2D extent, VkImageLayout sourceLayout,
                                    vk2::DeviceService& device) {
    if (!pending_ || readbackReady_) return;

    width_ = extent.width;
    height_ = extent.height;
    format_ = format;

    // Create readback buffer if needed
    VkDeviceSize bufferSize = static_cast<VkDeviceSize>(width_) * height_ * 4; // RGBA8

    if (readbackBuffer_ != VK_NULL_HANDLE) {
        vmaDestroyBuffer(device.vma(), readbackBuffer_, readbackAlloc_);
        readbackBuffer_ = VK_NULL_HANDLE;
    }

    VkBufferCreateInfo bufInfo{};
    bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufInfo.size = bufferSize;
    bufInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = VMA_MEMORY_USAGE_GPU_TO_CPU;

    VkResult r = vmaCreateBuffer(device.vma(), &bufInfo, &allocInfo,
                                  &readbackBuffer_, &readbackAlloc_, nullptr);
    if (r != VK_SUCCESS) {
        log::error("screenshot", "Failed to create readback buffer");
        pending_ = false;
        return;
    }

    // Transition swapchain image to TRANSFER_SRC
    VkImageMemoryBarrier2 toSrc{};
    toSrc.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    toSrc.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    toSrc.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    toSrc.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    toSrc.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
    toSrc.oldLayout = sourceLayout;
    toSrc.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    toSrc.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toSrc.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toSrc.image = swapImage;
    toSrc.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    VkDependencyInfo dep{};
    dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.imageMemoryBarrierCount = 1;
    dep.pImageMemoryBarriers = &toSrc;
    vkCmdPipelineBarrier2(cmd, &dep);

    // Copy image to buffer
    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageOffset = {0, 0, 0};
    region.imageExtent = {width_, height_, 1};

    vkCmdCopyImageToBuffer(cmd, swapImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                            readbackBuffer_, 1, &region);

    // Transition back to PRESENT_SRC (the present barrier will handle this)
    // Actually, we leave it in TRANSFER_SRC and let the caller's present barrier handle it.
    // The caller needs to know we changed the layout — for now, keep sourceLayout tracking simple
    // by transitioning back here:
    VkImageMemoryBarrier2 toPresent{};
    toPresent.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    toPresent.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    toPresent.srcAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
    toPresent.dstStageMask = VK_PIPELINE_STAGE_2_NONE;
    toPresent.dstAccessMask = VK_ACCESS_2_NONE;
    toPresent.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    toPresent.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    toPresent.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toPresent.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toPresent.image = swapImage;
    toPresent.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    dep.pImageMemoryBarriers = &toPresent;
    vkCmdPipelineBarrier2(cmd, &dep);

    readbackReady_ = true;
}

void ScreenshotCapture::saveIfReady(vk2::DeviceService& device) {
    if (!readbackReady_ || readbackBuffer_ == VK_NULL_HANDLE) return;

    void* data = nullptr;
    VkResult r = vmaMapMemory(device.vma(), readbackAlloc_, &data);
    if (r != VK_SUCCESS || !data) {
        log::error("screenshot", "Failed to map readback buffer");
        pending_ = false;
        readbackReady_ = false;
        return;
    }

    // Write PNG (RGBA8 assumed — swapchain is typically B8G8R8A8 so we swap channels)
    bool isBGR = (format_ == VK_FORMAT_B8G8R8A8_UNORM || format_ == VK_FORMAT_B8G8R8A8_SRGB);
    if (isBGR) {
        uint8_t* pixels = static_cast<uint8_t*>(data);
        for (uint32_t i = 0; i < width_ * height_; ++i) {
            std::swap(pixels[i * 4 + 0], pixels[i * 4 + 2]); // B<->R
        }
    }

    int ok = stbi_write_png(outputPath_.c_str(), width_, height_, 4,
                             data, width_ * 4);

    vmaUnmapMemory(device.vma(), readbackAlloc_);

    if (ok) {
        log::info("screenshot", "Saved: " + outputPath_ + " (" +
                  std::to_string(width_) + "x" + std::to_string(height_) + ")");
    } else {
        log::error("screenshot", "stbi_write_png failed for: " + outputPath_);
    }

    pending_ = false;
    readbackReady_ = false;
}

void ScreenshotCapture::shutdown(vk2::DeviceService& device) {
    if (readbackBuffer_ != VK_NULL_HANDLE) {
        vmaDestroyBuffer(device.vma(), readbackBuffer_, readbackAlloc_);
        readbackBuffer_ = VK_NULL_HANDLE;
        readbackAlloc_ = VK_NULL_HANDLE;
    }
}

} // namespace engine
