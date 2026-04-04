#pragma once

// Ownership: FrameScheduler (1:1).
// Thread: Main thread only.
// Usage: requestCapture() sets pending. recordCopy() inserts GPU commands.
//        saveIfReady() maps readback buffer and writes PNG after fence wait.

#include <cstdint>
#include <string>

#include <volk.h>
#include <vk_mem_alloc.h>

namespace engine {

namespace vk2 { class DeviceService; }

class ScreenshotCapture {
public:
    void requestCapture(const std::string& outputPath);
    bool isPending() const { return pending_; }

    // Insert copy from swapchain image to readback buffer.
    // Call AFTER rendering, BEFORE present-transition barrier.
    // sourceLayout: current layout of the image (e.g., COLOR_ATTACHMENT_OPTIMAL or TRANSFER_DST_OPTIMAL).
    void recordCopy(VkCommandBuffer cmd, VkImage swapImage, VkFormat format,
                    VkExtent2D extent, VkImageLayout sourceLayout,
                    vk2::DeviceService& device);

    // After fence wait: map buffer, write PNG. Resets pending state.
    void saveIfReady(vk2::DeviceService& device);

    void shutdown(vk2::DeviceService& device);

private:
    bool pending_ = false;
    bool readbackReady_ = false;
    std::string outputPath_;
    VkBuffer readbackBuffer_ = VK_NULL_HANDLE;
    VmaAllocation readbackAlloc_ = VK_NULL_HANDLE;
    uint32_t width_ = 0;
    uint32_t height_ = 0;
    VkFormat format_ = VK_FORMAT_UNDEFINED;
};

} // namespace engine
