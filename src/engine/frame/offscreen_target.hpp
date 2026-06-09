#pragma once

// Ownership: EngineServices (1:1).
// Thread: Main thread only.
// Dependencies: DeviceService (VkDevice, VMA).
//
// Owns one R16G16B16A16_SFLOAT image per frame-in-flight.
// Used as the intermediate render target before composite-to-swapchain blit.
// Recreated when swapchain extent changes.

#include "platform/vulkan/vk2_image.hpp"
#include "platform/vulkan/vk2_result.hpp"

#include <cstdint>
#include <vector>

namespace engine {

namespace vk2 { class DeviceService; }

class OffscreenTarget {
public:
    OffscreenTarget() = default;
    ~OffscreenTarget() = default;

    OffscreenTarget(const OffscreenTarget&) = delete;
    OffscreenTarget& operator=(const OffscreenTarget&) = delete;

    vk2::Result<void> init(vk2::DeviceService& device, uint32_t framesInFlight,
                           uint32_t width, uint32_t height);
    vk2::Result<void> recreate(vk2::DeviceService& device, uint32_t width, uint32_t height);
    void shutdown();

    bool isInitialized() const { return !images_.empty(); }

    vk2::Image& image(uint32_t frameIndex) { return images_[frameIndex]; }
    const vk2::Image& image(uint32_t frameIndex) const { return images_[frameIndex]; }

    VkFormat format() const { return VK_FORMAT_R16G16B16A16_SFLOAT; }
    uint32_t width() const { return width_; }
    uint32_t height() const { return height_; }

private:
    vk2::Result<void> createImages(vk2::DeviceService& device);

    std::vector<vk2::Image> images_;
    uint32_t framesInFlight_ = 0;
    uint32_t width_ = 0;
    uint32_t height_ = 0;
};

} // namespace engine
