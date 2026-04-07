#include "offscreen_target.hpp"
#include "platform/vulkan/vk2_device.hpp"
#include "diagnostics/log.hpp"

namespace engine {

vk2::Result<void> OffscreenTarget::init(vk2::DeviceService& device, uint32_t framesInFlight,
                                        uint32_t width, uint32_t height) {
    framesInFlight_ = framesInFlight;
    width_ = width;
    height_ = height;
    return createImages(device);
}

vk2::Result<void> OffscreenTarget::recreate(vk2::DeviceService& device,
                                            uint32_t width, uint32_t height) {
    if (width == width_ && height == height_) return {};

    width_ = width;
    height_ = height;
    images_.clear();
    return createImages(device);
}

void OffscreenTarget::shutdown() {
    images_.clear();
    width_ = 0;
    height_ = 0;
}

vk2::Result<void> OffscreenTarget::createImages(vk2::DeviceService& device) {
    images_.clear();
    images_.reserve(framesInFlight_);

    for (uint32_t i = 0; i < framesInFlight_; ++i) {
        vk2::Image::Desc desc{};
        desc.width = width_;
        desc.height = height_;
        desc.format = VK_FORMAT_R16G16B16A16_SFLOAT;
        desc.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
                   | VK_IMAGE_USAGE_TRANSFER_SRC_BIT
                   | VK_IMAGE_USAGE_TRANSFER_DST_BIT
                   | VK_IMAGE_USAGE_STORAGE_BIT
                   | VK_IMAGE_USAGE_SAMPLED_BIT;

        auto result = vk2::Image::create(device.device(), device.vma(), desc);
        if (!result) return vk2::makeError(result.error().vkResult,
                                           "OffscreenTarget image " + std::to_string(i) + " failed");
        images_.push_back(std::move(result.value()));
    }

    log::info("offscreen", "Created " + std::to_string(framesInFlight_) +
              " offscreen targets: " + std::to_string(width_) + "x" + std::to_string(height_) +
              " R16G16B16A16_SFLOAT");
    return {};
}

} // namespace engine
