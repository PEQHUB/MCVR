#pragma once

// Ownership: Shared (shared_ptr). Destroyed when last reference drops.
// Thread: Creation on any thread with VMA. Layout tracking is caller's responsibility.
// Dependencies: VMA allocator, Vulkan device.

#include "vk2_result.hpp"

#include <cstdint>
#include <memory>
#include <vector>
#include <vk_mem_alloc.h>

namespace engine::vk2 {

// RAII Vulkan image + view backed by VMA. No partial objects.
class Image {
public:
    ~Image();

    Image(const Image&) = delete;
    Image& operator=(const Image&) = delete;
    Image(Image&& other) noexcept;
    Image& operator=(Image&& other) noexcept;

    // --- Creation ---

    struct Desc {
        uint32_t width = 0;
        uint32_t height = 0;
        uint32_t depth = 1;                          // >1 for 3D images
        uint32_t layers = 1;                         // >1 for array images
        uint32_t mipLevels = 1;
        VkFormat format = VK_FORMAT_UNDEFINED;
        VkImageUsageFlags usage = 0;                 // Caller specifies all flags
        VkImageType imageType = VK_IMAGE_TYPE_2D;
        VkImageCreateFlags createFlags = 0;
        VmaAllocationCreateFlags vmaFlags = 0;
        VmaMemoryUsage memoryUsage = VMA_MEMORY_USAGE_AUTO;
    };

    // Create an image with a default image view. Returns Error on failure.
    static Result<Image> create(VkDevice device, VmaAllocator allocator, const Desc& desc);

    // --- Accessors ---

    VkImage handle() const { return image_; }
    VkImageView defaultView() const { return views_.empty() ? VK_NULL_HANDLE : views_[0]; }
    VkFormat format() const { return format_; }
    uint32_t width() const { return width_; }
    uint32_t height() const { return height_; }
    uint32_t depth() const { return depth_; }
    uint32_t layers() const { return layers_; }

    // Layout tracking (caller must keep in sync with actual transitions)
    VkImageLayout layout() const { return layout_; }
    void setLayout(VkImageLayout l) { layout_ = l; }

    // Add an additional image view (e.g., per-mip or per-layer view).
    // Returns the index into the views array.
    Result<uint32_t> addView(VkDevice device, const VkImageViewCreateInfo& viewInfo);

    Image() = default;

    VkImageView view(uint32_t index) const { return views_[index]; }

private:

    VkDevice device_ = VK_NULL_HANDLE;
    VmaAllocator allocator_ = VK_NULL_HANDLE;
    VkImage image_ = VK_NULL_HANDLE;
    VmaAllocation allocation_ = VK_NULL_HANDLE;
    std::vector<VkImageView> views_;
    VkFormat format_ = VK_FORMAT_UNDEFINED;
    VkImageLayout layout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    uint32_t width_ = 0, height_ = 0, depth_ = 1, layers_ = 1;

    void destroy();
};

} // namespace engine::vk2
