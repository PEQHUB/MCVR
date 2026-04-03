#include "vk2_image.hpp"
#include "diagnostics/log.hpp"

namespace engine::vk2 {

Image::~Image() {
    destroy();
}

Image::Image(Image&& other) noexcept
    : device_(other.device_), allocator_(other.allocator_),
      image_(other.image_), allocation_(other.allocation_),
      views_(std::move(other.views_)), format_(other.format_),
      layout_(other.layout_), width_(other.width_), height_(other.height_),
      depth_(other.depth_), layers_(other.layers_) {
    other.image_ = VK_NULL_HANDLE;
    other.allocation_ = VK_NULL_HANDLE;
}

Image& Image::operator=(Image&& other) noexcept {
    if (this != &other) {
        destroy();
        device_ = other.device_;
        allocator_ = other.allocator_;
        image_ = other.image_;
        allocation_ = other.allocation_;
        views_ = std::move(other.views_);
        format_ = other.format_;
        layout_ = other.layout_;
        width_ = other.width_;
        height_ = other.height_;
        depth_ = other.depth_;
        layers_ = other.layers_;
        other.image_ = VK_NULL_HANDLE;
        other.allocation_ = VK_NULL_HANDLE;
    }
    return *this;
}

Result<Image> Image::create(VkDevice device, VmaAllocator allocator, const Desc& desc) {
    if (desc.width == 0 || desc.height == 0) {
        return makeError(VK_ERROR_INITIALIZATION_FAILED, "Image dimensions must be > 0");
    }
    if (desc.format == VK_FORMAT_UNDEFINED) {
        return makeError(VK_ERROR_INITIALIZATION_FAILED, "Image format must not be VK_FORMAT_UNDEFINED");
    }

    bool is3D = (desc.imageType == VK_IMAGE_TYPE_3D);

    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.flags = desc.createFlags;
    imageInfo.imageType = desc.imageType;
    imageInfo.format = desc.format;
    imageInfo.extent = is3D
        ? VkExtent3D{desc.width, desc.height, desc.depth}
        : VkExtent3D{desc.width, desc.height, 1};
    imageInfo.mipLevels = desc.mipLevels;
    imageInfo.arrayLayers = is3D ? 1 : desc.layers;
    imageInfo.usage = desc.usage;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;

    VmaAllocationCreateInfo allocCreateInfo{};
    allocCreateInfo.flags = desc.vmaFlags;
    allocCreateInfo.usage = desc.memoryUsage;

    VkImage vkImage = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    VmaAllocationInfo allocInfo{};

    VkResult result = vmaCreateImage(
        allocator, &imageInfo, &allocCreateInfo,
        &vkImage, &allocation, &allocInfo);

    if (result != VK_SUCCESS) {
        return makeError(result, "vmaCreateImage failed");
    }

    // Create default image view
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = vkImage;
    viewInfo.format = desc.format;

    if (is3D) {
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_3D;
    } else if (desc.layers > 1) {
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    } else {
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    }

    bool isDepth = (desc.usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) != 0;
    viewInfo.subresourceRange.aspectMask = isDepth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = desc.mipLevels;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = is3D ? 1 : desc.layers;

    VkImageView view = VK_NULL_HANDLE;
    result = vkCreateImageView(device, &viewInfo, nullptr, &view);
    if (result != VK_SUCCESS) {
        // Clean up the image we already created — no partial objects
        vmaDestroyImage(allocator, vkImage, allocation);
        return makeError(result, "vkCreateImageView failed for default view");
    }

    Image img;
    img.device_ = device;
    img.allocator_ = allocator;
    img.image_ = vkImage;
    img.allocation_ = allocation;
    img.views_.push_back(view);
    img.format_ = desc.format;
    img.layout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    img.width_ = desc.width;
    img.height_ = desc.height;
    img.depth_ = desc.depth;
    img.layers_ = desc.layers;

    return img;
}

Result<uint32_t> Image::addView(VkDevice device, const VkImageViewCreateInfo& viewInfo) {
    VkImageView view = VK_NULL_HANDLE;
    VkResult result = vkCreateImageView(device, &viewInfo, nullptr, &view);
    if (result != VK_SUCCESS) {
        return makeError(result, "vkCreateImageView failed");
    }
    uint32_t index = static_cast<uint32_t>(views_.size());
    views_.push_back(view);
    return index;
}

void Image::destroy() {
    if (device_ != VK_NULL_HANDLE) {
        for (auto view : views_) {
            if (view != VK_NULL_HANDLE) {
                vkDestroyImageView(device_, view, nullptr);
            }
        }
        views_.clear();
    }
    if (image_ != VK_NULL_HANDLE && allocator_ != VK_NULL_HANDLE) {
        vmaDestroyImage(allocator_, image_, allocation_);
        image_ = VK_NULL_HANDLE;
        allocation_ = VK_NULL_HANDLE;
    }
}

} // namespace engine::vk2
