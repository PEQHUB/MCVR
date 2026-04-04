#include "vk2_buffer.hpp"
#include "diagnostics/log.hpp"

#include <volk.h>

namespace engine::vk2 {

Buffer::~Buffer() {
    destroy();
}

Buffer::Buffer(Buffer&& other) noexcept
    : device_(other.device_), allocator_(other.allocator_),
      buffer_(other.buffer_), allocation_(other.allocation_),
      size_(other.size_), mappedPtr_(other.mappedPtr_),
      deviceAddress_(other.deviceAddress_) {
    other.buffer_ = VK_NULL_HANDLE;
    other.allocation_ = VK_NULL_HANDLE;
    other.mappedPtr_ = nullptr;
    other.deviceAddress_ = 0;
}

Buffer& Buffer::operator=(Buffer&& other) noexcept {
    if (this != &other) {
        destroy();
        device_ = other.device_;
        allocator_ = other.allocator_;
        buffer_ = other.buffer_;
        allocation_ = other.allocation_;
        size_ = other.size_;
        mappedPtr_ = other.mappedPtr_;
        deviceAddress_ = other.deviceAddress_;
        other.buffer_ = VK_NULL_HANDLE;
        other.allocation_ = VK_NULL_HANDLE;
        other.mappedPtr_ = nullptr;
        other.deviceAddress_ = 0;
    }
    return *this;
}

Result<Buffer> Buffer::create(VkDevice device, VmaAllocator allocator, const Desc& desc) {
    if (desc.size == 0) {
        return makeError(VK_ERROR_INITIALIZATION_FAILED, "Buffer size must be > 0");
    }

    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = desc.size;
    bufferInfo.usage = desc.usage;

    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = desc.memoryUsage;
    allocInfo.flags = desc.vmaFlags;

    VkBuffer vkBuffer = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    VmaAllocationInfo allocationInfo{};

    VkResult result;
    if (desc.minAlignment > 0) {
        result = vmaCreateBufferWithAlignment(
            allocator, &bufferInfo, &allocInfo, desc.minAlignment,
            &vkBuffer, &allocation, &allocationInfo);
    } else {
        result = vmaCreateBuffer(
            allocator, &bufferInfo, &allocInfo,
            &vkBuffer, &allocation, &allocationInfo);
    }

    if (result != VK_SUCCESS) {
        return makeError(result, "vmaCreateBuffer failed");
    }

    Buffer buf;
    buf.device_ = device;
    buf.allocator_ = allocator;
    buf.buffer_ = vkBuffer;
    buf.allocation_ = allocation;
    buf.size_ = desc.size;
    buf.mappedPtr_ = allocationInfo.pMappedData;

    // Query device address if requested
    if (desc.usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) {
        VkBufferDeviceAddressInfo addrInfo{};
        addrInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
        addrInfo.buffer = vkBuffer;
        buf.deviceAddress_ = vkGetBufferDeviceAddress(device, &addrInfo);
    }

    return buf;
}

void Buffer::destroy() {
    if (buffer_ != VK_NULL_HANDLE && allocator_ != VK_NULL_HANDLE) {
        vmaDestroyBuffer(allocator_, buffer_, allocation_);
        buffer_ = VK_NULL_HANDLE;
        allocation_ = VK_NULL_HANDLE;
        mappedPtr_ = nullptr;
        deviceAddress_ = 0;
    }
}

} // namespace engine::vk2
