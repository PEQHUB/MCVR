#include "core/vulkan/buffer.hpp"

#include "core/vulkan/command.hpp"
#include "core/vulkan/device.hpp"
#include "core/vulkan/vma.hpp"

#include <cstring>
#include <iostream>

std::ostream &bufferCout() {
    return std::cout << "[Buffer] ";
}

std::ostream &bufferCerr() {
    return std::cerr << "[Buffer] ";
}

vk::HostVisibleBuffer::HostVisibleBuffer(std::shared_ptr<VMA> vma,
                                         std::shared_ptr<Device> device,
                                         size_t size,
                                         VkBufferUsageFlags usage)
    : vma_(vma), device_(device), size_(size), bufferUsage_(usage) {
    VkBufferCreateInfo bufferInfo = {};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size_;
    bufferInfo.usage = usage;

    VmaAllocationCreateInfo allocationInfo = {};
    allocationInfo.usage = VMA_MEMORY_USAGE_AUTO;
    allocationInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;

    // if (bufferUsage_ & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) {
    //     allocationInfo.flags |= VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    // }

    if (vmaCreateBuffer(vma_->allocator(), &bufferInfo, &allocationInfo, &buffer_, &allocation_, &allocationInfo_) !=
        VK_SUCCESS) {
        bufferCerr() << "failed to create staging buffer" << std::endl;
    }
    mappedPtr_ = allocationInfo_.pMappedData;

    if (bufferUsage_ & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) {
        VkBufferDeviceAddressInfo deviceAddressInfo{.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
                                                    .buffer = buffer_};
        bufferAddress_ = vkGetBufferDeviceAddress(device_->vkDevice(), &deviceAddressInfo);
    }
}

vk::HostVisibleBuffer::HostVisibleBuffer(std::shared_ptr<VMA> vma,
                                         std::shared_ptr<Device> device,
                                         size_t size,
                                         VkBufferUsageFlags usage,
                                         VkDeviceSize minAlignment)
    : vma_(vma), device_(device), size_(size), bufferUsage_(usage) {
    VkBufferCreateInfo bufferInfo = {};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size_;
    bufferInfo.usage = usage;

    VmaAllocationCreateInfo allocationInfo = {};
    allocationInfo.usage = VMA_MEMORY_USAGE_AUTO;
    allocationInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;

    // if (bufferUsage_ & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) {
    //     allocationInfo.flags |= VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    // }

    if (vmaCreateBufferWithAlignment(vma_->allocator(), &bufferInfo, &allocationInfo, minAlignment, &buffer_,
                                     &allocation_, &allocationInfo_) != VK_SUCCESS) {
        bufferCerr() << "failed to create staging buffer" << std::endl;
    }
    mappedPtr_ = allocationInfo_.pMappedData;

    if (bufferUsage_ & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) {
        VkBufferDeviceAddressInfo deviceAddressInfo{.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
                                                    .buffer = buffer_};
        bufferAddress_ = vkGetBufferDeviceAddress(device_->vkDevice(), &deviceAddressInfo);
    }
}

vk::HostVisibleBuffer::~HostVisibleBuffer() {
    vmaDestroyBuffer(vma_->allocator(), buffer_, allocation_);

#ifdef DEBUG
// bufferCout() << "host visible buffer deconstructed" << std::endl;
#endif
}

void vk::HostVisibleBuffer::downloadFromBuffer() {
    downloadFromBuffer(size_, 0);
}

void vk::HostVisibleBuffer::downloadFromBuffer(size_t size, size_t offset) {
    vmaInvalidateAllocation(vma_->allocator(), allocation_, offset, size);
}

void vk::HostVisibleBuffer::uploadToBuffer(void *src) {
    uploadToBuffer(src, size_, 0);
}

void vk::HostVisibleBuffer::uploadToBuffer(void *src, size_t size, size_t offset) {
    std::memcpy(mappedPtr_, src, size);
    vmaFlushAllocation(vma_->allocator(), allocation_, 0, size);
}

void vk::HostVisibleBuffer::flush() {
    vmaFlushAllocation(vma_->allocator(), allocation_, 0, size_);
}

size_t vk::HostVisibleBuffer::size() {
    return size_;
}

VkBuffer &vk::HostVisibleBuffer::vkBuffer() {
    return buffer_;
}

void *vk::HostVisibleBuffer::mappedPtr() {
    return mappedPtr_;
}

VkDeviceAddress &vk::HostVisibleBuffer::bufferAddress() {
    if (!(bufferUsage_ & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT)) {
        bufferCerr() << "VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT not specified when try to get bufferAddress"
                     << std::endl;
        exit(EXIT_FAILURE);
    }
    return bufferAddress_;
}

vk::DeviceLocalBuffer::DeviceLocalBuffer(std::shared_ptr<VMA> vma,
                                         std::shared_ptr<Device> device,
                                         size_t size,
                                         VkBufferUsageFlags usageExceptTransfer)
    : DeviceLocalBuffer(vma, device, true, size, usageExceptTransfer) {}

vk::DeviceLocalBuffer::DeviceLocalBuffer(std::shared_ptr<VMA> vma,
                                         std::shared_ptr<Device> device,
                                         bool persistStaging,
                                         size_t size,
                                         VkBufferUsageFlags usageExceptTransfer)
    : DeviceLocalBuffer(
          vma, device, persistStaging, size, usageExceptTransfer, 0, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE) {}

vk::DeviceLocalBuffer::DeviceLocalBuffer(std::shared_ptr<VMA> vma,
                                         std::shared_ptr<Device> device,
                                         bool persistStaging,
                                         size_t size,
                                         VkBufferUsageFlags usageExceptTransfer,
                                         VmaAllocationCreateFlags vmaAllocationFlags,
                                         VmaMemoryUsage vmaUsage)
    : vma_(vma),
      device_(device),
      persistStaging_(persistStaging),
      size_(size),
      vmaAllocationFlags_(vmaAllocationFlags),
      vmaUsage_(vmaUsage) {
#ifdef DEBUG
// bufferCout() << "created buffer with size: " << size_ << std::endl;
#endif

    if (persistStaging_) {
        // staging buffer
        VkBufferCreateInfo bufferInfo = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bufferInfo.size = size_;
        bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

        VmaAllocationCreateInfo allocationInfo = {};
        allocationInfo.usage = VMA_MEMORY_USAGE_AUTO;
        allocationInfo.flags =
            VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;

        if (vmaCreateBuffer(vma_->allocator(), &bufferInfo, &allocationInfo, &stagingBuffer_, &stagingAllocation_,
                            &stagingAllocationInfo_) != VK_SUCCESS) {
            bufferCerr() << "failed to create staging buffer" << std::endl;
        }
        mappedPtr_ = stagingAllocationInfo_.pMappedData;
    }

    // buffer
    VkBufferCreateInfo bufferInfo = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufferInfo.size = size_;
    bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | usageExceptTransfer;
    bufferUsage_ = bufferInfo.usage;

    VmaAllocationCreateInfo allocationInfo = {};
    allocationInfo.flags = vmaAllocationFlags;
    allocationInfo.usage = vmaUsage;
    // if (usageExceptTransfer & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) {
    //     allocationInfo.flags |= VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    //     std::cout << "already specified VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT" << std::endl;
    // }

    if (vmaCreateBuffer(vma_->allocator(), &bufferInfo, &allocationInfo, &buffer_, &allocation_, &allocationInfo_) !=
        VK_SUCCESS) {
        bufferCerr() << "failed to create buffer" << std::endl;
    }

    if (usageExceptTransfer & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) {
        VkBufferDeviceAddressInfo deviceAddressInfo{.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
                                                    .buffer = buffer_};
        bufferAddress_ = vkGetBufferDeviceAddress(device_->vkDevice(), &deviceAddressInfo);
    }
}

vk::DeviceLocalBuffer::DeviceLocalBuffer(std::shared_ptr<VMA> vma,
                                         std::shared_ptr<Device> device,
                                         bool persistStaging,
                                         size_t size,
                                         VkBufferUsageFlags usageExceptTransfer,
                                         VmaAllocationCreateFlags vmaAllocationFlags,
                                         VmaMemoryUsage vmaUsage,
                                         VkDeviceSize minAlignment)
    : vma_(vma),
      device_(device),
      persistStaging_(persistStaging),
      size_(size),
      vmaAllocationFlags_(vmaAllocationFlags),
      vmaUsage_(vmaUsage) {
#ifdef DEBUG
// bufferCout() << "created buffer with size: " << size_ << std::endl;
#endif

    if (persistStaging_) {
        // staging buffer
        VkBufferCreateInfo bufferInfo = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bufferInfo.size = size_;
        bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

        VmaAllocationCreateInfo allocationInfo = {};
        allocationInfo.usage = VMA_MEMORY_USAGE_AUTO;
        allocationInfo.flags =
            VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;

        if (vmaCreateBuffer(vma_->allocator(), &bufferInfo, &allocationInfo, &stagingBuffer_, &stagingAllocation_,
                            &stagingAllocationInfo_) != VK_SUCCESS) {
            bufferCerr() << "failed to create staging buffer" << std::endl;
        }
        mappedPtr_ = stagingAllocationInfo_.pMappedData;
    }

    // buffer
    VkBufferCreateInfo bufferInfo = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufferInfo.size = size_;
    bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | usageExceptTransfer;
    bufferUsage_ = bufferInfo.usage;

    VmaAllocationCreateInfo allocationInfo = {};
    allocationInfo.flags = vmaAllocationFlags;
    allocationInfo.usage = vmaUsage;
    // if (usageExceptTransfer & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) {
    //     allocationInfo.flags |= VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    //     std::cout << "already specified VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT" << std::endl;
    // }

    if (vmaCreateBufferWithAlignment(vma_->allocator(), &bufferInfo, &allocationInfo, minAlignment, &buffer_,
                                     &allocation_, &allocationInfo_) != VK_SUCCESS) {
        bufferCerr() << "failed to create buffer" << std::endl;
    }

    if (usageExceptTransfer & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) {
        VkBufferDeviceAddressInfo deviceAddressInfo{.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
                                                    .buffer = buffer_};
        bufferAddress_ = vkGetBufferDeviceAddress(device_->vkDevice(), &deviceAddressInfo);
    }
}

vk::DeviceLocalBuffer::~DeviceLocalBuffer() {
    vmaDestroyBuffer(vma_->allocator(), stagingBuffer_, stagingAllocation_);
    vmaDestroyBuffer(vma_->allocator(), buffer_, allocation_);

#ifdef DEBUG
// bufferCout() << "device local buffer deconstructed" << std::endl;
#endif
}

void vk::DeviceLocalBuffer::downloadFromStagingBuffer(void *dest) {
    downloadFromStagingBuffer(dest, size_, 0);
}

void vk::DeviceLocalBuffer::downloadFromStagingBuffer(void *dest, size_t size, size_t offset) {
    if (!persistStaging_) {
        if (stagingBuffer_ != VK_NULL_HANDLE || stagingAllocation_ != VK_NULL_HANDLE || mappedPtr_ != nullptr) {
            bufferCerr() << "if not persist staging, the staging buffer should not exist!" << std::endl;
            exit(EXIT_FAILURE);
        }

        // staging buffer
        VkBufferCreateInfo bufferInfo = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bufferInfo.size = size_;
        bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

        VmaAllocationCreateInfo allocationInfo = {};
        allocationInfo.usage = VMA_MEMORY_USAGE_AUTO;
        allocationInfo.flags =
            VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;

        if (vmaCreateBuffer(vma_->allocator(), &bufferInfo, &allocationInfo, &stagingBuffer_, &stagingAllocation_,
                            &stagingAllocationInfo_) != VK_SUCCESS) {
            bufferCerr() << "failed to create staging buffer" << std::endl;
        }
        mappedPtr_ = stagingAllocationInfo_.pMappedData;
    }

    vmaInvalidateAllocation(vma_->allocator(), stagingAllocation_, offset, size);
    std::memcpy(dest, mappedPtr_, size);

    if (!persistStaging_) {
        vmaDestroyBuffer(vma_->allocator(), stagingBuffer_, stagingAllocation_);
        stagingBuffer_ = VK_NULL_HANDLE;
        stagingAllocation_ = VK_NULL_HANDLE;
        mappedPtr_ = nullptr;
    }
}

void vk::DeviceLocalBuffer::uploadToStagingBuffer(void *src) {
    uploadToStagingBuffer(src, size_, 0);
}

void vk::DeviceLocalBuffer::uploadToStagingBuffer(void *src, size_t size, size_t offset) {
    if (!persistStaging_) {
        if (stagingBuffer_ != VK_NULL_HANDLE || stagingAllocation_ != VK_NULL_HANDLE || mappedPtr_ != nullptr) {
            bufferCerr() << "if not persist staging, the staging buffer should not exist!" << std::endl;
            exit(EXIT_FAILURE);
        }

        // staging buffer
        VkBufferCreateInfo bufferInfo = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bufferInfo.size = size_;
        bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

        VmaAllocationCreateInfo allocationInfo = {};
        allocationInfo.usage = VMA_MEMORY_USAGE_AUTO;
        allocationInfo.flags =
            VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;

        if (vmaCreateBuffer(vma_->allocator(), &bufferInfo, &allocationInfo, &stagingBuffer_, &stagingAllocation_,
                            &stagingAllocationInfo_) != VK_SUCCESS) {
            bufferCerr() << "failed to create staging buffer" << std::endl;
        }
        mappedPtr_ = stagingAllocationInfo_.pMappedData;
    }

    std::memcpy(mappedPtr_, src, size);
    vmaFlushAllocation(vma_->allocator(), stagingAllocation_, offset, size);

    if (!persistStaging_) {
        vmaDestroyBuffer(vma_->allocator(), stagingBuffer_, stagingAllocation_);
        stagingBuffer_ = VK_NULL_HANDLE;
        stagingAllocation_ = VK_NULL_HANDLE;
        mappedPtr_ = nullptr;
    }
}

void vk::DeviceLocalBuffer::flushStagingBuffer() {
    if (!persistStaging_) { return; }
    vmaFlushAllocation(vma_->allocator(), stagingAllocation_, 0, size_);
}

void vk::DeviceLocalBuffer::downloadFromBuffer(VkCommandBuffer cmdBuffer) {
    downloadFromBuffer(cmdBuffer, size_, 0, 0);
}

void vk::DeviceLocalBuffer::downloadFromBuffer(VkCommandBuffer cmdBuffer,
                                               size_t size,
                                               size_t srcOffset,
                                               size_t dstOffset) {
    VkBufferCopy copyRegion = {srcOffset, dstOffset, size};
    vkCmdCopyBuffer(cmdBuffer, buffer_, stagingBuffer_, 1, &copyRegion);
}

void vk::DeviceLocalBuffer::uploadToBuffer(VkCommandBuffer cmdBuffer) {
    uploadToBuffer(cmdBuffer, size_, 0, 0);
}

void vk::DeviceLocalBuffer::uploadToBuffer(VkCommandBuffer cmdBuffer, size_t size, size_t srcOffset, size_t dstOffset) {
    VkBufferCopy copyRegion = {srcOffset, dstOffset, size};
    vkCmdCopyBuffer(cmdBuffer, stagingBuffer_, buffer_, 1, &copyRegion);
}

void vk::DeviceLocalBuffer::uploadToBuffer(std::shared_ptr<CommandBuffer> cmdBuffer) {
    uploadToBuffer(cmdBuffer->vkCommandBuffer());
}

void vk::DeviceLocalBuffer::uploadToBuffer(std::shared_ptr<CommandBuffer> cmdBuffer,
                                           size_t size,
                                           size_t srcOffset,
                                           size_t dstOffset) {
    uploadToBuffer(cmdBuffer->vkCommandBuffer(), size, srcOffset, dstOffset);
}

size_t vk::DeviceLocalBuffer::size() {
    return size_;
}

VkBuffer &vk::DeviceLocalBuffer::vkStagingBuffer() {
    return stagingBuffer_;
}

VkBuffer &vk::DeviceLocalBuffer::vkBuffer() {
    return buffer_;
}

void *vk::DeviceLocalBuffer::mappedPtr() {
    return mappedPtr_;
}

VkDeviceAddress &vk::DeviceLocalBuffer::bufferAddress() {
    if (!(bufferUsage_ & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT)) {
        bufferCerr() << "VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT not specified when try to get bufferAddress"
                     << std::endl;
        exit(EXIT_FAILURE);
    }
    return bufferAddress_;
}