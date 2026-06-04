#include "core/vulkan/buffer.hpp"

#include "core/vulkan/command.hpp"
#include "core/vulkan/debug_utils.hpp"
#include "core/vulkan/device.hpp"
#include "core/vulkan/vma.hpp"
#include "core/render/renderer.hpp"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

// ── Buffer diagnostic file ──
static std::ofstream sBufferDiag;
static std::ofstream& bufferDiag() {
    if (!sBufferDiag.is_open()) {
        auto logDir = Renderer::folderPath / "logs";
        std::filesystem::create_directories(logDir);
        sBufferDiag.open((logDir / "buffer_diag.log").string(), std::ios::trunc);
    }
    return sBufferDiag;
}

// ── VMA budget diagnostic helper ──
static void logVmaBudgetOnFailure(VmaAllocator allocator) {
    VmaTotalStatistics stats{};
    vmaCalculateStatistics(allocator, &stats);
    VmaBudget budgets[VK_MAX_MEMORY_HEAPS]{};
    vmaGetHeapBudgets(allocator, budgets);
    uint64_t totalBudget = 0, totalUsage = 0;
    for (uint32_t i = 0; i < VK_MAX_MEMORY_HEAPS; ++i) {
        totalBudget += budgets[i].budget;
        totalUsage += budgets[i].usage;
    }
    float usedGB = (float)totalUsage / (1024.f * 1024.f * 1024.f);
    float budgetGB = (float)totalBudget / (1024.f * 1024.f * 1024.f);
    float allocGB = (float)stats.total.statistics.blockBytes / (1024.f * 1024.f * 1024.f);
    bufferDiag() << "[VRAM] usage=" << usedGB << "GB"
              << " budget=" << budgetGB << "GB"
              << " allocated=" << allocGB << "GB"
              << " allocCount=" << stats.total.statistics.allocationCount
              << " blockCount=" << stats.total.statistics.blockCount
              << std::endl;
    sBufferDiag.flush();
}

std::ostream &bufferCout() {
    return std::cout << "[Buffer] ";
}

std::ostream &bufferCerr() {
    return bufferDiag() << "";
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

	VkResult bufResult = vmaCreateBuffer(vma_->allocator(), &bufferInfo, &allocationInfo, &buffer_, &allocation_, &allocationInfo_);
	if (bufResult != VK_SUCCESS) {
		bufferCerr() << "FAILED vmaCreateBuffer(HostVisible): result=" << bufResult << " size=" << size_ << std::endl;
		logVmaBudgetOnFailure(vma_->allocator());
		buffer_ = VK_NULL_HANDLE; allocation_ = VK_NULL_HANDLE; mappedPtr_ = nullptr; return;
	}
	mappedPtr_ = allocationInfo_.pMappedData;

	if (bufferUsage_ & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) {
		VkBufferDeviceAddressInfo deviceAddressInfo{.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
		.buffer = buffer_};
		bufferAddress_ = vkGetBufferDeviceAddress(device_->vkDevice(), &deviceAddressInfo);
	}
    vk::DebugUtils::setObjectName(device_->vkDevice(), VK_OBJECT_TYPE_BUFFER, buffer_,
                                  "HostVisibleBuffer bytes=" + std::to_string(size_));
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

	VkResult bufAlignResult = vmaCreateBufferWithAlignment(vma_->allocator(), &bufferInfo, &allocationInfo, minAlignment, &buffer_,
		&allocation_, &allocationInfo_);
	if (bufAlignResult != VK_SUCCESS) {
		bufferCerr() << "FAILED vmaCreateBufferWithAlignment(HostVisible): result=" << bufAlignResult << " size=" << size_ << " align=" << minAlignment << std::endl;
		logVmaBudgetOnFailure(vma_->allocator());
		buffer_ = VK_NULL_HANDLE;
		allocation_ = VK_NULL_HANDLE;
		mappedPtr_ = nullptr;
		return;
	}
	mappedPtr_ = allocationInfo_.pMappedData;

    if (bufferUsage_ & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) {
        VkBufferDeviceAddressInfo deviceAddressInfo{.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
                                                    .buffer = buffer_};
        bufferAddress_ = vkGetBufferDeviceAddress(device_->vkDevice(), &deviceAddressInfo);
    }
    vk::DebugUtils::setObjectName(device_->vkDevice(), VK_OBJECT_TYPE_BUFFER, buffer_,
                                  "HostVisibleBuffer bytes=" + std::to_string(size_) +
                                      " aligned");
}

vk::HostVisibleBuffer::~HostVisibleBuffer() {
    vmaDestroyBuffer(vma_->allocator(), buffer_, allocation_);

#ifdef DEBUG
// bufferCout() << "host visible buffer deconstructed" << std::endl;
    sBufferDiag.flush();
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
	if (mappedPtr_ == nullptr) { bufferCerr() << "uploadToBuffer: null mappedPtr, skipping" << std::endl; sBufferDiag.flush(); return; }
    std::memcpy(static_cast<char *>(mappedPtr_) + offset, src, size);
    vmaFlushAllocation(vma_->allocator(), allocation_, offset, size);
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
    sBufferDiag.flush();
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

	VkResult stagingRes = vmaCreateBuffer(vma_->allocator(), &bufferInfo, &allocationInfo, &stagingBuffer_, &stagingAllocation_,
		&stagingAllocationInfo_);
	if (stagingRes != VK_SUCCESS) {
		bufferCerr() << "FAILED vmaCreateBuffer(DeviceLocal staging): result=" << stagingRes << " size=" << size_ << std::endl;
		logVmaBudgetOnFailure(vma_->allocator());
		stagingBuffer_ = VK_NULL_HANDLE;
		stagingAllocation_ = VK_NULL_HANDLE;
		mappedPtr_ = nullptr;
	} else {
		mappedPtr_ = stagingAllocationInfo_.pMappedData;
        vk::DebugUtils::setObjectName(device_->vkDevice(), VK_OBJECT_TYPE_BUFFER, stagingBuffer_,
                                      "DeviceLocalBuffer staging bytes=" + std::to_string(size_));
	}
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

	VkResult deviceRes = vmaCreateBuffer(vma_->allocator(), &bufferInfo, &allocationInfo, &buffer_, &allocation_,
		&allocationInfo_);
	if (deviceRes != VK_SUCCESS) {
		bufferCerr() << "FAILED vmaCreateBuffer(DeviceLocal GPU): result=" << deviceRes << " size=" << size_ << " usage=0x" << std::hex << usageExceptTransfer << std::dec << std::endl;
		logVmaBudgetOnFailure(vma_->allocator());
		buffer_ = VK_NULL_HANDLE;
		allocation_ = VK_NULL_HANDLE;
		return;
	}
    vk::DebugUtils::setObjectName(device_->vkDevice(), VK_OBJECT_TYPE_BUFFER, buffer_,
                                  "DeviceLocalBuffer GPU bytes=" + std::to_string(size_));

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
    sBufferDiag.flush();
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

	VkResult stagingAlignRes = vmaCreateBuffer(vma_->allocator(), &bufferInfo, &allocationInfo, &stagingBuffer_, &stagingAllocation_,
		&stagingAllocationInfo_);
	if (stagingAlignRes != VK_SUCCESS) {
		bufferCerr() << "FAILED vmaCreateBuffer(DeviceLocal aligned staging): result=" << stagingAlignRes << " size=" << size_ << std::endl;
		logVmaBudgetOnFailure(vma_->allocator());
		stagingBuffer_ = VK_NULL_HANDLE;
		stagingAllocation_ = VK_NULL_HANDLE;
		mappedPtr_ = nullptr;
	} else {
		mappedPtr_ = stagingAllocationInfo_.pMappedData;
        vk::DebugUtils::setObjectName(device_->vkDevice(), VK_OBJECT_TYPE_BUFFER, stagingBuffer_,
                                      "DeviceLocalBuffer staging bytes=" + std::to_string(size_) +
                                          " aligned");
	}
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

	VkResult deviceAlignRes = vmaCreateBufferWithAlignment(vma_->allocator(), &bufferInfo, &allocationInfo, minAlignment, &buffer_,
		&allocation_, &allocationInfo_);
	if (deviceAlignRes != VK_SUCCESS) {
		bufferCerr() << "FAILED vmaCreateBufferWithAlignment(DeviceLocal GPU): result=" << deviceAlignRes << " size=" << size_ << " align=" << minAlignment << " usage=0x" << std::hex << usageExceptTransfer << std::dec << std::endl;
		logVmaBudgetOnFailure(vma_->allocator());
		buffer_ = VK_NULL_HANDLE;
		allocation_ = VK_NULL_HANDLE;
		return;
	}
    vk::DebugUtils::setObjectName(device_->vkDevice(), VK_OBJECT_TYPE_BUFFER, buffer_,
                                  "DeviceLocalBuffer GPU bytes=" + std::to_string(size_) +
                                      " aligned");

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
    sBufferDiag.flush();
#endif
}

void vk::DeviceLocalBuffer::downloadFromStagingBuffer(void *dest) {
    downloadFromStagingBuffer(dest, size_, 0);
}

void vk::DeviceLocalBuffer::downloadFromStagingBuffer(void *dest, size_t size, size_t offset) {
    if (!persistStaging_) {
        if (stagingBuffer_ != VK_NULL_HANDLE || stagingAllocation_ != VK_NULL_HANDLE || mappedPtr_ != nullptr) {
            bufferCerr() << "if not persist staging, the staging buffer should not exist!" << std::endl;
    sBufferDiag.flush();
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

	VkResult dlRes = vmaCreateBuffer(vma_->allocator(), &bufferInfo, &allocationInfo, &stagingBuffer_, &stagingAllocation_,
		&stagingAllocationInfo_);
	if (dlRes != VK_SUCCESS) {
		bufferCerr() << "FAILED vmaCreateBuffer(downloadFromStaging): result=" << dlRes << " size=" << size_ << std::endl;
		logVmaBudgetOnFailure(vma_->allocator());
		return;
	}
	mappedPtr_ = stagingAllocationInfo_.pMappedData;
	}
	if (mappedPtr_ == nullptr) {
		bufferCerr() << "downloadFromStagingBuffer: null mappedPtr" << std::endl;
		sBufferDiag.flush();
		return;
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
    if (offset > size_ || size > size_ - offset) {
        bufferCerr() << "uploadToStagingBuffer: out-of-range upload size=" << size
                     << " offset=" << offset << " bufferSize=" << size_ << std::endl;
        sBufferDiag.flush();
        return;
    }
    if (!persistStaging_) {
        if (stagingBuffer_ != VK_NULL_HANDLE || stagingAllocation_ != VK_NULL_HANDLE || mappedPtr_ != nullptr) {
            bufferCerr() << "if not persist staging, the staging buffer should not exist!" << std::endl;
    sBufferDiag.flush();
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

	VkResult uploadRes = vmaCreateBuffer(vma_->allocator(), &bufferInfo, &allocationInfo, &stagingBuffer_, &stagingAllocation_,
		&stagingAllocationInfo_);
	if (uploadRes != VK_SUCCESS) {
		bufferCerr() << "FAILED vmaCreateBuffer(uploadToStaging): result=" << uploadRes << " size=" << size_ << std::endl;
		logVmaBudgetOnFailure(vma_->allocator());
		return;
	}
	mappedPtr_ = stagingAllocationInfo_.pMappedData;
	}
	if (mappedPtr_ == nullptr) {
		bufferCerr() << "uploadToStagingBuffer: null mappedPtr" << std::endl;
		sBufferDiag.flush();
		return;
	}

	std::memcpy(static_cast<char *>(mappedPtr_) + offset, src, size);
	VkResult flushRes = vmaFlushAllocation(vma_->allocator(), stagingAllocation_, offset, size);
	if (flushRes != VK_SUCCESS) {
		bufferCerr() << "uploadToStagingBuffer: vmaFlushAllocation failed result=" << flushRes
		             << " size=" << size << " offset=" << offset << std::endl;
		sBufferDiag.flush();
	}

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

void vk::DeviceLocalBuffer::releaseStagingBuffer() {
    if (stagingBuffer_ == VK_NULL_HANDLE) return;
    vmaDestroyBuffer(vma_->allocator(), stagingBuffer_, stagingAllocation_);
    stagingBuffer_ = VK_NULL_HANDLE;
    stagingAllocation_ = VK_NULL_HANDLE;
    mappedPtr_ = nullptr;
}

void vk::DeviceLocalBuffer::downloadFromBuffer(VkCommandBuffer cmdBuffer) {
    downloadFromBuffer(cmdBuffer, size_, 0, 0);
}

void vk::DeviceLocalBuffer::downloadFromBuffer(VkCommandBuffer cmdBuffer,
                                               size_t size,
                                               size_t srcOffset,
                                               size_t dstOffset) {
	if (stagingBuffer_ == VK_NULL_HANDLE || buffer_ == VK_NULL_HANDLE) return;
    VkBufferCopy copyRegion = {srcOffset, dstOffset, size};
    vkCmdCopyBuffer(cmdBuffer, buffer_, stagingBuffer_, 1, &copyRegion);
}

void vk::DeviceLocalBuffer::uploadToBuffer(VkCommandBuffer cmdBuffer) {
    uploadToBuffer(cmdBuffer, size_, 0, 0);
}

void vk::DeviceLocalBuffer::uploadToBuffer(VkCommandBuffer cmdBuffer, size_t size, size_t srcOffset, size_t dstOffset) {
	if (stagingBuffer_ == VK_NULL_HANDLE || buffer_ == VK_NULL_HANDLE) return;
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

bool vk::DeviceLocalBuffer::isValid() const {
    if (size_ == 0 || buffer_ == VK_NULL_HANDLE || allocation_ == VK_NULL_HANDLE) {
        return false;
    }
    if (persistStaging_ &&
        (stagingBuffer_ == VK_NULL_HANDLE || stagingAllocation_ == VK_NULL_HANDLE || mappedPtr_ == nullptr)) {
        return false;
    }
    if ((bufferUsage_ & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) && bufferAddress_ == 0) {
        return false;
    }
    return true;
}

VkDeviceAddress &vk::DeviceLocalBuffer::bufferAddress() {
    if (!(bufferUsage_ & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT)) {
        bufferCerr() << "VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT not specified when try to get bufferAddress"
                     << std::endl;
        exit(EXIT_FAILURE);
    }
    return bufferAddress_;
}
