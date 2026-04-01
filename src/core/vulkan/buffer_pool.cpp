#include "core/vulkan/buffer_pool.hpp"

#include "core/vulkan/device.hpp"
#include "core/vulkan/vma.hpp"

#include <iostream>

static std::ostream &poolCout() {
    return std::cout << "[BufferPool] ";
}

static std::ostream &poolCerr() {
    return std::cerr << "[BufferPool] ";
}

vk::BufferPool::BufferPool(std::shared_ptr<VMA> vma,
                           std::shared_ptr<Device> device,
                           VkBufferUsageFlags usage,
                           VkDeviceSize blockSize)
    : vma_(vma), device_(device), usage_(usage), blockSize_(blockSize) {}

vk::BufferPool::~BufferPool() {
    for (auto &block : blocks_) {
        if (block.virtualBlock) {
            vmaClearVirtualBlock(block.virtualBlock);
            vmaDestroyVirtualBlock(block.virtualBlock);
        }
        if (block.buffer) {
            vmaDestroyBuffer(vma_->allocator(), block.buffer, block.allocation);
        }
    }
    blocks_.clear();
}

uint32_t vk::BufferPool::addBlock() {
    Block block{};

    // Create VmaVirtualBlock for offset tracking
    VmaVirtualBlockCreateInfo vbci{};
    vbci.size = blockSize_;
    if (vmaCreateVirtualBlock(&vbci, &block.virtualBlock) != VK_SUCCESS) {
        poolCerr() << "failed to create VmaVirtualBlock (size=" << blockSize_ << ")" << std::endl;
        exit(EXIT_FAILURE);
    }

    // Create backing VkBuffer via VMA
    VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufferInfo.size = blockSize_;
    bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | usage_;

    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

    if (vmaCreateBuffer(vma_->allocator(), &bufferInfo, &allocInfo,
                        &block.buffer, &block.allocation, nullptr) != VK_SUCCESS) {
        poolCerr() << "failed to create backing VkBuffer (size=" << blockSize_ << ")" << std::endl;
        exit(EXIT_FAILURE);
    }

    // Get base BDA
    if (usage_ & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) {
        VkBufferDeviceAddressInfo addrInfo{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
        addrInfo.buffer = block.buffer;
        block.baseAddress = vkGetBufferDeviceAddress(device_->vkDevice(), &addrInfo);
    }

    uint32_t idx = static_cast<uint32_t>(blocks_.size());
    blocks_.push_back(block);

    poolCout() << "block " << idx << " created (" << (blockSize_ / (1024 * 1024)) << " MB)" << std::endl;
    return idx;
}

vk::SubAllocation vk::BufferPool::allocate(VkDeviceSize size, VkDeviceSize alignment) {
    std::lock_guard<std::mutex> lock(mutex_);

    VmaVirtualAllocationCreateInfo vaci{};
    vaci.size = size;
    vaci.alignment = alignment;

    // Try existing blocks first
    for (uint32_t i = 0; i < blocks_.size(); i++) {
        VmaVirtualAllocation va{};
        VkDeviceSize offset = 0;
        if (vmaVirtualAllocate(blocks_[i].virtualBlock, &vaci, &va, &offset) == VK_SUCCESS) {
            SubAllocation alloc{};
            alloc.buffer = blocks_[i].buffer;
            alloc.offset = offset;
            alloc.size = size;
            alloc.address = blocks_[i].baseAddress + offset;
            alloc.blockIndex = i;
            alloc.virtualAlloc = va;
            return alloc;
        }
    }

    // All blocks full — grow.
    // If requested size exceeds block size, create an oversized block.
    VkDeviceSize origBlockSize = blockSize_;
    if (size > blockSize_) {
        blockSize_ = size;
    }
    uint32_t newIdx = addBlock();
    blockSize_ = origBlockSize;

    VmaVirtualAllocation va{};
    VkDeviceSize offset = 0;
    if (vmaVirtualAllocate(blocks_[newIdx].virtualBlock, &vaci, &va, &offset) != VK_SUCCESS) {
        poolCerr() << "allocation failed even in fresh block (size=" << size << ")" << std::endl;
        exit(EXIT_FAILURE);
    }

    SubAllocation alloc{};
    alloc.buffer = blocks_[newIdx].buffer;
    alloc.offset = offset;
    alloc.size = size;
    alloc.address = blocks_[newIdx].baseAddress + offset;
    alloc.blockIndex = newIdx;
    alloc.virtualAlloc = va;
    return alloc;
}

void vk::BufferPool::free(const SubAllocation &alloc) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (alloc.blockIndex < blocks_.size() && blocks_[alloc.blockIndex].virtualBlock) {
        vmaVirtualFree(blocks_[alloc.blockIndex].virtualBlock, alloc.virtualAlloc);
    }
}

vk::BufferPool::Stats vk::BufferPool::getStats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    Stats stats{};
    stats.blockCount = static_cast<uint32_t>(blocks_.size());
    for (auto &block : blocks_) {
        stats.totalCapacity += blockSize_;
        VmaStatistics vmaStats{};
        vmaGetVirtualBlockStatistics(block.virtualBlock, &vmaStats);
        stats.totalUsed += vmaStats.allocationBytes;
    }
    return stats;
}
