#pragma once

#include "core/all_extern.hpp"

#include <mutex>
#include <vector>

namespace vk {
class VMA;
class Device;

/// Suballocation handle returned by BufferPool::allocate().
/// Stores everything needed to use the region and return it to the pool.
struct SubAllocation {
    VkBuffer buffer = VK_NULL_HANDLE;     ///< Pool's backing VkBuffer
    VkDeviceSize offset = 0;              ///< Byte offset within backing buffer
    VkDeviceSize size = 0;                ///< Allocation size
    VkDeviceAddress address = 0;          ///< Pre-computed BDA (baseAddr + offset)
    uint32_t blockIndex = 0;              ///< Which block in the pool
    VmaVirtualAllocation virtualAlloc{};  ///< Handle for returning to VmaVirtualBlock
};

/// VmaVirtualBlock-based buffer suballocator.
///
/// Replaces per-chunk VkBuffer allocations with suballocations within large
/// (default 8 MB) backing VkBuffers. Reduces 260K individual driver allocations
/// to ~200 pooled VkBuffers.
///
/// Thread-safe: mutex-protected for concurrent BLAS thread + render thread access.
class BufferPool : public SharedObject<BufferPool> {
  public:
    BufferPool(std::shared_ptr<VMA> vma,
               std::shared_ptr<Device> device,
               VkBufferUsageFlags usage,
               VkDeviceSize blockSize = 8 * 1024 * 1024);
    ~BufferPool();

    /// Allocate a region of the given size with the given alignment.
    /// Creates a new backing block if no existing block has space.
    SubAllocation allocate(VkDeviceSize size, VkDeviceSize alignment = 16);

    /// Return a suballocation to the pool.
    void free(const SubAllocation &alloc);

    struct Stats {
        uint32_t blockCount;
        VkDeviceSize totalCapacity;
        VkDeviceSize totalUsed;
    };
    Stats getStats() const;

  private:
    struct Block {
        VkBuffer buffer = VK_NULL_HANDLE;
        VmaAllocation allocation = VK_NULL_HANDLE;
        VkDeviceAddress baseAddress = 0;
        VmaVirtualBlock virtualBlock = VK_NULL_HANDLE;
    };

    std::shared_ptr<VMA> vma_;
    std::shared_ptr<Device> device_;
    VkBufferUsageFlags usage_;
    VkDeviceSize blockSize_;
    std::vector<Block> blocks_;
    mutable std::mutex mutex_;

    /// Create a new backing VkBuffer + VmaVirtualBlock. Returns block index.
    uint32_t addBlock();
};

}; // namespace vk
