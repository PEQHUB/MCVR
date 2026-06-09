#pragma once

// Ownership: Shared (shared_ptr). Destroyed when last reference drops.
// Thread: Creation on any thread with VMA. Access not internally synchronized.
// Dependencies: VMA allocator, Vulkan device.

#include "vk2_result.hpp"

#include <cstddef>
#include <memory>
#include <vk_mem_alloc.h>

namespace engine::vk2 {

// RAII Vulkan buffer backed by VMA. No partial objects — either fully valid or not created.
class Buffer {
public:
    ~Buffer();

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    Buffer(Buffer&& other) noexcept;
    Buffer& operator=(Buffer&& other) noexcept;

    // --- Creation ---

    struct Desc {
        VkDeviceSize size = 0;
        VkBufferUsageFlags usage = 0;               // Caller specifies all flags (including TRANSFER_DST if needed)
        VmaAllocationCreateFlags vmaFlags = 0;       // e.g., HOST_ACCESS_SEQUENTIAL_WRITE_BIT | MAPPED_BIT
        VmaMemoryUsage memoryUsage = VMA_MEMORY_USAGE_AUTO;
        VkDeviceSize minAlignment = 0;               // 0 = no alignment requirement
    };

    // Create a buffer. Returns Error on failure — never a partial object.
    static Result<Buffer> create(VkDevice device, VmaAllocator allocator, const Desc& desc);

    // --- Accessors ---

    VkBuffer handle() const { return buffer_; }
    VkDeviceSize size() const { return size_; }
    VmaAllocation allocation() const { return allocation_; }

    // Mapped pointer (only valid if created with MAPPED flag)
    void* mappedPtr() const { return mappedPtr_; }

    Buffer() = default;

    // Buffer device address (only valid if created with SHADER_DEVICE_ADDRESS usage)
    VkDeviceAddress deviceAddress() const { return deviceAddress_; }

private:

    VkDevice device_ = VK_NULL_HANDLE;       // Non-owning (device outlives buffer)
    VmaAllocator allocator_ = VK_NULL_HANDLE; // Non-owning
    VkBuffer buffer_ = VK_NULL_HANDLE;
    VmaAllocation allocation_ = VK_NULL_HANDLE;
    VkDeviceSize size_ = 0;
    void* mappedPtr_ = nullptr;
    VkDeviceAddress deviceAddress_ = 0;

    void destroy();
};

} // namespace engine::vk2
