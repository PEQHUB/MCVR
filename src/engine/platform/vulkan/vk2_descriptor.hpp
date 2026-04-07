#pragma once

// Ownership: Caller (unique ownership, RAII).
// Thread: Main thread only.
// Dependencies: VkDevice.
//
// DescriptorSetLayout: RAII wrapper around VkDescriptorSetLayout.
// DescriptorAllocator: Per-frame descriptor pool with reset-per-frame semantics.
// Write helpers: Typed descriptor update functions.

#include "vk2_result.hpp"

#include <cstdint>
#include <vector>

namespace engine::vk2 {

// --- RAII Descriptor Set Layout ---

class DescriptorSetLayout {
public:
    DescriptorSetLayout() = default;
    ~DescriptorSetLayout();

    DescriptorSetLayout(const DescriptorSetLayout&) = delete;
    DescriptorSetLayout& operator=(const DescriptorSetLayout&) = delete;
    DescriptorSetLayout(DescriptorSetLayout&& other) noexcept;
    DescriptorSetLayout& operator=(DescriptorSetLayout&& other) noexcept;

    static Result<DescriptorSetLayout> create(VkDevice device,
                                              const std::vector<VkDescriptorSetLayoutBinding>& bindings);

    VkDescriptorSetLayout handle() const { return layout_; }

private:
    VkDevice device_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout layout_ = VK_NULL_HANDLE;

    void destroy();
};

// --- Per-Frame Descriptor Allocator ---

// One pool per frame-in-flight. resetFrame() recycles all sets for that frame.
class DescriptorAllocator {
public:
    DescriptorAllocator() = default;
    ~DescriptorAllocator();

    DescriptorAllocator(const DescriptorAllocator&) = delete;
    DescriptorAllocator& operator=(const DescriptorAllocator&) = delete;

    Result<void> init(VkDevice device, uint32_t framesInFlight,
                      const std::vector<VkDescriptorPoolSize>& poolSizes,
                      uint32_t maxSetsPerFrame = 16);
    void shutdown();

    // Allocate a descriptor set from the current frame's pool.
    Result<VkDescriptorSet> allocate(VkDescriptorSetLayout layout, uint32_t frameIndex);

    // Reset the pool for the given frame (frees all sets allocated for that frame).
    void resetFrame(uint32_t frameIndex);

private:
    VkDevice device_ = VK_NULL_HANDLE;
    std::vector<VkDescriptorPool> pools_;  // one per frame-in-flight
};

// --- Write Helpers ---

namespace descriptor {

void writeStorageImage(VkDevice device, VkDescriptorSet set, uint32_t binding,
                       VkImageView view, VkImageLayout layout = VK_IMAGE_LAYOUT_GENERAL);

void writeCombinedImageSampler(VkDevice device, VkDescriptorSet set, uint32_t binding,
                               VkImageView view, VkSampler sampler,
                               VkImageLayout layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

void writeStorageBuffer(VkDevice device, VkDescriptorSet set, uint32_t binding,
                        VkBuffer buffer, VkDeviceSize offset, VkDeviceSize range);

void writeUniformBuffer(VkDevice device, VkDescriptorSet set, uint32_t binding,
                        VkBuffer buffer, VkDeviceSize offset, VkDeviceSize range);

} // namespace descriptor

} // namespace engine::vk2
