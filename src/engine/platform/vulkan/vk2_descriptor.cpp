#include "vk2_descriptor.hpp"

#include <volk.h>

namespace engine::vk2 {

// --- DescriptorSetLayout ---

DescriptorSetLayout::~DescriptorSetLayout() { destroy(); }

DescriptorSetLayout::DescriptorSetLayout(DescriptorSetLayout&& other) noexcept
    : device_(other.device_), layout_(other.layout_) {
    other.device_ = VK_NULL_HANDLE;
    other.layout_ = VK_NULL_HANDLE;
}

DescriptorSetLayout& DescriptorSetLayout::operator=(DescriptorSetLayout&& other) noexcept {
    if (this != &other) {
        destroy();
        device_ = other.device_;
        layout_ = other.layout_;
        other.device_ = VK_NULL_HANDLE;
        other.layout_ = VK_NULL_HANDLE;
    }
    return *this;
}

Result<DescriptorSetLayout> DescriptorSetLayout::create(
    VkDevice device, const std::vector<VkDescriptorSetLayoutBinding>& bindings) {

    VkDescriptorSetLayoutCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    ci.bindingCount = static_cast<uint32_t>(bindings.size());
    ci.pBindings = bindings.data();

    VkDescriptorSetLayout layout = VK_NULL_HANDLE;
    VkResult r = vkCreateDescriptorSetLayout(device, &ci, nullptr, &layout);
    if (r != VK_SUCCESS) return makeError(r, "vkCreateDescriptorSetLayout failed");

    DescriptorSetLayout dsl;
    dsl.device_ = device;
    dsl.layout_ = layout;
    return dsl;
}

void DescriptorSetLayout::destroy() {
    if (layout_ != VK_NULL_HANDLE && device_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device_, layout_, nullptr);
    }
    layout_ = VK_NULL_HANDLE;
    device_ = VK_NULL_HANDLE;
}

// --- DescriptorAllocator ---

DescriptorAllocator::~DescriptorAllocator() { shutdown(); }

Result<void> DescriptorAllocator::init(VkDevice device, uint32_t framesInFlight,
                                       const std::vector<VkDescriptorPoolSize>& poolSizes,
                                       uint32_t maxSetsPerFrame) {
    device_ = device;
    pools_.resize(framesInFlight);

    // Scale pool sizes by maxSetsPerFrame
    std::vector<VkDescriptorPoolSize> scaled = poolSizes;
    for (auto& ps : scaled) ps.descriptorCount *= maxSetsPerFrame;

    for (uint32_t i = 0; i < framesInFlight; ++i) {
        VkDescriptorPoolCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        ci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        ci.maxSets = maxSetsPerFrame;
        ci.poolSizeCount = static_cast<uint32_t>(scaled.size());
        ci.pPoolSizes = scaled.data();

        VkResult r = vkCreateDescriptorPool(device_, &ci, nullptr, &pools_[i]);
        if (r != VK_SUCCESS) return makeError(r, "vkCreateDescriptorPool failed");
    }

    return {};
}

void DescriptorAllocator::shutdown() {
    for (auto pool : pools_) {
        if (pool != VK_NULL_HANDLE) vkDestroyDescriptorPool(device_, pool, nullptr);
    }
    pools_.clear();
    device_ = VK_NULL_HANDLE;
}

Result<VkDescriptorSet> DescriptorAllocator::allocate(VkDescriptorSetLayout layout,
                                                      uint32_t frameIndex) {
    VkDescriptorSetAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool = pools_[frameIndex];
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &layout;

    VkDescriptorSet set = VK_NULL_HANDLE;
    VkResult r = vkAllocateDescriptorSets(device_, &ai, &set);
    if (r != VK_SUCCESS) return makeError(r, "vkAllocateDescriptorSets failed");

    return set;
}

void DescriptorAllocator::resetFrame(uint32_t frameIndex) {
    if (frameIndex < pools_.size()) {
        vkResetDescriptorPool(device_, pools_[frameIndex], 0);
    }
}

// --- Write Helpers ---

namespace descriptor {

void writeStorageImage(VkDevice device, VkDescriptorSet set, uint32_t binding,
                       VkImageView view, VkImageLayout layout) {
    VkDescriptorImageInfo info{};
    info.imageView = view;
    info.imageLayout = layout;

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = set;
    write.dstBinding = binding;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    write.pImageInfo = &info;

    vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
}

void writeCombinedImageSampler(VkDevice device, VkDescriptorSet set, uint32_t binding,
                               VkImageView view, VkSampler sampler, VkImageLayout layout) {
    VkDescriptorImageInfo info{};
    info.imageView = view;
    info.sampler = sampler;
    info.imageLayout = layout;

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = set;
    write.dstBinding = binding;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    write.pImageInfo = &info;

    vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
}

void writeStorageBuffer(VkDevice device, VkDescriptorSet set, uint32_t binding,
                        VkBuffer buffer, VkDeviceSize offset, VkDeviceSize range) {
    VkDescriptorBufferInfo info{};
    info.buffer = buffer;
    info.offset = offset;
    info.range = range;

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = set;
    write.dstBinding = binding;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.pBufferInfo = &info;

    vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
}

void writeUniformBuffer(VkDevice device, VkDescriptorSet set, uint32_t binding,
                        VkBuffer buffer, VkDeviceSize offset, VkDeviceSize range) {
    VkDescriptorBufferInfo info{};
    info.buffer = buffer;
    info.offset = offset;
    info.range = range;

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = set;
    write.dstBinding = binding;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    write.pBufferInfo = &info;

    vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
}

} // namespace descriptor

} // namespace engine::vk2
