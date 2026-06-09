#include "core/render/texture_rule_registry.hpp"
#include "core/render/renderer.hpp"
#include "core/render/render_framework.hpp"

#include <algorithm>
#include <iostream>
#include <utility>

bool TextureRuleRegistry::uploadRules(const vk::Data::TextureRuleEntry* entries, uint32_t count,
                                      std::shared_ptr<vk::VMA> vma,
                                      std::shared_ptr<vk::Device> device) {
    if (!entries || count == 0 || !vma || !device) return false;
    std::lock_guard<std::mutex> lock(mutex_);

    auto renderer = Renderer::try_instance();
    if (!renderer || !renderer->framework()) return false;
    auto framework = renderer->framework();

    vk::Data::TextureRuleEntry defaultEntry{};
    std::vector<vk::Data::TextureRuleEntry> uploadEntries(vk::Data::SPRITE_MAX_ENTRIES, defaultEntry);
    const size_t copyCount = std::min<size_t>(count, uploadEntries.size());
    std::copy(entries, entries + copyCount, uploadEntries.begin());

    VkDeviceSize dataSize = uploadEntries.size() * sizeof(vk::Data::TextureRuleEntry);
    auto nextSsbo = vk::DeviceLocalBuffer::create(
        vma, device, true, dataSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    nextSsbo->uploadToStagingBuffer(uploadEntries.data(), static_cast<size_t>(dataSize), 0);

    auto fence = vk::Fence::create(device);
    auto cmd = vk::CommandBuffer::create(device, framework->mainCommandPool());
    cmd->begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
    nextSsbo->uploadToBuffer(cmd);
    cmd->end();
    cmd->submitMainQueueIndividual(device, fence);
    vkWaitForFences(device->vkDevice(), 1, &fence->vkFence(), VK_TRUE, UINT64_MAX);

    if (ssbo_) {
        framework->gc().collect(ssbo_);
    }
    ssbo_ = std::move(nextSsbo);
    entries_.assign(entries, entries + copyCount);

    std::cout << "[TextureRuleRegistry] Uploaded "
              << std::min<uint32_t>(count, vk::Data::SPRITE_MAX_ENTRIES)
              << " texture rules (" << dataSize << " bytes padded)" << std::endl;
    return true;
}

void TextureRuleRegistry::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    entries_.clear();
    ssbo_.reset();
}

void TextureRuleRegistry::retire(GarbageCollector& gc) {
    std::lock_guard<std::mutex> lock(mutex_);
    gc.collect(ssbo_);
    entries_.clear();
    ssbo_.reset();
}
