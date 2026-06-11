#include "core/render/texture_rule_registry.hpp"
#include "core/render/renderer.hpp"
#include "core/render/render_framework.hpp"

#include <algorithm>
#include <iostream>
#include <sstream>
#include <utility>

namespace {
std::vector<vk::Data::TextureRuleEntry> makeDefaultTextureRules() {
    vk::Data::TextureRuleEntry defaultEntry{};
    return std::vector<vk::Data::TextureRuleEntry>(vk::Data::SPRITE_MAX_ENTRIES, defaultEntry);
}
}

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
    usingDefaultRules_ = false;
    revision_++;

    std::cout << "[TextureRuleRegistry] Uploaded "
              << std::min<uint32_t>(count, vk::Data::SPRITE_MAX_ENTRIES)
              << " texture rules (" << dataSize << " bytes padded)" << std::endl;
    return true;
}

bool TextureRuleRegistry::ensureDefaultRules(std::shared_ptr<vk::VMA> vma,
                                             std::shared_ptr<vk::Device> device) {
    if (!vma || !device) return false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (ssbo_ && ssbo_->isValid() && usingDefaultRules_) return true;
        if (ssbo_ && ssbo_->isValid() && !usingDefaultRules_) return true;
    }

    auto renderer = Renderer::try_instance();
    if (!renderer || !renderer->framework()) return false;
    auto framework = renderer->framework();

    auto uploadEntries = makeDefaultTextureRules();
    VkDeviceSize dataSize = uploadEntries.size() * sizeof(vk::Data::TextureRuleEntry);
    auto nextSsbo = vk::DeviceLocalBuffer::create(
        vma, device, true, dataSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    if (!nextSsbo || !nextSsbo->isValid()) return false;

    nextSsbo->uploadToStagingBuffer(uploadEntries.data(), static_cast<size_t>(dataSize), 0);

    auto fence = vk::Fence::create(device);
    auto cmd = vk::CommandBuffer::create(device, framework->mainCommandPool());
    cmd->begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
    nextSsbo->uploadToBuffer(cmd);
    cmd->end();
    cmd->submitMainQueueIndividual(device, fence);
    vkWaitForFences(device->vkDevice(), 1, &fence->vkFence(), VK_TRUE, UINT64_MAX);

    std::lock_guard<std::mutex> lock(mutex_);
    if (ssbo_) {
        framework->gc().collect(ssbo_);
    }
    ssbo_ = std::move(nextSsbo);
    entries_.clear();
    usingDefaultRules_ = true;
    revision_++;
    std::cout << "[TextureRuleRegistry] Uploaded default V4 fallback texture rules ("
              << dataSize << " bytes padded)" << std::endl;
    return true;
}

bool TextureRuleRegistry::hasBuffer() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return ssbo_ && ssbo_->isValid();
}

bool TextureRuleRegistry::usingDefaultRules() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return usingDefaultRules_;
}

uint64_t TextureRuleRegistry::revision() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return revision_;
}

std::string TextureRuleRegistry::statusJson() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::ostringstream out;
    out << "{"
        << "\"schema\":\"radser_texture_rule_status_v1\","
        << "\"hasBuffer\":" << ((ssbo_ && ssbo_->isValid()) ? "true" : "false") << ","
        << "\"usingDefaultRules\":" << (usingDefaultRules_ ? "true" : "false") << ","
        << "\"ruleCount\":" << entries_.size() << ","
        << "\"paddedRuleCapacity\":" << vk::Data::SPRITE_MAX_ENTRIES << ","
        << "\"revision\":" << revision_
        << "}";
    return out.str();
}

void TextureRuleRegistry::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    entries_.clear();
    ssbo_.reset();
    usingDefaultRules_ = false;
    revision_++;
}

void TextureRuleRegistry::retire(GarbageCollector& gc) {
    std::lock_guard<std::mutex> lock(mutex_);
    gc.collect(ssbo_);
    entries_.clear();
    ssbo_.reset();
    usingDefaultRules_ = false;
    revision_++;
}
