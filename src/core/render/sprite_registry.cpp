#include "core/render/sprite_registry.hpp"
#include "core/render/renderer.hpp"
#include "core/render/render_framework.hpp"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <utility>

void SpriteRegistry::registerSprite(uint16_t spriteId,
                                     uint32_t baseLayer,
                                     uint32_t frameCount,
                                     uint32_t tickRate,
                                     uint32_t flags,
                                     int32_t specularLayer,
                                     int32_t normalLayer,
                                     int32_t overlaySprite,
                                     int32_t maskLayer) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (spriteId >= vk::Data::SPRITE_MAX_ENTRIES) {
        std::cerr << "[SpriteRegistry] spriteId " << spriteId
                  << " exceeds max " << vk::Data::SPRITE_MAX_ENTRIES << std::endl;
        return;
    }

    if (entries_.size() <= spriteId) {
        vk::Data::SpriteEntry defaultEntry{};
        defaultEntry.baseLayer = 0;
        defaultEntry.frameCount = 1;
        defaultEntry.tickRate = 1;
        defaultEntry.flags = 0;
        defaultEntry.specularLayer = -1;
        defaultEntry.normalLayer = -1;
        defaultEntry.overlaySprite = -1;
        defaultEntry.maskLayer = -1;
        entries_.resize(spriteId + 1, defaultEntry);
    }

    uint32_t effectiveFlags = flags;
    int32_t effectiveMaskLayer = maskLayer;
    if (effectiveMaskLayer < 0) {
        effectiveFlags &= ~vk::Data::SPRITE_FLAG_HAS_HEIGHT;
    }
    if ((effectiveFlags & vk::Data::SPRITE_FLAG_HAS_HEIGHT) == 0u) {
        effectiveMaskLayer = -1;
    }

    auto& e = entries_[spriteId];
    e.baseLayer = baseLayer;
    e.frameCount = std::max(frameCount, 1u);
    e.tickRate = std::max(tickRate, 1u);
    e.flags = effectiveFlags;
    e.specularLayer = specularLayer;
    e.normalLayer = normalLayer;
    e.overlaySprite = overlaySprite;
    e.maskLayer = effectiveMaskLayer;

    spriteCount_ = std::max(spriteCount_, static_cast<uint32_t>(spriteId + 1));
}

const vk::Data::SpriteEntry* SpriteRegistry::getEntry(uint16_t spriteId) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (spriteId >= entries_.size()) return nullptr;
    return &entries_[spriteId];
}

bool SpriteRegistry::updateHeightMetadata(uint16_t spriteId, uint32_t flags, int32_t maskLayer) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (spriteId >= entries_.size()) return false;
    if (maskLayer < 0) {
        flags &= ~vk::Data::SPRITE_FLAG_HAS_HEIGHT;
    }
    if ((flags & vk::Data::SPRITE_FLAG_HAS_HEIGHT) == 0u) {
        maskLayer = -1;
    }
    auto& e = entries_[spriteId];
    constexpr uint32_t kSourceBits =
        vk::Data::SPRITE_FLAG_SPEC_SOURCE_MASK | vk::Data::SPRITE_FLAG_NORMAL_SOURCE_MASK;
    constexpr uint32_t kHeightBits = vk::Data::SPRITE_FLAG_HAS_NORMAL | vk::Data::SPRITE_FLAG_HAS_HEIGHT;
    e.flags = (e.flags & ~(kSourceBits | kHeightBits)) | (flags & (kSourceBits | kHeightBits));
    e.maskLayer = maskLayer;
    spriteCount_ = std::max(spriteCount_, static_cast<uint32_t>(spriteId + 1));
    return true;
}

bool SpriteRegistry::replacePrefix(const vk::Data::SpriteEntry* entries, uint32_t count) {
    if (!entries || count == 0 || count > vk::Data::SPRITE_MAX_ENTRIES) return false;
    std::lock_guard<std::mutex> lock(mutex_);
    if (entries_.size() < count) {
        vk::Data::SpriteEntry defaultEntry{};
        defaultEntry.baseLayer = 0;
        defaultEntry.frameCount = 1;
        defaultEntry.tickRate = 1;
        defaultEntry.flags = 0;
        defaultEntry.specularLayer = -1;
        defaultEntry.normalLayer = -1;
        defaultEntry.overlaySprite = -1;
        defaultEntry.maskLayer = -1;
        entries_.resize(count, defaultEntry);
    }
    std::copy(entries, entries + count, entries_.begin());
    spriteCount_ = std::max(spriteCount_, count);
    return true;
}

bool SpriteRegistry::uploadSSBO(std::shared_ptr<vk::VMA> vma, std::shared_ptr<vk::Device> device) {
    if (!vma || !device) return false;

    std::lock_guard<std::mutex> lock(mutex_);

    auto renderer = Renderer::try_instance();
    auto framework = renderer ? renderer->framework() : nullptr;
    if (!framework) {
        std::cerr << "[SpriteRegistry] Cannot upload SSBO without renderer framework" << std::endl;
        return false;
    }

    vk::Data::SpriteEntry defaultEntry{};
    defaultEntry.baseLayer = 0;
    defaultEntry.frameCount = 1;
    defaultEntry.tickRate = 1;
    defaultEntry.flags = 0;
    defaultEntry.specularLayer = -1;
    defaultEntry.normalLayer = -1;
    defaultEntry.overlaySprite = -1;
    defaultEntry.maskLayer = -1;

    std::vector<vk::Data::SpriteEntry> uploadEntries(vk::Data::SPRITE_MAX_ENTRIES, defaultEntry);
    const size_t copyCount = std::min(entries_.size(), uploadEntries.size());
    if (copyCount > 0) {
        std::copy(entries_.begin(), entries_.begin() + copyCount, uploadEntries.begin());
    }

    VkDeviceSize dataSize = uploadEntries.size() * sizeof(vk::Data::SpriteEntry);
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

    std::cout << "[SpriteRegistry] Uploaded " << spriteCount_ << " sprites ("
              << dataSize << " bytes padded) to GPU SSBO" << std::endl;
    return true;
}

void SpriteRegistry::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    entries_.clear();
    spriteCount_ = 0;
    ssbo_.reset();
}

void SpriteRegistry::clearEntries() {
    std::lock_guard<std::mutex> lock(mutex_);
    entries_.clear();
    spriteCount_ = 0;
}

void SpriteRegistry::retire(GarbageCollector& gc) {
    std::lock_guard<std::mutex> lock(mutex_);
    gc.collect(ssbo_);
    entries_.clear();
    spriteCount_ = 0;
    ssbo_.reset();
}
