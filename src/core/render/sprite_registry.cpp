#include "core/render/sprite_registry.hpp"
#include "core/render/renderer.hpp"
#include "core/render/render_framework.hpp"

#include <algorithm>
#include <cstring>
#include <iostream>

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
        // Default entry: single static layer, no aux textures
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

    auto& e = entries_[spriteId];
    e.baseLayer = baseLayer;
    e.frameCount = std::max(frameCount, 1u);
    e.tickRate = std::max(tickRate, 1u);
    e.flags = flags;
    e.specularLayer = specularLayer;
    e.normalLayer = normalLayer;
    e.overlaySprite = overlaySprite;
    e.maskLayer = maskLayer;

    spriteCount_ = std::max(spriteCount_, static_cast<uint32_t>(spriteId + 1));
}

const vk::Data::SpriteEntry* SpriteRegistry::getEntry(uint16_t spriteId) const {
    if (spriteId >= entries_.size()) return nullptr;
    return &entries_[spriteId];
}

bool SpriteRegistry::updateHeightMetadata(uint16_t spriteId, uint32_t flags, int32_t maskLayer) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (spriteId >= entries_.size()) return false;
    auto& e = entries_[spriteId];
    e.flags = flags;
    e.maskLayer = maskLayer;
    spriteCount_ = std::max(spriteCount_, static_cast<uint32_t>(spriteId + 1));
    return true;
}

void SpriteRegistry::uploadSSBO(std::shared_ptr<vk::VMA> vma, std::shared_ptr<vk::Device> device) {
    std::lock_guard<std::mutex> lock(mutex_);

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

    ssbo_ = vk::DeviceLocalBuffer::create(
        vma, device, true, dataSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    ssbo_->uploadToStagingBuffer(uploadEntries.data(), static_cast<size_t>(dataSize), 0);

    // One-shot staging → device-local transfer. Without this, the shader reads
    // uninitialized VRAM from the device-local buffer (staging is a separate VkBuffer).
    auto framework = Renderer::instance().framework();
    auto fence = vk::Fence::create(device);
    auto cmd = vk::CommandBuffer::create(device, framework->mainCommandPool());
    cmd->begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
    ssbo_->uploadToBuffer(cmd);
    cmd->end();
    cmd->submitMainQueueIndividual(device, fence);
    vkWaitForFences(device->vkDevice(), 1, &fence->vkFence(), VK_TRUE, UINT64_MAX);

    std::cout << "[SpriteRegistry] Uploaded " << spriteCount_ << " sprites ("
              << dataSize << " bytes padded) to GPU SSBO" << std::endl;
}

void SpriteRegistry::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    entries_.clear();
    spriteCount_ = 0;
    ssbo_.reset();
}

void SpriteRegistry::retire(GarbageCollector& gc) {
    std::lock_guard<std::mutex> lock(mutex_);
    gc.collect(ssbo_);
    entries_.clear();
    spriteCount_ = 0;
    ssbo_.reset();
}
