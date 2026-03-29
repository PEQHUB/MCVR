#include "core/render/texture_system.hpp"
#include "core/render/renderer.hpp"
#include "core/render/render_framework.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>

const TextureSystem::SpriteBounds TextureSystem::DEFAULT_BOUNDS = {0.0f, 1.0f, 0.0f, 1.0f};

void TextureSystem::receiveSpriteTable(const SpriteMetadata* table, uint32_t count,
                                        uint32_t atlasWidth, uint32_t atlasHeight) {
    std::lock_guard<std::mutex> lock(mutex_);

    sprites_.assign(table, table + count);
    atlasWidth_ = atlasWidth;
    atlasHeight_ = atlasHeight;

    // Compute atlas UV bounds for each sprite (for model UV normalization)
    float invW = atlasWidth > 0 ? 1.0f / static_cast<float>(atlasWidth) : 1.0f;
    float invH = atlasHeight > 0 ? 1.0f / static_cast<float>(atlasHeight) : 1.0f;

    spriteBounds_.resize(count);
    for (uint32_t i = 0; i < count; i++) {
        auto& s = sprites_[i];
        auto& b = spriteBounds_[i];
        b.minU = static_cast<float>(s.atlasX) * invW;
        b.maxU = static_cast<float>(s.atlasX + s.width) * invW;
        b.minV = static_cast<float>(s.atlasY) * invH;
        b.maxV = static_cast<float>(s.atlasY + s.height) * invH;
    }

    std::cout << "[TextureSystem] Received " << count << " sprites, atlas "
              << atlasWidth << "x" << atlasHeight << std::endl;
}

void TextureSystem::receiveSpritePixels(const uint8_t* data, uint32_t totalBytes) {
    std::lock_guard<std::mutex> lock(mutex_);

    spritePixels_.assign(data, data + totalBytes);

    std::cout << "[TextureSystem] Received " << (totalBytes / 1024) << " KB sprite pixels" << std::endl;
}

void TextureSystem::receiveAuxPixels(const uint8_t* specularData, const uint8_t* normalData,
                                      uint32_t totalBytesPerType) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (specularData && totalBytesPerType > 0)
        specularPixels_.assign(specularData, specularData + totalBytesPerType);
    if (normalData && totalBytesPerType > 0)
        normalPixels_.assign(normalData, normalData + totalBytesPerType);
    std::cout << "[TextureSystem] Received aux pixels: "
              << (totalBytesPerType / 1024) << " KB each" << std::endl;
}

void TextureSystem::receiveAnimationFrames(const uint8_t* data, uint32_t totalBytes) {
    std::lock_guard<std::mutex> lock(mutex_);

    animEntries_.clear();

    // Parse bulk format: [spriteId(u16), frameIndex(u16), pixels(w*h*4)] repeated
    const uint8_t* ptr = data;
    const uint8_t* end = data + totalBytes;

    while (ptr + 4 <= end) {
        uint16_t spriteId = *reinterpret_cast<const uint16_t*>(ptr);
        uint16_t frameIndex = *reinterpret_cast<const uint16_t*>(ptr + 2);
        ptr += 4;

        if (spriteId >= sprites_.size()) {
            std::cerr << "[TextureSystem] Animation spriteId " << spriteId
                      << " out of range (" << sprites_.size() << "), skipping" << std::endl;
            // Can't determine frame size without metadata — must abort
            break;
        }

        auto& meta = sprites_[spriteId];
        size_t frameBytes = static_cast<size_t>(meta.width) * meta.height * 4;
        if (ptr + frameBytes > end) {
            std::cerr << "[TextureSystem] Animation data truncated at spriteId " << spriteId
                      << " frame " << frameIndex
                      << " (need " << frameBytes << ", have " << (end - ptr) << ")" << std::endl;
            break;
        }

        // Find or create AnimEntry for this spriteId
        AnimEntry* entry = nullptr;
        for (auto& ae : animEntries_) {
            if (ae.spriteId == spriteId) {
                entry = &ae;
                break;
            }
        }
        if (!entry) {
            animEntries_.push_back({});
            entry = &animEntries_.back();
            entry->spriteId = spriteId;
            entry->tickRate = std::max<uint16_t>(meta.tickRate, 1);
            entry->frames.resize(meta.frameCount);
        }

        if (frameIndex < entry->frames.size()) {
            entry->frames[frameIndex].assign(ptr, ptr + frameBytes);
        }

        ptr += frameBytes;
    }

    std::cout << "[TextureSystem] Received animation data for " << animEntries_.size()
              << " animated sprites (" << totalBytes << " bytes)" << std::endl;
}

void TextureSystem::finalize(std::shared_ptr<vk::VMA> vma, std::shared_ptr<vk::Device> device) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (sprites_.empty()) {
        std::cerr << "[TextureSystem] Cannot finalize: no sprite data" << std::endl;
        return;
    }

    // Handle re-initialization (F3+T reload): destroy old arrays before creating new ones
    if (finalized_) {
        std::cout << "[TextureSystem] Re-initializing (destroying old arrays)" << std::endl;
        // Wait for ALL GPU work across all queues and all frames in flight.
        // Without this, in-flight command buffers still reference old texture arrays
        // via descriptors — destroying them causes GPU access violation (exit -805306369).
        auto framework = Renderer::instance().framework();
        vkDeviceWaitIdle(framework->device()->vkDevice());
        arrayManager_.reset();
        registry_.reset();
        blockAlbedoArrayId_ = UINT32_MAX;
        blockSpecularArrayId_ = UINT32_MAX;
        blockNormalArrayId_ = UINT32_MAX;
        albedoMipsInitialized_ = false;
        specMipsInitialized_ = false;
        normMipsInitialized_ = false;
        finalized_ = false;
    }

    uint32_t count = static_cast<uint32_t>(sprites_.size());
    uint32_t spriteSize = sprites_[0].width;

    // Validate against hardware limit
    auto framework = Renderer::instance().framework();
    auto physDevice = framework->physicalDevice();
    VkPhysicalDeviceProperties props = physDevice->properties();
    uint32_t maxLayers = props.limits.maxImageArrayLayers;

    if (count > maxLayers) {
        std::cerr << "[TextureSystem] WARNING: " << count
                  << " sprites exceeds maxImageArrayLayers=" << maxLayers
                  << ". Clamping." << std::endl;
        count = maxLayers;
    }

    std::cout << "[TextureSystem] Finalizing: " << count << " sprites, "
              << spriteSize << "x" << spriteSize
              << ", max layers=" << maxLayers << std::endl;

    // Create block albedo texture array
    blockAlbedoArrayId_ = arrayManager_.createArray(
        vma, device, spriteSize, count,
        VK_FORMAT_R8G8B8A8_SRGB, true /* generateMips */);

    // Stage frame-0 pixels for each sprite into the texture array.
    // Pixels are concatenated in sorted order in spritePixels_.
    // Java uses FIXED offset: i * spriteSize * spriteSize * 4 (all sprites assumed same size).
    size_t bytesPerSprite = static_cast<size_t>(spriteSize) * spriteSize * 4;
    for (uint32_t i = 0; i < count; i++) {
        // For animated sprites, prefer frame 0 from animation data (more reliable)
        const uint8_t* frameData = nullptr;
        for (auto& ae : animEntries_) {
            if (ae.spriteId == i && !ae.frames.empty() && !ae.frames[0].empty()) {
                frameData = ae.frames[0].data();
                break;
            }
        }

        // Fall back to concatenated pixel data (fixed offset, matching Java layout)
        size_t pixelOffset = static_cast<size_t>(i) * bytesPerSprite;
        if (!frameData && pixelOffset + bytesPerSprite <= spritePixels_.size()) {
            frameData = spritePixels_.data() + pixelOffset;
        }

        if (frameData) {
            arrayManager_.stageLayerPixels(blockAlbedoArrayId_, i, 0,
                                           frameData, bytesPerSprite);
        } else {
            std::cerr << "[TextureSystem] Missing pixel data for sprite " << i << std::endl;
        }
    }

    std::cout << "[TextureSystem] Staged " << count << " albedo layers" << std::endl;

    // Create block specular texture array (UNORM — LabPBR values are linear, NOT sRGB)
    if (!specularPixels_.empty()) {
        blockSpecularArrayId_ = arrayManager_.createArray(
            vma, device, spriteSize, count,
            VK_FORMAT_R8G8B8A8_UNORM, true /* generateMips */);
        for (uint32_t i = 0; i < count; i++) {
            size_t pixelOffset = static_cast<size_t>(i) * bytesPerSprite;
            if (pixelOffset + bytesPerSprite <= specularPixels_.size()) {
                arrayManager_.stageLayerPixels(blockSpecularArrayId_, i, 0,
                    specularPixels_.data() + pixelOffset, bytesPerSprite);
            }
        }
        std::cout << "[TextureSystem] Staged " << count << " specular layers" << std::endl;
    }

    // Create block normal texture array (UNORM — linear normal map data)
    if (!normalPixels_.empty()) {
        blockNormalArrayId_ = arrayManager_.createArray(
            vma, device, spriteSize, count,
            VK_FORMAT_R8G8B8A8_UNORM, true /* generateMips */);
        for (uint32_t i = 0; i < count; i++) {
            size_t pixelOffset = static_cast<size_t>(i) * bytesPerSprite;
            if (pixelOffset + bytesPerSprite <= normalPixels_.size()) {
                arrayManager_.stageLayerPixels(blockNormalArrayId_, i, 0,
                    normalPixels_.data() + pixelOffset, bytesPerSprite);
            }
        }
        std::cout << "[TextureSystem] Staged " << count << " normal layers" << std::endl;
    }

    // Build SpriteRegistry SSBO entries
    registry_.reset();
    for (uint32_t i = 0; i < count; i++) {
        auto& meta = sprites_[i];

        // Find overlay relationship: if sprite J has overlayOf == i,
        // then sprite i's overlaySprite = J.
        int32_t overlaySprite = -1;
        for (uint32_t j = 0; j < count; j++) {
            if (sprites_[j].overlayOf == static_cast<int16_t>(i)) {
                overlaySprite = static_cast<int32_t>(j);
                break;
            }
        }

        // Flags from Java metadata (padding field repurposed)
        uint32_t flags = 0;
        if (meta.padding & 1u) flags |= 1u; // SPRITE_FLAG_HAS_SPECULAR
        if (meta.padding & 2u) flags |= 2u; // SPRITE_FLAG_HAS_NORMAL

        // All sprites get a layer in aux arrays (defaults for missing)
        int32_t specLayer = (blockSpecularArrayId_ != UINT32_MAX) ? static_cast<int32_t>(i) : -1;
        int32_t normLayer = (blockNormalArrayId_ != UINT32_MAX) ? static_cast<int32_t>(i) : -1;

        registry_.registerSprite(
            static_cast<uint16_t>(i),
            i,                                          // baseLayer = spriteId
            1,                                          // frameCount=1 (animation via re-upload)
            std::max<uint32_t>(meta.tickRate, 1),       // tickRate
            flags,                                      // aux texture flags
            specLayer,                                  // specularLayer
            normLayer,                                  // normalLayer
            overlaySprite,                              // overlaySprite
            -1);                                        // maskLayer
    }

    registry_.uploadSSBO(vma, device);

    // Free CPU pixel data (no longer needed after staging)
    spritePixels_.clear();
    spritePixels_.shrink_to_fit();
    specularPixels_.clear();
    specularPixels_.shrink_to_fit();
    normalPixels_.clear();
    normalPixels_.shrink_to_fit();

    finalized_ = true;

    // Diagnostic log
    {
        std::ofstream diag("C:/RadSER/texture_system_diag.log", std::ios::trunc);
        diag << "=== TextureSystem Diagnostic ===" << std::endl;
        diag << "Sprites: " << count << " (" << spriteSize << "x" << spriteSize << ")" << std::endl;
        diag << "Atlas: " << atlasWidth_ << "x" << atlasHeight_ << std::endl;
        diag << "Animated: " << animEntries_.size() << std::endl;
        diag << "Array IDs: albedo=" << blockAlbedoArrayId_
             << ", specular=" << blockSpecularArrayId_
             << ", normal=" << blockNormalArrayId_ << std::endl;
        diag << std::endl;

        for (uint32_t i = 0; i < std::min(count, 20u); i++) {
            auto& m = sprites_[i];
            auto& b = spriteBounds_[i];
            auto* se = registry_.getEntry(static_cast<uint16_t>(i));
            diag << "Sprite[" << i << "]: "
                 << "atlas=(" << m.atlasX << "," << m.atlasY << ") "
                 << m.width << "x" << m.height
                 << " frames=" << m.frameCount
                 << " flags=" << (se ? se->flags : 0)
                 << " spec=" << (se ? se->specularLayer : -2)
                 << " norm=" << (se ? se->normalLayer : -2)
                 << " UV=[" << b.minU << "," << b.maxU << "]x[" << b.minV << "," << b.maxV << "]"
                 << " overlay=" << (se ? se->overlaySprite : -2)
                 << std::endl;
        }
        diag.close();
    }

    std::cout << "[TextureSystem] Finalized. " << count << " sprites ready, "
              << animEntries_.size() << " animated." << std::endl;
}

bool TextureSystem::tickAnimation(uint32_t gameTick) {
    if (!finalized_ || animEntries_.empty()) return false;

    bool anyUpdated = false;

    for (auto& ae : animEntries_) {
        if (ae.frames.empty()) continue;

        uint32_t frameCount = static_cast<uint32_t>(ae.frames.size());
        uint32_t tickRate = std::max<uint32_t>(ae.tickRate, 1);
        uint32_t currentFrame = (gameTick / tickRate) % frameCount;

        if (currentFrame != ae.lastFrame && currentFrame < ae.frames.size()
            && !ae.frames[currentFrame].empty()) {

            arrayManager_.stageLayerPixels(
                blockAlbedoArrayId_, ae.spriteId, 0,
                ae.frames[currentFrame].data(),
                ae.frames[currentFrame].size());

            ae.lastFrame = currentFrame;
            anyUpdated = true;
        }
    }

    return anyUpdated;
}

void TextureSystem::flushPendingUploads(std::shared_ptr<vk::VMA> vma,
                                         std::shared_ptr<vk::Device> device,
                                         std::shared_ptr<vk::CommandBuffer> cmdBuffer,
                                         GarbageCollector& gc) {
    if (!finalized_) return;
    if (!arrayManager_.hasPendingUploads()) return;

    // Single mutex lock: collect dirty layers for all arrays at once
    auto dirty = arrayManager_.collectAllDirtyLayers(
        blockAlbedoArrayId_, blockSpecularArrayId_, blockNormalArrayId_);

    // Flush all staged uploads
    arrayManager_.flushUploads(vma, device, cmdBuffer);

    // Route staging buffers through GarbageCollector for proper lifetime management.
    // GC keeps resources alive for imageCount*3 frames (matching all other GPU resources).
    for (auto& buf : arrayManager_.takeStagingBuffers()) {
        gc.collect(buf);
    }

    // Mipgen strategy:
    //   - First flush (init): bulk generateMipmaps for all layers
    //   - Many dirty layers (material change, >64): bulk to avoid TDR from thousands of barriers
    //   - Few dirty layers (animation tick, slider tweak, ≤64): per-layer for efficiency
    constexpr size_t BULK_MIPGEN_THRESHOLD = 64;

    auto mipgen = [&](uint32_t arrayId, const std::vector<uint32_t>& layers, bool& initialized) {
        if (layers.empty() || arrayId == UINT32_MAX) return;
        if (!initialized || layers.size() > BULK_MIPGEN_THRESHOLD) {
            arrayManager_.generateMipmaps(arrayId, cmdBuffer);
            initialized = true;
        } else {
            for (uint32_t layer : layers)
                arrayManager_.generateMipmapsForLayer(arrayId, layer, cmdBuffer);
        }
    };

    mipgen(blockAlbedoArrayId_, dirty.albedo, albedoMipsInitialized_);
    mipgen(blockSpecularArrayId_, dirty.specular, specMipsInitialized_);
    mipgen(blockNormalArrayId_, dirty.normal, normMipsInitialized_);
}

const TextureSystem::SpriteBounds& TextureSystem::getSpriteBounds(uint16_t spriteId) const {
    if (spriteId < spriteBounds_.size()) {
        return spriteBounds_[spriteId];
    }
    return DEFAULT_BOUNDS;
}

void TextureSystem::reset() {
    std::lock_guard<std::mutex> lock(mutex_);

    sprites_.clear();
    spriteBounds_.clear();
    spritePixels_.clear();
    specularPixels_.clear();
    normalPixels_.clear();
    atlasWidth_ = 0;
    atlasHeight_ = 0;
    animEntries_.clear();
    arrayManager_.reset();
    registry_.reset();
    blockAlbedoArrayId_ = UINT32_MAX;
    blockSpecularArrayId_ = UINT32_MAX;
    blockNormalArrayId_ = UINT32_MAX;
    albedoMipsInitialized_ = false;
    specMipsInitialized_ = false;
    normMipsInitialized_ = false;
    finalized_ = false;

    std::cout << "[TextureSystem] Reset." << std::endl;
}
