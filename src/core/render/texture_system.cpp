#include "core/render/texture_system.hpp"
#include "core/render/renderer.hpp"
#include "core/render/render_framework.hpp"
#include "core/render/chunks.hpp"
#include "core/render/world.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <cmath>
#include <limits>
#include <sstream>
#include <utility>
#include <vector>

const TextureSystem::SpriteBounds TextureSystem::DEFAULT_BOUNDS = {0.0f, 1.0f, 0.0f, 1.0f};

namespace {
uint64_t fnv1a64(const uint8_t* data, size_t size) {
    uint64_t hash = 1469598103934665603ull;
    for (size_t i = 0; i < size; i++) {
        hash ^= static_cast<uint64_t>(data[i]);
        hash *= 1099511628211ull;
    }
    return hash;
}

void appendArrayId(std::vector<uint32_t>& ids, uint32_t id) {
    if (id == UINT32_MAX) return;
    if (std::find(ids.begin(), ids.end(), id) != ids.end()) return;
    ids.push_back(id);
}

}

TextureSystem::TextureSystem() {
    resetMaterialTexturePagesLocked();
}

void TextureSystem::resetMaterialTexturePagesLocked() {
    for (uint32_t page = 0; page < vk::Data::MATERIAL_TEXTURE_PAGE_MAX; page++) {
        materialAlbedoPageArrayIds_[page].store(UINT32_MAX, std::memory_order_release);
        materialSpecularPageArrayIds_[page].store(UINT32_MAX, std::memory_order_release);
        materialNormalPageArrayIds_[page].store(UINT32_MAX, std::memory_order_release);
        materialFlagPageArrayIds_[page].store(UINT32_MAX, std::memory_order_release);
        materialPageReady_[page].store(false, std::memory_order_release);
        materialPageMipsDirty_[page] = false;
        materialPageLayerCapacity_[page] = 0;
        materialPageLayersUsed_[page] = 0;
    }
    materialPageUpdates_ = 0;
    materialPageImageAllocations_ = 0;
    lastMaterialPage_ = 0;
    lastMaterialPageStartLayer_ = 0;
    lastMaterialPageLayerCount_ = 0;
    lastMaterialPageLayerCapacity_ = 0;
}

bool TextureSystem::hasMaterialPageMipsDirtyLocked() const {
    for (bool dirty : materialPageMipsDirty_) {
        if (dirty) return true;
    }
    return false;
}

uint32_t TextureSystem::materialAlbedoPageArrayId(uint32_t page) const {
    if (page == 0) return blockAlbedoArrayId();
    if (page >= vk::Data::MATERIAL_TEXTURE_PAGE_MAX) return UINT32_MAX;
    if (!materialPageReady_[page].load(std::memory_order_acquire)) return UINT32_MAX;
    return materialAlbedoPageArrayIds_[page].load(std::memory_order_acquire);
}

uint32_t TextureSystem::materialSpecularPageArrayId(uint32_t page) const {
    if (page == 0) return blockSpecularArrayId();
    if (page >= vk::Data::MATERIAL_TEXTURE_PAGE_MAX) return UINT32_MAX;
    if (!materialPageReady_[page].load(std::memory_order_acquire)) return UINT32_MAX;
    return materialSpecularPageArrayIds_[page].load(std::memory_order_acquire);
}

uint32_t TextureSystem::materialNormalPageArrayId(uint32_t page) const {
    if (page == 0) return blockNormalArrayId();
    if (page >= vk::Data::MATERIAL_TEXTURE_PAGE_MAX) return UINT32_MAX;
    if (!materialPageReady_[page].load(std::memory_order_acquire)) return UINT32_MAX;
    return materialNormalPageArrayIds_[page].load(std::memory_order_acquire);
}

uint32_t TextureSystem::materialFlagPageArrayId(uint32_t page) const {
    if (page == 0) return blockFlagArrayId();
    if (page >= vk::Data::MATERIAL_TEXTURE_PAGE_MAX) return UINT32_MAX;
    if (!materialPageReady_[page].load(std::memory_order_acquire)) return UINT32_MAX;
    return materialFlagPageArrayIds_[page].load(std::memory_order_acquire);
}

void TextureSystem::waitForGpuIdleLocked(std::shared_ptr<vk::Device> device, const char* reason) {
    auto* renderer = Renderer::try_instance();
    auto framework = renderer ? renderer->framework() : nullptr;

    std::shared_ptr<ChunkBuildScheduler> scheduler;
    bool schedulerPaused = false;
    if (renderer && renderer->world() && renderer->world()->chunks()) {
        scheduler = renderer->world()->chunks()->chunkBuildScheduler();
    }

    if (scheduler) {
        schedulerPaused = scheduler->pause(std::chrono::milliseconds(5000));
        if (!schedulerPaused) {
            std::cerr << "[TextureSystem] WARNING: BLAS scheduler did not pause before "
                      << reason << "; preserving old texture resources through GC anyway"
                      << std::endl;
        }
    }

    if (framework && framework->device()) {
        vkDeviceWaitIdle(framework->device()->vkDevice());
    } else if (device) {
        vkDeviceWaitIdle(device->vkDevice());
    }

    if (scheduler && schedulerPaused) {
        scheduler->resume();
    }
}

void TextureSystem::retireGpuResourcesLocked(std::shared_ptr<vk::Device> device, const char* reason) {
    auto* renderer = Renderer::try_instance();
    auto framework = renderer ? renderer->framework() : nullptr;

    finalized_ = false;
    waitForGpuIdleLocked(device, reason);

    if (framework) {
        auto& gc = framework->gc();
        arrayManager_.retire(gc);
        registry_.retire(gc);
        materials_.retire(gc);
        textureRules_.retire(gc);
    } else {
        arrayManager_.reset();
        registry_.reset();
        materials_.reset();
        textureRules_.reset();
    }

    blockAlbedoArrayId_ = UINT32_MAX;
    blockSpecularArrayId_ = UINT32_MAX;
    blockNormalArrayId_ = UINT32_MAX;
    blockFlagArrayId_ = UINT32_MAX;
    resetMaterialTexturePagesLocked();
    materialTexturePageRevision_.fetch_add(1, std::memory_order_acq_rel);
    albedoMipsInitialized_ = false;
    specMipsInitialized_ = false;
    normMipsInitialized_ = false;
    flagMipsInitialized_ = false;
    finalized_ = false;
}

void TextureSystem::receiveSpriteTable(const SpriteMetadata* table, uint32_t count,
                                        uint32_t atlasWidth, uint32_t atlasHeight) {
    std::lock_guard<std::mutex> lock(mutex_);

    generation_.fetch_add(1, std::memory_order_acq_rel);
    finalized_ = false;
    arrayManager_.discardPendingUploads();
    spritePixels_.clear();
    specularPixels_.clear();
    normalPixels_.clear();
    flagPixels_.clear();
    animEntries_.clear();
    albedoChecksums_.clear();
    specularChecksums_.clear();
    normalChecksums_.clear();
    flagChecksums_.clear();

    if (!table || count == 0) {
        sprites_.clear();
        spriteBounds_.clear();
        atlasWidth_ = 0;
        atlasHeight_ = 0;
        layerSize_ = 0;
        std::cerr << "[TextureSystem] Received empty sprite table" << std::endl;
        return;
    }

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

    if (!data || totalBytes == 0) {
        spritePixels_.clear();
        layerSize_ = 0;
        std::cerr << "[TextureSystem] Received empty sprite pixel payload" << std::endl;
        return;
    }
    if (sprites_.empty()) {
        spritePixels_.clear();
        layerSize_ = 0;
        std::cerr << "[TextureSystem] Ignoring sprite pixels before sprite table" << std::endl;
        return;
    }

    size_t bytesPerLayer = totalBytes / sprites_.size();
    size_t pixelsPerLayer = bytesPerLayer / 4;
    uint32_t inferredLayerSize = static_cast<uint32_t>(
        std::sqrt(static_cast<double>(pixelsPerLayer)));
    if (bytesPerLayer == 0 || bytesPerLayer % 4 != 0 ||
        static_cast<size_t>(inferredLayerSize) * inferredLayerSize != pixelsPerLayer ||
        bytesPerLayer * sprites_.size() != totalBytes) {
        spritePixels_.clear();
        layerSize_ = 0;
        std::cerr << "[TextureSystem] Invalid fixed-layer sprite payload: bytes="
                  << totalBytes << ", sprites=" << sprites_.size() << std::endl;
        return;
    }

    layerSize_ = inferredLayerSize;
    spritePixels_.assign(data, data + totalBytes);

    std::cout << "[TextureSystem] Received " << (totalBytes / 1024)
              << " KB sprite pixels (" << layerSize_ << "x" << layerSize_
              << " layers)" << std::endl;
}

void TextureSystem::receiveAuxPixels(const uint8_t* specularData, const uint8_t* normalData,
                                      const uint8_t* flagData, uint32_t totalBytesPerType) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (specularData && totalBytesPerType > 0)
        specularPixels_.assign(specularData, specularData + totalBytesPerType);
    if (normalData && totalBytesPerType > 0)
        normalPixels_.assign(normalData, normalData + totalBytesPerType);
    if (flagData && totalBytesPerType > 0)
        flagPixels_.assign(flagData, flagData + totalBytesPerType);
    std::cout << "[TextureSystem] Received aux pixels: "
              << (totalBytesPerType / 1024) << " KB each" << std::endl;
}

void TextureSystem::receiveAnimationFrames(const uint8_t* data, uint32_t totalBytes) {
    std::lock_guard<std::mutex> lock(mutex_);

    animEntries_.clear();
    if (!data || totalBytes == 0) {
        std::cout << "[TextureSystem] Received animation data for 0 animated sprites (0 bytes)" << std::endl;
        return;
    }
    if (sprites_.empty()) {
        std::cerr << "[TextureSystem] Ignoring animation data before sprite table" << std::endl;
        return;
    }

    // Parse bulk format: [spriteId(u16), frameIndex(u16), pixels(layerSize^2*4)] repeated
    // Java writes every frame as a fixed-size texture-array layer.
    size_t frameBytes = static_cast<size_t>(layerSize_) * layerSize_ * 4;
    if (frameBytes == 0) {
        std::cerr << "[TextureSystem] Ignoring animation data with zero frame size" << std::endl;
        return;
    }
    const uint8_t* ptr = data;
    const uint8_t* end = data + totalBytes;

    while (ptr + 4 + frameBytes <= end) {
        uint16_t spriteId = *reinterpret_cast<const uint16_t*>(ptr);
        uint16_t frameIndex = *reinterpret_cast<const uint16_t*>(ptr + 2);
        ptr += 4;

        if (spriteId >= sprites_.size()) {
            std::cerr << "[TextureSystem] Animation spriteId " << spriteId
                      << " out of range (" << sprites_.size() << "), skipping" << std::endl;
            ptr += frameBytes;
            continue;
        }

        auto& meta = sprites_[spriteId];

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
    if (ptr != end) {
        std::cerr << "[TextureSystem] Animation payload has " << (end - ptr)
                  << " trailing bytes after fixed-layer parsing" << std::endl;
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

    std::vector<uint32_t> oldArrayIds;
    appendArrayId(oldArrayIds, blockAlbedoArrayId_.load(std::memory_order_acquire));
    appendArrayId(oldArrayIds, blockSpecularArrayId_.load(std::memory_order_acquire));
    appendArrayId(oldArrayIds, blockNormalArrayId_.load(std::memory_order_acquire));
    appendArrayId(oldArrayIds, blockFlagArrayId_.load(std::memory_order_acquire));

    auto framework = Renderer::instance().framework();
    if (!oldArrayIds.empty() || registry_.getBuffer() || materials_.getBuffer() || textureRules_.getBuffer()) {
        std::cout << "[TextureSystem] Re-initializing (preserving old published textures until swap)" << std::endl;
        waitForGpuIdleLocked(device, "texture reinitialization");
    }

    // Handle re-initialization (F3+T reload): retire old arrays before creating new ones.
/*
        std::cout << "[TextureSystem] Re-initializing (retiring old arrays)" << std::endl;
        // Wait all queues and keep old arrays/SSBO alive through frame GC.
        // Without this, in-flight command buffers still reference old texture arrays
        // via descriptors — destroying them causes GPU access violation (exit -805306369).
        retireGpuResourcesLocked(device, "texture reinitialization");
*/

    uint32_t count = static_cast<uint32_t>(sprites_.size());
    uint32_t spriteSize = layerSize_;
    if (spriteSize == 0 && !spritePixels_.empty() && count > 0) {
        size_t bytesPerLayer = spritePixels_.size() / count;
        size_t pixelsPerLayer = bytesPerLayer / 4;
        uint32_t inferredLayerSize = static_cast<uint32_t>(
            std::sqrt(static_cast<double>(pixelsPerLayer)));
        if (static_cast<size_t>(inferredLayerSize) * inferredLayerSize == pixelsPerLayer) {
            spriteSize = inferredLayerSize;
            layerSize_ = inferredLayerSize;
        }
    }
    const bool metadataOnlyFallbackArrays = spriteSize == 1 && !spritePixels_.empty();
    if (spriteSize == 0) {
        std::cerr << "[TextureSystem] Cannot finalize: no valid fixed sprite layer size" << std::endl;
        return;
    }

    // Validate against hardware limit
    auto physDevice = framework->physicalDevice();
    VkPhysicalDeviceProperties props = physDevice->properties();
    uint32_t maxLayers = props.limits.maxImageArrayLayers;

    uint32_t maxSupportedSprites = std::min(maxLayers, vk::Data::SPRITE_MAX_ENTRIES);
    if (count > maxSupportedSprites) {
        std::cerr << "[TextureSystem] WARNING: " << count
                  << " sprites exceeds supported texture-array registry layers="
                  << maxSupportedSprites
                  << ". Clamping." << std::endl;
        count = maxSupportedSprites;
    }

    std::cout << "[TextureSystem] Finalizing: " << count << " sprites, "
              << spriteSize << "x" << spriteSize
              << (metadataOnlyFallbackArrays ? " fallback arrays" : "")
              << ", max layers=" << maxLayers << std::endl;

    uint32_t newAlbedoArrayId = UINT32_MAX;
    uint32_t newSpecularArrayId = UINT32_MAX;
    uint32_t newNormalArrayId = UINT32_MAX;
    uint32_t newFlagArrayId = UINT32_MAX;
    std::vector<uint32_t> newArrayIds;

    // Create block albedo texture array
    newAlbedoArrayId = arrayManager_.createArray(
        vma, device, spriteSize, count,
        VK_FORMAT_R8G8B8A8_SRGB, true /* generateMips */);
    appendArrayId(newArrayIds, newAlbedoArrayId);

    // Stage frame-0 pixels for each sprite into the texture array.
    // Pixels are concatenated in sorted order in spritePixels_.
    // Java uses FIXED offset: i * spriteSize * spriteSize * 4 (all sprites assumed same size).
    size_t bytesPerSprite = static_cast<size_t>(spriteSize) * spriteSize * 4;
    std::vector<uint8_t> defaultAlbedo(bytesPerSprite, 255);
    const std::vector<uint8_t> defaultSpecular(bytesPerSprite, 0);
    std::vector<uint8_t> defaultNormal(bytesPerSprite, 0);
    std::vector<uint8_t> defaultFlag(bytesPerSprite, 0);
    for (size_t px = 0; px + 3 < defaultNormal.size(); px += 4) {
        defaultNormal[px + 0] = 128;
        defaultNormal[px + 1] = 128;
        defaultNormal[px + 2] = 255;
        defaultNormal[px + 3] = 255;
    }
    albedoChecksums_.assign(count, 0);
    specularChecksums_.assign(count, 0);
    normalChecksums_.assign(count, 0);
    flagChecksums_.assign(count, 0);
    uint32_t stagedAlbedoLayers = 0;
    uint32_t stagedSpecularLayers = 0;
    uint32_t stagedNormalLayers = 0;
    uint32_t stagedFlagLayers = 0;
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

        const uint8_t* layer = frameData ? frameData : defaultAlbedo.data();
        if (arrayManager_.stageLayerPixels(newAlbedoArrayId, i, 0, layer, bytesPerSprite)) {
            stagedAlbedoLayers++;
            if (frameData) {
                albedoChecksums_[i] = fnv1a64(frameData, bytesPerSprite);
            }
        }
    }

    std::cout << "[TextureSystem] Staged " << stagedAlbedoLayers << " / " << count
              << " albedo layers" << std::endl;

    // Create block specular texture array (UNORM — LabPBR values are linear, NOT sRGB)
    newSpecularArrayId = arrayManager_.createArray(
        vma, device, spriteSize, count,
        VK_FORMAT_R8G8B8A8_UNORM, true /* generateMips */);
    appendArrayId(newArrayIds, newSpecularArrayId);
    for (uint32_t i = 0; i < count; i++) {
        size_t pixelOffset = static_cast<size_t>(i) * bytesPerSprite;
        const bool hasLayer = pixelOffset + bytesPerSprite <= specularPixels_.size();
        const uint8_t* layer = hasLayer ? specularPixels_.data() + pixelOffset : defaultSpecular.data();
        if (arrayManager_.stageLayerPixels(newSpecularArrayId, i, 0,
                                           layer, bytesPerSprite)) {
            stagedSpecularLayers++;
            if (hasLayer) {
                specularChecksums_[i] = fnv1a64(layer, bytesPerSprite);
            }
        }
    }
    std::cout << "[TextureSystem] Staged " << stagedSpecularLayers << " / " << count
              << " specular layers" << std::endl;

    // Create block normal texture array (UNORM — linear normal map data)
    newNormalArrayId = arrayManager_.createArray(
        vma, device, spriteSize, count,
        VK_FORMAT_R8G8B8A8_UNORM, true /* generateMips */);
    appendArrayId(newArrayIds, newNormalArrayId);
    for (uint32_t i = 0; i < count; i++) {
        size_t pixelOffset = static_cast<size_t>(i) * bytesPerSprite;
        const bool hasLayer = pixelOffset + bytesPerSprite <= normalPixels_.size();
        const uint8_t* layer = hasLayer ? normalPixels_.data() + pixelOffset : defaultNormal.data();
        if (arrayManager_.stageLayerPixels(newNormalArrayId, i, 0,
                                           layer, bytesPerSprite)) {
            stagedNormalLayers++;
            if (hasLayer) {
                normalChecksums_[i] = fnv1a64(layer, bytesPerSprite);
            }
        }
    }
    std::cout << "[TextureSystem] Staged " << stagedNormalLayers << " / " << count
              << " normal layers" << std::endl;

    // Create block flag texture array (UNORM raw LabPBR flag bytes)
    newFlagArrayId = arrayManager_.createArray(
        vma, device, spriteSize, count,
        VK_FORMAT_R8G8B8A8_UNORM, true /* generateMips */);
    appendArrayId(newArrayIds, newFlagArrayId);
    for (uint32_t i = 0; i < count; i++) {
        size_t pixelOffset = static_cast<size_t>(i) * bytesPerSprite;
        const bool hasLayer = pixelOffset + bytesPerSprite <= flagPixels_.size();
        const uint8_t* layer = hasLayer ? flagPixels_.data() + pixelOffset : defaultFlag.data();
        if (arrayManager_.stageLayerPixels(newFlagArrayId, i, 0,
                                           layer, bytesPerSprite)) {
            stagedFlagLayers++;
            if (hasLayer) {
                flagChecksums_[i] = fnv1a64(layer, bytesPerSprite);
            }
        }
    }
    std::cout << "[TextureSystem] Staged " << stagedFlagLayers << " / " << count
              << " flag layers" << std::endl;

    const uint32_t minRequiredLayers = std::max<uint32_t>(1, count / 2);
    const bool albedoReady = stagedAlbedoLayers >= minRequiredLayers;
    const bool specularReady = stagedSpecularLayers >= minRequiredLayers;
    const bool normalReady = stagedNormalLayers >= minRequiredLayers;
    const bool flagReady = stagedFlagLayers >= minRequiredLayers;
    if (!albedoReady || !specularReady || !normalReady || !flagReady) {
        std::cerr << "[TextureSystem] Cannot finalize: staged layer counts too low"
                  << " albedo=" << stagedAlbedoLayers << "/" << count
                  << " specular=" << stagedSpecularLayers << "/" << count
                  << " normal=" << stagedNormalLayers << "/" << count
                  << " flag=" << stagedFlagLayers << "/" << count << std::endl;
        if (framework) {
            arrayManager_.retireArrays(framework->gc(), newArrayIds);
        }
        return;
    }

    // Build SpriteRegistry SSBO entries. Authored normal alpha remains shader metadata only.
    registry_.clearEntries();
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
        if (meta.padding & vk::Data::SPRITE_FLAG_HAS_SPECULAR) {
            flags |= vk::Data::SPRITE_FLAG_HAS_SPECULAR;
        }
        if (meta.padding & vk::Data::SPRITE_FLAG_HAS_NORMAL) {
            flags |= vk::Data::SPRITE_FLAG_HAS_NORMAL;
        }
        if (meta.padding & vk::Data::SPRITE_FLAG_HAS_HEIGHT) {
            flags |= vk::Data::SPRITE_FLAG_HAS_HEIGHT;
        }
        if (meta.padding & vk::Data::SPRITE_FLAG_EMISSIVE_OVERLAY) {
            flags |= vk::Data::SPRITE_FLAG_EMISSIVE_OVERLAY;
        }
        flags |= (meta.padding &
            (vk::Data::SPRITE_FLAG_SPEC_SOURCE_MASK | vk::Data::SPRITE_FLAG_NORMAL_SOURCE_MASK));
        uint32_t normalSource =
            (flags >> vk::Data::SPRITE_FLAG_NORMAL_SOURCE_SHIFT) & vk::Data::SPRITE_FLAG_SOURCE_MASK;
        bool authoredHeightSource =
            normalSource == vk::Data::SPRITE_SOURCE_PACK_AUTHORED ||
            normalSource == vk::Data::SPRITE_SOURCE_USER_CUSTOM;

        // All sprites get a layer in aux arrays (defaults for missing)
        int32_t specLayer = (newSpecularArrayId != UINT32_MAX) ? static_cast<int32_t>(i) : -1;
        int32_t normLayer = (newNormalArrayId != UINT32_MAX) ? static_cast<int32_t>(i) : -1;
        int32_t heightRangePacked = -1;
        if ((flags & vk::Data::SPRITE_FLAG_HAS_HEIGHT) != 0 && authoredHeightSource && normLayer >= 0) {
            size_t pixelOffset = static_cast<size_t>(i) * bytesPerSprite;
            if (pixelOffset + bytesPerSprite <= normalPixels_.size()) {
                const uint8_t* layer = normalPixels_.data() + pixelOffset;
                const uint8_t* albedoLayer =
                    pixelOffset + bytesPerSprite <= spritePixels_.size()
                        ? spritePixels_.data() + pixelOffset
                        : nullptr;
                uint8_t minAlpha = 255;
                uint8_t maxAlpha = 0;
                bool foundOpaquePixel = false;
                for (size_t px = 0; px < bytesPerSprite; px += 4) {
                    if (albedoLayer && albedoLayer[px + 3] == 0) {
                        continue;
                    }
                    uint8_t alpha = layer[px + 3];
                    minAlpha = std::min(minAlpha, alpha);
                    maxAlpha = std::max(maxAlpha, alpha);
                    foundOpaquePixel = true;
                }
                if (foundOpaquePixel && maxAlpha > minAlpha) {
                    heightRangePacked = static_cast<int32_t>(minAlpha) |
                                        (static_cast<int32_t>(maxAlpha) << 8);
                }
            }
        }

        registry_.registerSprite(
            static_cast<uint16_t>(i),
            i,                                          // baseLayer = spriteId
            1,                                          // frameCount=1 (animation via re-upload)
            std::max<uint32_t>(meta.tickRate, 1),       // tickRate
            flags,                                      // aux texture flags
            specLayer,                                  // specularLayer
            normLayer,                                  // normalLayer
            overlaySprite,                              // overlaySprite
            heightRangePacked);                         // packed height range
    }

    if (!registry_.uploadSSBO(vma, device)) {
        std::cerr << "[TextureSystem] Cannot finalize: sprite registry upload failed" << std::endl;
        if (framework) {
            arrayManager_.retireArrays(framework->gc(), newArrayIds);
        }
        return;
    }
    std::vector<vk::Data::TextureRuleEntry> emptyRules(vk::Data::SPRITE_MAX_ENTRIES);
    if (!textureRules_.uploadRules(emptyRules.data(), static_cast<uint32_t>(emptyRules.size()), vma, device)) {
        std::cerr << "[TextureSystem] WARNING: default texture rule upload failed" << std::endl;
    }
    std::vector<vk::Data::MaterialEntry> wrappedMaterials(count);
    for (uint32_t i = 0; i < count; i++) {
        auto* se = registry_.getEntry(static_cast<uint16_t>(i));
        auto& material = wrappedMaterials[i];
        material.materialId = i;
        material.baseSpriteId = static_cast<int32_t>(i);
        material.fallbackMaterialId = 0;
        material.flags = vk::Data::MATERIAL_FLAG_VALID |
                         vk::Data::MATERIAL_FLAG_VANILLA_SPRITE |
                         vk::Data::MATERIAL_FLAG_GPU_RESIDENT;
        if (se && se->specularLayer >= 0) material.flags |= vk::Data::MATERIAL_FLAG_HAS_SPECULAR;
        if (se && se->normalLayer >= 0) material.flags |= vk::Data::MATERIAL_FLAG_HAS_NORMAL;
        if (se && (se->flags & vk::Data::SPRITE_FLAG_HAS_HEIGHT) != 0u) {
            material.flags |= vk::Data::MATERIAL_FLAG_DISPLACEMENT_ELIGIBLE;
        }
        material.albedoPage = 0;
        material.albedoLayer = static_cast<int32_t>(i);
        material.specularPage = 0;
        material.specularLayer = se ? se->specularLayer : -1;
        material.normalPage = 0;
        material.normalLayer = se ? se->normalLayer : -1;
        material.flagPage = 0;
        material.flagLayer = static_cast<int32_t>(i);
        material.overlayMaterialId = se ? se->overlaySprite : -1;
        material.displacementPolicy =
            (material.flags & vk::Data::MATERIAL_FLAG_DISPLACEMENT_ELIGIBLE) != 0u
                ? vk::Data::MATERIAL_DISPLACEMENT_AUTHORED_HEIGHT
                : vk::Data::MATERIAL_DISPLACEMENT_DISABLED;
        material.displacementScale =
            material.displacementPolicy == vk::Data::MATERIAL_DISPLACEMENT_AUTHORED_HEIGHT ? 1.0f : 0.0f;
        material.heightRangePacked = se ? se->maskLayer : -1;
        material.uvScaleU = 1.0f;
        material.uvScaleV = 1.0f;
        material.uvOffsetU = 0.0f;
        material.uvOffsetV = 0.0f;
    }
    if (!materials_.uploadMaterials(wrappedMaterials.data(), static_cast<uint32_t>(wrappedMaterials.size()),
                                    vma, device)) {
        std::cerr << "[TextureSystem] WARNING: default material table upload failed" << std::endl;
    }

    // Free CPU pixel data (no longer needed after staging)
    spritePixels_.clear();
    spritePixels_.shrink_to_fit();
    specularPixels_.clear();
    specularPixels_.shrink_to_fit();
    normalPixels_.clear();
    normalPixels_.shrink_to_fit();
    flagPixels_.clear();
    flagPixels_.shrink_to_fit();

    blockAlbedoArrayId_.store(newAlbedoArrayId, std::memory_order_release);
    blockSpecularArrayId_.store(newSpecularArrayId, std::memory_order_release);
    blockNormalArrayId_.store(newNormalArrayId, std::memory_order_release);
    blockFlagArrayId_.store(newFlagArrayId, std::memory_order_release);
    resetMaterialTexturePagesLocked();
    materialAlbedoPageArrayIds_[0].store(newAlbedoArrayId, std::memory_order_release);
    materialSpecularPageArrayIds_[0].store(newSpecularArrayId, std::memory_order_release);
    materialNormalPageArrayIds_[0].store(newNormalArrayId, std::memory_order_release);
    materialFlagPageArrayIds_[0].store(newFlagArrayId, std::memory_order_release);
    materialPageReady_[0].store(true, std::memory_order_release);
    materialTexturePageRevision_.fetch_add(1, std::memory_order_acq_rel);
    albedoMipsInitialized_ = false;
    specMipsInitialized_ = false;
    normMipsInitialized_ = false;
    flagMipsInitialized_ = false;
    finalized_ = true;

    if (framework) {
        arrayManager_.retireArrays(framework->gc(), oldArrayIds);
    }

    // Diagnostic log
    {
        std::ofstream diag("C:/RadSER/texture_system_diag.log", std::ios::trunc);
        diag << "=== TextureSystem Diagnostic ===" << std::endl;
        diag << "Sprites: " << count << " (" << spriteSize << "x" << spriteSize << ")" << std::endl;
        diag << "Atlas: " << atlasWidth_ << "x" << atlasHeight_ << std::endl;
        diag << "Animated: " << animEntries_.size() << std::endl;
        diag << "Array IDs: albedo=" << blockAlbedoArrayId_
             << ", specular=" << blockSpecularArrayId_
             << ", normal=" << blockNormalArrayId_
             << ", flag=" << blockFlagArrayId_ << std::endl;
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
                 << " heightRange=0x" << std::hex << (se ? se->maskLayer : -1) << std::dec
                 << " UV=[" << b.minU << "," << b.maxU << "]x[" << b.minV << "," << b.maxV << "]"
                 << " overlay=" << (se ? se->overlaySprite : -2)
                 << std::endl;
        }
        diag.close();
    }

    // Free animation pixel data when animation updates are disabled (default).
    // The animation frames are only needed for tickAnimation() which is gated
    // behind textureArrayAnimationUpdatesEnabled. Keeping 51 animated sprite
    // frames (potentially MB of pixel data) in CPU memory is wasteful when
    // animation is frozen. If animation is later enabled, a resource reload
    // (F3+T) will repopulate the data.
    if (!Renderer::options.textureArrayAnimationUpdatesEnabled) {
        for (auto& ae : animEntries_) {
            for (auto& frame : ae.frames) {
                frame.clear();
                frame.shrink_to_fit();
            }
        }
        std::cout << "[TextureSystem] Freed animation pixel data (animation updates disabled)" << std::endl;
    }

    std::cout << "[TextureSystem] Finalized. " << count << " sprites ready, "
        << animEntries_.size() << " animated." << std::endl;
}

bool TextureSystem::tickAnimation(uint32_t gameTick, uint64_t generation) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (generation != 0 && generation != generation_.load(std::memory_order_acquire)) return false;
    if (!finalized_ || animEntries_.empty()) return false;
    uint32_t albedoArrayId = blockAlbedoArrayId_.load(std::memory_order_acquire);
    if (albedoArrayId == UINT32_MAX) return false;

    // Per-frame animation upload budget: limit total bytes staged per tick
    // to prevent 51 animated sprites from causing a single-frame upload spike.
    // At 64x64 sprites (16KB each), 8 sprites = 128KB — reasonable per frame.
    // Sprites that exceed the budget are deferred to the next tick.
    constexpr size_t kAnimBudgetBytes = 128 * 1024;  // 128 KB per frame
    size_t budgetUsed = 0;

    bool anyUpdated = false;

    for (auto& ae : animEntries_) {
        if (ae.frames.empty()) continue;

        uint32_t frameCount = static_cast<uint32_t>(ae.frames.size());
        uint32_t tickRate = std::max<uint32_t>(ae.tickRate, 1);
        uint32_t currentFrame = (gameTick / tickRate) % frameCount;

        if (currentFrame != ae.lastFrame && currentFrame < ae.frames.size()
            && !ae.frames[currentFrame].empty()) {
            size_t frameBytes = ae.frames[currentFrame].size();
            if (budgetUsed + frameBytes > kAnimBudgetBytes) {
                // Budget exceeded — defer this sprite to next tick.
                // Don't update lastFrame so it will be re-evaluated next tick.
                continue;
            }

            if (arrayManager_.stageLayerPixels(
                    albedoArrayId, ae.spriteId, 0,
                    ae.frames[currentFrame].data(),
                    ae.frames[currentFrame].size())) {
                ae.lastFrame = currentFrame;
                budgetUsed += frameBytes;
                anyUpdated = true;
            }
        }
    }

    return anyUpdated;
}

void TextureSystem::flushPendingUploads(std::shared_ptr<vk::VMA> vma,
	std::shared_ptr<vk::Device> device,
	std::shared_ptr<vk::CommandBuffer> cmdBuffer,
	GarbageCollector& gc) {
    std::lock_guard<std::mutex> lock(mutex_);
	if (!finalized_) return;
    const bool hasPendingUploads = arrayManager_.hasPendingUploads();
	if (!hasPendingUploads && !hasMaterialPageMipsDirtyLocked()) return;

	// Flush all staged uploads (snapshot-and-process: mutex held only ~1us)
	TextureArrayManager::DirtyLayers dirty{};
    if (hasPendingUploads) {
        dirty = arrayManager_.flushUploads(
            vma, device, cmdBuffer,
            blockAlbedoArrayId_, blockSpecularArrayId_, blockNormalArrayId_, blockFlagArrayId_);
    }
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
			arrayManager_.generateMipmapsForLayers(arrayId, layers, cmdBuffer);
		}
	};
    mipgen(blockAlbedoArrayId_, dirty.albedo, albedoMipsInitialized_);
    mipgen(blockSpecularArrayId_, dirty.specular, specMipsInitialized_);
    mipgen(blockNormalArrayId_, dirty.normal, normMipsInitialized_);
    mipgen(blockFlagArrayId_, dirty.flag, flagMipsInitialized_);

    for (uint32_t page = 1; page < vk::Data::MATERIAL_TEXTURE_PAGE_MAX; page++) {
        if (!materialPageMipsDirty_[page]) continue;
        uint32_t albedoId = materialAlbedoPageArrayIds_[page].load(std::memory_order_acquire);
        uint32_t specId = materialSpecularPageArrayIds_[page].load(std::memory_order_acquire);
        uint32_t normalId = materialNormalPageArrayIds_[page].load(std::memory_order_acquire);
        uint32_t flagId = materialFlagPageArrayIds_[page].load(std::memory_order_acquire);
        if (albedoId != UINT32_MAX) arrayManager_.generateMipmaps(albedoId, cmdBuffer);
        if (specId != UINT32_MAX) arrayManager_.generateMipmaps(specId, cmdBuffer);
        if (normalId != UINT32_MAX) arrayManager_.generateMipmaps(normalId, cmdBuffer);
        if (flagId != UINT32_MAX) arrayManager_.generateMipmaps(flagId, cmdBuffer);
        materialPageMipsDirty_[page] = false;
        materialPageReady_[page].store(true, std::memory_order_release);
        materialTexturePageRevision_.fetch_add(1, std::memory_order_acq_rel);
    }
}

const TextureSystem::SpriteBounds& TextureSystem::getSpriteBounds(uint16_t spriteId) const {
    if (spriteId < spriteBounds_.size()) {
        return spriteBounds_[spriteId];
    }
    return DEFAULT_BOUNDS;
}

void TextureSystem::setGeneration(uint64_t generation) {
    std::lock_guard<std::mutex> lock(mutex_);
    uint64_t current = generation_.load(std::memory_order_acquire);
    if (generation >= current) {
        generation_.store(generation, std::memory_order_release);
    }
}

bool TextureSystem::stageLayerUpdate(LayerKind layerKind, uint32_t spriteId,
                                     const uint8_t* pixels, size_t pixelSize,
                                     uint64_t generation) {
    if (!pixels || pixelSize == 0) return false;

    std::lock_guard<std::mutex> lock(mutex_);
    if (generation != 0 && generation != generation_.load(std::memory_order_acquire)) return false;
    if (!finalized_ || spriteId >= sprites_.size()) return false;

    uint32_t arrayId = UINT32_MAX;
    switch (layerKind) {
        case LayerKind::Albedo:
            arrayId = blockAlbedoArrayId_.load(std::memory_order_acquire);
            break;
        case LayerKind::Specular:
            arrayId = blockSpecularArrayId_.load(std::memory_order_acquire);
            break;
        case LayerKind::Normal:
            arrayId = blockNormalArrayId_.load(std::memory_order_acquire);
            break;
        case LayerKind::Flag:
            arrayId = blockFlagArrayId_.load(std::memory_order_acquire);
            break;
    }
    if (arrayId == UINT32_MAX) return false;

    return arrayManager_.stageLayerPixels(arrayId, spriteId, 0, pixels, pixelSize);
}

bool TextureSystem::updateSpriteHeightMetadata(uint32_t spriteId, uint32_t flags, int32_t maskLayer,
                                               uint64_t generation,
                                               std::shared_ptr<vk::VMA> vma,
                                               std::shared_ptr<vk::Device> device) {
    if (!vma || !device) return false;

    std::lock_guard<std::mutex> lock(mutex_);
    if (generation != 0 && generation != generation_.load(std::memory_order_acquire)) return false;
    if (!finalized_ || spriteId >= sprites_.size()) return false;
    if (!registry_.updateHeightMetadata(static_cast<uint16_t>(spriteId), flags, maskLayer)) {
        return false;
    }
    return registry_.uploadSSBO(std::move(vma), std::move(device));
}

bool TextureSystem::receiveSparseAuxBatch(const SparseAuxUpdate* updates, uint32_t updateCount,
                                          const uint8_t* pixels, size_t pixelBytes,
                                          const SparseAuxMetadata* metadata, uint32_t metadataCount,
                                          uint64_t generation,
                                          std::shared_ptr<vk::VMA> vma,
                                          std::shared_ptr<vk::Device> device) {
    if (!updates || updateCount == 0 || !pixels || pixelBytes == 0 || !vma || !device) return false;

    std::lock_guard<std::mutex> lock(mutex_);
    if (generation != 0 && generation != generation_.load(std::memory_order_acquire)) return false;
    if (!finalized_ || sprites_.empty() || layerSize_ == 0) return false;

    uint32_t specularArrayId = blockSpecularArrayId_.load(std::memory_order_acquire);
    uint32_t normalArrayId = blockNormalArrayId_.load(std::memory_order_acquire);
    uint32_t flagArrayId = blockFlagArrayId_.load(std::memory_order_acquire);
    const size_t bytesPerLayer = static_cast<size_t>(layerSize_) * layerSize_ * 4u;
    if (bytesPerLayer == 0) return false;

    auto validPayload = [&](int32_t offset) {
        if (offset < 0) return false;
        size_t byteOffset = static_cast<size_t>(offset);
        return byteOffset <= pixelBytes && bytesPerLayer <= pixelBytes - byteOffset;
    };

    uint32_t stagedSpecular = 0;
    uint32_t stagedNormal = 0;
    uint32_t stagedFlag = 0;
    constexpr uint32_t CHANNEL_SPECULAR = 1u;
    constexpr uint32_t CHANNEL_NORMAL = 2u;
    constexpr uint32_t CHANNEL_FLAG = 4u;

    for (uint32_t i = 0; i < updateCount; i++) {
        const auto& update = updates[i];
        if (update.spriteId < 0 || static_cast<uint32_t>(update.spriteId) >= sprites_.size()) {
            continue;
        }
        uint32_t spriteId = static_cast<uint32_t>(update.spriteId);
        if ((update.channelMask & CHANNEL_SPECULAR) != 0u) {
            if (specularArrayId == UINT32_MAX || !validPayload(update.specularOffset)) return false;
            if (arrayManager_.stageLayerPixels(specularArrayId, spriteId, 0,
                    pixels + static_cast<size_t>(update.specularOffset), bytesPerLayer)) {
                stagedSpecular++;
            }
        }
        if ((update.channelMask & CHANNEL_NORMAL) != 0u) {
            if (normalArrayId == UINT32_MAX || !validPayload(update.normalOffset)) return false;
            if (arrayManager_.stageLayerPixels(normalArrayId, spriteId, 0,
                    pixels + static_cast<size_t>(update.normalOffset), bytesPerLayer)) {
                stagedNormal++;
            }
        }
        if ((update.channelMask & CHANNEL_FLAG) != 0u) {
            if (flagArrayId == UINT32_MAX || !validPayload(update.flagOffset)) return false;
            if (arrayManager_.stageLayerPixels(flagArrayId, spriteId, 0,
                    pixels + static_cast<size_t>(update.flagOffset), bytesPerLayer)) {
                stagedFlag++;
            }
        }
    }

    uint32_t metadataApplied = 0;
    if (metadata && metadataCount > 0) {
        for (uint32_t i = 0; i < metadataCount; i++) {
            const auto& entry = metadata[i];
            if (entry.spriteId < 0 || static_cast<uint32_t>(entry.spriteId) >= sprites_.size()) {
                continue;
            }
            if (registry_.updateHeightMetadata(static_cast<uint16_t>(entry.spriteId),
                    entry.flags, entry.heightRangePacked)) {
                metadataApplied++;
            }
        }
        if (metadataApplied > 0 && !registry_.uploadSSBO(vma, device)) {
            return false;
        }
    }

    std::cout << "[TextureSystem] Sparse aux batch staged: updates=" << updateCount
              << " bytes=" << pixelBytes
              << " metadataUpdates=" << metadataApplied
              << " specular=" << stagedSpecular
              << " normal=" << stagedNormal
              << " flag=" << stagedFlag << std::endl;
    return true;
}

bool TextureSystem::uploadTextureRules(const vk::Data::TextureRuleEntry* entries, uint32_t count,
                                       uint64_t generation,
                                       std::shared_ptr<vk::VMA> vma,
                                       std::shared_ptr<vk::Device> device) {
    if (!entries || count == 0 || !vma || !device) return false;

    std::lock_guard<std::mutex> lock(mutex_);
    if (generation != 0 && generation != generation_.load(std::memory_order_acquire)) return false;
    if (!finalized_) return false;
    return textureRules_.uploadRules(entries, count, std::move(vma), std::move(device));
}

bool TextureSystem::uploadMaterialTable(const vk::Data::MaterialEntry* entries, uint32_t count,
                                        uint64_t generation,
                                        std::shared_ptr<vk::VMA> vma,
                                        std::shared_ptr<vk::Device> device) {
    if (!entries || count == 0 || !vma || !device) return false;

    std::lock_guard<std::mutex> lock(mutex_);
    if (generation != 0 && generation != generation_.load(std::memory_order_acquire)) return false;
    if (!finalized_) return false;
    return materials_.uploadMaterials(entries, count, std::move(vma), std::move(device));
}

bool TextureSystem::updateMaterialTableSparse(const vk::Data::MaterialEntry* entries, uint32_t count,
                                              uint64_t generation,
                                              std::shared_ptr<vk::VMA> vma,
                                              std::shared_ptr<vk::Device> device) {
    if (!entries || count == 0 || !vma || !device) return false;

    std::lock_guard<std::mutex> lock(mutex_);
    if (generation != 0 && generation != generation_.load(std::memory_order_acquire)) return false;
    if (!finalized_) return false;
    return materials_.updateMaterialsSparse(entries, count, std::move(vma), std::move(device));
}

bool TextureSystem::uploadMaterialTexturePage(uint32_t page, uint32_t spriteSize, uint32_t layerCount,
                                              const uint8_t* albedoData,
                                              const uint8_t* specularData,
                                              const uint8_t* normalData,
                                              const uint8_t* flagData,
                                              uint64_t generation,
                                              std::shared_ptr<vk::VMA> vma,
                                              std::shared_ptr<vk::Device> device) {
    return uploadMaterialTextureLayers(page, spriteSize, 0, layerCount, layerCount,
        albedoData, specularData, normalData, flagData, generation, std::move(vma), std::move(device));
}

bool TextureSystem::uploadMaterialTextureLayers(uint32_t page, uint32_t spriteSize, uint32_t startLayer,
                                                uint32_t layerCount, uint32_t layerCapacity,
                                                const uint8_t* albedoData,
                                                const uint8_t* specularData,
                                                const uint8_t* normalData,
                                                const uint8_t* flagData,
                                                uint64_t generation,
                                                std::shared_ptr<vk::VMA> vma,
                                                std::shared_ptr<vk::Device> device) {
    if (page == 0 || page >= vk::Data::MATERIAL_TEXTURE_PAGE_MAX) return false;
    if (spriteSize == 0 || layerCount == 0 || layerCapacity == 0) return false;
    if (startLayer >= layerCapacity || layerCount > layerCapacity - startLayer) return false;
    if (!albedoData || !specularData || !normalData || !flagData || !vma || !device) return false;
    const size_t spriteSizeBytes = static_cast<size_t>(spriteSize);
    if (spriteSizeBytes > std::numeric_limits<size_t>::max() / spriteSizeBytes
        || spriteSizeBytes * spriteSizeBytes > std::numeric_limits<size_t>::max() / 4u) {
        std::cerr << "[TextureSystem] Material page upload rejected: byte count overflow size="
                  << spriteSize << std::endl;
        return false;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    if (generation != 0 && generation != generation_.load(std::memory_order_acquire)) return false;
    if (!finalized_) return false;

    const size_t bytesPerLayer = spriteSizeBytes * spriteSizeBytes * 4u;
    if (bytesPerLayer == 0
        || layerCount > std::numeric_limits<size_t>::max() / bytesPerLayer) {
        std::cerr << "[TextureSystem] Material page upload rejected: upload byte count overflow layers="
                  << layerCount << " bytesPerLayer=" << bytesPerLayer << std::endl;
        return false;
    }

    std::vector<uint32_t> oldArrayIds;
    uint32_t albedoArrayId = materialAlbedoPageArrayIds_[page].load(std::memory_order_acquire);
    uint32_t specularArrayId = materialSpecularPageArrayIds_[page].load(std::memory_order_acquire);
    uint32_t normalArrayId = materialNormalPageArrayIds_[page].load(std::memory_order_acquire);
    uint32_t flagArrayId = materialFlagPageArrayIds_[page].load(std::memory_order_acquire);

    std::vector<uint32_t> newArrayIds;
    bool createdPageArrays = false;
    if (albedoArrayId == UINT32_MAX || specularArrayId == UINT32_MAX
        || normalArrayId == UINT32_MAX || flagArrayId == UINT32_MAX) {
        appendArrayId(oldArrayIds, albedoArrayId);
        appendArrayId(oldArrayIds, specularArrayId);
        appendArrayId(oldArrayIds, normalArrayId);
        appendArrayId(oldArrayIds, flagArrayId);
        albedoArrayId = arrayManager_.createArray(
            vma, device, spriteSize, layerCapacity, VK_FORMAT_R8G8B8A8_SRGB, true);
        specularArrayId = arrayManager_.createArray(
            vma, device, spriteSize, layerCapacity, VK_FORMAT_R8G8B8A8_UNORM, true);
        normalArrayId = arrayManager_.createArray(
            vma, device, spriteSize, layerCapacity, VK_FORMAT_R8G8B8A8_UNORM, true);
        flagArrayId = arrayManager_.createArray(
            vma, device, spriteSize, layerCapacity, VK_FORMAT_R8G8B8A8_UNORM, true);
        appendArrayId(newArrayIds, albedoArrayId);
        appendArrayId(newArrayIds, specularArrayId);
        appendArrayId(newArrayIds, normalArrayId);
        appendArrayId(newArrayIds, flagArrayId);
        createdPageArrays = true;
    }

    auto stageAllLayers = [&](uint32_t arrayId, const uint8_t* data) -> uint32_t {
        uint32_t staged = 0;
        for (uint32_t layer = 0; layer < layerCount; layer++) {
            const uint8_t* layerData = data + static_cast<size_t>(layer) * bytesPerLayer;
            if (arrayManager_.stageLayerPixels(arrayId, startLayer + layer, 0, layerData, bytesPerLayer)) {
                staged++;
            }
        }
        return staged;
    };

    const uint32_t stagedAlbedo = stageAllLayers(albedoArrayId, albedoData);
    const uint32_t stagedSpecular = stageAllLayers(specularArrayId, specularData);
    const uint32_t stagedNormal = stageAllLayers(normalArrayId, normalData);
    const uint32_t stagedFlag = stageAllLayers(flagArrayId, flagData);
    if (stagedAlbedo != layerCount || stagedSpecular != layerCount ||
        stagedNormal != layerCount || stagedFlag != layerCount) {
        auto* renderer = Renderer::try_instance();
        auto framework = renderer ? renderer->framework() : nullptr;
        if (framework) {
            arrayManager_.retireArrays(framework->gc(), newArrayIds);
        }
        std::cerr << "[TextureSystem] Material page upload failed for page " << page
                  << " staged albedo=" << stagedAlbedo << "/" << layerCount
                  << " specular=" << stagedSpecular << "/" << layerCount
                  << " normal=" << stagedNormal << "/" << layerCount
                  << " flag=" << stagedFlag << "/" << layerCount << std::endl;
        return false;
    }

    materialPageReady_[page].store(false, std::memory_order_release);
    materialAlbedoPageArrayIds_[page].store(albedoArrayId, std::memory_order_release);
    materialSpecularPageArrayIds_[page].store(specularArrayId, std::memory_order_release);
    materialNormalPageArrayIds_[page].store(normalArrayId, std::memory_order_release);
    materialFlagPageArrayIds_[page].store(flagArrayId, std::memory_order_release);
    materialPageLayerCapacity_[page] = std::max(materialPageLayerCapacity_[page], layerCapacity);
    materialPageLayersUsed_[page] = std::max(materialPageLayersUsed_[page], startLayer + layerCount);
    materialPageUpdates_++;
    if (createdPageArrays) {
        materialPageImageAllocations_++;
    }
    lastMaterialPage_ = page;
    lastMaterialPageStartLayer_ = startLayer;
    lastMaterialPageLayerCount_ = layerCount;
    lastMaterialPageLayerCapacity_ = layerCapacity;
    materialPageMipsDirty_[page] = true;
    materialTexturePageRevision_.fetch_add(1, std::memory_order_acq_rel);

    auto* renderer = Renderer::try_instance();
    auto framework = renderer ? renderer->framework() : nullptr;
    if (framework) {
        arrayManager_.retireArrays(framework->gc(), oldArrayIds);
    }

    if (createdPageArrays) {
        std::cout << "[TextureArrayManager] Created material page " << page
                  << ": " << spriteSize << "x" << spriteSize
                  << " x " << layerCapacity << " layers" << std::endl;
    }
    std::cout << "[TextureSystem] Updated material page " << page
              << " startLayer=" << startLayer
              << " layers=" << layerCount
              << " capacity=" << layerCapacity
              << " size=" << spriteSize << "x" << spriteSize << std::endl;
    return true;
}

std::string TextureSystem::materialPagePoolStatusJson() const {
    std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
    if (!lock.owns_lock()) {
        return "{\"schema\":\"radser_material_page_pool_status_v1\",\"busy\":true}";
    }
    uint32_t pagesAllocated = 0;
    uint32_t layersUsed = 0;
    uint32_t layersFree = 0;
    uint32_t allocatedLayers = 0;
    for (uint32_t page = 1; page < vk::Data::MATERIAL_TEXTURE_PAGE_MAX; page++) {
        uint32_t capacity = materialPageLayerCapacity_[page];
        if (capacity == 0 && materialAlbedoPageArrayIds_[page].load(std::memory_order_acquire) != UINT32_MAX) {
            capacity = materialPageLayersUsed_[page];
        }
        if (capacity == 0) continue;
        pagesAllocated++;
        allocatedLayers += capacity;
        uint32_t used = std::min(materialPageLayersUsed_[page], capacity);
        layersUsed += used;
        layersFree += capacity - used;
    }
    std::ostringstream out;
    out << "{"
        << "\"schema\":\"radser_material_page_pool_status_v1\","
        << "\"generation\":" << generation() << ","
        << "\"materialPagePools\":true,"
        << "\"pagesAllocated\":" << pagesAllocated << ","
        << "\"layersUsed\":" << layersUsed << ","
        << "\"layersFree\":" << layersFree << ","
        << "\"allocatedLayers\":" << allocatedLayers << ","
        << "\"updates\":" << materialPageUpdates_ << ","
        << "\"newPageImageAllocations\":" << materialPageImageAllocations_ << ","
        << "\"lastUpdatePage\":" << lastMaterialPage_ << ","
        << "\"lastUpdateStartLayer\":" << lastMaterialPageStartLayer_ << ","
        << "\"lastUpdateLayerCount\":" << lastMaterialPageLayerCount_ << ","
        << "\"lastUpdateLayerCapacity\":" << lastMaterialPageLayerCapacity_ << ","
        << "\"materialPageRevision\":" << materialTexturePageRevision()
        << "}";
    return out.str();
}

std::string TextureSystem::materialTableStatusJson() const {
    return materials_.statusJson();
}

std::string TextureSystem::nativeUploadSafetyStatusJson() const {
    std::ostringstream out;
    out << "{"
        << "\"schema\":\"radser_native_upload_safety_status_v1\","
        << "\"materialLayerRangeValidation\":true,"
        << "\"materialLayerPointerValidation\":true,"
        << "\"materialLayerByteOverflowValidation\":true,"
        << "\"invalidMaterialUploadsReturnFalse\":true,"
        << "\"generation\":" << generation()
        << "}";
    return out.str();
}

std::string TextureSystem::statusString() const {
    std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
    if (!lock.owns_lock()) return "texturesBusy:1,generation:" + std::to_string(generation());
    uint32_t materialPages = 0;
    for (uint32_t page = 1; page < vk::Data::MATERIAL_TEXTURE_PAGE_MAX; page++) {
        if (materialPageReady_[page].load(std::memory_order_acquire) &&
            materialAlbedoPageArrayIds_[page].load(std::memory_order_acquire) != UINT32_MAX) {
            materialPages++;
        }
    }
    std::ostringstream out;
    out << "finalized:" << (finalized_ ? 1 : 0)
        << ",generation:" << generation()
        << ",sprites:" << sprites_.size()
        << ",atlasWidth:" << atlasWidth_
        << ",atlasHeight:" << atlasHeight_
        << ",layerSize:" << layerSize_
        << ",animated:" << animEntries_.size()
        << ",materials:" << materials_.count()
        << ",materialPages:" << materialPages
        << ",materialPageRevision:" << materialTexturePageRevision()
        << ",albedoArray:" << blockAlbedoArrayId_
        << ",specularArray:" << blockSpecularArrayId_
        << ",normalArray:" << blockNormalArrayId_
        << ",flagArray:" << blockFlagArrayId_;
    return out.str();
}

bool TextureSystem::dumpDebug(const std::string& path, uint32_t limit) const {
    std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
    if (!lock.owns_lock()) return false;
    std::ofstream out(path, std::ios::trunc);
    if (!out.is_open()) return false;

    uint32_t count = static_cast<uint32_t>(sprites_.size());
    uint32_t n = limit == 0 ? count : std::min(limit, count);
    out << "TextureSystem generation=" << generation()
        << " finalized=" << finalized_
        << " sprites=" << count
        << " atlas=" << atlasWidth_ << "x" << atlasHeight_
        << " layerSize=" << layerSize_
        << "\n";
    out << "arrays albedo=" << blockAlbedoArrayId_
        << " specular=" << blockSpecularArrayId_
        << " normal=" << blockNormalArrayId_
        << " flag=" << blockFlagArrayId_ << "\n";
    out << "spriteId,atlasX,atlasY,width,height,frames,flags,specLayer,normalLayer,overlaySprite,heightRange,albedoHash,specHash,normalHash,flagHash\n";

    for (uint32_t i = 0; i < n; i++) {
        const auto& m = sprites_[i];
        const auto* se = registry_.getEntry(static_cast<uint16_t>(i));
        out << i << ','
            << m.atlasX << ','
            << m.atlasY << ','
            << m.width << ','
            << m.height << ','
            << m.frameCount << ','
            << (se ? se->flags : 0) << ','
            << (se ? se->specularLayer : -1) << ','
            << (se ? se->normalLayer : -1) << ','
            << (se ? se->overlaySprite : -1) << ','
            << (se ? se->maskLayer : -1) << ','
            << (i < albedoChecksums_.size() ? albedoChecksums_[i] : 0) << ','
            << (i < specularChecksums_.size() ? specularChecksums_[i] : 0) << ','
            << (i < normalChecksums_.size() ? normalChecksums_[i] : 0) << ','
            << (i < flagChecksums_.size() ? flagChecksums_[i] : 0)
            << "\n";
    }
    return true;
}

void TextureSystem::reset() {
    std::lock_guard<std::mutex> lock(mutex_);

    sprites_.clear();
    spriteBounds_.clear();
    spritePixels_.clear();
    specularPixels_.clear();
    normalPixels_.clear();
    flagPixels_.clear();
    albedoChecksums_.clear();
    specularChecksums_.clear();
    normalChecksums_.clear();
    flagChecksums_.clear();
    atlasWidth_ = 0;
    atlasHeight_ = 0;
    layerSize_ = 0;
    animEntries_.clear();
    retireGpuResourcesLocked(nullptr, "texture reset");

    std::cout << "[TextureSystem] Reset." << std::endl;
}
