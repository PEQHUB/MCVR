#include "core/render/texture_system.hpp"
#include "core/render/renderer.hpp"
#include "core/render/render_framework.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <cmath>
#include <sstream>

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

}

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
    if (spriteSize == 0) {
        std::cerr << "[TextureSystem] Cannot finalize: no valid fixed sprite layer size" << std::endl;
        return;
    }

    // Validate against hardware limit
    auto framework = Renderer::instance().framework();
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
              << ", max layers=" << maxLayers << std::endl;

    // Create block albedo texture array
    blockAlbedoArrayId_ = arrayManager_.createArray(
        vma, device, spriteSize, count,
        VK_FORMAT_R8G8B8A8_SRGB, true /* generateMips */);

    // Stage frame-0 pixels for each sprite into the texture array.
    // Pixels are concatenated in sorted order in spritePixels_.
    // Java uses FIXED offset: i * spriteSize * spriteSize * 4 (all sprites assumed same size).
    size_t bytesPerSprite = static_cast<size_t>(spriteSize) * spriteSize * 4;
    albedoChecksums_.assign(count, 0);
    specularChecksums_.assign(count, 0);
    normalChecksums_.assign(count, 0);
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
            albedoChecksums_[i] = fnv1a64(frameData, bytesPerSprite);
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
                const uint8_t* layer = specularPixels_.data() + pixelOffset;
                arrayManager_.stageLayerPixels(blockSpecularArrayId_, i, 0,
                    layer, bytesPerSprite);
                specularChecksums_[i] = fnv1a64(layer, bytesPerSprite);
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
                const uint8_t* layer = normalPixels_.data() + pixelOffset;
                arrayManager_.stageLayerPixels(blockNormalArrayId_, i, 0,
                    layer, bytesPerSprite);
                normalChecksums_[i] = fnv1a64(layer, bytesPerSprite);
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
        if (meta.padding & vk::Data::SPRITE_FLAG_HAS_SPECULAR) {
            flags |= vk::Data::SPRITE_FLAG_HAS_SPECULAR;
        }
        if (meta.padding & vk::Data::SPRITE_FLAG_HAS_NORMAL) {
            flags |= vk::Data::SPRITE_FLAG_HAS_NORMAL;
        }
        if (meta.padding & vk::Data::SPRITE_FLAG_HAS_HEIGHT) {
            flags |= vk::Data::SPRITE_FLAG_HAS_HEIGHT;
        }
        flags |= (meta.padding &
            (vk::Data::SPRITE_FLAG_SPEC_SOURCE_MASK | vk::Data::SPRITE_FLAG_NORMAL_SOURCE_MASK));

        // All sprites get a layer in aux arrays (defaults for missing)
        int32_t specLayer = (blockSpecularArrayId_ != UINT32_MAX) ? static_cast<int32_t>(i) : -1;
        int32_t normLayer = (blockNormalArrayId_ != UINT32_MAX) ? static_cast<int32_t>(i) : -1;
        int32_t heightRangePacked = -1;
        if ((flags & vk::Data::SPRITE_FLAG_HAS_HEIGHT) != 0 && normLayer >= 0) {
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

bool TextureSystem::tickAnimation(uint32_t gameTick) {
    if (!finalized_ || animEntries_.empty()) return false;

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

            arrayManager_.stageLayerPixels(
                blockAlbedoArrayId_, ae.spriteId, 0,
                ae.frames[currentFrame].data(),
                ae.frames[currentFrame].size());

            ae.lastFrame = currentFrame;
            budgetUsed += frameBytes;
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

	// Flush all staged uploads (snapshot-and-process: mutex held only ~1us)
	auto dirty = arrayManager_.flushUploads(
		vma, device, cmdBuffer,
		blockAlbedoArrayId_, blockSpecularArrayId_, blockNormalArrayId_);
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
}

const TextureSystem::SpriteBounds& TextureSystem::getSpriteBounds(uint16_t spriteId) const {
    if (spriteId < spriteBounds_.size()) {
        return spriteBounds_[spriteId];
    }
    return DEFAULT_BOUNDS;
}

std::string TextureSystem::statusString() const {
    std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
    if (!lock.owns_lock()) return "texturesBusy:1,generation:" + std::to_string(generation());
    std::ostringstream out;
    out << "finalized:" << (finalized_ ? 1 : 0)
        << ",generation:" << generation()
        << ",sprites:" << sprites_.size()
        << ",atlasWidth:" << atlasWidth_
        << ",atlasHeight:" << atlasHeight_
        << ",layerSize:" << layerSize_
        << ",animated:" << animEntries_.size()
        << ",albedoArray:" << blockAlbedoArrayId_
        << ",specularArray:" << blockSpecularArrayId_
        << ",normalArray:" << blockNormalArrayId_;
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
        << " layerSize=" << layerSize_ << "\n";
    out << "arrays albedo=" << blockAlbedoArrayId_
        << " specular=" << blockSpecularArrayId_
        << " normal=" << blockNormalArrayId_ << "\n";
    out << "spriteId,atlasX,atlasY,width,height,frames,flags,specLayer,normalLayer,overlaySprite,heightRange,albedoHash,specHash,normalHash\n";

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
            << (i < normalChecksums_.size() ? normalChecksums_[i] : 0)
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
    albedoChecksums_.clear();
    specularChecksums_.clear();
    normalChecksums_.clear();
    atlasWidth_ = 0;
    atlasHeight_ = 0;
    layerSize_ = 0;
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
