#pragma once

#include "core/render/sprite_registry.hpp"
#include "core/render/texture_rule_registry.hpp"
#include "core/render/texture_arrays.hpp"
#include "core/vulkan/all_core_vulkan.hpp"

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

class Framework;
class GarbageCollector;

/// Full C++ texture system. Single owner of the block texture pipeline.
///
/// Java sends (3 JNI calls + finalize):
///   1. Sorted sprite metadata table + atlas dimensions (one call)
///   2. Concatenated frame-0 pixel data for all sprites in sorted order (one call)
///   3. Bulk animation frame data for animated sprites (one call)
///   4. Finalize trigger
///
/// C++ owns:
///   - Texture array creation and management
///   - Deterministic spriteId assignment (= sorted index)
///   - SpriteRegistry SSBO for shader access
///   - Per-tick animation updates
///   - Mipmap generation
///   - Atlas UV bounds for model UV normalization
///
/// spriteId is simply the index in the sorted metadata table.
/// Java sorts sprites alphabetically by Identifier.toString() and sends them in order.
/// C++ trusts this ordering — no hash collisions, no mapping needed.
class TextureSystem {
  public:
    /// Sprite metadata received from Java (packed, in sorted order).
    /// Java writes these in Identifier-sorted order; spriteId = array index.
#pragma pack(push, 1)
    struct SpriteMetadata {
        uint16_t atlasX;       // pixel X in atlas (for UV normalization)
        uint16_t atlasY;       // pixel Y in atlas
        uint16_t width;        // sprite width (pixels)
        uint16_t height;       // sprite height (pixels)
        uint16_t frameCount;   // 1 = static, N = animated
        uint16_t tickRate;     // ticks per animation frame (0 treated as 1)
        int16_t  overlayOf;    // spriteId this is an overlay FOR (-1 = none)
        uint16_t padding;
    };
    static_assert(sizeof(SpriteMetadata) == 16, "SpriteMetadata must be 16 bytes");
#pragma pack(pop)

    /// UV bounds for a sprite within the atlas (computed from atlas pixel coords).
    struct SpriteBounds {
        float minU, maxU, minV, maxV;
    };

    enum class LayerKind {
        Albedo,
        Specular,
        Normal,
        Flag,
    };

    TextureSystem() = default;

    // ---- Data reception from Java (called on game thread via JNI) ----

    /// Receive sorted sprite metadata table with atlas dimensions for UV bound computation.
    /// spriteId = index in the table. Atlas dimensions used to convert pixel coords → UV [0,1].
    void receiveSpriteTable(const SpriteMetadata* table, uint32_t count,
                            uint32_t atlasWidth, uint32_t atlasHeight);

    /// Receive concatenated fixed-layer frame-0 pixel data for all sprites (sorted order).
    /// Java resamples every frame into the selected square texture-array layer size.
    /// Pixels are RGBA8, in the same order as the sprite table.
    void receiveSpritePixels(const uint8_t* data, uint32_t totalBytes);

    /// Receive concatenated specular + normal + flag pixel data for all sprites (sorted order).
    /// Each buffer is count * (layerSize * layerSize * 4) bytes, RGBA8 UNORM.
    void receiveAuxPixels(const uint8_t* specularData, const uint8_t* normalData,
                          const uint8_t* flagData,
                          uint32_t totalBytesPerType);

    /// Receive bulk animation frame data for all animated sprites.
    /// Format: repeated entries of
    /// [spriteId(uint16), frameIndex(uint16), pixels(layerSize*layerSize*4 bytes)].
    void receiveAnimationFrames(const uint8_t* data, uint32_t totalBytes);

    // ---- Finalization (called on game thread, creates GPU resources) ----

    /// Create texture array, stage all pixels, build SpriteRegistry SSBO.
    /// Requires renderer to be initialized (VMA + Device available).
    /// Mipmap generation and first flush happen on the render thread.
    void finalize(std::shared_ptr<vk::VMA> vma, std::shared_ptr<vk::Device> device);

    /// Whether finalize() has been called and textures are ready.
    bool isFinalized() const { return finalized_.load(std::memory_order_acquire); }

    // ---- Per-frame operations (called on render thread) ----

    /// Tick animation: stage changed frames for upload. Returns true if any layers changed.
    bool tickAnimation(uint32_t gameTick, uint64_t generation);

    /// Flush any pending texture uploads + selective mipgen.
    /// Must be called within a valid command buffer recording (render thread).
    void flushPendingUploads(std::shared_ptr<vk::VMA> vma,
                             std::shared_ptr<vk::Device> device,
                             std::shared_ptr<vk::CommandBuffer> cmdBuffer,
                             GarbageCollector& gc);

    // ---- Query (thread-safe after finalize) ----

    /// Get atlas UV bounds for a spriteId. Used during model table UV normalization.
    const SpriteBounds& getSpriteBounds(uint16_t spriteId) const;

    /// Sprite count.
    uint32_t spriteCount() const { return static_cast<uint32_t>(sprites_.size()); }
    uint32_t layerSize() const { return layerSize_; }
    uint32_t atlasWidth() const { return atlasWidth_; }
    uint32_t atlasHeight() const { return atlasHeight_; }

    void setGeneration(uint64_t generation);
    uint64_t generation() const { return generation_.load(std::memory_order_acquire); }

    std::string statusString() const;
    bool dumpDebug(const std::string& path, uint32_t limit) const;

    /// Access underlying managers for descriptor binding.
    TextureArrayManager& arrayManager() { return arrayManager_; }
    SpriteRegistry& registry() { return registry_; }
    TextureRuleRegistry& textureRules() { return textureRules_; }
    bool stageLayerUpdate(LayerKind layerKind, uint32_t spriteId,
                          const uint8_t* pixels, size_t pixelSize,
                          uint64_t generation);
    bool updateSpriteHeightMetadata(uint32_t spriteId, uint32_t flags, int32_t maskLayer,
                                    uint64_t generation,
                                    std::shared_ptr<vk::VMA> vma,
                                    std::shared_ptr<vk::Device> device);
    bool uploadTextureRules(const vk::Data::TextureRuleEntry* entries, uint32_t count,
                            uint64_t generation,
                            std::shared_ptr<vk::VMA> vma,
                            std::shared_ptr<vk::Device> device);

    /// Get texture array IDs (for descriptor binding).
    uint32_t blockAlbedoArrayId() const { return blockAlbedoArrayId_.load(std::memory_order_acquire); }
    uint32_t blockSpecularArrayId() const { return blockSpecularArrayId_.load(std::memory_order_acquire); }
    uint32_t blockNormalArrayId() const { return blockNormalArrayId_.load(std::memory_order_acquire); }
    uint32_t blockFlagArrayId() const { return blockFlagArrayId_.load(std::memory_order_acquire); }
    /// Reset on resource reload.
    void reset();

  private:
    void waitForGpuIdleLocked(std::shared_ptr<vk::Device> device, const char* reason);
    void retireGpuResourcesLocked(std::shared_ptr<vk::Device> device, const char* reason);

    // Sprite metadata (sorted by identifier, spriteId = index)
    std::vector<SpriteMetadata> sprites_;
    std::vector<SpriteBounds> spriteBounds_;
    uint32_t atlasWidth_ = 0;
    uint32_t atlasHeight_ = 0;
    uint32_t layerSize_ = 0;
    static const SpriteBounds DEFAULT_BOUNDS;

    // Frame-0 pixel data for all sprites (concatenated, RGBA8)
    std::vector<uint8_t> spritePixels_;

    // Auxiliary pixel data (specular + normal + flags, concatenated per sprite, RGBA8)
    std::vector<uint8_t> specularPixels_;
    std::vector<uint8_t> normalPixels_;
    std::vector<uint8_t> flagPixels_;
    std::vector<uint64_t> albedoChecksums_;
    std::vector<uint64_t> specularChecksums_;
    std::vector<uint64_t> normalChecksums_;
    std::vector<uint64_t> flagChecksums_;
    // Animation: per-sprite frame data (RGBA pixels per frame)
    struct AnimEntry {
        uint16_t spriteId;
        uint16_t tickRate;
        std::vector<std::vector<uint8_t>> frames; // [frameIndex] → RGBA pixels
        uint32_t lastFrame = UINT32_MAX;           // last displayed frame (for change detection)
    };
    std::vector<AnimEntry> animEntries_;

    // GPU resources (owned)
    TextureArrayManager arrayManager_;
    SpriteRegistry registry_;
    TextureRuleRegistry textureRules_;
    std::atomic<uint32_t> blockAlbedoArrayId_{UINT32_MAX};
    std::atomic<uint32_t> blockSpecularArrayId_{UINT32_MAX};
    std::atomic<uint32_t> blockNormalArrayId_{UINT32_MAX};
    std::atomic<uint32_t> blockFlagArrayId_{UINT32_MAX};
    bool albedoMipsInitialized_ = false;
    bool specMipsInitialized_ = false;
    bool normMipsInitialized_ = false;
    bool flagMipsInitialized_ = false;

    std::atomic<bool> finalized_{false};
    std::atomic<uint64_t> generation_{0};
    mutable std::mutex mutex_;
};
