#pragma once

// Ownership: EngineServices (1:1 lifetime).
// Thread: receive*() called from main thread (bridge handler). finalize() on main thread.
// Dependencies: DeviceService (VMA + device), ResourceGC for staging buffer retirement.
//
// Receives sprite data staged by bridge commands (CmdSpriteTableUpload,
// CmdSpritePixelsUpload, CmdSpriteAuxPixelsUpload), then on finalize() creates:
//   - 3 VkImages as sampler2DArray (albedo sRGB, specular UNORM, normal UNORM)
//   - A SpriteRegistry SSBO (SpriteEntry[spriteCount])
//   - VkSamplers for texture arrays
// Public accessors expose handles for descriptor binding in the RT adapter.

#include "platform/vulkan/vk2_buffer.hpp"
#include "platform/vulkan/vk2_image.hpp"
#include "platform/vulkan/vk2_result.hpp"
#include <cstdint>
#include <glm/glm.hpp>
#include <unordered_map>
#include <vector>
#include <vulkan/vulkan.h>

namespace engine {
namespace vk2 { class DeviceService; }
class ResourceGC;

class TextureService {
public:
    TextureService() = default;
    ~TextureService();

    vk2::Result<void> init(vk2::DeviceService& device, ResourceGC* gc = nullptr);
    void shutdown();
    bool isInitialized() const { return initialized_; }
    bool isFinalized() const { return finalized_; }

    // Called from bridge handlers in order:
    void receiveSpriteTable(const uint8_t* metadata, uint32_t count,
                           uint32_t atlasW, uint32_t atlasH, uint32_t spriteSize);
    void receiveSpritePixels(const uint8_t* data, size_t size);
    void receiveAuxPixels(const uint8_t* specular, size_t specSize,
                         const uint8_t* normal, size_t normSize);
    void receiveAnimationFrames(const uint8_t* data, size_t size);

    // Create GPU resources from staged data. Must be called with a command buffer
    // for staging copies. Returns true if commands were recorded.
    bool finalize(VkCommandBuffer cmd);

    // Descriptor accessors (valid after finalize)
    VkImageView albedoArrayView() const { return albedoArray_.defaultView(); }
    VkImageView specularArrayView() const { return specularArray_.defaultView(); }
    VkImageView normalArrayView() const { return normalArray_.defaultView(); }
    VkSampler arrayNearestSampler() const { return nearestSampler_; }
    VkSampler arrayLinearSampler() const { return linearSampler_; }
    VkBuffer spriteRegistrySSBO() const { return spriteRegistrySSBO_.handle(); }
    VkDeviceSize spriteRegistrySize() const;
    uint32_t spriteCount() const { return spriteCount_; }

    // Sprite atlas bounds in [0,1] atlas-relative space: (minU, maxU, minV, maxV).
    // Populated during finalize() and retained after spriteTable_ is freed —
    // the BlockModelTable UV normalization in engine_app::processScene needs
    // these to convert atlas-space model UVs into sprite-local [0,1] UVs that
    // the V2 RT shaders expect.
    glm::vec4 spriteBounds(uint16_t spriteId) const {
        if (spriteId >= atlasBounds_.size()) return {0.0f, 1.0f, 0.0f, 1.0f};
        return atlasBounds_[spriteId];
    }

private:
    vk2::DeviceService* device_ = nullptr;
    ResourceGC* gc_ = nullptr;
    bool initialized_ = false;
    bool finalized_ = false;
    bool pendingFinalize_ = false;

    // Staged CPU data (cleared after finalize).
    // Layout matches Java SpriteAtlasTextureMixins packing (16 bytes per sprite).
    struct SpriteMetadata {
        uint16_t x, y, w, h;           // atlas pixel coords
        uint16_t frameCount;           // animation frames (1 = static)
        uint16_t tickRate;             // game ticks per animation advance
        int16_t  overlayOf;            // spriteId this overlays (-1 = none)
        uint16_t flags;                // bit 0: hasSpecular, bit 1: hasNormal
    };
    static_assert(sizeof(SpriteMetadata) == 16, "SpriteMetadata must be 16 bytes");

    std::vector<SpriteMetadata> spriteTable_;
    std::vector<uint8_t> albedoPixels_;     // frame-0 pixels for all sprites
    std::vector<uint8_t> specularPixels_;
    std::vector<uint8_t> normalPixels_;
    uint32_t spriteCount_ = 0;
    uint32_t atlasWidth_ = 0;
    uint32_t atlasHeight_ = 0;
    uint32_t spriteSize_ = 16;

    // Animation frame storage (populated by receiveAnimationFrames).
    // Key = spriteId. Value = per-frame RGBA8 pixel buffers indexed by frameIndex.
    struct AnimFrames {
        std::vector<std::vector<uint8_t>> frames;  // [frameIndex] -> RGBA8 pixels
    };
    std::unordered_map<uint16_t, AnimFrames> animData_;
    uint32_t totalAlbedoLayers_ = 0;  // computed in finalize: sum of max(1,frameCount)

    // GPU resources
    vk2::Image albedoArray_;      // sampler2DArray, sRGB
    vk2::Image specularArray_;    // sampler2DArray, UNORM
    vk2::Image normalArray_;      // sampler2DArray, UNORM
    vk2::Buffer spriteRegistrySSBO_;
    vk2::Buffer stagingBuffer_;   // temporary, freed after finalize

    // Per-sprite atlas bounds (minU, maxU, minV, maxV) in [0,1] space.
    // Computed in finalize() before spriteTable_ is cleared; used by
    // engine_app::processScene to normalize BlockModelTable quad UVs.
    std::vector<glm::vec4> atlasBounds_;
    VkSampler nearestSampler_ = VK_NULL_HANDLE;
    VkSampler linearSampler_ = VK_NULL_HANDLE;

    // SpriteRegistry entry layout — matches V1 shared.hpp SpriteEntry exactly.
    // The shader (sprite_fetch.glsl) reads baseLayer, frameCount, tickRate for animation.
    struct SpriteEntry {
        uint32_t baseLayer;        // first albedo layer in the texture array
        uint32_t frameCount;       // 1 = static, N = animated (N consecutive layers)
        uint32_t tickRate;         // game ticks per animation frame (1 = every tick)
        uint32_t flags;            // bit 0: hasSpecular, bit 1: hasNormal, bit 2: hasHeight
        int32_t  specularLayer;    // layer in specular array (-1 = none)
        int32_t  normalLayer;      // layer in normal array (-1 = none)
        int32_t  overlaySprite;    // spriteId of overlay texture (-1 = none)
        int32_t  maskLayer;        // material class mask layer (-1 = none)
    };
    static_assert(sizeof(SpriteEntry) == 32, "SpriteEntry must be 32 bytes");

    void destroySamplers();
};
} // namespace engine
