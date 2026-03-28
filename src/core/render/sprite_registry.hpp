#pragma once

#include "common/shared.hpp"
#include "core/vulkan/all_core_vulkan.hpp"

#include <cstdint>
#include <mutex>
#include <vector>

/// GPU-accessible sprite metadata registry.
/// Maps sequential spriteId → SpriteEntry for shader-side texture array lookups.
/// Uploaded as an SSBO at set 1, binding 13.
class SpriteRegistry {
  public:
    SpriteRegistry() = default;

    /// Register a sprite's metadata. Thread-safe.
    void registerSprite(uint16_t spriteId,
                        uint32_t baseLayer,
                        uint32_t frameCount,
                        uint32_t tickRate,
                        uint32_t flags,
                        int32_t specularLayer,
                        int32_t normalLayer,
                        int32_t overlaySprite,
                        int32_t maskLayer);

    /// Get a registered entry (CPU-side). Returns nullptr if not registered.
    const vk::Data::SpriteEntry* getEntry(uint16_t spriteId) const;

    /// Upload all entries to the GPU SSBO. Call after all sprites are registered.
    /// Requires valid VMA and Device from the renderer.
    void uploadSSBO(std::shared_ptr<vk::VMA> vma, std::shared_ptr<vk::Device> device);

    /// Get the GPU buffer for descriptor binding. Returns nullptr if not uploaded.
    std::shared_ptr<vk::DeviceLocalBuffer> getBuffer() const { return ssbo_; }

    /// Number of registered sprites.
    uint32_t count() const { return spriteCount_; }

    /// Reset all entries (resource reload).
    void reset();

  private:
    std::vector<vk::Data::SpriteEntry> entries_;
    uint32_t spriteCount_ = 0;
    std::shared_ptr<vk::DeviceLocalBuffer> ssbo_;
    mutable std::mutex mutex_;
};
