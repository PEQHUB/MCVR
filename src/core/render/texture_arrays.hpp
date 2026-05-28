#pragma once

#include "core/vulkan/all_core_vulkan.hpp"

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

class Framework;

/// Manages sampler2DArray images for the sprite-based texture system.
/// Each array is a single VkImage with N layers at uniform WxW resolution.
/// Supports per-layer pixel upload, mipmap generation, and descriptor binding.
class TextureArrayManager {
  public:
	struct ArrayInfo {
		std::shared_ptr<vk::DeviceLocalImage> image;
		std::shared_ptr<vk::Sampler> sampler;
		uint32_t spriteSize; // WxW per layer (e.g., 16, 128, 1024)
		uint32_t layerCount;
		uint32_t mipLevels;
		VkFormat format;
		std::vector<VkImageLayout> subresourceLayouts; // [layer * mipLevels + mip]
	};
    TextureArrayManager() = default;

    /// Create a new texture array. Returns an array ID for subsequent operations.
    /// The image is created immediately with all layers allocated but uninitialized.
    uint32_t createArray(std::shared_ptr<vk::VMA> vma,
                         std::shared_ptr<vk::Device> device,
                         uint32_t spriteSize,
                         uint32_t layerCount,
                         VkFormat format,
                         bool generateMips);

    /// Stage pixel data for a specific layer and mip level.
    /// Pixels are buffered CPU-side until flushUploads() is called.
	void stageLayerPixels(uint32_t arrayId,
		uint32_t layer,
		uint32_t mipLevel,
		const uint8_t* pixels,
		size_t pixelSize);

	struct DirtyLayers {
		std::vector<uint32_t> albedo, specular, normal;
	};

	/// Flush all staged uploads to the GPU and return dirty layer info.
	/// Must be called within a valid command buffer recording.
	/// Uses snapshot-and-process: mutex held only for ~1us to swap staged uploads,
	/// then processing (VMA alloc, copy, barriers) runs without the mutex.
	DirtyLayers flushUploads(std::shared_ptr<vk::VMA> vma,
		std::shared_ptr<vk::Device> device,
		std::shared_ptr<vk::CommandBuffer> cmdBuffer,
		uint32_t albedoId = UINT32_MAX,
		uint32_t specId = UINT32_MAX,
		uint32_t normId = UINT32_MAX,
		size_t maxBytes = 0);

    /// Only for uncompressed formats (RGBA8). BC7 mips must be pre-computed on CPU.
    void generateMipmaps(uint32_t arrayId,
                         std::shared_ptr<vk::CommandBuffer> cmdBuffer);

    /// Get the array info for descriptor binding.
    const ArrayInfo* getArray(uint32_t arrayId) const;

    /// Check if there are pending uploads that need flushing.
    bool hasPendingUploads() const;

    /// Reset all arrays (resource reload).
    void reset();

    /// Compute mip level count for a given sprite size.
    static uint32_t computeMipLevels(uint32_t spriteSize);

    /// Generate mipmaps for a single layer of an array.
	void generateMipmapsForLayer(uint32_t arrayId, uint32_t layer,
		std::shared_ptr<vk::CommandBuffer> cmdBuffer);

	/// Generate mipmaps for multiple layers of an array.
	/// More efficient than calling generateMipmapsForLayer() in a loop because
	/// it looks up the array once instead of N times.
	void generateMipmapsForLayers(uint32_t arrayId,
		const std::vector<uint32_t>& layers,
		std::shared_ptr<vk::CommandBuffer> cmdBuffer);

    /// Collect which layers of an array have pending uploads (without flushing).
    std::vector<uint32_t> collectDirtyLayers(uint32_t arrayId) const;

    /// Take ownership of this frame's staging buffers (for GC collection).
    std::vector<std::shared_ptr<vk::HostVisibleBuffer>> takeStagingBuffers();

  private:

    struct StagedUpload {
        uint32_t arrayId;
        uint32_t layer;
        uint32_t mipLevel;
        std::vector<uint8_t> pixels;
    };

    std::map<uint32_t, ArrayInfo> arrays_;
    std::vector<StagedUpload> stagedUploads_;
    // Per-frame staging buffers: kept alive until GC reclaims them (imageCount*3 frames).
    // This matches the existing GarbageCollector pattern used for all GPU resources.
    std::vector<std::shared_ptr<vk::HostVisibleBuffer>> currentFrameStagingBuffers_;
    uint32_t nextArrayId_ = 0;
    mutable std::mutex mutex_;
};
