#pragma once

// Ownership: EngineServices (1:1).
// Thread: Main thread only.
// Dependencies: DeviceService (VkDevice, VMA, queues).
//
// Uploads ChunkGeometry from ExtractedScene into device-local GPU buffers.
// Uses a staging buffer ring (host-visible, persistently mapped) for async transfer.
// Per-chunk GPU data is tracked in a map keyed by ChunkId.

#include "scene_types.hpp"
#include "platform/vulkan/vk2_buffer.hpp"
#include "platform/vulkan/vk2_result.hpp"

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace engine {

struct ChunkGeometry;

namespace vk2 { class DeviceService; }

// GPU-side data for one chunk.
struct GpuChunkData {
    vk2::Buffer vertexBuffer;
    vk2::Buffer indexBuffer;
    uint32_t triangleCount = 0;
    uint32_t indexCount = 0;
    VkDeviceAddress vertexAddress = 0;
    VkDeviceAddress indexAddress = 0;
    RevisionId revision;

    // World-space origin
    float originX = 0.0f;
    float originY = 0.0f;
    float originZ = 0.0f;
};

class GpuUploadService {
public:
    GpuUploadService() = default;
    ~GpuUploadService() = default;

    GpuUploadService(const GpuUploadService&) = delete;
    GpuUploadService& operator=(const GpuUploadService&) = delete;

    vk2::Result<void> init(vk2::DeviceService& device, uint32_t framesInFlight,
                           VkDeviceSize stagingSize = 64 * 1024 * 1024);
    void shutdown();

    // Upload dirty chunks from an extracted scene.
    // Records copy commands into cmd. Caller must submit the command buffer.
    // frameIndex selects the staging region to avoid overwriting in-flight data.
    // Returns the number of chunks uploaded this frame.
    uint32_t uploadDirtyChunks(VkCommandBuffer cmd,
                               const std::vector<ChunkGeometry>& dirtyChunks,
                               uint32_t frameIndex);

    // Remove GPU data for chunks that no longer exist.
    void removeChunk(ChunkId id);

    // Access GPU data for a chunk.
    const GpuChunkData* getChunk(ChunkId id) const;

    // Iterate all uploaded chunks.
    template<typename Fn>
    void forEachChunk(Fn&& fn) const {
        for (const auto& [id, data] : chunks_) fn(id, data);
    }

    uint32_t chunkCount() const { return static_cast<uint32_t>(chunks_.size()); }
    const std::unordered_map<ChunkId, GpuChunkData>& allChunks() const { return chunks_; }
    bool isInitialized() const { return initialized_; }

private:
    vk2::DeviceService* device_ = nullptr;

    // Staging buffer (host-visible, persistently mapped, per-frame-slot regions)
    vk2::Buffer stagingBuffer_;
    VkDeviceSize stagingSize_ = 0;
    VkDeviceSize stagingOffset_ = 0;
    VkDeviceSize slotSize_ = 0;      // stagingSize_ / framesInFlight_
    VkDeviceSize slotBase_ = 0;      // current slot's base offset
    uint32_t framesInFlight_ = 1;

    // Per-chunk GPU data
    std::unordered_map<ChunkId, GpuChunkData> chunks_;

    bool initialized_ = false;

    // Allocate from the current frame's staging region. Returns offset, or UINT64_MAX if full.
    VkDeviceSize allocStaging(VkDeviceSize size, VkDeviceSize alignment = 16);

    // Reset staging to the beginning of the given frame slot's region.
    void resetStaging(uint32_t frameIndex);
};

} // namespace engine
