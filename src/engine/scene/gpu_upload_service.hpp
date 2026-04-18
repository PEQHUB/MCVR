#pragma once

// Ownership: EngineServices (1:1).
// Thread: Main thread only.
// Dependencies: DeviceService (VkDevice, VMA, queues), optional ResourceGC for deferred-delete.
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
class ResourceGC;

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

// Result of an upload pass. `uploaded` is the number of chunks successfully
// committed to GPU memory this frame. `skipped` is the list of ChunkIds that
// could not fit in the per-frame staging slot and need to be retried next frame
// (caller should mark them dirty in ChunkRegistry to avoid silent data loss).
struct UploadResult {
    uint32_t uploaded = 0;
    std::vector<ChunkId> skipped;
};

class GpuUploadService {
public:
    GpuUploadService() = default;
    ~GpuUploadService() = default;

    GpuUploadService(const GpuUploadService&) = delete;
    GpuUploadService& operator=(const GpuUploadService&) = delete;

    // gc is optional. When provided, vertex/index buffer destruction on chunk re-upload
    // or removal is deferred into the ResourceGC ring so the GPU finishes in-flight work first.
    //
    // stagingSize defaults to 256 MB — large enough to absorb the initial chunk burst
    // when a 32-render-distance world first loads (~30k chunks, peak ~50KB each).
    // Undersizing this causes silent chunk drops that only surface as missing geometry.
    vk2::Result<void> init(vk2::DeviceService& device, uint32_t framesInFlight,
                           ResourceGC* gc = nullptr,
                           VkDeviceSize stagingSize = 256 * 1024 * 1024);
    void shutdown();

    // Upload dirty chunks from an extracted scene.
    // Records copy commands into cmd. Caller must submit the command buffer.
    // frameIndex selects the staging region to avoid overwriting in-flight data.
    // Returns an UploadResult containing the upload count and the list of skipped
    // chunk ids (caller should re-mark these dirty in ChunkRegistry).
    UploadResult uploadDirtyChunks(VkCommandBuffer cmd,
                                    const std::vector<ChunkGeometry>& dirtyChunks,
                                    uint32_t frameIndex);

    // Remove GPU data for chunks that no longer exist.
    void removeChunk(ChunkId id);

    // Remove all GPU chunk data (world unload). Retires all buffers via GC.
    void clearAll();

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

    // Diagnostic flags — set each frame from EngineConfig::diagFlags before upload.
    void setDiagFlags(int f) { diagFlags_ = f; }

private:
    vk2::DeviceService* device_ = nullptr;
    ResourceGC* gc_ = nullptr;

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
    int diagFlags_ = 0;  // Set from EngineConfig::diagFlags each frame via setDiagFlags()

    // Allocate from the current frame's staging region. Returns offset, or UINT64_MAX if full.
    VkDeviceSize allocStaging(VkDeviceSize size, VkDeviceSize alignment = 16);

    // Reset staging to the beginning of the given frame slot's region.
    void resetStaging(uint32_t frameIndex);

    // Retire (or defer) the buffers owned by `data`. After the call, data's vk2::Buffer
    // members are empty (either destroyed synchronously or captured into the GC lambda).
    void retireChunk(GpuChunkData data);
};

} // namespace engine
