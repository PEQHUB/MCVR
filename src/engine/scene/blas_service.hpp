#pragma once

// Ownership: EngineServices (1:1).
// Thread: Main thread only.
// Dependencies: DeviceService, GpuUploadService, optional ResourceGC for deferred-delete.
//
// Builds per-chunk BLAS from uploaded GPU vertex/index buffers.
// Uses a shared scratch buffer for builds. Batched per-frame.

#include "scene_types.hpp"
#include "platform/vulkan/vk2_buffer.hpp"
#include "platform/vulkan/vk2_result.hpp"

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>
#include <vulkan/vulkan.h>

namespace engine {

struct GpuChunkData;
class ResourceGC;

namespace vk2 { class DeviceService; }

struct BlasData {
    VkAccelerationStructureKHR handle = VK_NULL_HANDLE;
    vk2::Buffer buffer;
    VkDeviceAddress address = 0;
    // Cached for the RT shader's vertex fetch path (TlasService bundles these into SSBOs)
    VkDeviceAddress vertexAddress = 0;
    VkDeviceAddress indexAddress = 0;
    // Section origin in world space — baked into TLAS instance transform (section-local
    // vertex data * translation = world-space ray intersection).
    float originX = 0.0f;
    float originY = 0.0f;
    float originZ = 0.0f;
    RevisionId revision;
};

// Result of a BLAS-build pass. `built` is the number of BLAS successfully
// constructed this frame. `skipped` is the list of ChunkIds that could not
// fit in the shared scratch buffer and need to be retried next frame (caller
// should mark them dirty in ChunkRegistry to avoid silent data loss).
struct BlasBuildResult {
    uint32_t built = 0;
    std::vector<ChunkId> skipped;
};

class BlasService {
public:
    BlasService() = default;
    ~BlasService();

    BlasService(const BlasService&) = delete;
    BlasService& operator=(const BlasService&) = delete;

    // gc is optional. When provided, BLAS destruction on chunk re-upload/removal is
    // deferred into the ResourceGC ring so the GPU finishes any in-flight work first.
    //
    // scratchSize defaults to 128 MB — sized for the same worst-case chunk-burst
    // scenario as GpuUploadService::stagingSize. Undersizing causes silent drops
    // where uploaded chunks never get a BLAS and remain invisible to RT.
    vk2::Result<void> init(vk2::DeviceService& device, ResourceGC* gc = nullptr,
                           VkDeviceSize scratchSize = 128 * 1024 * 1024,
                           uint32_t framesInFlight = 2);
    void shutdown();

    // Build BLAS for dirty chunks. Records build commands into cmd.
    // frameIndex selects the double-buffered scratch slot (frameIndex & 1);
    // the fence wait before this call guarantees the same slot's previous
    // GPU BLAS build has completed, making scratch reuse safe.
    // Returns a BlasBuildResult with the built count and the list of skipped
    // chunk ids (caller should re-mark these dirty in ChunkRegistry).
    BlasBuildResult buildDirtyChunks(VkCommandBuffer cmd,
                                      const std::vector<ChunkId>& dirtyIds,
                                      const std::unordered_map<ChunkId, GpuChunkData>& gpuChunks,
                                      uint32_t frameIndex = 0);

    // Remove a chunk's BLAS.
    void removeChunk(ChunkId id);

    // Remove all chunk BLASes (world unload). Retires all via GC.
    void clearAll();

    // Access BLAS data.
    const BlasData* getBlas(ChunkId id) const;

    template<typename Fn>
    void forEachBlas(Fn&& fn) const {
        for (const auto& [id, data] : blas_) fn(id, data);
    }

    uint32_t blasCount() const { return static_cast<uint32_t>(blas_.size()); }
    bool isInitialized() const { return initialized_; }

    // Enable OMM fallback mode. When true, VK_GEOMETRY_OPAQUE_BIT_KHR is omitted
    // from all geometry descriptions so any-hit shaders fire for alpha testing.
    // This is the correct V2 fallback: alpha texture data is not available CPU-side
    // at BLAS build time (it lives GPU-side after CmdTextureFinalize), so the full
    // VkMicromapEXT path cannot be taken. Transparent geometry remains correct at a
    // small performance cost versus opaque-flagged geometry without OMM.
    void setOmmEnabled(bool enabled) { ommEnabled_ = enabled; }

    // Diagnostic flags — set each frame from EngineConfig::diagFlags before build.
    void setDiagFlags(int f) { diagFlags_ = f; }

private:
    vk2::DeviceService* device_ = nullptr;
    ResourceGC* gc_ = nullptr;

    // Double-buffered scratch buffers — one per frame-in-flight slot.
    // Frame N uses scratch[N & 1]; the fence wait guarantees Frame N-2
    // (same slot, previous use) is done before we reuse the scratch.
    // This prevents Frame N-1's GPU BLAS build from aliasing Frame N's scratch.
    vk2::Buffer scratchBuffers_[2];
    VkDeviceAddress scratchAddresses_[2] = {0, 0};
    VkDeviceSize scratchSize_ = 0;

    std::unordered_map<ChunkId, BlasData> blas_;
    bool initialized_ = false;
    bool ommEnabled_ = false;  // Set from config; strips OPAQUE flag to enable any-hit for alpha test
    int diagFlags_ = 0;         // Set from EngineConfig::diagFlags each frame via setDiagFlags()

    // Destroy (or defer destruction of) a BLAS, leaving `data` in a zeroed state.
    void retireBlas(BlasData data);
};

} // namespace engine
