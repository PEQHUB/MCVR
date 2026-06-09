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

class BlasService {
public:
    BlasService() = default;
    ~BlasService();

    BlasService(const BlasService&) = delete;
    BlasService& operator=(const BlasService&) = delete;

    // gc is optional. When provided, BLAS destruction on chunk re-upload/removal is
    // deferred into the ResourceGC ring so the GPU finishes any in-flight work first.
    vk2::Result<void> init(vk2::DeviceService& device, ResourceGC* gc = nullptr,
                           VkDeviceSize scratchSize = 32 * 1024 * 1024);
    void shutdown();

    // Build BLAS for dirty chunks. Records build commands into cmd.
    // Returns number of BLAS built this frame.
    uint32_t buildDirtyChunks(VkCommandBuffer cmd,
                              const std::vector<ChunkId>& dirtyIds,
                              const std::unordered_map<ChunkId, GpuChunkData>& gpuChunks);

    // Remove a chunk's BLAS.
    void removeChunk(ChunkId id);

    // Access BLAS data.
    const BlasData* getBlas(ChunkId id) const;

    template<typename Fn>
    void forEachBlas(Fn&& fn) const {
        for (const auto& [id, data] : blas_) fn(id, data);
    }

    uint32_t blasCount() const { return static_cast<uint32_t>(blas_.size()); }
    bool isInitialized() const { return initialized_; }

private:
    vk2::DeviceService* device_ = nullptr;
    ResourceGC* gc_ = nullptr;

    // Shared scratch buffer for all AS builds
    vk2::Buffer scratchBuffer_;
    VkDeviceAddress scratchAddress_ = 0;
    VkDeviceSize scratchSize_ = 0;

    std::unordered_map<ChunkId, BlasData> blas_;
    bool initialized_ = false;

    // Destroy (or defer destruction of) a BLAS, leaving `data` in a zeroed state.
    void retireBlas(BlasData data);
};

} // namespace engine
