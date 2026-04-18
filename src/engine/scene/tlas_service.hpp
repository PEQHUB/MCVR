#pragma once

// Ownership: EngineServices (1:1).
// Thread: Main thread only.
// Dependencies: DeviceService, BlasService, optional ResourceGC for deferred-delete.
//
// Builds the world TLAS from all chunk BLAS instances.
// Full rebuild each frame initially. UPDATE mode can be added later.
//
// Each TLAS instance carries a translation transform derived from the section
// origin (originX/Y/Z in BlasData) so the BLAS triangles can stay in the
// section-local space the mesher produces.
//
// The TLAS handle, backing buffer, and device address are double-buffered
// (slot = frameIndex & 1) to prevent the Frame N CPU build from writing the AS
// storage while Frame N-1's GPU vkCmdTraceRaysKHR is still reading it
// (WAR hazard, manifests as type=6 instruction-fetch DEVICE_FAULT).

#include "scene_types.hpp"
#include "platform/vulkan/vk2_buffer.hpp"
#include "platform/vulkan/vk2_result.hpp"
#include "frame/frame_context.hpp"

#include <cstdint>
#include <vulkan/vulkan.h>

namespace engine {

class BlasService;
class EntityBlasService;
class ResourceGC;

namespace vk2 { class DeviceService; }

class TlasService {
public:
    TlasService() = default;
    ~TlasService();

    TlasService(const TlasService&) = delete;
    TlasService& operator=(const TlasService&) = delete;

    // gc is optional. When provided, old TLAS destruction on rebuild is deferred
    // into the ResourceGC ring so the GPU finishes any in-flight work first.
    vk2::Result<void> init(vk2::DeviceService& device, ResourceGC* gc = nullptr);
    void shutdown();

    // Rebuild TLAS from all current BLAS instances.
    // Records build commands into cmd. Returns true if TLAS was built.
    // frameIndex is used to select the per-frame BDA SSBO slot (double-buffered).
    bool rebuild(VkCommandBuffer cmd, const BlasService& blasService,
                 uint32_t frameIndex,
                 const EntityBlasService* entityService = nullptr,
                 const CameraData* camera = nullptr);

    // Pass frameIndex to select the correct double-buffered slot.
    // The handle/address for slot N is safe to read after the per-frame fence wait.
    VkAccelerationStructureKHR handle(uint32_t frameIndex = 0) const { return tlas_[frameIndex & 1]; }
    VkDeviceAddress address(uint32_t frameIndex = 0) const { return tlasAddress_[frameIndex & 1]; }
    uint32_t instanceCount() const { return instanceCount_; }
    bool isValid() const { return (tlas_[0] != VK_NULL_HANDLE || tlas_[1] != VK_NULL_HANDLE) && instanceCount_ > 0; }
    bool isInitialized() const { return initialized_; }

    // Diagnostic flags — set each frame from EngineConfig::diagFlags before rebuild().
    void setDiagFlags(int f) { diagFlags_ = f; }

    // Per-instance vertex/index BDA SSBOs (built alongside TLAS).
    // Double-buffered (one slot per frame-in-flight) to prevent CPU write racing
    // with GPU reads from the previous frame. frameIndex must match rebuild().
    VkBuffer vertexBdaBuffer(uint32_t frameIndex = 0) const { return vertexBdaBuffers_[frameIndex & 1].handle(); }
    VkBuffer indexBdaBuffer(uint32_t frameIndex = 0) const  { return indexBdaBuffers_[frameIndex & 1].handle(); }
    VkDeviceSize bdaArraySize(uint32_t frameIndex = 0) const { return bdaInstanceCounts_[frameIndex & 1] * sizeof(uint64_t); }

    // Per-instance BLAS offset SSBO (double-buffered, same slot as BDA).
    VkBuffer blasOffsetsBuffer(uint32_t frameIndex = 0) const { return blasOffsetsBuffers_[frameIndex & 1].handle(); }
    VkDeviceSize blasOffsetsSize(uint32_t frameIndex = 0) const { return bdaInstanceCounts_[frameIndex & 1] * sizeof(uint32_t); }

    // Previous-frame BDA SSBOs and transforms for motion vector correctness.
    // Bound at set 1 bindings 4, 5, 6. Pass the CURRENT frameIndex — each slot
    // holds its own snapshot written after the matching fence wait. Double-buffered
    // so CPU writes for slot N never race with GPU reads of the in-flight Frame N-1.
    VkBuffer lastVertexBdaBuffer(uint32_t frameIndex = 0) const { return lastVertexBdaBuffers_[frameIndex & 1].handle(); }
    VkBuffer lastIndexBdaBuffer(uint32_t frameIndex = 0) const  { return lastIndexBdaBuffers_[frameIndex & 1].handle(); }
    VkBuffer lastTransformsBuffer(uint32_t frameIndex = 0) const { return lastTransformsBuffers_[frameIndex & 1].handle(); }
    VkDeviceSize lastBdaArraySize(uint32_t frameIndex = 0) const { return lastInstanceCounts_[frameIndex & 1] * sizeof(uint64_t); }
    VkDeviceSize lastTransformsSize(uint32_t frameIndex = 0) const { return lastInstanceCounts_[frameIndex & 1] * sizeof(float) * 12; }
private:
    vk2::DeviceService* device_ = nullptr;
    ResourceGC* gc_ = nullptr;

    // TLAS double-buffered (slot = frameIndex & 1):
    //   Frame N's CPU rebuild writes tlas_[N&1] while Frame N-1's GPU reads tlas_[(N-1)&1].
    //   The per-frame fence wait guarantees Frame N-2 is done before we overwrite slot N.
    VkAccelerationStructureKHR tlas_[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    vk2::Buffer tlasBuffer_[2];
    VkDeviceAddress tlasAddress_[2] = {0, 0};
    vk2::Buffer instanceBuffers_[2]; // VkAccelerationStructureInstanceKHR array (double-buffered, one per frame slot)
    vk2::Buffer scratchBuffer_;

    // Auxiliary SSBOs for the RT shader's vertex fetch path.
    // Double-buffered (index 0 and 1) to prevent CPU write racing with in-flight GPU reads.
    // Slot = frameIndex & 1 is written this frame; the other slot was written last frame.
    vk2::Buffer vertexBdaBuffers_[2];      // host-visible uint64[][2]
    vk2::Buffer indexBdaBuffers_[2];       // host-visible uint64[][2]
    vk2::Buffer blasOffsetsBuffers_[2];    // host-visible uint32[][2]
    uint32_t    bdaInstanceCounts_[2] = {0, 0};  // live instance counts per slot
    VkDeviceSize bdaCapacity_[2] = {0, 0}; // allocated capacity per slot

    // Previous-frame copies for motion vectors.
    // Double-buffered (index 0 and 1) — slot N is written during Frame N,
    // overwriting what was written in Frame N-2 (fence-waited, safe).
    vk2::Buffer lastVertexBdaBuffers_[2];
    vk2::Buffer lastIndexBdaBuffers_[2];
    vk2::Buffer lastTransformsBuffers_[2];
    VkDeviceSize lastBdaCapacities_[2] = {0, 0};
    uint32_t lastInstanceCounts_[2] = {0, 0};

    uint32_t instanceCount_ = 0;
    bool initialized_ = false;
    int diagFlags_ = 0;  // Set from EngineConfig::diagFlags each frame via setDiagFlags()

    // Destroy (or defer destruction of) the old TLAS handle + buffer on rebuild.
    void retireTlas(VkAccelerationStructureKHR handle, vk2::Buffer buffer);

    // Defer a plain buffer to the GC ring (used for instanceBuffers_[slot] and scratchBuffer_
    // to ensure in-flight GPU commands that reference them finish before the memory is freed).
    void retireBuffer(vk2::Buffer buffer);
};

} // namespace engine
