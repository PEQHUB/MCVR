#pragma once

// Ownership: EngineServices (1:1).
// Thread: Main thread only.
// Dependencies: DeviceService, optional ResourceGC for deferred-delete.
//
// Builds per-entity BLAS from batched vertex/index data submitted each frame.
// All entity BLASes are rebuilt every frame (PREFER_FAST_BUILD).
// Uses a shared scratch buffer for builds.

#include "scene_types.hpp"
#include "platform/vulkan/vk2_buffer.hpp"
#include "platform/vulkan/vk2_result.hpp"

#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <vulkan/vulkan.h>

namespace engine {

class ResourceGC;
namespace vk2 { class DeviceService; }

// Previous-frame data for entity motion vector computation.
struct PrevEntityData {
    float posX = 0.0f, posY = 0.0f, posZ = 0.0f;
    uint64_t vertexAddr = 0;
    uint64_t indexAddr = 0;
};

// Per-entity motion delta written to the SSBO (std430, 16 bytes).
struct EntityMotion {
    float dx = 0.0f, dy = 0.0f, dz = 0.0f;
    float pad = 0.0f;
};

struct EntityBlasData {
    VkAccelerationStructureKHR handle = VK_NULL_HANDLE;
    vk2::Buffer buffer;
    VkDeviceAddress address = 0;
    VkDeviceAddress vertexAddress = 0;
    VkDeviceAddress indexAddress = 0;
    // Entity world position (baked into TLAS instance transform)
    float posX = 0.0f;
    float posY = 0.0f;
    float posZ = 0.0f;
    uint8_t rtFlag = 0;
    uint8_t coordSystem = 0;  // 0=WORLD, 1=CAMERA, 2=CAMERA_SHIFT
    uint32_t hashCode = 0;
};

class EntityBlasService {
public:
    EntityBlasService() = default;
    ~EntityBlasService();

    EntityBlasService(const EntityBlasService&) = delete;
    EntityBlasService& operator=(const EntityBlasService&) = delete;

    vk2::Result<void> init(vk2::DeviceService& device, ResourceGC* gc = nullptr,
                           VkDeviceSize scratchSize = 32 * 1024 * 1024);
    void shutdown();

    // Submit a batch of entities for this frame. Uploads vertex/index data to GPU
    // and records BLAS build commands into cmd.
    // frameIndex (0..framesInFlight-1) selects the double-buffer slot for the
    // shared vertex/index buffers, preventing UAF while the previous slot is in-flight.
    // Returns the number of entity BLASes built.
    uint32_t submitBatch(VkCommandBuffer cmd,
                         std::vector<uint8_t> vertexData,
                         std::vector<uint32_t> indexData,
                         const std::vector<struct EntityBatchEntry>& entries,
                         uint32_t frameIndex = 0);

    // Clear all entity BLASes (called at start of each frame before submitBatch).
    void clear();

    const std::vector<EntityBlasData>& entities() const { return entities_; }
    uint32_t entityCount() const { return static_cast<uint32_t>(entities_.size()); }
    bool isInitialized() const { return initialized_; }

    // Motion-vector SSBO (one EntityMotion per entity, indexed by TLAS custom index).
    VkBuffer entityMotionBuffer() const;
    VkDeviceSize entityMotionBufferSize() const;

private:
    vk2::DeviceService* device_ = nullptr;
    ResourceGC* gc_ = nullptr;

    // Double-buffered scratch buffers — one per frame-in-flight slot.
    // Frame N uses scratch[N & 1]; the fence wait guarantees Frame N-2
    // (same slot, previous use) is done before we reuse the scratch.
    vk2::Buffer scratchBuffers_[2];
    VkDeviceAddress scratchAddresses_[2] = {0, 0};
    VkDeviceSize scratchSize_ = 0;

    // GPU buffers for combined vertex/index data — double-buffered by frameIndex & 1
    // to prevent UAF while the previous frame slot is still in-flight on the GPU.
    vk2::Buffer vertexBuffers_[2];
    vk2::Buffer indexBuffers_[2];

    // Per-entity BLAS data
    std::vector<EntityBlasData> entities_;
    bool initialized_ = false;

    // --- Motion-vector tracking ---
    std::unordered_map<uint32_t, PrevEntityData> prevEntities_;
    vk2::Buffer entityMotionBuffer_;

    void destroyAllBlas();
    void buildMotionBuffer();
};

// Intermediate entry type used by submitBatch (matches CmdEntityBatchSubmit::EntityEntry layout)
struct EntityBatchEntry {
    uint32_t hashCode;
    float posX, posY, posZ;
    uint8_t rtFlag;
    uint8_t coordSystem;
    uint32_t vertexOffset;  // Byte offset into vertex buffer
    uint32_t indexOffset;   // Element offset into index buffer
    uint32_t triangleCount;
};

} // namespace engine
