#pragma once

#include "common/shared.hpp"
#include "common/singleton.hpp"
#include "core/all_extern.hpp"
#include "core/vulkan/all_core_vulkan.hpp"
#include "core/render/world.hpp"

#include <unordered_map>

class Framework;
class FrameworkContext;
class RayTracingModule;
struct RayTracingModuleContext;

struct WorldPrepareContext;

class WorldPrepare : public SharedObject<WorldPrepare> {
    friend RayTracingModule;
    friend RayTracingModuleContext;
    friend WorldPrepareContext;

  public:
    WorldPrepare();

    void init(std::shared_ptr<Framework> framework, std::shared_ptr<RayTracingModule> rayTracingModule);

    void build();

  private:
    std::weak_ptr<Framework> framework_;
    std::weak_ptr<RayTracingModule> rayTracingModule_;

    std::vector<std::shared_ptr<WorldPrepareContext>> contexts_;
};

struct WorldPrepareContext : public SharedObject<WorldPrepareContext> {
    std::weak_ptr<FrameworkContext> frameworkContext;
    std::weak_ptr<RayTracingModuleContext> rayTracingModuleContext;
    std::weak_ptr<WorldPrepare> worldPrepare;

    std::shared_ptr<vk::TLAS> tlas;
    std::shared_ptr<vk::TLASBuilder> tlasBuilder;

    // TLAS update tracking: reuse TLAS when only transforms changed (same BLAS set)
    uint32_t prevTlasInstanceCount_ = 0;
    std::shared_ptr<vk::DeviceLocalBuffer> tlasScratchBuffer_;  // persisted for UPDATE mode reuse
    VkDeviceSize tlasScratchSize_ = 0;  // max(buildScratchSize, updateScratchSize)

    // BLAS lifetime snapshot: keeps shared_ptrs alive while GPU references the TLAS.
    // Released when this swapchain context is reused (GPU guaranteed done by then).
    struct TlasBlasSnapshot {
        std::vector<std::shared_ptr<vk::BLAS>> blases;
        std::vector<uint64_t> generations;  // Chunk1::blasGeneration per instance
        uint32_t instanceCount = 0;
    };
    TlasBlasSnapshot prevBlasSnapshot_;

    // Mega-chunk system: groups distant chunks into single BLASes
    struct MegaChunk {
        int64_t key;                    // spatial hash of mega-chunk grid position
        glm::dvec3 origin;             // world origin (min corner)
        std::shared_ptr<vk::BLAS> blas;
        uint64_t contentHash = 0;      // hash of constituent chunk generations (dirty detection)

        // Flattened geometry metadata for SBT
        std::vector<World::GeometryTypes> geometryTypes;
        std::vector<std::shared_ptr<vk::DeviceLocalBuffer>> vertexBuffers;
        std::vector<std::shared_ptr<vk::DeviceLocalBuffer>> indexBuffers;

        // Per-sub-chunk transform buffers (3x4 VkTransformMatrixKHR each)
        // HostVisibleBuffer avoids staging copy issues and guarantees alignment
        std::vector<std::shared_ptr<vk::HostVisibleBuffer>> transformBuffers;
    };
    std::unordered_map<int64_t, std::shared_ptr<MegaChunk>> megaChunkCache_;

    // Dedicated secondary-queue resources for mega-BLAS builds
    std::shared_ptr<vk::CommandBuffer> megaCmdBuffer_;
    std::shared_ptr<vk::Fence> megaFence_;

    // Cached per-chunk instance metadata — only regenerated when blasGeneration changes.
    // Eliminates shared_ptr dereferences and vector copies on stable frames.
    struct CachedChunkData {
        uint64_t blasGeneration = UINT64_MAX; // invalid → forces rebuild
        std::shared_ptr<vk::BLAS> blas;
        std::shared_ptr<vk::BLAS> displacedBlas;
        std::shared_ptr<vk::DeviceLocalBuffer> displacedFaceDataBuffer;
        int x, y, z;
        uint32_t geometryCount = 0;
        std::vector<World::GeometryTypes> geoTypes; // SHADOW prefix + chunk types
        std::vector<uint64_t> vertBufAddrs;
        std::vector<uint64_t> idxBufAddrs;
        bool hasDisplaced = false;
    };
    std::vector<CachedChunkData> cachedChunks_;

    std::shared_ptr<vk::DeviceLocalBuffer> blasOffsetsBuffer;
    std::shared_ptr<vk::DeviceLocalBuffer> vertexBufferAddr;
    std::shared_ptr<vk::DeviceLocalBuffer> indexBufferAddr;
    std::shared_ptr<vk::DeviceLocalBuffer> lastVertexBufferAddr;
    std::shared_ptr<vk::DeviceLocalBuffer> lastIndexBufferAddr;
    std::shared_ptr<vk::DeviceLocalBuffer> lastObjToWorldMat;
    std::shared_ptr<vk::DeviceLocalBuffer> areaLightBuffer;
    int areaLightCount = 0;

    WorldPrepareContext(std::shared_ptr<FrameworkContext> frameworkContext, std::shared_ptr<WorldPrepare> worldprepare);

    // Called after vkDeviceWaitIdle during render distance change.
    // Releases all acceleration structure state so the next frame does a clean rebuild.
    void resetChunkState() {
        tlas = nullptr;
        tlasBuilder = nullptr;
        prevBlasSnapshot_ = {};
        prevTlasInstanceCount_ = 0;
        tlasScratchBuffer_ = nullptr;
        tlasScratchSize_ = 0;
        megaChunkCache_.clear();
        cachedChunks_.clear();
    }

    void uploadBuffer(std::vector<uint32_t> &blasOffsets,
                      std::vector<uint64_t> &vertexBufferAddrs,
                      std::vector<uint64_t> &indexBufferAddrs,
                      std::vector<uint64_t> &lastVertexBufferAddrs,
                      std::vector<uint64_t> &lastIndexBufferAddrs,
                      std::vector<glm::mat4> &lastObjToWorldMats);
    void render();
};