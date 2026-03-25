#pragma once

#include "common/shared.hpp"
#include "common/singleton.hpp"
#include "core/all_extern.hpp"
#include "core/vulkan/all_core_vulkan.hpp"

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

    std::shared_ptr<vk::DeviceLocalBuffer> blasOffsetsBuffer;
    std::shared_ptr<vk::DeviceLocalBuffer> vertexBufferAddr;
    std::shared_ptr<vk::DeviceLocalBuffer> indexBufferAddr;
    std::shared_ptr<vk::DeviceLocalBuffer> lastVertexBufferAddr;
    std::shared_ptr<vk::DeviceLocalBuffer> lastIndexBufferAddr;
    std::shared_ptr<vk::DeviceLocalBuffer> lastObjToWorldMat;
    std::shared_ptr<vk::DeviceLocalBuffer> areaLightBuffer;
    int areaLightCount = 0;

    WorldPrepareContext(std::shared_ptr<FrameworkContext> frameworkContext, std::shared_ptr<WorldPrepare> worldprepare);

    void uploadBuffer(std::vector<uint32_t> &blasOffsets,
                      std::vector<uint64_t> &vertexBufferAddrs,
                      std::vector<uint64_t> &indexBufferAddrs,
                      std::vector<uint64_t> &lastVertexBufferAddrs,
                      std::vector<uint64_t> &lastIndexBufferAddrs,
                      std::vector<glm::mat4> &lastObjToWorldMats);
    void render();
};