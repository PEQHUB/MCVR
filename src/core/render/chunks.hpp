#pragma once

#include "common/shared.hpp"
#include "common/singleton.hpp"
#include "core/all_extern.hpp"
#include "core/vulkan/all_core_vulkan.hpp"

#include "core/render/world.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <list>
#include <map>
#include <mutex>
#include <queue>
#include <set>
#include <unordered_map>
#include <vector>

class Framework;

struct ChunkLightEntry {
    float worldX, worldY, worldZ;
    int lightTypeId;
};

struct ChunkBuildTask {
    int x, y, z;
    int64_t id;
    int geometryCount;
    int *geometryTypes;
    int *geometryTextures;
    int *vertexFormats;
    int *vertexCounts;
    vk::VertexFormat::PBRTriangle **vertices;
    bool isImportant;
    uint64_t textureGeneration = 0;
    std::vector<ChunkLightEntry> lightSources;
};

struct ChunkBuildTaskV2 {
    int x, y, z;
    int64_t id;
    uint32_t blockStates[4096];
    uint16_t biomes[64];
    uint32_t neighborStates[6][256];
    uint32_t blockAtlasTextureId;
    bool isImportant;
    uint32_t biomeGrassColor;
    uint32_t biomeFoliageColor;
    uint32_t biomeWaterColor;
    uint64_t textureGeneration = 0;
    uint32_t haloStates[4096] = {};
    bool hasHaloStates = false;
};

struct ChunkBuildData : public SharedObject<ChunkBuildData> {
    int x, y, z;
    int64_t id;
    int64_t version;
    uint32_t allVertexCount;
    uint32_t allIndexCount;
    uint32_t geometryCount;
    std::vector<World::GeometryTypes> geometryTypes;
    std::vector<std::vector<vk::VertexFormat::PBRTriangle>> vertices;
    std::vector<std::vector<uint32_t>> indices;
    std::vector<std::shared_ptr<vk::DeviceLocalBuffer>> vertexBuffers;
    std::vector<std::shared_ptr<vk::DeviceLocalBuffer>> indexBuffers;
    std::vector<std::shared_ptr<vk::DeviceLocalBuffer>> ommIndexBuffers;
    struct OMMGeometryData {
        std::shared_ptr<vk::DeviceLocalBuffer> arrayBuffer;
        std::shared_ptr<vk::DeviceLocalBuffer> descBuffer;
        std::vector<VkMicromapUsageEXT> descHistogram;
        std::vector<VkMicromapUsageEXT> indexHistogram;
        VkMicromapEXT micromap = VK_NULL_HANDLE;
        std::shared_ptr<vk::Device> device;
        std::shared_ptr<vk::DeviceLocalBuffer> micromapBuffer;
        std::shared_ptr<vk::DeviceLocalBuffer> micromapScratchBuffer;
        bool hasMicromap = false;
    };
    std::vector<OMMGeometryData> ommGeometryData;
    uint8_t vertexFormat = 0;
    struct OMMCpuResult {
        std::vector<int32_t> ommIndices;
        std::vector<uint8_t> mergedArrayData;
        std::vector<VkMicromapTriangleEXT> mergedDescs;
        std::vector<VkMicromapUsageEXT> descHistogram;
        std::vector<VkMicromapUsageEXT> indexHistogram;
        bool bakingDone = false;
        bool isTransparentOMM = false;
        bool isOpaqueOMM = false;
    };
    std::vector<OMMCpuResult> ommCpuResults;

    std::vector<VkAabbPositionsKHR> displacedAABBs;
    std::vector<vk::Data::DisplacedFaceData> displacedFaceData;
    uint32_t displacedFaceCount = 0;
    std::shared_ptr<vk::DeviceLocalBuffer> displacedAABBBuffer;
    std::shared_ptr<vk::DeviceLocalBuffer> displacedFaceDataBuffer;
    std::shared_ptr<vk::BLASBuilder> displacedBlasBuilder;
    std::shared_ptr<vk::BLAS> displacedBlas;

    uint32_t biomeGrassColor = 0x91BD59;
    uint32_t biomeFoliageColor = 0x77AB2F;
    uint32_t biomeWaterColor = 0x3F76E4;

    uint64_t textureGeneration = 0;
    std::vector<ChunkLightEntry> lightSources;

    std::shared_ptr<vk::BLAS> blas;
    std::shared_ptr<vk::BLASBuilder> blasBuilder;
    std::shared_ptr<vk::BLAS> preCompactionBlas;

    ChunkBuildData(int64_t id,
                   int x,
                   int y,
                   int z,
                   int64_t version,
                   uint32_t allVertexCount,
                   uint32_t allIndexCount,
                   uint32_t geometryCount,
                   std::vector<World::GeometryTypes> &&geometryTypes,
                   std::vector<std::vector<vk::VertexFormat::PBRTriangle>> &&vertices,
                   std::vector<std::vector<uint32_t>> &&indices);
    ~ChunkBuildData();

    void build(bool allowMicromapBake = true, bool skipOMM = false, glm::vec3 cameraPos = glm::vec3(0));
    void prepareCPU(bool allowMicromapBake, bool skipOMM, glm::vec3 cameraPos);
    void uploadGPU();

    void releaseStagingBuffers();
    void releaseHostGeometry();
};

struct Chunk1;

struct ChunkBuildDataBatch : public SharedObject<ChunkBuildDataBatch> {
    std::vector<std::shared_ptr<ChunkBuildData>> batchData;
    glm::vec3 cameraPos;

    VkQueryPool compactionQueryPool = VK_NULL_HANDLE;
    uint32_t compactionQueryCount = 0;
    std::shared_ptr<vk::Device> queryDevice;

    ~ChunkBuildDataBatch();

    ChunkBuildDataBatch() = default;

    ChunkBuildDataBatch(uint32_t maxBatchSize,
                        std::set<int64_t> &queuedIndex,
                        std::vector<std::shared_ptr<Chunk1>> &chunks,
                        std::vector<std::shared_ptr<ChunkBuildData>> &chunkBuildDatas,
                        glm::vec3 cameraPos);
};

class ChunkBuildScheduler : public SharedObject<ChunkBuildScheduler> {
  public:
    ChunkBuildScheduler(std::vector<std::shared_ptr<Chunk1>> &chunks,
                        std::vector<std::shared_ptr<ChunkBuildData>> &chunkBuildDatas,
                        std::recursive_mutex &mutex,
                        std::shared_ptr<vk::HostVisibleBuffer> &chunkPackedData,
                        uint32_t chunkBuildingBatchSize);
    ~ChunkBuildScheduler();

    void integrateCompleted();
    void waitAllFinish();

    void enqueue(std::shared_ptr<ChunkBuildData> data, glm::vec3 cameraPos, bool important = false);
    void updateCameraPos(glm::vec3 pos);

    uint32_t chunkBuildingBatchSize();

    bool pause(std::chrono::milliseconds timeout = std::chrono::milliseconds(5000));
    void resume();

    uint64_t lastSubmittedTimelineValue() const { return lastSubmittedTimeline_.load(std::memory_order_acquire); }

    uint32_t getInputQueueSize() {
        std::lock_guard<std::mutex> lock(inputMtx_);
        return static_cast<uint32_t>(inputQueue_.size());
    }

  private:
    std::vector<std::shared_ptr<Chunk1>> &chunks_;
    std::vector<std::shared_ptr<ChunkBuildData>> &chunkBuildDatas_;
    std::recursive_mutex &mutex_;
    std::shared_ptr<vk::HostVisibleBuffer> &chunkPackedData_;
    uint32_t chunkBuildingBatchSize_;

    std::thread blasThread_;
    std::atomic<bool> stop_{false};
    std::atomic<bool> blasThreadExited_{false};
    std::atomic<bool> paused_{false};
    std::atomic<bool> pausedAck_{false};
    std::atomic<float> cameraPosX_{0}, cameraPosY_{0}, cameraPosZ_{0};

    struct WorkerTask {
        std::shared_ptr<ChunkBuildData> data;
        float priority;
    };
    std::mutex inputMtx_;
    std::condition_variable inputCv_;
    std::deque<WorkerTask> inputQueue_;

    std::mutex completedMtx_;
    std::deque<std::shared_ptr<ChunkBuildData>> completedQueue_;

    struct InFlightBatch {
        uint64_t timelineValue;
        std::shared_ptr<vk::CommandBuffer> cmd;
        std::vector<std::shared_ptr<ChunkBuildData>> chunks;
        VkQueryPool compactionQP = VK_NULL_HANDLE;
        uint32_t compactionCount = 0;
        bool isCompaction = false;
    };
    std::deque<InFlightBatch> inFlight_;
    std::atomic<uint32_t> inFlightCount_{0};

    uint64_t blasTimelineCounter_{0};
    std::atomic<uint64_t> lastSubmittedTimeline_{0};

    std::atomic<uint64_t> totalEnqueued_{0};
    std::atomic<uint64_t> totalSubmitted_{0};
    std::atomic<uint64_t> totalCompleted_{0};
    std::atomic<uint64_t> totalIntegrated_{0};

    std::queue<std::shared_ptr<vk::CommandBuffer>> cmdPool_;

    void blasThreadLoop();
};

struct ChunkRenderData : public SharedObject<ChunkRenderData> {
    int x, y, z;
    std::shared_ptr<vk::BLAS> blas;
    uint32_t allVertexCount;
    uint32_t allIndexCount;
    uint32_t geometryCount;
    std::shared_ptr<std::vector<World::GeometryTypes>> geometryTypes;
    std::shared_ptr<std::vector<std::shared_ptr<vk::DeviceLocalBuffer>>> vertexBuffers;
    std::shared_ptr<std::vector<std::shared_ptr<vk::DeviceLocalBuffer>>> indexBuffers;
};

struct Chunk1 : public SharedObject<Chunk1> {
    constexpr static float T_HALF = 200;
    constexpr static float T_WEIGHT = 1.0;

    constexpr static float D_HALF = 48;
    constexpr static float D_SENSITIVITY = 1.5;
    constexpr static float D_WEIGHT = 2.0;

    int x, y, z;
    int64_t latestVersion = 0;
    std::chrono::steady_clock::time_point lastUpdate;

    std::shared_ptr<vk::BLAS> blas;
    int64_t blasVersion = -1;
    uint64_t blasGeneration = 0;
    uint8_t vertexFormat = 0;
    bool secondaryQueueOwnershipPending = false;
    std::shared_ptr<std::vector<std::shared_ptr<vk::DeviceLocalBuffer>>> vertexBuffers;
    std::shared_ptr<std::vector<std::shared_ptr<vk::DeviceLocalBuffer>>> indexBuffers;

    uint32_t allVertexCount;
    uint32_t allIndexCount;
    uint32_t geometryCount;
    std::shared_ptr<std::vector<World::GeometryTypes>> geometryTypes;

    uint32_t biomeGrassColor = 0x91BD59;
    uint32_t biomeFoliageColor = 0x77AB2F;
    uint32_t biomeWaterColor = 0x3F76E4;
    uint64_t textureGeneration = 0;
    std::vector<ChunkLightEntry> lightSources;

    std::shared_ptr<vk::BLAS> displacedBlas;
    std::shared_ptr<vk::DeviceLocalBuffer> displacedFaceDataBuffer;
    uint32_t displacedFaceCount = 0;

    float buildFactor(std::chrono::steady_clock::time_point currentTime, glm::vec3 cameraPos);
    void enqueue(std::shared_ptr<ChunkBuildData> chunkBuildData);
    void invalidate();
    std::shared_ptr<ChunkRenderData> tryGetValid();
};

struct ChunkPackedData {
    uint32_t geometryCount;
};

class Chunks : public SharedObject<Chunks> {
    friend World;

  public:
    Chunks(std::shared_ptr<Framework> framework);

    void reset(uint32_t numChunks);
    void resetScheduler();
    void resetFrame();
    void invalidateChunk(int id);
    void queueChunkBuild(ChunkBuildTask task);
    void queueBlockStateBuild(ChunkBuildTaskV2 task);
    void submitExtendedBuild(uint32_t extId, std::shared_ptr<ChunkBuildData> cbd);
    void ensureCapacity(uint32_t totalSlots);

    bool isChunkReady(int64_t id);
    uint32_t getInputQueueSize();

    void setChunkLights(int64_t id, const std::vector<ChunkLightEntry> &lights);
    void close();

    std::recursive_mutex &mutex();
    std::vector<std::shared_ptr<Chunk1>> &chunks();
    std::shared_ptr<ChunkBuildScheduler> chunkBuildScheduler();
    std::vector<std::shared_ptr<vk::BLASBuilder>> &importantBLASBuilders();
    std::shared_ptr<vk::HostVisibleBuffer> chunkPackedData();

  private:
    std::recursive_mutex mutex_;
    std::vector<std::shared_ptr<Chunk1>> chunks_;
    std::shared_ptr<vk::HostVisibleBuffer> chunkPackedData_ = nullptr;
    std::vector<std::shared_ptr<ChunkBuildData>> chunkBuildDatas_;
    std::set<int64_t> queuedIndex_;
    std::shared_ptr<ChunkBuildScheduler> chunkBuildScheduler_;

    std::shared_ptr<std::vector<std::shared_ptr<vk::BLASBuilder>>> importantBLASBuilders_;
};
