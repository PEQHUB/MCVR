#pragma once

#include "common/shared.hpp"
#include "common/singleton.hpp"
#include "core/all_extern.hpp"
#include "core/vulkan/all_core_vulkan.hpp"

#include "core/render/world.hpp"

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

    // Per-section biome colors for shader-side tinting (packed 0x00RRGGBB)
    uint32_t biomeGrassColor = 0x91BD59;
    uint32_t biomeFoliageColor = 0x77AB2F;
    uint32_t biomeWaterColor = 0x3F76E4;

    uint64_t textureGeneration = 0;

    std::shared_ptr<vk::BLAS> blas;
    std::shared_ptr<vk::BLASBuilder> blasBuilder;
    std::shared_ptr<vk::BLAS> preCompactionBlas; // kept alive until render thread GCs via Chunk1::enqueue()
    bool uploadValid = true;
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

    void build();

    // Split build into two phases for parallelization:
    // prepareCPU() counts full triangle geometry (no Vulkan calls, thread-safe).
    // uploadGPU() allocates/stages full PBRTriangle buffers and builds BLAS on the render thread.
    void prepareCPU();
    void uploadGPU();

    // Release staging buffers after GPU copy completes (timeline semaphore confirmed).
    // Frees ~50% of per-chunk buffer memory — staging halves are only needed during upload.
    void releaseStagingBuffers();

    // Release CPU-side vertex/index data after GPU upload + BLAS build.
    // Frees ~4GB of RAM across all loaded chunks.
    void releaseHostGeometry();
};

struct Chunk1;

struct ChunkBuildDataBatch : public SharedObject<ChunkBuildDataBatch> {
    std::vector<std::shared_ptr<ChunkBuildData>> batchData;
    glm::vec3 cameraPos;

    // BLAS compaction: query pool for compacted sizes (one query per BLAS in batch)
    VkQueryPool compactionQueryPool = VK_NULL_HANDLE;
    uint32_t compactionQueryCount = 0;
    std::shared_ptr<vk::Device> queryDevice;  // for cleanup

    ~ChunkBuildDataBatch();

    // Default constructor for pre-built batches (Phase 1: background workers fill batchData)
    ChunkBuildDataBatch() = default;

    // Legacy constructor (dequeues + prepareCPU + uploadGPU inline — used for important chunks)
    ChunkBuildDataBatch(uint32_t maxBatchSize,
                        std::set<int64_t> &queuedIndex,
                        std::vector<std::shared_ptr<Chunk1>> &chunks,
                        std::vector<std::shared_ptr<ChunkBuildData>> &chunkBuildDatas,
                        glm::vec3 cameraPos);
};

struct ChunkBuildSchedulerStats {
    uint32_t inputQueue = 0;
    uint32_t completedQueue = 0;
    uint32_t inFlight = 0;
    uint64_t enqueued = 0;
    uint64_t submitted = 0;
    uint64_t completed = 0;
    uint64_t integrated = 0;
    uint64_t lastSubmittedTimeline = 0;
    uint64_t currentTimeline = 0;
};

class ChunkBuildScheduler : public SharedObject<ChunkBuildScheduler> {
  public:
    ChunkBuildScheduler(std::vector<std::shared_ptr<Chunk1>> &chunks,
                        std::vector<std::shared_ptr<ChunkBuildData>> &chunkBuildDatas,
                        std::recursive_mutex &mutex,
                        std::shared_ptr<vk::HostVisibleBuffer> &chunkPackedData,
                        uint32_t chunkBuildingBatchSize);
    ~ChunkBuildScheduler();

    // Render thread only: drain completed chunks into Chunk1 (zero Vulkan work)
    void integrateCompleted();
    void waitAllFinish();

    // Push a chunk for background processing (called from JNI thread)
    void enqueue(std::shared_ptr<ChunkBuildData> data, glm::vec3 cameraPos, bool important = false);
    void updateCameraPos(glm::vec3 pos);

    uint32_t chunkBuildingBatchSize();

    // Pause BLAS thread during swapchain recreate to prevent submits in the critical window.
    // Returns false if the thread does not acknowledge before the timeout.
    bool pause(std::chrono::milliseconds timeout = std::chrono::milliseconds(5000));
    void resume();

    // Last BLAS timeline value submitted to secondary queue (for cross-queue GC sync)
    uint64_t lastSubmittedTimelineValue() const { return lastSubmittedTimeline_.load(std::memory_order_acquire); }

    // Queue depth for Java-side adaptive throttling
    uint32_t getInputQueueSize() {
        std::lock_guard<std::mutex> lock(inputMtx_);
        return static_cast<uint32_t>(inputQueue_.size());
    }

    // Diagnostic snapshot — non-blocking, no waits
    ChunkBuildSchedulerStats stats();

  private:
    std::vector<std::shared_ptr<Chunk1>> &chunks_;
    std::vector<std::shared_ptr<ChunkBuildData>> &chunkBuildDatas_;
    std::recursive_mutex &mutex_;
    std::shared_ptr<vk::HostVisibleBuffer> &chunkPackedData_;
    uint32_t chunkBuildingBatchSize_;

    // Background BLAS builder thread — sole owner of secondary queue.
    // Does everything: CPU geometry → VMA alloc → cmd recording → submit → compaction.
    // Hands off fully-built, compacted BLASes to render thread via completedQueue_.
    std::thread blasThread_;
    std::atomic<bool> stop_{false};
    std::atomic<bool> blasThreadExited_{false};
    std::atomic<bool> paused_{false};
    std::atomic<bool> pausedAck_{false};
    std::atomic<float> cameraPosX_{0}, cameraPosY_{0}, cameraPosZ_{0};

    // Input queue: JNI pushes chunks, BLAS thread consumes them.
    struct WorkerTask {
        std::shared_ptr<ChunkBuildData> data;
        float priority;
    };
    std::mutex inputMtx_;
    std::condition_variable inputCv_;
    std::deque<WorkerTask> inputQueue_;

    // Output queue: BLAS thread pushes completed chunks, render thread drains them.
    std::mutex completedMtx_;
    std::deque<std::shared_ptr<ChunkBuildData>> completedQueue_;

    // In-flight batches on GPU — BLAS thread owns these exclusively (no mutex needed)
    struct InFlightBatch {
        uint64_t timelineValue;
        std::shared_ptr<vk::CommandBuffer> cmd;
        std::vector<std::shared_ptr<ChunkBuildData>> chunks;
        VkQueryPool compactionQP = VK_NULL_HANDLE;
        uint32_t compactionCount = 0;
        bool isCompaction = false; // true = compaction phase, false = build phase
    };
    std::deque<InFlightBatch> inFlight_;

    // BLAS timeline counter — only incremented by BLAS thread (sole owner, no atomic needed)
    uint64_t blasTimelineCounter_{0};

    // Last submitted timeline value — atomic for cross-thread read by render thread GC sync
    std::atomic<uint64_t> lastSubmittedTimeline_{0};

    // Diagnostic counters (atomic for cross-thread reads)
    std::atomic<uint64_t> totalEnqueued_{0};
    std::atomic<uint64_t> totalSubmitted_{0};
    std::atomic<uint64_t> totalCompleted_{0};
    std::atomic<uint64_t> totalIntegrated_{0};

    // BLAS thread cmd pool — owned exclusively, no mutex
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
    constexpr static float T_HALF = 200; // ms
    constexpr static float T_WEIGHT = 1.0;

    constexpr static float D_HALF = 48; // blocks (sharper distance falloff for spiral loading)
    constexpr static float D_SENSITIVITY = 1.5;
    constexpr static float D_WEIGHT = 2.0; // stronger distance preference

    int x, y, z;
    int64_t latestVersion = 0;
    std::chrono::steady_clock::time_point lastUpdate;

    std::shared_ptr<vk::BLAS> blas;
    int64_t blasVersion = -1;
    uint64_t blasGeneration = 0;  // incremented on each BLAS swap, for TLAS UPDATE change detection
    std::shared_ptr<std::vector<std::shared_ptr<vk::DeviceLocalBuffer>>> vertexBuffers;
    std::shared_ptr<std::vector<std::shared_ptr<vk::DeviceLocalBuffer>>> indexBuffers;

    uint32_t allVertexCount;
    uint32_t allIndexCount;
    uint32_t geometryCount;
    std::shared_ptr<std::vector<World::GeometryTypes>> geometryTypes;

    // Per-section biome colors for shader-side tinting (packed 0x00RRGGBB)
    uint32_t biomeGrassColor = 0x91BD59;    // default plains green
    uint32_t biomeFoliageColor = 0x77AB2F;  // default foliage
    uint32_t biomeWaterColor = 0x3F76E4;    // default water blue
    uint64_t textureGeneration = 0;

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
    void resetScheduler(bool invalidateExisting = false);
    void resetFrame();
    void invalidateChunk(int id);
    void queueChunkBuild(ChunkBuildTask task);

    bool isChunkReady(int64_t id);
    uint32_t getInputQueueSize();

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
