#include "core/render/chunks.hpp"

#include "core/render/buffers.hpp"
#include "core/render/crash_ring_buffer.hpp"
#include "core/render/pipeline.hpp"
#include "core/render/render_framework.hpp"
#include "core/render/renderer.hpp"
#include "core/render/modules/world/ray_tracing/ray_tracing_module.hpp"
#include "core/render/modules/world/ray_tracing/submodules/world_prepare.hpp"
#include "core/vulkan/debug_utils.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <set>


ChunkBuildData::ChunkBuildData(int64_t id,
                               int x,
                               int y,
                               int z,
                               int64_t version,
                               uint32_t allVertexCount,
                               uint32_t allIndexCount,
                               uint32_t geometryCount,
                               std::vector<World::GeometryTypes> &&geometryTypes,
                               std::vector<std::vector<vk::VertexFormat::PBRTriangle>> &&vertices,
                               std::vector<std::vector<uint32_t>> &&indices)
    : id(id),
      x(x),
      y(y),
      z(z),
      version(version),
      allVertexCount(allVertexCount),
      allIndexCount(allIndexCount),
      geometryCount(geometryCount),
      geometryTypes(std::move(geometryTypes)),
      vertices(std::move(vertices)),
      indices(std::move(indices)),
      blas(nullptr),
      blasBuilder(nullptr) {}

ChunkBuildData::~ChunkBuildData() = default;

void ChunkBuildData::build() {
    prepareCPU();
    uploadGPU();
}

// ---- CPU PHASE: count full triangle geometry ----
// No Vulkan or VMA calls. Safe to call from worker threads.
void ChunkBuildData::prepareCPU() {
    allVertexCount = 0;
    allIndexCount = 0;
    for (int i = 0; i < geometryCount; i++) {
        allVertexCount += static_cast<uint32_t>(vertices[i].size());
        allIndexCount += static_cast<uint32_t>(indices[i].size());
    }
}
// ---- GPU PHASE: VMA allocation, staging uploads, BLAS builder setup ----
// Must run on render thread (single-threaded Vulkan access).
void ChunkBuildData::uploadGPU() {
    uploadValid = true;
    vertexBuffers.clear();
    indexBuffers.clear();

    auto framework = Renderer::instance().framework();
    auto vma = framework->vma();
    auto device = framework->device();
    auto physicalDevice = framework->physicalDevice();

    auto failUpload = [&](const char *what, int geometryIndex, VkDeviceSize bytes) {
        uploadValid = false;
        std::cerr << "[ChunkBuildData] Dropping chunk build after GPU upload failure: id=" << id
                  << " geometry=" << geometryIndex
                  << " resource=" << what
                  << " bytes=" << bytes
                  << " vertices=" << allVertexCount
                  << " indices=" << allIndexCount << std::endl;
        vertexBuffers.clear();
        indexBuffers.clear();
        blas.reset();
        blasBuilder.reset();
    };

    auto requireBuffer = [&](const std::shared_ptr<vk::DeviceLocalBuffer> &buffer,
                             const char *what,
                             int geometryIndex,
                             VkDeviceSize bytes) -> bool {
        if (!buffer || !buffer->isValid()) {
            failUpload(what, geometryIndex, bytes);
            return false;
        }
        return true;
    };

    for (int i = 0; i < geometryCount; i++) {
        VkDeviceSize vertexBufferSize = vertices[i].size() * sizeof(vk::VertexFormat::PBRTriangle);
        if (vertexBufferSize == 0 || indices[i].empty()) {
            vertexBuffers.push_back(nullptr);
            indexBuffers.push_back(nullptr);
            continue;
        }

        auto vertexBuffer = vk::DeviceLocalBuffer::create(
            vma, device, true, vertexBufferSize,
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        if (!requireBuffer(vertexBuffer, "vertex", i, vertexBufferSize)) return;
        vertexBuffer->uploadToStagingBuffer(vertices[i].data());
        vertexBuffers.push_back(vertexBuffer);

        VkDeviceSize indexBufferSize = indices[i].size() * sizeof(uint32_t);
        auto indexBuffer = vk::DeviceLocalBuffer::create(
            vma, device, true, indexBufferSize,
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        if (!requireBuffer(indexBuffer, "index", i, indexBufferSize)) return;
        indexBuffer->uploadToStagingBuffer(indices[i].data());
        indexBuffers.push_back(indexBuffer);
    }

    blasBuilder = vk::BLASBuilder::create();
    auto blasGeometryBuilder = blasBuilder->beginGeometries();
    for (int i = 0; i < geometryCount; i++) {
        bool isOpaque = geometryTypes[i] == World::WORLD_SOLID;
        uint32_t indexCount = static_cast<uint32_t>(indices[i].size());
        uint32_t vertexCount = static_cast<uint32_t>(vertices[i].size());

        if (indexCount == 0) {
            blasGeometryBuilder->definePlaceholderGeometry();
            continue;
        }

        blasGeometryBuilder->defineTriangleGeomrtry<vk::VertexFormat::PBRTriangle>(
            vertexBuffers[i], vertexCount, indexBuffers[i], indexCount, isOpaque);
    }
    blasGeometryBuilder->endGeometries();
    blas = blasBuilder->defineBuildProperty(VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR)
               ->querySizeInfo(device)
               ->allocateBuffers(physicalDevice, device, vma)
               ->build(device);
    if (!blas || !blas->blasBuffer() || !blas->blasBuffer()->isValid()) {
        failUpload("blas", -1, 0);
    }
}
void ChunkBuildData::releaseHostGeometry() {
    for (auto &v : vertices) {
        v.clear();
        v.shrink_to_fit();
    }
    vertices.clear();
    vertices.shrink_to_fit();

    for (auto &idx : indices) {
        idx.clear();
        idx.shrink_to_fit();
    }
    indices.clear();
    indices.shrink_to_fit();

}

void ChunkBuildData::releaseStagingBuffers() {
    for (auto &vb : vertexBuffers) { if (vb) vb->releaseStagingBuffer(); }
    for (auto &ib : indexBuffers) { if (ib) ib->releaseStagingBuffer(); }
}

ChunkBuildDataBatch::ChunkBuildDataBatch(uint32_t maxBatchSize,
                                         std::set<int64_t> &queuedIndexSet,
                                         std::vector<std::shared_ptr<Chunk1>> &chunks,
                                         std::vector<std::shared_ptr<ChunkBuildData>> &chunkBuildDatas,
                                         glm::vec3 cameraPos)
    : cameraPos(cameraPos) {
    std::vector<int64_t> queuedIndices;
    std::copy(queuedIndexSet.begin(), queuedIndexSet.end(), std::back_inserter(queuedIndices));
    auto currentTime = std::chrono::steady_clock::now();
    std::sort(queuedIndices.begin(), queuedIndices.end(), [&](int64_t a, int64_t b) -> bool {
        return chunks[a]->buildFactor(currentTime, cameraPos) > chunks[b]->buildFactor(currentTime, cameraPos);
    });

    // Two-phase build: CPU work parallelized across thread pool, GPU work sequential.
    // prepareCPU() has zero Vulkan/VMA calls — safe for concurrent execution.
    // uploadGPU() does all VMA allocation + staging uploads on the render thread.
    size_t count = std::min(static_cast<size_t>(maxBatchSize), queuedIndices.size());
    batchData.resize(count);
    for (size_t i = 0; i < count; i++) {
        auto iter = queuedIndexSet.find(queuedIndices[i]);
        if (iter != queuedIndexSet.end()) { queuedIndexSet.erase(iter); }
        batchData[i] = chunkBuildDatas[queuedIndices[i]];
    }

    // Two-phase build: CPU work parallelized, GPU work sequential.
    // prepareCPU() has zero Vulkan/VMA calls — safe for concurrent execution.
    // SharedState in parallelFor lives on the heap to prevent use-after-free.
    Renderer::threadPool.parallelFor(static_cast<uint32_t>(count), [&](uint32_t i) {
        batchData[i]->prepareCPU();
    });

    for (size_t i = 0; i < count; i++) {
        batchData[i]->uploadGPU();
    }
    batchData.erase(std::remove_if(batchData.begin(), batchData.end(),
        [](const std::shared_ptr<ChunkBuildData> &data) {
            if (!data || data->uploadValid) return false;
            data->releaseHostGeometry();
            data->releaseStagingBuffers();
            return true;
        }), batchData.end());
}

ChunkBuildDataBatch::~ChunkBuildDataBatch() {
    if (compactionQueryPool != VK_NULL_HANDLE && queryDevice) {
        vkDestroyQueryPool(queryDevice->vkDevice(), compactionQueryPool, nullptr);
    }
}

ChunkBuildScheduler::ChunkBuildScheduler(std::vector<std::shared_ptr<Chunk1>> &chunks,
                                         std::vector<std::shared_ptr<ChunkBuildData>> &chunkBuildDatas,
                                         std::recursive_mutex &mutex,
                                         std::shared_ptr<vk::HostVisibleBuffer> &chunkPackedData,
                                         uint32_t chunkBuildingBatchSize)
    : chunks_(chunks),
      chunkBuildDatas_(chunkBuildDatas),
      mutex_(mutex),
      chunkPackedData_(chunkPackedData),
      chunkBuildingBatchSize_(chunkBuildingBatchSize) {
    blasThreadExited_.store(false, std::memory_order_release);
    blasThread_ = std::thread(&ChunkBuildScheduler::blasThreadLoop, this);
}

ChunkBuildScheduler::~ChunkBuildScheduler() {
    stop_.store(true);
    inputCv_.notify_one();
    if (blasThread_.joinable()) blasThread_.join();
}

void ChunkBuildScheduler::enqueue(std::shared_ptr<ChunkBuildData> data, glm::vec3 cameraPos, bool important) {
    float dx = data->x * 16.0f + 8.0f - cameraPos.x;
    float dy = data->y * 16.0f + 8.0f - cameraPos.y;
    float dz = data->z * 16.0f + 8.0f - cameraPos.z;
    float priority = 1.0f / (1.0f + (dx * dx + dy * dy + dz * dz) * 0.001f);
    if (important) priority *= 10.0f;  // important chunks processed first
    {
        std::lock_guard<std::mutex> lock(inputMtx_);
        inputQueue_.push_back({std::move(data), priority});
    }
    totalEnqueued_.fetch_add(1, std::memory_order_relaxed);
    inputCv_.notify_one();
}

void ChunkBuildScheduler::updateCameraPos(glm::vec3 pos) {
    cameraPosX_.store(pos.x, std::memory_order_relaxed);
    cameraPosY_.store(pos.y, std::memory_order_relaxed);
    cameraPosZ_.store(pos.z, std::memory_order_relaxed);
}

// Render thread only: integrate completed chunks from BLAS thread into Chunk1.
// Zero Vulkan work — just pointer swaps.
void ChunkBuildScheduler::integrateCompleted() {
    std::deque<std::shared_ptr<ChunkBuildData>> completed;
    {
        std::lock_guard<std::mutex> lock(completedMtx_);
        completed.swap(completedQueue_);
    }

    if (completed.empty()) return;
    totalCompleted_.fetch_add(static_cast<uint64_t>(completed.size()), std::memory_order_relaxed);

    std::unique_lock<std::recursive_mutex> lock(mutex_);
    for (auto &cbd : completed) {
        if (cbd->id >= static_cast<int64_t>(chunks_.size())) continue;
        chunks_[cbd->id]->enqueue(cbd);

        ChunkPackedData data = { .geometryCount = cbd->geometryCount };
        chunkPackedData_->uploadToBuffer(&data, sizeof(ChunkPackedData),
                                         cbd->id * sizeof(ChunkPackedData));

        // Release ChunkBuildData slot to free scratch geometry buffers
        // that were NOT transferred to Chunk1 (blasBuilder.).
        // Only clear if slot still points to this build (a newer build may have replaced it).
        if (cbd->id < static_cast<int64_t>(chunkBuildDatas_.size()) &&
            chunkBuildDatas_[cbd->id] == cbd) {
            chunkBuildDatas_[cbd->id] = nullptr;
        }
    }
    totalIntegrated_.fetch_add(static_cast<uint64_t>(completed.size()), std::memory_order_relaxed);
}

void ChunkBuildScheduler::waitAllFinish() {
    // Signal stop, join, then integrate remaining
    stop_.store(true);
    inputCv_.notify_one();
    if (blasThread_.joinable()) {
        // Never detach this owner thread. A detached BLAS worker can keep using
        // scheduler-owned queues, command pools, and buffers after reset frees them.
        auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (!blasThreadExited_.load(std::memory_order_acquire) &&
               std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        if (!blasThreadExited_.load(std::memory_order_acquire)) {
            std::cerr << "[Chunks] WARNING: BLAS thread did not finish in 5s; joining to preserve resource lifetime"
                      << std::endl;
            g_crashRing.record("BLAS:joinWait");
            g_crashRing.dumpToFile(Renderer::folderPath / "logs");
        }
        blasThread_.join();
    }
    integrateCompleted();
}

// BLAS builder thread: sole owner of the secondary Vulkan queue.
// Runs the complete chunk pipeline synchronously: CPU → GPU → fence wait → compaction.
// Blocking waits are free because this is a dedicated thread.
void ChunkBuildScheduler::blasThreadLoop() {
    vk::DebugUtils::setCurrentThreadName("Radiance BLAS Builder");
    auto framework = Renderer::instance().framework();
    auto vma = framework->vma();
    auto device = framework->device();
    auto physicalDevice = framework->physicalDevice();
    auto asyncPool = framework->asyncCommandPool();
    auto secondaryQueueIndex = physicalDevice->secondaryQueueIndex();

    // Pre-allocate cmd pool — sole owner, no mutex needed.
    const uint32_t maxInFlight = Renderer::options.displacementEnabled ? 2u : 12u;
    for (uint32_t i = 0; i < maxInFlight; i++) {
        cmdPool_.push(vk::CommandBuffer::create(device, asyncPool));
    }

    // Timeline semaphore for BLAS thread — secondary queue only.
    auto blasSem = device->blasSemaphore();
    // Initialize counter from current semaphore value — handles scheduler restart
    // (render distance change recreates scheduler but Device-level semaphore persists).
    blasTimelineCounter_ = blasSem->getValue();

    static uint64_t totalOrig = 0, totalComp = 0;
    static uint32_t totalN = 0;

    // --- BLAS diagnostic log: persistent, survives TDR ---
    // Deferred open: Renderer::folderPath may not be set yet at BLAS thread start
    std::ofstream diagLog;
    auto diagTs = []() {
        auto now = std::chrono::steady_clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
    };
    uint64_t diagIter = 0;
    auto ensureDiagOpen = [&]() {
        if (!diagLog.is_open() && !Renderer::folderPath.empty()) {
            auto diagPath = Renderer::folderPath / "logs" / "blas_diag.log";
            std::filesystem::create_directories(diagPath.parent_path());
            diagLog.open(diagPath, std::ios::trunc);
            if (diagLog.is_open()) {
                diagLog << "BLAS_DIAG_START timelineInit=" << blasTimelineCounter_
                        << " maxInFlight=" << maxInFlight
                        << " path=" << diagPath.string() << std::endl;
                diagLog.flush();
            }
        }
    };
    ensureDiagOpen();

    while (!stop_.load()) {
        if (diagLog.is_open() && diagIter % 2000 == 0) { diagLog << diagTs() << " HEARTBEAT iter=" << diagIter << " inFlight=" << inFlight_.size() << " cmdPool=" << cmdPool_.size() << std::endl; diagLog.flush(); }
        // Pause gate: render thread requests pause during swapchain recreate
        if (paused_.load(std::memory_order_acquire)) {
            pausedAck_.store(true, std::memory_order_release);
            if (diagLog.is_open()) {
                diagLog << diagTs() << " PAUSE_ACK iter=" << diagIter
                        << " inFlight=" << inFlight_.size() << std::endl;
                diagLog.flush();
            }
            while (paused_.load(std::memory_order_acquire) && !stop_.load()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            pausedAck_.store(false, std::memory_order_release);
            if (diagLog.is_open()) {
                diagLog << diagTs() << " PAUSE_RESUME iter=" << diagIter
                        << " stop=" << stop_.load() << std::endl;
                diagLog.flush();
            }
            if (stop_.load()) break;
        }

        diagIter++;
        ensureDiagOpen();
        // ---- POLL: check completed batches via timeline counter (non-blocking) ----
        uint64_t currentTimelineVal = blasSem->getValue();
        while (!inFlight_.empty()) {
            auto &front = inFlight_.front();
            if (currentTimelineVal < front.timelineValue) break;  // oldest not done yet

            if (diagLog.is_open()) {
                diagLog << diagTs() << " COMPLETE iter=" << diagIter
                        << " tv=" << front.timelineValue
                        << " type=" << (front.isCompaction ? "compact" : "build")
                        << " chunks=" << front.chunks.size()
                        << " inFlight=" << inFlight_.size()
                        << " gpuVal=" << currentTimelineVal << std::endl;
            }

            if (front.isCompaction) {
                // Compaction phase complete — destroy query pool and hand off
                if (front.compactionQP) vkDestroyQueryPool(device->vkDevice(), front.compactionQP, nullptr);

                if (totalN > 0 && totalN % 50 == 0) {
                    float pct = totalOrig > 0 ? 100.0f * (1.0f - (float)totalComp / totalOrig) : 0;
                    std::cout << "[Compaction] " << totalN << " BLASes: " << (totalOrig/1024)
                              << " KB -> " << (totalComp/1024) << " KB (" << pct << "% savings)" << std::endl;
                }

                // Hand off to render thread — release staging buffers first (GPU copy done)
                for (auto &cbd : front.chunks) cbd->releaseStagingBuffers();
                {
                    std::lock_guard<std::mutex> lock(completedMtx_);
                    for (auto &cbd : front.chunks) completedQueue_.push_back(std::move(cbd));
                }
                cmdPool_.push(std::move(front.cmd));
                inFlight_.pop_front();

            } else if (front.compactionQP && front.compactionCount > 0 && !cmdPool_.empty()) {
                // Build phase complete with compaction queries — submit compaction async (non-blocking)
                std::vector<VkDeviceSize> sizes(front.compactionCount);
                bool queriesOk = vkGetQueryPoolResults(device->vkDevice(), front.compactionQP, 0,
                    front.compactionCount, sizes.size() * sizeof(VkDeviceSize), sizes.data(),
                    sizeof(VkDeviceSize), VK_QUERY_RESULT_64_BIT) == VK_SUCCESS;

                if (queriesOk) {
                    // Recycle build cmd, take fresh one for compaction
                    cmdPool_.push(std::move(front.cmd));
                    auto compCmd = std::move(cmdPool_.front()); cmdPool_.pop();

                    compCmd->begin();
                    uint32_t qi = 0;
                    for (auto &cbd : front.chunks) {
                        if (!cbd->blas || qi >= front.compactionCount) continue;
                        VkDeviceSize compSz = sizes[qi++], origSz = cbd->blas->blasBuffer()->size();
                        if (compSz == 0 || compSz >= origSz * 9 / 10) {
                            totalOrig += origSz; totalComp += origSz; totalN++; continue;
                        }
                        auto buf = vk::DeviceLocalBuffer::create(vma, device, false, compSz,
                            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR|VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                            0, VMA_MEMORY_USAGE_GPU_ONLY);
                        if (!buf || !buf->isValid()) {
                            totalOrig += origSz;
                            totalComp += origSz;
                            totalN++;
                            continue;
                        }
                        VkAccelerationStructureCreateInfoKHR ci{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
                        ci.buffer = buf->vkBuffer(); ci.size = compSz; ci.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
                        VkAccelerationStructureKHR as;
                        if (vkCreateAccelerationStructureKHR(device->vkDevice(), &ci, nullptr, &as) == VK_SUCCESS) {
                            VkCopyAccelerationStructureInfoKHR cp{VK_STRUCTURE_TYPE_COPY_ACCELERATION_STRUCTURE_INFO_KHR};
                            cp.src = cbd->blas->blas(); cp.dst = as; cp.mode = VK_COPY_ACCELERATION_STRUCTURE_MODE_COMPACT_KHR;
                            vkCmdCopyAccelerationStructureKHR(compCmd->vkCommandBuffer(), &cp);
                            cbd->preCompactionBlas = cbd->blas;
                            cbd->blas = vk::BLAS::create(device, as, buf);
                        }
                        totalOrig += origSz; totalComp += compSz; totalN++;
                    }
                    compCmd->end();

                    // Submit compaction with timeline signal (non-blocking — polled next iteration)
                    uint64_t compactValue = ++blasTimelineCounter_;
                    VkSemaphore blasSemHandle = blasSem->vkSemaphore();
                    VkTimelineSemaphoreSubmitInfo tsi{};
                    tsi.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
                    tsi.signalSemaphoreValueCount = 1;
                    tsi.pSignalSemaphoreValues = &compactValue;
                    VkSubmitInfo csi{VK_STRUCTURE_TYPE_SUBMIT_INFO};
                    csi.pNext = &tsi;
                    csi.commandBufferCount = 1; csi.pCommandBuffers = &compCmd->vkCommandBuffer();
                    csi.signalSemaphoreCount = 1; csi.pSignalSemaphores = &blasSemHandle;
                    VkResult cr;
                    {
                        std::lock_guard<std::mutex> qLock(device->queueMutex());
                        vk::DebugUtils::ScopedQueueLabel queueLabel(
                            device->secondaryQueue(), "Radiance QueueSubmit: BLAS Compact", 0.9f, 0.55f, 0.2f);
                        cr = vkQueueSubmit(device->secondaryQueue(), 1, &csi, VK_NULL_HANDLE);
                    }
                    if (diagLog.is_open()) {
                        diagLog << diagTs() << " COMPACT_SUBMIT iter=" << diagIter
                                << " tv=" << compactValue << " vk=" << cr
                                << " inFlight=" << inFlight_.size() << std::endl;
                        diagLog.flush();
                    }
                    if (cr == VK_SUCCESS) {
                        lastSubmittedTimeline_.store(compactValue, std::memory_order_release);
                        auto chunks = std::move(front.chunks);
                        auto qp = front.compactionQP;
                        auto qpCount = front.compactionCount;
                        inFlight_.pop_front();
                        inFlight_.push_back({compactValue, std::move(compCmd),
                            std::move(chunks), qp, qpCount, true});
                    } else {
                        // Submit failed — propagate device lost
                        --blasTimelineCounter_;
                        g_crashRing.record("compactionSubmitFail", cr);
                        if (cr == VK_ERROR_DEVICE_LOST) {
                            if (diagLog.is_open()) { diagLog << diagTs() << " DEVICE_LOST_COMPACT" << std::endl; diagLog.flush(); }
                            crashExitWithQueue(cr, "BLAS compaction submit DEVICE_LOST", device->secondaryQueue());
                        }
                        if (front.compactionQP) vkDestroyQueryPool(device->vkDevice(), front.compactionQP, nullptr);
                        for (auto &cbd : front.chunks) cbd->releaseStagingBuffers();
                        {
                            std::lock_guard<std::mutex> lock(completedMtx_);
                            for (auto &cbd : front.chunks) completedQueue_.push_back(std::move(cbd));
                        }
                        cmdPool_.push(std::move(compCmd));
                        inFlight_.pop_front();
                    }
                } else {
                    // Query failed — hand off without compaction
                    if (front.compactionQP) vkDestroyQueryPool(device->vkDevice(), front.compactionQP, nullptr);
                    for (auto &cbd : front.chunks) cbd->releaseStagingBuffers();
                    {
                        std::lock_guard<std::mutex> lock(completedMtx_);
                        for (auto &cbd : front.chunks) completedQueue_.push_back(std::move(cbd));
                    }
                    cmdPool_.push(std::move(front.cmd));
                    inFlight_.pop_front();
                }

            } else {
                // Build phase complete, no compaction needed — hand off directly
                if (front.compactionQP) vkDestroyQueryPool(device->vkDevice(), front.compactionQP, nullptr);
                for (auto &cbd : front.chunks) cbd->releaseStagingBuffers();
                {
                    std::lock_guard<std::mutex> lock(completedMtx_);
                    for (auto &cbd : front.chunks) completedQueue_.push_back(std::move(cbd));
                }
                cmdPool_.push(std::move(front.cmd));
                inFlight_.pop_front();
            }
        }

        // ---- SUBMIT: prepare + submit new batch if we have work AND capacity ----
        if (diagLog.is_open() && diagIter % 100 == 0) {
            std::lock_guard<std::mutex> ilock(inputMtx_);
            diagLog << diagTs() << " POLL iter=" << diagIter
                    << " inputQ=" << inputQueue_.size()
                    << " inFlight=" << inFlight_.size()
                    << " cmdPool=" << cmdPool_.size()
                    << " paused=" << paused_.load()
                    << " stop=" << stop_.load() << std::endl;
            diagLog.flush();
        }
        bool hasWork = false;
        {
            std::lock_guard<std::mutex> lock(inputMtx_);
            hasWork = !inputQueue_.empty();
        }

        if (hasWork && inFlight_.size() < maxInFlight && !cmdPool_.empty()) {
            // Dequeue batch — dynamic size based on GPU headroom
            std::vector<std::shared_ptr<ChunkBuildData>> batch;
            {
                std::lock_guard<std::mutex> lock(inputMtx_);
                std::sort(inputQueue_.begin(), inputQueue_.end(),
                          [](const WorkerTask &a, const WorkerTask &b) { return a.priority > b.priority; });
                uint32_t count = std::min(chunkBuildingBatchSize_, static_cast<uint32_t>(inputQueue_.size()));
                for (uint32_t i = 0; i < count; i++) {
                    batch.push_back(std::move(inputQueue_.front().data));
                    inputQueue_.pop_front();
                }
            }

            if (!batch.empty()) {
                glm::vec3 cameraPos(cameraPosX_.load(std::memory_order_relaxed),
                                    cameraPosY_.load(std::memory_order_relaxed),
                                    cameraPosZ_.load(std::memory_order_relaxed));

                // CPU phase (parallel)
                            if (diagLog.is_open()) { diagLog << diagTs() << " PREPARE_CPU_BEGIN iter=" << diagIter << " batch=" << batch.size() << std::endl; diagLog.flush(); }
// DIAGNOSTIC: sequential prepareCPU to avoid parallelFor deadlock
            for (uint32_t i = 0; i < static_cast<uint32_t>(batch.size()); i++) {
                if (diagLog.is_open()) { diagLog << diagTs() << " prepareCPU_item iter=" << diagIter << " i=" << i << " id=" << batch[i]->id << " geo=" << batch[i]->geometryCount << std::endl; diagLog.flush(); }
                batch[i]->prepareCPU();
                if (diagLog.is_open()) { diagLog << diagTs() << " prepareCPU_item_done iter=" << diagIter << " i=" << i << std::endl; diagLog.flush(); }
            }
            if (diagLog.is_open()) { diagLog << diagTs() << " PREPARE_CPU_END iter=" << diagIter << std::endl; diagLog.flush(); }
                // GPU upload (sequential — VMA alloc + staging)
                            if (diagLog.is_open()) { diagLog << diagTs() << " UPLOAD_GPU_BEGIN iter=" << diagIter << std::endl; diagLog.flush(); }
for (auto &cbd : batch) cbd->uploadGPU();
                batch.erase(std::remove_if(batch.begin(), batch.end(),
                    [&](const std::shared_ptr<ChunkBuildData> &data) {
                        if (!data || data->uploadValid) return false;
                        if (diagLog.is_open()) {
                            diagLog << diagTs() << " DROP_UPLOAD_FAILED iter=" << diagIter
                                    << " id=" << (data ? data->id : -1)
                                    << " vertices=" << (data ? data->allVertexCount : 0)
                                    << " indices=" << (data ? data->allIndexCount : 0)
                                    << std::endl;
                            diagLog.flush();
                        }
                        if (data) {
                            data->releaseHostGeometry();
                            data->releaseStagingBuffers();
                        }
                        return true;
                    }), batch.end());
                if (batch.empty()) {
                    if (diagLog.is_open()) {
                        diagLog << diagTs() << " DROP_EMPTY_BATCH_AFTER_UPLOAD iter=" << diagIter << std::endl;
                        diagLog.flush();
                    }
                    continue;
                }
            if (diagLog.is_open()) { diagLog << diagTs() << " UPLOAD_GPU_END iter=" << diagIter << std::endl; diagLog.flush(); }

                // Release CPU-side geometry data — staging buffers hold the copy for GPU transfer.
                // This frees ~200KB/chunk of system RAM (PBRTriangle arrays + index arrays).
                for (auto &cbd : batch) cbd->releaseHostGeometry();

                // Take cmd from pool
                auto cmd = std::move(cmdPool_.front()); cmdPool_.pop();

                // Record
                cmd->begin();
                for (auto &cbd : batch) {
                    for (int i = 0; i < cbd->geometryCount; i++) {
                        if (!cbd->vertexBuffers[i] || !cbd->indexBuffers[i]) continue;
                        cbd->vertexBuffers[i]->uploadToBuffer(cmd);
                        cbd->indexBuffers[i]->uploadToBuffer(cmd);
                    }
                }

                // Barriers
                std::vector<vk::CommandBuffer::BufferMemoryBarrier> barriers;
                for (auto &cbd : batch) {
                    for (int i = 0; i < cbd->geometryCount; i++) {
                        if (!cbd->vertexBuffers[i] || !cbd->indexBuffers[i]) continue;
                        auto bar = [&](std::shared_ptr<vk::DeviceLocalBuffer> &b) {
                            barriers.push_back({VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                VK_ACCESS_2_MEMORY_READ_BIT|VK_ACCESS_2_MEMORY_WRITE_BIT,
                                VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                                VK_ACCESS_2_MEMORY_READ_BIT|VK_ACCESS_2_MEMORY_WRITE_BIT,
                                secondaryQueueIndex, secondaryQueueIndex, b});
                        };
                        bar(cbd->vertexBuffers[i]); bar(cbd->indexBuffers[i]);
                    }
                }
                cmd->barriersBufferImage(barriers, {});

                // BLAS builds
                std::vector<std::shared_ptr<vk::BLASBuilder>> builders;
                for (auto &cbd : batch) {
                    builders.push_back(cbd->blasBuilder);
                }
                vk::BLASBuilder::batchSubmit(builders, cmd);

                // Compaction queries — only valid when BLAS was built with ALLOW_COMPACTION
                VkQueryPool qp = VK_NULL_HANDLE; uint32_t qpCount = 0;

                cmd->end();

                // Submit with timeline signal (sole owner of secondary queue — no mutex)
                uint64_t batchValue = ++blasTimelineCounter_;
                VkSemaphore blasSemHandle = blasSem->vkSemaphore();
                VkTimelineSemaphoreSubmitInfo btsi{};
                btsi.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
                btsi.signalSemaphoreValueCount = 1;
                btsi.pSignalSemaphoreValues = &batchValue;
                VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
                si.pNext = &btsi;
                si.commandBufferCount = 1; si.pCommandBuffers = &cmd->vkCommandBuffer();
                si.signalSemaphoreCount = 1; si.pSignalSemaphores = &blasSemHandle;
                VkResult r;
                {
                    std::lock_guard<std::mutex> qLock(device->queueMutex());
                    vk::DebugUtils::ScopedQueueLabel queueLabel(
                        device->secondaryQueue(), "Radiance QueueSubmit: BLAS Build", 0.9f, 0.35f, 0.25f);
                    r = vkQueueSubmit(device->secondaryQueue(), 1, &si, VK_NULL_HANDLE);
                }
                if (diagLog.is_open()) {
                    diagLog << diagTs() << " BUILD_SUBMIT iter=" << diagIter
                            << " tv=" << batchValue << " vk=" << r
                            << " chunks=" << batch.size()
                            << " inFlight=" << inFlight_.size() << std::endl;
                    diagLog.flush();
                }
                if (r != VK_SUCCESS) {
                    --blasTimelineCounter_;  // rollback — signal never issued
                    g_crashRing.record("blasThreadSubmitFail", r);
                    if (r == VK_ERROR_DEVICE_LOST) {
                        if (diagLog.is_open()) { diagLog << diagTs() << " DEVICE_LOST_BUILD" << std::endl; diagLog.flush(); }
                        crashExitWithQueue(r, "BLAS build submit DEVICE_LOST", device->secondaryQueue());
                    }
                    cmdPool_.push(std::move(cmd));
                    if (qp) vkDestroyQueryPool(device->vkDevice(), qp, nullptr);
                    continue;
                }

                // Track in-flight — store timeline AFTER confirmed success
                lastSubmittedTimeline_.store(batchValue, std::memory_order_release);
                totalSubmitted_.fetch_add(static_cast<uint64_t>(batch.size()), std::memory_order_relaxed);
                inFlight_.push_back({batchValue, std::move(cmd), std::move(batch), qp, qpCount});
            }
        } else if (!hasWork && inFlight_.empty()) {
            if (diagLog.is_open() && diagIter % 500 == 0) {
                diagLog << diagTs() << " WAIT_INPUT iter=" << diagIter << std::endl;
                diagLog.flush();
            }
            // Nothing to do — wait for input
            std::unique_lock<std::mutex> lock(inputMtx_);
            inputCv_.wait_for(lock, std::chrono::milliseconds(5), [this] {
                return stop_.load() || !inputQueue_.empty();
            });
        } else {
            if (diagLog.is_open() && diagIter % 500 == 0) {
                diagLog << diagTs() << " YIELD iter=" << diagIter
                        << " hasWork=" << hasWork
                        << " inFlight=" << inFlight_.size()
                        << " cmdPool=" << cmdPool_.size() << std::endl;
                diagLog.flush();
            }
            // In-flight batches but no new work or no capacity — brief yield
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }

        // Periodic crash ring flush (every 500 iterations ≈ 50ms) so TDR doesn't lose data
        if (diagIter % 500 == 0) {
            auto logsDir = Renderer::folderPath / "logs";
            g_crashRing.dumpToFile(logsDir);
        }
    }

    if (diagLog.is_open()) {
        diagLog << diagTs() << " SHUTDOWN iter=" << diagIter
                << " tv=" << blasTimelineCounter_
                << " inFlight=" << inFlight_.size() << std::endl;
        diagLog.flush();
    }

    // Drain remaining in-flight batches on shutdown — single timeline wait covers all
    if (!inFlight_.empty()) {
        blasSem->waitValue(inFlight_.back().timelineValue);
    }
    for (auto &f : inFlight_) {
        if (f.compactionQP) vkDestroyQueryPool(device->vkDevice(), f.compactionQP, nullptr);
        for (auto &cbd : f.chunks) cbd->releaseStagingBuffers();
        std::lock_guard<std::mutex> lock(completedMtx_);
        for (auto &cbd : f.chunks) completedQueue_.push_back(std::move(cbd));
    }
    blasThreadExited_.store(true, std::memory_order_release);
}

uint32_t ChunkBuildScheduler::chunkBuildingBatchSize() {
    return chunkBuildingBatchSize_;
}

ChunkBuildSchedulerStats ChunkBuildScheduler::stats() {
    ChunkBuildSchedulerStats s;
    { std::lock_guard<std::mutex> lock(inputMtx_); s.inputQueue = static_cast<uint32_t>(inputQueue_.size()); }
    { std::lock_guard<std::mutex> lock(completedMtx_); s.completedQueue = static_cast<uint32_t>(completedQueue_.size()); }
    s.inFlight = static_cast<uint32_t>(inFlight_.size());
    s.enqueued = totalEnqueued_.load(std::memory_order_relaxed);
    s.submitted = totalSubmitted_.load(std::memory_order_relaxed);
    s.completed = totalCompleted_.load(std::memory_order_relaxed);
    s.integrated = totalIntegrated_.load(std::memory_order_relaxed);
    s.lastSubmittedTimeline = lastSubmittedTimeline_.load(std::memory_order_acquire);
    auto blasSem = Renderer::instance().framework()->device()->blasSemaphore();
    s.currentTimeline = blasSem ? blasSem->getValue() : 0;
    return s;
}

bool ChunkBuildScheduler::pause(std::chrono::milliseconds timeout) {
    if (stop_.load(std::memory_order_acquire)) return true;
    pausedAck_.store(false, std::memory_order_release);
    paused_.store(true, std::memory_order_release);
    inputCv_.notify_one();
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    // Spin until BLAS thread acknowledges — guarantees no submit is in progress
    while (!pausedAck_.load(std::memory_order_acquire)) {
        if (stop_.load(std::memory_order_acquire)) return true;
        if (std::chrono::steady_clock::now() >= deadline) {
            paused_.store(false, std::memory_order_release);
            inputCv_.notify_one();
            return false;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    return true;
}

void ChunkBuildScheduler::resume() {
    pausedAck_.store(false, std::memory_order_release);
    paused_.store(false, std::memory_order_release);
}

float Chunk1::buildFactor(std::chrono::steady_clock::time_point currentTime, glm::vec3 cameraPos) {
    double tDiff = std::chrono::duration<double, std::milli>(currentTime - lastUpdate).count();
    double dDiff = glm::distance(cameraPos, glm::vec3{x, y, z});

    double tScore = 1 - exp(-tDiff / T_HALF);
    double dScore = 1 / (1 + pow(dDiff / D_HALF, D_SENSITIVITY));
    double score = pow(tScore, T_WEIGHT) * pow(dScore, D_WEIGHT);

    return score;
}

void Chunk1::enqueue(std::shared_ptr<ChunkBuildData> chunkBuildData) {
    auto framework = Renderer::instance().framework();
    auto &gc = framework->gc();

    lastUpdate = std::chrono::steady_clock::now();
    x = chunkBuildData->x;
    y = chunkBuildData->y;
    z = chunkBuildData->z;

    if (chunkBuildData->version > blasVersion) {
        blasVersion = chunkBuildData->version;

        gc.collect(blas);
        if (chunkBuildData->preCompactionBlas) gc.collect(chunkBuildData->preCompactionBlas);
        blas = chunkBuildData->blas;
        blasGeneration++;

        gc.collect(vertexBuffers);
        vertexBuffers = std::make_shared<std::vector<std::shared_ptr<vk::DeviceLocalBuffer>>>(
            std::move(chunkBuildData->vertexBuffers));

        gc.collect(indexBuffers);
        indexBuffers = std::make_shared<std::vector<std::shared_ptr<vk::DeviceLocalBuffer>>>(
            std::move(chunkBuildData->indexBuffers));
    } else {
        gc.collect(chunkBuildData->blas);
        if (chunkBuildData->preCompactionBlas) gc.collect(chunkBuildData->preCompactionBlas);

        gc.collect(std::make_shared<std::vector<std::shared_ptr<vk::DeviceLocalBuffer>>>(
            std::move(chunkBuildData->vertexBuffers)));

        gc.collect(std::make_shared<std::vector<std::shared_ptr<vk::DeviceLocalBuffer>>>(
            std::move(chunkBuildData->indexBuffers)));
    }

    allVertexCount = chunkBuildData->allVertexCount;
    allIndexCount = chunkBuildData->allIndexCount;
    geometryCount = chunkBuildData->geometryCount;
    geometryTypes = std::make_shared<std::vector<World::GeometryTypes>>(std::move(chunkBuildData->geometryTypes));
    biomeGrassColor = chunkBuildData->biomeGrassColor;
    biomeFoliageColor = chunkBuildData->biomeFoliageColor;
    biomeWaterColor = chunkBuildData->biomeWaterColor;
    textureGeneration = chunkBuildData->textureGeneration;
}
void Chunk1::invalidate() {
    auto framework = Renderer::instance().framework();
    auto &gc = framework->gc();

    lastUpdate = std::chrono::steady_clock::now();

    blasVersion = latestVersion++;
    blasGeneration++;  // TLAS UPDATE detection: invalidation changes BLAS composition

    gc.collect(blas);
    blas = nullptr;

    gc.collect(vertexBuffers);
    vertexBuffers = nullptr;

    gc.collect(indexBuffers);
    indexBuffers = nullptr;

}

std::shared_ptr<ChunkRenderData> Chunk1::tryGetValid() {
    auto ret = ChunkRenderData::create();
    ret->x = x;
    ret->y = y;
    ret->z = z;
    ret->blas = blas;
    ret->vertexBuffers = vertexBuffers;
    ret->indexBuffers = indexBuffers;
    ret->allVertexCount = allVertexCount;
    ret->allIndexCount = allIndexCount;
    ret->geometryCount = geometryCount;
    ret->geometryTypes = geometryTypes;

    return ret;
}

Chunks::Chunks(std::shared_ptr<Framework> framework) {
    importantBLASBuilders_ = std::make_shared<std::vector<std::shared_ptr<vk::BLASBuilder>>>();
}

void Chunks::reset(uint32_t numChunks) {
    std::unique_lock<std::recursive_mutex> lock(mutex_);

    // 1. Stop BLAS builder thread — drains in-flight GPU work on secondary queue
    if (chunkBuildScheduler_) {
        chunkBuildScheduler_->waitAllFinish();
        chunkBuildScheduler_.reset();
    }

    auto framework = Renderer::instance().framework();
    auto device = framework->device();
    auto vma = framework->vma();

    // 2. Wait for ALL GPU work across all queues and all frames in flight.
    //    After this, no command buffer references old BLASes or vertex buffers.
    vkDeviceWaitIdle(device->vkDevice());

    // GC flush intentionally omitted — shared_ptrs naturally prevent use-after-free,
    // and the frame-indexed ring will clean up when each slot is reused.

    // Note: WorldPrepareContext caches (cachedChunks_, prevBlasSnapshot_, TLAS) are
    // cleared on the next render() call when chunk1s.size() changes.
    // vkDeviceWaitIdle above guarantees no GPU work references old data.

    int size = Renderer::instance().framework()->swapchain()->imageCount();

    importantBLASBuilders_ = std::make_shared<std::vector<std::shared_ptr<vk::BLASBuilder>>>();

    chunks_.clear();
    chunks_.resize(numChunks);
    chunkPackedData_ =
        vk::HostVisibleBuffer::create(vma, device, numChunks * sizeof(ChunkPackedData),
                                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    chunkBuildDatas_.clear();
    chunkBuildDatas_.resize(numChunks);
    queuedIndex_.clear();

    for (int i = 0; i < numChunks; i++) {
        chunks_[i] = Chunk1::create();
        chunkBuildDatas_[i] = nullptr;
    }

    uint32_t chunkBuildingBatchSize = Renderer::instance().options.chunkBuildingBatchSize;
    chunkBuildScheduler_ =
        ChunkBuildScheduler::create(chunks_, chunkBuildDatas_, mutex_, chunkPackedData_,
                                    chunkBuildingBatchSize);
}

void Chunks::resetScheduler(bool invalidateExisting) {
    std::unique_lock<std::recursive_mutex> lock(mutex_);

    if (chunkBuildScheduler_ == nullptr) return;

    chunkBuildScheduler_->waitAllFinish();
    queuedIndex_.clear();

    if (invalidateExisting) {
        for (size_t i = 0; i < chunks_.size(); ++i) {
            if (chunks_[i]) {
                chunks_[i]->invalidate();
            }
            if (i < chunkBuildDatas_.size()) {
                chunkBuildDatas_[i] = nullptr;
            }
        }
    }

    uint32_t chunkBuildingBatchSize = Renderer::instance().options.chunkBuildingBatchSize;
    chunkBuildScheduler_ =
        ChunkBuildScheduler::create(chunks_, chunkBuildDatas_, mutex_, chunkPackedData_,
                                    chunkBuildingBatchSize);
}

void Chunks::resetFrame() {
    std::unique_lock<std::recursive_mutex> lock(mutex_);
    auto framework = Renderer::instance().framework();
    auto &gc = framework->gc();

    gc.collect(importantBLASBuilders_);
    importantBLASBuilders_ = std::make_shared<std::vector<std::shared_ptr<vk::BLASBuilder>>>();
}

void Chunks::invalidateChunk(int id) {
    std::unique_lock<std::recursive_mutex> lock(mutex_);
    if (id < 0 || id >= static_cast<int>(chunks_.size())) return;
    chunks_[id]->invalidate();

    // Release ChunkBuildData to free scratch buffers for out-of-range chunks
    if (id < static_cast<int>(chunkBuildDatas_.size())) {
        chunkBuildDatas_[id] = nullptr;
    }

    ChunkPackedData data = {
        .geometryCount = 0,
    };

    chunkPackedData_->uploadToBuffer(&data, sizeof(ChunkPackedData), id * sizeof(ChunkPackedData));
}

// maybe called async
void Chunks::queueChunkBuild(ChunkBuildTask task) {
    uint64_t currentTextureGeneration = Renderer::textureSystem.generation();
    if (currentTextureGeneration != 0 && task.textureGeneration != currentTextureGeneration) {
        static int sStaleCount = 0;
        sStaleCount++;
        bool hasVisibleChunk = false;
        {
            std::unique_lock<std::recursive_mutex> lock(mutex_);
            hasVisibleChunk = task.id >= 0 &&
                task.id < static_cast<int64_t>(chunks_.size()) &&
                chunks_[task.id] &&
                chunks_[task.id]->blas;
        }
        if (sStaleCount <= 5 || sStaleCount % 1000 == 0) {
            std::cerr << "[Chunks] "
                      << (hasVisibleChunk ? "Dropping" : "Accepting")
                      << " stale Java-meshed chunk build (stale texture): taskGen="
                      << task.textureGeneration << " currentGen=" << currentTextureGeneration
                      << " chunk=" << task.id << " staleCount=" << sStaleCount << std::endl;
        }
        if (hasVisibleChunk) {
            // Preserve the existing visible chunk until a current-generation rebuild arrives.
            return;
        }
    }

    uint32_t allVertexCount = 0, allIndexCount = 0;
    std::vector<World::GeometryTypes> geometryTypes;
    std::vector<std::vector<vk::VertexFormat::PBRTriangle>> vertices;
    std::vector<std::vector<uint32_t>> indices;

    for (int i = 0; i < task.geometryCount; i++) {
        World::GeometryTypes geometryType = static_cast<World::GeometryTypes>(task.geometryTypes[i]);
        int geometryTexture = task.geometryTextures[i];
        geometryTypes.push_back(geometryType);

        auto &geometryVertices = vertices.emplace_back();
        auto &geometryIndices = indices.emplace_back();

        geometryVertices.resize(task.vertexCounts[i]);
        std::memcpy(geometryVertices.data(), task.vertices[i],
                    task.vertexCounts[i] * sizeof(vk::VertexFormat::PBRTriangle));

        for (int j = 0; j < task.vertexCounts[i]; j += 4) {
            geometryIndices.push_back(j + 0);
            geometryIndices.push_back(j + 1);
            geometryIndices.push_back(j + 2);
            geometryIndices.push_back(j + 2);
            geometryIndices.push_back(j + 3);
            geometryIndices.push_back(j + 0);
        }

        allVertexCount += geometryVertices.size();
        allIndexCount += geometryIndices.size();
    }

    auto framework = Renderer::instance().framework();
    auto vma = framework->vma();
    auto device = framework->device();
    auto physicalDevice = framework->physicalDevice();

    std::unique_lock<std::recursive_mutex> lock(mutex_);

    // Bounds check under lock: after render distance change, stale Java background
    // threads may submit builds with old chunk indices that exceed the resized array.
    if (task.id < 0 || task.id >= static_cast<int64_t>(chunks_.size())) return;

    std::shared_ptr<ChunkBuildData> chunkBuildData = ChunkBuildData::create(
        task.id, task.x, task.y, task.z, chunks_[task.id]->latestVersion++, allVertexCount, allIndexCount,
        task.geometryCount, std::move(geometryTypes), std::move(vertices), std::move(indices));
    chunkBuildData->textureGeneration = task.textureGeneration;

    // ALL chunks go to the BLAS thread — render thread does zero chunk building.
    // Important chunks get higher priority (2x) but are still async.
    // This means chunks appear 1-2 frames later but the render thread never stalls.
    chunkBuildDatas_[task.id] = chunkBuildData;
    glm::vec3 camPos = Renderer::instance().world() ? glm::vec3(Renderer::instance().world()->getCameraPos()) : glm::vec3(0);
    if (chunkBuildScheduler_) {
        chunkBuildScheduler_->enqueue(chunkBuildData, camPos, task.isImportant);
    }
}

uint32_t Chunks::getInputQueueSize() {
    if (!chunkBuildScheduler_) return 0;
    return chunkBuildScheduler_->getInputQueueSize();
}

bool Chunks::isChunkReady(int64_t id) {
    std::unique_lock<std::recursive_mutex> lock(mutex_);
    if (id < 0 || id >= static_cast<int64_t>(chunks_.size())) return false;
    auto chunkRenderData = chunks_[id]->tryGetValid();
    return chunkRenderData->blas != nullptr;
}

void Chunks::close() {
    std::unique_lock<std::recursive_mutex> lock(mutex_);
    queuedIndex_.clear();
}

std::recursive_mutex &Chunks::mutex() {
    return mutex_;
}

std::vector<std::shared_ptr<Chunk1>> &Chunks::chunks() {
    return chunks_;
}

std::shared_ptr<ChunkBuildScheduler> Chunks::chunkBuildScheduler() {
    std::unique_lock<std::recursive_mutex> lock(mutex_);
    return chunkBuildScheduler_;
}

std::vector<std::shared_ptr<vk::BLASBuilder>> &Chunks::importantBLASBuilders() {
    return *importantBLASBuilders_;
}

std::shared_ptr<vk::HostVisibleBuffer> Chunks::chunkPackedData() {
    return chunkPackedData_;
}
