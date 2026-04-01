#include "core/render/modules/world/ray_tracing/submodules/world_prepare.hpp"

#include "core/render/buffers.hpp"
#include "core/render/gpu_diagnostics.hpp"
#include "core/render/chunks.hpp"
#include "core/render/entities.hpp"
#include "core/render/lights.hpp"
#include "core/render/colorspace.hpp"
#include "core/render/modules/world/ray_tracing/ray_tracing_module.hpp"
#include "core/render/render_framework.hpp"
#include "core/render/renderer.hpp"
#include "core/render/world.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <unordered_map>
#include <glm/gtc/type_ptr.hpp>

WorldPrepare::WorldPrepare() {}

void WorldPrepare::init(std::shared_ptr<Framework> framework, std::shared_ptr<RayTracingModule> rayTracingModule) {
    framework_ = framework;
    rayTracingModule_ = rayTracingModule;
}

void WorldPrepare::build() {
    auto framework = framework_.lock();
    if (!framework) return;
    auto rayTracingModule = rayTracingModule_.lock();
    if (!rayTracingModule) return;
    uint32_t size = framework->swapchain()->imageCount();

    contexts_.resize(size);

    for (int i = 0; i < size; i++) {
        contexts_[i] = WorldPrepareContext::create(framework->contexts()[i], shared_from_this());
    }
}

WorldPrepareContext::WorldPrepareContext(std::shared_ptr<FrameworkContext> frameworkContext,
                                         std::shared_ptr<WorldPrepare> worldPrepare)
    : frameworkContext(frameworkContext), worldPrepare(worldPrepare) {}

void WorldPrepareContext::uploadBuffer(std::vector<uint32_t> &blasOffsets,
                                       std::vector<uint64_t> &vertexBufferAddrs,
                                       std::vector<uint64_t> &indexBufferAddrs,
                                       std::vector<uint64_t> &lastVertexBufferAddrs,
                                       std::vector<uint64_t> &lastIndexBufferAddrs,
                                       std::vector<glm::mat4> &lastObjToWorldMats,
                                       std::vector<glm::uvec4> &biomeColors) {
    auto context = frameworkContext.lock();
    if (!context) return;
    auto framework = context->framework.lock();
    if (!framework) return;
    auto vma = framework->vma();
    auto device = framework->device();
    auto physicalDevice = framework->physicalDevice();
    auto mainQueueIndex = physicalDevice->mainQueueIndex();
    auto cmdBuffer = context->worldCommandBuffer;

    constexpr VkBufferUsageFlags metaUsage =
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

    // Persistent per-context buffers: reuse when size matches, reallocate when size changes.
    // Safe because acquireContext() waits for this context's previous GPU work before reuse.
    // Exact-size: no over-allocation, no stale BDA addresses in buffer tail.
    auto ensureBuffer = [&](std::shared_ptr<vk::DeviceLocalBuffer> &buf,
                            VkDeviceSize &prevSize, const void *data, VkDeviceSize dataSize) {
        if (!buf || prevSize != dataSize) {
            buf = vk::DeviceLocalBuffer::create(vma, device, true, dataSize, metaUsage);
            prevSize = dataSize;
        }
        buf->uploadToStagingBuffer(const_cast<void*>(data));
    };

    ensureBuffer(blasOffsetsBuffer, blasOffsetsCapacity_,
                 blasOffsets.data(), blasOffsets.size() * sizeof(uint32_t));
    ensureBuffer(vertexBufferAddr, vertexBufferAddrCapacity_,
                 vertexBufferAddrs.data(), vertexBufferAddrs.size() * sizeof(uint64_t));
    ensureBuffer(indexBufferAddr, indexBufferAddrCapacity_,
                 indexBufferAddrs.data(), indexBufferAddrs.size() * sizeof(uint64_t));
    ensureBuffer(lastVertexBufferAddr, lastVertexBufferAddrCapacity_,
                 lastVertexBufferAddrs.data(), lastVertexBufferAddrs.size() * sizeof(uint64_t));
    ensureBuffer(lastIndexBufferAddr, lastIndexBufferAddrCapacity_,
                 lastIndexBufferAddrs.data(), lastIndexBufferAddrs.size() * sizeof(uint64_t));
    ensureBuffer(lastObjToWorldMat, lastObjToWorldMatCapacity_,
                 lastObjToWorldMats.data(), lastObjToWorldMats.size() * sizeof(glm::mat4));
    ensureBuffer(biomeColorBuffer, biomeColorCapacity_,
                 biomeColors.data(), biomeColors.size() * sizeof(glm::uvec4));

    std::vector<std::shared_ptr<vk::DeviceLocalBuffer>> rayTracingMetaData{{
        blasOffsetsBuffer,
        vertexBufferAddr,
        indexBufferAddr,
        lastVertexBufferAddr,
        lastIndexBufferAddr,
        lastObjToWorldMat,
        areaLightBuffer,
        biomeColorBuffer,
    }};

    std::vector<vk::CommandBuffer::BufferMemoryBarrier> uploadPreBufferBarriers, uploadPostBufferBarriers;

    for (auto buffer : rayTracingMetaData) {
        if (buffer == nullptr) continue;
        uploadPreBufferBarriers.push_back({
            .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            .dstAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
            .srcQueueFamilyIndex = mainQueueIndex,
            .dstQueueFamilyIndex = mainQueueIndex,
            .buffer = buffer,
        });
        uploadPostBufferBarriers.push_back({
            .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                            VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR |
                            VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
            .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
            .srcQueueFamilyIndex = mainQueueIndex,
            .dstQueueFamilyIndex = mainQueueIndex,
            .buffer = buffer,
        });
    }

    cmdBuffer->barriersBufferImage(uploadPreBufferBarriers, {});
    for (auto buffer : rayTracingMetaData) {
        if (buffer == nullptr) continue;
        buffer->uploadToBuffer(cmdBuffer);
    }
    cmdBuffer->barriersBufferImage(uploadPostBufferBarriers, {});
}

void WorldPrepareContext::render() {
    auto module = worldPrepare.lock();
    if (!module) return;

    std::shared_ptr<Framework> framework = Renderer::instance().framework();
    std::shared_ptr<FrameworkContext> context = frameworkContext.lock();
    if (!context) return;
    std::shared_ptr<vk::VMA> vma = framework->vma();
    std::shared_ptr<vk::Device> device = framework->device();
    std::shared_ptr<vk::PhysicalDevice> physicalDevice = framework->physicalDevice();
    std::shared_ptr<vk::CommandBuffer> worldCommandBuffer = context->worldCommandBuffer;

    auto chunks = Renderer::instance().world()->chunks();
    auto entities = Renderer::instance().world()->entities();
    auto cameraPos = Renderer::instance().world()->getCameraPos();

    GpuDiag::checkpoint(worldCommandBuffer->vkCommandBuffer(), GpuDiag::WORLD_PREPARE_BEGIN);

    // CPU profiling: accumulate sub-phase timing (logged every 120 frames with TLAS stats)
    using Clock = std::chrono::steady_clock;
    static float cpuAccCheck = 0, cpuAccSchedule = 0, cpuAccImportant = 0;
    static float cpuAccEntity = 0, cpuAccInstances = 0, cpuAccTlas = 0, cpuAccTotal = 0;
    static uint32_t cpuFrameCount = 0;
    auto cpuT0 = Clock::now();
    auto cpuMsSince = [](Clock::time_point t0) {
        return std::chrono::duration<float, std::milli>(Clock::now() - t0).count();
    };

    std::unique_lock<std::recursive_mutex> lock(chunks->mutex());

    auto cpuT1 = Clock::now();
    if (chunks->chunkBuildScheduler() != nullptr) {
        chunks->chunkBuildScheduler()->updateCameraPos(cameraPos);
        chunks->chunkBuildScheduler()->integrateCompleted();
        cpuAccCheck += cpuMsSince(cpuT1);
    }

    auto cpuT3 = Clock::now();
    GpuDiag::checkpoint(worldCommandBuffer->vkCommandBuffer(), GpuDiag::CHUNK_SCHEDULE_DONE);

    // Barrier: ensure vertex/index buffer TRANSFER writes from uploadCommandBuffer
    // are visible before BLAS builds read them. Without this, the GPU may speculatively
    // start BLAS builds while staging→device copies are still in flight.
    worldCommandBuffer->barriersMemory({vk::CommandBuffer::MemoryBarrier{
        .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
        .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        .dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR |
                         VK_ACCESS_2_SHADER_READ_BIT,
    }});

    auto cpuT4 = Clock::now();
    if (chunks->importantBLASBuilders().size() > 0) {
        GpuDiag::checkpoint(worldCommandBuffer->vkCommandBuffer(), GpuDiag::BLAS_BUILD_IMPORTANT);
        auto &builders = chunks->importantBLASBuilders();
        // Cap important BLAS builds per frame to prevent GPU TDR on teleport/world load.
        // Extra builders stay in the vector and get submitted next frame.
        constexpr size_t MAX_IMPORTANT_BLAS_PER_FRAME = 16;
        if (builders.size() <= MAX_IMPORTANT_BLAS_PER_FRAME) {
            vk::BLASBuilder::batchSubmit(builders, worldCommandBuffer);
        } else {
            std::vector<std::shared_ptr<vk::BLASBuilder>> thisFrame(
                builders.begin(), builders.begin() + MAX_IMPORTANT_BLAS_PER_FRAME);
            vk::BLASBuilder::batchSubmit(thisFrame, worldCommandBuffer);
            // Keep remaining for next frame
            builders.erase(builders.begin(), builders.begin() + MAX_IMPORTANT_BLAS_PER_FRAME);
        }
    }
    cpuAccImportant += cpuMsSince(cpuT4);

    auto cpuT5 = Clock::now();
    if (entities->blasBatchBuilder() != nullptr) {
        GpuDiag::checkpoint(worldCommandBuffer->vkCommandBuffer(), GpuDiag::BLAS_BUILD_ENTITY);
        entities->blasBatchBuilder()->submit(worldCommandBuffer);
    }
    cpuAccEntity += cpuMsSince(cpuT5);

    worldCommandBuffer->barriersMemory({vk::CommandBuffer::MemoryBarrier{
        .srcStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        .srcAccessMask =
            VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
        .dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        .dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR,
    }});

    auto cpuT6 = Clock::now();
    uint32_t blasAccu = 0, blasGroupAccu = 0;
    std::vector<uint32_t> blasOffset;
    std::vector<uint32_t> geometryTypes;
    std::vector<uint64_t> vertexBufferAddrs, indexBufferAddrs;
    std::vector<uint64_t> lastVertexBufferAddrs, lastIndexBufferAddrs;
    std::vector<glm::mat4> lastObjToWorldMats;
    std::vector<glm::uvec4> biomeColors;

    tlasBuilder = vk::TLASBuilder::create();
    auto &instanceBuilder = tlasBuilder->beginInstanceBuilder();
    int blasIndex = 0;

    // BLAS lifetime snapshot: keep shared_ptrs alive while GPU references the TLAS.
    // prevBlasSnapshot_ from the last use of this swapchain context is safe to release
    // (GPU finished with it before we re-acquired this context).
    // Declared BEFORE entity section so entity BLASes are tracked for UPDATE validity.
    TlasBlasSnapshot currBlasSnapshot;

    // Entity
    {
        auto entityBatch = entities->entityBatch();

        if (entityBatch != nullptr) {
            static std::queue<std::unordered_map<int, std::pair<std::shared_ptr<Entity>, VkTransformMatrixKHR>>>
                previousEntityRenderDataBatches;
            static std::unordered_map<int, std::pair<std::shared_ptr<Entity>, VkTransformMatrixKHR>> emptyMap;

            std::unordered_map<int, std::pair<std::shared_ptr<Entity>, VkTransformMatrixKHR>> &previousEntityRenderDataBatch =
                previousEntityRenderDataBatches.empty() ? emptyMap : previousEntityRenderDataBatches.back();
            if (previousEntityRenderDataBatches.size() > Renderer::instance().framework()->swapchain()->imageCount())
                previousEntityRenderDataBatches.pop();
            std::unordered_map<int, std::pair<std::shared_ptr<Entity>, VkTransformMatrixKHR>> &currentEntityRenderDataBatch =
                previousEntityRenderDataBatches.emplace();

            auto worldUniformBuffer = Renderer::instance().buffers()->worldUniformBuffer();
            auto ubo = static_cast<vk::Data::WorldUBO *>(worldUniformBuffer->mappedPtr());

            auto &entities1 = entityBatch->entities;
            for (int i = 0; i < entities1.size(); i++) {
                VkGeometryInstanceFlagsKHR flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
                // VkGeometryInstanceFlagsKHR flags = 0;
                VkTransformMatrixKHR transform;

                if (entities1[i]->prebuiltBLAS < 0) {
                    if (entities1[i]->coordinate == World::Coordinates::WORLD || !ubo) {
                        transform = {
                            1, 0, 0, static_cast<float>(entities1[i]->x - cameraPos.x), //
                            0, 1, 0, static_cast<float>(entities1[i]->y - cameraPos.y), //
                            0, 0, 1, static_cast<float>(entities1[i]->z - cameraPos.z), //
                        };
                    } else if (entities1[i]->coordinate == World::Coordinates::CAMERA) {
                        glm::mat4 viewMat = glm::transpose(ubo->cameraViewMatInv); // column major to row major

                        transform = {
                            viewMat[0][0], viewMat[0][1], viewMat[0][2], viewMat[0][3], //
                            viewMat[1][0], viewMat[1][1], viewMat[1][2], viewMat[1][3], //
                            viewMat[2][0], viewMat[2][1], viewMat[2][2], viewMat[2][3], //
                        };
                    } else if (entities1[i]->coordinate == World::Coordinates::CAMERA_SHIFT) {
                        glm::vec3 shift = glm::vec3(ubo->cameraViewMatInv[3]);
                        transform = {
                            1, 0, 0, shift.x, //
                            0, 1, 0, shift.y, //
                            0, 0, 1, shift.z, //
                        };
                    }

                    instanceBuilder.defineInstance(transform, blasIndex, entities1[i]->rtFlag, blasGroupAccu, flags,
                                                   entities1[i]->blas);
                    // Track entity BLAS in snapshot for TLAS UPDATE validity check + GPU lifetime.
                    // Entity BLASes are rebuilt every frame (new VkAccelerationStructureKHR handles),
                    // so using blasDeviceAddress as generation ensures canUpdate detects the change.
                    currBlasSnapshot.blases.push_back(entities1[i]->blas);
                    currBlasSnapshot.generations.push_back(entities1[i]->blas->blasDeviceAddress());
                } else {
                    // auto &prebuiltBLAS =
                    //     Renderer::instance().framework()->prebuiltBLASs()[entityRenderData->prebuiltBLAS];
                    // transform = prebuiltBLAS.align(*entityRenderData->vertices, *entityRenderData->indices);

                    // instanceBuilder.defineInstance(transform, blasIndex, entityRenderData->rtFlag, blasGroupAccu,
                    // flags,
                    //                                prebuiltBLAS.blas);
                    throw std::runtime_error("prebuilt blas not implemented yet!");
                }

                geometryTypes.push_back(World::GeometryTypes::SHADOW);
                geometryTypes.insert(geometryTypes.end(), entities1[i]->geometryTypes->begin(),
                                     entities1[i]->geometryTypes->end());

                for (int j = 0; j < entities1[i]->geometryCount; j++) {
                    vertexBufferAddrs.push_back((*entities1[i]->vertexBufferAddresses)[j]);
                    indexBufferAddrs.push_back((*entities1[i]->indexBufferAddresses)[j]);
                }

                // store current render data
                {
                    if (entities1[i]->hashCode) {
                        currentEntityRenderDataBatch[entities1[i]->hashCode].first = entities1[i];
                        currentEntityRenderDataBatch[entities1[i]->hashCode].second = transform;
                    }
                }

                // read previous render data
                {
                    glm::mat4 lastObjToWorldMat(1);
                    auto iter = previousEntityRenderDataBatch.find(entities1[i]->hashCode);
                    if (iter != previousEntityRenderDataBatch.end()) {
                        auto &previousEntityRenderData = (*iter).second.first;
                        if (previousEntityRenderData->geometryCount == entities1[i]->geometryCount) {
                            for (int j = 0; j < entities1[i]->geometryCount; j++) {
                                if ((*previousEntityRenderData->vertices)[j].size() ==
                                        (*entities1[i]->vertices)[j].size() &&
                                    (*previousEntityRenderData->indices)[j].size() ==
                                        (*entities1[i]->indices)[j].size()) {
                                    lastVertexBufferAddrs.push_back(
                                        (*previousEntityRenderData->vertexBufferAddresses)[j]);
                                    lastIndexBufferAddrs.push_back(
                                        (*previousEntityRenderData->indexBufferAddresses)[j]);
                                } else {
                                    lastVertexBufferAddrs.push_back(0);
                                    lastIndexBufferAddrs.push_back(0);
                                }
                            }
                        } else {
                            for (int j = 0; j < entities1[i]->geometryCount; j++) {
                                lastVertexBufferAddrs.push_back(0);
                                lastIndexBufferAddrs.push_back(0);
                            }
                        }

                        VkTransformMatrixKHR lastObjToWorldVkMat = iter->second.second;
                        lastObjToWorldMat = glm::transpose(glm::mat4(glm::make_vec4(lastObjToWorldVkMat.matrix[0]), //
                                                                     glm::make_vec4(lastObjToWorldVkMat.matrix[1]), //
                                                                     glm::make_vec4(lastObjToWorldVkMat.matrix[2]), //
                                                                     glm::vec4(0.0f, 0.0f, 0.0f, 1.0f)));
                    } else {
                        for (int j = 0; j < entities1[i]->geometryCount; j++) {
                            lastVertexBufferAddrs.push_back(0);
                            lastIndexBufferAddrs.push_back(0);
                        }
                    }
                    lastObjToWorldMats.push_back(lastObjToWorldMat);
                }

                blasOffset.push_back(blasAccu);
                blasAccu += entities1[i]->geometryCount;
                blasGroupAccu += entities1[i]->geometryCount + 1; // shadow

                blasIndex++;
            }
        }
    }

    // Chunk — cached + parallel instance population
    {
        auto &chunk1s = chunks->chunks();
        // Effective cull distance: max of user setting and extended render distance requirement
        float cullDist = Renderer::options.chunkCullDistance;
        if (Renderer::options.extendedRenderDistance > 0) {
            float extMinCull = static_cast<float>(
                (Renderer::javaRenderDistance + Renderer::options.extendedRenderDistance + 2) * 16);
            if (extMinCull > cullDist) cullDist = extMinCull;
        }
        float cullDist2 = cullDist * cullDist;

        // Update cache: only recompute metadata when blasGeneration changes
        if (cachedChunks_.size() != chunk1s.size()) cachedChunks_.resize(chunk1s.size());

        // Parallel pass 1: update cache + distance cull → collect visible chunk indices
        // Each thread fills a local list of visible chunk indices + GC items (thread-safe)
        uint32_t numChunks = static_cast<uint32_t>(chunk1s.size());
        uint32_t numThreads = std::max(1u, Renderer::threadPool.threadCount());
        struct LocalResult {
            std::vector<int> visible;
            std::vector<std::shared_ptr<void>> gcItems; // thread-local GC collection
        };
        std::vector<LocalResult> locals(numThreads);

        Renderer::threadPool.parallelFor(numThreads, [&](uint32_t t) {
            uint32_t start = t * numChunks / numThreads;
            uint32_t end = (t + 1) * numChunks / numThreads;
            auto &local = locals[t];

            for (uint32_t i = start; i < end; i++) {
                auto &chunk1 = chunk1s[i];
                if (chunk1->blas == nullptr) continue;

                // Update cache if generation changed
                auto &cc = cachedChunks_[i];
                if (cc.blasGeneration != chunk1->blasGeneration) {
                    // Defer GC to thread-local vector — gc().collect() is not thread-safe.
                    // Keeps old BLAS + vertex/index buffers alive so pool sub-allocations
                    // aren't reused while the GPU still reads them.
                    if (cc.blas) local.gcItems.push_back(cc.blas);
                    if (cc.displacedBlas) local.gcItems.push_back(cc.displacedBlas);
                    if (cc.vertexBuffers) local.gcItems.push_back(cc.vertexBuffers);
                    if (cc.indexBuffers) local.gcItems.push_back(cc.indexBuffers);
                    if (cc.displacedFaceDataBuffer) local.gcItems.push_back(cc.displacedFaceDataBuffer);
                    cc.blasGeneration = chunk1->blasGeneration;
                    cc.blas = chunk1->blas;
                    cc.x = chunk1->x; cc.y = chunk1->y; cc.z = chunk1->z;
                    cc.biomeGrassColor = chunk1->biomeGrassColor;
                    cc.biomeFoliageColor = chunk1->biomeFoliageColor;
                    cc.biomeWaterColor = chunk1->biomeWaterColor;
                    cc.geometryCount = chunk1->geometryCount;
                    cc.geoTypes.clear();
                    cc.geoTypes.push_back(World::GeometryTypes::SHADOW);
                    cc.geoTypes.insert(cc.geoTypes.end(), chunk1->geometryTypes->begin(), chunk1->geometryTypes->end());
                    cc.vertBufAddrs.clear();
                    cc.idxBufAddrs.clear();
                    for (uint32_t j = 0; j < chunk1->geometryCount; j++) {
                        cc.vertBufAddrs.push_back((*chunk1->vertexBuffers)[j]->bufferAddress());
                        cc.idxBufAddrs.push_back((*chunk1->indexBuffers)[j]->bufferAddress());
                    }
                    cc.vertexBuffers = chunk1->vertexBuffers;
                    cc.indexBuffers = chunk1->indexBuffers;
                    cc.displacedBlas = chunk1->displacedBlas;
                    cc.displacedFaceDataBuffer = chunk1->displacedFaceDataBuffer;
                    cc.hasDisplaced = chunk1->displacedBlas && chunk1->displacedFaceDataBuffer;
                    cc.vertexFormat = chunk1->vertexFormat;
                }

                // Distance cull
                float cx = static_cast<float>(static_cast<double>(cc.x) + 8.0 - cameraPos.x);
                float cy = static_cast<float>(static_cast<double>(cc.y) + 8.0 - cameraPos.y);
                float cz = static_cast<float>(static_cast<double>(cc.z) + 8.0 - cameraPos.z);
                float dist2 = cx * cx + cy * cy + cz * cz;
                if (dist2 > cullDist2) continue;

                local.visible.push_back(static_cast<int>(i));
            }
        });

        // Drain thread-local GC items (single-threaded — gc().collect() is not thread-safe)
        for (auto &local : locals) {
            for (auto &item : local.gcItems) {
                framework->gc().collect(std::move(item));
            }
            local.gcItems.clear();
        }

        // Flatten visible chunks from all threads
        std::vector<int> allVisible;
        for (auto &local : locals) {
            allVisible.insert(allVisible.end(), local.visible.begin(), local.visible.end());
        }

        // Prefix sums: compute exact offsets for every visible chunk (sequential, ~50µs)
        uint32_t chunkVisCount = static_cast<uint32_t>(allVisible.size());
        std::vector<uint32_t> prefixInst(chunkVisCount);   // instance index
        std::vector<uint32_t> prefixGeo(chunkVisCount);    // geometry data offset
        std::vector<uint32_t> prefixSbt(chunkVisCount);    // SBT group offset
        uint32_t totalChunkInst = 0, totalChunkGeo = 0, totalChunkSbt = 0;

        for (uint32_t vi = 0; vi < chunkVisCount; vi++) {
            auto &cc = cachedChunks_[allVisible[vi]];
            prefixInst[vi] = totalChunkInst;
            prefixGeo[vi] = totalChunkGeo;
            prefixSbt[vi] = totalChunkSbt;
            uint32_t instForChunk = 1 + (cc.hasDisplaced ? 1 : 0);
            totalChunkInst += instForChunk;
            totalChunkGeo += cc.geometryCount + (cc.hasDisplaced ? 1 : 0); // +1 for displaced geo
            totalChunkSbt += cc.geometryCount + instForChunk; // +instForChunk for SHADOW entries
        }

        // Pre-allocate all arrays to exact sizes (entity prefix already accumulated)
        uint32_t chunkInstBase = blasIndex;
        chunkInstBase_ = chunkInstBase; // store for PTLAS partition assignment
        uint32_t chunkGeoBase = static_cast<uint32_t>(vertexBufferAddrs.size());
        uint32_t chunkSbtBase = blasGroupAccu;
        uint32_t chunkBlasBase = blasAccu;

        instanceBuilder.instances.resize(chunkInstBase + totalChunkInst);
        currBlasSnapshot.blases.resize(chunkInstBase + totalChunkInst);
        currBlasSnapshot.generations.resize(chunkInstBase + totalChunkInst);
        currBlasSnapshot.vertexBuffers.resize(chunkInstBase + totalChunkInst);
        currBlasSnapshot.indexBuffers.resize(chunkInstBase + totalChunkInst);
        geometryTypes.resize(chunkSbtBase + totalChunkSbt);
        vertexBufferAddrs.resize(chunkGeoBase + totalChunkGeo);
        indexBufferAddrs.resize(chunkGeoBase + totalChunkGeo);
        lastVertexBufferAddrs.resize(chunkGeoBase + totalChunkGeo, 0);
        lastIndexBufferAddrs.resize(chunkGeoBase + totalChunkGeo, 0);
        lastObjToWorldMats.resize(chunkInstBase + totalChunkInst);
        blasOffset.resize(chunkInstBase + totalChunkInst);
        biomeColors.resize(chunkInstBase + totalChunkInst);

        // Parallel fill: each thread writes to pre-computed offsets (zero contention)
        Renderer::threadPool.parallelFor(numThreads, [&](uint32_t t) {
            uint32_t start = t * chunkVisCount / numThreads;
            uint32_t end = (t + 1) * chunkVisCount / numThreads;

            for (uint32_t vi = start; vi < end; vi++) {
                auto &cc = cachedChunks_[allVisible[vi]];
                uint32_t instIdx = chunkInstBase + prefixInst[vi];
                uint32_t geoIdx = chunkGeoBase + prefixGeo[vi];
                uint32_t sbtIdx = chunkSbtBase + prefixSbt[vi];
                // blasAccu for this chunk = chunkBlasBase + prefixGeo[vi] (geometry data is 1:1 with blasAccu)
                uint32_t myBlasAccu = chunkBlasBase + prefixGeo[vi];

                VkTransformMatrixKHR transform = {
                    1, 0, 0, static_cast<float>(static_cast<double>(cc.x) - cameraPos.x),
                    0, 1, 0, static_cast<float>(static_cast<double>(cc.y) - cameraPos.y),
                    0, 0, 1, static_cast<float>(static_cast<double>(cc.z) - cameraPos.z),
                };

                // Instance
                instanceBuilder.instances[instIdx] = std::make_tuple(
                    transform, instIdx, uint32_t(0x01), sbtIdx,
                    VkGeometryInstanceFlagsKHR(0), cc.blas);
                currBlasSnapshot.blases[instIdx] = cc.blas;
                currBlasSnapshot.generations[instIdx] = cc.blasGeneration;
                currBlasSnapshot.vertexBuffers[instIdx] = cc.vertexBuffers;
                currBlasSnapshot.indexBuffers[instIdx] = cc.indexBuffers;

                // SBT geometry types: SHADOW + chunk types
                geometryTypes[sbtIdx] = World::GeometryTypes::SHADOW;
                for (uint32_t g = 0; g < cc.geometryCount; g++) {
                    geometryTypes[sbtIdx + 1 + g] = cc.geoTypes[g + 1]; // skip cached SHADOW
                }

                // Buffer addresses
                for (uint32_t g = 0; g < cc.geometryCount; g++) {
                    vertexBufferAddrs[geoIdx + g] = cc.vertBufAddrs[g];
                    indexBufferAddrs[geoIdx + g] = cc.idxBufAddrs[g];
                }

                // Per-instance metadata
                glm::mat4 mat = glm::transpose(glm::mat4(
                    glm::vec4(1, 0, 0, static_cast<float>(static_cast<double>(cc.x) - cameraPos.x)),
                    glm::vec4(0, 1, 0, static_cast<float>(static_cast<double>(cc.y) - cameraPos.y)),
                    glm::vec4(0, 0, 1, static_cast<float>(static_cast<double>(cc.z) - cameraPos.z)),
                    glm::vec4(0, 0, 0, 1)));
                lastObjToWorldMats[instIdx] = mat;
                // Encode vertex format in upper 2 bits of blasOffset (shader extracts via >> 30)
                blasOffset[instIdx] = myBlasAccu | (static_cast<uint32_t>(cc.vertexFormat) << 30);
                biomeColors[instIdx] = glm::uvec4(cc.biomeGrassColor, cc.biomeFoliageColor, cc.biomeWaterColor, 0);

                // DDA displacement instance
                if (cc.hasDisplaced) {
                    uint32_t dInstIdx = instIdx + 1;
                    uint32_t dGeoIdx = geoIdx + cc.geometryCount;
                    uint32_t dSbtIdx = sbtIdx + cc.geometryCount + 1;
                    uint32_t dBlasAccu = myBlasAccu + cc.geometryCount;

                    instanceBuilder.instances[dInstIdx] = std::make_tuple(
                        transform, dInstIdx, uint32_t(0x01), dSbtIdx,
                        VkGeometryInstanceFlagsKHR(0), cc.displacedBlas);
                    currBlasSnapshot.blases[dInstIdx] = cc.displacedBlas;
                    currBlasSnapshot.generations[dInstIdx] = cc.blasGeneration;
                    geometryTypes[dSbtIdx] = World::GeometryTypes::WORLD_DISPLACED;
                    vertexBufferAddrs[dGeoIdx] = cc.displacedFaceDataBuffer->bufferAddress();
                    indexBufferAddrs[dGeoIdx] = 0;
                    lastObjToWorldMats[dInstIdx] = mat;
                    blasOffset[dInstIdx] = dBlasAccu;
                    biomeColors[dInstIdx] = glm::uvec4(cc.biomeGrassColor, cc.biomeFoliageColor, cc.biomeWaterColor, 0);
                }
            }
        });

        // Update accumulators after chunk instance population
        blasIndex += totalChunkInst;
        blasAccu += totalChunkGeo;
        blasGroupAccu += totalChunkSbt;
    }

    // Area light gathering: collect from loaded chunks, cull by vertical range + distance
    // NOTE: No frustum culling — lights behind the camera still contribute via bounced
    // illumination and removing them causes visible pop-in when rotating the camera.
    // Vertical culling is safe because deep underground lights are behind solid rock.
    {
        constexpr int MAX_AREA_LIGHTS = 512;
        constexpr float VERTICAL_CULL_BELOW = 48.0f; // skip lights more than N blocks below camera
        struct LightWithDist {
            vk::Data::AreaLight light;
            float dist2;
            float contribution; // effectiveIntensity / max(dist2, 1.0)
        };
        std::vector<LightWithDist> gatheredLights;

        auto &chunk1s = chunks->chunks();
        for (int i = 0; i < chunk1s.size(); i++) {
            auto &chunk1 = chunk1s[i];
            if (chunk1->blas == nullptr) continue;
            if (chunk1->lightSources.empty()) continue;

            // Chunk-level vertical early out: skip entire chunk if too far below camera
            float chunkTopY = static_cast<float>(static_cast<double>(chunk1->y) + 16.0 - cameraPos.y);
            if (-chunkTopY > VERTICAL_CULL_BELOW) continue;

            for (auto &src : chunk1->lightSources) {
                if (src.lightTypeId < 0 || src.lightTypeId >= LIGHT_TYPE_COUNT) continue;
                auto &def = LIGHT_DEFS[src.lightTypeId];

                // Camera-relative position
                float rx = static_cast<float>(static_cast<double>(src.worldX) - cameraPos.x);
                float ry = static_cast<float>(static_cast<double>(src.worldY) - cameraPos.y);
                float rz = static_cast<float>(static_cast<double>(src.worldZ) - cameraPos.z);

                // Per-light vertical cull: skip lights too far below camera (sealed caves)
                if (-ry > VERTICAL_CULL_BELOW) continue;

                float d2 = rx * rx + ry * ry + rz * rz;

                float perBlock = Renderer::options.perBlockIntensity[src.lightTypeId];
                if (perBlock < 0.001f) continue;  // Skip disabled lights

                // CPU-side distance cull: skip lights beyond their defined radius
                float effectiveIntensity = def.lumens * LUMENS_TO_INTENSITY * Renderer::options.areaLightIntensity * perBlock;
                float maxRange = Renderer::options.areaLightRange;
                if (d2 > maxRange * maxRange) continue;

                float contribution = effectiveIntensity / std::max(d2, 1.0f);

                vk::Data::AreaLight al{};
                auto &opts = Renderer::options;
                int tid = src.lightTypeId;
                al.position = glm::vec3(rx, ry + def.yOffset + opts.perBlockYOffset[tid], rz);
                al.halfExtent = def.halfExtent * opts.perBlockScale[tid];
                if (opts.perBlockColorR[tid] >= 0) {
                    // Per-block override: user BT.709 color -> BT.2020
                    glm::vec3 userColor(opts.perBlockColorR[tid], opts.perBlockColorG[tid], opts.perBlockColorB[tid]);
                    al.color = colorspace::BT709_TO_BT2020 * userColor;
                } else {
                    // Blackbody (XYZ -> BT.2020) or spectral uplift (BT.709 -> BT.2020 + chroma boost)
                    // Per-block temperature override from UI sliders (0 = use LIGHT_DEFS default)
                    float colorTemp = def.colorTemperature;
                    if (opts.perBlockTemperatureK[tid] > 0) {
                        colorTemp = opts.perBlockTemperatureK[tid];
                    }
                    al.color = colorspace::computeEmissionColor(colorTemp, def.color, def.spectralPurity);
                }
                al.intensity = effectiveIntensity;
                al.radius = maxRange;

                // Stable ID for cross-frame light tracking (ReSTIR DI)
                uint32_t bx = static_cast<uint32_t>(static_cast<int>(src.worldX)) & 0xFFFF;
                uint32_t by = static_cast<uint32_t>(static_cast<int>(src.worldY)) & 0xFFFF;
                uint32_t bz = static_cast<uint32_t>(static_cast<int>(src.worldZ)) & 0xFFFF;
                uint32_t stableId = (bx | (by << 16)) ^ (bz * 2654435761u);
                std::memcpy(&al._unused.x, &stableId, sizeof(float));
                al._unused.y = LIGHT_DEFS[src.lightTypeId].flickerStrength;

                gatheredLights.push_back({al, d2, contribution});
            }
        }

        // IMPORTANT: Do NOT unlock the chunks mutex here. Although chunk data has
        // been read into local vectors, the underlying GPU resources (BLASes, vertex
        // buffers) referenced by those addresses must remain alive through TLAS build
        // and RT dispatch. Unlocking here allows the chunk build thread to free BLASes
        // while the render thread still references them, causing DEVICE_LOST.
        // The lock is released automatically at scope exit via RAII.

        // Sort by contribution (brightest/nearest first)
        std::sort(gatheredLights.begin(), gatheredLights.end(),
                  [](const LightWithDist &a, const LightWithDist &b) { return a.contribution > b.contribution; });

        // Clamp to max
        if (gatheredLights.size() > MAX_AREA_LIGHTS) {
            gatheredLights.resize(MAX_AREA_LIGHTS);
        }

        // Use areaLightRange directly — contribution sort already prioritizes nearby lights.
        // Frostbite windowing (1-(d/R)^2)^2 handles smooth falloff at boundary.
        for (auto &lwd : gatheredLights) {
            lwd.light.radius = Renderer::options.areaLightRange;
        }

        areaLightCount = static_cast<int>(gatheredLights.size());

        // Pre-allocate for max lights on first use; reuse on subsequent frames
        constexpr size_t AREA_LIGHT_BUFFER_CAPACITY = MAX_AREA_LIGHTS;
        if (!areaLightBuffer) {
            areaLightBuffer = vk::DeviceLocalBuffer::create(
                vma, device, true, AREA_LIGHT_BUFFER_CAPACITY * sizeof(vk::Data::AreaLight),
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        }

        if (areaLightCount > 0) {
            std::vector<vk::Data::AreaLight> lightData;
            lightData.reserve(areaLightCount);
            for (auto &lwd : gatheredLights) {
                lightData.push_back(lwd.light);
            }

            areaLightBuffer->uploadToStagingBuffer(lightData.data(),
                                                    lightData.size() * sizeof(vk::Data::AreaLight), 0);
        } else {
            // Upload a dummy light so the descriptor binding remains valid
            vk::Data::AreaLight dummy{};
            areaLightBuffer->uploadToStagingBuffer(&dummy, sizeof(vk::Data::AreaLight), 0);
        }
    }

    cpuAccInstances += cpuMsSince(cpuT6);

    if (instanceBuilder.instances.empty()) {
        tlas = nullptr;
        ptlas = nullptr;
        prevTlasInstanceCount_ = 0;
        return;
    }

    // PTLAS: disabled by default pending driver investigation (595.97 AS Build fault).
    // Standard TLAS UPDATE path is stable. PTLAS can be enabled via Options when ready.
    // if (!usePTLAS_ && device->hasPTLAS()) {
    //     usePTLAS_ = true;
    //     std::cout << "[PTLAS] Enabled — using partitioned TLAS for incremental updates" << std::endl;
    // }

    // Log TLAS instance count and build mode for profiling (every 120 frames ~ 1/sec at 120fps)
    static uint32_t tlasLogCounter = 0;
    static uint32_t tlasUpdateCount = 0;
    static uint32_t tlasBuildCount = 0;

    auto cpuT7 = Clock::now();
    currBlasSnapshot.instanceCount = static_cast<uint32_t>(instanceBuilder.instances.size());
    uint32_t currentInstanceCount = currBlasSnapshot.instanceCount;

    if (usePTLAS_) {
        // PTLAS path: convert instances and build partitioned TLAS
        // Recreate if capacity exceeded or first frame
        if (!ptlas || ptlas->needsRecreate(currentInstanceCount)) {
            vk::PartitionedTLAS::Config ptlasConfig{};
            ptlasConfig.maxInstances = std::max(currentInstanceCount + 4096, 40000u);
            ptlasConfig.maxPartitions = 300;
            ptlasConfig.maxInstancePerPartition = 512;
            ptlasConfig.maxGlobalInstances = 2048;
            ptlas = vk::PartitionedTLAS::create(device, physicalDevice, vma, ptlasConfig);
        }

        // Convert TLASBuilder instances to PTLAS write data
        std::vector<VkPartitionedAccelerationStructureWriteInstanceDataNV> ptlasInstances(currentInstanceCount);
        uint32_t entityInstCount = static_cast<uint32_t>(chunkInstBase_);

        for (uint32_t i = 0; i < currentInstanceCount; i++) {
            auto &[transform, customIndex, mask, sbtOffset, flags, blas] = instanceBuilder.instances[i];
            auto &wd = ptlasInstances[i];
            wd.transform = transform;
            std::memset(wd.explicitAABB, 0, sizeof(wd.explicitAABB));
            wd.instanceID = customIndex;
            wd.instanceMask = mask;
            wd.instanceContributionToHitGroupIndex = sbtOffset;
            wd.instanceFlags = static_cast<VkPartitionedAccelerationStructureInstanceFlagsNV>(flags);
            wd.instanceIndex = i;
            wd.accelerationStructure = blas->blasDeviceAddress();

            // Partition assignment: entities → global, chunks → spatial hash
            if (i < entityInstCount) {
                wd.partitionIndex = VK_PARTITIONED_ACCELERATION_STRUCTURE_PARTITION_INDEX_GLOBAL_NV;
            } else {
                // Look up chunk coordinates from cachedChunks_ via the visible chunk index
                // Use a spatial hash to distribute chunks across partitions
                uint32_t chunkLocalIdx = i - entityInstCount;
                // Compute partition from the instance transform translation (camera-relative chunk position)
                // Use the transform directly — it encodes the spatial position
                int gx = static_cast<int>(std::floor(transform.matrix[0][3])) >> 6;
                int gz = static_cast<int>(std::floor(transform.matrix[2][3])) >> 6;
                uint32_t h = static_cast<uint32_t>(gx * 73856093) ^ static_cast<uint32_t>(gz * 19349663);
                wd.partitionIndex = 1 + (h % (ptlas->config().maxPartitions - 1));
            }
        }

        GpuDiag::checkpoint(worldCommandBuffer->vkCommandBuffer(), GpuDiag::TLAS_BUILD);
        ptlas->buildAllInstances(ptlasInstances, worldCommandBuffer);
        tlasUpdateCount++; // PTLAS builds are always incremental

    } else {
        // Standard TLAS path (non-Blackwell GPUs)
        // TLAS UPDATE only requires same instance count. Per Vulkan spec, instances can have
        // different BLAS references / transforms — the driver refits the BVH.
        bool canUpdate = tlas != nullptr
            && currBlasSnapshot.instanceCount == prevBlasSnapshot_.instanceCount;

        constexpr VkBuildAccelerationStructureFlagsKHR tlasFlags =
            VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
            VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;

        if (canUpdate) {
            instanceBuilder.endInstanceBuilder(device, vma);
            tlasBuilder->defineBuildProperty(tlasFlags);
            GpuDiag::checkpoint(worldCommandBuffer->vkCommandBuffer(), GpuDiag::TLAS_UPDATE);
            tlasBuilder->updateAndSubmit(tlas, tlasScratchBuffer_, worldCommandBuffer);
            tlasUpdateCount++;
        } else {
            instanceBuilder.endInstanceBuilder(device, vma);
            tlasBuilder->defineBuildProperty(tlasFlags);
            tlasBuilder->querySizeInfo(device);
            tlasBuilder->allocateBuffers(physicalDevice, device, vma);
            GpuDiag::checkpoint(worldCommandBuffer->vkCommandBuffer(), GpuDiag::TLAS_BUILD);
            tlas = tlasBuilder->buildAndSubmit(device, worldCommandBuffer);

            VkAccelerationStructureBuildGeometryInfoKHR sizeQueryInfo{};
            sizeQueryInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
            sizeQueryInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
            sizeQueryInfo.flags = tlasFlags;

            VkAccelerationStructureBuildSizesInfoKHR buildSizeInfo{};
            buildSizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
            sizeQueryInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
            vkGetAccelerationStructureBuildSizesKHR(device->vkDevice(),
                VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                &sizeQueryInfo, &currentInstanceCount, &buildSizeInfo);

            VkAccelerationStructureBuildSizesInfoKHR updateSizeInfo{};
            updateSizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
            sizeQueryInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;
            vkGetAccelerationStructureBuildSizesKHR(device->vkDevice(),
                VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                &sizeQueryInfo, &currentInstanceCount, &updateSizeInfo);

            VkDeviceSize requiredScratchSize = std::max(buildSizeInfo.buildScratchSize,
                                                         updateSizeInfo.updateScratchSize);

            if (!tlasScratchBuffer_ || tlasScratchSize_ < requiredScratchSize) {
                tlasScratchSize_ = requiredScratchSize;
                tlasScratchBuffer_ = vk::DeviceLocalBuffer::create(
                    vma, device, false, tlasScratchSize_,
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                    0, VMA_MEMORY_USAGE_GPU_ONLY,
                    physicalDevice->accelerationStructProperties().minAccelerationStructureScratchOffsetAlignment);
            }
            tlasBuildCount++;
        }
    }

    cpuAccTlas += cpuMsSince(cpuT7);
    cpuAccTotal += cpuMsSince(cpuT0);
    cpuFrameCount++;

    // Save BLAS snapshot: keeps shared_ptrs alive until this context is reused,
    // preventing GC from freeing BLASes while the GPU still references their addresses.
    prevBlasSnapshot_ = std::move(currBlasSnapshot);
    prevTlasInstanceCount_ = currentInstanceCount;

    // Save entity batch references: Entities::build() replaces the global entityBatch_
    // every frame, freeing entity vertex/index buffers. Without this, in-flight GPU
    // contexts read BDA addresses pointing to freed buffers → READ_AFTER_DESTROY.
    prevEntityBatch_ = entities->entityBatch();
    prevEntityPostBatch_ = entities->entityPostBatch();

    if (++tlasLogCounter >= 120) {
        float n = static_cast<float>(cpuFrameCount);
        std::cout << "[Profiler] TLAS instances: " << currentInstanceCount
                  << "  builds: " << tlasBuildCount << "  updates: " << tlasUpdateCount << std::endl;
        if (cpuFrameCount > 0) {
            std::cout << "[CPU] WorldPrepare avg(ms): check=" << (cpuAccCheck / n)
                      << " schedule=" << (cpuAccSchedule / n)
                      << " important=" << (cpuAccImportant / n)
                      << " entity=" << (cpuAccEntity / n)
                      << " instances=" << (cpuAccInstances / n)
                      << " tlas=" << (cpuAccTlas / n)
                      << " TOTAL=" << (cpuAccTotal / n) << std::endl;
            // Also write to file since C++ stdout may not reach Minecraft log
            std::ofstream cpuLog("C:/RadSER/results/cpu_timing.log", std::ios::app);
            if (cpuLog.is_open()) {
                cpuLog << "[CPU] instances=" << currentInstanceCount
                       << " check=" << (cpuAccCheck / n)
                       << " schedule=" << (cpuAccSchedule / n)
                       << " important=" << (cpuAccImportant / n)
                       << " entity=" << (cpuAccEntity / n)
                       << " instances_pop=" << (cpuAccInstances / n)
                       << " tlas=" << (cpuAccTlas / n)
                       << " TOTAL=" << (cpuAccTotal / n)
                       << " builds=" << tlasBuildCount
                       << " updates=" << tlasUpdateCount
                       << " threads=" << Renderer::threadPool.threadCount()
                       << "\n";
            }
        }
        // Pool utilization stats
        if (Renderer::vertexPool && Renderer::indexPool && Renderer::blasPool) {
            auto vs = Renderer::vertexPool->getStats();
            auto is = Renderer::indexPool->getStats();
            auto bs = Renderer::blasPool->getStats();
            uint32_t totalBlocks = vs.blockCount + is.blockCount + bs.blockCount;
            VkDeviceSize totalUsedMB = (vs.totalUsed + is.totalUsed + bs.totalUsed) / (1024 * 1024);
            VkDeviceSize totalCapMB = (vs.totalCapacity + is.totalCapacity + bs.totalCapacity) / (1024 * 1024);
            std::cout << "[Pool] blocks: vtx=" << vs.blockCount << " idx=" << is.blockCount
                      << " blas=" << bs.blockCount << "  total=" << totalBlocks
                      << "  used=" << totalUsedMB << "/" << totalCapMB << " MB" << std::endl;
        }

        tlasLogCounter = 0;
        tlasUpdateCount = 0;
        tlasBuildCount = 0;
        cpuAccCheck = cpuAccSchedule = cpuAccImportant = 0;
        cpuAccEntity = cpuAccInstances = cpuAccTlas = cpuAccTotal = 0;
        cpuFrameCount = 0;
    }

    worldCommandBuffer->barriersMemory({vk::CommandBuffer::MemoryBarrier{
        .srcStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        .srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
        .dstStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
        .dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR,
    }});

    auto rtModuleCtx = rayTracingModuleContext.lock();
    if (!rtModuleCtx) return;
    rtModuleCtx->sbt->setupHitSBT(geometryTypes);
    if (rtModuleCtx->sharcUpdateSbt) {
        rtModuleCtx->sharcUpdateSbt->setupHitSBT(geometryTypes);
    }

    uploadBuffer(blasOffset, vertexBufferAddrs, indexBufferAddrs, lastVertexBufferAddrs, lastIndexBufferAddrs,
                 lastObjToWorldMats, biomeColors);
}