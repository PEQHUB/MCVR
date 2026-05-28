#include "core/render/modules/world/ray_tracing/submodules/world_prepare.hpp"
#include "core/render/buffers.hpp"
#include "core/render/crash_ring_buffer.hpp"
#include "core/render/gpu_diagnostics.hpp"
#include "core/render/chunks.hpp"
#include "core/render/entities.hpp"
#include "core/render/radiance_logger.hpp"
#include "core/render/colorspace.hpp"
#include "core/render/lights.hpp"
#include "core/render/modules/world/ray_tracing/ray_tracing_module.hpp"
#include "core/render/render_framework.hpp"
#include "core/render/renderer.hpp"
#include "core/render/world.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <thread>
#include <unordered_map>
#include <glm/gtc/type_ptr.hpp>

namespace {
class ScopedGpuProfile {
public:
    ScopedGpuProfile(VkCommandBuffer cmd, const char* name) : cmd_(cmd) {
        if (cmd_ != VK_NULL_HANDLE && Renderer::gpuProfiler.isEnabled()) {
            Renderer::gpuProfiler.beginModule(cmd_, name);
            open_ = true;
        }
    }

    ~ScopedGpuProfile() {
        close();
    }

    void close() {
        if (open_) {
            Renderer::gpuProfiler.endModule(cmd_);
            open_ = false;
        }
    }

private:
    VkCommandBuffer cmd_ = VK_NULL_HANDLE;
    bool open_ = false;
};
} // namespace

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
    VkCommandBuffer profileCmd = (Renderer::gpuProfiler.isEnabled() && worldCommandBuffer)
        ? worldCommandBuffer->vkCommandBuffer()
        : VK_NULL_HANDLE;

    auto chunks = Renderer::instance().world()->chunks();
    auto entities = Renderer::instance().world()->entities();
    auto cameraPos = Renderer::instance().world()->getCameraPos();

    GpuDiag::checkpoint(worldCommandBuffer->vkCommandBuffer(), GpuDiag::WORLD_PREPARE_BEGIN);
    g_crashRing.record("WP:enter");

    // CPU profiling: accumulate sub-phase timing (logged every 120 frames with TLAS stats)
    using Clock = std::chrono::steady_clock;
    static float cpuAccCheck = 0, cpuAccSchedule = 0, cpuAccImportant = 0;
    static float cpuAccEntity = 0, cpuAccInstances = 0, cpuAccTlas = 0, cpuAccTotal = 0;
    static uint32_t cpuFrameCount = 0;
    // Per-frame CSV accumulators (reset each frame when perFrameTiming is on)
    static float pfCheck = 0, pfImportant = 0, pfEntity = 0;
    static float pfInstances = 0, pfTlas = 0;
    auto cpuT0 = Clock::now();
    auto cpuMsSince = [](Clock::time_point t0) {
        return std::chrono::duration<float, std::milli>(Clock::now() - t0).count();
    };

    g_crashRing.record("WP:lock");
    auto lockStart = Clock::now();
    std::unique_lock<std::recursive_mutex> lock(chunks->mutex(), std::defer_lock);
    while (!lock.try_lock()) {
        if (std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - lockStart).count() >= 250) {
            renderDiag("WP chunk mutex wait timed out after %.3f ms; skipping RT frame",
                       cpuMsSince(lockStart));
            g_crashRing.record("WP:lockTimeout");
            tlas = nullptr;
            prevBlasSnapshot_ = {};
            prevTlasInstanceCount_ = 0;
            return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    float lockWaitMs = cpuMsSince(lockStart);
    if (lockWaitMs > 2.0f) {
        renderDiag("WP chunk mutex waited %.3f ms", lockWaitMs);
    }
    g_crashRing.record("WP:lockOk");

    auto cpuT1 = Clock::now();
    if (chunks->chunkBuildScheduler() != nullptr) {
        chunks->chunkBuildScheduler()->updateCameraPos(cameraPos);
        chunks->chunkBuildScheduler()->integrateCompleted();
        cpuAccCheck += cpuMsSince(cpuT1);
        pfCheck += cpuMsSince(cpuT1);
    }

    auto cpuT3 = Clock::now();
    GpuDiag::checkpoint(worldCommandBuffer->vkCommandBuffer(), GpuDiag::CHUNK_SCHEDULE_DONE);
    g_crashRing.record("WP:schedulerDone");

    // Barrier: ensure vertex/index buffer TRANSFER writes from uploadCommandBuffer
    // are visible before BLAS builds read them. Without this, the GPU may speculatively
    // start BLAS builds while staging→device copies are still in flight.
    {
        ScopedGpuProfile profile(profileCmd, "RT.WP.InputBarrier");
        worldCommandBuffer->barriersMemory({vk::CommandBuffer::MemoryBarrier{
            .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
            .dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR |
                             VK_ACCESS_2_SHADER_READ_BIT,
        }});
    }

    auto cpuT4 = Clock::now();
    if (chunks->importantBLASBuilders().size() > 0) {
        ScopedGpuProfile profile(profileCmd, "RT.WP.ImportantBLAS");
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
    pfImportant += cpuMsSince(cpuT4);

    auto cpuT5 = Clock::now();
    if (entities->blasBatchBuilder() != nullptr) {
        ScopedGpuProfile profile(profileCmd, "RT.WP.EntityBLAS");
        GpuDiag::checkpoint(worldCommandBuffer->vkCommandBuffer(), GpuDiag::BLAS_BUILD_ENTITY);
        entities->blasBatchBuilder()->submit(worldCommandBuffer);
    }
    cpuAccEntity += cpuMsSince(cpuT5);
    pfEntity += cpuMsSince(cpuT5);

    {
        ScopedGpuProfile profile(profileCmd, "RT.WP.BLASBuildBarrier");
        worldCommandBuffer->barriersMemory({vk::CommandBuffer::MemoryBarrier{
            .srcStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
            .srcAccessMask =
                VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
            .dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
            .dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR,
        }});
    }

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

                    if (!entities1[i]->blas) continue; // entity BLAS not yet built
                    instanceBuilder.defineInstance(transform, blasIndex, entities1[i]->rtFlag, blasGroupAccu, flags,
                                                   entities1[i]->blas);
		} else {
			// Prebuilt BLAS not implemented — skip this entity
			RadianceLogger::log("world_prepare", "WARN", "Skipping entity with prebuiltBLAS=%d", entities1[i]->prebuiltBLAS);
			continue;
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

    // BLAS lifetime snapshot: keep shared_ptrs alive while GPU references the TLAS.
    // prevBlasSnapshot_ from the last use of this swapchain context is safe to release
    // (GPU finished with it before we re-acquired this context).
    TlasBlasSnapshot currBlasSnapshot;

    // Chunk — cached + parallel instance population
    {
        auto &chunk1s = chunks->chunks();
        float cullDist = Renderer::options.chunkCullDistance;
        float cullDist2 = cullDist * cullDist;
        float mergeDist = Renderer::options.megaMergeDistance;
        float mergeDist2 = mergeDist * mergeDist;
        bool megaEnabled = mergeDist > 0 && mergeDist < cullDist;

        constexpr int MEGA_SIZE = 64;
        auto megaKey = [](int x, int y, int z) -> int64_t {
            int mx = (x >= 0) ? x / MEGA_SIZE : (x - MEGA_SIZE + 1) / MEGA_SIZE;
            int my = (y >= 0) ? y / MEGA_SIZE : (y - MEGA_SIZE + 1) / MEGA_SIZE;
            int mz = (z >= 0) ? z / MEGA_SIZE : (z - MEGA_SIZE + 1) / MEGA_SIZE;
            return (int64_t(mx) & 0xFFFFF) | ((int64_t(my) & 0xFFFFF) << 20) | ((int64_t(mz) & 0xFFFFF) << 40);
        };
        std::unordered_map<int64_t, std::vector<int>> farChunksByMega;

        // Update cache: only recompute metadata when blasGeneration changes.
        // On render-distance change the chunk grid is rebuilt — clear the entire cache
        // to prevent stale BLAS/buffer pointers from surviving across the reset.
        if (cachedChunks_.size() != chunk1s.size()) {
            cachedChunks_.clear();
            cachedChunks_.resize(chunk1s.size());
            // Force full TLAS BUILD (stale snapshot would falsely match new generations)
            prevBlasSnapshot_ = {};
            prevTlasInstanceCount_ = 0;
            tlas = nullptr;
        }

        // Deterministic render-thread pass: update cache + distance cull.
        // This used to use Renderer::threadPool.parallelFor(), but crash logs proved
        // a hang at WP:chunkPass1. Keep WorldPrepare render-critical work local.
        uint32_t numChunks = static_cast<uint32_t>(chunk1s.size());
        std::vector<int> allVisible;
        allVisible.reserve(numChunks);
        uint64_t textureGeneration = Renderer::textureSystem.generation();
        uint32_t staleTextureChunks = 0;
        bool scan25Recorded = numChunks < 4;
        bool scan50Recorded = numChunks < 4;
        bool scan75Recorded = numChunks < 4;

        g_crashRing.record("WP:chunkPass1");
        for (uint32_t i = 0; i < numChunks; i++) {
            if (!scan25Recorded && i >= numChunks / 4) {
                g_crashRing.record("WP:scan25");
                scan25Recorded = true;
            }
            if (!scan50Recorded && i >= numChunks / 2) {
                g_crashRing.record("WP:scan50");
                scan50Recorded = true;
            }
            if (!scan75Recorded && i >= (numChunks * 3) / 4) {
                g_crashRing.record("WP:scan75");
                scan75Recorded = true;
            }
            auto &chunk1 = chunk1s[i];
            if (!chunk1 || chunk1->blas == nullptr) continue;
            if (!chunk1->geometryTypes || !chunk1->vertexBuffers || !chunk1->indexBuffers) continue;
            if (chunk1->geometryCount > chunk1->geometryTypes->size() ||
                chunk1->geometryCount > chunk1->vertexBuffers->size() ||
                chunk1->geometryCount > chunk1->indexBuffers->size()) {
                continue;
            }
            if (textureGeneration != 0 && chunk1->textureGeneration != textureGeneration) {
                staleTextureChunks++;
            }

            // Update cache if generation changed.
            auto &cc = cachedChunks_[i];
            if (cc.blasGeneration != chunk1->blasGeneration) {
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
                cc.vertexFormat = chunk1->vertexFormat;
            }

            // Distance cull
            float cx = static_cast<float>(static_cast<double>(cc.x) + 8.0 - cameraPos.x);
            float cy = static_cast<float>(static_cast<double>(cc.y) + 8.0 - cameraPos.y);
            float cz = static_cast<float>(static_cast<double>(cc.z) + 8.0 - cameraPos.z);
            float dist2 = cx * cx + cy * cy + cz * cz;
            if (dist2 > cullDist2) continue;

            if (megaEnabled && dist2 >= mergeDist2) {
                farChunksByMega[megaKey(cc.x, cc.y, cc.z)].push_back(static_cast<int>(i));
            } else {
                allVisible.push_back(static_cast<int>(i));
            }
        }
        if (staleTextureChunks > 0) {
            static uint32_t staleTextureLogCounter = 0;
            staleTextureLogCounter++;
            if (staleTextureLogCounter <= 5 || staleTextureLogCounter % 120 == 0) {
                renderDiag("WP accepted stale-texture chunks count=%u currentGen=%llu",
                           staleTextureChunks,
                           static_cast<unsigned long long>(textureGeneration));
            }
        }
        g_crashRing.record("WP:chunkPass1Done");

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
            uint32_t instForChunk = 1;
            totalChunkInst += instForChunk;
            totalChunkGeo += cc.geometryCount;
            totalChunkSbt += cc.geometryCount + instForChunk; // +instForChunk for SHADOW entries
        }

        // Pre-allocate all arrays to exact sizes (entity prefix already accumulated)
        uint32_t chunkInstBase = blasIndex;
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

        // Sequential fill: avoids render-thread dependency on the global worker pool.
        g_crashRing.record("WP:chunkFill");
        for (uint32_t vi = 0; vi < chunkVisCount; vi++) {
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

        }
        g_crashRing.record("WP:chunkFillDone");

        // Update accumulators for mega-BLAS pass that follows
        blasIndex += totalChunkInst;
        blasAccu += totalChunkGeo;
        blasGroupAccu += totalChunkSbt;

        // Mega-BLAS: build merged BLASes for far chunk groups
        // Each mega-chunk group gets a single TLAS instance with all sub-chunk geometries.
        // Per-geometry transforms offset each sub-chunk within the mega-BLAS.
        if (megaEnabled && !farChunksByMega.empty()) {
            std::unordered_map<int64_t, std::shared_ptr<MegaChunk>> newMegaCache;
            static uint32_t megaBuildCount = 0, megaCacheHit = 0;

            for (auto &[mk, chunkIndices] : farChunksByMega) {
                // Content hash: detect when any constituent chunk changed
                uint64_t contentHash = 0;
                for (int idx : chunkIndices) {
                    uint64_t chunkHash = chunk1s[idx]->blasGeneration;
                    chunkHash ^= chunk1s[idx]->textureGeneration + 0x9e3779b97f4a7c15ull + (chunkHash << 6) + (chunkHash >> 2);
                    contentHash ^= chunkHash * 2654435761u + static_cast<uint64_t>(idx);
                }

                // Check cache
                auto cacheIt = megaChunkCache_.find(mk);
                if (cacheIt != megaChunkCache_.end() && cacheIt->second->contentHash == contentHash) {
                    newMegaCache[mk] = cacheIt->second;
                    megaCacheHit++;
                } else {
                    // Build new mega-BLAS
                    auto mega = std::make_shared<MegaChunk>();
                    mega->key = mk;
                    mega->contentHash = contentHash;

                    // Compute mega-chunk origin (aligned to MEGA_SIZE grid)
                    int ox = chunk1s[chunkIndices[0]]->x;
                    int oy = chunk1s[chunkIndices[0]]->y;
                    int oz = chunk1s[chunkIndices[0]]->z;
                    ox = (ox >= 0) ? (ox / MEGA_SIZE) * MEGA_SIZE : ((ox - MEGA_SIZE + 1) / MEGA_SIZE) * MEGA_SIZE;
                    oy = (oy >= 0) ? (oy / MEGA_SIZE) * MEGA_SIZE : ((oy - MEGA_SIZE + 1) / MEGA_SIZE) * MEGA_SIZE;
                    oz = (oz >= 0) ? (oz / MEGA_SIZE) * MEGA_SIZE : ((oz - MEGA_SIZE + 1) / MEGA_SIZE) * MEGA_SIZE;
                    mega->origin = glm::dvec3(ox, oy, oz);

                    auto megaBuilder = vk::BLASBuilder::create();
                    auto megaGeom = megaBuilder->beginGeometries();
                    uint32_t totalGeoms = 0;

                    for (int idx : chunkIndices) {
                        auto &chunk1 = chunk1s[idx];

                        // Per-sub-chunk transform: chunk_origin - mega_origin
                        VkTransformMatrixKHR subTransform = {
                            1, 0, 0, static_cast<float>(chunk1->x - ox),
                            0, 1, 0, static_cast<float>(chunk1->y - oy),
                            0, 0, 1, static_cast<float>(chunk1->z - oz),
                        };
                        auto transformBuf = vk::HostVisibleBuffer::create(
                            vma, device, sizeof(VkTransformMatrixKHR),
                            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                            16);
                        transformBuf->uploadToBuffer(&subTransform);
                        mega->transformBuffers.push_back(transformBuf);

                        for (uint32_t g = 0; g < chunk1->geometryCount; g++) {
                            bool isOpaque = (*chunk1->geometryTypes)[g] == World::WORLD_SOLID;
                            auto &vb = (*chunk1->vertexBuffers)[g];
                            auto &ib = (*chunk1->indexBuffers)[g];
                            uint32_t vertCount = static_cast<uint32_t>(vb->size() / sizeof(vk::VertexFormat::PBRTriangle));
                            uint32_t idxCount = static_cast<uint32_t>(ib->size() / sizeof(uint32_t));

                            megaGeom->defineTriangleGeometryWithTransform<vk::VertexFormat::PBRTriangle>(
                                vb, vertCount, ib, idxCount, isOpaque,
                                transformBuf->bufferAddress());

                            mega->geometryTypes.push_back((*chunk1->geometryTypes)[g]);
                            mega->vertexBuffers.push_back(vb);
                            mega->indexBuffers.push_back(ib);
                            totalGeoms++;
                        }
                    }
                    megaGeom->endGeometries();

                    // Build mega-BLAS on SECONDARY queue (same queue that uploaded vertex data).
                    // Building on the main queue crashes — cross-queue memory visibility issue.
                    if (!megaCmdBuffer_) {
                        megaCmdBuffer_ = vk::CommandBuffer::create(device, framework->asyncCommandPool());
                        megaFence_ = vk::Fence::create(device);
                    }

                    auto megaBuilt = megaBuilder
                        ->defineBuildProperty(VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR |
                                              VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR)
                        ->querySizeInfo(device)
                        ->allocateBuffers(physicalDevice, device, vma)
                        ->build(device);

                    megaCmdBuffer_->begin();
                    megaBuilder->submit(megaCmdBuffer_);
                    megaCmdBuffer_->end();

                    VkSubmitInfo si{};
                    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
                    si.commandBufferCount = 1;
                    si.pCommandBuffers = &megaCmdBuffer_->vkCommandBuffer();
                    vkQueueSubmit(device->secondaryQueue(), 1, &si, megaFence_->vkFence());
                    // 10-second timeout: prevents infinite hang on GPU TDR.
            // Mega-BLAS builds typically complete in <1s; 10s is very generous.
            constexpr uint64_t kMegaBlasTimeoutNs = 10000000000ULL;
            VkResult megaFenceResult = vkWaitForFences(device->vkDevice(), 1, &megaFence_->vkFence(), true, kMegaBlasTimeoutNs);
            if (megaFenceResult == VK_TIMEOUT) {
                std::cerr << "[MegaBLAS] vkWaitForFences timed out (10s) — GPU may be hung" << std::endl;
                vkDeviceWaitIdle(device->vkDevice());
                // Continue with potentially incomplete BLAS — the next frame will rebuild
            }
                    vkResetFences(device->vkDevice(), 1, &megaFence_->vkFence());

                    mega->blas = megaBuilt;

                    newMegaCache[mk] = mega;
                    megaBuildCount++;

                    if (megaBuildCount % 10 == 0) {
                        std::cout << "[MegaBLAS] Built " << megaBuildCount << " mega-chunks ("
                                  << totalGeoms << " geoms in latest, "
                                  << chunkIndices.size() << " sub-chunks), "
                                  << megaCacheHit << " cache hits" << std::endl;
                    }
                }

                // Add mega-chunk as TLAS instance
                auto &mega = newMegaCache[mk];
                VkTransformMatrixKHR megaTransform = {
                    1, 0, 0, static_cast<float>(mega->origin.x - cameraPos.x),
                    0, 1, 0, static_cast<float>(mega->origin.y - cameraPos.y),
                    0, 0, 1, static_cast<float>(mega->origin.z - cameraPos.z),
                };

                instanceBuilder.defineInstance(megaTransform, blasIndex, 0x01, blasGroupAccu, 0, mega->blas);
                currBlasSnapshot.blases.push_back(mega->blas);
                currBlasSnapshot.generations.push_back(mega->contentHash);

                // SBT: SHADOW prefix + all sub-chunk geometry types
                geometryTypes.push_back(World::GeometryTypes::SHADOW);
                geometryTypes.insert(geometryTypes.end(), mega->geometryTypes.begin(), mega->geometryTypes.end());

                for (size_t g = 0; g < mega->vertexBuffers.size(); g++) {
                    vertexBufferAddrs.push_back(mega->vertexBuffers[g]->bufferAddress());
                    indexBufferAddrs.push_back(mega->indexBuffers[g]->bufferAddress());
                    lastVertexBufferAddrs.push_back(0);
                    lastIndexBufferAddrs.push_back(0);
                }

                {
                    glm::mat4 lastObjToWorldMat = glm::transpose(glm::mat4(
                        glm::vec4(1, 0, 0, static_cast<float>(mega->origin.x - cameraPos.x)),
                        glm::vec4(0, 1, 0, static_cast<float>(mega->origin.y - cameraPos.y)),
                        glm::vec4(0, 0, 1, static_cast<float>(mega->origin.z - cameraPos.z)),
                        glm::vec4(0, 0, 0, 1)));
                    lastObjToWorldMats.push_back(lastObjToWorldMat);
                }

                blasOffset.push_back(blasAccu);
                blasAccu += static_cast<uint32_t>(mega->geometryTypes.size());
                blasGroupAccu += static_cast<uint32_t>(mega->geometryTypes.size()) + 1;
                blasIndex++;
            }

            // Evict stale mega-chunks — GC ALL resources, not just BLAS.
            // In-flight frames may still reference vertex/index buffer addresses
            // from the SSBO, so we must defer destruction via GC ring buffer.
            auto &gc = framework->gc();
            for (auto &[mk, oldMega] : megaChunkCache_) {
                if (newMegaCache.find(mk) == newMegaCache.end()) {
                    gc.collect(oldMega->blas);
                    for (auto &vb : oldMega->vertexBuffers) gc.collect(vb);
                    for (auto &ib : oldMega->indexBuffers) gc.collect(ib);
                    for (auto &tb : oldMega->transformBuffers) gc.collect(tb);
                }
            }
            megaChunkCache_ = std::move(newMegaCache);

            // Barrier: make secondary-queue mega-BLAS builds visible to main-queue TLAS build.
            // The fence wait ensures completion, but a memory barrier ensures visibility.
            worldCommandBuffer->barriersMemory({vk::CommandBuffer::MemoryBarrier{
                .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
                .dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                .dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR,
            }});
        } else if (!megaChunkCache_.empty()) {
            auto &gc = framework->gc();
            for (auto &[mk, oldMega] : megaChunkCache_) {
                gc.collect(oldMega->blas);
                for (auto &vb : oldMega->vertexBuffers) gc.collect(vb);
                for (auto &ib : oldMega->indexBuffers) gc.collect(ib);
                for (auto &tb : oldMega->transformBuffers) gc.collect(tb);
            }
            megaChunkCache_.clear();
        }
    }

    // Area light gathering: collect from loaded chunks, cull by vertical range + distance
    // NOTE: No frustum culling — lights behind the camera still contribute via bounced
    // illumination and removing them causes visible pop-in when rotating the camera.
    // Vertical culling is safe because deep underground lights are behind solid rock.
    {
        g_crashRing.record("WP:lights");
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
        g_crashRing.record("WP:lightsDone");
    }

    cpuAccInstances += cpuMsSince(cpuT6);
    pfInstances += cpuMsSince(cpuT6);

    if (instanceBuilder.instances.empty()) {
        g_crashRing.record("WP:noInstances");
        tlas = nullptr;
        prevTlasInstanceCount_ = 0;
        return;
    }

    // Log TLAS instance count and build mode for profiling (every 120 frames ~ 1/sec at 120fps)
    static uint32_t tlasLogCounter = 0;
    static uint32_t tlasUpdateCount = 0;
    static uint32_t tlasBuildCount = 0;
    static uint32_t tlasBuildFirstCount = 0;
    static uint32_t tlasBuildInstanceCount = 0;
    static uint32_t tlasBuildGenerationCount = 0;
    static uint32_t tlasBuildBlasHandleCount = 0;

    auto cpuT7 = Clock::now();
    // Determine if TLAS UPDATE is possible by comparing per-instance BLAS generations.
    // UPDATE requires identical BLAS handles at each instance index. If any chunk BLAS
    // was rebuilt (generation changed) or instance count changed, we must full BUILD.
    currBlasSnapshot.instanceCount = static_cast<uint32_t>(instanceBuilder.instances.size());
    uint32_t currentInstanceCount = currBlasSnapshot.instanceCount;

    bool canUpdate = tlas != nullptr
        && currBlasSnapshot.instanceCount == prevBlasSnapshot_.instanceCount
        && currBlasSnapshot.generations == prevBlasSnapshot_.generations
        && currBlasSnapshot.blases == prevBlasSnapshot_.blases;

    auto countDiffs = [](const auto &prev, const auto &curr, size_t &firstDiff) {
        size_t diffCount = 0;
        firstDiff = std::numeric_limits<size_t>::max();
        const size_t common = std::min(prev.size(), curr.size());
        for (size_t i = 0; i < common; ++i) {
            if (!(prev[i] == curr[i])) {
                if (firstDiff == std::numeric_limits<size_t>::max()) {
                    firstDiff = i;
                }
                ++diffCount;
            }
        }
        if (prev.size() != curr.size()) {
            if (firstDiff == std::numeric_limits<size_t>::max()) {
                firstDiff = common;
            }
            diffCount += prev.size() > curr.size() ? prev.size() - curr.size() : curr.size() - prev.size();
        }
        return diffCount;
    };

    size_t firstGenDiff = std::numeric_limits<size_t>::max();
    size_t firstBlasDiff = std::numeric_limits<size_t>::max();
    size_t genDiffCount = 0;
    size_t blasDiffCount = 0;
    const bool tlasFirstBuild = tlas == nullptr;
    const bool tlasInstanceCountChanged =
        currBlasSnapshot.instanceCount != prevBlasSnapshot_.instanceCount;
    bool tlasGenerationChanged = false;
    bool tlasBlasHandleChanged = false;
    const char *tlasBuildReason = "none";
    if (!canUpdate) {
        genDiffCount = countDiffs(prevBlasSnapshot_.generations,
                                  currBlasSnapshot.generations,
                                  firstGenDiff);
        blasDiffCount = countDiffs(prevBlasSnapshot_.blases,
                                   currBlasSnapshot.blases,
                                   firstBlasDiff);
        tlasGenerationChanged = genDiffCount != 0;
        tlasBlasHandleChanged = blasDiffCount != 0;
        if (tlasFirstBuild) {
            tlasBuildReason = "firstBuild";
        } else if (tlasInstanceCountChanged) {
            tlasBuildReason = "instanceCountChanged";
        } else if (tlasGenerationChanged) {
            tlasBuildReason = "generationChanged";
        } else if (tlasBlasHandleChanged) {
            tlasBuildReason = "blasHandleChanged";
        } else {
            tlasBuildReason = "unknown";
        }
    }
    const bool tlasBuildReasonFirst = !canUpdate && tlasFirstBuild;
    const bool tlasBuildReasonInstance = !canUpdate && !tlasBuildReasonFirst
        && tlasInstanceCountChanged;
    const bool tlasBuildReasonGeneration = !canUpdate && !tlasBuildReasonFirst
        && !tlasBuildReasonInstance && tlasGenerationChanged;
    const bool tlasBuildReasonBlasHandle = !canUpdate && !tlasBuildReasonFirst
        && !tlasBuildReasonInstance && !tlasBuildReasonGeneration && tlasBlasHandleChanged;

    constexpr VkBuildAccelerationStructureFlagsKHR tlasFlags =
        VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
        VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;

    g_crashRing.record("WP:tlas");
    if (canUpdate) {
        ScopedGpuProfile profile(profileCmd, "RT.WP.TLASUpdate");
        // UPDATE path: reuse existing TLAS, only transforms changed
        // Upload new instance data (endInstanceBuilder writes to a new host-visible buffer)
        instanceBuilder.endInstanceBuilder(device, vma);
        tlasBuilder->defineBuildProperty(tlasFlags);  // sets flags_ for the geometry info

        g_crashRing.record("WP:tlasUpdate");
        GpuDiag::checkpoint(worldCommandBuffer->vkCommandBuffer(), GpuDiag::TLAS_UPDATE);
        tlasBuilder->updateAndSubmit(tlas, tlasScratchBuffer_, worldCommandBuffer);
        tlasUpdateCount++;
    } else {
        ScopedGpuProfile profile(profileCmd, "RT.WP.TLASBuild");
        // Full BUILD path: instance count or BLAS composition changed
        instanceBuilder.endInstanceBuilder(device, vma);
        tlasBuilder->defineBuildProperty(tlasFlags);
        tlasBuilder->querySizeInfo(device);
        tlasBuilder->allocateBuffers(physicalDevice, device, vma);
        g_crashRing.record("WP:tlasBuild");
        GpuDiag::checkpoint(worldCommandBuffer->vkCommandBuffer(), GpuDiag::TLAS_BUILD);
        tlas = tlasBuilder->buildAndSubmit(device, worldCommandBuffer);
        if (tlasBuildReasonFirst) ++tlasBuildFirstCount;
        if (tlasBuildReasonInstance) ++tlasBuildInstanceCount;
        if (tlasBuildReasonGeneration) ++tlasBuildGenerationCount;
        if (tlasBuildReasonBlasHandle) ++tlasBuildBlasHandleCount;
        const long long firstGenDiffLog = firstGenDiff == std::numeric_limits<size_t>::max()
            ? -1LL
            : static_cast<long long>(firstGenDiff);
        const long long firstBlasDiffLog = firstBlasDiff == std::numeric_limits<size_t>::max()
            ? -1LL
            : static_cast<long long>(firstBlasDiff);
        renderDiag("WP TLAS full build reason=%s prevInstances=%u currInstances=%u genDiffs=%zu firstGenDiff=%lld blasDiffs=%zu firstBlasDiff=%lld",
                   tlasBuildReason,
                   prevBlasSnapshot_.instanceCount,
                   currBlasSnapshot.instanceCount,
                   genDiffCount,
                   firstGenDiffLog,
                   blasDiffCount,
                   firstBlasDiffLog);

        // Persist scratch buffer sized for max(build, update) for future UPDATE calls.
        // Query both BUILD and UPDATE scratch sizes to ensure the buffer is large enough.
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
    g_crashRing.record("WP:tlasDone");

    cpuAccTlas += cpuMsSince(cpuT7);
    pfTlas += cpuMsSince(cpuT7);
    cpuAccTotal += cpuMsSince(cpuT0);
    cpuFrameCount++;

    // Per-frame CSV logging (when Renderer::options.perFrameTiming is on)
    {
        static uint64_t wpFrameIndex = 0;
        static bool wpHeaderWritten = false;
        std::ofstream wpLog("C:/RadSER/results/per_frame_world_prepare.log", std::ios::app);
        if (wpLog.is_open()) {
            if (!wpHeaderWritten) {
                wpLog << "frame,instances,check,important,entity,instances_pop,tlas,total\n";
                wpHeaderWritten = true;
            }
            wpLog << wpFrameIndex << ","
                << currentInstanceCount << ","
                << pfCheck << ","
                << pfImportant << ","
                << pfEntity << ","
                << pfInstances << ","
                << pfTlas << ","
                << cpuMsSince(cpuT0) << "\n";
            wpFrameIndex++;
        }
        pfCheck = pfImportant = pfEntity = 0;
        pfInstances = pfTlas = 0;
    }

    // Save BLAS snapshot: keeps shared_ptrs alive until this context is reused,
    // preventing GC from freeing BLASes while the GPU still references their addresses.
    prevBlasSnapshot_ = std::move(currBlasSnapshot);
    prevTlasInstanceCount_ = currentInstanceCount;

    if (++tlasLogCounter >= 120) {
        float n = static_cast<float>(cpuFrameCount);
        std::cout << "[Profiler] TLAS instances: " << currentInstanceCount
                  << "  builds: " << tlasBuildCount << "  updates: " << tlasUpdateCount
                  << "  reasons[first=" << tlasBuildFirstCount
                  << ", instance=" << tlasBuildInstanceCount
                  << ", generation=" << tlasBuildGenerationCount
                  << ", blasHandle=" << tlasBuildBlasHandleCount
                  << "]" << std::endl;
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
                       << " tlasReasonFirst=" << tlasBuildFirstCount
                       << " tlasReasonInstance=" << tlasBuildInstanceCount
                       << " tlasReasonGeneration=" << tlasBuildGenerationCount
                       << " tlasReasonBlasHandle=" << tlasBuildBlasHandleCount
                       << " threads=" << Renderer::threadPool.threadCount()
                       << "\n";
            }
        }
        tlasLogCounter = 0;
        tlasUpdateCount = 0;
        tlasBuildCount = 0;
        tlasBuildFirstCount = 0;
        tlasBuildInstanceCount = 0;
        tlasBuildGenerationCount = 0;
        tlasBuildBlasHandleCount = 0;
        cpuAccCheck = cpuAccSchedule = cpuAccImportant = 0;
        cpuAccEntity = cpuAccInstances = cpuAccTlas = cpuAccTotal = 0;
        cpuFrameCount = 0;
    }

    {
        ScopedGpuProfile profile(profileCmd, "RT.WP.ASReadBarrier");
        worldCommandBuffer->barriersMemory({vk::CommandBuffer::MemoryBarrier{
            .srcStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
            .srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
            .dstStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
            .dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR,
        }});
    }

    auto rtModuleCtx = rayTracingModuleContext.lock();
    if (!rtModuleCtx) return;
    g_crashRing.record("WP:sbt");
    {
        ScopedGpuProfile profile(profileCmd, "RT.WP.SBTSetup");
        rtModuleCtx->sbt->setupHitSBT(geometryTypes);
        if (rtModuleCtx->sharcUpdateSbt) {
            rtModuleCtx->sharcUpdateSbt->setupHitSBT(geometryTypes);
        }
    }

    g_crashRing.record("WP:upload");
    {
        ScopedGpuProfile profile(profileCmd, "RT.WP.MetadataUpload");
        uploadBuffer(blasOffset, vertexBufferAddrs, indexBufferAddrs, lastVertexBufferAddrs, lastIndexBufferAddrs,
                     lastObjToWorldMats, biomeColors);
    }
    g_crashRing.record("WP:done");
}
