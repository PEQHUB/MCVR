#include "core/render/modules/world/ray_tracing/submodules/world_prepare.hpp"

#include "core/render/buffers.hpp"
#include "core/render/chunks.hpp"
#include "core/render/entities.hpp"
#include "core/render/modules/world/ray_tracing/ray_tracing_module.hpp"
#include "core/render/render_framework.hpp"
#include "core/render/renderer.hpp"
#include "core/render/world.hpp"

#include <filesystem>
#include <glm/gtc/type_ptr.hpp>

WorldPrepare::WorldPrepare() {}

void WorldPrepare::init(std::shared_ptr<Framework> framework, std::shared_ptr<RayTracingModule> rayTracingModule) {
    framework_ = framework;
    rayTracingModule_ = rayTracingModule;
}

void WorldPrepare::build() {
    auto framework = framework_.lock();
    auto rayTracingModule = rayTracingModule_.lock();
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
                                       std::vector<glm::mat4> &lastObjToWorldMats) {
    auto context = frameworkContext.lock();
    auto framework = context->framework.lock();
    auto vma = framework->vma();
    auto device = framework->device();
    auto physicalDevice = framework->physicalDevice();
    auto mainQueueIndex = physicalDevice->mainQueueIndex();
    auto cmdBuffer = context->worldCommandBuffer;

    blasOffsetsBuffer = vk::DeviceLocalBuffer::create(
        vma, device, blasOffsets.size() * sizeof(uint32_t),
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    blasOffsetsBuffer->uploadToStagingBuffer(blasOffsets.data());

    vertexBufferAddr = vk::DeviceLocalBuffer::create(
        vma, device, vertexBufferAddrs.size() * sizeof(uint64_t),
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    vertexBufferAddr->uploadToStagingBuffer(vertexBufferAddrs.data());

    indexBufferAddr = vk::DeviceLocalBuffer::create(
        vma, device, indexBufferAddrs.size() * sizeof(uint64_t),
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    indexBufferAddr->uploadToStagingBuffer(indexBufferAddrs.data());

    lastVertexBufferAddr = vk::DeviceLocalBuffer::create(
        vma, device, lastVertexBufferAddrs.size() * sizeof(uint64_t),
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    lastVertexBufferAddr->uploadToStagingBuffer(lastVertexBufferAddrs.data());

    lastIndexBufferAddr = vk::DeviceLocalBuffer::create(
        vma, device, lastIndexBufferAddrs.size() * sizeof(uint64_t),
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    lastIndexBufferAddr->uploadToStagingBuffer(lastIndexBufferAddrs.data());

    lastObjToWorldMat = vk::DeviceLocalBuffer::create(
        vma, device, lastObjToWorldMats.size() * sizeof(glm::mat4),
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    lastObjToWorldMat->uploadToStagingBuffer(lastObjToWorldMats.data());

    std::vector<std::shared_ptr<vk::DeviceLocalBuffer>> rayTracingMetaData{{
        blasOffsetsBuffer,
        vertexBufferAddr,
        indexBufferAddr,
        lastVertexBufferAddr,
        lastIndexBufferAddr,
        lastObjToWorldMat,
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

    std::shared_ptr<Framework> framework = Renderer::instance().framework();
    std::shared_ptr<FrameworkContext> context = frameworkContext.lock();
    std::shared_ptr<vk::VMA> vma = framework->vma();
    std::shared_ptr<vk::Device> device = framework->device();
    std::shared_ptr<vk::PhysicalDevice> physicalDevice = framework->physicalDevice();
    std::shared_ptr<vk::CommandBuffer> worldCommandBuffer = context->worldCommandBuffer;

    auto chunks = Renderer::instance().world()->chunks();
    auto entities = Renderer::instance().world()->entities();
    auto cameraPos = Renderer::instance().world()->getCameraPos();

    std::unique_lock<std::recursive_mutex> lock(chunks->mutex());

    if (chunks->chunkBuildScheduler() != nullptr) {
        chunks->chunkBuildScheduler()->tryCheckBatchesFinish();
        chunks->chunkBuildScheduler()->tryScheduleBatches(
            Renderer::instance().world()->chunks()->chunkBuildScheduler()->chunkBuildingBatchSize());
    }

    if (chunks->importantBLASBuilders().size() > 0) {
        vk::BLASBuilder::batchSubmit(chunks->importantBLASBuilders(), worldCommandBuffer);
    }

    if (entities->blasBatchBuilder() != nullptr) { entities->blasBatchBuilder()->submit(worldCommandBuffer); }

    worldCommandBuffer->barriersMemory({vk::CommandBuffer::MemoryBarrier{
        .srcStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        .srcAccessMask =
            VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
        .dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        .dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR,
    }});

    uint32_t blasAccu = 0, blasGroupAccu = 0;
    std::vector<uint32_t> blasOffset;
    std::vector<uint32_t> geometryTypes;
    std::vector<uint64_t> vertexBufferAddrs, indexBufferAddrs;
    std::vector<uint64_t> lastVertexBufferAddrs, lastIndexBufferAddrs;
    std::vector<glm::mat4> lastObjToWorldMats;

    tlasBuilder = vk::TLASBuilder::create();
    auto &instanceBuilder = tlasBuilder->beginInstanceBuilder();
    int blasIndex = 0;

    // Entity
    {
        auto entityBatch = entities->entityBatch();

        if (entityBatch != nullptr) {
            static std::queue<std::map<int, std::pair<std::shared_ptr<Entity>, VkTransformMatrixKHR>>>
                previousEntityRenderDataBatches;
            static std::map<int, std::pair<std::shared_ptr<Entity>, VkTransformMatrixKHR>> emptyMap;

            std::map<int, std::pair<std::shared_ptr<Entity>, VkTransformMatrixKHR>> &previousEntityRenderDataBatch =
                previousEntityRenderDataBatches.empty() ? emptyMap : previousEntityRenderDataBatches.back();
            if (previousEntityRenderDataBatches.size() > Renderer::instance().framework()->swapchain()->imageCount())
                previousEntityRenderDataBatches.pop();
            std::map<int, std::pair<std::shared_ptr<Entity>, VkTransformMatrixKHR>> &currentEntityRenderDataBatch =
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

    // Chunk
    {
        auto &chunk1s = chunks->chunks();
        for (int i = 0; i < chunk1s.size(); i++) {
            auto &chunk1 = chunk1s[i];
            if (chunk1->blas == nullptr) continue;

            VkTransformMatrixKHR transform = {
                1, 0, 0, static_cast<float>(static_cast<double>(chunk1->x) - cameraPos.x), //
                0, 1, 0, static_cast<float>(static_cast<double>(chunk1->y) - cameraPos.y), //
                0, 0, 1, static_cast<float>(static_cast<double>(chunk1->z) - cameraPos.z), //
            };

            instanceBuilder.defineInstance(transform, blasIndex, 0x01, blasGroupAccu, 0, chunk1->blas);

            geometryTypes.push_back(World::GeometryTypes::SHADOW);
            geometryTypes.insert(geometryTypes.end(), chunk1->geometryTypes->begin(), chunk1->geometryTypes->end());

            for (int j = 0; j < chunk1->geometryCount; j++) {
                vertexBufferAddrs.push_back((*chunk1->vertexBuffers)[j]->bufferAddress());
                indexBufferAddrs.push_back((*chunk1->indexBuffers)[j]->bufferAddress());
                lastVertexBufferAddrs.push_back(0);
                lastIndexBufferAddrs.push_back(0);
            }

            // read (fake, since chunk is not moving) previous render data
            {
                glm::mat4 lastObjToWorldMat = glm::transpose(glm::mat4(
                    glm::vec4(1.0f, 0.0f, 0.0f, static_cast<float>(static_cast<double>(chunk1->x) - cameraPos.x)), //
                    glm::vec4(0.0f, 1.0f, 0.0f, static_cast<float>(static_cast<double>(chunk1->y) - cameraPos.y)), //
                    glm::vec4(0.0f, 0.0f, 1.0f, static_cast<float>(static_cast<double>(chunk1->z) - cameraPos.z)), //
                    glm::vec4(0.0f, 0.0f, 0.0f, 1.0f)));
                lastObjToWorldMats.push_back(lastObjToWorldMat);
            }

            blasOffset.push_back(blasAccu);
            blasAccu += chunk1->geometryCount;
            blasGroupAccu += chunk1->geometryCount + 1; // shadow

            blasIndex++;
        }
    }

    if (instanceBuilder.instances.empty()) {
        tlas = nullptr;
        return;
    }

    tlas = instanceBuilder.endInstanceBuilder(device, vma)
               ->defineBuildProperty(VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR)
               ->querySizeInfo(device)
               ->allocateBuffers(physicalDevice, device, vma)
               ->buildAndSubmit(device, worldCommandBuffer);

    worldCommandBuffer->barriersMemory({vk::CommandBuffer::MemoryBarrier{
        .srcStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        .srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
        .dstStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
        .dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR,
    }});

    rayTracingModuleContext.lock()->sbt->setupHitSBT(geometryTypes);

    uploadBuffer(blasOffset, vertexBufferAddrs, indexBufferAddrs, lastVertexBufferAddrs, lastIndexBufferAddrs,
                 lastObjToWorldMats);
}