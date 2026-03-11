#include "core/render/chunks.hpp"

#include "core/render/buffers.hpp"
#include "core/render/render_framework.hpp"
#include "core/render/renderer.hpp"
#include "core/render/textures.hpp"
#ifdef MCVR_ENABLE_OMM
#include "core/render/omm_baker.hpp"
#endif

#include <algorithm>
#include <cassert>
#include <cmath>

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

ChunkBuildData::~ChunkBuildData() {
    for (auto &gd : ommGeometryData) {
        if (gd.micromap != VK_NULL_HANDLE && gd.device) {
            vkDestroyMicromapEXT(gd.device->vkDevice(), gd.micromap, nullptr);
            gd.micromap = VK_NULL_HANDLE;
        }
    }
}

void ChunkBuildData::build(bool allowMicromapBake, bool skipOMM) {
    auto framework = Renderer::instance().framework();
    auto vma = framework->vma();
    auto device = framework->device();
    auto physicalDevice = framework->physicalDevice();
    auto textures = Renderer::instance().textures();
    bool useOMM = !skipOMM && device->hasOMM() && Renderer::options.ommEnabled && textures != nullptr;

#ifdef MCVR_ENABLE_OMM
    // Thread-local OMM baker (one per thread, SDK is not thread-safe per instance)
    static thread_local std::unique_ptr<OMMBaker> tlBaker;
    if (useOMM && !tlBaker) {
        tlBaker = std::make_unique<OMMBaker>();
    }
#endif

    ommGeometryData.resize(geometryCount);

    for (int i = 0; i < geometryCount; i++) {
        auto vertexBuffer =
            vk::DeviceLocalBuffer::create(vma, device, vertices[i].size() * sizeof(vk::VertexFormat::PBRTriangle),
                                          VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                                              VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        vertexBuffer->uploadToStagingBuffer(vertices[i].data());
        vertexBuffers.push_back(vertexBuffer);

        auto indexBuffer =
            vk::DeviceLocalBuffer::create(vma, device, indices[i].size() * sizeof(uint32_t),
                                          VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                                              VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        indexBuffer->uploadToStagingBuffer(indices[i].data());
        indexBuffers.push_back(indexBuffer);

        // OMM: per-triangle opacity for WORLD_TRANSPARENT geometry
        if (useOMM && geometryTypes[i] == World::WORLD_TRANSPARENT) {
#ifdef MCVR_ENABLE_OMM
            uint32_t numTriangles = static_cast<uint32_t>(indices[i].size()) / 3;

            if (!allowMicromapBake) {
                // Phase 1 fallback: special indices only (for important/immediate chunks)
                std::vector<int32_t> ommIndices(numTriangles);
                for (uint32_t t = 0; t < numTriangles; t++) {
                    uint32_t vertIdx = indices[i][t * 3];
                    uint32_t texId = vertices[i][vertIdx].textureID;
                    auto alphaClass = textures->getTextureAlphaClass(texId);
                    switch (alphaClass) {
                        case Textures::AlphaClass::FULLY_OPAQUE:
                            ommIndices[t] = VK_OPACITY_MICROMAP_SPECIAL_INDEX_FULLY_OPAQUE_EXT;
                            break;
                        case Textures::AlphaClass::FULLY_TRANSPARENT:
                            // Translucent textures (e.g. water) have no fully-opaque pixels but ARE
                            // visible. Fall back to AHS instead of marking invisible.
                            ommIndices[t] = VK_OPACITY_MICROMAP_SPECIAL_INDEX_FULLY_UNKNOWN_OPAQUE_EXT;
                            break;
                        default:
                            ommIndices[t] = VK_OPACITY_MICROMAP_SPECIAL_INDEX_FULLY_UNKNOWN_OPAQUE_EXT;
                            break;
                    }
                }
                auto ommIdxBuffer = vk::DeviceLocalBuffer::create(
                    vma, device, numTriangles * sizeof(int32_t),
                    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                        VK_BUFFER_USAGE_MICROMAP_BUILD_INPUT_READ_ONLY_BIT_EXT);
                ommIdxBuffer->uploadToStagingBuffer(ommIndices.data());
                ommIndexBuffers.push_back(ommIdxBuffer);
            } else {

            // Phase 2: full per-micro-triangle baking
            bool bakingDone = false;

            // Group triangles by textureID to bake each texture group separately
            std::map<uint32_t, std::vector<uint32_t>> texGroups; // texID -> list of tri indices
            for (uint32_t t = 0; t < numTriangles; t++) {
                uint32_t vertIdx = indices[i][t * 3];
                uint32_t texId = vertices[i][vertIdx].textureID;
                texGroups[texId].push_back(t);
            }

            // Per-triangle OMM index buffer (maps each triangle to an OMM block or special index)
            std::vector<int32_t> ommIndices(numTriangles, VK_OPACITY_MICROMAP_SPECIAL_INDEX_FULLY_UNKNOWN_OPAQUE_EXT);
            // Merged OMM array data and descriptors across all texture groups
            std::vector<uint8_t> mergedArrayData;
            std::vector<VkMicromapTriangleEXT> mergedDescs;
            // Track usage counts
            std::map<uint64_t, uint32_t> descHistMap, indexHistMap; // key = (subdiv << 16 | format)

            for (auto &[texId, triList] : texGroups) {
                const auto *alphaData = textures->getTextureAlphaData(texId);
                if (!alphaData || alphaData->alpha.empty()) {
                    // No alpha data or animated → fall back to special index
                    for (uint32_t t : triList) {
                        auto alphaClass = textures->getTextureAlphaClass(texId);
                        switch (alphaClass) {
                            case Textures::AlphaClass::FULLY_OPAQUE:
                                ommIndices[t] = VK_OPACITY_MICROMAP_SPECIAL_INDEX_FULLY_OPAQUE_EXT;
                                break;
                            case Textures::AlphaClass::FULLY_TRANSPARENT:
                                // Translucent textures (e.g. water) have no fully-opaque pixels but ARE
                                // visible. Fall back to AHS instead of marking invisible.
                                ommIndices[t] = VK_OPACITY_MICROMAP_SPECIAL_INDEX_FULLY_UNKNOWN_OPAQUE_EXT;
                                break;
                            default:
                                ommIndices[t] = VK_OPACITY_MICROMAP_SPECIAL_INDEX_FULLY_UNKNOWN_OPAQUE_EXT;
                                break;
                        }
                    }
                    // Update index histogram for special indices
                    for (uint32_t t : triList) {
                        // Special indices don't contribute to desc histogram, only index histogram
                        // Vulkan spec: special indices counted with subdivisionLevel=0, format matching
                    }
                    continue;
                }

                // Translucent textures (e.g. water): all alpha < 255 but visible.
                // The OMM baker would mark them OPAQUE (alpha > cutoff), which
                // skips AHS and blocks shadow rays. Use UNKNOWN_OPAQUE to force AHS.
                auto alphaClass = textures->getTextureAlphaClass(texId);
                if (alphaClass == Textures::AlphaClass::FULLY_TRANSPARENT) {
                    for (uint32_t t : triList) {
                        ommIndices[t] = VK_OPACITY_MICROMAP_SPECIAL_INDEX_FULLY_UNKNOWN_OPAQUE_EXT;
                    }
                    continue;
                }

                // Build local index buffer for this texture group
                std::vector<uint32_t> localIndices;
                localIndices.reserve(triList.size() * 3);
                for (uint32_t t : triList) {
                    localIndices.push_back(indices[i][t * 3 + 0]);
                    localIndices.push_back(indices[i][t * 3 + 1]);
                    localIndices.push_back(indices[i][t * 3 + 2]);
                }

                OMMBaker::BakeInput input{};
                input.alphaData = alphaData->alpha.data();
                input.texWidth = alphaData->width;
                input.texHeight = alphaData->height;
                input.uvData = &vertices[i][0].textureUV;
                input.uvStrideBytes = sizeof(vk::VertexFormat::PBRTriangle);
                input.indexData = localIndices.data();
                input.indexCount = static_cast<uint32_t>(localIndices.size());
                input.alphaCutoff = 0.05f;
                input.maxSubdivisionLevel = Renderer::options.ommBakerLevel;

                OMMBaker::BakeResult result;
                if (tlBaker && tlBaker->bake(input, result)) {
                    bakingDone = true;
                    uint32_t baseOffset = static_cast<uint32_t>(mergedArrayData.size());
                    uint32_t baseDescIndex = static_cast<uint32_t>(mergedDescs.size());

                    // Append array data
                    mergedArrayData.insert(mergedArrayData.end(), result.arrayData.begin(), result.arrayData.end());

                    // Append descriptors with adjusted offsets
                    for (uint32_t d = 0; d < result.descArrayCount; d++) {
                        VkMicromapTriangleEXT desc{};
                        desc.dataOffset = result.descOffsets[d] + baseOffset;
                        desc.subdivisionLevel = result.descSubdivisionLevels[d];
                        desc.format = result.descFormats[d];
                        mergedDescs.push_back(desc);
                    }

                    // Map per-triangle indices back to the original triangle positions
                    // result.indexBuffer has one entry per triangle in triList order
                    for (uint32_t li = 0; li < triList.size(); li++) {
                        int32_t idx = result.indexBuffer[li];
                        if (idx >= 0) {
                            // Remap to merged desc array
                            ommIndices[triList[li]] = idx + static_cast<int32_t>(baseDescIndex);
                        } else {
                            // Special index, keep as-is
                            ommIndices[triList[li]] = idx;
                        }
                    }

                    // Accumulate desc array histogram
                    for (auto &uc : result.descArrayHistogram) {
                        uint64_t key = (static_cast<uint64_t>(uc.subdivisionLevel) << 16) | uc.format;
                        descHistMap[key] += uc.count;
                    }
                    // Accumulate index histogram
                    for (auto &uc : result.indexHistogram) {
                        uint64_t key = (static_cast<uint64_t>(uc.subdivisionLevel) << 16) | uc.format;
                        indexHistMap[key] += uc.count;
                    }
                } else {
                    // Bake failed → fallback to special indices
                    for (uint32_t t : triList) {
                        ommIndices[t] = VK_OPACITY_MICROMAP_SPECIAL_INDEX_FULLY_UNKNOWN_OPAQUE_EXT;
                    }
                }
            }

            // Upload OMM index buffer (always needed if useOMM)
            auto ommIdxBuffer = vk::DeviceLocalBuffer::create(
                vma, device, numTriangles * sizeof(int32_t),
                VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                    VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                    VK_BUFFER_USAGE_MICROMAP_BUILD_INPUT_READ_ONLY_BIT_EXT);
            ommIdxBuffer->uploadToStagingBuffer(ommIndices.data());
            ommIndexBuffers.push_back(ommIdxBuffer);

            if (bakingDone && !mergedDescs.empty()) {
                auto &gd = ommGeometryData[i];

                // Upload OMM array data
                gd.arrayBuffer = vk::DeviceLocalBuffer::create(
                    vma, device, mergedArrayData.size(),
                    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                        VK_BUFFER_USAGE_MICROMAP_BUILD_INPUT_READ_ONLY_BIT_EXT);
                gd.arrayBuffer->uploadToStagingBuffer(mergedArrayData.data());

                // Upload OMM descriptor array (VkMicromapTriangleEXT)
                gd.descBuffer = vk::DeviceLocalBuffer::create(
                    vma, device, mergedDescs.size() * sizeof(VkMicromapTriangleEXT),
                    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                        VK_BUFFER_USAGE_MICROMAP_BUILD_INPUT_READ_ONLY_BIT_EXT);
                gd.descBuffer->uploadToStagingBuffer(mergedDescs.data());

                // Convert histograms
                for (auto &[key, count] : descHistMap) {
                    VkMicromapUsageEXT usage{};
                    usage.count = count;
                    usage.subdivisionLevel = static_cast<uint32_t>(key >> 16);
                    usage.format = static_cast<uint32_t>(key & 0xFFFF);
                    gd.descHistogram.push_back(usage);
                }
                for (auto &[key, count] : indexHistMap) {
                    VkMicromapUsageEXT usage{};
                    usage.count = count;
                    usage.subdivisionLevel = static_cast<uint32_t>(key >> 16);
                    usage.format = static_cast<uint32_t>(key & 0xFFFF);
                    gd.indexHistogram.push_back(usage);
                }

                // Query micromap build sizes
                VkMicromapBuildInfoEXT buildInfo{};
                buildInfo.sType = VK_STRUCTURE_TYPE_MICROMAP_BUILD_INFO_EXT;
                buildInfo.type = VK_MICROMAP_TYPE_OPACITY_MICROMAP_EXT;
                buildInfo.mode = VK_BUILD_MICROMAP_MODE_BUILD_EXT;
                buildInfo.usageCountsCount = static_cast<uint32_t>(gd.descHistogram.size());
                buildInfo.pUsageCounts = gd.descHistogram.data();

                VkMicromapBuildSizesInfoEXT sizeInfo{};
                sizeInfo.sType = VK_STRUCTURE_TYPE_MICROMAP_BUILD_SIZES_INFO_EXT;
                vkGetMicromapBuildSizesEXT(device->vkDevice(), VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                                           &buildInfo, &sizeInfo);

                // Allocate micromap buffer and scratch
                VkDeviceSize micromapSize = (sizeInfo.micromapSize + 255) & ~255ULL; // 256-byte align
                gd.micromapBuffer = vk::DeviceLocalBuffer::create(
                    vma, device, micromapSize,
                    VK_BUFFER_USAGE_MICROMAP_STORAGE_BIT_EXT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);

                if (sizeInfo.buildScratchSize > 0) {
                    VkDeviceSize scratchSize = (sizeInfo.buildScratchSize + 255) & ~255ULL;
                    gd.micromapScratchBuffer = vk::DeviceLocalBuffer::create(
                        vma, device, scratchSize,
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
                }

                // Create VkMicromapEXT
                VkMicromapCreateInfoEXT createInfo{};
                createInfo.sType = VK_STRUCTURE_TYPE_MICROMAP_CREATE_INFO_EXT;
                createInfo.type = VK_MICROMAP_TYPE_OPACITY_MICROMAP_EXT;
                createInfo.size = sizeInfo.micromapSize;
                createInfo.buffer = gd.micromapBuffer->vkBuffer();
                createInfo.offset = 0;
                vkCreateMicromapEXT(device->vkDevice(), &createInfo, nullptr, &gd.micromap);
                gd.device = device;

                gd.hasMicromap = true;
            }

            } // end Phase 2 (allowMicromapBake)
#endif
        } else if (useOMM) {
            // WORLD_SOLID with OMM enabled: all-opaque special indices
            // When pipeline has VK_PIPELINE_CREATE_RAY_TRACING_OPACITY_MICROMAP_BIT_EXT,
            // ALL geometries in the BLAS must have OMM pNext attached
            uint32_t numTriangles = static_cast<uint32_t>(indices[i].size()) / 3;
            std::vector<int32_t> ommIndices(numTriangles, VK_OPACITY_MICROMAP_SPECIAL_INDEX_FULLY_OPAQUE_EXT);
            auto ommIdxBuffer = vk::DeviceLocalBuffer::create(
                vma, device, numTriangles * sizeof(int32_t),
                VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                    VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                    VK_BUFFER_USAGE_MICROMAP_BUILD_INPUT_READ_ONLY_BIT_EXT);
            ommIdxBuffer->uploadToStagingBuffer(ommIndices.data());
            ommIndexBuffers.push_back(ommIdxBuffer);
        } else {
            ommIndexBuffers.push_back(nullptr);
        }
    }

    blasBuilder = vk::BLASBuilder::create();
    auto blasGeometryBuilder = blasBuilder->beginGeometries();
    for (int i = 0; i < geometryCount; i++) {
        bool isOpaque = geometryTypes[i] == World::WORLD_SOLID;
        if (ommIndexBuffers[i] != nullptr) {
            uint32_t numTriangles = static_cast<uint32_t>(indices[i].size()) / 3;
            if (ommGeometryData[i].hasMicromap) {
                blasGeometryBuilder->defineTriangleGeomrtryWithMicromap<vk::VertexFormat::PBRTriangle>(
                    vertexBuffers[i], vertices[i].size(), indexBuffers[i], indices[i].size(),
                    isOpaque, ommIndexBuffers[i]->bufferAddress(), numTriangles,
                    ommGeometryData[i].micromap,
                    ommGeometryData[i].indexHistogram.data(),
                    static_cast<uint32_t>(ommGeometryData[i].indexHistogram.size()));
            } else {
                blasGeometryBuilder->defineTriangleGeomrtry<vk::VertexFormat::PBRTriangle>(
                    vertexBuffers[i], vertices[i].size(), indexBuffers[i], indices[i].size(),
                    isOpaque, ommIndexBuffers[i]->bufferAddress(), numTriangles);
            }
        } else {
            blasGeometryBuilder->defineTriangleGeomrtry<vk::VertexFormat::PBRTriangle>(
                vertexBuffers[i], vertices[i].size(), indexBuffers[i], indices[i].size(),
                isOpaque);
        }
    }
    blasGeometryBuilder->endGeometries();
    blas = blasBuilder->defineBuildProperty(VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR)
               ->querySizeInfo(device)
               ->allocateBuffers(physicalDevice, device, vma)
               ->build(device);
}

ChunkBuildDataBatch::ChunkBuildDataBatch(uint32_t maxBatchSize,
                                         std::set<int64_t> &queuedIndexSet,
                                         std::vector<std::shared_ptr<Chunk1>> &chunks,
                                         std::vector<std::shared_ptr<ChunkBuildData>> &chunkBuildDatas,
                                         glm::vec3 cameraPos) {
    std::vector<int64_t> queuedIndices;
    std::copy(queuedIndexSet.begin(), queuedIndexSet.end(), std::back_inserter(queuedIndices));
    auto currentTime = std::chrono::steady_clock::now();
    std::sort(queuedIndices.begin(), queuedIndices.end(), [&](int64_t a, int64_t b) -> bool {
        return chunks[a]->buildFactor(currentTime, cameraPos) > chunks[b]->buildFactor(currentTime, cameraPos);
    });

    auto batchStart = std::chrono::steady_clock::now();
    static constexpr auto TIME_BUDGET = std::chrono::microseconds(4000); // 4ms

    for (int i = 0; i < std::min((size_t)maxBatchSize, queuedIndices.size()); i++) {
        // After the first chunk, enforce time budget to cap frame time spikes from OMM baking
        if (i > 0) {
            auto elapsed = std::chrono::steady_clock::now() - batchStart;
            if (elapsed > TIME_BUDGET) break;
        }

        auto iter = queuedIndexSet.find(queuedIndices[i]);
        if (iter != queuedIndexSet.end()) { queuedIndexSet.erase(iter); }

        auto data = chunkBuildDatas[queuedIndices[i]];
        data->build();
        batchData.push_back(data);
    }
}

ChunkBuildScheduler::ChunkBuildScheduler(std::set<int64_t> &queuedIndex,
                                         std::vector<std::shared_ptr<Chunk1>> &chunks,
                                         std::vector<std::shared_ptr<ChunkBuildData>> &chunkBuildDatas,
                                         std::recursive_mutex &mutex,
                                         std::shared_ptr<vk::HostVisibleBuffer> &chunkPackedData,
                                         uint32_t chunkBuildingBatchSize,
                                         uint32_t chunkBuildingTotalBatches)
    : queuedIndex_(queuedIndex),
      chunks_(chunks),
      chunkBuildDatas_(chunkBuildDatas),
      mutex_(mutex),
      chunkPackedData_(chunkPackedData),
      chunkBuildingBatchSize_(chunkBuildingBatchSize),
      chunkBuildingTotalBatches_(chunkBuildingTotalBatches) {
    auto framework = Renderer::instance().framework();
    auto device = framework->device();

    uint32_t numFences = chunkBuildingTotalBatches_;
    for (int i = 0; i < numFences; i++) { freeFences_.push(vk::Fence::create(device)); }
}

void ChunkBuildScheduler::tryCheckBatchesFinish() {
    auto framework = Renderer::instance().framework();
    auto device = framework->device();

    std::unique_lock<std::recursive_mutex> lock(mutex_);
    auto iterFence = buildingFences_.begin();
    auto iterBatch = buildingBatches_.begin();
    for (; iterFence != buildingFences_.end() && iterBatch != buildingBatches_.end();) {
        if (vkWaitForFences(device->vkDevice(), 1, &(*iterFence)->vkFence(), true, 0) == VK_SUCCESS) {
            vkResetFences(device->vkDevice(), 1, &(*iterFence)->vkFence());
            freeFences_.push(*iterFence);

            for (auto chunkBuildData : (*iterBatch)->batchData) {
                chunks_[chunkBuildData->id]->enqueue(chunkBuildData);

                ChunkPackedData data = {
                    .geometryCount = chunkBuildData->geometryCount,
                };

                chunkPackedData_->uploadToBuffer(&data, sizeof(ChunkPackedData),
                                                 chunkBuildData->id * sizeof(ChunkPackedData));
            }

            iterFence = buildingFences_.erase(iterFence);
            iterBatch = buildingBatches_.erase(iterBatch);
        }
    }
}

void ChunkBuildScheduler::waitAllBatchesFinish() {
    auto framework = Renderer::instance().framework();
    auto device = framework->device();

    std::unique_lock<std::recursive_mutex> lock(mutex_);
    auto iterFence = buildingFences_.begin();
    auto iterBatch = buildingBatches_.begin();
    for (; iterFence != buildingFences_.end() && iterBatch != buildingBatches_.end();) {
        if (vkWaitForFences(device->vkDevice(), 1, &(*iterFence)->vkFence(), true, UINT64_MAX) == VK_SUCCESS) {
            vkResetFences(device->vkDevice(), 1, &(*iterFence)->vkFence());
            freeFences_.push(*iterFence);

            for (auto chunkBuildData : (*iterBatch)->batchData) {
                chunks_[chunkBuildData->id]->enqueue(chunkBuildData);

                ChunkPackedData data = {
                    .geometryCount = chunkBuildData->geometryCount,
                };

                chunkPackedData_->uploadToBuffer(&data, sizeof(ChunkPackedData),
                                                 chunkBuildData->id * sizeof(ChunkPackedData));
            }

            iterFence = buildingFences_.erase(iterFence);
            iterBatch = buildingBatches_.erase(iterBatch);
        }
    }
}

void ChunkBuildScheduler::tryScheduleBatches(uint32_t maxBatchSize) {
    if (!Renderer::instance().framework()->isRunning()) return;
    std::unique_lock<std::recursive_mutex> lock(mutex_);
    if (!freeFences_.empty() && !queuedIndex_.empty()) {
        auto fence = freeFences_.front();

        glm::vec3 cameraPos = Renderer::instance().world()->getCameraPos();
        auto chunkBuildDataBatch =
            ChunkBuildDataBatch::create(maxBatchSize, queuedIndex_, chunks_, chunkBuildDatas_, cameraPos);

        auto framework = Renderer::instance().framework();
        auto vma = framework->vma();
        auto device = framework->device();
        auto physicalDevice = Renderer::instance().framework()->physicalDevice();
        auto secondaryQueueIndex = physicalDevice->secondaryQueueIndex();

        auto worldAsyncBuffer = framework->worldAsyncCommandBuffer();

        if (chunkBuildDataBatch->batchData.size() > 0) {
            worldAsyncBuffer->begin();

            // Upload all buffers (vertex, index, OMM)
            for (auto chunkBuildData : chunkBuildDataBatch->batchData) {
                for (int i = 0; i < chunkBuildData->geometryCount; i++) {
                    chunkBuildData->vertexBuffers[i]->uploadToBuffer(worldAsyncBuffer);
                    chunkBuildData->indexBuffers[i]->uploadToBuffer(worldAsyncBuffer);
                    if (chunkBuildData->ommIndexBuffers[i] != nullptr) {
                        chunkBuildData->ommIndexBuffers[i]->uploadToBuffer(worldAsyncBuffer);
                    }
                    // Upload Phase 2 OMM data buffers
                    auto &gd = chunkBuildData->ommGeometryData[i];
                    if (gd.hasMicromap) {
                        gd.arrayBuffer->uploadToBuffer(worldAsyncBuffer);
                        gd.descBuffer->uploadToBuffer(worldAsyncBuffer);
                    }
                }
            }

            // Barrier: transfer → micromap build + BLAS build
            std::vector<vk::CommandBuffer::BufferMemoryBarrier> bufferBarriers;
            for (auto chunkBuildData : chunkBuildDataBatch->batchData) {
                for (int i = 0; i < chunkBuildData->geometryCount; i++) {
                    VkPipelineStageFlags2 dstStage = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
                    auto &gd = chunkBuildData->ommGeometryData[i];
                    if (gd.hasMicromap) {
                        dstStage |= VK_PIPELINE_STAGE_2_MICROMAP_BUILD_BIT_EXT;
                    }

                    bufferBarriers.push_back(vk::CommandBuffer::BufferMemoryBarrier{
                        .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                        .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                        .dstStageMask = dstStage,
                        .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                        .srcQueueFamilyIndex = secondaryQueueIndex,
                        .dstQueueFamilyIndex = secondaryQueueIndex,
                        .buffer = chunkBuildData->vertexBuffers[i],
                    });

                    bufferBarriers.push_back(vk::CommandBuffer::BufferMemoryBarrier{
                        .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                        .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                        .dstStageMask = dstStage,
                        .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                        .srcQueueFamilyIndex = secondaryQueueIndex,
                        .dstQueueFamilyIndex = secondaryQueueIndex,
                        .buffer = chunkBuildData->indexBuffers[i],
                    });

                    if (chunkBuildData->ommIndexBuffers[i] != nullptr) {
                        bufferBarriers.push_back(vk::CommandBuffer::BufferMemoryBarrier{
                            .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                            .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                            .dstStageMask = dstStage,
                            .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                            .srcQueueFamilyIndex = secondaryQueueIndex,
                            .dstQueueFamilyIndex = secondaryQueueIndex,
                            .buffer = chunkBuildData->ommIndexBuffers[i],
                        });
                    }

                    if (gd.hasMicromap) {
                        bufferBarriers.push_back(vk::CommandBuffer::BufferMemoryBarrier{
                            .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                            .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                            .dstStageMask = VK_PIPELINE_STAGE_2_MICROMAP_BUILD_BIT_EXT,
                            .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                            .srcQueueFamilyIndex = secondaryQueueIndex,
                            .dstQueueFamilyIndex = secondaryQueueIndex,
                            .buffer = gd.arrayBuffer,
                        });
                        bufferBarriers.push_back(vk::CommandBuffer::BufferMemoryBarrier{
                            .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                            .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                            .dstStageMask = VK_PIPELINE_STAGE_2_MICROMAP_BUILD_BIT_EXT,
                            .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                            .srcQueueFamilyIndex = secondaryQueueIndex,
                            .dstQueueFamilyIndex = secondaryQueueIndex,
                            .buffer = gd.descBuffer,
                        });
                    }
                }
            }
            worldAsyncBuffer->barriersBufferImage(bufferBarriers, {});

            // Build micromaps (before BLAS build)
            bool anyMicromaps = false;
            for (auto chunkBuildData : chunkBuildDataBatch->batchData) {
                for (int i = 0; i < chunkBuildData->geometryCount; i++) {
                    auto &gd = chunkBuildData->ommGeometryData[i];
                    if (!gd.hasMicromap) continue;
                    anyMicromaps = true;

                    VkMicromapBuildInfoEXT buildInfo{};
                    buildInfo.sType = VK_STRUCTURE_TYPE_MICROMAP_BUILD_INFO_EXT;
                    buildInfo.type = VK_MICROMAP_TYPE_OPACITY_MICROMAP_EXT;
                    buildInfo.flags = VK_BUILD_MICROMAP_PREFER_FAST_TRACE_BIT_EXT;
                    buildInfo.mode = VK_BUILD_MICROMAP_MODE_BUILD_EXT;
                    buildInfo.dstMicromap = gd.micromap;
                    buildInfo.data.deviceAddress = gd.arrayBuffer->bufferAddress();
                    buildInfo.triangleArray.deviceAddress = gd.descBuffer->bufferAddress();
                    buildInfo.triangleArrayStride = sizeof(VkMicromapTriangleEXT);
                    buildInfo.usageCountsCount = static_cast<uint32_t>(gd.descHistogram.size());
                    buildInfo.pUsageCounts = gd.descHistogram.data();
                    if (gd.micromapScratchBuffer) {
                        buildInfo.scratchData.deviceAddress = gd.micromapScratchBuffer->bufferAddress();
                    }

                    vkCmdBuildMicromapsEXT(worldAsyncBuffer->vkCommandBuffer(), 1, &buildInfo);
                }
            }

            // Barrier: micromap build → BLAS build
            if (anyMicromaps) {
                VkMemoryBarrier2 mmBarrier{};
                mmBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
                mmBarrier.srcStageMask = VK_PIPELINE_STAGE_2_MICROMAP_BUILD_BIT_EXT;
                mmBarrier.srcAccessMask = VK_ACCESS_2_MICROMAP_WRITE_BIT_EXT;
                mmBarrier.dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
                mmBarrier.dstAccessMask = VK_ACCESS_2_MICROMAP_READ_BIT_EXT;

                VkDependencyInfo depInfo{};
                depInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                depInfo.memoryBarrierCount = 1;
                depInfo.pMemoryBarriers = &mmBarrier;
                vkCmdPipelineBarrier2(worldAsyncBuffer->vkCommandBuffer(), &depInfo);
            }

            // Build BLAS
            std::vector<std::shared_ptr<vk::BLASBuilder>> builders;
            for (auto chunkBuildData : chunkBuildDataBatch->batchData) {
                builders.push_back(chunkBuildData->blasBuilder);
            }
            vk::BLASBuilder::batchSubmit(builders, worldAsyncBuffer);

            worldAsyncBuffer->end();

            VkSubmitInfo vkSubmitInfo = {};
            vkSubmitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            vkSubmitInfo.waitSemaphoreCount = 0;
            vkSubmitInfo.pWaitSemaphores = nullptr;
            vkSubmitInfo.pWaitDstStageMask = nullptr;
            vkSubmitInfo.commandBufferCount = 1;
            vkSubmitInfo.pCommandBuffers = &worldAsyncBuffer->vkCommandBuffer();
            vkSubmitInfo.signalSemaphoreCount = 0;
            vkSubmitInfo.pSignalSemaphores = nullptr;

            vkQueueSubmit(device->secondaryQueue(), 1, &vkSubmitInfo, fence->vkFence());

            freeFences_.pop();
            buildingFences_.push_back(fence);
            buildingBatches_.push_back(chunkBuildDataBatch);
        }
    }
}

uint32_t ChunkBuildScheduler::chunkBuildingBatchSize() {
    return chunkBuildingBatchSize_;
}

uint32_t ChunkBuildScheduler::chunkBuildingTotalBatches() {
    return chunkBuildingTotalBatches_;
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
        blas = chunkBuildData->blas;

        gc.collect(vertexBuffers);
        vertexBuffers = std::make_shared<std::vector<std::shared_ptr<vk::DeviceLocalBuffer>>>(
            std::move(chunkBuildData->vertexBuffers));

        gc.collect(indexBuffers);
        indexBuffers = std::make_shared<std::vector<std::shared_ptr<vk::DeviceLocalBuffer>>>(
            std::move(chunkBuildData->indexBuffers));
    } else {
        gc.collect(chunkBuildData->blas);

        gc.collect(std::make_shared<std::vector<std::shared_ptr<vk::DeviceLocalBuffer>>>(
            std::move(chunkBuildData->vertexBuffers)));

        gc.collect(std::make_shared<std::vector<std::shared_ptr<vk::DeviceLocalBuffer>>>(
            std::move(chunkBuildData->indexBuffers)));
    }

    allVertexCount = chunkBuildData->allVertexCount;
    allIndexCount = chunkBuildData->allIndexCount;
    geometryCount = chunkBuildData->geometryCount;
    geometryTypes = std::make_shared<std::vector<World::GeometryTypes>>(std::move(chunkBuildData->geometryTypes));
    vertices =
        std::make_shared<std::vector<std::vector<vk::VertexFormat::PBRTriangle>>>(std::move(chunkBuildData->vertices));
    indices = std::make_shared<std::vector<std::vector<uint32_t>>>(std::move(chunkBuildData->indices));
}

void Chunk1::invalidate() {
    auto framework = Renderer::instance().framework();
    auto &gc = framework->gc();

    lastUpdate = std::chrono::steady_clock::now();

    blasVersion = latestVersion++;

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
    ret->vertices = vertices;
    ret->indices = indices;

    return ret;
}

Chunks::Chunks(std::shared_ptr<Framework> framework) {
    importantBLASBuilders_ = std::make_shared<std::vector<std::shared_ptr<vk::BLASBuilder>>>();
}

void Chunks::reset(uint32_t numChunks) {
    std::unique_lock<std::recursive_mutex> lock(mutex_);

    auto framework = Renderer::instance().framework();
    auto device = framework->device();
    auto vma = framework->vma();
    vkQueueWaitIdle(device->mainVkQueue());
    vkQueueWaitIdle(device->secondaryQueue());

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
    uint32_t chunkBuildingTotalBatches = Renderer::instance().options.chunkBuildingTotalBatches;
    chunkBuildScheduler_ =
        ChunkBuildScheduler::create(queuedIndex_, chunks_, chunkBuildDatas_, mutex_, chunkPackedData_,
                                    chunkBuildingBatchSize, chunkBuildingTotalBatches);
}

void Chunks::resetScheduler() {
    std::unique_lock<std::recursive_mutex> lock(mutex_);

    if (chunkBuildScheduler_ == nullptr) return;

    chunkBuildScheduler_->waitAllBatchesFinish();

    uint32_t chunkBuildingBatchSize = Renderer::instance().options.chunkBuildingBatchSize;
    uint32_t chunkBuildingTotalBatches = Renderer::instance().options.chunkBuildingTotalBatches;
    chunkBuildScheduler_ =
        ChunkBuildScheduler::create(queuedIndex_, chunks_, chunkBuildDatas_, mutex_, chunkPackedData_,
                                    chunkBuildingBatchSize, chunkBuildingTotalBatches);
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
    chunks_[id]->invalidate();

    ChunkPackedData data = {
        .geometryCount = 0,
    };

    chunkPackedData_->uploadToBuffer(&data, sizeof(ChunkPackedData), id * sizeof(ChunkPackedData));
}

// maybe called async
void Chunks::queueChunkBuild(ChunkBuildTask task) {
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

    std::shared_ptr<ChunkBuildData> chunkBuildData = ChunkBuildData::create(
        task.id, task.x, task.y, task.z, chunks_[task.id]->latestVersion++, allVertexCount, allIndexCount,
        task.geometryCount, std::move(geometryTypes), std::move(vertices), std::move(indices));

    if (task.isImportant) {
        bool ommEnabled = device->hasOMM() && Renderer::options.ommEnabled;
        // Skip OMM entirely for important chunks when OMM is enabled — avoids
        // VK_NULL_HANDLE micromap in BLAS pNext which causes invisibility on some drivers.
        // The chunk is queued for async Phase 2 rebuild below.
        chunkBuildData->build(false, ommEnabled);
        for (int i = 0; i < chunkBuildData->geometryCount; i++) {
            Renderer::instance().buffers()->queueImportantWorldUpload(chunkBuildData->vertexBuffers[i],
                                                                      chunkBuildData->indexBuffers[i]);
            if (chunkBuildData->ommIndexBuffers[i] != nullptr) {
                Renderer::instance().buffers()->queueImportantWorldUpload(chunkBuildData->ommIndexBuffers[i],
                                                                          nullptr);
            }
        }
        importantBLASBuilders_->push_back(chunkBuildData->blasBuilder);

        // Copy geometry data BEFORE enqueue (enqueue moves them out)
        std::shared_ptr<ChunkBuildData> asyncRebuildData;
        if (ommEnabled) {
            asyncRebuildData = ChunkBuildData::create(
                task.id, task.x, task.y, task.z,
                chunks_[task.id]->latestVersion++, // higher version → will replace Phase 1 BLAS
                allVertexCount, allIndexCount, task.geometryCount,
                std::vector<World::GeometryTypes>(chunkBuildData->geometryTypes),
                std::vector<std::vector<vk::VertexFormat::PBRTriangle>>(chunkBuildData->vertices),
                std::vector<std::vector<uint32_t>>(chunkBuildData->indices));
        }

        chunks_[task.id]->enqueue(chunkBuildData);

        // Queue async Phase 2 rebuild with full OMM baking
        if (asyncRebuildData) {
            queuedIndex_.insert(task.id);
            chunkBuildDatas_[task.id] = asyncRebuildData;
        }

        ChunkPackedData data = {
            .geometryCount = chunkBuildData->geometryCount,
        };

        chunkPackedData_->uploadToBuffer(&data, sizeof(ChunkPackedData), chunkBuildData->id * sizeof(ChunkPackedData));
    } else {
        queuedIndex_.insert(task.id);
        chunkBuildDatas_[task.id] = chunkBuildData;
    }
}

bool Chunks::isChunkReady(int64_t id) {
    std::unique_lock<std::recursive_mutex> lock(mutex_);
    auto chunkRenderData = chunks_[id]->tryGetValid();
    return chunkRenderData->blas != nullptr;
}

void Chunks::setChunkLights(int64_t id, const std::vector<ChunkLightEntry> &lights) {
    std::unique_lock<std::recursive_mutex> lock(mutex_);
    if (id >= 0 && id < static_cast<int64_t>(chunks_.size()) && chunks_[id]) {
        chunks_[id]->lightSources = lights;
    }
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