#include "core/render/chunks.hpp"

#include "core/render/buffers.hpp"
#include "core/render/crash_ring_buffer.hpp"
#include "core/render/render_framework.hpp"
#include "core/render/renderer.hpp"
#include "core/render/textures.hpp"
#ifdef MCVR_ENABLE_OMM
#include "core/render/omm_baker.hpp"
#endif

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>

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

    // Displacement Tier 1: partition WORLD_SOLID quads with height maps into displaced AABBs
    bool useDisplacement = Renderer::options.displacementQuality == 1;
    vk::Data::TextureMapping *texMappingPtr = nullptr;
    if (useDisplacement) {
        auto buffers = Renderer::instance().buffers();
        auto texMappingBuf = buffers ? buffers->textureMappingBuffer() : nullptr;
        texMappingPtr = texMappingBuf ? static_cast<vk::Data::TextureMapping *>(texMappingBuf->mappedPtr()) : nullptr;
    }

    // For each WORLD_SOLID geometry, split indices into regular and displaced
    // filteredIndices[i] replaces indices[i] for displaced geometries (triangles without height maps)
    std::vector<std::vector<uint32_t>> filteredIndices(geometryCount);
    std::vector<bool> geometryFiltered(geometryCount, false);

    if (useDisplacement && texMappingPtr) {
        float heightScale = Renderer::options.pomHeightScale;

        for (int i = 0; i < geometryCount; i++) {
            if (geometryTypes[i] != World::WORLD_SOLID) continue;

            auto &idx = indices[i];
            auto &verts = vertices[i];
            std::vector<uint32_t> regularIdx;
            regularIdx.reserve(idx.size());

            // Process quads (6 indices = 2 triangles per quad)
            for (size_t t = 0; t + 5 < idx.size(); t += 6) {
                uint32_t i0 = idx[t], i1 = idx[t + 1], i2 = idx[t + 2];
                uint32_t i3 = idx[t + 3], i4 = idx[t + 4], i5 = idx[t + 5];

                auto &v0 = verts[i0];
                uint32_t texID = v0.textureID;
                bool displaced = false;

                if (texID < 4096) {
                    auto &entry = texMappingPtr->entries[texID];
                    if ((entry.properties & vk::Data::TEX_PROP_HAS_HEIGHT_MAP) && entry.normal >= 0) {
                        // Determine face axis from normal
                        glm::vec3 norm = v0.norm;
                        float ax = std::abs(norm.x), ay = std::abs(norm.y), az = std::abs(norm.z);
                        if (ax > 0.9f || ay > 0.9f || az > 0.9f) {
                            // Axis-aligned face — extract as displaced AABB
                            uint32_t faceAxis;
                            if (ax > 0.9f) faceAxis = (norm.x > 0) ? 0 : 1;
                            else if (ay > 0.9f) faceAxis = (norm.y > 0) ? 2 : 3;
                            else faceAxis = (norm.z > 0) ? 4 : 5;

                            // Collect unique vertices (quad = 4 unique verts from 6 indices)
                            std::array<uint32_t, 6> quadIdx = {i0, i1, i2, i3, i4, i5};
                            glm::vec3 posMin(1e9f), posMax(-1e9f);
                            glm::vec2 uvMin(1e9f), uvMax(-1e9f);
                            glm::vec4 avgColor(0);
                            int vertCount = 0;
                            std::array<glm::vec3, 4> uniquePos;
                            std::array<glm::vec2, 4> uniqueUV;
                            int uniqueCount = 0;

                            for (int q = 0; q < 6; q++) {
                                auto &vq = verts[quadIdx[q]];
                                bool isDup = false;
                                for (int u = 0; u < uniqueCount; u++) {
                                    if (glm::length(uniquePos[u] - vq.pos) < 0.001f) {
                                        isDup = true;
                                        break;
                                    }
                                }
                                if (!isDup && uniqueCount < 4) {
                                    uniquePos[uniqueCount] = vq.pos;
                                    uniqueUV[uniqueCount] = vq.textureUV;
                                    uniqueCount++;
                                }
                                posMin = glm::min(posMin, vq.pos);
                                posMax = glm::max(posMax, vq.pos);
                                uvMin = glm::min(uvMin, vq.textureUV);
                                uvMax = glm::max(uvMax, vq.textureUV);
                                avgColor += vq.colorLayer;
                                vertCount++;
                            }
                            avgColor /= float(vertCount);

                            if (uniqueCount == 4) {
                                // Build AABB: extend inward from face by heightScale
                                VkAabbPositionsKHR aabb;
                                glm::vec3 aabbMin = posMin;
                                glm::vec3 aabbMax = posMax;
                                // Extend along face normal direction (inward = opposite to normal)
                                if (faceAxis <= 1) { // ±X
                                    float ext = heightScale;
                                    aabbMin.x -= ext;
                                    aabbMax.x += ext;
                                } else if (faceAxis <= 3) { // ±Y
                                    float ext = heightScale;
                                    aabbMin.y -= ext;
                                    aabbMax.y += ext;
                                } else { // ±Z
                                    float ext = heightScale;
                                    aabbMin.z -= ext;
                                    aabbMax.z += ext;
                                }
                                aabb = {aabbMin.x, aabbMin.y, aabbMin.z, aabbMax.x, aabbMax.y, aabbMax.z};
                                displacedAABBs.push_back(aabb);

                                // Compute corner and edges for the face quad
                                // Corner = min UV vertex, edgeU/edgeV = edges from corner
                                // Find the vertex with smallest UV as the corner
                                int cornerIdx = 0;
                                for (int u = 1; u < 4; u++) {
                                    if (uniqueUV[u].x < uniqueUV[cornerIdx].x - 0.0001f ||
                                        (std::abs(uniqueUV[u].x - uniqueUV[cornerIdx].x) < 0.0001f &&
                                         uniqueUV[u].y < uniqueUV[cornerIdx].y)) {
                                        cornerIdx = u;
                                    }
                                }
                                glm::vec3 corner = uniquePos[cornerIdx];
                                glm::vec2 cornerUV = uniqueUV[cornerIdx];

                                // Find edges from corner to adjacent vertices
                                glm::vec3 edgeU(0), edgeV(0);
                                bool foundU = false, foundV = false;
                                for (int u = 0; u < 4; u++) {
                                    if (u == cornerIdx) continue;
                                    glm::vec2 dUV = uniqueUV[u] - cornerUV;
                                    if (!foundU && std::abs(dUV.x) > 0.0001f && std::abs(dUV.y) < 0.0001f) {
                                        edgeU = uniquePos[u] - corner;
                                        foundU = true;
                                    } else if (!foundV && std::abs(dUV.y) > 0.0001f && std::abs(dUV.x) < 0.0001f) {
                                        edgeV = uniquePos[u] - corner;
                                        foundV = true;
                                    }
                                }
                                // Fallback: if UV axes aren't perfectly aligned, use first two non-corner verts
                                if (!foundU || !foundV) {
                                    int idx2 = 0;
                                    for (int u = 0; u < 4; u++) {
                                        if (u == cornerIdx) continue;
                                        if (!foundU) { edgeU = uniquePos[u] - corner; foundU = true; }
                                        else if (!foundV) { edgeV = uniquePos[u] - corner; foundV = true; break; }
                                    }
                                }

                                vk::Data::DisplacedFaceData faceData{};
                                faceData.corner = corner;
                                faceData.faceAxis = faceAxis;
                                faceData.edgeU = edgeU;
                                faceData.heightScale = heightScale;
                                faceData.edgeV = edgeV;
                                faceData.textureID = texID;
                                faceData.normalTexID = entry.normal;
                                faceData.specularTexID = entry.specular;
                                faceData.uvMin = uvMin;
                                faceData.uvMax = uvMax;
                                faceData.colorLayer = avgColor;
                                faceData.emissiveBlockType = v0.emissiveBlockType;
                                faceData.properties = entry.properties;
                                faceData._pad0 = 0;
                                faceData._pad1 = 0;
                                displacedFaceData.push_back(faceData);

                                displaced = true;
                            }
                        }
                    }
                }

                if (!displaced) {
                    // Keep in regular triangle mesh
                    regularIdx.push_back(i0);
                    regularIdx.push_back(i1);
                    regularIdx.push_back(i2);
                    regularIdx.push_back(i3);
                    regularIdx.push_back(i4);
                    regularIdx.push_back(i5);
                }
            }

            // Handle remaining triangles (not part of a full quad)
            size_t remainder = idx.size() % 6;
            if (remainder > 0) {
                for (size_t t = idx.size() - remainder; t < idx.size(); t++) {
                    regularIdx.push_back(idx[t]);
                }
            }

            if (regularIdx.size() != idx.size()) {
                filteredIndices[i] = std::move(regularIdx);
                geometryFiltered[i] = true;

                // Rebuild index buffer with filtered indices
                if (!filteredIndices[i].empty()) {
                    indexBuffers[i] = vk::DeviceLocalBuffer::create(
                        vma, device, filteredIndices[i].size() * sizeof(uint32_t),
                        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
                    indexBuffers[i]->uploadToStagingBuffer(filteredIndices[i].data());
                }
            }
        }
    }

    blasBuilder = vk::BLASBuilder::create();
    auto blasGeometryBuilder = blasBuilder->beginGeometries();
    for (int i = 0; i < geometryCount; i++) {
        bool isOpaque = geometryTypes[i] == World::WORLD_SOLID;
        uint32_t indexCount = geometryFiltered[i] ? static_cast<uint32_t>(filteredIndices[i].size())
                                                  : static_cast<uint32_t>(indices[i].size());

        if (indexCount == 0) {
            // All faces were displaced — add placeholder geometry to maintain geometry indexing
            blasGeometryBuilder->definePlaceholderGeometry();
            continue;
        }

        if (ommIndexBuffers[i] != nullptr && !geometryFiltered[i]) {
            uint32_t numTriangles = indexCount / 3;
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
                vertexBuffers[i], vertices[i].size(), indexBuffers[i], indexCount,
                isOpaque);
        }
    }
    blasGeometryBuilder->endGeometries();
    blas = blasBuilder->defineBuildProperty(VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR)
               ->querySizeInfo(device)
               ->allocateBuffers(physicalDevice, device, vma)
               ->build(device);

    // Build displaced BLAS if we extracted any faces
    if (!displacedAABBs.empty()) {
        displacedAABBBuffer = vk::DeviceLocalBuffer::create(
            vma, device, displacedAABBs.size() * sizeof(VkAabbPositionsKHR),
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
        displacedAABBBuffer->uploadToStagingBuffer(displacedAABBs.data());

        displacedFaceDataBuffer = vk::DeviceLocalBuffer::create(
            vma, device, displacedFaceData.size() * sizeof(vk::Data::DisplacedFaceData),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
        displacedFaceDataBuffer->uploadToStagingBuffer(displacedFaceData.data());

        displacedBlasBuilder = vk::BLASBuilder::create();
        auto displacedGeomBuilder = displacedBlasBuilder->beginGeometries();
        displacedGeomBuilder->defineAABBGeometry(
            displacedAABBBuffer, static_cast<uint32_t>(displacedAABBs.size()), true);
        displacedGeomBuilder->endGeometries();
        displacedBlas = displacedBlasBuilder
                            ->defineBuildProperty(VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR)
                            ->querySizeInfo(device)
                            ->allocateBuffers(physicalDevice, device, vma)
                            ->build(device);
    }
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
        VkResult fenceResult = vkWaitForFences(device->vkDevice(), 1, &(*iterFence)->vkFence(), true, 0);
        if (fenceResult == VK_SUCCESS) {
            vkResetFences(device->vkDevice(), 1, &(*iterFence)->vkFence());
            freeFences_.push(*iterFence);

            for (auto chunkBuildData : (*iterBatch)->batchData) {
                if (chunkBuildData->id >= static_cast<int>(chunks_.size())) continue;
                chunks_[chunkBuildData->id]->enqueue(chunkBuildData);

                ChunkPackedData data = {
                    .geometryCount = chunkBuildData->geometryCount,
                };

                chunkPackedData_->uploadToBuffer(&data, sizeof(ChunkPackedData),
                                                 chunkBuildData->id * sizeof(ChunkPackedData));
            }

            iterFence = buildingFences_.erase(iterFence);
            iterBatch = buildingBatches_.erase(iterBatch);
        } else if (fenceResult != VK_TIMEOUT) {
            g_crashRing.record("chunkFenceFail", fenceResult);
            std::cout << "Chunk build fence failed with error: " << std::dec << fenceResult << std::endl;
            break;
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
        VkResult fenceResult = vkWaitForFences(device->vkDevice(), 1, &(*iterFence)->vkFence(), true, UINT64_MAX);
        if (fenceResult == VK_SUCCESS) {
            vkResetFences(device->vkDevice(), 1, &(*iterFence)->vkFence());
            freeFences_.push(*iterFence);

            for (auto chunkBuildData : (*iterBatch)->batchData) {
                if (chunkBuildData->id >= static_cast<int>(chunks_.size())) continue;
                chunks_[chunkBuildData->id]->enqueue(chunkBuildData);

                ChunkPackedData data = {
                    .geometryCount = chunkBuildData->geometryCount,
                };

                chunkPackedData_->uploadToBuffer(&data, sizeof(ChunkPackedData),
                                                 chunkBuildData->id * sizeof(ChunkPackedData));
            }

            iterFence = buildingFences_.erase(iterFence);
            iterBatch = buildingBatches_.erase(iterBatch);
        } else {
            g_crashRing.record("chunkWaitAllFail", fenceResult);
            std::cout << "Chunk build waitAll fence failed with error: " << std::dec << fenceResult << std::endl;
            break;
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

            // Build BLAS (including displaced BLAS if any)
            std::vector<std::shared_ptr<vk::BLASBuilder>> builders;
            for (auto chunkBuildData : chunkBuildDataBatch->batchData) {
                builders.push_back(chunkBuildData->blasBuilder);
                if (chunkBuildData->displacedBlasBuilder) {
                    builders.push_back(chunkBuildData->displacedBlasBuilder);
                }
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

    // Displacement: transfer displaced BLAS and face data
    gc.collect(displacedBlas);
    displacedBlas = chunkBuildData->displacedBlas;
    gc.collect(displacedFaceDataBuffer);
    displacedFaceDataBuffer = chunkBuildData->displacedFaceDataBuffer;
    displacedFaceCount = static_cast<uint32_t>(chunkBuildData->displacedFaceData.size());
    if (!chunkBuildData->displacedFaceData.empty()) {
        displacedFaceDataCPU = std::make_shared<std::vector<vk::Data::DisplacedFaceData>>(
            std::move(chunkBuildData->displacedFaceData));
    } else {
        displacedFaceDataCPU = nullptr;
    }
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

    gc.collect(displacedBlas);
    displacedBlas = nullptr;
    gc.collect(displacedFaceDataBuffer);
    displacedFaceDataBuffer = nullptr;
    displacedFaceCount = 0;
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
    if (id < 0 || id >= static_cast<int>(chunks_.size())) return;
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
        if (chunkBuildData->displacedBlasBuilder) {
            Renderer::instance().buffers()->queueImportantWorldUpload(chunkBuildData->displacedAABBBuffer,
                                                                      chunkBuildData->displacedFaceDataBuffer);
            importantBLASBuilders_->push_back(chunkBuildData->displacedBlasBuilder);
        }

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