#include "core/render/chunks.hpp"

#include "core/render/buffers.hpp"
#include "core/render/crash_ring_buffer.hpp"
#include "core/render/render_framework.hpp"
#include "core/render/renderer.hpp"
#include "core/render/tessellator.hpp"
#include "core/render/textures.hpp"
#ifdef MCVR_ENABLE_OMM
#include "core/render/omm_baker.hpp"
#endif

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <map>
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

ChunkBuildData::~ChunkBuildData() {
    for (auto &gd : ommGeometryData) {
        if (gd.micromap != VK_NULL_HANDLE && gd.device) {
            vkDestroyMicromapEXT(gd.device->vkDevice(), gd.micromap, nullptr);
            gd.micromap = VK_NULL_HANDLE;
        }
    }
}

void ChunkBuildData::build(bool allowMicromapBake, bool skipOMM, glm::vec3 cameraPos) {
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
            vk::DeviceLocalBuffer::create(vma, device, true, vertices[i].size() * sizeof(vk::VertexFormat::PBRTriangle),
                                          VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                                              VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        vertexBuffer->uploadToStagingBuffer(vertices[i].data());
        vertexBuffers.push_back(vertexBuffer);

        auto indexBuffer =
            vk::DeviceLocalBuffer::create(vma, device, true, indices[i].size() * sizeof(uint32_t),
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
                    vma, device, true, numTriangles * sizeof(int32_t),
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
                vma, device, true, numTriangles * sizeof(int32_t),
                VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                    VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                    VK_BUFFER_USAGE_MICROMAP_BUILD_INPUT_READ_ONLY_BIT_EXT);
            ommIdxBuffer->uploadToStagingBuffer(ommIndices.data());
            ommIndexBuffers.push_back(ommIdxBuffer);

            if (bakingDone && !mergedDescs.empty()) {
                auto &gd = ommGeometryData[i];

                // Upload OMM array data
                gd.arrayBuffer = vk::DeviceLocalBuffer::create(
                    vma, device, true, mergedArrayData.size(),
                    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                        VK_BUFFER_USAGE_MICROMAP_BUILD_INPUT_READ_ONLY_BIT_EXT);
                gd.arrayBuffer->uploadToStagingBuffer(mergedArrayData.data());

                // Upload OMM descriptor array (VkMicromapTriangleEXT)
                gd.descBuffer = vk::DeviceLocalBuffer::create(
                    vma, device, true, mergedDescs.size() * sizeof(VkMicromapTriangleEXT),
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
                vma, device, true, numTriangles * sizeof(int32_t),
                VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                    VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                    VK_BUFFER_USAGE_MICROMAP_BUILD_INPUT_READ_ONLY_BIT_EXT);
            ommIdxBuffer->uploadToStagingBuffer(ommIndices.data());
            ommIndexBuffers.push_back(ommIdxBuffer);
        } else {
            ommIndexBuffers.push_back(nullptr);
        }
    }

    // Geometric displacement: per-block method dispatch (DDA collects AABBs, Tessellation generates triangles)
    bool useDisplacement = Renderer::options.displacementQuality >= 1;
    vk::Data::TextureMapping *texMappingPtr = nullptr;
    vk::Data::MaterialClassMapping *matClassPtr = nullptr;
    auto tessTextures = textures; // reuse existing textures ptr from line 57
    if (useDisplacement) {
        auto buffers = Renderer::instance().buffers();
        auto texMappingBuf = buffers ? buffers->textureMappingBuffer() : nullptr;
        texMappingPtr = texMappingBuf ? static_cast<vk::Data::TextureMapping *>(texMappingBuf->mappedPtr()) : nullptr;
        auto matClassBuf = buffers ? buffers->materialClassMappingBuffer() : nullptr;
        matClassPtr = matClassBuf ? static_cast<vk::Data::MaterialClassMapping *>(matClassBuf->mappedPtr()) : nullptr;
    }

    if (useDisplacement && texMappingPtr) {
        uint32_t maxTessLevel = Renderer::options.tessMaxLevel;
        // Chunk center distance to camera for LOD tessellation level
        glm::vec3 chunkCenter(x * 16.0f + 8.0f, y * 16.0f + 8.0f, z * 16.0f + 8.0f);
        float distToCamera = glm::length(chunkCenter - cameraPos);

        for (int i = 0; i < geometryCount; i++) {
            if (geometryTypes[i] != World::WORLD_SOLID) continue;

            auto &idx = indices[i];
            auto &verts = vertices[i];

            // Collect tessellated geometry to append after the loop
            std::vector<vk::VertexFormat::PBRTriangle> newVerts;
            std::vector<uint32_t> newIndices;
            std::vector<uint32_t> keptIndices;
            keptIndices.reserve(idx.size());
            bool anyTessellated = false;

            // Pre-scan: collect block positions of all faces that will be displaced,
            // so we can detect coplanar neighbors and skip edge fade between them.
            struct FaceKey {
                int16_t x, y, z;
                uint8_t axis;
                bool operator<(const FaceKey &o) const {
                    if (x != o.x) return x < o.x;
                    if (y != o.y) return y < o.y;
                    if (z != o.z) return z < o.z;
                    return axis < o.axis;
                }
            };
            std::set<FaceKey> displacedFaceSet;
            struct SeamEdgeInfo {
                std::vector<float> u0, u1, v0, v1;
                glm::ivec3 uDir, vDir;
                uint32_t faceAxis;
                glm::vec3 norm;
                Tessellator::Input tessInput; // for interpVertex in seam wall generation
            };
            std::map<FaceKey, SeamEdgeInfo> seamEdgeData;
            for (size_t t = 0; t + 5 < idx.size(); t += 6) {
                auto &sv0 = verts[idx[t]];
                uint32_t stID = sv0.textureID;
                if (stID >= 4096) continue;
                auto &se = texMappingPtr->entries[stID];
                uint32_t sMT = (sv0.emissiveBlockType >> 8u) & 0xFFu;
                if (sMT == 0 || !matClassPtr) continue;
                uint32_t sMCI = sMT - 1;
                if (sMCI >= vk::Data::MAX_MATERIAL_CLASSES) continue;
                auto &smc = matClassPtr->entries[sMCI];
                float sPD = smc.pomDepth;
                bool sAP = (smc.flags & 0x8u) != 0;
                bool sLH = (se.properties & vk::Data::TEX_PROP_HAS_HEIGHT_MAP) && se.normal >= 0;
                bool sHH = sLH || (sAP && sPD > 0.0f);
                int sM = (smc.pomPacked0 >> 3) & 0x7;
                if (sM == 0 && sHH && sPD > 0.0f) sM = (Renderer::options.displacementQuality >= 1) ? 2 : 0;
                if (sM == 1) sM = 2;
                if (!sHH || (sM != 2 && sM != 3)) continue;
                glm::vec3 sn = sv0.norm;
                float sax = std::abs(sn.x), say = std::abs(sn.y), saz = std::abs(sn.z);
                if (sax <= 0.9f && say <= 0.9f && saz <= 0.9f) continue;
                uint8_t sFA;
                if (sax > 0.9f) sFA = (sn.x > 0) ? 0 : 1;
                else if (say > 0.9f) sFA = (sn.y > 0) ? 2 : 3;
                else sFA = (sn.z > 0) ? 4 : 5;
                // Block position = floor(face_center - 0.5 * normal)
                glm::vec3 sc = 0.25f * (verts[idx[t]].pos + verts[idx[t+1]].pos + verts[idx[t+2]].pos + verts[idx[t+4]].pos);
                glm::ivec3 sbp = glm::ivec3(glm::floor(sc - sn * 0.5f));
                displacedFaceSet.insert({(int16_t)sbp.x, (int16_t)sbp.y, (int16_t)sbp.z, sFA});
            }

            // Process quads (6 indices = 2 triangles per quad)
            for (size_t t = 0; t + 5 < idx.size(); t += 6) {
                uint32_t i0 = idx[t], i1 = idx[t + 1], i2 = idx[t + 2];
                uint32_t i3 = idx[t + 3], i4 = idx[t + 4], i5 = idx[t + 5];

                auto &v0 = verts[i0];
                uint32_t texID = v0.textureID;
                bool tessellated = false;

                if (texID < 4096) {
                    auto &entry = texMappingPtr->entries[texID];

                    // Check for LabPBR height map
                    bool hasLabPBRHeight = (entry.properties & vk::Data::TEX_PROP_HAS_HEIGHT_MAP) && entry.normal >= 0;

                    // Read material class for displacement method dispatch
                    bool isAutoPBR = false;
                    float pomDepth = 0.0f;
                    const vk::Data::MaterialClassEntry *matEntry = nullptr;
                    uint32_t materialType = (v0.emissiveBlockType >> 8u) & 0xFFu;
                    int dispMethod = 0; // 0=Off, 1=DDA, 2=Tessellation, 3=Hybrid, 4=CLAS
                    if (materialType > 0 && matClassPtr) {
                        uint32_t mcIdx = materialType - 1;
                        if (mcIdx < vk::Data::MAX_MATERIAL_CLASSES) {
                            matEntry = &matClassPtr->entries[mcIdx];
                            pomDepth = matEntry->pomDepth;
                            isAutoPBR = (matEntry->flags & 0x8u) != 0;
                            dispMethod = (matEntry->pomPacked0 >> 3) & 0x7; // bits 3-5
                        }
                    }

                    // Resolve method: if Off (0) but block has height data + depth, use global default
                    bool hasHeight = hasLabPBRHeight || (isAutoPBR && pomDepth > 0.0f);
                    if (dispMethod == 0 && hasHeight && pomDepth > 0.0f) {
                        // Global default maps to tessellation (DDA disabled pending VRAM budgeting)
                        dispMethod = (Renderer::options.displacementQuality >= 1) ? 2 : 0;
                    }
                    // Force DDA→Tessellation until VRAM budget + distance culling are implemented
                    if (dispMethod == 1) dispMethod = 2;

                    // Displace when method is Tessellation(2) or Hybrid(3)
                    bool shouldDisplace = hasHeight && (dispMethod == 2 || dispMethod == 3);

                    if (shouldDisplace) {
                        // Common setup: face axis, UV bounds, texture data
                        glm::vec3 norm = v0.norm;
                        float ax = std::abs(norm.x), ay = std::abs(norm.y), az = std::abs(norm.z);
                        if (ax > 0.9f || ay > 0.9f || az > 0.9f) {
                            uint32_t faceAxis;
                            if (ax > 0.9f) faceAxis = (norm.x > 0) ? 0 : 1;
                            else if (ay > 0.9f) faceAxis = (norm.y > 0) ? 2 : 3;
                            else faceAxis = (norm.z > 0) ? 4 : 5;

                            glm::vec2 uvMin(1e9f), uvMax(-1e9f);
                            std::array<uint32_t, 6> quadIdx = {i0, i1, i2, i3, i4, i5};
                            for (int q = 0; q < 6; q++) {
                                uvMin = glm::min(uvMin, verts[quadIdx[q]].textureUV);
                                uvMax = glm::max(uvMax, verts[quadIdx[q]].textureUV);
                            }

                            // texRes = tile width in texels, NOT atlas width
                            auto *albedoRGBA = tessTextures ? tessTextures->getTextureRGBAData(texID) : nullptr;
                            uint32_t texRes = 16;
                            if (albedoRGBA && albedoRGBA->width > 0) {
                                // Compute tile width from UV span * atlas width
                                float uvSpanX = uvMax.x - uvMin.x;
                                texRes = std::max(2u, static_cast<uint32_t>(uvSpanX * albedoRGBA->width + 0.5f));
                            }

                            const Textures::TextureRGBAData *normalRGBA = nullptr;
                            if (hasLabPBRHeight && entry.normal >= 0 && tessTextures) {
                                normalRGBA = tessTextures->getTextureRGBAData(static_cast<uint32_t>(entry.normal));
                            }

                            float heightScale = std::clamp(
                                (pomDepth > 0.0f) ? pomDepth : Renderer::options.pomHeightScale,
                                0.0f, 0.5f); // Cap at 0.5 blocks
                            uint32_t i4v = idx[t + 4]; // 4th unique vertex

                            // Compute per-edge fade mask: check 4 in-plane neighbors
                            // for coplanar displaced faces. If a neighbor exists, skip
                            // fading that edge so adjacent blocks blend seamlessly.
                            // Shared by both tessellation and DDA paths.
                            uint32_t fadeEdgeMask = 0xF; // default: fade all
                            {
                                glm::vec3 fc = 0.25f * (verts[i0].pos + verts[i1].pos + verts[i2].pos + verts[i4v].pos);
                                glm::ivec3 bp = glm::ivec3(glm::floor(fc - norm * 0.5f));

                                const vk::VertexFormat::PBRTriangle *qv[4] = {&verts[i0], &verts[i1], &verts[i2], &verts[i4v]};
                                int mi = 0;
                                for (int qi = 1; qi < 4; qi++) {
                                    if (qv[qi]->textureUV.x < qv[mi]->textureUV.x - 0.0001f ||
                                        (std::abs(qv[qi]->textureUV.x - qv[mi]->textureUV.x) < 0.0001f &&
                                         qv[qi]->textureUV.y < qv[mi]->textureUV.y))
                                        mi = qi;
                                }
                                glm::vec3 cPos = qv[mi]->pos;
                                glm::vec2 cUV = qv[mi]->textureUV;
                                glm::vec3 dirU(0), dirV(0);
                                for (int qi = 0; qi < 4; qi++) {
                                    if (qi == mi) continue;
                                    glm::vec2 dUV = qv[qi]->textureUV - cUV;
                                    if (std::abs(dUV.x) > 0.0001f && std::abs(dUV.y) < 0.0001f && glm::length(dirU) < 0.001f)
                                        dirU = glm::normalize(qv[qi]->pos - cPos);
                                    else if (std::abs(dUV.y) > 0.0001f && std::abs(dUV.x) < 0.0001f && glm::length(dirV) < 0.001f)
                                        dirV = glm::normalize(qv[qi]->pos - cPos);
                                }

                                auto roundDir = [](glm::vec3 d) -> glm::ivec3 {
                                    glm::ivec3 r(0);
                                    float ax = std::abs(d.x), ay = std::abs(d.y), az = std::abs(d.z);
                                    if (ax >= ay && ax >= az) r.x = (d.x > 0) ? 1 : -1;
                                    else if (ay >= ax && ay >= az) r.y = (d.y > 0) ? 1 : -1;
                                    else r.z = (d.z > 0) ? 1 : -1;
                                    return r;
                                };
                                glm::ivec3 uDir = roundDir(dirU);
                                glm::ivec3 vDir = roundDir(dirV);

                                glm::ivec3 nU0 = bp - uDir, nU1 = bp + uDir;
                                glm::ivec3 nV0 = bp - vDir, nV1 = bp + vDir;
                                if (displacedFaceSet.count({(int16_t)nU0.x, (int16_t)nU0.y, (int16_t)nU0.z, (uint8_t)faceAxis}))
                                    fadeEdgeMask &= ~0x1u;
                                if (displacedFaceSet.count({(int16_t)nU1.x, (int16_t)nU1.y, (int16_t)nU1.z, (uint8_t)faceAxis}))
                                    fadeEdgeMask &= ~0x2u;
                                if (displacedFaceSet.count({(int16_t)nV0.x, (int16_t)nV0.y, (int16_t)nV0.z, (uint8_t)faceAxis}))
                                    fadeEdgeMask &= ~0x4u;
                                if (displacedFaceSet.count({(int16_t)nV1.x, (int16_t)nV1.y, (int16_t)nV1.z, (uint8_t)faceAxis}))
                                    fadeEdgeMask &= ~0x8u;
                            }

                            // --- Tessellation path (method 2 or 3) ---
                            if (dispMethod == 2 || dispMethod == 3) {
                                // Use chunk-local camera distance: if global cameraPos is unset (0,0,0 default),
                                // distToCamera is wildly wrong. Fall back to max level until camera is valid.
                                uint32_t tessLevel;
                                if (distToCamera > 500.0f) {
                                    // Camera position likely uninitialized — use max tessellation
                                    tessLevel = std::clamp(std::min(texRes, maxTessLevel), 2u, 32u);
                                } else {
                                    tessLevel = Tessellator::computeTessLevel(
                                        distToCamera, texRes, maxTessLevel,
                                        Renderer::options.tessNearDist, Renderer::options.tessMidDist, Renderer::options.tessFarDist);
                                }

                                // If lumMin/lumMax are uninitialized defaults (0/1), compute
                                // the tile's actual luminance range for proper height normalization
                                vk::Data::MaterialClassEntry correctedMat{};
                                const vk::Data::MaterialClassEntry *effectiveMat = matEntry;
                                if (matEntry && albedoRGBA && isAutoPBR &&
                                    matEntry->lumMin < 0.001f && matEntry->lumMax > 0.99f) {
                                    correctedMat = *matEntry;
                                    float tileMin = 1.0f, tileMax = 0.0f;
                                    uint32_t tw = albedoRGBA->width, th = albedoRGBA->height;
                                    int x0 = static_cast<int>(uvMin.x * tw);
                                    int x1 = static_cast<int>(uvMax.x * tw);
                                    int y0 = static_cast<int>(uvMin.y * th);
                                    int y1 = static_cast<int>(uvMax.y * th);
                                    for (int py = y0; py < y1 && py < static_cast<int>(th); py++) {
                                        for (int px = x0; px < x1 && px < static_cast<int>(tw); px++) {
                                            size_t pi = (py * tw + px) * 4;
                                            float r = albedoRGBA->rgba[pi] / 255.0f;
                                            float g = albedoRGBA->rgba[pi+1] / 255.0f;
                                            float b = albedoRGBA->rgba[pi+2] / 255.0f;
                                            float lum = r * 0.2627f + g * 0.6780f + b * 0.0593f;
                                            tileMin = std::min(tileMin, lum);
                                            tileMax = std::max(tileMax, lum);
                                        }
                                    }
                                    if (tileMax > tileMin + 1e-4f) {
                                        correctedMat.lumMin = tileMin;
                                        correctedMat.lumMax = tileMax;
                                    }
                                    effectiveMat = &correctedMat;
                                }

                                Tessellator::Input tessInput{};
                                tessInput.v0 = &verts[i0];
                                tessInput.v1 = &verts[i1];
                                tessInput.v2 = &verts[i2];
                                tessInput.v3 = &verts[i4v];
                                tessInput.faceAxis = faceAxis;
                                tessInput.normalRGBA = normalRGBA;
                                tessInput.albedoRGBA = albedoRGBA;
                                tessInput.material = effectiveMat;
                                tessInput.hasLabPBRHeight = hasLabPBRHeight;
                                tessInput.isAutoPBR = isAutoPBR;
                                tessInput.uvMinX = uvMin.x;
                                tessInput.uvMinY = uvMin.y;
                                tessInput.uvMaxX = uvMax.x;
                                tessInput.uvMaxY = uvMax.y;
                                tessInput.tessLevel = tessLevel;
                                tessInput.heightScale = heightScale;
                                tessInput.fadeEdgeMask = fadeEdgeMask;

                                auto tessOutput = Tessellator::tessellate(tessInput);
                                if (!tessOutput.vertices.empty()) {
                                    uint32_t baseVertex = static_cast<uint32_t>(verts.size()) + static_cast<uint32_t>(newVerts.size());
                                    for (uint32_t ti : tessOutput.indices) {
                                        newIndices.push_back(baseVertex + ti);
                                    }
                                    newVerts.insert(newVerts.end(), tessOutput.vertices.begin(), tessOutput.vertices.end());
                                    tessellated = true;
                                    anyTessellated = true;

                                    // Store edge data for seam stitching post-pass
                                    glm::vec3 fc2 = 0.25f * (verts[i0].pos + verts[i1].pos + verts[i2].pos + verts[i4v].pos);
                                    glm::ivec3 bp2 = glm::ivec3(glm::floor(fc2 - norm * 0.5f));
                                    // Recompute UV directions for seam wall generation
                                    const vk::VertexFormat::PBRTriangle *qv2[4] = {&verts[i0], &verts[i1], &verts[i2], &verts[i4v]};
                                    int mi2 = 0;
                                    for (int qi = 1; qi < 4; qi++)
                                        if (qv2[qi]->textureUV.x < qv2[mi2]->textureUV.x - 0.0001f ||
                                            (std::abs(qv2[qi]->textureUV.x - qv2[mi2]->textureUV.x) < 0.0001f &&
                                             qv2[qi]->textureUV.y < qv2[mi2]->textureUV.y))
                                            mi2 = qi;
                                    glm::vec3 dirU2(0), dirV2(0);
                                    for (int qi = 0; qi < 4; qi++) {
                                        if (qi == mi2) continue;
                                        glm::vec2 dUV2 = qv2[qi]->textureUV - qv2[mi2]->textureUV;
                                        if (std::abs(dUV2.x) > 0.0001f && std::abs(dUV2.y) < 0.0001f && glm::length(dirU2) < 0.001f)
                                            dirU2 = glm::normalize(qv2[qi]->pos - qv2[mi2]->pos);
                                        else if (std::abs(dUV2.y) > 0.0001f && std::abs(dUV2.x) < 0.0001f && glm::length(dirV2) < 0.001f)
                                            dirV2 = glm::normalize(qv2[qi]->pos - qv2[mi2]->pos);
                                    }
                                    auto roundDir2 = [](glm::vec3 d) -> glm::ivec3 {
                                        glm::ivec3 r(0);
                                        float ax = std::abs(d.x), ay = std::abs(d.y), az = std::abs(d.z);
                                        if (ax >= ay && ax >= az) r.x = (d.x > 0) ? 1 : -1;
                                        else if (ay >= ax && ay >= az) r.y = (d.y > 0) ? 1 : -1;
                                        else r.z = (d.z > 0) ? 1 : -1;
                                        return r;
                                    };
                                    FaceKey fk{(int16_t)bp2.x, (int16_t)bp2.y, (int16_t)bp2.z, (uint8_t)faceAxis};
                                    seamEdgeData[fk] = {std::move(tessOutput.edgeDispU0), std::move(tessOutput.edgeDispU1),
                                                        std::move(tessOutput.edgeDispV0), std::move(tessOutput.edgeDispV1),
                                                        roundDir2(dirU2), roundDir2(dirV2), faceAxis, norm, tessInput};
                                }
                            }

                            // --- DDA path (method 1) ---
                            if (dispMethod == 1) {
                                // Map quad corners by UV: find min-UV corner, U-neighbor, V-neighbor
                                const auto *qVerts = &verts[0];
                                uint32_t qIdx[4] = {i0, i1, i2, i4v};
                                // Find min-UV corner
                                int minI = 0;
                                for (int qi = 1; qi < 4; qi++) {
                                    auto &a = qVerts[qIdx[qi]].textureUV, &b = qVerts[qIdx[minI]].textureUV;
                                    if (a.x < b.x - 0.0001f || (std::abs(a.x - b.x) < 0.0001f && a.y < b.y))
                                        minI = qi;
                                }
                                glm::vec3 cornerPos = qVerts[qIdx[minI]].pos;
                                glm::vec3 edgeU(0), edgeV(0);
                                glm::vec2 cornerUV = qVerts[qIdx[minI]].textureUV;
                                for (int qi = 0; qi < 4; qi++) {
                                    if (qi == minI) continue;
                                    glm::vec2 dUV = qVerts[qIdx[qi]].textureUV - cornerUV;
                                    if (std::abs(dUV.x) > 0.0001f && std::abs(dUV.y) < 0.0001f && glm::length(edgeU) < 0.001f)
                                        edgeU = qVerts[qIdx[qi]].pos - cornerPos;
                                    else if (std::abs(dUV.y) > 0.0001f && std::abs(dUV.x) < 0.0001f && glm::length(edgeV) < 0.001f)
                                        edgeV = qVerts[qIdx[qi]].pos - cornerPos;
                                }
                                // Fallback: if UV mapping failed, assign remaining edges
                                if (glm::length(edgeU) < 0.001f || glm::length(edgeV) < 0.001f) {
                                    for (int qi = 0; qi < 4; qi++) {
                                        if (qi == minI) continue;
                                        glm::vec3 e = qVerts[qIdx[qi]].pos - cornerPos;
                                        if (glm::length(edgeU) < 0.001f) edgeU = e;
                                        else if (glm::length(edgeV) < 0.001f) edgeV = e;
                                    }
                                }

                                // Compute AABB: expand along negative face normal by heightScale
                                static const glm::vec3 FACE_NORMALS[6] = {
                                    {1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}};
                                glm::vec3 fn = FACE_NORMALS[faceAxis];
                                glm::vec3 p[4];
                                for (int qi = 0; qi < 4; qi++) p[qi] = qVerts[qIdx[qi]].pos;
                                glm::vec3 bmin(1e9f), bmax(-1e9f);
                                for (int qi = 0; qi < 4; qi++) {
                                    bmin = glm::min(bmin, p[qi]);
                                    bmax = glm::max(bmax, p[qi]);
                                    glm::vec3 displaced = p[qi] - fn * heightScale;
                                    bmin = glm::min(bmin, displaced);
                                    bmax = glm::max(bmax, displaced);
                                }
                                // Small epsilon expansion to avoid precision issues
                                bmin -= glm::vec3(0.001f);
                                bmax += glm::vec3(0.001f);

                                VkAabbPositionsKHR aabb{bmin.x, bmin.y, bmin.z, bmax.x, bmax.y, bmax.z};
                                displacedAABBs.push_back(aabb);

                                // Build DisplacedFaceData
                                vk::Data::DisplacedFaceData fd{};
                                fd.corner = cornerPos;
                                fd.faceAxis = faceAxis;
                                fd.edgeU = edgeU;
                                fd.heightScale = heightScale;
                                fd.edgeV = edgeV;
                                fd.textureID = texID;
                                fd.normalTexID = entry.normal;
                                fd.specularTexID = entry.specular;
                                fd.uvMin = uvMin;
                                fd.uvMax = uvMax;
                                fd.pomPacked0 = matEntry ? matEntry->pomPacked0 : 0;
                                uint32_t mcIdx = (materialType > 0) ? (materialType - 1) : 0;
                                fd.materialClassIdx = mcIdx;
                                fd.emissiveBlockType = v0.emissiveBlockType;
                                fd.properties = entry.properties;
                                fd.lumMin = matEntry ? matEntry->lumMin : 0.0f;
                                fd.lumMax = matEntry ? matEntry->lumMax : 1.0f;
                                fd.colorLayer = v0.colorLayer;
                                fd.pomPacked1 = matEntry ? matEntry->pomPacked1 : 0;
                                fd.pomPacked2 = matEntry ? matEntry->pomPacked2 : 0;
                                fd.flags = matEntry ? matEntry->flags : 0;
                                fd.fadeEdgeMask = fadeEdgeMask;
                                displacedFaceData.push_back(fd);

                                tessellated = true; // consumed — don't keep original quad
                                anyTessellated = true;
                            }
                        }
                    }
                }

                if (!tessellated) {
                    keptIndices.push_back(i0);
                    keptIndices.push_back(i1);
                    keptIndices.push_back(i2);
                    keptIndices.push_back(i3);
                    keptIndices.push_back(i4);
                    keptIndices.push_back(i5);
                }
            }

            // Handle remaining triangles (not part of a full quad)
            size_t remainder = idx.size() % 6;
            if (remainder > 0) {
                for (size_t t = idx.size() - remainder; t < idx.size(); t++) {
                    keptIndices.push_back(idx[t]);
                }
            }

            // Post-pass: generate seam walls between adjacent displaced faces
            // For each face with a coplanar neighbor (fadeEdgeMask bit cleared),
            // compare edge displacements and add wall quads where they differ.
            if (!seamEdgeData.empty()) {
                for (auto &[fk, info] : seamEdgeData) {
                    uint32_t N = static_cast<uint32_t>(info.u0.size());
                    if (N == 0) continue;
                    float invN = 1.0f / static_cast<float>(N);
                    glm::vec3 fN3 = -info.norm; // displacement direction

                    // UV-ordered corners for correct bilinear interpolation
                    auto q = Tessellator::mapCornersPublic(info.tessInput);
                    if (!q.c00 || !q.c10 || !q.c01 || !q.c11) continue;

                    // Check each edge direction for coplanar neighbor
                    auto checkEdge = [&](const std::vector<float> &myEdge, glm::ivec3 neighborDir,
                                         bool isU, bool isMax) {
                        glm::ivec3 nPos = glm::ivec3(fk.x, fk.y, fk.z) + neighborDir;
                        FaceKey nk{(int16_t)nPos.x, (int16_t)nPos.y, (int16_t)nPos.z, fk.axis};
                        auto nit = seamEdgeData.find(nk);
                        if (nit == seamEdgeData.end()) return;

                        // Get the neighbor's OPPOSITE edge
                        const std::vector<float> &theirEdge = isU
                            ? (isMax ? nit->second.u0 : nit->second.u1)
                            : (isMax ? nit->second.v0 : nit->second.v1);

                        uint32_t nN = static_cast<uint32_t>(theirEdge.size());
                        uint32_t minN = std::min(N, nN);

                        for (uint32_t k = 0; k < minN; k++) {
                            float dA = myEdge[k], dB = theirEdge[k];
                            if (std::abs(dA - dB) < 1e-5f) continue;

                            // Generate seam wall at the shared boundary
                            float dNear = std::min(dA, dB), dFar = std::max(dA, dB);
                            float p0 = k * invN, p1 = (k + 1) * invN;
                            float edgeParam = isMax ? 1.0f : 0.0f;

                            // Create 4 wall vertices (double-sided = 8 verts, 4 tris)
                            for (int side = 0; side < 2; side++) {
                                uint32_t base = static_cast<uint32_t>(verts.size()) + static_cast<uint32_t>(newVerts.size());
                                vk::VertexFormat::PBRTriangle wv[4];

                                float eu0, ev0, eu1, ev1;
                                if (isU) {
                                    eu0 = edgeParam; ev0 = p0;
                                    eu1 = edgeParam; ev1 = p1;
                                } else {
                                    eu0 = p0; ev0 = edgeParam;
                                    eu1 = p1; ev1 = edgeParam;
                                }

                                // Use UV-ordered corners (mapCorners) for correct interpolation
                                Tessellator::interpVertexPublic(wv[0], q, eu0, ev0);
                                Tessellator::interpVertexPublic(wv[1], q, eu1, ev1);
                                wv[2] = wv[1]; wv[3] = wv[0];

                                // Offset top pair by dNear, bottom pair by dFar
                                wv[0].pos += fN3 * dNear; wv[0].postBase += fN3 * dNear;
                                wv[1].pos += fN3 * dNear; wv[1].postBase += fN3 * dNear;
                                wv[2].pos += fN3 * dFar;  wv[2].postBase += fN3 * dFar;
                                wv[3].pos += fN3 * dFar;  wv[3].postBase += fN3 * dFar;

                                glm::vec3 wallNorm = glm::normalize(glm::cross(
                                    wv[1].pos - wv[0].pos, fN3));
                                if (side == 1) wallNorm = -wallNorm;
                                for (int vi = 0; vi < 4; vi++) wv[vi].norm = wallNorm;

                                newVerts.push_back(wv[0]); newVerts.push_back(wv[1]);
                                newVerts.push_back(wv[2]); newVerts.push_back(wv[3]);

                                if (side == 0) {
                                    newIndices.push_back(base); newIndices.push_back(base+1); newIndices.push_back(base+2);
                                    newIndices.push_back(base); newIndices.push_back(base+2); newIndices.push_back(base+3);
                                } else {
                                    newIndices.push_back(base); newIndices.push_back(base+2); newIndices.push_back(base+1);
                                    newIndices.push_back(base); newIndices.push_back(base+3); newIndices.push_back(base+2);
                                }
                            }
                        }
                    };

                    // Only generate seam walls for edges where fade was DISABLED (coplanar neighbor)
                    if (!(info.tessInput.fadeEdgeMask & 0x1)) checkEdge(info.u0, -info.uDir, true, false);
                    if (!(info.tessInput.fadeEdgeMask & 0x2)) checkEdge(info.u1, info.uDir, true, true);
                    if (!(info.tessInput.fadeEdgeMask & 0x4)) checkEdge(info.v0, -info.vDir, false, false);
                    if (!(info.tessInput.fadeEdgeMask & 0x8)) checkEdge(info.v1, info.vDir, false, true);
                }
                anyTessellated = true; // seam walls add geometry
            }

            if (anyTessellated) {
                // Append new tessellated vertices to the geometry's vertex array
                verts.insert(verts.end(), newVerts.begin(), newVerts.end());

                // Combine kept + tessellated indices
                keptIndices.insert(keptIndices.end(), newIndices.begin(), newIndices.end());
                idx = std::move(keptIndices);

                // Rebuild vertex and index buffers with expanded data
                vertexBuffers[i] = vk::DeviceLocalBuffer::create(
                    vma, device, true, verts.size() * sizeof(vk::VertexFormat::PBRTriangle),
                    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
                vertexBuffers[i]->uploadToStagingBuffer(verts.data());

                indexBuffers[i] = vk::DeviceLocalBuffer::create(
                    vma, device, true, idx.size() * sizeof(uint32_t),
                    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
                indexBuffers[i]->uploadToStagingBuffer(idx.data());
            }
        }
    }

    blasBuilder = vk::BLASBuilder::create();
    auto blasGeometryBuilder = blasBuilder->beginGeometries();
    for (int i = 0; i < geometryCount; i++) {
        bool isOpaque = geometryTypes[i] == World::WORLD_SOLID;
        uint32_t indexCount = static_cast<uint32_t>(indices[i].size());

        if (indexCount == 0) {
            blasGeometryBuilder->definePlaceholderGeometry();
            continue;
        }

        if (ommIndexBuffers[i] != nullptr) {
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
    blas = blasBuilder->defineBuildProperty(VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR |
                                           VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR)
               ->querySizeInfo(device)
               ->allocateBuffers(physicalDevice, device, vma)
               ->build(device);

    // Build separate AABB BLAS for DDA displaced faces
    if (!displacedAABBs.empty()) {
        displacedAABBBuffer = vk::DeviceLocalBuffer::create(
            vma, device, true, displacedAABBs.size() * sizeof(VkAabbPositionsKHR),
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
        displacedAABBBuffer->uploadToStagingBuffer(displacedAABBs.data());

        displacedFaceDataBuffer = vk::DeviceLocalBuffer::create(
            vma, device, true, displacedFaceData.size() * sizeof(vk::Data::DisplacedFaceData),
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
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

    // Cache the face count before releaseHostGeometry() clears the vector
    displacedFaceCount = static_cast<uint32_t>(displacedFaceData.size());
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

    // Also release CPU-side displacement data (GPU buffers are already uploaded)
    displacedAABBs.clear();
    displacedAABBs.shrink_to_fit();
    displacedFaceCount = static_cast<uint32_t>(displacedFaceData.size());
    displacedFaceData.clear();
    displacedFaceData.shrink_to_fit();
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

    for (int i = 0; i < std::min((size_t)maxBatchSize, queuedIndices.size()); i++) {
        auto iter = queuedIndexSet.find(queuedIndices[i]);
        if (iter != queuedIndexSet.end()) { queuedIndexSet.erase(iter); }

        auto data = chunkBuildDatas[queuedIndices[i]];
        data->build(true, false, cameraPos);
        // data->releaseHostGeometry(); // DISABLED: investigating staging buffer crash
        batchData.push_back(data);
    }
}

ChunkBuildDataBatch::~ChunkBuildDataBatch() {
    if (compactionQueryPool != VK_NULL_HANDLE && queryDevice) {
        vkDestroyQueryPool(queryDevice->vkDevice(), compactionQueryPool, nullptr);
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
    auto asyncPool = framework->asyncCommandPool();

    uint32_t numFences = chunkBuildingTotalBatches_;
    for (int i = 0; i < numFences; i++) {
        freeFences_.push(vk::Fence::create(device));
        freeCmdBuffers_.push(vk::CommandBuffer::create(device, asyncPool));
    }
}

void ChunkBuildScheduler::tryCheckBatchesFinish() {
    auto framework = Renderer::instance().framework();
    auto device = framework->device();

    std::unique_lock<std::recursive_mutex> lock(mutex_);
    auto iterFence = buildingFences_.begin();
    auto iterCmd = buildingCmdBuffers_.begin();
    auto iterBatch = buildingBatches_.begin();
    for (; iterFence != buildingFences_.end() && iterBatch != buildingBatches_.end();) {
        VkResult fenceResult = vkWaitForFences(device->vkDevice(), 1, &(*iterFence)->vkFence(), true, 0);
        if (fenceResult == VK_SUCCESS) {
            vkResetFences(device->vkDevice(), 1, &(*iterFence)->vkFence());
            freeFences_.push(*iterFence);
            freeCmdBuffers_.push(*iterCmd);

            // Read BLAS compaction query results (measurement only — no compaction yet)
            auto &batch = *iterBatch;
            if (batch->compactionQueryPool != VK_NULL_HANDLE && batch->compactionQueryCount > 0) {
                std::vector<VkDeviceSize> compactedSizes(batch->compactionQueryCount);
                VkResult qr = vkGetQueryPoolResults(
                    device->vkDevice(), batch->compactionQueryPool, 0, batch->compactionQueryCount,
                    compactedSizes.size() * sizeof(VkDeviceSize), compactedSizes.data(),
                    sizeof(VkDeviceSize), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);

                if (qr == VK_SUCCESS) {
                    static uint64_t totalOriginal = 0, totalCompacted = 0;
                    static uint32_t totalCount = 0;
                    uint32_t qi = 0;
                    for (auto &cbd : batch->batchData) {
                        if (cbd->blas && qi < batch->compactionQueryCount) {
                            uint64_t original = cbd->blas->blasBuffer()->size();
                            uint64_t compacted = compactedSizes[qi];
                            totalOriginal += original;
                            totalCompacted += compacted;
                            totalCount++;
                            qi++;
                        }
                    }
                    // Log every 50 chunks
                    if (totalCount > 0 && totalCount % 50 == 0) {
                        float pct = totalOriginal > 0 ? 100.0f * (1.0f - (float)totalCompacted / totalOriginal) : 0;
                        std::cout << "[Compaction] " << totalCount << " BLASes: "
                                  << (totalOriginal / 1024) << " KB original, "
                                  << (totalCompacted / 1024) << " KB compacted, "
                                  << pct << "% savings" << std::endl;
                    }
                }
            }

            for (auto chunkBuildData : batch->batchData) {
                if (chunkBuildData->id >= static_cast<int>(chunks_.size())) continue;
                chunks_[chunkBuildData->id]->enqueue(chunkBuildData);

                ChunkPackedData data = {
                    .geometryCount = chunkBuildData->geometryCount,
                };

                chunkPackedData_->uploadToBuffer(&data, sizeof(ChunkPackedData),
                                                 chunkBuildData->id * sizeof(ChunkPackedData));
            }

            iterFence = buildingFences_.erase(iterFence);
            iterCmd = buildingCmdBuffers_.erase(iterCmd);
            iterBatch = buildingBatches_.erase(iterBatch);
        } else if (fenceResult == VK_TIMEOUT) {
            ++iterFence;
            ++iterCmd;
            ++iterBatch;
        } else {
            g_crashRing.record("chunkFenceFail", fenceResult);
            std::cout << "Chunk build fence failed with error: " << std::dec << fenceResult << std::endl;
            // Device lost — all remaining fences are poison, drain them
            buildingFences_.clear();
            buildingCmdBuffers_.clear();
            buildingBatches_.clear();
            break;
        }
    }
}

void ChunkBuildScheduler::waitAllBatchesFinish() {
    auto framework = Renderer::instance().framework();
    auto device = framework->device();

    std::unique_lock<std::recursive_mutex> lock(mutex_);
    auto iterFence = buildingFences_.begin();
    auto iterCmd = buildingCmdBuffers_.begin();
    auto iterBatch = buildingBatches_.begin();
    for (; iterFence != buildingFences_.end() && iterBatch != buildingBatches_.end();) {
        VkResult fenceResult = vkWaitForFences(device->vkDevice(), 1, &(*iterFence)->vkFence(), true, UINT64_MAX);
        if (fenceResult == VK_SUCCESS) {
            vkResetFences(device->vkDevice(), 1, &(*iterFence)->vkFence());
            freeFences_.push(*iterFence);
            freeCmdBuffers_.push(*iterCmd);

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
            iterCmd = buildingCmdBuffers_.erase(iterCmd);
            iterBatch = buildingBatches_.erase(iterBatch);
        } else {
            g_crashRing.record("chunkWaitAllFail", fenceResult);
            std::cout << "Chunk build waitAll fence failed with error: " << std::dec << fenceResult << std::endl;
            // Device lost — all remaining fences are poison, drain them
            buildingFences_.clear();
            buildingCmdBuffers_.clear();
            buildingBatches_.clear();
            break;
        }
    }
}

void ChunkBuildScheduler::tryScheduleBatches(uint32_t maxBatchSize) {
    if (!Renderer::instance().framework()->isRunning()) return;
    std::unique_lock<std::recursive_mutex> lock(mutex_);
    if (!freeFences_.empty() && !queuedIndex_.empty()) {
        auto fence = freeFences_.front();
        auto worldAsyncBuffer = freeCmdBuffers_.front();

        glm::vec3 cameraPos = Renderer::instance().world()->getCameraPos();
        auto chunkBuildDataBatch =
            ChunkBuildDataBatch::create(maxBatchSize, queuedIndex_, chunks_, chunkBuildDatas_, cameraPos);

        auto framework = Renderer::instance().framework();
        auto vma = framework->vma();
        auto device = framework->device();
        auto physicalDevice = Renderer::instance().framework()->physicalDevice();
        auto secondaryQueueIndex = physicalDevice->secondaryQueueIndex();

        if (chunkBuildDataBatch->batchData.size() > 0) {
            worldAsyncBuffer->begin();

            // Upload all buffers (vertex, index, OMM, displaced)
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
                // DDA displacement buffer uploads
                if (chunkBuildData->displacedAABBBuffer) {
                    chunkBuildData->displacedAABBBuffer->uploadToBuffer(worldAsyncBuffer);
                }
                if (chunkBuildData->displacedFaceDataBuffer) {
                    chunkBuildData->displacedFaceDataBuffer->uploadToBuffer(worldAsyncBuffer);
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
            // Add barriers for displaced AABB buffers
            for (auto chunkBuildData : chunkBuildDataBatch->batchData) {
                if (chunkBuildData->displacedAABBBuffer) {
                    bufferBarriers.push_back(vk::CommandBuffer::BufferMemoryBarrier{
                        .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                        .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                        .dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                        .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                        .srcQueueFamilyIndex = secondaryQueueIndex,
                        .dstQueueFamilyIndex = secondaryQueueIndex,
                        .buffer = chunkBuildData->displacedAABBBuffer,
                    });
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

            // Build BLAS (regular + displaced)
            std::vector<std::shared_ptr<vk::BLASBuilder>> builders;
            for (auto chunkBuildData : chunkBuildDataBatch->batchData) {
                builders.push_back(chunkBuildData->blasBuilder);
                if (chunkBuildData->displacedBlasBuilder) {
                    builders.push_back(chunkBuildData->displacedBlasBuilder);
                }
            }
            vk::BLASBuilder::batchSubmit(builders, worldAsyncBuffer);

            // BLAS compaction: query compacted sizes after build completes
            {
                // Collect all chunk BLASes (not displaced — those use AABB, compaction less useful)
                std::vector<VkAccelerationStructureKHR> blasHandles;
                for (auto &cbd : chunkBuildDataBatch->batchData) {
                    if (cbd->blas) blasHandles.push_back(cbd->blas->blas());
                }

                if (!blasHandles.empty()) {
                    VkQueryPoolCreateInfo qpci{};
                    qpci.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
                    qpci.queryType = VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR;
                    qpci.queryCount = static_cast<uint32_t>(blasHandles.size());

                    VkQueryPool qp;
                    if (vkCreateQueryPool(device->vkDevice(), &qpci, nullptr, &qp) == VK_SUCCESS) {
                        vkCmdResetQueryPool(worldAsyncBuffer->vkCommandBuffer(), qp, 0, qpci.queryCount);
                        vkCmdWriteAccelerationStructuresPropertiesKHR(
                            worldAsyncBuffer->vkCommandBuffer(),
                            static_cast<uint32_t>(blasHandles.size()), blasHandles.data(),
                            VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR,
                            qp, 0);
                        chunkBuildDataBatch->compactionQueryPool = qp;
                        chunkBuildDataBatch->compactionQueryCount = qpci.queryCount;
                        chunkBuildDataBatch->queryDevice = device;
                    }
                }
            }

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

            VkResult submitResult = vkQueueSubmit(device->secondaryQueue(), 1, &vkSubmitInfo, fence->vkFence());
            if (submitResult != VK_SUCCESS) {
                g_crashRing.record("chunkSubmitFail", submitResult);
                std::cout << "Chunk batch vkQueueSubmit failed with error: " << std::dec << submitResult << std::endl;
                return;
            }

            freeFences_.pop();
            freeCmdBuffers_.pop();
            buildingFences_.push_back(fence);
            buildingCmdBuffers_.push_back(worldAsyncBuffer);
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
        blasGeneration++;

        gc.collect(vertexBuffers);
        vertexBuffers = std::make_shared<std::vector<std::shared_ptr<vk::DeviceLocalBuffer>>>(
            std::move(chunkBuildData->vertexBuffers));

        gc.collect(indexBuffers);
        indexBuffers = std::make_shared<std::vector<std::shared_ptr<vk::DeviceLocalBuffer>>>(
            std::move(chunkBuildData->indexBuffers));
    } else {
        gc.collect(chunkBuildData->blas);
        if (chunkBuildData->displacedBlas) gc.collect(chunkBuildData->displacedBlas);
        if (chunkBuildData->displacedFaceDataBuffer) gc.collect(chunkBuildData->displacedFaceDataBuffer);

        gc.collect(std::make_shared<std::vector<std::shared_ptr<vk::DeviceLocalBuffer>>>(
            std::move(chunkBuildData->vertexBuffers)));

        gc.collect(std::make_shared<std::vector<std::shared_ptr<vk::DeviceLocalBuffer>>>(
            std::move(chunkBuildData->indexBuffers)));
    }

    allVertexCount = chunkBuildData->allVertexCount;
    allIndexCount = chunkBuildData->allIndexCount;
    geometryCount = chunkBuildData->geometryCount;
    geometryTypes = std::make_shared<std::vector<World::GeometryTypes>>(std::move(chunkBuildData->geometryTypes));
    // Defer destruction of old displaced resources (GPU may still reference from in-flight frames)
    if (displacedBlas) gc.collect(displacedBlas);
    if (displacedFaceDataBuffer) gc.collect(displacedFaceDataBuffer);
    displacedFaceDataBuffer = chunkBuildData->displacedFaceDataBuffer;
    displacedBlas = chunkBuildData->displacedBlas;
    displacedFaceCount = chunkBuildData->displacedFaceCount;
    // CPU vertex/index data kept alive in ChunkBuildData (releaseHostGeometry disabled)
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
        glm::vec3 importantCameraPos = Renderer::instance().world() ? glm::vec3(Renderer::instance().world()->getCameraPos()) : glm::vec3(0);
        chunkBuildData->build(false, ommEnabled, importantCameraPos);
        for (int i = 0; i < chunkBuildData->geometryCount; i++) {
            Renderer::instance().buffers()->queueImportantWorldUpload(chunkBuildData->vertexBuffers[i],
                                                                      chunkBuildData->indexBuffers[i]);
            if (chunkBuildData->ommIndexBuffers[i] != nullptr) {
                Renderer::instance().buffers()->queueImportantWorldUpload(chunkBuildData->ommIndexBuffers[i],
                                                                          nullptr);
            }
        }
        importantBLASBuilders_->push_back(chunkBuildData->blasBuilder);

        // DDA displacement: upload AABB buffer and queue displaced BLAS build
        if (chunkBuildData->displacedBlasBuilder) {
            Renderer::instance().buffers()->queueImportantWorldUpload(chunkBuildData->displacedAABBBuffer, nullptr);
            Renderer::instance().buffers()->queueImportantWorldUpload(chunkBuildData->displacedFaceDataBuffer, nullptr);
            importantBLASBuilders_->push_back(chunkBuildData->displacedBlasBuilder);
        }

        // Copy geometry data BEFORE releasing host data (enqueue no longer stores CPU data)
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

        // Release CPU vertex/index data now that GPU buffers are uploaded and async copy is made
        // chunkBuildData->releaseHostGeometry(); // DISABLED: investigating staging buffer crash

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