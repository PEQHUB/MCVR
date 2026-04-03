#include "core/render/chunks.hpp"

#include "core/render/block_mesher.hpp"
#include "core/render/buffers.hpp"
#include "core/render/crash_ring_buffer.hpp"
#include "core/render/greedy_mesher.hpp"
#include "core/render/pipeline.hpp"
#include "core/render/render_framework.hpp"
#include "core/render/renderer.hpp"
#include "core/render/tessellator.hpp"
#include "core/render/textures.hpp"
#include "core/render/modules/world/ray_tracing/ray_tracing_module.hpp"
#include "core/render/modules/world/ray_tracing/submodules/world_prepare.hpp"
#ifdef MCVR_ENABLE_OMM
#include "core/render/omm_baker.hpp"
#endif

#include <algorithm>
#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <set>

#include <glm/gtc/packing.hpp>

// Convert full PBRTriangle vertices to compact 32-byte format.
// Drops norm, postBase, lightPacked, glintUV/glintTexture, overlayPacked.
// Compresses colorLayer to RGBA8, albedoEmission to fp16.
static std::vector<vk::VertexFormat::PBRTriangleCompact>
packCompactVertices(const std::vector<vk::VertexFormat::PBRTriangle> &src) {
    std::vector<vk::VertexFormat::PBRTriangleCompact> dst(src.size());
    for (size_t j = 0; j < src.size(); j++) {
        auto &s = src[j];
        auto &d = dst[j];
        d.pos = s.pos;
        // packed0: flags (bits 0-10) | vivid flag (bit 11) | textureID (bits 16-31)
        // Clear USE_GLINT, USE_OVERLAY, and GREEDY_MERGED — compact format zeros those data
        // fields, so leaving the flags set would cause the shader to sample garbage or divide by zero.
        uint32_t flags = s.flags & 0x77FFu; // preserve bits 0-10 + 12-13 (BIOME_TINT) + 14 (BLOCK_GEOMETRY)
        flags &= ~(vk::VertexFormat::PBR_FLAG_USE_GLINT | vk::VertexFormat::PBR_FLAG_USE_OVERLAY |
                    vk::VertexFormat::PBR_FLAG_GREEDY_MERGED);
        if (s.emissiveBlockType & 0x10000u) flags |= vk::VertexFormat::PBR_FLAG_COMPACT_VIVID;
        d.packed0 = flags | ((s.textureID & 0xFFFFu) << 16);
        d.textureUV = s.textureUV;
        // colorLayer vec4 → RGBA8
        uint8_t r = static_cast<uint8_t>(glm::clamp(s.colorLayer.r * 255.0f, 0.0f, 255.0f));
        uint8_t g = static_cast<uint8_t>(glm::clamp(s.colorLayer.g * 255.0f, 0.0f, 255.0f));
        uint8_t b = static_cast<uint8_t>(glm::clamp(s.colorLayer.b * 255.0f, 0.0f, 255.0f));
        uint8_t a = static_cast<uint8_t>(glm::clamp(s.colorLayer.a * 255.0f, 0.0f, 255.0f));
        d.colorPacked = uint32_t(r) | (uint32_t(g) << 8) | (uint32_t(b) << 16) | (uint32_t(a) << 24);
        // packed1: albedoEmission as fp16 (lower 16) | emissiveBlockType bits 0-15 (upper 16)
        uint32_t halfEmission = glm::packHalf2x16(glm::vec2(s.albedoEmission, 0.0f)) & 0xFFFFu;
        d.packed1 = halfEmission | ((s.emissiveBlockType & 0xFFFFu) << 16);
    }
    return dst;
}

// Convert full PBRTriangle vertices to lossless 64-byte format.
// Drops only dead fields: norm, postBase, lightPacked.
// All shader-read fields preserved at full precision — bit-identical output.
static std::vector<vk::VertexFormat::PBRTriangleLossless>
packLosslessVertices(const std::vector<vk::VertexFormat::PBRTriangle> &src) {
    std::vector<vk::VertexFormat::PBRTriangleLossless> dst(src.size());
    for (size_t j = 0; j < src.size(); j++) {
        auto &s = src[j];
        auto &d = dst[j];
        d.pos = s.pos;
        d.flags = s.flags;
        d.colorLayer = s.colorLayer;
        d.textureUV = s.textureUV;
        d.glintUV = s.glintUV;
        d.albedoEmission = s.albedoEmission;
        d.emissiveBlockType = s.emissiveBlockType;
        d.textureID_glint = (s.textureID & 0xFFFFu) | ((s.glintTexture & 0xFFFFu) << 16);
        d.overlayPacked = s.overlayPacked;
    }
    return dst;
}

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
    prepareCPU(allowMicromapBake, skipOMM, cameraPos);
    uploadGPU();
}

// ---- CPU PHASE: greedy meshing, OMM classification/baking, tessellation ----
// No Vulkan or VMA calls. Safe to call from worker threads.
void ChunkBuildData::prepareCPU(bool allowMicromapBake, bool skipOMM, glm::vec3 cameraPos) {
    auto textures = Renderer::instance().textures();
    auto device = Renderer::instance().framework()->device();
    bool useOMM = !skipOMM && device->hasOMM() && Renderer::options.ommEnabled && textures != nullptr;

#ifdef MCVR_ENABLE_OMM
    // Thread-local OMM baker (one per thread, SDK is not thread-safe per instance)
    static thread_local std::unique_ptr<OMMBaker> tlBaker;
    if (useOMM && !tlBaker) {
        tlBaker = std::make_unique<OMMBaker>();
    }
#endif

    ommGeometryData.resize(geometryCount);
    ommCpuResults.resize(geometryCount);

    // Greedy meshing: merge coplanar block faces for WORLD_SOLID (50-70% triangle reduction)
    if (Renderer::options.greedyMeshingEnabled) {
        for (int i = 0; i < geometryCount; i++) {
            if (geometryTypes[i] == World::WORLD_SOLID) {
                auto merged = GreedyMesher::merge(vertices[i], indices[i]);
                vertices[i] = std::move(merged.vertices);
                indices[i] = std::move(merged.indices);
            }
        }
    }

    // Full 96-byte PBRTriangle — lossless format has struct layout issues.
    // Block vs entity is identified via PBR_FLAG_BLOCK_GEOMETRY in vertex flags.
    vertexFormat = 0;

    // CPU-only OMM classification: compute per-triangle opacity indices without VMA/Vulkan calls.
    // Results stored in ommCpuResults[], consumed by uploadGPU().

    for (int i = 0; i < geometryCount; i++) {
        auto &cr = ommCpuResults[i];

        if (useOMM && geometryTypes[i] == World::WORLD_TRANSPARENT) {
#ifdef MCVR_ENABLE_OMM
            cr.isTransparentOMM = true;
            uint32_t numTriangles = static_cast<uint32_t>(indices[i].size()) / 3;

            if (!allowMicromapBake) {
                // Phase 1 fallback: special indices only
                cr.ommIndices.resize(numTriangles);
                for (uint32_t t = 0; t < numTriangles; t++) {
                    uint32_t vertIdx = indices[i][t * 3];
                    uint32_t texId = vertices[i][vertIdx].textureID;
                    auto alphaClass = textures->getTextureAlphaClass(texId);
                    switch (alphaClass) {
                        case Textures::AlphaClass::FULLY_OPAQUE:
                            cr.ommIndices[t] = VK_OPACITY_MICROMAP_SPECIAL_INDEX_FULLY_OPAQUE_EXT;
                            break;
                        default:
                            cr.ommIndices[t] = VK_OPACITY_MICROMAP_SPECIAL_INDEX_FULLY_UNKNOWN_OPAQUE_EXT;
                            break;
                    }
                }
            } else {
                // Phase 2: full per-micro-triangle baking (CPU-only computation)
                std::map<uint32_t, std::vector<uint32_t>> texGroups;
                for (uint32_t t = 0; t < numTriangles; t++) {
                    uint32_t vertIdx = indices[i][t * 3];
                    uint32_t texId = vertices[i][vertIdx].textureID;
                    texGroups[texId].push_back(t);
                }

                cr.ommIndices.assign(numTriangles, VK_OPACITY_MICROMAP_SPECIAL_INDEX_FULLY_UNKNOWN_OPAQUE_EXT);
                std::map<uint64_t, uint32_t> descHistMap, indexHistMap;

                for (auto &[texId, triList] : texGroups) {
                    const auto *alphaData = textures->getTextureAlphaData(texId);
                    if (!alphaData || alphaData->alpha.empty()) {
                        for (uint32_t t : triList) {
                            auto alphaClass = textures->getTextureAlphaClass(texId);
                            cr.ommIndices[t] = (alphaClass == Textures::AlphaClass::FULLY_OPAQUE)
                                ? VK_OPACITY_MICROMAP_SPECIAL_INDEX_FULLY_OPAQUE_EXT
                                : VK_OPACITY_MICROMAP_SPECIAL_INDEX_FULLY_UNKNOWN_OPAQUE_EXT;
                        }
                        continue;
                    }

                    auto alphaClass = textures->getTextureAlphaClass(texId);
                    if (alphaClass == Textures::AlphaClass::FULLY_TRANSPARENT) {
                        for (uint32_t t : triList) {
                            cr.ommIndices[t] = VK_OPACITY_MICROMAP_SPECIAL_INDEX_FULLY_UNKNOWN_OPAQUE_EXT;
                        }
                        continue;
                    }

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
                        cr.bakingDone = true;
                        uint32_t baseOffset = static_cast<uint32_t>(cr.mergedArrayData.size());
                        uint32_t baseDescIndex = static_cast<uint32_t>(cr.mergedDescs.size());

                        cr.mergedArrayData.insert(cr.mergedArrayData.end(), result.arrayData.begin(), result.arrayData.end());
                        for (uint32_t d = 0; d < result.descArrayCount; d++) {
                            VkMicromapTriangleEXT desc{};
                            desc.dataOffset = result.descOffsets[d] + baseOffset;
                            desc.subdivisionLevel = result.descSubdivisionLevels[d];
                            desc.format = result.descFormats[d];
                            cr.mergedDescs.push_back(desc);
                        }
                        for (uint32_t li = 0; li < triList.size(); li++) {
                            int32_t idx = result.indexBuffer[li];
                            cr.ommIndices[triList[li]] = (idx >= 0) ? idx + static_cast<int32_t>(baseDescIndex) : idx;
                        }
                        for (auto &uc : result.descArrayHistogram) {
                            descHistMap[(static_cast<uint64_t>(uc.subdivisionLevel) << 16) | uc.format] += uc.count;
                        }
                        for (auto &uc : result.indexHistogram) {
                            indexHistMap[(static_cast<uint64_t>(uc.subdivisionLevel) << 16) | uc.format] += uc.count;
                        }
                    } else {
                        for (uint32_t t : triList) {
                            cr.ommIndices[t] = VK_OPACITY_MICROMAP_SPECIAL_INDEX_FULLY_UNKNOWN_OPAQUE_EXT;
                        }
                    }
                }

                // Convert histograms to Vulkan structs (CPU data, no Vulkan calls)
                for (auto &[key, count] : descHistMap) {
                    VkMicromapUsageEXT usage{};
                    usage.count = count;
                    usage.subdivisionLevel = static_cast<uint32_t>(key >> 16);
                    usage.format = static_cast<uint32_t>(key & 0xFFFF);
                    cr.descHistogram.push_back(usage);
                }
                for (auto &[key, count] : indexHistMap) {
                    VkMicromapUsageEXT usage{};
                    usage.count = count;
                    usage.subdivisionLevel = static_cast<uint32_t>(key >> 16);
                    usage.format = static_cast<uint32_t>(key & 0xFFFF);
                    cr.indexHistogram.push_back(usage);
                }
            }
#endif
        } else if (useOMM) {
            // WORLD_SOLID with OMM enabled: all-opaque special indices
            cr.isOpaqueOMM = true;
            uint32_t numTriangles = static_cast<uint32_t>(indices[i].size()) / 3;
            cr.ommIndices.assign(numTriangles, VK_OPACITY_MICROMAP_SPECIAL_INDEX_FULLY_OPAQUE_EXT);
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
                // Buffer creation deferred to uploadGPU()
            }
        }
    }

    // Cache the face count before any potential releaseHostGeometry()
    displacedFaceCount = static_cast<uint32_t>(displacedFaceData.size());
}

// ---- GPU PHASE: VMA allocation, staging uploads, BLAS builder setup ----
// Must run on render thread (single-threaded Vulkan access).
void ChunkBuildData::uploadGPU() {
    auto framework = Renderer::instance().framework();
    auto vma = framework->vma();
    auto device = framework->device();
    auto physicalDevice = framework->physicalDevice();

    // Pack vertices into the selected format before uploading
    std::vector<std::vector<vk::VertexFormat::PBRTriangleCompact>> compactVerts;
    std::vector<std::vector<vk::VertexFormat::PBRTriangleLossless>> losslessVerts;
    if (vertexFormat == 1) {
        compactVerts.resize(geometryCount);
        for (int i = 0; i < geometryCount; i++)
            compactVerts[i] = packCompactVertices(vertices[i]);
    } else if (vertexFormat == 2) {
        losslessVerts.resize(geometryCount);
        for (int i = 0; i < geometryCount; i++)
            losslessVerts[i] = packLosslessVertices(vertices[i]);
    }

    // Create vertex/index buffers from final geometry (post-mesh, post-tessellation)
    for (int i = 0; i < geometryCount; i++) {
        VkDeviceSize bufSize;
        const void *bufData;
        if (vertexFormat == 1) {
            bufSize = compactVerts[i].size() * sizeof(vk::VertexFormat::PBRTriangleCompact);
            bufData = compactVerts[i].data();
        } else if (vertexFormat == 2) {
            bufSize = losslessVerts[i].size() * sizeof(vk::VertexFormat::PBRTriangleLossless);
            bufData = losslessVerts[i].data();
        } else {
            bufSize = vertices[i].size() * sizeof(vk::VertexFormat::PBRTriangle);
            bufData = vertices[i].data();
        }

        auto vertexBuffer =
            vk::DeviceLocalBuffer::create(vma, device, true, bufSize,
                                          VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                                              VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        vertexBuffer->uploadToStagingBuffer(const_cast<void *>(bufData));
        vertexBuffers.push_back(vertexBuffer);

        auto indexBuffer =
            vk::DeviceLocalBuffer::create(vma, device, true, indices[i].size() * sizeof(uint32_t),
                                          VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                                              VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        indexBuffer->uploadToStagingBuffer(indices[i].data());
        indexBuffers.push_back(indexBuffer);

        // Upload OMM data from CPU results
        auto &cr = ommCpuResults[i];
        if (cr.isTransparentOMM || cr.isOpaqueOMM) {
            auto ommIdxBuffer = vk::DeviceLocalBuffer::create(
                vma, device, true, cr.ommIndices.size() * sizeof(int32_t),
                VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                    VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                    VK_BUFFER_USAGE_MICROMAP_BUILD_INPUT_READ_ONLY_BIT_EXT);
            ommIdxBuffer->uploadToStagingBuffer(cr.ommIndices.data());
            ommIndexBuffers.push_back(ommIdxBuffer);

            // Phase 2 OMM: create micromap from baked data
            if (cr.bakingDone && !cr.mergedDescs.empty()) {
                auto &gd = ommGeometryData[i];
                gd.arrayBuffer = vk::DeviceLocalBuffer::create(
                    vma, device, true, cr.mergedArrayData.size(),
                    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                        VK_BUFFER_USAGE_MICROMAP_BUILD_INPUT_READ_ONLY_BIT_EXT);
                gd.arrayBuffer->uploadToStagingBuffer(cr.mergedArrayData.data());

                gd.descBuffer = vk::DeviceLocalBuffer::create(
                    vma, device, true, cr.mergedDescs.size() * sizeof(VkMicromapTriangleEXT),
                    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                        VK_BUFFER_USAGE_MICROMAP_BUILD_INPUT_READ_ONLY_BIT_EXT);
                gd.descBuffer->uploadToStagingBuffer(cr.mergedDescs.data());

                gd.descHistogram = std::move(cr.descHistogram);
                gd.indexHistogram = std::move(cr.indexHistogram);

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

                VkDeviceSize micromapSize = (sizeInfo.micromapSize + 255) & ~255ULL;
                gd.micromapBuffer = vk::DeviceLocalBuffer::create(
                    vma, device, micromapSize,
                    VK_BUFFER_USAGE_MICROMAP_STORAGE_BIT_EXT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);

                if (sizeInfo.buildScratchSize > 0) {
                    VkDeviceSize scratchSize = (sizeInfo.buildScratchSize + 255) & ~255ULL;
                    gd.micromapScratchBuffer = vk::DeviceLocalBuffer::create(
                        vma, device, scratchSize,
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
                }

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
        } else {
            ommIndexBuffers.push_back(nullptr);
        }
    }

    // Free CPU-side OMM results now that they're uploaded
    ommCpuResults.clear();
    ommCpuResults.shrink_to_fit();

    blasBuilder = vk::BLASBuilder::create();
    auto blasGeometryBuilder = blasBuilder->beginGeometries();

    // Lambda: define BLAS geometry with correct vertex stride template
    auto defineGeom = [&](int i, bool isOpaque, uint32_t numVerts, uint32_t numIndices) {
        if (vertexFormat == 1) {
            blasGeometryBuilder->defineTriangleGeomrtry<vk::VertexFormat::PBRTriangleCompact>(
                vertexBuffers[i], numVerts, indexBuffers[i], numIndices, isOpaque);
        } else if (vertexFormat == 2) {
            blasGeometryBuilder->defineTriangleGeomrtry<vk::VertexFormat::PBRTriangleLossless>(
                vertexBuffers[i], numVerts, indexBuffers[i], numIndices, isOpaque);
        } else {
            blasGeometryBuilder->defineTriangleGeomrtry<vk::VertexFormat::PBRTriangle>(
                vertexBuffers[i], numVerts, indexBuffers[i], numIndices, isOpaque);
        }
    };
    auto defineGeomOMM = [&](int i, bool isOpaque, uint32_t numVerts, uint32_t numIndices,
                             VkDeviceAddress ommAddr, uint32_t numTris) {
        if (vertexFormat == 1) {
            blasGeometryBuilder->defineTriangleGeomrtry<vk::VertexFormat::PBRTriangleCompact>(
                vertexBuffers[i], numVerts, indexBuffers[i], numIndices, isOpaque, ommAddr, numTris);
        } else if (vertexFormat == 2) {
            blasGeometryBuilder->defineTriangleGeomrtry<vk::VertexFormat::PBRTriangleLossless>(
                vertexBuffers[i], numVerts, indexBuffers[i], numIndices, isOpaque, ommAddr, numTris);
        } else {
            blasGeometryBuilder->defineTriangleGeomrtry<vk::VertexFormat::PBRTriangle>(
                vertexBuffers[i], numVerts, indexBuffers[i], numIndices, isOpaque, ommAddr, numTris);
        }
    };
    auto defineGeomMicromap = [&](int i, bool isOpaque, uint32_t numVerts, uint32_t numIndices,
                                  VkDeviceAddress ommAddr, uint32_t numTris,
                                  VkMicromapEXT mm, const VkMicromapUsageEXT *uc, uint32_t ucc) {
        if (vertexFormat == 1) {
            blasGeometryBuilder->defineTriangleGeomrtryWithMicromap<vk::VertexFormat::PBRTriangleCompact>(
                vertexBuffers[i], numVerts, indexBuffers[i], numIndices, isOpaque, ommAddr, numTris, mm, uc, ucc);
        } else if (vertexFormat == 2) {
            blasGeometryBuilder->defineTriangleGeomrtryWithMicromap<vk::VertexFormat::PBRTriangleLossless>(
                vertexBuffers[i], numVerts, indexBuffers[i], numIndices, isOpaque, ommAddr, numTris, mm, uc, ucc);
        } else {
            blasGeometryBuilder->defineTriangleGeomrtryWithMicromap<vk::VertexFormat::PBRTriangle>(
                vertexBuffers[i], numVerts, indexBuffers[i], numIndices, isOpaque, ommAddr, numTris, mm, uc, ucc);
        }
    };

    for (int i = 0; i < geometryCount; i++) {
        bool isOpaque = geometryTypes[i] == World::WORLD_SOLID;
        uint32_t indexCount = static_cast<uint32_t>(indices[i].size());
        uint32_t vertCount = static_cast<uint32_t>(vertices[i].size());

        if (indexCount == 0) {
            blasGeometryBuilder->definePlaceholderGeometry();
            continue;
        }

        if (ommIndexBuffers[i] != nullptr) {
            uint32_t numTriangles = indexCount / 3;
            if (ommGeometryData[i].hasMicromap) {
                defineGeomMicromap(i, isOpaque, vertCount, indexCount,
                    ommIndexBuffers[i]->bufferAddress(), numTriangles,
                    ommGeometryData[i].micromap,
                    ommGeometryData[i].indexHistogram.data(),
                    static_cast<uint32_t>(ommGeometryData[i].indexHistogram.size()));
            } else {
                defineGeomOMM(i, isOpaque, vertCount, indexCount,
                    ommIndexBuffers[i]->bufferAddress(), numTriangles);
            }
        } else {
            defineGeom(i, isOpaque, vertCount, indexCount);
        }
    }
    blasGeometryBuilder->endGeometries();
    blas = blasBuilder->defineBuildProperty(VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR)
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

void ChunkBuildData::releaseStagingBuffers() {
    for (auto &vb : vertexBuffers) { if (vb) vb->releaseStagingBuffer(); }
    for (auto &ib : indexBuffers) { if (ib) ib->releaseStagingBuffer(); }
    for (auto &ob : ommIndexBuffers) { if (ob) ob->releaseStagingBuffer(); }
    for (auto &gd : ommGeometryData) {
        if (gd.arrayBuffer) gd.arrayBuffer->releaseStagingBuffer();
        if (gd.descBuffer) gd.descBuffer->releaseStagingBuffer();
    }
    if (displacedAABBBuffer) displacedAABBBuffer->releaseStagingBuffer();
    if (displacedFaceDataBuffer) displacedFaceDataBuffer->releaseStagingBuffer();
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
        batchData[i]->prepareCPU(true, false, cameraPos);
    });

    for (size_t i = 0; i < count; i++) {
        batchData[i]->uploadGPU();
    }
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

    std::unique_lock<std::recursive_mutex> lock(mutex_);
    for (auto &cbd : completed) {
        if (cbd->id >= static_cast<int64_t>(chunks_.size())) continue;
        chunks_[cbd->id]->enqueue(cbd);

        ChunkPackedData data = { .geometryCount = cbd->geometryCount };
        chunkPackedData_->uploadToBuffer(&data, sizeof(ChunkPackedData),
                                         cbd->id * sizeof(ChunkPackedData));

        // Release ChunkBuildData slot to free scratch/OMM/displacement buffers
        // that were NOT transferred to Chunk1 (blasBuilder, ommGeometryData, etc.).
        // Only clear if slot still points to this build (a newer build may have replaced it).
        if (cbd->id < static_cast<int64_t>(chunkBuildDatas_.size()) &&
            chunkBuildDatas_[cbd->id] == cbd) {
            chunkBuildDatas_[cbd->id] = nullptr;
        }
    }
}

void ChunkBuildScheduler::waitAllFinish() {
    // Signal stop, join, then integrate remaining
    stop_.store(true);
    inputCv_.notify_one();
    if (blasThread_.joinable()) blasThread_.join();
    integrateCompleted();
}

// BLAS builder thread: sole owner of the secondary Vulkan queue.
// Runs the complete chunk pipeline synchronously: CPU → GPU → fence wait → compaction.
// Blocking waits are free because this is a dedicated thread.
void ChunkBuildScheduler::blasThreadLoop() {
    auto framework = Renderer::instance().framework();
    auto vma = framework->vma();
    auto device = framework->device();
    auto physicalDevice = framework->physicalDevice();
    auto asyncPool = framework->asyncCommandPool();
    auto secondaryQueueIndex = physicalDevice->secondaryQueueIndex();

    // Pre-allocate cmd pool — sole owner, no mutex needed.
    constexpr uint32_t MAX_IN_FLIGHT = 12;
    for (uint32_t i = 0; i < MAX_IN_FLIGHT; i++) {
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
                        << " maxInFlight=" << MAX_IN_FLIGHT
                        << " path=" << diagPath.string() << std::endl;
                diagLog.flush();
            }
        }
    };
    ensureDiagOpen();

    while (!stop_.load()) {
        // Pause gate: render thread requests pause during swapchain recreate
        if (paused_.load(std::memory_order_acquire)) {
            pausedAck_.store(true, std::memory_order_release);
            while (paused_.load(std::memory_order_acquire) && !stop_.load()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            pausedAck_.store(false, std::memory_order_release);
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
        bool hasWork = false;
        {
            std::lock_guard<std::mutex> lock(inputMtx_);
            hasWork = !inputQueue_.empty();
        }

        if (hasWork && inFlight_.size() < MAX_IN_FLIGHT && !cmdPool_.empty()) {
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
                Renderer::threadPool.parallelFor(static_cast<uint32_t>(batch.size()), [&](uint32_t i) {
                    batch[i]->prepareCPU(true, false, cameraPos);
                });
                // GPU upload (sequential — VMA alloc + staging)
                for (auto &cbd : batch) cbd->uploadGPU();

                // Release CPU-side geometry data — staging buffers hold the copy for GPU transfer.
                // This frees ~200KB/chunk of system RAM (PBRTriangle arrays + index arrays).
                for (auto &cbd : batch) cbd->releaseHostGeometry();

                // Take cmd from pool
                auto cmd = std::move(cmdPool_.front()); cmdPool_.pop();

                // Record
                cmd->begin();
                for (auto &cbd : batch) {
                    for (int i = 0; i < cbd->geometryCount; i++) {
                        cbd->vertexBuffers[i]->uploadToBuffer(cmd);
                        cbd->indexBuffers[i]->uploadToBuffer(cmd);
                        if (cbd->ommIndexBuffers[i]) cbd->ommIndexBuffers[i]->uploadToBuffer(cmd);
                        auto &gd = cbd->ommGeometryData[i];
                        if (gd.hasMicromap) { gd.arrayBuffer->uploadToBuffer(cmd); gd.descBuffer->uploadToBuffer(cmd); }
                    }
                    if (cbd->displacedAABBBuffer) cbd->displacedAABBBuffer->uploadToBuffer(cmd);
                    if (cbd->displacedFaceDataBuffer) cbd->displacedFaceDataBuffer->uploadToBuffer(cmd);
                }

                // Barriers
                std::vector<vk::CommandBuffer::BufferMemoryBarrier> barriers;
                for (auto &cbd : batch) {
                    for (int i = 0; i < cbd->geometryCount; i++) {
                        VkPipelineStageFlags2 dst = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
                        if (cbd->ommGeometryData[i].hasMicromap) dst |= VK_PIPELINE_STAGE_2_MICROMAP_BUILD_BIT_EXT;
                        auto bar = [&](std::shared_ptr<vk::DeviceLocalBuffer> &b) {
                            barriers.push_back({VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                VK_ACCESS_2_MEMORY_READ_BIT|VK_ACCESS_2_MEMORY_WRITE_BIT, dst,
                                VK_ACCESS_2_MEMORY_READ_BIT|VK_ACCESS_2_MEMORY_WRITE_BIT,
                                secondaryQueueIndex, secondaryQueueIndex, b});
                        };
                        bar(cbd->vertexBuffers[i]); bar(cbd->indexBuffers[i]);
                        if (cbd->ommIndexBuffers[i]) bar(cbd->ommIndexBuffers[i]);
                        if (cbd->ommGeometryData[i].hasMicromap) {
                            auto mm = VK_PIPELINE_STAGE_2_MICROMAP_BUILD_BIT_EXT;
                            barriers.push_back({VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_MEMORY_READ_BIT|VK_ACCESS_2_MEMORY_WRITE_BIT,
                                mm, VK_ACCESS_2_MEMORY_READ_BIT|VK_ACCESS_2_MEMORY_WRITE_BIT,
                                secondaryQueueIndex, secondaryQueueIndex, cbd->ommGeometryData[i].arrayBuffer});
                            barriers.push_back({VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_MEMORY_READ_BIT|VK_ACCESS_2_MEMORY_WRITE_BIT,
                                mm, VK_ACCESS_2_MEMORY_READ_BIT|VK_ACCESS_2_MEMORY_WRITE_BIT,
                                secondaryQueueIndex, secondaryQueueIndex, cbd->ommGeometryData[i].descBuffer});
                        }
                    }
                    if (cbd->displacedAABBBuffer) {
                        barriers.push_back({VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_MEMORY_READ_BIT|VK_ACCESS_2_MEMORY_WRITE_BIT,
                            VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, VK_ACCESS_2_MEMORY_READ_BIT|VK_ACCESS_2_MEMORY_WRITE_BIT,
                            secondaryQueueIndex, secondaryQueueIndex, cbd->displacedAABBBuffer});
                    }
                }
                cmd->barriersBufferImage(barriers, {});

                // Micromaps
                bool anyMM = false;
                for (auto &cbd : batch) {
                    for (int i = 0; i < cbd->geometryCount; i++) {
                        auto &gd = cbd->ommGeometryData[i];
                        if (!gd.hasMicromap) continue;
                        anyMM = true;
                        VkMicromapBuildInfoEXT bi{VK_STRUCTURE_TYPE_MICROMAP_BUILD_INFO_EXT};
                        bi.type = VK_MICROMAP_TYPE_OPACITY_MICROMAP_EXT; bi.flags = VK_BUILD_MICROMAP_PREFER_FAST_TRACE_BIT_EXT;
                        bi.mode = VK_BUILD_MICROMAP_MODE_BUILD_EXT; bi.dstMicromap = gd.micromap;
                        bi.data.deviceAddress = gd.arrayBuffer->bufferAddress();
                        bi.triangleArray.deviceAddress = gd.descBuffer->bufferAddress();
                        bi.triangleArrayStride = sizeof(VkMicromapTriangleEXT);
                        bi.usageCountsCount = static_cast<uint32_t>(gd.descHistogram.size());
                        bi.pUsageCounts = gd.descHistogram.data();
                        if (gd.micromapScratchBuffer) bi.scratchData.deviceAddress = gd.micromapScratchBuffer->bufferAddress();
                        vkCmdBuildMicromapsEXT(cmd->vkCommandBuffer(), 1, &bi);
                    }
                }
                if (anyMM) {
                    VkMemoryBarrier2 mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
                    mb.srcStageMask = VK_PIPELINE_STAGE_2_MICROMAP_BUILD_BIT_EXT; mb.srcAccessMask = VK_ACCESS_2_MICROMAP_WRITE_BIT_EXT;
                    mb.dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR; mb.dstAccessMask = VK_ACCESS_2_MICROMAP_READ_BIT_EXT;
                    VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO}; dep.memoryBarrierCount = 1; dep.pMemoryBarriers = &mb;
                    vkCmdPipelineBarrier2(cmd->vkCommandBuffer(), &dep);
                }

                // BLAS builds
                std::vector<std::shared_ptr<vk::BLASBuilder>> builders;
                for (auto &cbd : batch) {
                    builders.push_back(cbd->blasBuilder);
                    if (cbd->displacedBlasBuilder) builders.push_back(cbd->displacedBlasBuilder);
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
                inFlight_.push_back({batchValue, std::move(cmd), std::move(batch), qp, qpCount});
            }
        } else if (!hasWork && inFlight_.empty()) {
            // Nothing to do — wait for input
            std::unique_lock<std::mutex> lock(inputMtx_);
            inputCv_.wait_for(lock, std::chrono::milliseconds(5), [this] {
                return stop_.load() || !inputQueue_.empty();
            });
        } else {
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
}

uint32_t ChunkBuildScheduler::chunkBuildingBatchSize() {
    return chunkBuildingBatchSize_;
}

void ChunkBuildScheduler::pause() {
    paused_.store(true, std::memory_order_release);
    // Spin until BLAS thread acknowledges — guarantees no submit is in progress
    while (!pausedAck_.load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
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
        vertexFormat = chunkBuildData->vertexFormat;

        gc.collect(blas);
        // GC the pre-compaction BLAS (from BLAS thread) — GPU TLAS may still reference it
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
    biomeGrassColor = chunkBuildData->biomeGrassColor;
    biomeFoliageColor = chunkBuildData->biomeFoliageColor;
    biomeWaterColor = chunkBuildData->biomeWaterColor;
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

void Chunks::resetScheduler() {
    std::unique_lock<std::recursive_mutex> lock(mutex_);

    if (chunkBuildScheduler_ == nullptr) return;

    chunkBuildScheduler_->waitAllFinish();

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

    // Release ChunkBuildData to free scratch/OMM buffers for out-of-range chunks
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

    // ALL chunks go to the BLAS thread — render thread does zero chunk building.
    // Important chunks get higher priority (2x) but are still async.
    // This means chunks appear 1-2 frames later but the render thread never stalls.
    chunkBuildDatas_[task.id] = chunkBuildData;
    glm::vec3 camPos = Renderer::instance().world() ? glm::vec3(Renderer::instance().world()->getCameraPos()) : glm::vec3(0);
    if (chunkBuildScheduler_) {
        chunkBuildScheduler_->enqueue(chunkBuildData, camPos, task.isImportant);
    }
}

void Chunks::queueBlockStateBuild(ChunkBuildTaskV2 task) {
    if (!Renderer::blockModelTable.isLoaded()) {
        // Model table not ready — can't mesh in C++
        return;
    }

    // Build mesher input from task data
    BlockMesher::SectionInput input;
    std::memcpy(input.blockStates, task.blockStates, sizeof(input.blockStates));
    std::memcpy(input.biomes, task.biomes, sizeof(input.biomes));
    std::memcpy(input.neighborStates, task.neighborStates, sizeof(input.neighborStates));
    input.originX = task.x;
    input.originY = task.y;
    input.originZ = task.z;
    input.blockAtlasTextureId = task.blockAtlasTextureId;

    // C++ meshing — generates identical PBRTriangle output
    auto meshOutput = BlockMesher::mesh(input, Renderer::blockModelTable);

    // Build geometry arrays matching the existing ChunkBuildData format
    std::vector<World::GeometryTypes> geometryTypes;
    std::vector<std::vector<vk::VertexFormat::PBRTriangle>> vertices;
    std::vector<std::vector<uint32_t>> indices;
    uint32_t allVertexCount = 0, allIndexCount = 0;

    if (!meshOutput.solidVertices.empty()) {
        geometryTypes.push_back(World::WORLD_SOLID);
        allVertexCount += static_cast<uint32_t>(meshOutput.solidVertices.size());
        allIndexCount += static_cast<uint32_t>(meshOutput.solidIndices.size());
        vertices.push_back(std::move(meshOutput.solidVertices));
        indices.push_back(std::move(meshOutput.solidIndices));
    }
    if (!meshOutput.cutoutVertices.empty()) {
        geometryTypes.push_back(World::WORLD_TRANSPARENT);
        allVertexCount += static_cast<uint32_t>(meshOutput.cutoutVertices.size());
        allIndexCount += static_cast<uint32_t>(meshOutput.cutoutIndices.size());
        vertices.push_back(std::move(meshOutput.cutoutVertices));
        indices.push_back(std::move(meshOutput.cutoutIndices));
    }
    if (!meshOutput.translucentVertices.empty()) {
        geometryTypes.push_back(World::WORLD_TRANSPARENT);
        allVertexCount += static_cast<uint32_t>(meshOutput.translucentVertices.size());
        allIndexCount += static_cast<uint32_t>(meshOutput.translucentIndices.size());
        vertices.push_back(std::move(meshOutput.translucentVertices));
        indices.push_back(std::move(meshOutput.translucentIndices));
    }

    if (geometryTypes.empty()) {
        // Empty section (all air) — invalidate
        invalidateChunk(task.id);
        return;
    }

    auto framework = Renderer::instance().framework();

    std::unique_lock<std::recursive_mutex> lock(mutex_);
    if (task.id < 0 || task.id >= static_cast<int64_t>(chunks_.size())) return;

    int geomCount = static_cast<int>(geometryTypes.size());
    auto chunkBuildData = ChunkBuildData::create(
        task.id, task.x, task.y, task.z, chunks_[task.id]->latestVersion++,
        allVertexCount, allIndexCount, geomCount,
        std::move(geometryTypes), std::move(vertices), std::move(indices));
    chunkBuildData->biomeGrassColor = task.biomeGrassColor;
    chunkBuildData->biomeFoliageColor = task.biomeFoliageColor;
    chunkBuildData->biomeWaterColor = task.biomeWaterColor;

    chunkBuildDatas_[task.id] = chunkBuildData;
    glm::vec3 camPos = Renderer::instance().world()
        ? glm::vec3(Renderer::instance().world()->getCameraPos()) : glm::vec3(0);
    if (chunkBuildScheduler_) {
        chunkBuildScheduler_->enqueue(chunkBuildData, camPos, task.isImportant);
    }

    // Set chunk lights
    if (!meshOutput.lights.empty()) {
        setChunkLights(task.id, meshOutput.lights);
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

void Chunks::setChunkLights(int64_t id, const std::vector<ChunkLightEntry> &lights) {
    std::unique_lock<std::recursive_mutex> lock(mutex_);
    if (id >= 0 && id < static_cast<int64_t>(chunks_.size()) && chunks_[id]) {
        chunks_[id]->lightSources = lights;
    }
}

void Chunks::ensureCapacity(uint32_t totalSlots) {
    std::unique_lock<std::recursive_mutex> lock(mutex_);
    if (totalSlots <= chunks_.size()) return;

    auto framework = Renderer::instance().framework();
    auto vma = framework->vma();
    auto device = framework->device();

    uint32_t oldSize = static_cast<uint32_t>(chunks_.size());
    chunks_.resize(totalSlots);
    chunkBuildDatas_.resize(totalSlots);

    for (uint32_t i = oldSize; i < totalSlots; i++) {
        chunks_[i] = Chunk1::create();
        chunkBuildDatas_[i] = nullptr;
    }

    // Reallocate chunkPackedData SSBO to fit new size
    auto newPackedData = vk::HostVisibleBuffer::create(
        vma, device, totalSlots * sizeof(ChunkPackedData),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    // Copy existing data
    if (chunkPackedData_ && oldSize > 0) {
        std::memcpy(newPackedData->mappedPtr(), chunkPackedData_->mappedPtr(),
                    oldSize * sizeof(ChunkPackedData));
    }
    // Zero new entries
    std::memset(static_cast<uint8_t*>(newPackedData->mappedPtr()) + oldSize * sizeof(ChunkPackedData),
                0, (totalSlots - oldSize) * sizeof(ChunkPackedData));
    chunkPackedData_ = newPackedData;

    std::cout << "[ExtendedRD] Chunk array grown: " << oldSize << " -> " << totalSlots << std::endl;
}

void Chunks::submitExtendedBuild(uint32_t extId, std::shared_ptr<ChunkBuildData> cbd) {
    std::unique_lock<std::recursive_mutex> lock(mutex_);

    // Grow array if needed
    if (extId >= chunks_.size()) {
        ensureCapacity(extId + 256); // Grow in batches of 256
    }

    chunkBuildDatas_[extId] = cbd;
    glm::vec3 camPos = Renderer::instance().world()
        ? glm::vec3(Renderer::instance().world()->getCameraPos()) : glm::vec3(0);
    if (chunkBuildScheduler_) {
        chunkBuildScheduler_->enqueue(cbd, camPos, false);
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