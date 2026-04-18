#include "com_radiance_client_proxy_world_ChunkProxy.h"

#include "core/render/chunks.hpp"
#include "core/render/block_mesher.hpp"
#include "core/render/renderer.hpp"

#include "engine/diagnostics/log.hpp"

#ifdef MCVR_ENABLE_ENGINE_V2
#include "engine/app/engine_app.hpp"
#include "engine/app/engine_services.hpp"
#include "engine/bridge/bridge_service.hpp"
#include "engine/scene/scene_service.hpp"
#include "engine/scene/chunk_registry.hpp"
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <mutex>

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_proxy_world_ChunkProxy_initNative(JNIEnv *, jclass, jint chunkNum) {
    if (!Renderer::is_initialized()) return;
    Renderer::instance().world()->chunks()->reset(chunkNum);

    // Compute Java's render distance from chunk count:
    // numChunks = (2*RD+1)^2 * 24 → RD = (sqrt(numChunks/24) - 1) / 2
    uint32_t javaRD = static_cast<uint32_t>((std::sqrt(chunkNum / 24.0) - 1) / 2);
    Renderer::instance().world()->startExtendedChunkLoading(javaRD, chunkNum);
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_proxy_world_ChunkProxy_nativeSetWorldRegionPath(
    JNIEnv *env, jclass, jstring jpath) {
    if (!Renderer::is_initialized()) return;
    if (!jpath) return;
    const char* utf = env->GetStringUTFChars(jpath, nullptr);
    if (utf) {
        Renderer::worldRegionPath = utf;
        engine::log::info("ExtendedRD", "World region path: " + Renderer::worldRegionPath);
        env->ReleaseStringUTFChars(jpath, utf);
    }
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_proxy_world_ChunkProxy_rebuildSingle(JNIEnv *,
                                                                                     jclass,
                                                                                     jint originX,
                                                                                     jint originY,
                                                                                     jint originZ,
                                                                                     jlong index,
                                                                                     jint geometryCount,
                                                                                     jlong geometryTypes,
                                                                                     jlong geometryTextures,
                                                                                     jlong vertexFormats,
                                                                                     jlong vertexCounts,
                                                                                     jlong vertexAddrs,
                                                                                     jboolean important) {
    if (Renderer::is_initialized()) {
        auto world = Renderer::instance().world();
        if (world != nullptr) {
            world->chunks()->queueChunkBuild(ChunkBuildTask{
                .x = originX,
                .y = originY,
                .z = originZ,
                .id = index,
                .geometryCount = geometryCount,
                .geometryTypes = reinterpret_cast<int *>(geometryTypes),
                .geometryTextures = reinterpret_cast<int *>(geometryTextures),
                .vertexFormats = reinterpret_cast<int *>(vertexFormats),
                .vertexCounts = reinterpret_cast<int *>(vertexCounts),
                .vertices = reinterpret_cast<vk::VertexFormat::PBRTriangle **>(vertexAddrs),
                .isImportant = static_cast<bool>(important),
            });
        }
    }

#ifdef MCVR_ENABLE_ENGINE_V2
    // V2 path: combine all layers into a single ChunkGeometry.
    // Section-Y is included in the key so all 24 vertical sections of a column
    // get distinct BLAS entries instead of overwriting each other.
    auto* app = engine::EngineApp::get();
    if (app && app->isInitialized() && app->mode() == engine::EngineMode::V2) {
        const int* vCounts = reinterpret_cast<const int*>(vertexCounts);
        const auto** vAddrs = reinterpret_cast<const uint8_t**>(vertexAddrs);
        constexpr int VERTEX_STRIDE = 96;

        // Calculate totals
        uint32_t totalVerts = 0;
        for (int i = 0; i < geometryCount; i++) totalVerts += vCounts[i];
        if (totalVerts == 0) return;

        // Quads → triangles: 4 verts per quad → 6 indices
        uint32_t totalTris = totalVerts / 4 * 2;

        engine::ChunkGeometry geo;
        // Section-level key. originX/Y/Z are world-space block coords (multiples of 16).
        geo.id = {originX >> 4, originY >> 4, originZ >> 4};
        geo.originX = static_cast<float>(originX);
        geo.originY = static_cast<float>(originY);
        geo.originZ = static_cast<float>(originZ);
        geo.triangleCount = totalTris;

        // Copy vertex data from all layers
        geo.vertexData.resize(totalVerts * VERTEX_STRIDE);
        size_t offset = 0;
        for (int i = 0; i < geometryCount; i++) {
            size_t layerSize = vCounts[i] * VERTEX_STRIDE;
            if (vAddrs[i] && layerSize > 0) {
                std::memcpy(geo.vertexData.data() + offset, vAddrs[i], layerSize);
                offset += layerSize;
            }
        }

        // Generate indices: quads → triangles (0,1,2, 2,3,0)
        geo.indexData.reserve(totalTris * 3);
        uint32_t baseVert = 0;
        for (int i = 0; i < geometryCount; i++) {
            uint32_t quadCount = vCounts[i] / 4;
            for (uint32_t q = 0; q < quadCount; q++) {
                uint32_t v = baseVert + q * 4;
                geo.indexData.push_back(v);
                geo.indexData.push_back(v + 1);
                geo.indexData.push_back(v + 2);
                geo.indexData.push_back(v + 2);
                geo.indexData.push_back(v + 3);
                geo.indexData.push_back(v);
            }
            baseVert += vCounts[i];
        }

        engine::CmdChunkSubmit cmd;
        cmd.chunkX   = geo.id.x;
        cmd.sectionY = geo.id.y;
        cmd.chunkZ   = geo.id.z;
        cmd.originX  = static_cast<int32_t>(geo.originX);
        cmd.originY  = static_cast<int32_t>(geo.originY);
        cmd.originZ  = static_cast<int32_t>(geo.originZ);
        cmd.triangleCount = geo.triangleCount;
        cmd.vertexData = std::move(geo.vertexData);
        cmd.indexData = std::move(geo.indexData);
        app->services().bridge().post(std::move(cmd));
    }
#endif
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_proxy_world_ChunkProxy_rebuildSingleBlockStates(
    JNIEnv *, jclass,
    jint originX, jint originY, jint originZ, jlong index,
    jlong blockStateArrayPtr, jlong palettePtr, jint paletteSize,
    jlong biomeDataPtr, jlong neighborFacesPtr, jint blockAtlasTextureId,
    jboolean important,
    jint biomeGrassColor, jint biomeFoliageColor, jint biomeWaterColor) {
    // DIAGNOSTIC: log first calls to verify this JNI fires
    static std::atomic<uint32_t> rsbsCount{0};
    uint32_t rn = rsbsCount.fetch_add(1) + 1;
    if (rn <= 10 || rn % 500 == 0) {
#ifdef MCVR_ENABLE_ENGINE_V2
        const bool v2Init = (engine::EngineApp::get() && engine::EngineApp::get()->isInitialized());
#else
        const bool v2Init = false;
#endif
        engine::log::info("rebuildSingleBlockStates",
            "#" + std::to_string(rn) +
            " origin=(" + std::to_string(originX) + "," + std::to_string(originY) + "," + std::to_string(originZ) + ")" +
            " rendererInit=" + (Renderer::is_initialized() ? "Y" : "N") +
            " mtl=" + (Renderer::blockModelTable.isLoaded() ? "Y" : "N") +
            " v2App=" + (v2Init ? "Y" : "N"));
    }

    // Decode palette-indexed block states to global state IDs (needed by both V1 and V2 paths)
    uint32_t decodedBlockStates[4096];
    const uint16_t* paletteIndices = reinterpret_cast<const uint16_t*>(blockStateArrayPtr);
    const uint32_t* palette = reinterpret_cast<const uint32_t*>(palettePtr);
    for (int i = 0; i < 4096; i++) {
        uint16_t idx = paletteIndices[i];
        decodedBlockStates[i] = (idx < paletteSize) ? palette[idx] : 0;
    }

#ifdef MCVR_ENABLE_ENGINE_V2
    // V2 direct path: when V2 engine is active (and therefore legacy Renderer is NOT initialized),
    // we mesh the section in-process and post CmdChunkSubmit straight to the V2 bridge.
    // This bypasses the legacy Chunks::queueBlockStateBuild system entirely.
    auto* app = engine::EngineApp::get();
    if (app != nullptr && app->isInitialized() && Renderer::blockModelTable.isLoaded()) {
        // Task 1.3 diagnostic (copper-grate hunt): one-shot pre-mesh dump of
        // BlockModelTable entries for the first 20 non-empty renderType==1 states.
        // Runs BEFORE BlockMesher::mesh, so it is not contaminated by greedy merging.
        // Distinguishes H-A (Java uploaded all-zero spriteIds) vs H-C (mesher merging)
        // vs H-D/E (corruption downstream of mesher).
        static std::atomic<bool> preMeshDumped{false};
        bool expected = false;
        if (preMeshDumped.compare_exchange_strong(expected, true)) {
            uint32_t dumped = 0;
            uint32_t totalQuads = 0;
            uint64_t sumIds = 0;
            uint32_t nonZeroIds = 0;
            std::string sample;
            sample.reserve(4096);
            for (uint32_t s = 1; s < Renderer::blockModelTable.maxStateId() && dumped < 20; ++s) {
                const auto* e = Renderer::blockModelTable.getEntry(s);
                if (!e || e->renderType != 1 || e->totalQuadCount == 0) continue;
                sample.append(" S").append(std::to_string(s)).append(":[");
                for (uint8_t dir = 0; dir < 6; ++dir) {
                    uint8_t cnt = 0;
                    const auto* q = Renderer::blockModelTable.getFaceQuads(*e, dir, cnt);
                    if (dir > 0) sample.append("|");
                    for (uint8_t qi = 0; qi < cnt; ++qi) {
                        uint16_t sid = q[qi].spriteId;
                        sample.append(std::to_string(sid));
                        if (qi + 1 < cnt) sample.append(",");
                        ++totalQuads;
                        sumIds += sid;
                        if (sid != 0) ++nonZeroIds;
                    }
                }
                sample.append("]");
                ++dumped;
            }
            engine::log::warn("V2-DIAG",
                "pre-mesh BlockModelTable dump: states=" + std::to_string(dumped) +
                " totalQuads=" + std::to_string(totalQuads) +
                " nonZeroSpriteIds=" + std::to_string(nonZeroIds) +
                " avgSpriteId=" + std::to_string(totalQuads ? (sumIds / totalQuads) : 0) +
                " sample:" + sample);
        }

        BlockMesher::SectionInput input;
        std::memcpy(input.blockStates, decodedBlockStates, sizeof(input.blockStates));
        if (biomeDataPtr != 0) {
            std::memcpy(input.biomes, reinterpret_cast<const void*>(biomeDataPtr), sizeof(input.biomes));
        } else {
            std::memset(input.biomes, 0, sizeof(input.biomes));
        }
        if (neighborFacesPtr != 0) {
            std::memcpy(input.neighborStates, reinterpret_cast<const void*>(neighborFacesPtr),
                        sizeof(input.neighborStates));
        } else {
            std::memset(input.neighborStates, 0, sizeof(input.neighborStates));
        }
        input.originX = originX;
        input.originY = originY;
        input.originZ = originZ;
        input.blockAtlasTextureId = static_cast<uint32_t>(blockAtlasTextureId);

        auto meshOutput = BlockMesher::mesh(input, Renderer::blockModelTable);

        // Count vertices + indices across all layers. PBRTriangle is a single vertex (96B).
        const size_t solidV = meshOutput.solidVertices.size();
        const size_t solidI = meshOutput.solidIndices.size();
        const size_t cutoutV = meshOutput.cutoutVertices.size();
        const size_t cutoutI = meshOutput.cutoutIndices.size();
        const size_t transV = meshOutput.translucentVertices.size();
        const size_t transI = meshOutput.translucentIndices.size();

        const size_t totalV = solidV + cutoutV + transV;
        const size_t totalI = solidI + cutoutI + transI;

        if (totalI == 0) {
            return;  // empty section (air only)
        }

        const size_t vertexByteSize = totalV * sizeof(vk::VertexFormat::PBRTriangle);

        std::vector<uint8_t> vertexBuffer(vertexByteSize);
        std::vector<uint32_t> indexBuffer(totalI);

        auto* vOut = reinterpret_cast<vk::VertexFormat::PBRTriangle*>(vertexBuffer.data());
        uint32_t* iOut = indexBuffer.data();

        // Solid layer - copy vertices verbatim, copy indices unchanged (base=0).
        if (solidV > 0) {
            std::memcpy(vOut, meshOutput.solidVertices.data(),
                        solidV * sizeof(vk::VertexFormat::PBRTriangle));
        }
        if (solidI > 0) {
            std::memcpy(iOut, meshOutput.solidIndices.data(), solidI * sizeof(uint32_t));
        }

        // Cutout layer - copy vertices with offset, re-offset indices by solidV.
        if (cutoutV > 0) {
            std::memcpy(vOut + solidV, meshOutput.cutoutVertices.data(),
                        cutoutV * sizeof(vk::VertexFormat::PBRTriangle));
        }
        for (size_t i = 0; i < cutoutI; i++) {
            iOut[solidI + i] = meshOutput.cutoutIndices[i] + static_cast<uint32_t>(solidV);
        }

        // Translucent layer - copy vertices with offset, re-offset indices by solidV+cutoutV.
        if (transV > 0) {
            std::memcpy(vOut + solidV + cutoutV, meshOutput.translucentVertices.data(),
                        transV * sizeof(vk::VertexFormat::PBRTriangle));
        }
        for (size_t i = 0; i < transI; i++) {
            iOut[solidI + cutoutI + i] =
                meshOutput.translucentIndices[i] + static_cast<uint32_t>(solidV + cutoutV);
        }

        const size_t totalTris = totalI / 3;

        // DIAGNOSTIC: dump unique textureIDs in emitted vertex buffer for first few sections.
        // This is the decisive check — if all verts share one ID, bug is upstream (mesher/table).
        // If varied, bug is GPU-side (BDA/descriptor/sampler/layout).
        static std::atomic<uint32_t> v2DiagDumpCount{0};
        uint32_t dumpN = v2DiagDumpCount.fetch_add(1) + 1;
        if (dumpN <= 5) {
            const auto* verts = reinterpret_cast<const vk::VertexFormat::PBRTriangle*>(vertexBuffer.data());
            uint32_t uniq[64]; uint32_t uniqCount = 0;
            uint32_t minId = 0xFFFFFFFFu, maxId = 0;
            for (size_t v = 0; v < totalV; ++v) {
                uint32_t tid = verts[v].textureID;
                if (tid < minId) minId = tid;
                if (tid > maxId) maxId = tid;
                bool found = false;
                for (uint32_t u = 0; u < uniqCount; ++u) if (uniq[u] == tid) { found = true; break; }
                if (!found && uniqCount < 64) uniq[uniqCount++] = tid;
            }
            std::string sample;
            for (uint32_t u = 0; u < std::min(uniqCount, 32u); ++u) {
                if (u > 0) sample.append(",");
                sample.append(std::to_string(uniq[u]));
            }
            engine::log::warn("V2-DIAG",
                "vbuf#" + std::to_string(dumpN) +
                " origin=(" + std::to_string(originX) + "," + std::to_string(originY) + "," + std::to_string(originZ) + ")" +
                " verts=" + std::to_string(totalV) +
                " uniqIDs=" + std::to_string(uniqCount) +
                " range=[" + std::to_string(minId) + ".." + std::to_string(maxId) + "]" +
                " ids:" + sample);
        }

        // Task 1.2 diagnostic (copper-grate hunt): whole-frame textureID histogram.
        // Accumulates across ALL sections during a 10 s window starting at first submit,
        // then emits a single summary line. Low-bucket dominance => all quads share one
        // (or a few) spriteIds (H-A/H-B). Even distribution => bug is downstream.
        {
            static std::mutex histMu;
            static std::array<uint64_t, 16> histBuckets{};
            // Top-unique tracking: small LRU of (textureID -> count).
            static std::array<std::pair<uint32_t, uint64_t>, 64> topUniqueArr{};
            static uint32_t topUniqueCount = 0;
            static uint64_t totalSamples = 0;
            static uint64_t uniqueSamples = 0;
            static std::chrono::steady_clock::time_point windowStart{};
            static bool windowStarted = false;
            static bool windowEmitted = false;

            const auto* vertsHist = reinterpret_cast<const vk::VertexFormat::PBRTriangle*>(vertexBuffer.data());
            std::lock_guard<std::mutex> lock(histMu);
            if (!windowStarted) {
                windowStart = std::chrono::steady_clock::now();
                windowStarted = true;
            }
            const auto now = std::chrono::steady_clock::now();
            const auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(now - windowStart).count();

            if (!windowEmitted && elapsedMs <= 10000) {
                for (size_t v = 0; v < totalV; ++v) {
                    uint32_t tid = vertsHist[v].textureID;
                    histBuckets[tid & 0xFu]++;
                    totalSamples++;
                    // Count unique textureIDs (bounded to 64 distinct values).
                    bool found = false;
                    for (uint32_t u = 0; u < topUniqueCount; ++u) {
                        if (topUniqueArr[u].first == tid) { topUniqueArr[u].second++; found = true; break; }
                    }
                    if (!found) {
                        if (topUniqueCount < topUniqueArr.size()) {
                            topUniqueArr[topUniqueCount++] = {tid, 1};
                            uniqueSamples++;
                        }
                        // else: drop — table saturated
                    }
                }
            } else if (!windowEmitted && elapsedMs > 10000 && totalSamples > 0) {
                windowEmitted = true;
                std::string bucketStr;
                bucketStr.reserve(256);
                for (uint32_t b = 0; b < 16; ++b) {
                    if (b > 0) bucketStr.append(",");
                    bucketStr.append(std::to_string(histBuckets[b]));
                }
                // Sort top-unique descending by count, take top 8.
                std::array<std::pair<uint32_t, uint64_t>, 64> sorted = topUniqueArr;
                uint32_t n = topUniqueCount;
                std::sort(sorted.begin(), sorted.begin() + n,
                    [](const auto& a, const auto& b){ return a.second > b.second; });
                std::string topStr;
                topStr.reserve(256);
                uint32_t top = std::min<uint32_t>(8, n);
                for (uint32_t u = 0; u < top; ++u) {
                    if (u > 0) topStr.append(" ");
                    topStr.append("id")
                          .append(std::to_string(sorted[u].first))
                          .append("=")
                          .append(std::to_string(sorted[u].second));
                }
                engine::log::warn("V2-DIAG",
                    "textureID histogram over " + std::to_string(totalSamples) + " verts ("
                    + std::to_string(uniqueSamples) + "+ unique) buckets[0..15]=" + bucketStr
                    + " top8:" + topStr);
            }
        }

        // Post CmdChunkSubmit to the V2 bridge.
        engine::CmdChunkSubmit cmd{};
        cmd.chunkX = originX >> 4;
        cmd.sectionY = originY >> 4;
        cmd.chunkZ = originZ >> 4;
        cmd.originX = originX;
        cmd.originY = originY;
        cmd.originZ = originZ;
        cmd.vertexData = std::move(vertexBuffer);
        cmd.indexData = std::move(indexBuffer);
        cmd.triangleCount = static_cast<uint32_t>(totalTris);

        app->services().bridge().post(cmd);
        // DIAGNOSTIC: log first 5 + every 200th V2 submit
        static std::atomic<uint32_t> v2SubmitCount{0};
        uint32_t sn = v2SubmitCount.fetch_add(1) + 1;
        if (sn <= 5 || sn % 200 == 0) {
            engine::log::info("V2 submit",
                "#" + std::to_string(sn) +
                " chunk=(" + std::to_string(cmd.chunkX) + "," + std::to_string(cmd.sectionY) + "," + std::to_string(cmd.chunkZ) + ")" +
                " origin=(" + std::to_string(originX) + "," + std::to_string(originY) + "," + std::to_string(originZ) + ")" +
                " tris=" + std::to_string(totalTris) + " verts=" + std::to_string(totalV) +
                " (sV=" + std::to_string(solidV) + " cV=" + std::to_string(cutoutV) + " tV=" + std::to_string(transV) + ")");
        }
        return;
    }
#endif

    // Legacy V1 path: queue to Chunks via Renderer
    if (!Renderer::is_initialized()) return;

    auto world = Renderer::instance().world();
    if (world == nullptr) return;

    ChunkBuildTaskV2 task;
    task.x = originX;
    task.y = originY;
    task.z = originZ;
    task.id = index;
    task.blockAtlasTextureId = static_cast<uint32_t>(blockAtlasTextureId);
    task.isImportant = static_cast<bool>(important);
    task.biomeGrassColor = static_cast<uint32_t>(biomeGrassColor);
    task.biomeFoliageColor = static_cast<uint32_t>(biomeFoliageColor);
    task.biomeWaterColor = static_cast<uint32_t>(biomeWaterColor);

    std::memcpy(task.blockStates, decodedBlockStates, sizeof(task.blockStates));

    // Copy biome data
    if (biomeDataPtr != 0) {
        std::memcpy(task.biomes, reinterpret_cast<const void*>(biomeDataPtr), sizeof(task.biomes));
    } else {
        std::memset(task.biomes, 0, sizeof(task.biomes));
    }

    // Copy neighbor face data
    if (neighborFacesPtr != 0) {
        std::memcpy(task.neighborStates, reinterpret_cast<const void*>(neighborFacesPtr),
                    sizeof(task.neighborStates));
    } else {
        std::memset(task.neighborStates, 0, sizeof(task.neighborStates));
    }

    world->chunks()->queueBlockStateBuild(std::move(task));
}

extern "C" JNIEXPORT jboolean JNICALL Java_com_radiance_client_proxy_world_ChunkProxy_isChunkReady(JNIEnv *, jclass, jlong id) {
    if (!Renderer::is_initialized()) return JNI_FALSE;
    auto world = Renderer::instance().world();
    if (world == nullptr)
        return false;
    else
        return world->chunks()->isChunkReady(id);
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_proxy_world_ChunkProxy_invalidateSingle(JNIEnv *, jclass, jlong index) {
    if (!Renderer::is_initialized()) return;
    auto world = Renderer::instance().world();
    if (world == nullptr) return;
    world->chunks()->invalidateChunk(index);
}

extern "C" JNIEXPORT jint JNICALL Java_com_radiance_client_proxy_world_ChunkProxy_nativeGetInputQueueSize(JNIEnv *, jclass) {
    if (!Renderer::is_initialized()) return 0;
    auto world = Renderer::instance().world();
    if (!world) return 0;
    return static_cast<jint>(world->chunks()->getInputQueueSize());
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_proxy_world_ChunkProxy_setChunkLights(JNIEnv *, jclass, jlong chunkIndex, jint lightCount, jlong lightDataPtr) {
    if (!Renderer::is_initialized()) return;
    auto world = Renderer::instance().world();
    if (world == nullptr) return;

    std::vector<ChunkLightEntry> lights;
    if (lightCount > 0 && lightDataPtr != 0) {
        const uint8_t *ptr = reinterpret_cast<const uint8_t *>(lightDataPtr);
        lights.reserve(lightCount);
        for (int i = 0; i < lightCount; i++) {
            ChunkLightEntry entry;
            entry.worldX = *reinterpret_cast<const float *>(ptr);
            entry.worldY = *reinterpret_cast<const float *>(ptr + 4);
            entry.worldZ = *reinterpret_cast<const float *>(ptr + 8);
            entry.lightTypeId = *reinterpret_cast<const int *>(ptr + 12);
            lights.push_back(entry);
            ptr += 16;
        }
    }
    world->chunks()->setChunkLights(chunkIndex, lights);
}