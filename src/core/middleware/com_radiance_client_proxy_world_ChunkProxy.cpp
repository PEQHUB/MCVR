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

#include <cmath>
#include <cstring>

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