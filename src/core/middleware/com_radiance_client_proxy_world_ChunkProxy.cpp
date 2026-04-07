#include "com_radiance_client_proxy_world_ChunkProxy.h"

#include "core/render/chunks.hpp"
#include "core/render/renderer.hpp"

#ifdef MCVR_ENABLE_ENGINE_V2
#include "engine/app/engine_app.hpp"
#include "engine/app/engine_services.hpp"
#include "engine/bridge/bridge_service.hpp"
#include "engine/scene/scene_service.hpp"
#include "engine/scene/chunk_registry.hpp"
#endif

#include <cmath>
#include <cstring>
#include <iostream>

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
        std::cout << "[ExtendedRD] World region path: " << Renderer::worldRegionPath << std::endl;
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
    // V2 path: combine all layers into a single ChunkGeometry
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
        geo.id = {originX >> 4, originZ >> 4}; // chunk coords from block coords
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
        cmd.chunkX = geo.id.x;
        cmd.chunkZ = geo.id.z;
        cmd.originX = static_cast<int32_t>(geo.originX);
        cmd.originY = static_cast<int32_t>(geo.originY);
        cmd.originZ = static_cast<int32_t>(geo.originZ);
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


    // Decode palette-indexed block states to global state IDs
    const uint16_t* paletteIndices = reinterpret_cast<const uint16_t*>(blockStateArrayPtr);
    const uint32_t* palette = reinterpret_cast<const uint32_t*>(palettePtr);
    for (int i = 0; i < 4096; i++) {
        uint16_t idx = paletteIndices[i];
        task.blockStates[i] = (idx < paletteSize) ? palette[idx] : 0;
    }

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