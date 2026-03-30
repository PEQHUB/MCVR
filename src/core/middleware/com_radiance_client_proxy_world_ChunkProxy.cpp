#include "com_radiance_client_proxy_world_ChunkProxy.h"

#include "core/render/chunks.hpp"
#include "core/render/renderer.hpp"

#include <cstring>
#include <iostream>

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_proxy_world_ChunkProxy_initNative(JNIEnv *, jclass, jint chunkNum) {
    Renderer::instance().world()->chunks()->reset(chunkNum);
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
    auto world = Renderer::instance().world();
    if (world == nullptr) return;
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

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_proxy_world_ChunkProxy_rebuildSingleBlockStates(
    JNIEnv *, jclass,
    jint originX, jint originY, jint originZ, jlong index,
    jlong blockStateArrayPtr, jlong palettePtr, jint paletteSize,
    jlong biomeDataPtr, jlong neighborFacesPtr, jint blockAtlasTextureId,
    jboolean important,
    jint biomeGrassColor, jint biomeFoliageColor, jint biomeWaterColor) {

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
    auto world = Renderer::instance().world();
    if (world == nullptr)
        return false;
    else
        return world->chunks()->isChunkReady(id);
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_proxy_world_ChunkProxy_invalidateSingle(JNIEnv *, jclass, jlong index) {
    auto world = Renderer::instance().world();
    if (world == nullptr) return;
    world->chunks()->invalidateChunk(index);
}

extern "C" JNIEXPORT jint JNICALL Java_com_radiance_client_proxy_world_ChunkProxy_nativeGetInputQueueSize(JNIEnv *, jclass) {
    auto world = Renderer::instance().world();
    if (!world) return 0;
    return static_cast<jint>(world->chunks()->getInputQueueSize());
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_proxy_world_ChunkProxy_setChunkLights(JNIEnv *, jclass, jlong chunkIndex, jint lightCount, jlong lightDataPtr) {
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