#include "com_radiance_client_proxy_world_ChunkProxy.h"

#include "core/render/chunks.hpp"
#include "core/render/renderer.hpp"

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