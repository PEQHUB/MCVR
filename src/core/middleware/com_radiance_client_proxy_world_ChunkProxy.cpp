#include "com_radiance_client_proxy_world_ChunkProxy.h"

#include "core/render/chunks.hpp"
#include "core/render/renderer.hpp"

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_proxy_world_ChunkProxy_initNative(
    JNIEnv *, jclass, jint chunkNum) {
    Renderer::instance().world()->chunks()->reset(chunkNum);
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_proxy_world_ChunkProxy_rebuildSingle(
    JNIEnv *,
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
    jboolean important,
    jlong textureGeneration) {

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
        .textureGeneration = static_cast<uint64_t>(textureGeneration),
    });
}

extern "C" JNIEXPORT jboolean JNICALL Java_com_radiance_client_proxy_world_ChunkProxy_isChunkReady(
    JNIEnv *, jclass, jlong id) {
    auto world = Renderer::instance().world();
    if (world == nullptr) {
        return false;
    }
    return world->chunks()->isChunkReady(id);
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_proxy_world_ChunkProxy_invalidateSingle(
    JNIEnv *, jclass, jlong index) {
    auto world = Renderer::instance().world();
    if (world == nullptr) return;
    world->chunks()->invalidateChunk(index);
}
