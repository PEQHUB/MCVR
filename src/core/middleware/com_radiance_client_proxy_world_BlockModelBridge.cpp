#include <jni.h>
#include "core/render/renderer.hpp"
#include "core/render/render_framework.hpp"
#include "core/render/block_model_table.hpp"
#include "core/render/block_state_registry.hpp"

#include <iostream>

// ---- Block model table (unchanged) ----

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_proxy_world_BlockModelBridge_uploadBlockModelTable(
    JNIEnv *, jclass,
    jlong entryPtr, jint entryCount,
    jlong quadPtr, jint quadCount) {

    auto* entries = reinterpret_cast<const BlockModelEntry*>(entryPtr);
    auto* quads = reinterpret_cast<const BlockModelQuad*>(quadPtr);

    Renderer::blockModelTable.load(entries, static_cast<uint32_t>(entryCount),
                                   quads, static_cast<uint32_t>(quadCount));

    // Pre-normalize quad UVs if TextureSystem is already finalized.
    // (Texture finalize happens before model table upload, so this is the normal path.)
    if (Renderer::textureSystem.isFinalized()) {
        Renderer::blockModelTable.normalizeQuadUVs();
    }
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_proxy_world_BlockModelBridge_uploadBiomeTintTable(
    JNIEnv *, jclass,
    jlong tintPtr, jint tintCount) {

    auto* tints = reinterpret_cast<const BiomeTintEntry*>(tintPtr);
    Renderer::blockModelTable.loadBiomeTints(tints, static_cast<uint32_t>(tintCount));
}

// ---- TextureSystem: new C++-owned texture pipeline ----

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_proxy_world_BlockModelBridge_nativeReceiveSpriteTable(
    JNIEnv *, jclass,
    jlong metaPtr, jint count, jint atlasWidth, jint atlasHeight) {

    auto* table = reinterpret_cast<const TextureSystem::SpriteMetadata*>(metaPtr);
    Renderer::textureSystem.receiveSpriteTable(
        table, static_cast<uint32_t>(count),
        static_cast<uint32_t>(atlasWidth), static_cast<uint32_t>(atlasHeight));
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_proxy_world_BlockModelBridge_nativeReceiveSpritePixels(
    JNIEnv *, jclass,
    jlong dataPtr, jint totalBytes) {

    auto* data = reinterpret_cast<const uint8_t*>(dataPtr);
    Renderer::textureSystem.receiveSpritePixels(data, static_cast<uint32_t>(totalBytes));
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_proxy_world_BlockModelBridge_nativeReceiveAnimationFrames(
    JNIEnv *, jclass,
    jlong dataPtr, jint totalBytes) {

    auto* data = reinterpret_cast<const uint8_t*>(dataPtr);
    Renderer::textureSystem.receiveAnimationFrames(data, static_cast<uint32_t>(totalBytes));
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_proxy_world_BlockModelBridge_nativeReceiveSpriteAuxPixels(
    JNIEnv *, jclass,
    jlong specPtr, jlong normPtr, jint totalBytesPerType) {

    auto* specData = reinterpret_cast<const uint8_t*>(specPtr);
    auto* normData = reinterpret_cast<const uint8_t*>(normPtr);
    Renderer::textureSystem.receiveAuxPixels(
        specData, normData, static_cast<uint32_t>(totalBytesPerType));
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_proxy_world_BlockModelBridge_nativeUpdateSpecularLayer(
    JNIEnv *, jclass, jint spriteId, jlong pixelPtr, jint sizeBytes) {

    auto& ts = Renderer::textureSystem;
    if (!ts.isFinalized() || ts.blockSpecularArrayId() == UINT32_MAX) return;
    if (spriteId < 0 || static_cast<uint32_t>(spriteId) >= ts.spriteCount()) return;
    if (pixelPtr == 0 || sizeBytes <= 0) return;
    ts.arrayManager().stageLayerPixels(
        ts.blockSpecularArrayId(), static_cast<uint32_t>(spriteId), 0,
        reinterpret_cast<const uint8_t*>(pixelPtr), static_cast<size_t>(sizeBytes));
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_proxy_world_BlockModelBridge_nativeUpdateNormalLayer(
    JNIEnv *, jclass, jint spriteId, jlong pixelPtr, jint sizeBytes) {

    auto& ts = Renderer::textureSystem;
    if (!ts.isFinalized() || ts.blockNormalArrayId() == UINT32_MAX) return;
    if (spriteId < 0 || static_cast<uint32_t>(spriteId) >= ts.spriteCount()) return;
    if (pixelPtr == 0 || sizeBytes <= 0) return;
    ts.arrayManager().stageLayerPixels(
        ts.blockNormalArrayId(), static_cast<uint32_t>(spriteId), 0,
        reinterpret_cast<const uint8_t*>(pixelPtr), static_cast<size_t>(sizeBytes));
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_proxy_world_BlockModelBridge_nativeTextureFinalize(
    JNIEnv *, jclass) {

    auto framework = Renderer::instance().framework();
    auto vma = framework->vma();
    auto device = framework->device();

    Renderer::textureSystem.finalize(vma, device);

    // Pre-normalize quad UVs now that TextureSystem has atlas bounds.
    // This converts atlas-space UVs in all BlockModelQuads to [0,1] within sprite bounds,
    // so the mesher doesn't need to do per-vertex normalization.
    if (Renderer::blockModelTable.isLoaded()) {
        Renderer::blockModelTable.normalizeQuadUVs();
    }
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_proxy_world_BlockModelBridge_nativeSetTextureGeneration(
    JNIEnv *, jclass, jlong generation) {
    Renderer::textureSystem.setGeneration(static_cast<uint64_t>(generation));
}

// ---- Block state registry (for C++-only chunk loading from region files) ----

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_proxy_world_BlockModelBridge_nativeUploadBlockStateRegistry(
    JNIEnv *, jclass,
    jlong dataPtr, jint dataSize) {

    auto* data = reinterpret_cast<const uint8_t*>(dataPtr);
    Renderer::blockStateRegistry.load(data, static_cast<uint32_t>(dataSize));
}

// ---- Animation tick (called per game tick from Java) ----

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_proxy_world_BlockModelBridge_nativeTickAnimation(
    JNIEnv *, jclass, jint gameTick, jlong generation) {
    Renderer::options.textureArrayAnimationUpdatesEnabled = true;
    // Reject animation ticks from stale texture generations (e.g., after F3+T reload
    // but before the new texture system is finalized). This prevents uploading
    // animation frames from the old atlas into the new texture arrays.
    auto currentGen = Renderer::textureSystem.generation();
    if (generation != 0 && generation != currentGen) {
        return;  // stale tick — skip
    }
    Renderer::textureSystem.tickAnimation(static_cast<uint32_t>(gameTick));
}

// ---- Set texture array animation enabled state ----
extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_proxy_world_BlockModelBridge_nativeSetTextureArrayAnimationEnabled(
    JNIEnv *, jclass, jboolean enabled) {
    Renderer::options.textureArrayAnimationUpdatesEnabled = (enabled == JNI_TRUE);
}
