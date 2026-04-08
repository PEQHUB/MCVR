#include <jni.h>
#include "core/render/renderer.hpp"
#include "core/render/render_framework.hpp"
#include "core/render/block_model_table.hpp"
#include "core/render/block_state_registry.hpp"
#include "engine/diagnostics/log.hpp"

// ---- Block model table (unchanged) ----

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_proxy_world_BlockModelBridge_uploadBlockModelTable(
    JNIEnv *, jclass,
    jlong entryPtr, jint entryCount,
    jlong quadPtr, jint quadCount) {
    // V2 mode: Renderer is never initialized but blockModelTable is a static class
    // member that exists independently. The mesher only needs the static data.
    auto* entries = reinterpret_cast<const BlockModelEntry*>(entryPtr);
    auto* quads = reinterpret_cast<const BlockModelQuad*>(quadPtr);

    Renderer::blockModelTable.load(entries, static_cast<uint32_t>(entryCount),
                                   quads, static_cast<uint32_t>(quadCount));

    engine::log::info("BlockModelBridge",
        "uploadBlockModelTable: entries=" + std::to_string(entryCount) +
        " quads=" + std::to_string(quadCount) +
        " rendererInit=" + (Renderer::is_initialized() ? "yes" : "no") +
        " loaded=" + (Renderer::blockModelTable.isLoaded() ? "yes" : "no"));

    // Pre-normalize quad UVs if TextureSystem is already finalized.
    // (Texture finalize happens before model table upload, so this is the normal path.)
    // Skip if Renderer not initialized — TextureSystem state is moot in V2.
    if (Renderer::is_initialized() && Renderer::textureSystem.isFinalized()) {
        Renderer::blockModelTable.normalizeQuadUVs();
    }
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_proxy_world_BlockModelBridge_uploadBiomeTintTable(
    JNIEnv *, jclass,
    jlong tintPtr, jint tintCount) {
    // V2 mode: blockModelTable is static, no Renderer init needed.
    auto* tints = reinterpret_cast<const BiomeTintEntry*>(tintPtr);
    Renderer::blockModelTable.loadBiomeTints(tints, static_cast<uint32_t>(tintCount));

    engine::log::info("BlockModelBridge",
        "uploadBiomeTintTable: tints=" + std::to_string(tintCount) +
        " rendererInit=" + (Renderer::is_initialized() ? "yes" : "no"));
}

// ---- TextureSystem: new C++-owned texture pipeline ----

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_proxy_world_BlockModelBridge_nativeReceiveSpriteTable(
    JNIEnv *, jclass,
    jlong metaPtr, jint count, jint atlasWidth, jint atlasHeight) {
    if (!Renderer::is_initialized()) return;

    auto* table = reinterpret_cast<const TextureSystem::SpriteMetadata*>(metaPtr);
    Renderer::textureSystem.receiveSpriteTable(
        table, static_cast<uint32_t>(count),
        static_cast<uint32_t>(atlasWidth), static_cast<uint32_t>(atlasHeight));
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_proxy_world_BlockModelBridge_nativeReceiveSpritePixels(
    JNIEnv *, jclass,
    jlong dataPtr, jint totalBytes) {
    if (!Renderer::is_initialized()) return;

    auto* data = reinterpret_cast<const uint8_t*>(dataPtr);
    Renderer::textureSystem.receiveSpritePixels(data, static_cast<uint32_t>(totalBytes));
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_proxy_world_BlockModelBridge_nativeReceiveAnimationFrames(
    JNIEnv *, jclass,
    jlong dataPtr, jint totalBytes) {
    if (!Renderer::is_initialized()) return;

    auto* data = reinterpret_cast<const uint8_t*>(dataPtr);
    Renderer::textureSystem.receiveAnimationFrames(data, static_cast<uint32_t>(totalBytes));
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_proxy_world_BlockModelBridge_nativeReceiveSpriteAuxPixels(
    JNIEnv *, jclass,
    jlong specPtr, jlong normPtr, jint totalBytesPerType) {
    if (!Renderer::is_initialized()) return;

    auto* specData = reinterpret_cast<const uint8_t*>(specPtr);
    auto* normData = reinterpret_cast<const uint8_t*>(normPtr);
    Renderer::textureSystem.receiveAuxPixels(
        specData, normData, static_cast<uint32_t>(totalBytesPerType));
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_proxy_world_BlockModelBridge_nativeUpdateSpecularLayer(
    JNIEnv *, jclass, jint spriteId, jlong pixelPtr, jint sizeBytes) {
    if (!Renderer::is_initialized()) return;

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
    if (!Renderer::is_initialized()) return;

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
    if (!Renderer::is_initialized()) return;

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

// ---- Block state registry (for C++-only chunk loading from region files) ----

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_proxy_world_BlockModelBridge_nativeUploadBlockStateRegistry(
    JNIEnv *, jclass,
    jlong dataPtr, jint dataSize) {
    // V2 mode: blockStateRegistry is static, no Renderer init needed.
    auto* data = reinterpret_cast<const uint8_t*>(dataPtr);
    Renderer::blockStateRegistry.load(data, static_cast<uint32_t>(dataSize));

    engine::log::info("BlockModelBridge",
        "uploadBlockStateRegistry: bytes=" + std::to_string(dataSize) +
        " rendererInit=" + (Renderer::is_initialized() ? "yes" : "no") +
        " loaded=" + (Renderer::blockStateRegistry.isLoaded() ? "yes" : "no"));
}

// ---- Animation tick (called per game tick from Java) ----

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_proxy_world_BlockModelBridge_nativeTickAnimation(
    JNIEnv *, jclass,
    jint gameTick) {
    if (!Renderer::is_initialized()) return;

    Renderer::textureSystem.tickAnimation(static_cast<uint32_t>(gameTick));
}
