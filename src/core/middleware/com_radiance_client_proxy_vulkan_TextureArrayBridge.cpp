#include <jni.h>
#include "core/render/renderer.hpp"
#include "core/render/render_framework.hpp"

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_TextureArrayBridge_nativeReceiveSpriteTable(
    JNIEnv *, jclass,
    jlong metaPtr, jint count, jint atlasWidth, jint atlasHeight) {

    auto* table = reinterpret_cast<const TextureSystem::SpriteMetadata*>(metaPtr);
    Renderer::textureSystem.receiveSpriteTable(
        table, static_cast<uint32_t>(count),
        static_cast<uint32_t>(atlasWidth), static_cast<uint32_t>(atlasHeight));
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_TextureArrayBridge_nativeReceiveSpritePixels(
    JNIEnv *, jclass,
    jlong dataPtr, jint totalBytes) {

    auto* data = reinterpret_cast<const uint8_t*>(dataPtr);
    Renderer::textureSystem.receiveSpritePixels(data, static_cast<uint32_t>(totalBytes));
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_TextureArrayBridge_nativeReceiveAnimationFrames(
    JNIEnv *, jclass,
    jlong dataPtr, jint totalBytes) {

    auto* data = reinterpret_cast<const uint8_t*>(dataPtr);
    Renderer::textureSystem.receiveAnimationFrames(data, static_cast<uint32_t>(totalBytes));
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_TextureArrayBridge_nativeReceiveSpriteAuxPixels(
    JNIEnv *, jclass,
    jlong specPtr, jlong normPtr, jlong flagPtr, jint totalBytesPerType) {

    auto* specData = reinterpret_cast<const uint8_t*>(specPtr);
    auto* normData = reinterpret_cast<const uint8_t*>(normPtr);
    auto* flagData = reinterpret_cast<const uint8_t*>(flagPtr);
    Renderer::textureSystem.receiveAuxPixels(
        specData, normData, flagData, static_cast<uint32_t>(totalBytesPerType));
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_TextureArrayBridge_nativeUpdateAlbedoLayer(
    JNIEnv *, jclass, jint spriteId, jlong pixelPtr, jint sizeBytes, jlong generation) {

    auto& ts = Renderer::textureSystem;
    if (generation != 0 && static_cast<uint64_t>(generation) != ts.generation()) return;
    if (!ts.isFinalized() || ts.blockAlbedoArrayId() == UINT32_MAX) return;
    if (spriteId < 0 || static_cast<uint32_t>(spriteId) >= ts.spriteCount()) return;
    if (pixelPtr == 0 || sizeBytes <= 0) return;
    ts.arrayManager().stageLayerPixels(
        ts.blockAlbedoArrayId(), static_cast<uint32_t>(spriteId), 0,
        reinterpret_cast<const uint8_t*>(pixelPtr), static_cast<size_t>(sizeBytes));
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_TextureArrayBridge_nativeUpdateSpecularLayer(
    JNIEnv *, jclass, jint spriteId, jlong pixelPtr, jint sizeBytes, jlong generation) {

    auto& ts = Renderer::textureSystem;
    if (generation != 0 && static_cast<uint64_t>(generation) != ts.generation()) return;
    if (!ts.isFinalized() || ts.blockSpecularArrayId() == UINT32_MAX) return;
    if (spriteId < 0 || static_cast<uint32_t>(spriteId) >= ts.spriteCount()) return;
    if (pixelPtr == 0 || sizeBytes <= 0) return;
    ts.arrayManager().stageLayerPixels(
        ts.blockSpecularArrayId(), static_cast<uint32_t>(spriteId), 0,
        reinterpret_cast<const uint8_t*>(pixelPtr), static_cast<size_t>(sizeBytes));
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_TextureArrayBridge_nativeUpdateNormalLayer(
    JNIEnv *, jclass, jint spriteId, jlong pixelPtr, jint sizeBytes, jlong generation) {

    auto& ts = Renderer::textureSystem;
    if (generation != 0 && static_cast<uint64_t>(generation) != ts.generation()) return;
    if (!ts.isFinalized() || ts.blockNormalArrayId() == UINT32_MAX) return;
    if (spriteId < 0 || static_cast<uint32_t>(spriteId) >= ts.spriteCount()) return;
    if (pixelPtr == 0 || sizeBytes <= 0) return;
    ts.arrayManager().stageLayerPixels(
        ts.blockNormalArrayId(), static_cast<uint32_t>(spriteId), 0,
        reinterpret_cast<const uint8_t*>(pixelPtr), static_cast<size_t>(sizeBytes));
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_TextureArrayBridge_nativeUpdateFlagLayer(
    JNIEnv *, jclass, jint spriteId, jlong pixelPtr, jint sizeBytes, jlong generation) {

    auto& ts = Renderer::textureSystem;
    if (generation != 0 && static_cast<uint64_t>(generation) != ts.generation()) return;
    if (!ts.isFinalized() || ts.blockFlagArrayId() == UINT32_MAX) return;
    if (spriteId < 0 || static_cast<uint32_t>(spriteId) >= ts.spriteCount()) return;
    if (pixelPtr == 0 || sizeBytes <= 0) return;
    ts.arrayManager().stageLayerPixels(
        ts.blockFlagArrayId(), static_cast<uint32_t>(spriteId), 0,
        reinterpret_cast<const uint8_t*>(pixelPtr), static_cast<size_t>(sizeBytes));
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_TextureArrayBridge_nativeUpdateSpriteHeightMetadata(
    JNIEnv *, jclass, jint spriteId, jint flags, jint heightRangePacked, jlong generation) {

    auto& ts = Renderer::textureSystem;
    if (generation != 0 && static_cast<uint64_t>(generation) != ts.generation()) return;
    if (!ts.isFinalized()) return;
    if (spriteId < 0 || static_cast<uint32_t>(spriteId) >= ts.spriteCount()) return;

    auto renderer = Renderer::try_instance();
    if (!renderer || !renderer->framework()) return;
    auto framework = renderer->framework();
    ts.updateSpriteHeightMetadata(
        static_cast<uint32_t>(spriteId),
        static_cast<uint32_t>(flags),
        static_cast<int32_t>(heightRangePacked),
        framework->vma(),
        framework->device());
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_TextureArrayBridge_nativeReceiveTextureRules(
    JNIEnv *, jclass, jlong dataPtr, jint count, jlong generation) {

    auto& ts = Renderer::textureSystem;
    if (generation != 0 && static_cast<uint64_t>(generation) != ts.generation()) return;
    if (dataPtr == 0 || count <= 0) return;

    auto renderer = Renderer::try_instance();
    if (!renderer || !renderer->framework()) return;
    auto framework = renderer->framework();
    ts.textureRules().uploadRules(
        reinterpret_cast<const vk::Data::TextureRuleEntry*>(dataPtr),
        static_cast<uint32_t>(count),
        framework->vma(),
        framework->device());
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_TextureArrayBridge_nativeTextureFinalize(
    JNIEnv *, jclass) {

    auto framework = Renderer::instance().framework();
    auto vma = framework->vma();
    auto device = framework->device();

    Renderer::textureSystem.finalize(vma, device);
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_TextureArrayBridge_nativeSetTextureGeneration(
    JNIEnv *, jclass, jlong generation) {
    Renderer::textureSystem.setGeneration(static_cast<uint64_t>(generation));
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_TextureArrayBridge_nativeTickAnimation(
    JNIEnv *, jclass, jint gameTick, jlong generation) {
    Renderer::options.textureArrayAnimationUpdatesEnabled = true;
    auto currentGen = Renderer::textureSystem.generation();
    if (generation != 0 && generation != currentGen) {
        return;
    }
    Renderer::textureSystem.tickAnimation(static_cast<uint32_t>(gameTick));
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_TextureArrayBridge_nativeSetTextureArrayAnimationEnabled(
    JNIEnv *, jclass, jboolean enabled) {
    Renderer::options.textureArrayAnimationUpdatesEnabled = (enabled == JNI_TRUE);
}
