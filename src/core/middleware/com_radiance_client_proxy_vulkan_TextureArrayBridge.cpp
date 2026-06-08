#include <jni.h>
#include "core/render/renderer.hpp"
#include "core/render/render_framework.hpp"

#include <algorithm>

extern "C" JNIEXPORT jint JNICALL Java_com_radiance_client_proxy_vulkan_TextureArrayBridge_nativeMaxRenderableSpriteCount(
    JNIEnv *, jclass) {

    auto renderer = Renderer::try_instance();
    if (!renderer || !renderer->framework()) {
        return static_cast<jint>(vk::Data::SPRITE_MAX_ENTRIES);
    }
    auto framework = renderer->framework();
    if (!framework || !framework->physicalDevice()) {
        return static_cast<jint>(vk::Data::SPRITE_MAX_ENTRIES);
    }
    VkPhysicalDeviceProperties props = framework->physicalDevice()->properties();
    uint32_t capacity = std::min(props.limits.maxImageArrayLayers, vk::Data::SPRITE_MAX_ENTRIES);
    return static_cast<jint>(capacity);
}

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

extern "C" JNIEXPORT jboolean JNICALL Java_com_radiance_client_proxy_vulkan_TextureArrayBridge_nativeUpdateAlbedoLayer(
    JNIEnv *, jclass, jint spriteId, jlong pixelPtr, jint sizeBytes, jlong generation) {

    auto& ts = Renderer::textureSystem;
    if (spriteId < 0 || pixelPtr == 0 || sizeBytes <= 0) return JNI_FALSE;
    return ts.stageLayerUpdate(TextureSystem::LayerKind::Albedo,
        static_cast<uint32_t>(spriteId),
        reinterpret_cast<const uint8_t*>(pixelPtr),
        static_cast<size_t>(sizeBytes),
        static_cast<uint64_t>(generation)) ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jboolean JNICALL Java_com_radiance_client_proxy_vulkan_TextureArrayBridge_nativeUpdateSpecularLayer(
    JNIEnv *, jclass, jint spriteId, jlong pixelPtr, jint sizeBytes, jlong generation) {

    auto& ts = Renderer::textureSystem;
    if (spriteId < 0 || pixelPtr == 0 || sizeBytes <= 0) return JNI_FALSE;
    return ts.stageLayerUpdate(TextureSystem::LayerKind::Specular,
        static_cast<uint32_t>(spriteId),
        reinterpret_cast<const uint8_t*>(pixelPtr),
        static_cast<size_t>(sizeBytes),
        static_cast<uint64_t>(generation)) ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jboolean JNICALL Java_com_radiance_client_proxy_vulkan_TextureArrayBridge_nativeUpdateNormalLayer(
    JNIEnv *, jclass, jint spriteId, jlong pixelPtr, jint sizeBytes, jlong generation) {

    auto& ts = Renderer::textureSystem;
    if (spriteId < 0 || pixelPtr == 0 || sizeBytes <= 0) return JNI_FALSE;
    return ts.stageLayerUpdate(TextureSystem::LayerKind::Normal,
        static_cast<uint32_t>(spriteId),
        reinterpret_cast<const uint8_t*>(pixelPtr),
        static_cast<size_t>(sizeBytes),
        static_cast<uint64_t>(generation)) ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jboolean JNICALL Java_com_radiance_client_proxy_vulkan_TextureArrayBridge_nativeUpdateFlagLayer(
    JNIEnv *, jclass, jint spriteId, jlong pixelPtr, jint sizeBytes, jlong generation) {

    auto& ts = Renderer::textureSystem;
    if (spriteId < 0 || pixelPtr == 0 || sizeBytes <= 0) return JNI_FALSE;
    return ts.stageLayerUpdate(TextureSystem::LayerKind::Flag,
        static_cast<uint32_t>(spriteId),
        reinterpret_cast<const uint8_t*>(pixelPtr),
        static_cast<size_t>(sizeBytes),
        static_cast<uint64_t>(generation)) ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jboolean JNICALL Java_com_radiance_client_proxy_vulkan_TextureArrayBridge_nativeUpdateSpriteHeightMetadata(
    JNIEnv *, jclass, jint spriteId, jint flags, jint heightRangePacked, jlong generation) {

    auto& ts = Renderer::textureSystem;
    if (spriteId < 0) return JNI_FALSE;

    auto renderer = Renderer::try_instance();
    if (!renderer || !renderer->framework()) return JNI_FALSE;
    auto framework = renderer->framework();
    return ts.updateSpriteHeightMetadata(
        static_cast<uint32_t>(spriteId),
        static_cast<uint32_t>(flags),
        static_cast<int32_t>(heightRangePacked),
        static_cast<uint64_t>(generation),
        framework->vma(),
        framework->device()) ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jboolean JNICALL Java_com_radiance_client_proxy_vulkan_TextureArrayBridge_nativeReceiveTextureRules(
    JNIEnv *, jclass, jlong dataPtr, jint count, jlong generation) {

    auto& ts = Renderer::textureSystem;
    if (dataPtr == 0 || count <= 0) return JNI_FALSE;

    auto renderer = Renderer::try_instance();
    if (!renderer || !renderer->framework()) return JNI_FALSE;
    auto framework = renderer->framework();
    return ts.uploadTextureRules(
        reinterpret_cast<const vk::Data::TextureRuleEntry*>(dataPtr),
        static_cast<uint32_t>(count),
        static_cast<uint64_t>(generation),
        framework->vma(),
        framework->device()) ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jboolean JNICALL Java_com_radiance_client_proxy_vulkan_TextureArrayBridge_nativeReceiveMaterialTable(
    JNIEnv *, jclass, jlong dataPtr, jint count, jlong generation) {

    auto& ts = Renderer::textureSystem;
    if (dataPtr == 0 || count <= 0) return JNI_FALSE;

    auto renderer = Renderer::try_instance();
    if (!renderer || !renderer->framework()) return JNI_FALSE;
    auto framework = renderer->framework();
    return ts.uploadMaterialTable(
        reinterpret_cast<const vk::Data::MaterialEntry*>(dataPtr),
        static_cast<uint32_t>(count),
        static_cast<uint64_t>(generation),
        framework->vma(),
        framework->device()) ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jboolean JNICALL Java_com_radiance_client_proxy_vulkan_TextureArrayBridge_nativeReceiveMaterialTexturePage(
    JNIEnv *, jclass, jint page, jint layerSize, jint layerCount,
    jlong albedoPtr, jlong specularPtr, jlong normalPtr, jlong flagPtr,
    jlong generation) {

    auto& ts = Renderer::textureSystem;
    if (page <= 0 || layerSize <= 0 || layerCount <= 0) return JNI_FALSE;
    if (albedoPtr == 0 || specularPtr == 0 || normalPtr == 0 || flagPtr == 0) return JNI_FALSE;

    auto renderer = Renderer::try_instance();
    if (!renderer || !renderer->framework()) return JNI_FALSE;
    auto framework = renderer->framework();
    return ts.uploadMaterialTexturePage(
        static_cast<uint32_t>(page),
        static_cast<uint32_t>(layerSize),
        static_cast<uint32_t>(layerCount),
        reinterpret_cast<const uint8_t*>(albedoPtr),
        reinterpret_cast<const uint8_t*>(specularPtr),
        reinterpret_cast<const uint8_t*>(normalPtr),
        reinterpret_cast<const uint8_t*>(flagPtr),
        static_cast<uint64_t>(generation),
        framework->vma(),
        framework->device()) ? JNI_TRUE : JNI_FALSE;
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
    if (!Renderer::textureSystem.isFinalized()) return;
    Renderer::textureSystem.setGeneration(static_cast<uint64_t>(generation));
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_TextureArrayBridge_nativeTickAnimation(
    JNIEnv *, jclass, jint gameTick, jlong generation) {
    Renderer::options.textureArrayAnimationUpdatesEnabled = true;
    auto currentGen = Renderer::textureSystem.generation();
    if (generation != 0 && generation != currentGen) {
        return;
    }
    Renderer::textureSystem.tickAnimation(static_cast<uint32_t>(gameTick), static_cast<uint64_t>(generation));
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_TextureArrayBridge_nativeSetTextureArrayAnimationEnabled(
    JNIEnv *, jclass, jboolean enabled) {
    Renderer::options.textureArrayAnimationUpdatesEnabled = (enabled == JNI_TRUE);
}
