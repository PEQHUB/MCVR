#include <jni.h>
#include "core/render/renderer.hpp"
#include "core/render/render_framework.hpp"

#include <algorithm>
#include <iostream>
#include <sstream>

namespace {
jstring makeString(JNIEnv* env, const std::string& value) {
    return env->NewStringUTF(value.c_str());
}
}

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

extern "C" JNIEXPORT jint JNICALL Java_com_radiance_client_proxy_vulkan_TextureArrayBridge_nativeTextureUploadCapabilities(
    JNIEnv *, jclass) {
    constexpr jint kDefaultAuxTextures = 1 << 0;
    constexpr jint kSparseSpriteAuxUpload = 1 << 1;
    constexpr jint kGenerationBeginEndCancel = 1 << 2;
    return kDefaultAuxTextures | kSparseSpriteAuxUpload | kGenerationBeginEndCancel;
}

extern "C" JNIEXPORT jstring JNICALL Java_com_radiance_client_proxy_vulkan_TextureArrayBridge_nativeTextureStreamingStatusJson(
    JNIEnv *env, jclass) {
    auto& ts = Renderer::textureSystem;
    std::ostringstream out;
    out << "{"
        << "\"schema\":\"radser_native_texture_streaming_status_v1\","
        << "\"generation\":" << ts.generation() << ","
        << "\"finalized\":" << (ts.isFinalized() ? "true" : "false") << ","
        << "\"spriteCount\":" << ts.spriteCount() << ","
        << "\"layerSize\":" << ts.layerSize() << ","
        << "\"activeUploadMode\":\"texture_loader_v4\","
        << "\"materialTexturePageMax\":" << vk::Data::MATERIAL_TEXTURE_PAGE_MAX << ","
        << "\"nativeDefaultAuxTextures\":true,"
        << "\"sparseSpriteAuxUpload\":true,"
        << "\"sparseAuxBatchUpload\":true,"
        << "\"materialPagePools\":true,"
        << "\"sparseMaterialTableUpdates\":true,"
        << "\"blockingMaterialTableUploads\":false,"
        << "\"asyncMaterialTableMainQueueUpload\":true,"
        << "\"tieredArrays\":true,"
        << "\"vanillaMaterialPageTiers\":true,"
        << "\"asyncTransferQueueUpload\":true,"
        << "\"pendingTextureUploads\":" << (ts.hasPendingTextureUploads() ? "true" : "false") << ","
        << "\"materialTable\":" << ts.materialTableStatusJson() << ","
        << "\"textureUploadCapabilities\":" << Java_com_radiance_client_proxy_vulkan_TextureArrayBridge_nativeTextureUploadCapabilities(nullptr, nullptr)
        << "}";
    return makeString(env, out.str());
}

extern "C" JNIEXPORT jstring JNICALL Java_com_radiance_client_proxy_vulkan_TextureArrayBridge_nativeTextureTierStatusJson(
    JNIEnv *env, jclass) {
    auto& ts = Renderer::textureSystem;
    std::ostringstream out;
    out << "{"
        << "\"schema\":\"radser_native_texture_tier_status_v1\","
        << "\"tieredArrays\":true,"
        << "\"currentCompatibilityMode\":\"texture_loader_v4_namespace_tier_page_layer\","
        << "\"layerSize\":" << ts.layerSize() << ","
        << "\"spriteCount\":" << ts.spriteCount() << ","
        << "\"materialTexturePageMax\":" << vk::Data::MATERIAL_TEXTURE_PAGE_MAX << ","
        << "\"materialPageNamespaceMask\":\"0xF0000000\","
        << "\"vanillaTierPageNamespace\":\"0x80000000\","
        << "\"reservedTierPages\":\"1-7\","
        << "\"tierSizes\":\"16,32,64,128,256,512,1024\","
        << "\"compatMaterialFirstPage\":8"
        << "}";
    return makeString(env, out.str());
}

extern "C" JNIEXPORT jstring JNICALL Java_com_radiance_client_proxy_vulkan_TextureArrayBridge_nativeGpuUploadQueueStatusJson(
    JNIEnv *env, jclass) {
    auto& ts = Renderer::textureSystem;
    std::ostringstream out;
    out << "{"
        << "\"schema\":\"radser_native_gpu_upload_queue_status_v1\","
        << "\"pendingTextureUploads\":" << (ts.hasPendingTextureUploads() ? "true" : "false") << ","
        << "\"textureGeneration\":" << ts.generation() << ","
        << "\"finalized\":" << (ts.isFinalized() ? "true" : "false") << ","
        << "\"materialPageRevision\":" << ts.materialTexturePageRevision()
        << ",\"materialPagePool\":" << ts.materialPagePoolStatusJson()
        << ",\"materialTable\":" << ts.materialTableStatusJson()
        << "}";
    return makeString(env, out.str());
}

extern "C" JNIEXPORT jstring JNICALL Java_com_radiance_client_proxy_vulkan_TextureArrayBridge_nativeTextureMemoryStatusJson(
    JNIEnv *env, jclass) {
    auto& ts = Renderer::textureSystem;
    const uint64_t bytesPerLayer = static_cast<uint64_t>(ts.layerSize()) * ts.layerSize() * 4ull;
    std::ostringstream out;
    out << "{"
        << "\"schema\":\"radser_native_texture_memory_status_v1\","
        << "\"generation\":" << ts.generation() << ","
        << "\"spriteCount\":" << ts.spriteCount() << ","
        << "\"layerSize\":" << ts.layerSize() << ","
        << "\"bytesPerLayer\":" << bytesPerLayer << ","
        << "\"fixedAlbedoBytes\":0,"
        << "\"javaFullAuxArraysRequired\":false,"
        << "\"nativeDefaultAuxTextures\":true"
        << "}";
    return makeString(env, out.str());
}

extern "C" JNIEXPORT jstring JNICALL Java_com_radiance_client_proxy_vulkan_TextureArrayBridge_nativeMaterialResidencyStatusJson(
    JNIEnv *env, jclass) {
    auto& ts = Renderer::textureSystem;
    std::ostringstream out;
    out << "{"
        << "\"schema\":\"radser_native_material_residency_status_v1\","
        << "\"generation\":" << ts.generation() << ","
        << "\"finalized\":" << (ts.isFinalized() ? "true" : "false") << ","
        << "\"backend\":\"renderer_owned_material_pages\","
        << "\"materialTexturePageMax\":" << vk::Data::MATERIAL_TEXTURE_PAGE_MAX << ","
        << "\"dirtyMaterialTableUpdates\":true,"
        << "\"sparseMaterialTableUpdates\":true,"
        << "\"persistentMaterialPagePools\":true,"
        << "\"materialPageRevision\":" << ts.materialTexturePageRevision()
        << "}";
    return makeString(env, out.str());
}

extern "C" JNIEXPORT jstring JNICALL Java_com_radiance_client_proxy_vulkan_TextureArrayBridge_nativeMaterialPagePoolStatusJson(
    JNIEnv *env, jclass) {
    return makeString(env, Renderer::textureSystem.materialPagePoolStatusJson());
}

extern "C" JNIEXPORT jstring JNICALL Java_com_radiance_client_proxy_vulkan_TextureArrayBridge_nativeMaterialTableStatusJson(
    JNIEnv *env, jclass) {
    return makeString(env, Renderer::textureSystem.materialTableStatusJson());
}

extern "C" JNIEXPORT jstring JNICALL Java_com_radiance_client_proxy_vulkan_TextureArrayBridge_nativeUploadSafetyStatusJson(
    JNIEnv *env, jclass) {
    return makeString(env, Renderer::textureSystem.nativeUploadSafetyStatusJson());
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

extern "C" JNIEXPORT jboolean JNICALL Java_com_radiance_client_proxy_vulkan_TextureArrayBridge_nativeReceiveSparseAuxBatch(
    JNIEnv *, jclass, jlong updatePtr, jint updateCount,
    jlong pixelPtr, jint pixelBytes, jlong metadataPtr, jint metadataCount,
    jlong generation) {

    auto& ts = Renderer::textureSystem;
    if (updatePtr == 0 || updateCount <= 0 || pixelPtr == 0 || pixelBytes <= 0) return JNI_FALSE;

    auto renderer = Renderer::try_instance();
    if (!renderer || !renderer->framework()) return JNI_FALSE;
    auto framework = renderer->framework();
    return ts.receiveSparseAuxBatch(
        reinterpret_cast<const TextureSystem::SparseAuxUpdate*>(updatePtr),
        static_cast<uint32_t>(updateCount),
        reinterpret_cast<const uint8_t*>(pixelPtr),
        static_cast<size_t>(pixelBytes),
        reinterpret_cast<const TextureSystem::SparseAuxMetadata*>(metadataPtr),
        static_cast<uint32_t>(std::max(metadataCount, 0)),
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

    std::cerr << "[TextureLoaderV4] Rejected retired legacy material table upload" << std::endl;
    return JNI_FALSE;
}

extern "C" JNIEXPORT jboolean JNICALL Java_com_radiance_client_proxy_vulkan_TextureArrayBridge_nativeUpdateMaterialTableSparse(
    JNIEnv *, jclass, jlong dataPtr, jint count, jlong generation) {

    std::cerr << "[TextureLoaderV4] Rejected retired legacy sparse material table upload" << std::endl;
    return JNI_FALSE;
}

extern "C" JNIEXPORT jboolean JNICALL Java_com_radiance_client_proxy_vulkan_TextureArrayBridge_nativeReceiveMaterialTexturePage(
    JNIEnv *, jclass, jint page, jint layerSize, jint layerCount,
    jlong albedoPtr, jlong specularPtr, jlong normalPtr, jlong flagPtr,
    jlong generation) {

    std::cerr << "[TextureLoaderV4] Rejected retired legacy material texture page upload" << std::endl;
    return JNI_FALSE;
}

extern "C" JNIEXPORT jboolean JNICALL Java_com_radiance_client_proxy_vulkan_TextureArrayBridge_nativeReceiveMaterialTextureLayers(
    JNIEnv *, jclass, jint page, jint layerSize, jint startLayer, jint layerCount, jint layerCapacity,
    jlong albedoPtr, jlong specularPtr, jlong normalPtr, jlong flagPtr,
    jlong generation) {

    std::cerr << "[TextureLoaderV4] Rejected retired legacy material texture layer upload" << std::endl;
    return JNI_FALSE;
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
