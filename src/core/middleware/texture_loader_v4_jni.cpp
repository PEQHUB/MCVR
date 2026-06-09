#include <jni.h>
#include "core/render/texture_loader_v4.hpp"
#include "core/render/renderer.hpp"
#include "core/render/render_framework.hpp"
#include "core/build/build_info.hpp"
#include <algorithm>
#include <iostream>
#include <sstream>
#include <cstring>

namespace {

jstring makeString(JNIEnv* env, const std::string& value) {
    return env->NewStringUTF(value.c_str());
}

static constexpr uint32_t CHANNEL_ALBEDO   = 1u << 0;
static constexpr uint32_t CHANNEL_SPECULAR = 1u << 1;
static constexpr uint32_t CHANNEL_NORMAL   = 1u << 2;
static constexpr uint32_t CHANNEL_FLAG     = 1u << 3;

uint32_t tierSizePixels(uint32_t tier) {
    static const uint32_t SIZES[] = {16, 32, 64, 128, 256, 512, 1024};
    if (tier >= 7) return 0;
    return SIZES[tier];
}

bool validateLayerUpload(uint64_t generation, uint32_t namespaceId, uint32_t tier,
                         uint32_t startLayer, uint32_t layerCount,
                         uint32_t layerCapacity, uint32_t width, uint32_t height, uint32_t vkFormat,
                         uint32_t channelMask, jlong albedoPtr, jlong specularPtr,
                         jlong normalPtr, jlong flagPtr, jlong bytesPerLayer) {
    if (generation == 0) return false;
    if (namespaceId > 3) return false;
    if (tier >= 7) return false;
    if (layerCount == 0) return false;
    if (layerCapacity == 0) return false;
    if (startLayer > UINT32_MAX - layerCount) return false;
    if (startLayer >= layerCapacity || layerCount > layerCapacity - startLayer) return false;
    uint32_t pixelSize = tierSizePixels(tier);
    if (pixelSize == 0) return false;
    if (width != pixelSize || height != pixelSize) return false;
    if (vkFormat != VK_FORMAT_R8G8B8A8_UNORM) return false;
    jlong expectedBytes = static_cast<jlong>(pixelSize) * pixelSize * 4;
    if (bytesPerLayer < expectedBytes) return false;
    if (layerCount > UINT64_MAX / static_cast<uint64_t>(bytesPerLayer)) return false;
    if ((channelMask & CHANNEL_ALBEDO) && albedoPtr == 0) return false;
    if ((channelMask & CHANNEL_SPECULAR) && specularPtr == 0) return false;
    if ((channelMask & CHANNEL_NORMAL) && normalPtr == 0) return false;
    if ((channelMask & CHANNEL_FLAG) && flagPtr == 0) return false;
    return true;
}

} // anonymous namespace

// ---- V4 lifecycle ----

extern "C" JNIEXPORT jboolean JNICALL
Java_com_radiance_client_proxy_vulkan_TextureArrayBridgeV4_nativeBeginTextureLoaderV4(
    JNIEnv*, jclass, jlong generation, jlong manifestPtr, jint manifestBytes) {
    if (generation <= 0) return JNI_FALSE;
    // manifestPtr/manifestBytes can be 0 for initial begin
    auto* renderer = Renderer::try_instance();
    if (!renderer || !renderer->framework()) return JNI_FALSE;
    auto& loader = renderer->textureLoaderV4();
    if (!loader.isInitialized()) {
        auto framework = renderer->framework();
        if (!loader.initialize(framework->device(), framework->vma())) {
            return JNI_FALSE;
        }
    }
    return loader.beginGeneration(static_cast<uint64_t>(generation)) ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_radiance_client_proxy_vulkan_TextureArrayBridgeV4_nativeUploadTexturePageV4(
    JNIEnv*, jclass, jlong generation, jint namespaceId, jint tier, jint page,
    jint startLayer, jint layerCount, jint layerCapacity, jint width, jint height, jint vkFormat,
    jlong albedoPtr, jlong specularPtr, jlong normalPtr, jlong flagPtr,
    jlong bytesPerLayer, jint channelMask, jboolean visible) {
    if (namespaceId < 0 || tier < 0 || page < -1 || startLayer < -1 || layerCount <= 0
        || layerCapacity <= 0 || width <= 0 || height <= 0 || bytesPerLayer <= 0) {
        return JNI_FALSE;
    }
    if (!validateLayerUpload(static_cast<uint64_t>(generation), static_cast<uint32_t>(namespaceId),
            static_cast<uint32_t>(tier), static_cast<uint32_t>(std::max(0, startLayer)),
            static_cast<uint32_t>(layerCount), static_cast<uint32_t>(layerCapacity),
            static_cast<uint32_t>(width),
            static_cast<uint32_t>(height), static_cast<uint32_t>(vkFormat),
            static_cast<uint32_t>(channelMask),
            albedoPtr, specularPtr, normalPtr, flagPtr, bytesPerLayer)) {
        return JNI_FALSE;
    }
    auto* renderer = Renderer::try_instance();
    if (!renderer || !renderer->framework()) return JNI_FALSE;

    TextureLoaderV4::UploadRequest req{};
    req.generation = static_cast<uint64_t>(generation);
    req.namespaceId = static_cast<uint32_t>(namespaceId);
    req.tier = static_cast<uint32_t>(tier);
    req.page = static_cast<uint32_t>(std::max(0, page));
    req.startLayer = static_cast<uint32_t>(std::max(0, startLayer));
    req.layerCount = static_cast<uint32_t>(layerCount);
    req.layerCapacity = static_cast<uint32_t>(layerCapacity);
    req.width = static_cast<uint32_t>(width);
    req.height = static_cast<uint32_t>(height);
    req.format = static_cast<VkFormat>(vkFormat);
    req.albedoData = albedoPtr ? reinterpret_cast<const uint8_t*>(albedoPtr) : nullptr;
    req.specularData = specularPtr ? reinterpret_cast<const uint8_t*>(specularPtr) : nullptr;
    req.normalData = normalPtr ? reinterpret_cast<const uint8_t*>(normalPtr) : nullptr;
    req.flagData = flagPtr ? reinterpret_cast<const uint8_t*>(flagPtr) : nullptr;
    req.bytesPerLayer = static_cast<uint64_t>(bytesPerLayer);
    req.channelMask = static_cast<uint32_t>(channelMask);
    req.visible = visible == JNI_TRUE;

    bool queued = renderer->textureLoaderV4().enqueueUpload(req);
    if (!queued) {
        return JNI_FALSE;
    }

    auto framework = renderer->framework();
    if (page > 0
        && ((static_cast<uint32_t>(channelMask)
            & (CHANNEL_ALBEDO | CHANNEL_SPECULAR | CHANNEL_NORMAL | CHANNEL_FLAG))
            == (CHANNEL_ALBEDO | CHANNEL_SPECULAR | CHANNEL_NORMAL | CHANNEL_FLAG))) {
        bool shaderVisible = Renderer::textureSystem.uploadMaterialTextureLayers(
            static_cast<uint32_t>(page),
            static_cast<uint32_t>(width),
            static_cast<uint32_t>(std::max(0, startLayer)),
            static_cast<uint32_t>(layerCount),
            static_cast<uint32_t>(layerCapacity),
            reinterpret_cast<const uint8_t*>(albedoPtr),
            reinterpret_cast<const uint8_t*>(specularPtr),
            reinterpret_cast<const uint8_t*>(normalPtr),
            reinterpret_cast<const uint8_t*>(flagPtr),
            static_cast<uint64_t>(generation),
            framework->vma(),
            framework->device());
        if (!shaderVisible) {
            renderer->textureLoaderV4().cancelGeneration(static_cast<uint64_t>(generation), 4);
            return JNI_FALSE;
        }
    }

    return JNI_TRUE;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_radiance_client_proxy_vulkan_TextureArrayBridgeV4_nativeCommitTextureLoaderV4(
    JNIEnv*, jclass, jlong generation) {
    if (generation <= 0) return JNI_FALSE;
    auto* renderer = Renderer::try_instance();
    if (!renderer || !renderer->framework()) return JNI_FALSE;
    return renderer->textureLoaderV4().commitGeneration(static_cast<uint64_t>(generation))
        ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_radiance_client_proxy_vulkan_TextureArrayBridgeV4_nativeCancelTextureLoaderV4(
    JNIEnv*, jclass, jlong generation, jint reasonCode) {
    if (generation <= 0) return JNI_FALSE;
    auto* renderer = Renderer::try_instance();
    if (!renderer || !renderer->framework()) return JNI_FALSE;
    return renderer->textureLoaderV4().cancelGeneration(
        static_cast<uint64_t>(generation), static_cast<int>(reasonCode))
        ? JNI_TRUE : JNI_FALSE;
}

// ---- Sparse registry updates ----

extern "C" JNIEXPORT jboolean JNICALL
Java_com_radiance_client_proxy_vulkan_TextureArrayBridgeV4_nativeUpdateMaterialTableSparseV4(
    JNIEnv*, jclass, jlong generation, jlong entriesPtr, jint entryCount) {
    if (generation <= 0 || entriesPtr == 0 || entryCount <= 0) return JNI_FALSE;
    auto* renderer = Renderer::try_instance();
    if (!renderer || !renderer->framework()) return JNI_FALSE;
    auto framework = renderer->framework();
    return Renderer::textureSystem.updateMaterialTableSparse(
        reinterpret_cast<const vk::Data::MaterialEntry*>(entriesPtr),
        static_cast<uint32_t>(entryCount),
        static_cast<uint64_t>(generation),
        framework->vma(), framework->device()) ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_radiance_client_proxy_vulkan_TextureArrayBridgeV4_nativeUpdateSpriteRegistrySparseV4(
    JNIEnv*, jclass, jlong generation, jlong entriesPtr, jint entryCount) {
    if (generation <= 0 || entriesPtr == 0 || entryCount <= 0) return JNI_FALSE;
    // Sprite registry sparse updates will be implemented in the full v4 path
    return JNI_FALSE;
}

// ---- Status JSON ----

extern "C" JNIEXPORT jstring JNICALL
Java_com_radiance_client_proxy_vulkan_TextureArrayBridgeV4_nativeTextureLoaderV4StatusJson(
    JNIEnv* env, jclass) {
    auto* renderer = Renderer::try_instance();
    if (!renderer) return makeString(env, "{\"error\":\"no_renderer\"}");
    return makeString(env, renderer->textureLoaderV4().statusJson());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_radiance_client_proxy_vulkan_TextureArrayBridgeV4_nativeTextureTierStatusJsonV4(
    JNIEnv* env, jclass) {
    auto* renderer = Renderer::try_instance();
    if (!renderer) return makeString(env, "{\"error\":\"no_renderer\"}");
    return makeString(env, renderer->textureLoaderV4().tierStatusJson());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_radiance_client_proxy_vulkan_TextureArrayBridgeV4_nativeGpuUploadQueueStatusJsonV4(
    JNIEnv* env, jclass) {
    auto* renderer = Renderer::try_instance();
    if (!renderer) return makeString(env, "{\"error\":\"no_renderer\"}");
    return makeString(env, renderer->textureLoaderV4().uploadQueueStatusJson());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_radiance_client_proxy_vulkan_TextureArrayBridgeV4_nativeMaterialPagePoolStatusJsonV4(
    JNIEnv* env, jclass) {
    auto* renderer = Renderer::try_instance();
    if (!renderer) return makeString(env, "{\"error\":\"no_renderer\"}");
    return makeString(env, renderer->textureLoaderV4().pagePoolStatusJson());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_radiance_client_proxy_vulkan_TextureArrayBridgeV4_nativeFirstFrameNativeReadinessJsonV4(
    JNIEnv* env, jclass, jlong generation) {
    auto* renderer = Renderer::try_instance();
    if (!renderer) return makeString(env, "{\"error\":\"no_renderer\"}");
    return makeString(env, renderer->textureLoaderV4().firstFrameReadinessJson(
        static_cast<uint64_t>(generation)));
}
