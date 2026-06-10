#include <jni.h>
#include "core/render/texture_loader_v4.hpp"
#include "core/render/renderer.hpp"
#include "core/render/render_framework.hpp"
#include "core/build/build_info.hpp"
#include <algorithm>
#include <iostream>
#include <sstream>
#include <cstring>
#include <climits>

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

/// Validate signed JNI inputs before any unsigned cast.
/// Accepts four-plane channel masks; requires a non-null pointer for every set bit.
bool validateLayerUploadSigned(jlong generation, jint namespaceId, jint tier,
    jint page, jint startLayer, jint layerCount, jint layerCapacity,
    jint width, jint height, jint vkFormat, jint channelMask,
    jlong albedoPtr, jlong specularPtr, jlong normalPtr, jlong flagPtr,
    jlong bytesPerLayer) {
    if (generation <= 0) return false;
    if (namespaceId < 0 || namespaceId > 3) return false;
    if (tier < 0 || tier >= 7) return false;
    if (page < -1) return false;
    if (startLayer < -1) return false;
    if (layerCount <= 0) return false;
    if (layerCapacity <= 0) return false;
    if (width <= 0 || height <= 0) return false;
    const uint32_t tierSize = tierSizePixels(static_cast<uint32_t>(tier));
    if (tierSize == 0) return false;
    if (width != static_cast<jint>(tierSize) || height != static_cast<jint>(tierSize)) return false;
    if (vkFormat != static_cast<jint>(VK_FORMAT_R8G8B8A8_UNORM)) return false;
    const uint64_t expectedBytes = uint64_t(static_cast<uint32_t>(width)) * uint64_t(static_cast<uint32_t>(height)) * 4ull;
    if (expectedBytes == 0 || expectedBytes > static_cast<uint64_t>(INT64_MAX)) return false;
    if (bytesPerLayer <= 0) return false;
    if (static_cast<uint64_t>(bytesPerLayer) != expectedBytes) return false;
    // Four-plane validation: require a non-null pointer for every set bit
    const uint32_t mask = static_cast<uint32_t>(channelMask);
    if ((mask & CHANNEL_ALBEDO) == 0) return false;
    if (albedoPtr == 0) return false;
    if ((mask & CHANNEL_SPECULAR) && specularPtr == 0) return false;
    if ((mask & CHANNEL_NORMAL) && normalPtr == 0) return false;
    if ((mask & CHANNEL_FLAG) && flagPtr == 0) return false;
    // Reject unknown bits
    static constexpr uint32_t ALL_CHANNELS = CHANNEL_ALBEDO | CHANNEL_SPECULAR | CHANNEL_NORMAL | CHANNEL_FLAG;
    if ((mask & ~ALL_CHANNELS) != 0) return false;
    // Reject nonzero aux pointers when their bit is not set
    if ((mask & CHANNEL_SPECULAR) == 0 && specularPtr != 0) return false;
    if ((mask & CHANNEL_NORMAL) == 0 && normalPtr != 0) return false;
    if ((mask & CHANNEL_FLAG) == 0 && flagPtr != 0) return false;
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
    // Validate all signed JNI inputs before any unsigned cast.
    // Negative values would become huge native values if cast without validation.
    if (!validateLayerUploadSigned(generation, namespaceId, tier, page, startLayer,
        layerCount, layerCapacity, width, height, vkFormat, channelMask,
        albedoPtr, specularPtr, normalPtr, flagPtr, bytesPerLayer)) {
        return JNI_FALSE;
    }

    auto* renderer = Renderer::try_instance();
    if (!renderer || !renderer->framework()) return JNI_FALSE;

    // Build request only after validation — safe to cast now
    TextureLoaderV4::UploadRequest req{};
    req.generation = static_cast<uint64_t>(generation);
    req.namespaceId = static_cast<uint32_t>(namespaceId);
    req.tier = static_cast<uint32_t>(tier);
    req.page = page >= 0 ? static_cast<uint32_t>(page) : UINT32_MAX;
    req.startLayer = startLayer >= 0 ? static_cast<uint32_t>(startLayer) : UINT32_MAX;
    req.layerCount = static_cast<uint32_t>(layerCount);
    req.layerCapacity = static_cast<uint32_t>(layerCapacity);
    req.width = static_cast<uint32_t>(width);
    req.height = static_cast<uint32_t>(height);
    req.format = static_cast<VkFormat>(vkFormat);
    req.albedoData = reinterpret_cast<const uint8_t*>(albedoPtr);
    req.specularData = (channelMask & CHANNEL_SPECULAR) ? reinterpret_cast<const uint8_t*>(specularPtr) : nullptr;
    req.normalData   = (channelMask & CHANNEL_NORMAL)   ? reinterpret_cast<const uint8_t*>(normalPtr)   : nullptr;
    req.flagData     = (channelMask & CHANNEL_FLAG)     ? reinterpret_cast<const uint8_t*>(flagPtr)     : nullptr;
    req.bytesPerLayer = static_cast<uint64_t>(bytesPerLayer);
    req.channelMask = static_cast<uint32_t>(channelMask);
    req.visible = visible == JNI_TRUE;

    bool queued = renderer->textureLoaderV4().enqueueUpload(req);
    if (!queued) {
        return JNI_FALSE;
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

extern "C" JNIEXPORT jint JNICALL
Java_com_radiance_client_proxy_vulkan_TextureArrayBridgeV4_nativePageLayerCapacityForTier(
    JNIEnv*, jclass, jint tier) {
    if (tier < 0 || tier >= static_cast<jint>(TexturePagePool::kMaxTiers)) return 0;
    // Mirrors TexturePagePool::pageLayerCapacity() — kept as a JNI query so
    // Java can chunk uploads to stay within per-page layer limits.
    static constexpr uint32_t CAPACITIES[] = {2048, 1024, 512, 256, 128, 64, 32};
    return static_cast<jint>(CAPACITIES[static_cast<uint32_t>(tier)]);
}
