#include <jni.h>
#include "core/render/texture_loader_v4.hpp"
#include "core/render/renderer.hpp"
#include "core/render/render_framework.hpp"
#include "core/build/build_info.hpp"
#include <algorithm>
#include <exception>
#include <iostream>
#include <sstream>
#include <cstring>
#include <climits>
#include <mutex>
#include <string>

namespace {

jstring makeString(JNIEnv* env, const std::string& value) {
    return env->NewStringUTF(value.c_str());
}

std::string readJString(JNIEnv* env, jstring value) {
    if (!value) return {};
    const char* chars = env->GetStringUTFChars(value, nullptr);
    if (!chars) return {};
    std::string out(chars);
    env->ReleaseStringUTFChars(value, chars);
    return out;
}

std::string jsonEscape(const std::string& value) {
    std::ostringstream out;
    for (char c : value) {
        switch (c) {
            case '\\': out << "\\\\"; break;
            case '"': out << "\\\""; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    const auto uc = static_cast<unsigned char>(c);
                    out << "\\u";
                    static const char* HEX = "0123456789abcdef";
                    out << "00" << HEX[(uc >> 4) & 0xF] << HEX[uc & 0xF];
                } else {
                    out << c;
                }
                break;
        }
    }
    return out.str();
}

struct PackIndexSnapshotState {
    bool hasSnapshot = false;
    uint64_t generation = 0;
    uint64_t submitCount = 0;
    size_t snapshotBytes = 0;
    uint32_t packCount = 0;
    uint32_t resourceCount = 0;
    uint32_t ruleFileCount = 0;
    uint32_t sidecarCount = 0;
    uint64_t javaCaptureMillis = 0;
    std::string packStackHash;
};

std::mutex g_packIndexMutex;
PackIndexSnapshotState g_packIndexState;

std::string packIndexStatusJson() {
    std::lock_guard<std::mutex> lock(g_packIndexMutex);
    std::ostringstream out;
    out << "{";
    out << "\"ok\":true";
    out << ",\"schema\":\"radser_native_pack_index_status_v1\"";
    out << ",\"indexMode\":\"snapshot_ingest_only\"";
    out << ",\"hasSnapshot\":" << (g_packIndexState.hasSnapshot ? "true" : "false");
    out << ",\"generation\":" << g_packIndexState.generation;
    out << ",\"submitCount\":" << g_packIndexState.submitCount;
    out << ",\"snapshotBytes\":" << g_packIndexState.snapshotBytes;
    out << ",\"packCount\":" << g_packIndexState.packCount;
    out << ",\"resourceCount\":" << g_packIndexState.resourceCount;
    out << ",\"ruleFileCount\":" << g_packIndexState.ruleFileCount;
    out << ",\"sidecarCount\":" << g_packIndexState.sidecarCount;
    out << ",\"javaCaptureMillis\":" << g_packIndexState.javaCaptureMillis;
    out << ",\"packStackHash\":\"" << jsonEscape(g_packIndexState.packStackHash) << "\"";
    out << "}";
    return out.str();
}

void logNativeException(const char* method, jlong generation, const std::exception& ex) {
    std::cerr << "[TextureLoaderV4JNI] " << method << " caught native exception";
    if (generation > 0) {
        std::cerr << " generation=" << generation;
    }
    std::cerr << " what=\"" << ex.what() << "\"" << std::endl;
}

void logNativeException(const char* method, jlong generation) {
    std::cerr << "[TextureLoaderV4JNI] " << method << " caught unknown native exception";
    if (generation > 0) {
        std::cerr << " generation=" << generation;
    }
    std::cerr << std::endl;
}

jstring makeNativeExceptionJson(JNIEnv* env, const char* method) {
    std::ostringstream out;
    out << "{\"error\":\"native_exception\",\"method\":\"" << method << "\"}";
    return makeString(env, out.str());
}

template <typename Fn>
jboolean guardedJniBool(const char* method, jlong generation, Fn fn) {
    try {
        return fn();
    } catch (const std::exception& ex) {
        logNativeException(method, generation, ex);
    } catch (...) {
        logNativeException(method, generation);
    }
    return JNI_FALSE;
}

template <typename Fn>
jstring guardedJniString(JNIEnv* env, const char* method, Fn fn) {
    try {
        return fn();
    } catch (const std::exception& ex) {
        logNativeException(method, 0, ex);
    } catch (...) {
        logNativeException(method, 0);
    }
    return makeNativeExceptionJson(env, method);
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

uint32_t descriptorPageForV4(uint32_t namespaceId, uint32_t tier,
    uint32_t page, uint32_t startLayer, uint32_t* nativeStartLayer,
    uint32_t* nativeCapacity) {
    if (nativeStartLayer) *nativeStartLayer = startLayer;
    if (nativeCapacity) *nativeCapacity = 0;

    if (namespaceId == 1u) {
        const uint32_t capacity = TexturePagePool::pageLayerCapacityStatic(tier);
        if (capacity == 0) return UINT32_MAX;
        const uint32_t nativePage = startLayer / capacity;
        if (nativeStartLayer) *nativeStartLayer = startLayer % capacity;
        if (nativeCapacity) *nativeCapacity = capacity;
        return 1u + tier * 8u + nativePage;
    }

    if (namespaceId == 2u) {
        const uint32_t capacity = TexturePagePool::pageLayerCapacityStatic(tier);
        if (capacity == 0) return UINT32_MAX;
        const uint32_t basePage = page >= 8u ? page - 8u : page;
        const uint32_t nativePage = basePage + startLayer / capacity;
        if (nativeStartLayer) *nativeStartLayer = startLayer % capacity;
        if (nativeCapacity) *nativeCapacity = capacity;
        return 64u + nativePage;
    }

    return page;
}

uint32_t nullPlaneMask(jlong albedoPtr, jlong specularPtr, jlong normalPtr, jlong flagPtr) {
    uint32_t mask = 0;
    if (albedoPtr == 0) mask |= CHANNEL_ALBEDO;
    if (specularPtr == 0) mask |= CHANNEL_SPECULAR;
    if (normalPtr == 0) mask |= CHANNEL_NORMAL;
    if (flagPtr == 0) mask |= CHANNEL_FLAG;
    return mask;
}

uint32_t clampJintToU32(jint value) {
    return value < 0 ? 0u : static_cast<uint32_t>(value);
}

TextureLoaderV4::UploadRequest requestForRejection(
    jlong generation, jint namespaceId, jint tier, jint page,
    jint startLayer, jint layerCount, jint layerCapacity, jint width, jint height,
    jint vkFormat, jlong albedoPtr, jlong specularPtr, jlong normalPtr, jlong flagPtr,
    jlong bytesPerLayer, jint channelMask, jboolean visible) {
    TextureLoaderV4::UploadRequest req{};
    req.generation = generation <= 0 ? 0u : static_cast<uint64_t>(generation);
    req.namespaceId = clampJintToU32(namespaceId);
    req.tier = clampJintToU32(tier);
    req.page = clampJintToU32(page);
    req.startLayer = clampJintToU32(startLayer);
    req.layerCount = clampJintToU32(layerCount);
    req.layerCapacity = clampJintToU32(layerCapacity);
    req.width = clampJintToU32(width);
    req.height = clampJintToU32(height);
    req.format = static_cast<VkFormat>(vkFormat);
    req.albedoData = reinterpret_cast<const uint8_t*>(albedoPtr);
    req.specularData = reinterpret_cast<const uint8_t*>(specularPtr);
    req.normalData = reinterpret_cast<const uint8_t*>(normalPtr);
    req.flagData = reinterpret_cast<const uint8_t*>(flagPtr);
    req.bytesPerLayer = bytesPerLayer <= 0 ? 0u : static_cast<uint64_t>(bytesPerLayer);
    req.channelMask = clampJintToU32(channelMask);
    req.visible = visible == JNI_TRUE;
    return req;
}

/// Validate signed JNI inputs before any unsigned cast.
/// Accepts four-plane channel masks; requires a non-null pointer for every set bit.
/// page and startLayer must be non-negative (no -1 sentinel; Java provides explicit values).
bool validateLayerUploadSigned(jlong generation, jint namespaceId, jint tier,
    jint page, jint startLayer, jint layerCount, jint layerCapacity,
    jint width, jint height, jint vkFormat, jint channelMask,
    jlong albedoPtr, jlong specularPtr, jlong normalPtr, jlong flagPtr,
    jlong bytesPerLayer) {
    if (generation <= 0) return false;
    if (namespaceId < 0 || namespaceId > 3) return false;
    if (tier < 0 || tier >= 7) return false;
    if (page < 0) return false;
    if (startLayer < 0) return false;
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
    return true;
}

} // anonymous namespace

// ---- V4 lifecycle ----

extern "C" JNIEXPORT jboolean JNICALL
Java_com_radiance_client_proxy_vulkan_TextureArrayBridgeV4_nativeBeginTextureLoaderV4(
    JNIEnv*, jclass, jlong generation, jlong manifestPtr, jint manifestBytes) {
    return guardedJniBool("nativeBeginTextureLoaderV4", generation, [&]() -> jboolean {
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
    if (!loader.beginGeneration(static_cast<uint64_t>(generation))) {
        return JNI_FALSE;
    }
    const uint64_t nativeGeneration = static_cast<uint64_t>(generation);
    Renderer::textureSystem.beginV4MaterialPages(nativeGeneration);
    auto framework = renderer->framework();
    if (!Renderer::textureSystem.ensureV4ShaderFallbackResources(
            nativeGeneration, framework->vma(), framework->device())) {
        std::cerr << "[TextureLoaderV4JNI] nativeBeginTextureLoaderV4 rejected: V4 fallback resources unavailable"
                  << " generation=" << generation << std::endl;
        loader.cancelGeneration(nativeGeneration, -2);
        return JNI_FALSE;
    }
    return JNI_TRUE;
    });
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_radiance_client_proxy_vulkan_TextureArrayBridgeV4_nativeEnsureV4ShaderFallbackResources(
    JNIEnv*, jclass, jlong generation) {
    return guardedJniBool("nativeEnsureV4ShaderFallbackResources", generation, [&]() -> jboolean {
    if (generation <= 0) return JNI_FALSE;
    auto* renderer = Renderer::try_instance();
    if (!renderer || !renderer->framework()) return JNI_FALSE;
    auto framework = renderer->framework();
    return Renderer::textureSystem.ensureV4ShaderFallbackResources(
        static_cast<uint64_t>(generation), framework->vma(), framework->device())
        ? JNI_TRUE : JNI_FALSE;
    });
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_radiance_client_proxy_vulkan_TextureArrayBridgeV4_nativeUploadTexturePageV4(
    JNIEnv*, jclass, jlong generation, jint namespaceId, jint tier, jint page,
    jint startLayer, jint layerCount, jint layerCapacity, jint width, jint height, jint vkFormat,
    jlong albedoPtr, jlong specularPtr, jlong normalPtr, jlong flagPtr,
    jlong bytesPerLayer, jint channelMask, jboolean visible) {
    return guardedJniBool("nativeUploadTexturePageV4", generation, [&]() -> jboolean {
    // Validate all signed JNI inputs before any unsigned cast.
    // Negative values would become huge native values if cast without validation.
    if (!validateLayerUploadSigned(generation, namespaceId, tier, page, startLayer,
        layerCount, layerCapacity, width, height, vkFormat, channelMask,
        albedoPtr, specularPtr, normalPtr, flagPtr, bytesPerLayer)) {
        if (auto* renderer = Renderer::try_instance()) {
            auto rejected = requestForRejection(generation, namespaceId, tier, page, startLayer,
                layerCount, layerCapacity, width, height, vkFormat,
                albedoPtr, specularPtr, normalPtr, flagPtr, bytesPerLayer, channelMask, visible);
            const uint32_t planeMask = nullPlaneMask(albedoPtr, specularPtr, normalPtr, flagPtr);
            renderer->textureLoaderV4().recordUploadRejection(rejected,
                planeMask != 0 ? "invalid_plane_pointer" : "invalid_arguments",
                UINT32_MAX, UINT32_MAX, UINT32_MAX, 0, UINT32_MAX, planeMask, "validation");
        }
        std::cerr << "[TextureLoaderV4JNI] nativeUploadTexturePageV4 rejected invalid arguments"
                  << " generation=" << generation
                  << " namespace=" << namespaceId
                  << " tier=" << tier
                  << " page=" << page
                  << " startLayer=" << startLayer
                  << " layers=" << layerCount
                  << " capacity=" << layerCapacity
                  << " width=" << width
                  << " height=" << height
                  << " bytesPerLayer=" << bytesPerLayer
                  << " channelMask=" << channelMask << std::endl;
        return JNI_FALSE;
    }

    auto* renderer = Renderer::try_instance();
    if (!renderer || !renderer->framework()) {
        std::cerr << "[TextureLoaderV4JNI] nativeUploadTexturePageV4 rejected: renderer unavailable"
                  << " generation=" << generation
                  << " tier=" << tier
                  << " page=" << page
                  << " startLayer=" << startLayer
                  << " layers=" << layerCount
                  << " capacity=" << layerCapacity << std::endl;
        return JNI_FALSE;
    }

    // Build request only after validation — safe to cast now
    TextureLoaderV4::UploadRequest req{};
    req.generation = static_cast<uint64_t>(generation);
    req.namespaceId = static_cast<uint32_t>(namespaceId);
    req.tier = static_cast<uint32_t>(tier);
    req.page = static_cast<uint32_t>(page);
    req.startLayer = static_cast<uint32_t>(startLayer);
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
        std::cerr << "[TextureLoaderV4JNI] nativeUploadTexturePageV4 rejected: queue failed"
                  << " generation=" << generation
                  << " tier=" << tier
                  << " page=" << page
                  << " startLayer=" << startLayer
                  << " layers=" << layerCount
                  << " capacity=" << layerCapacity << std::endl;
        return JNI_FALSE;
    }

    uint32_t nativeStartLayer = static_cast<uint32_t>(startLayer);
    uint32_t nativeCapacity = static_cast<uint32_t>(layerCapacity);
    uint32_t descriptorPage = descriptorPageForV4(
        static_cast<uint32_t>(namespaceId),
        static_cast<uint32_t>(tier),
        static_cast<uint32_t>(page),
        static_cast<uint32_t>(startLayer),
        &nativeStartLayer,
        &nativeCapacity);
    if (descriptorPage >= vk::Data::MATERIAL_TEXTURE_PAGE_MAX || nativeCapacity == 0) {
        renderer->textureLoaderV4().recordUploadRejection(req,
            "invalid_descriptor_page", UINT32_MAX,
            UINT32_MAX, nativeStartLayer, nativeCapacity,
            descriptorPage, nullPlaneMask(albedoPtr, specularPtr, normalPtr, flagPtr),
            "descriptor_map");
        std::cerr << "[TextureLoaderV4JNI] nativeUploadTexturePageV4 rejected invalid descriptor page"
                  << " generation=" << generation
                  << " namespace=" << namespaceId
                  << " tier=" << tier
                  << " page=" << page
                  << " descriptorPage=" << descriptorPage
                  << " startLayer=" << startLayer
                  << " nativeStartLayer=" << nativeStartLayer
                  << " layers=" << layerCount
                  << " capacity=" << layerCapacity
                  << " nativeCapacity=" << nativeCapacity << std::endl;
        return JNI_FALSE;
    }

    auto framework = renderer->framework();
    bool published = Renderer::textureSystem.uploadMaterialTextureLayersV4(
        descriptorPage,
        static_cast<uint32_t>(width),
        nativeStartLayer,
        static_cast<uint32_t>(layerCount),
        nativeCapacity,
        reinterpret_cast<const uint8_t*>(albedoPtr),
        reinterpret_cast<const uint8_t*>(specularPtr),
        reinterpret_cast<const uint8_t*>(normalPtr),
        reinterpret_cast<const uint8_t*>(flagPtr),
        static_cast<uint32_t>(channelMask),
        static_cast<uint64_t>(generation),
        framework->vma(),
        framework->device());
    if (!published) {
        renderer->textureLoaderV4().recordUploadRejection(req,
            "descriptor_publish_failed", UINT32_MAX,
            descriptorPage, nativeStartLayer, nativeCapacity,
            descriptorPage, nullPlaneMask(albedoPtr, specularPtr, normalPtr, flagPtr),
            "descriptor_publish");
        std::cerr << "[TextureLoaderV4JNI] nativeUploadTexturePageV4 rejected: descriptor/material page publication failed"
                  << " generation=" << generation
                  << " namespace=" << namespaceId
                  << " tier=" << tier
                  << " page=" << page
                  << " descriptorPage=" << descriptorPage
                  << " startLayer=" << startLayer
                  << " nativeStartLayer=" << nativeStartLayer
                  << " layers=" << layerCount
                  << " capacity=" << layerCapacity
                  << " nativeCapacity=" << nativeCapacity << std::endl;
        return JNI_FALSE;
    }

    return JNI_TRUE;
    });
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_radiance_client_proxy_vulkan_TextureArrayBridgeV4_nativeCommitTextureLoaderV4(
    JNIEnv*, jclass, jlong generation) {
    return guardedJniBool("nativeCommitTextureLoaderV4", generation, [&]() -> jboolean {
    if (generation <= 0) return JNI_FALSE;
    auto* renderer = Renderer::try_instance();
    if (!renderer || !renderer->framework()) return JNI_FALSE;
    auto framework = renderer->framework();
    if (!Renderer::textureSystem.ensureV4ShaderFallbackResources(
            static_cast<uint64_t>(generation), framework->vma(), framework->device())) {
        std::cerr << "[TextureLoaderV4JNI] nativeCommitTextureLoaderV4 rejected: V4 fallback contract unavailable"
                  << " generation=" << generation << std::endl;
        return JNI_FALSE;
    }
    return renderer->textureLoaderV4().commitGeneration(static_cast<uint64_t>(generation))
        ? JNI_TRUE : JNI_FALSE;
    });
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_radiance_client_proxy_vulkan_TextureArrayBridgeV4_nativeCancelTextureLoaderV4(
    JNIEnv*, jclass, jlong generation, jint reasonCode) {
    return guardedJniBool("nativeCancelTextureLoaderV4", generation, [&]() -> jboolean {
    if (generation <= 0) return JNI_FALSE;
    auto* renderer = Renderer::try_instance();
    if (!renderer || !renderer->framework()) return JNI_FALSE;
    return renderer->textureLoaderV4().cancelGeneration(
        static_cast<uint64_t>(generation), static_cast<int>(reasonCode))
        ? JNI_TRUE : JNI_FALSE;
    });
}

// ---- Sparse registry updates ----

extern "C" JNIEXPORT jboolean JNICALL
Java_com_radiance_client_proxy_vulkan_TextureArrayBridgeV4_nativeUpdateMaterialTableSparseV4(
    JNIEnv*, jclass, jlong generation, jlong entriesPtr, jint entryCount) {
    return guardedJniBool("nativeUpdateMaterialTableSparseV4", generation, [&]() -> jboolean {
    if (generation <= 0 || entriesPtr == 0 || entryCount <= 0) return JNI_FALSE;
    auto* renderer = Renderer::try_instance();
    if (!renderer || !renderer->framework()) return JNI_FALSE;
    auto framework = renderer->framework();
    return Renderer::textureSystem.updateMaterialTableSparse(
        reinterpret_cast<const vk::Data::MaterialEntry*>(entriesPtr),
        static_cast<uint32_t>(entryCount),
        static_cast<uint64_t>(generation),
        framework->vma(), framework->device()) ? JNI_TRUE : JNI_FALSE;
    });
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_radiance_client_proxy_vulkan_TextureArrayBridgeV4_nativeUpdateSpriteRegistrySparseV4(
    JNIEnv*, jclass, jlong generation, jlong entriesPtr, jint entryCount) {
    return guardedJniBool("nativeUpdateSpriteRegistrySparseV4", generation, [&]() -> jboolean {
    if (generation <= 0 || entriesPtr == 0 || entryCount <= 0) return JNI_FALSE;
    auto* renderer = Renderer::try_instance();
    if (!renderer || !renderer->framework()) return JNI_FALSE;
    auto framework = renderer->framework();
    return Renderer::textureSystem.updateSpriteRegistrySparse(
        reinterpret_cast<const vk::Data::SpriteEntry*>(entriesPtr),
        static_cast<uint32_t>(entryCount),
        static_cast<uint64_t>(generation),
        framework->vma(), framework->device()) ? JNI_TRUE : JNI_FALSE;
    });
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_radiance_client_proxy_vulkan_TextureArrayBridgeV4_nativeUpdateTextureRulesV4(
    JNIEnv*, jclass, jlong generation, jlong entriesPtr, jint entryCount) {
    return guardedJniBool("nativeUpdateTextureRulesV4", generation, [&]() -> jboolean {
    if (generation <= 0 || entryCount < 0) return JNI_FALSE;
    auto* renderer = Renderer::try_instance();
    if (!renderer || !renderer->framework()) return JNI_FALSE;
    auto framework = renderer->framework();
    const auto* entries = entriesPtr == 0
        ? nullptr
        : reinterpret_cast<const vk::Data::TextureRuleEntry*>(entriesPtr);
    const uint32_t count = entryCount <= 0 ? 0u : static_cast<uint32_t>(entryCount);
    return Renderer::textureSystem.uploadTextureRules(
        entries,
        count,
        static_cast<uint64_t>(generation),
        framework->vma(), framework->device()) ? JNI_TRUE : JNI_FALSE;
    });
}

// ---- Status JSON ----

extern "C" JNIEXPORT jboolean JNICALL
Java_com_radiance_client_proxy_vulkan_TextureArrayBridgeV4_nativeSubmitPackStackSnapshotV1(
    JNIEnv* env, jclass, jlong generation, jstring snapshotJson,
    jint packCount, jint resourceCount, jint ruleFileCount, jint sidecarCount,
    jstring packStackHash, jlong javaCaptureMillis) {
    return guardedJniBool("nativeSubmitPackStackSnapshotV1", generation, [&]() -> jboolean {
    if (generation <= 0 || !snapshotJson || packCount < 0 || resourceCount < 0
        || ruleFileCount < 0 || sidecarCount < 0 || javaCaptureMillis < 0) {
        return JNI_FALSE;
    }
    std::string snapshot = readJString(env, snapshotJson);
    if (snapshot.empty()) {
        return JNI_FALSE;
    }
    std::string hash = readJString(env, packStackHash);
    {
        std::lock_guard<std::mutex> lock(g_packIndexMutex);
        g_packIndexState.hasSnapshot = true;
        g_packIndexState.generation = static_cast<uint64_t>(generation);
        g_packIndexState.submitCount++;
        g_packIndexState.snapshotBytes = snapshot.size();
        g_packIndexState.packCount = static_cast<uint32_t>(packCount);
        g_packIndexState.resourceCount = static_cast<uint32_t>(resourceCount);
        g_packIndexState.ruleFileCount = static_cast<uint32_t>(ruleFileCount);
        g_packIndexState.sidecarCount = static_cast<uint32_t>(sidecarCount);
        g_packIndexState.javaCaptureMillis = static_cast<uint64_t>(javaCaptureMillis);
        g_packIndexState.packStackHash = std::move(hash);
    }
    return JNI_TRUE;
    });
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_radiance_client_proxy_vulkan_TextureArrayBridgeV4_nativePackIndexStatusJson(
    JNIEnv* env, jclass) {
    return guardedJniString(env, "nativePackIndexStatusJson", [&]() -> jstring {
    return makeString(env, packIndexStatusJson());
    });
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_radiance_client_proxy_vulkan_TextureArrayBridgeV4_nativeTextureLoaderV4StatusJson(
    JNIEnv* env, jclass) {
    return guardedJniString(env, "nativeTextureLoaderV4StatusJson", [&]() -> jstring {
    auto* renderer = Renderer::try_instance();
    if (!renderer) return makeString(env, "{\"error\":\"no_renderer\"}");
    return makeString(env, renderer->textureLoaderV4().statusJson());
    });
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_radiance_client_proxy_vulkan_TextureArrayBridgeV4_nativeTextureTierStatusJsonV4(
    JNIEnv* env, jclass) {
    return guardedJniString(env, "nativeTextureTierStatusJsonV4", [&]() -> jstring {
    auto* renderer = Renderer::try_instance();
    if (!renderer) return makeString(env, "{\"error\":\"no_renderer\"}");
    return makeString(env, renderer->textureLoaderV4().tierStatusJson());
    });
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_radiance_client_proxy_vulkan_TextureArrayBridgeV4_nativeGpuUploadQueueStatusJsonV4(
    JNIEnv* env, jclass) {
    return guardedJniString(env, "nativeGpuUploadQueueStatusJsonV4", [&]() -> jstring {
    auto* renderer = Renderer::try_instance();
    if (!renderer) return makeString(env, "{\"error\":\"no_renderer\"}");
    return makeString(env, renderer->textureLoaderV4().uploadQueueStatusJson());
    });
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_radiance_client_proxy_vulkan_TextureArrayBridgeV4_nativeMaterialPagePoolStatusJsonV4(
    JNIEnv* env, jclass) {
    return guardedJniString(env, "nativeMaterialPagePoolStatusJsonV4", [&]() -> jstring {
    auto* renderer = Renderer::try_instance();
    if (!renderer) return makeString(env, "{\"error\":\"no_renderer\"}");
    return makeString(env, renderer->textureLoaderV4().pagePoolStatusJson());
    });
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_radiance_client_proxy_vulkan_TextureArrayBridgeV4_nativeV4FrameResourceStatusJson(
    JNIEnv* env, jclass) {
    return guardedJniString(env, "nativeV4FrameResourceStatusJson", [&]() -> jstring {
    auto* renderer = Renderer::try_instance();
    if (!renderer) return makeString(env, "{\"error\":\"no_renderer\"}");
    return makeString(env, Renderer::textureSystem.v4FrameResourceStatusJson());
    });
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_radiance_client_proxy_vulkan_TextureArrayBridgeV4_nativeFirstFrameNativeReadinessJsonV4(
    JNIEnv* env, jclass, jlong generation) {
    return guardedJniString(env, "nativeFirstFrameNativeReadinessJsonV4", [&]() -> jstring {
    auto* renderer = Renderer::try_instance();
    if (!renderer) return makeString(env, "{\"error\":\"no_renderer\"}");
    return makeString(env, renderer->textureLoaderV4().firstFrameReadinessJson(
        static_cast<uint64_t>(generation)));
    });
}

extern "C" JNIEXPORT jint JNICALL
Java_com_radiance_client_proxy_vulkan_TextureArrayBridgeV4_nativePageLayerCapacityForTier(
    JNIEnv*, jclass, jint tier) {
    if (tier < 0 || tier >= static_cast<jint>(TexturePagePool::kMaxTiers)) return 0;
    return static_cast<jint>(TexturePagePool::pageLayerCapacityStatic(static_cast<uint32_t>(tier)));
}
