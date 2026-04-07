#ifdef MCVR_ENABLE_ENGINE_V2

#include <jni.h>
#include <atomic>
#include <cstring>
#include <mutex>

#include "engine/app/engine_app.hpp"
#include "engine/app/engine_session.hpp"
#include "engine/app/engine_services.hpp"
#include "engine/bridge/bridge_service.hpp"
#include "engine/platform/vulkan/vk2_device.hpp"

#if defined(_WIN32)
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#endif

namespace {
std::recursive_mutex g_v2JniMtx;
engine::EngineApp g_engineApp;
std::atomic<bool> g_v2Initialized{false};
} // namespace

// --- Lifecycle ---

extern "C" JNIEXPORT jboolean JNICALL Java_com_radiance_v2_bridge_EngineBridge_nativeInitV2(
    JNIEnv* env, jclass, jstring configDir, jlong nativeWindowHandle, jboolean enableValidation) {
    std::lock_guard<std::recursive_mutex> guard(g_v2JniMtx);

    if (g_v2Initialized.load(std::memory_order_acquire)) return JNI_TRUE;

    const char* configDirStr = env->GetStringUTFChars(configDir, nullptr);
    if (!configDirStr) return JNI_FALSE;
    std::string configDirCpp(configDirStr);
    env->ReleaseStringUTFChars(configDir, configDirStr);

    engine::EngineInitConfig config;
    config.configDir = configDirCpp;
    config.mode = engine::EngineMode::V2;
    config.enableValidation = static_cast<bool>(enableValidation);
    config.nativeWindowHandle = reinterpret_cast<void*>(static_cast<intptr_t>(nativeWindowHandle));

    bool ok = false;
    try {
        ok = g_engineApp.init(config);
    } catch (const std::exception& e) {
        fprintf(stderr, "[V2-JNI] init exception: %s\n", e.what());
        return JNI_FALSE;
    } catch (...) {
        fprintf(stderr, "[V2-JNI] init unknown exception\n");
        return JNI_FALSE;
    }
    if (ok) {
        g_v2Initialized.store(true, std::memory_order_release);
    }
    return ok ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jboolean JNICALL Java_com_radiance_v2_bridge_EngineBridge_nativeTick0(
    JNIEnv*, jclass) {
    std::lock_guard<std::recursive_mutex> guard(g_v2JniMtx);
    if (!g_v2Initialized.load(std::memory_order_acquire)) return JNI_FALSE;
    try {
        return g_engineApp.tick() ? JNI_TRUE : JNI_FALSE;
    } catch (const std::exception& e) {
        fprintf(stderr, "[V2-JNI] tick exception: %s\n", e.what());
        return JNI_FALSE;
    } catch (...) {
        fprintf(stderr, "[V2-JNI] tick unknown exception\n");
        return JNI_FALSE;
    }
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_v2_bridge_EngineBridge_nativeShutdown0(
    JNIEnv*, jclass) {
    std::lock_guard<std::recursive_mutex> guard(g_v2JniMtx);
    if (!g_v2Initialized.load(std::memory_order_acquire)) return;
    g_engineApp.shutdown();
    g_v2Initialized.store(false, std::memory_order_release);
}

extern "C" JNIEXPORT jboolean JNICALL Java_com_radiance_v2_bridge_EngineBridge_nativeIsInitialized0(
    JNIEnv*, jclass) {
    return g_v2Initialized.load(std::memory_order_acquire) ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jboolean JNICALL Java_com_radiance_v2_bridge_EngineBridge_nativeIsInWorld0(
    JNIEnv*, jclass) {
    if (!g_v2Initialized.load(std::memory_order_acquire)) return JNI_FALSE;
    return g_engineApp.isInWorld() ? JNI_TRUE : JNI_FALSE;
}

// --- Bridge commands ---

extern "C" JNIEXPORT void JNICALL Java_com_radiance_v2_bridge_EngineBridge_nativePostResize0(
    JNIEnv*, jclass, jint width, jint height) {
    if (!g_v2Initialized.load(std::memory_order_acquire)) return;
    g_engineApp.services().bridge().post(
        engine::CmdWindowResize{static_cast<uint32_t>(width), static_cast<uint32_t>(height)});
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_v2_bridge_EngineBridge_nativePostShutdown0(
    JNIEnv*, jclass) {
    if (!g_v2Initialized.load(std::memory_order_acquire)) return;
    g_engineApp.services().bridge().post(engine::CmdShutdown{});
}

extern "C" JNIEXPORT jstring JNICALL Java_com_radiance_v2_bridge_EngineBridge_nativeGetDeviceName0(
    JNIEnv* env, jclass) {
    if (!g_v2Initialized.load(std::memory_order_acquire)) return env->NewStringUTF("N/A");
    const auto& name = g_engineApp.services().device().caps().deviceName;
    return env->NewStringUTF(name.c_str());
}

// --- Chunk submission ---

extern "C" JNIEXPORT void JNICALL Java_com_radiance_v2_bridge_EngineBridge_nativeSubmitChunk0(
    JNIEnv* env, jclass,
    jint chunkX, jint sectionY, jint chunkZ,
    jint originX, jint originY, jint originZ,
    jlong vertexPtr, jint vertexSize,
    jlong indexPtr, jint indexCount,
    jint triangleCount) {
    if (!g_v2Initialized.load(std::memory_order_acquire)) return;

    engine::CmdChunkSubmit cmd;
    cmd.chunkX   = chunkX;
    cmd.sectionY = sectionY;
    cmd.chunkZ   = chunkZ;
    cmd.originX  = originX;
    cmd.originY  = originY;
    cmd.originZ  = originZ;
    cmd.triangleCount = static_cast<uint32_t>(triangleCount);

    // Copy vertex data
    if (vertexPtr && vertexSize > 0) {
        const auto* src = reinterpret_cast<const uint8_t*>(vertexPtr);
        cmd.vertexData.assign(src, src + vertexSize);
    }

    // Copy index data
    if (indexPtr && indexCount > 0) {
        const auto* src = reinterpret_cast<const uint32_t*>(indexPtr);
        cmd.indexData.assign(src, src + indexCount);
    }

    g_engineApp.services().bridge().post(std::move(cmd));
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_v2_bridge_EngineBridge_nativeRemoveChunk0(
    JNIEnv*, jclass, jint chunkX, jint sectionY, jint chunkZ) {
    if (!g_v2Initialized.load(std::memory_order_acquire)) return;
    g_engineApp.services().bridge().post(engine::CmdChunkRemove{chunkX, sectionY, chunkZ});
}

// --- Camera update ---

extern "C" JNIEXPORT void JNICALL Java_com_radiance_v2_bridge_EngineBridge_nativeUpdateCamera0(
    JNIEnv* env, jclass,
    jfloatArray viewMatrix, jfloatArray projMatrix,
    jfloat posX, jfloat posY, jfloat posZ,
    jfloat dirX, jfloat dirY, jfloat dirZ,
    jfloat nearPlane, jfloat farPlane) {
    if (!g_v2Initialized.load(std::memory_order_acquire)) return;

    engine::CmdCameraUpdate cmd;
    cmd.posX = posX; cmd.posY = posY; cmd.posZ = posZ;
    cmd.dirX = dirX; cmd.dirY = dirY; cmd.dirZ = dirZ;
    cmd.nearPlane = nearPlane; cmd.farPlane = farPlane;

    if (viewMatrix) {
        jfloat* v = env->GetFloatArrayElements(viewMatrix, nullptr);
        if (v) { std::memcpy(cmd.view, v, 16 * sizeof(float)); env->ReleaseFloatArrayElements(viewMatrix, v, 0); }
    }
    if (projMatrix) {
        jfloat* p = env->GetFloatArrayElements(projMatrix, nullptr);
        if (p) { std::memcpy(cmd.projection, p, 16 * sizeof(float)); env->ReleaseFloatArrayElements(projMatrix, p, 0); }
    }

    g_engineApp.services().bridge().post(std::move(cmd));
}

// --- Sky update ---

extern "C" JNIEXPORT void JNICALL Java_com_radiance_v2_bridge_EngineBridge_nativeUpdateSky0(
    JNIEnv*, jclass,
    jfloat baseColorR, jfloat baseColorG, jfloat baseColorB,
    jfloat horizonColorR, jfloat horizonColorG, jfloat horizonColorB, jfloat horizonColorA,
    jfloat sunDirX, jfloat sunDirY, jfloat sunDirZ,
    jfloat moonDirX, jfloat moonDirY, jfloat moonDirZ,
    jint skyType, jboolean sunRisingOrSetting, jboolean skyDark,
    jboolean hasBlindnessOrDarkness, jint submersionType, jint moonPhase,
    jfloat rainGradient, jfloat thunderGradient,
    jint sunTextureID, jint moonTextureID) {
    if (!g_v2Initialized.load(std::memory_order_acquire)) return;

    engine::CmdSkyUpdate cmd;
    cmd.baseColor[0] = baseColorR;
    cmd.baseColor[1] = baseColorG;
    cmd.baseColor[2] = baseColorB;
    cmd.horizonColor[0] = horizonColorR;
    cmd.horizonColor[1] = horizonColorG;
    cmd.horizonColor[2] = horizonColorB;
    cmd.horizonColor[3] = horizonColorA;
    cmd.sunDirection[0] = sunDirX;
    cmd.sunDirection[1] = sunDirY;
    cmd.sunDirection[2] = sunDirZ;
    cmd.moonDirection[0] = moonDirX;
    cmd.moonDirection[1] = moonDirY;
    cmd.moonDirection[2] = moonDirZ;
    cmd.skyType = static_cast<uint32_t>(skyType);
    cmd.sunRisingOrSetting = sunRisingOrSetting ? 1u : 0u;
    cmd.skyDark = skyDark ? 1u : 0u;
    cmd.hasBlindnessOrDarkness = hasBlindnessOrDarkness ? 1u : 0u;
    cmd.cameraSubmersionType = static_cast<uint32_t>(submersionType);
    cmd.moonPhase = static_cast<uint32_t>(moonPhase);
    cmd.rainGradient = rainGradient;
    cmd.thunderGradient = thunderGradient;
    cmd.sunTextureID = static_cast<uint32_t>(sunTextureID);
    cmd.moonTextureID = static_cast<uint32_t>(moonTextureID);

    g_engineApp.services().bridge().post(std::move(cmd));
}

// --- Texture mapping upload ---

extern "C" JNIEXPORT void JNICALL Java_com_radiance_v2_bridge_EngineBridge_nativeUpdateTextureMapping0(
    JNIEnv*, jclass, jlong dataPtr, jint dataSize) {
    if (!g_v2Initialized.load(std::memory_order_acquire)) return;
    if (!dataPtr || dataSize <= 0) return;

    engine::CmdTextureMappingUpdate cmd;
    const auto* src = reinterpret_cast<const uint8_t*>(dataPtr);
    cmd.data.assign(src, src + dataSize);

    g_engineApp.services().bridge().post(std::move(cmd));
}

#endif // MCVR_ENABLE_ENGINE_V2
