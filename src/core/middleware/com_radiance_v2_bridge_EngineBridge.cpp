#ifdef MCVR_ENABLE_ENGINE_V2

#include <jni.h>
#include <atomic>
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

#endif // MCVR_ENABLE_ENGINE_V2
