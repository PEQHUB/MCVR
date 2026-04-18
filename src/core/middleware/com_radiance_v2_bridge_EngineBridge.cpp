#ifdef MCVR_ENABLE_ENGINE_V2

#include <jni.h>
#include <atomic>
#include <cstring>
#include <mutex>
#include <string>

#include "engine/app/engine_app.hpp"
#include "engine/app/engine_session.hpp"
#include "engine/app/engine_services.hpp"
#include "engine/bridge/bridge_service.hpp"
#include "engine/diagnostics/boot_trace.hpp"
#include "engine/diagnostics/metrics_service.hpp"
#include "engine/platform/vulkan/vk2_device.hpp"
#include "engine/platform/vulkan/vk2_probe.hpp"

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

// Phase 2 lifetime overhaul. Engine is heap-allocated, owned by a unique_ptr,
// constructed lazily on first nativeInitV2 and torn down by nativeShutdown.
// This removes the dependency on CRT static-destructor ordering at DLL detach
// — the previous file-scope `static engine::EngineApp` would call into spdlog
// during its destructor after the CRT had already destroyed the spdlog sinks,
// causing EXCEPTION_ACCESS_VIOLATION inside isCategoryEnabled.
//
// State machine:
//   Uninit       — engine pointer is null, nothing to tear down.
//   Initializing — nativeInitV2 holds the JNI mutex, building the engine.
//   Ready        — engine is built and tickable.
//   ShuttingDown — nativeShutdown is running; further calls are no-ops.
//   Dead         — engine has been destroyed; resurrection requires init again.
//
// Transitions are guarded by both g_state (cheap, lock-free for fast-path
// readers like nativeIsInitialized) and g_v2JniMtx (required for any actual
// engine method call).

enum class EngineState : uint8_t {
    Uninit       = 0,
    Initializing = 1,
    Ready        = 2,
    ShuttingDown = 3,
    Dead         = 4,
};

// Recursive (not plain std::mutex) because the engine occasionally posts
// follow-up bridge commands from inside a flush(), and ConfigPatch handlers
// can call back into JNI methods that re-acquire this mutex. A plain mutex
// would deadlock; demoting it would require restructuring the bridge to
// drain re-entrant posts asynchronously, which is out of scope for the
// stability work and would mask real reentrancy bugs as silent skips.
std::recursive_mutex g_v2JniMtx;
std::unique_ptr<engine::EngineApp> g_engineApp;
std::atomic<EngineState> g_state{EngineState::Uninit};

inline bool stateAtLeast(EngineState s) {
    auto cur = g_state.load(std::memory_order_acquire);
    return cur == EngineState::Ready || cur == s;
}

inline bool isReady() {
    return g_state.load(std::memory_order_acquire) == EngineState::Ready;
}

} // namespace

// --- Lifecycle ---

// Dry-run Vulkan probe. Returns a UTF-8 string the Java side parses:
//   "OK|<gpuName>"               on success
//   "FAIL|<reason>|<missing>"    on failure (missing = comma-joined extension list)
// Writes a v2_cannot_start.txt / v2_probe_ok.txt to {logDir}/logs.
//
// Safe to call multiple times; safe to call before nativeInitV2; never
// touches GLFW or window state.
extern "C" JNIEXPORT jstring JNICALL Java_com_radiance_v2_bridge_EngineBridge_nativeProbe0(
    JNIEnv* env, jclass, jstring configDir, jboolean enableValidation) {
    const char* configDirStr = configDir ? env->GetStringUTFChars(configDir, nullptr) : nullptr;
    std::string configDirCpp = configDirStr ? std::string(configDirStr) : std::string{};
    if (configDirStr) env->ReleaseStringUTFChars(configDir, configDirStr);

    // Initialize the boot trace early so the probe breadcrumb lands even if
    // EngineApp::init never runs.
    if (!configDirCpp.empty()) {
        engine::boot_trace::init(configDirCpp + "/logs");
        engine::boot_trace::breadcrumb("probe", "begin");
    }

    auto result = engine::vk2::probeVulkan(static_cast<bool>(enableValidation));

    if (!configDirCpp.empty()) {
        engine::vk2::writeProbeReport(configDirCpp + "/logs", result);
    }

    std::string payload;
    if (result.ok) {
        payload = "OK|" + result.gpuName;
    } else {
        std::string missing;
        for (size_t i = 0; i < result.missingExtensions.size(); ++i) {
            if (i) missing += ",";
            missing += result.missingExtensions[i];
        }
        payload = "FAIL|" + result.reason + "|" + missing;
    }
    return env->NewStringUTF(payload.c_str());
}

extern "C" JNIEXPORT jboolean JNICALL Java_com_radiance_v2_bridge_EngineBridge_nativeInitV2(
    JNIEnv* env, jclass, jstring configDir, jlong nativeWindowHandle, jboolean enableValidation) {
    std::lock_guard<std::recursive_mutex> guard(g_v2JniMtx);

    auto cur = g_state.load(std::memory_order_acquire);
    if (cur == EngineState::Ready) return JNI_TRUE;
    if (cur == EngineState::Initializing || cur == EngineState::ShuttingDown) {
        // Re-entrant init or init during teardown — both indicate a logic bug
        // upstream. Refuse rather than corrupt state.
        engine::boot_trace::breadcrumb("jni.init", "fail",
            "called while engine state != Uninit/Dead");
        return JNI_FALSE;
    }
    g_state.store(EngineState::Initializing, std::memory_order_release);

    const char* configDirStr = env->GetStringUTFChars(configDir, nullptr);
    if (!configDirStr) {
        g_state.store(EngineState::Uninit, std::memory_order_release);
        return JNI_FALSE;
    }
    std::string configDirCpp(configDirStr);
    env->ReleaseStringUTFChars(configDir, configDirStr);

    engine::EngineInitConfig config;
    config.configDir = configDirCpp;
    config.mode = engine::EngineMode::V2;
    config.enableValidation = static_cast<bool>(enableValidation);
    config.nativeWindowHandle = reinterpret_cast<void*>(static_cast<intptr_t>(nativeWindowHandle));

    // Heap-allocate the engine on first init. Reuse the existing instance on
    // subsequent re-inits (e.g. after a clean shutdown) only if it's still in
    // the Dead state — otherwise a fresh instance is built.
    if (!g_engineApp) {
        g_engineApp = std::make_unique<engine::EngineApp>();
    }

    bool ok = false;
    try {
        ok = g_engineApp->init(config);
    } catch (const std::exception& e) {
        fprintf(stderr, "[V2-JNI] init exception: %s\n", e.what());
        engine::boot_trace::writeCrashDump(
            std::string("init threw std::exception: ") + e.what());
        g_engineApp.reset();
        g_state.store(EngineState::Dead, std::memory_order_release);
        return JNI_FALSE;
    } catch (...) {
        fprintf(stderr, "[V2-JNI] init unknown exception\n");
        engine::boot_trace::writeCrashDump("init threw unknown exception");
        g_engineApp.reset();
        g_state.store(EngineState::Dead, std::memory_order_release);
        return JNI_FALSE;
    }
    if (ok) {
        g_state.store(EngineState::Ready, std::memory_order_release);
    } else {
        // init() returned false without throwing — the offending stage already
        // wrote a crash dump (see rollback paths in EngineApp::init). Add one
        // more line to v2_boot.log so the JNI boundary is recorded too.
        engine::boot_trace::breadcrumb("jni.init", "fail",
            "EngineApp::init returned false");
        g_engineApp.reset();
        g_state.store(EngineState::Dead, std::memory_order_release);
    }
    return ok ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jboolean JNICALL Java_com_radiance_v2_bridge_EngineBridge_nativeTick0(
    JNIEnv*, jclass) {
    std::lock_guard<std::recursive_mutex> guard(g_v2JniMtx);
    if (!isReady() || !g_engineApp) return JNI_FALSE;
    try {
        const bool ok = g_engineApp->tick();
        if (!ok) {
            // tick() returns false on device-lost or shutdown-requested. In both
            // cases the engine has already entered ShuttingDown — the classifier
            // entry has been written from inside tick() (device-lost path) or is
            // a clean shutdown (no dump needed).
        }
        return ok ? JNI_TRUE : JNI_FALSE;
    } catch (const std::exception& e) {
        fprintf(stderr, "[V2-JNI] tick exception: %s\n", e.what());
        engine::boot_trace::writeCrashDump(
            std::string("tick threw std::exception: ") + e.what());
        return JNI_FALSE;
    } catch (...) {
        fprintf(stderr, "[V2-JNI] tick unknown exception\n");
        engine::boot_trace::writeCrashDump("tick threw unknown exception");
        return JNI_FALSE;
    }
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_v2_bridge_EngineBridge_nativeShutdown0(
    JNIEnv*, jclass) {
    std::lock_guard<std::recursive_mutex> guard(g_v2JniMtx);
    auto cur = g_state.load(std::memory_order_acquire);
    // Idempotent: safe to call from any thread, any number of times.
    if (cur == EngineState::Uninit || cur == EngineState::Dead ||
        cur == EngineState::ShuttingDown || !g_engineApp) {
        return;
    }
    g_state.store(EngineState::ShuttingDown, std::memory_order_release);
    try {
        g_engineApp->shutdown();
    } catch (const std::exception& e) {
        fprintf(stderr, "[V2-JNI] shutdown exception: %s\n", e.what());
        engine::boot_trace::writeCrashDump(
            std::string("shutdown threw std::exception: ") + e.what());
    } catch (...) {
        fprintf(stderr, "[V2-JNI] shutdown unknown exception\n");
        engine::boot_trace::writeCrashDump("shutdown threw unknown exception");
    }
    // Destroy the engine object explicitly while we still hold the JNI mutex.
    // This avoids the static-destructor-ordering hazard at DLL detach.
    g_engineApp.reset();
    g_state.store(EngineState::Dead, std::memory_order_release);
}

extern "C" JNIEXPORT jboolean JNICALL Java_com_radiance_v2_bridge_EngineBridge_nativeIsInitialized0(
    JNIEnv*, jclass) {
    return isReady() ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jboolean JNICALL Java_com_radiance_v2_bridge_EngineBridge_nativeIsInWorld0(
    JNIEnv*, jclass) {
    if (!isReady() || !g_engineApp) return JNI_FALSE;
    return g_engineApp->isInWorld() ? JNI_TRUE : JNI_FALSE;
}

// V2 engine snapshot for crash-time diagnostics (A1). Deliberately lock-free
// on the fast path — reading g_state and the boot_trace "last X" accessors,
// which are themselves thread-safe. Never dereferences g_engineApp unless the
// state is Ready/ShuttingDown, which is the window during which the engine
// pointer is guaranteed live.
//
// Format (one key=value per line, no locale-dependent formatting):
//   state=<Uninit|Initializing|Ready|ShuttingDown|Dead>
//   frame=<uint64>
//   breadcrumb=<string or (none)>
//   bridge_cmd=<string or (none)>
//   mode=<V1|V2|Unknown>
//   in_world=<0|1>
//
// Java callers are expected to embed this as-is in crash-context.txt.
extern "C" JNIEXPORT jstring JNICALL Java_com_radiance_v2_bridge_EngineBridge_nativeGetV2Snapshot0(
    JNIEnv* env, jclass) {
    const auto cur = g_state.load(std::memory_order_acquire);
    const char* stateStr =
        cur == EngineState::Uninit       ? "Uninit"       :
        cur == EngineState::Initializing ? "Initializing" :
        cur == EngineState::Ready        ? "Ready"        :
        cur == EngineState::ShuttingDown ? "ShuttingDown" :
        cur == EngineState::Dead         ? "Dead"         : "?";

    std::string breadcrumb  = engine::boot_trace::lastBreadcrumb();
    std::string bridgeCmd   = engine::boot_trace::lastBridgeCommand();
    uint64_t    frameIndex  = engine::boot_trace::lastFrameIndex();

    // mode/in_world require g_engineApp, which can be reset under the JNI
    // mutex. Use try_lock so a shutdown-hook caller never blocks a dying
    // render thread — if we can't get the lock, just report "Unknown".
    const char* modeStr = "Unknown";
    int inWorld = 0;
    if (g_v2JniMtx.try_lock()) {
        if ((cur == EngineState::Ready || cur == EngineState::ShuttingDown) && g_engineApp) {
            modeStr = g_engineApp->mode() == engine::EngineMode::V2 ? "V2" : "V1";
            inWorld = g_engineApp->isInWorld() ? 1 : 0;
        }
        g_v2JniMtx.unlock();
    }

    std::string out;
    out.reserve(512);
    out += "state=";      out += stateStr;    out += '\n';
    out += "frame=";      out += std::to_string(frameIndex); out += '\n';
    out += "breadcrumb="; out += breadcrumb.empty() ? "(none)" : breadcrumb; out += '\n';
    out += "bridge_cmd="; out += bridgeCmd.empty()  ? "(none)" : bridgeCmd;  out += '\n';
    out += "mode=";       out += modeStr;     out += '\n';
    out += "in_world=";   out += (inWorld ? "1" : "0"); out += '\n';
    return env->NewStringUTF(out.c_str());
}

// --- World lifecycle ---

extern "C" JNIEXPORT void JNICALL Java_com_radiance_v2_bridge_EngineBridge_nativeWorldUnload0(
    JNIEnv*, jclass) {
    if (!isReady() || !g_engineApp) return;
    g_engineApp->services().bridge().post(engine::CmdWorldUnload{});
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_v2_bridge_EngineBridge_nativeWorldLoad0(
    JNIEnv*, jclass) {
    if (!isReady() || !g_engineApp) return;
    g_engineApp->services().bridge().post(engine::CmdWorldLoad{});
}

// --- Bridge commands ---

extern "C" JNIEXPORT void JNICALL Java_com_radiance_v2_bridge_EngineBridge_nativePostResize0(
    JNIEnv*, jclass, jint width, jint height) {
    if (!isReady() || !g_engineApp) return;
    g_engineApp->services().bridge().post(
        engine::CmdWindowResize{static_cast<uint32_t>(width), static_cast<uint32_t>(height)});
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_v2_bridge_EngineBridge_nativePostShutdown0(
    JNIEnv*, jclass) {
    if (!isReady() || !g_engineApp) return;
    g_engineApp->services().bridge().post(engine::CmdShutdown{});
}

extern "C" JNIEXPORT jstring JNICALL Java_com_radiance_v2_bridge_EngineBridge_nativeGetDeviceName0(
    JNIEnv* env, jclass) {
    if (!isReady() || !g_engineApp) return env->NewStringUTF("N/A");
    const auto& name = g_engineApp->services().device().caps().deviceName;
    return env->NewStringUTF(name.c_str());
}

extern "C" JNIEXPORT jstring JNICALL Java_com_radiance_v2_bridge_EngineBridge_nativeGetGpuProfile0(
    JNIEnv* env, jclass) {
    if (!isReady() || !g_engineApp) return env->NewStringUTF("");
    std::string profile = g_engineApp->services().metrics().getProfileString();
    return env->NewStringUTF(profile.c_str());
}

// --- Chunk submission ---

extern "C" JNIEXPORT void JNICALL Java_com_radiance_v2_bridge_EngineBridge_nativeSubmitChunk0(
    JNIEnv* env, jclass,
    jint chunkX, jint sectionY, jint chunkZ,
    jint originX, jint originY, jint originZ,
    jlong vertexPtr, jint vertexSize,
    jlong indexPtr, jint indexCount,
    jint triangleCount,
    jlong lightDataPtr, jint lightCount) {
    if (!isReady() || !g_engineApp) return;

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

    // Copy per-chunk light data (16 bytes per entry: float x, y, z + int typeId)
    if (lightDataPtr && lightCount > 0) {
        cmd.lightCount = static_cast<uint32_t>(lightCount);
        const auto* src = reinterpret_cast<const uint8_t*>(lightDataPtr);
        cmd.lightData.assign(src, src + lightCount * 16);
    }

    g_engineApp->services().bridge().post(std::move(cmd));
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_v2_bridge_EngineBridge_nativeRemoveChunk0(
    JNIEnv*, jclass, jint chunkX, jint sectionY, jint chunkZ) {
    if (!isReady() || !g_engineApp) return;
    g_engineApp->services().bridge().post(engine::CmdChunkRemove{chunkX, sectionY, chunkZ});
}

// --- Camera update ---

extern "C" JNIEXPORT void JNICALL Java_com_radiance_v2_bridge_EngineBridge_nativeUpdateCamera0(
    JNIEnv* env, jclass,
    jfloatArray viewMatrix, jfloatArray projMatrix,
    jfloat posX, jfloat posY, jfloat posZ,
    jfloat dirX, jfloat dirY, jfloat dirZ,
    jfloat nearPlane, jfloat farPlane) {
    if (!isReady() || !g_engineApp) return;

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

    g_engineApp->services().bridge().post(std::move(cmd));
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
    if (!isReady() || !g_engineApp) return;

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

    g_engineApp->services().bridge().post(std::move(cmd));
}

// --- Texture mapping upload ---

extern "C" JNIEXPORT void JNICALL Java_com_radiance_v2_bridge_EngineBridge_nativeUpdateTextureMapping0(
    JNIEnv*, jclass, jlong dataPtr, jint dataSize) {
    if (!isReady() || !g_engineApp) return;
    if (!dataPtr || dataSize <= 0) return;

    engine::CmdTextureMappingUpdate cmd;
    const auto* src = reinterpret_cast<const uint8_t*>(dataPtr);
    cmd.data.assign(src, src + dataSize);

    g_engineApp->services().bridge().post(std::move(cmd));
}

// --- Texture upload ---

extern "C" JNIEXPORT void JNICALL Java_com_radiance_v2_bridge_EngineBridge_nativeSubmitSpriteTable0(
    JNIEnv*, jclass, jlong metaPtr, jint count, jint atlasWidth, jint atlasHeight, jint spriteSize) {
    if (!isReady() || !g_engineApp) return;
    if (!metaPtr || count <= 0) return;

    engine::CmdSpriteTableUpload cmd;
    cmd.count = static_cast<uint32_t>(count);
    cmd.atlasWidth = static_cast<uint32_t>(atlasWidth);
    cmd.atlasHeight = static_cast<uint32_t>(atlasHeight);
    cmd.spriteSize = static_cast<uint32_t>(spriteSize);

    const auto* src = reinterpret_cast<const uint8_t*>(metaPtr);
    cmd.metadata.assign(src, src + count * 16);

    g_engineApp->services().bridge().post(std::move(cmd));
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_v2_bridge_EngineBridge_nativeSubmitSpritePixels0(
    JNIEnv*, jclass, jlong dataPtr, jint totalBytes) {
    if (!isReady() || !g_engineApp) return;
    if (!dataPtr || totalBytes <= 0) return;

    engine::CmdSpritePixelsUpload cmd;
    const auto* src = reinterpret_cast<const uint8_t*>(dataPtr);
    cmd.pixels.assign(src, src + totalBytes);

    g_engineApp->services().bridge().post(std::move(cmd));
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_v2_bridge_EngineBridge_nativeSubmitSpriteAuxPixels0(
    JNIEnv*, jclass, jlong specPtr, jlong normPtr, jint bytesPerType) {
    if (!isReady() || !g_engineApp) return;
    if (bytesPerType <= 0) return;

    engine::CmdSpriteAuxPixelsUpload cmd;
    if (specPtr) {
        const auto* src = reinterpret_cast<const uint8_t*>(specPtr);
        cmd.specularPixels.assign(src, src + bytesPerType);
    }
    if (normPtr) {
        const auto* src = reinterpret_cast<const uint8_t*>(normPtr);
        cmd.normalPixels.assign(src, src + bytesPerType);
    }

    g_engineApp->services().bridge().post(std::move(cmd));
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_v2_bridge_EngineBridge_nativeSubmitAnimationFrames0(
    JNIEnv*, jclass, jlong dataPtr, jint totalBytes) {
    if (!isReady() || !g_engineApp) return;

    engine::CmdAnimationFramesUpload cmd;
    if (dataPtr && totalBytes > 0) {
        const auto* src = reinterpret_cast<const uint8_t*>(dataPtr);
        cmd.data.assign(src, src + totalBytes);
    }

    g_engineApp->services().bridge().post(std::move(cmd));
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_v2_bridge_EngineBridge_nativeTextureFinalize0(
    JNIEnv*, jclass) {
    if (!isReady() || !g_engineApp) return;
    g_engineApp->services().bridge().post(engine::CmdTextureFinalize{});
}

// --- Area light upload ---

extern "C" JNIEXPORT void JNICALL Java_com_radiance_v2_bridge_EngineBridge_nativeSubmitAreaLights0(
    JNIEnv*, jclass, jlong dataPtr, jint dataSize, jint lightCount) {
    if (!isReady() || !g_engineApp) return;

    engine::CmdAreaLightUpload cmd;
    cmd.lightCount = static_cast<uint32_t>(lightCount);

    if (dataPtr && dataSize > 0) {
        const auto* src = reinterpret_cast<const uint8_t*>(dataPtr);
        cmd.data.assign(src, src + dataSize);
    }

    g_engineApp->services().bridge().post(std::move(cmd));
}

// --- Emission data upload ---

extern "C" JNIEXPORT void JNICALL Java_com_radiance_v2_bridge_EngineBridge_nativeUploadEmissionData0(
    JNIEnv*, jclass, jlong dataPtr) {
    if (!isReady() || !g_engineApp) return;
    if (!dataPtr) return;

    // Layout: 50*4 floats (emissionData) followed by 13*4 floats (emissiveGamut)
    // Total: (50 + 13) * 4 * 4 = 1008 bytes
    const auto* src = reinterpret_cast<const float*>(dataPtr);

    engine::CmdEmissionDataUpload cmd;
    std::memcpy(cmd.emissionData, src, sizeof(cmd.emissionData));
    std::memcpy(cmd.emissiveGamut, src + 50 * 4, sizeof(cmd.emissiveGamut));

    g_engineApp->services().bridge().post(std::move(cmd));
}

// --- Entity batch submission ---

extern "C" JNIEXPORT void JNICALL Java_com_radiance_v2_bridge_EngineBridge_nativeSubmitEntityBatch0(
    JNIEnv*, jclass,
    jlong entryPtr, jint entryCount,
    jlong vertexPtr, jint vertexBytes,
    jlong indexPtr, jint indexCount) {
    if (!isReady() || !g_engineApp) return;
    if (!entryPtr || entryCount <= 0) return;

    engine::CmdEntityBatchSubmit cmd;

    // Each entry is 36 bytes packed from Java:
    // Per entry: int hash(4), float x(4), float y(4), float z(4), int rtFlag(4),
    //            int coordSystem(4), int vertexOffset(4), int indexOffset(4), int triangleCount(4) = 36 bytes
    const auto* src = reinterpret_cast<const uint8_t*>(entryPtr);
    cmd.entities.resize(static_cast<size_t>(entryCount));
    for (int i = 0; i < entryCount; ++i) {
        const auto* p = src + i * 36;
        uint32_t hash;       std::memcpy(&hash, p + 0, 4);
        float px, py, pz;    std::memcpy(&px, p + 4, 4);
                             std::memcpy(&py, p + 8, 4);
                             std::memcpy(&pz, p + 12, 4);
        int32_t rtf;         std::memcpy(&rtf, p + 16, 4);
        int32_t cs;          std::memcpy(&cs, p + 20, 4);
        uint32_t vo;         std::memcpy(&vo, p + 24, 4);
        uint32_t io;         std::memcpy(&io, p + 28, 4);
        uint32_t tc;         std::memcpy(&tc, p + 32, 4);

        auto& e = cmd.entities[i];
        e.hashCode = hash;
        e.posX = px; e.posY = py; e.posZ = pz;
        e.rtFlag = static_cast<uint8_t>(rtf);
        e.coordSystem = static_cast<uint8_t>(cs);
        e.vertexOffset = vo;
        e.indexOffset = io;
        e.triangleCount = tc;
    }

    // Copy vertex data
    if (vertexPtr && vertexBytes > 0) {
        const auto* vsrc = reinterpret_cast<const uint8_t*>(vertexPtr);
        cmd.vertexData.assign(vsrc, vsrc + vertexBytes);
    }

    // Copy index data
    if (indexPtr && indexCount > 0) {
        const auto* isrc = reinterpret_cast<const uint32_t*>(indexPtr);
        cmd.indexData.assign(isrc, isrc + indexCount);
    }

    g_engineApp->services().bridge().post(std::move(cmd));
}

// --- Post-entity submission (particles, hand items, weather) ---

extern "C" JNIEXPORT void JNICALL Java_com_radiance_v2_bridge_EngineBridge_nativeSubmitPostEntities0(
    JNIEnv* env, jclass,
    jbyteArray vertexData, jintArray indexData, jintArray drawCallData) {
    if (!isReady() || !g_engineApp) return;
    if (!vertexData || !indexData || !drawCallData) return;

    engine::CmdEntityPostDraw cmd;

    // Copy vertex data (PBRTriangle, 96 bytes per vertex)
    {
        jsize len = env->GetArrayLength(vertexData);
        if (len > 0) {
            jbyte* src = env->GetByteArrayElements(vertexData, nullptr);
            if (src) {
                cmd.vertexData.assign(reinterpret_cast<const uint8_t*>(src),
                                      reinterpret_cast<const uint8_t*>(src) + len);
                env->ReleaseByteArrayElements(vertexData, src, JNI_ABORT);
            }
        }
    }

    // Copy index data (uint32 indices)
    {
        jsize len = env->GetArrayLength(indexData);
        if (len > 0) {
            jint* src = env->GetIntArrayElements(indexData, nullptr);
            if (src) {
                cmd.indexData.resize(static_cast<size_t>(len));
                for (jsize i = 0; i < len; ++i) {
                    cmd.indexData[i] = static_cast<uint32_t>(src[i]);
                }
                env->ReleaseIntArrayElements(indexData, src, JNI_ABORT);
            }
        }
    }

    // Copy draw call data (each draw call = 3 ints: vertexByteOffset, indexElementOffset, indexCount)
    {
        jsize len = env->GetArrayLength(drawCallData);
        if (len > 0 && len % 3 == 0) {
            jint* src = env->GetIntArrayElements(drawCallData, nullptr);
            if (src) {
                int numDrawCalls = len / 3;
                cmd.drawCalls.resize(static_cast<size_t>(numDrawCalls));
                for (int i = 0; i < numDrawCalls; ++i) {
                    cmd.drawCalls[i].vertexByteOffset = static_cast<uint32_t>(src[i * 3 + 0]);
                    cmd.drawCalls[i].indexElementOffset = static_cast<uint32_t>(src[i * 3 + 1]);
                    cmd.drawCalls[i].indexCount = static_cast<uint32_t>(src[i * 3 + 2]);
                }
                env->ReleaseIntArrayElements(drawCallData, src, JNI_ABORT);
            }
        }
    }

    g_engineApp->services().bridge().post(std::move(cmd));
}

// --- Offline accumulation reset ---

extern "C" JNIEXPORT void JNICALL Java_com_radiance_v2_bridge_EngineBridge_nativeResetAccumulation0(
    JNIEnv*, jclass) {
    if (!isReady() || !g_engineApp) return;
    g_engineApp->services().bridge().post(engine::CmdResetAccumulation{});
}

#endif // MCVR_ENABLE_ENGINE_V2
