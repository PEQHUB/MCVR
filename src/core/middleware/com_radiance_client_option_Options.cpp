#include "com_radiance_client_option_Options.h"

#include "core/all_extern.hpp"
#include "core/render/buffers.hpp"
#include "core/render/chunks.hpp"
#include "core/render/lights.hpp"
#include "core/render/render_framework.hpp"
#include "core/render/radiance_logger.hpp"
#include "core/render/renderer.hpp"
#include "core/render/streamline_context.hpp"
#include "core/render/modules/world/frame_gen/frame_gen_manager.hpp"
#include "core/render/textures.hpp"
#include "core/render/world.hpp"

#include <algorithm>

static void applyReflexSettings(); // forward decl — defined below

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetMaxFps(JNIEnv *,
                                                                               jclass,
                                                                               jint maxFps,
                                                                               jboolean write) {
    Renderer::options.maxFps = maxFps;
    // Don't call applyReflexSettings() here — slider drag fires per-pixel.
    // Reflex frame limit is applied lazily via Renderer::options.reflexDirty flag.
    Renderer::options.reflexDirty = true;
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeApplyReflexSettings(
    JNIEnv *, jclass) {
    applyReflexSettings();
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetInactivityFpsLimit(JNIEnv *,
                                                                                           jclass,
                                                                                           jint inactivityFpsLimit,
                                                                                           jboolean write) {
    Renderer::options.inactivityFpsLimit = inactivityFpsLimit;
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetVsync(JNIEnv *,
                                                                              jclass,
                                                                              jboolean vsync,
                                                                              jboolean write) {
    Renderer::options.vsync = vsync;
    if (write) Renderer::options.needRecreate = true;
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetChunkBuildingBatchSize(
    JNIEnv *, jclass, jint chunkBuildingBatchSize, jboolean write) {
    Renderer::options.chunkBuildingBatchSize = chunkBuildingBatchSize;
    if (write) Renderer::instance().world()->chunks()->resetScheduler();
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetChunkBuildingTotalBatches(
    JNIEnv *, jclass, jint chunkBuildingTotalBatches, jboolean write) {
    Renderer::options.chunkBuildingTotalBatches = chunkBuildingTotalBatches;
    if (write) Renderer::instance().world()->chunks()->resetScheduler();
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetTonemappingMode(
    JNIEnv *, jclass, jint mode, jboolean write) {
    Renderer::options.tonemappingMode = mode;
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetSdrTransferFunction(
    JNIEnv *, jclass, jint mode, jboolean write) {
    Renderer::options.sdrTransferFunction = static_cast<uint32_t>(std::clamp(mode, 0, 1));
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetRayBounces(
    JNIEnv *, jclass, jint bounces, jboolean write) {
    Renderer::options.rayBounces = bounces;
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetOMMEnabled(
    JNIEnv *, jclass, jboolean enabled, jboolean write) {
    Renderer::options.ommEnabled = enabled;
    if (write) {
        Renderer::options.needRecreate = true;
        Renderer::instance().world()->chunks()->resetScheduler();
    }
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetGreedyMeshingEnabled(
    JNIEnv *, jclass, jboolean enabled, jboolean write) {
    Renderer::options.greedyMeshingEnabled = enabled;
    if (write) {
        Renderer::options.needRecreate = true;
        Renderer::instance().world()->chunks()->resetScheduler();
    }
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetOMMBakerLevel(
    JNIEnv *, jclass, jint level, jboolean write) {
    Renderer::options.ommBakerLevel = static_cast<uint32_t>(std::clamp(level, 1, 8));
    if (write) {
        Renderer::options.needRecreate = true;
        Renderer::instance().world()->chunks()->resetScheduler();
    }
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetSimplifiedIndirect(
    JNIEnv *, jclass, jboolean enabled, jboolean write) {
    Renderer::options.simplifiedIndirect = enabled;
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetSEREnabled(
    JNIEnv *, jclass, jboolean enabled, jboolean write) {
    Renderer::options.serEnabled = (enabled == JNI_TRUE);
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetSERHintsEnabled(
    JNIEnv *, jclass, jboolean enabled, jboolean write) {
    Renderer::options.serHintsEnabled = (enabled == JNI_TRUE);
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetSharcEnabled(
    JNIEnv *, jclass, jboolean enabled, jboolean write) {
    Renderer::options.sharcEnabled = (enabled == JNI_TRUE);
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetSharcSceneScale(
    JNIEnv *, jclass, jfloat scale, jboolean write) {
    Renderer::options.sharcSceneScale = scale;
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetSharcRoughnessThreshold(
    JNIEnv *, jclass, jfloat threshold, jboolean write) {
    Renderer::options.sharcRoughnessThreshold = threshold;
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetSharcAccumulationFrames(
    JNIEnv *, jclass, jint frames, jboolean write) {
    Renderer::options.sharcAccumulationFrames = frames;
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetSharcStaleFrames(
    JNIEnv *, jclass, jint frames, jboolean write) {
    Renderer::options.sharcStaleFrames = frames;
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetSharcDownscale(
    JNIEnv *, jclass, jint downscale, jboolean write) {
    Renderer::options.sharcDownscale = downscale;
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetSharcUpdateBlockSize(
    JNIEnv *, jclass, jint blockSize, jboolean write) {
    Renderer::options.sharcUpdateBlockSize = blockSize;
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetSharcUpdateBounces(
    JNIEnv *, jclass, jint bounces, jboolean write) {
    Renderer::options.sharcUpdateBounces = bounces;
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetSharcCapacityExponent(
    JNIEnv *, jclass, jint exponent, jboolean write) {
    Renderer::options.sharcCapacityExponent = exponent;
    // Note: buffer reallocation requires restart. The exponent is used at init time.
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetOutputScale2x(
    JNIEnv *, jclass, jboolean enabled, jboolean write) {
    Renderer::options.outputScale2x = enabled;
    if (write) Renderer::options.needRecreate = true;
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetDlssQuality(
    JNIEnv *, jclass, jint quality, jboolean write) {
    Renderer::options.upscalerMode = quality;
    if (write) Renderer::options.needRecreate = true;
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetDlssResOverride(
    JNIEnv *, jclass, jint resOverride, jboolean write) {
    Renderer::options.upscalerResOverride = resOverride;
    if (write) Renderer::options.needRecreate = true;
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetMinExposure(
    JNIEnv *, jclass, jfloat minExposure, jboolean write) {
    Renderer::options.minExposure = minExposure;
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetMaxExposure(
    JNIEnv *, jclass, jint maxExposure, jboolean write) {
    Renderer::options.maxExposure = static_cast<float>(maxExposure) * 0.1f;
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetExposureCompensation(
    JNIEnv *, jclass, jfloat ec, jboolean write) {
    Renderer::options.exposureCompensation = ec;
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetManualExposureEnabled(
    JNIEnv *, jclass, jboolean enabled, jboolean write) {
    Renderer::options.manualExposureEnabled = enabled;
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetManualExposure(
    JNIEnv *, jclass, jfloat exposure, jboolean write) {
    Renderer::options.manualExposure = std::max(0.0001f, exposure);
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetSharpenerMode(
    JNIEnv *, jclass, jint mode, jboolean write) {
    Renderer::options.sharpenerMode = static_cast<uint32_t>(std::clamp(static_cast<int>(mode), 0, 2));
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetCasSharpness(
    JNIEnv *, jclass, jfloat sharpness, jboolean write) {
    Renderer::options.casSharpness = std::clamp(sharpness, 0.0f, 1.0f);
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetMiddleGrey(
    JNIEnv *, jclass, jfloat mg, jboolean write) {
    Renderer::options.middleGrey = mg;
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetLwhite(
    JNIEnv *, jclass, jfloat lw, jboolean write) {
    Renderer::options.Lwhite = lw;
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetBrightAdaptSpeed(
    JNIEnv *, jclass, jfloat speed, jboolean write) {
    Renderer::options.brightAdaptSpeed = speed;
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetDarkAdaptSpeed(
    JNIEnv *, jclass, jfloat speed, jboolean write) {
    Renderer::options.darkAdaptSpeed = speed;
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetSceneChangeThreshold(
    JNIEnv *, jclass, jfloat threshold, jboolean write) {
    Renderer::options.sceneChangeThreshold = threshold;
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetCenterWeightStrength(
    JNIEnv *, jclass, jfloat strength, jboolean write) {
    Renderer::options.centerWeightStrength = strength;
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetDlssPreset(
    JNIEnv *, jclass, jint preset, jboolean write) {
    // Clamp to valid DLSS RR preset range (A=0 through G=6)
    Renderer::options.upscalerPreset = static_cast<uint32_t>(std::clamp(preset, 0, 6));
    if (write) Renderer::options.needRecreate = true;
}

// --- HDR10 Output ---

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetHdrEnabled(
    JNIEnv *, jclass, jboolean enabled, jboolean write) {
    Renderer::options.hdrEnabled = enabled;
    // Toggling HDR requires swapchain recreation (format + color space change)
    if (write) Renderer::options.needRecreate = true;
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetHdrScrgbMode(
    JNIEnv *, jclass, jboolean scrgb, jboolean write) {
    Renderer::options.hdrScrgbMode = scrgb;
    // Changing HDR format requires swapchain recreation
    if (write) Renderer::options.needRecreate = true;
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetHdrPeakNits(
    JNIEnv *, jclass, jint nits, jboolean write) {
    Renderer::options.hdrPeakNits = static_cast<float>(nits);
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetHdrPaperWhiteNits(
    JNIEnv *, jclass, jint nits, jboolean write) {
    Renderer::options.hdrPaperWhiteNits = static_cast<float>(nits);
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetHdrUiBrightnessNits(
    JNIEnv *, jclass, jint nits, jboolean write) {
    Renderer::options.hdrUiBrightnessNits = static_cast<float>(nits);
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetSaturation(
    JNIEnv *, jclass, jfloat saturation, jboolean write) {
    Renderer::options.saturation = saturation;
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetSaturationAdaptive(
    JNIEnv *, jclass, jboolean enabled, jboolean write) {
    Renderer::options.saturationAdaptive = enabled;
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetNoiseLOD(
    JNIEnv *, jclass, jboolean enabled, jboolean write) {
    Renderer::options.noiseLOD = enabled;
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetMultiScatterGGX(
    JNIEnv *, jclass, jboolean enabled, jboolean write) {
    Renderer::options.multiScatterGGX = enabled;
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetEonDiffuse(
    JNIEnv *, jclass, jboolean enabled, jboolean write) {
    Renderer::options.eonDiffuse = enabled;
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetColorExpansion(
    JNIEnv *, jclass, jfloat colorExpansion, jboolean write) {
    Renderer::options.colorExpansion = colorExpansion;
}

// PsychoV tonemapper setters
extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetPsychoEnabled(
    JNIEnv *, jclass, jboolean enabled, jboolean write) {
    Renderer::options.psychoEnabled = enabled;
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetHdrTonemapMode(
    JNIEnv *, jclass, jint mode, jboolean write) {
    Renderer::options.hdrTonemapMode = mode;
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetPsychoHighlights(
    JNIEnv *, jclass, jfloat v, jboolean write) { Renderer::options.psychoHighlights = v; }

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetPsychoShadows(
    JNIEnv *, jclass, jfloat v, jboolean write) { Renderer::options.psychoShadows = v; }

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetPsychoContrast(
    JNIEnv *, jclass, jfloat v, jboolean write) { Renderer::options.psychoContrast = v; }

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetPsychoPurity(
    JNIEnv *, jclass, jfloat v, jboolean write) { Renderer::options.psychoPurity = v; }

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetPsychoBleaching(
    JNIEnv *, jclass, jfloat v, jboolean write) { Renderer::options.psychoBleaching = v; }

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetPsychoClipPoint(
    JNIEnv *, jclass, jfloat v, jboolean write) { Renderer::options.psychoClipPoint = v; }

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetPsychoHueRestore(
    JNIEnv *, jclass, jfloat v, jboolean write) { Renderer::options.psychoHueRestore = v; }

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetPsychoAdaptContrast(
    JNIEnv *, jclass, jfloat v, jboolean write) { Renderer::options.psychoAdaptContrast = v; }

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetPsychoWhiteCurve(
    JNIEnv *, jclass, jint v, jboolean write) { Renderer::options.psychoWhiteCurve = v; }

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetPsychoConeExponent(
    JNIEnv *, jclass, jfloat v, jboolean write) { Renderer::options.psychoConeExponent = v; }

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetTonemapParam(
    JNIEnv *, jclass, jint index, jfloat value, jboolean write) {
    if (index >= 0 && index < 8) Renderer::options.tonemapParams[index] = value;
}

extern "C" JNIEXPORT jboolean JNICALL Java_com_radiance_client_option_Options_nativeIsHdrActive(
    JNIEnv *, jclass) {
    auto *renderer = Renderer::try_instance();
    if (renderer == nullptr) {
        return JNI_FALSE;
    }

    auto framework = renderer->framework();
    if (framework == nullptr || framework->swapchain() == nullptr) {
        return JNI_FALSE;
    }

    bool hdrActive = Renderer::options.hdrEnabled && framework->swapchain()->isHDR();
    return hdrActive ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jboolean JNICALL Java_com_radiance_client_option_Options_nativeIsHdrSupported(
    JNIEnv *, jclass) {
    auto *renderer = Renderer::try_instance();
    if (renderer == nullptr) {
        return JNI_FALSE;
    }

    auto framework = renderer->framework();
    if (framework == nullptr || framework->swapchain() == nullptr) {
        return JNI_FALSE;
    }

    return framework->swapchain()->isHDRSupported() ? JNI_TRUE : JNI_FALSE;
}

// --- NVIDIA Reflex + VRR helpers ---

static int getDisplayRefreshRate() {
    auto *renderer = Renderer::try_instance();
    if (!renderer) return 0;
    auto fw = renderer->framework();
    if (!fw) return 0;
    auto window = fw->window();
    if (!window) return 0;

    GLFWmonitor *monitor = GLFW_GetWindowMonitor(window->window());
    if (!monitor) monitor = GLFW_GetPrimaryMonitor();
    if (!monitor) return 0;

    const GLFWvidmode *mode = GLFW_GetVideoMode(monitor);
    return mode ? mode->refreshRate : 0;
}

// Shared helper: pushes current reflexEnabled/reflexBoost/vrrMode to Streamline.
// Called whenever any of the three toggles change. No-op on non-Windows platforms.
static void applyReflexSettings() {
#ifdef _WIN32
    if (!StreamlineContext::isReflexAvailable()) return;

    sl::ReflexMode mode = sl::ReflexMode::eOff;
    // DLSS-G requires Reflex — force at least LowLatency when FG is enabled
    bool reflexOn = Renderer::options.reflexEnabled || Renderer::options.frameGenEnabled;
    if (reflexOn) {
        mode = Renderer::options.reflexBoost
            ? sl::ReflexMode::eLowLatencyWithBoost
            : sl::ReflexMode::eLowLatency;
    }

    // FPS limit via Reflex frame limiter (0 = unlimited)
    // FPS limit via Reflex frame limiter (0 = unlimited)
    uint32_t frameLimitUs = 0;
    uint32_t maxFps = Renderer::options.maxFps;
    if (maxFps > 0 && maxFps < 1000000) {
        frameLimitUs = 1000000 / maxFps;
    }

    StreamlineContext::setReflexOptions(mode, frameLimitUs);
#endif
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetReflexEnabled(
    JNIEnv *, jclass, jboolean enabled, jboolean write) {
    Renderer::options.reflexEnabled = enabled;
    applyReflexSettings();
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetReflexBoost(
    JNIEnv *, jclass, jboolean enabled, jboolean write) {
    Renderer::options.reflexBoost = enabled;
    applyReflexSettings();
}

extern "C" JNIEXPORT jboolean JNICALL Java_com_radiance_client_option_Options_nativeIsReflexSupported(
    JNIEnv *, jclass) {
    return StreamlineContext::isReflexAvailable() ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetVrrMode(
    JNIEnv *, jclass, jboolean enabled, jboolean write) {
    Renderer::options.vrrMode = enabled;
    // Don't call applyReflexSettings() — vrrMode is a UI hint only.
    // Frame limit is controlled solely by maxFps via the deferred reflexDirty path.
}

extern "C" JNIEXPORT jint JNICALL Java_com_radiance_client_option_Options_nativeGetDisplayRefreshRate(
    JNIEnv *, jclass) {
    return static_cast<jint>(getDisplayRefreshRate());
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeRebuildChunks(
    JNIEnv *, jclass) {
    Renderer::instance().world()->chunks()->resetScheduler();
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeResetExposureAdaptation(
    JNIEnv *, jclass) {
    Renderer::resetExposureAdaptation = true;
}

// --- Area Lights ---

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetAreaLightsEnabled(
    JNIEnv *, jclass, jboolean enabled, jboolean write) {
    Renderer::options.areaLightsEnabled = enabled;
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetRestirEnabled(
    JNIEnv *, jclass, jboolean enabled, jboolean write) {
    Renderer::options.restirEnabled = enabled;
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetAreaLightIntensity(
    JNIEnv *, jclass, jfloat intensity, jboolean write) {
    Renderer::options.areaLightIntensity = std::clamp(intensity, 0.0f, 5.0f);
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetAreaLightRange(
    JNIEnv *, jclass, jint range, jboolean write) {
    Renderer::options.areaLightRange = static_cast<float>(std::clamp(range, 8, 512));
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetChunkCullDistance(
    JNIEnv *, jclass, jint distance, jboolean write) {
    // UI is in chunks (0-512), C++ pipeline works in blocks (×16). 0 = unlimited (100000).
    int chunks = std::clamp(distance, 0, 512);
    Renderer::options.chunkCullDistance = chunks == 0 ? 100000.0f : static_cast<float>(chunks * 16);
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetChunkLodDistance(
    JNIEnv *, jclass, jint distance, jboolean write) {
    // UI is in chunks (0-512), C++ pipeline works in blocks (×16). 0 = all full quality (100000).
    int chunks = std::clamp(distance, 0, 512);
    Renderer::options.chunkLodDistance = chunks == 0 ? 100000.0f : static_cast<float>(chunks * 16);
    // LOD format is selected at BLAS build time based on chunk distance from camera.
    // No invalidation needed — chunks retain their current format until naturally rebuilt.
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetExtendedRenderDistance(
    JNIEnv *, jclass, jint distance, jboolean write) {
    Renderer::options.extendedRenderDistance = static_cast<uint32_t>(std::clamp(distance, 0, 512));
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetShadowSoftness(
    JNIEnv *, jclass, jfloat softness, jboolean write) {
    Renderer::options.shadowSoftness = std::clamp(softness, 0.0f, 2.0f);
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetAreaLightBlockIntensity(
    JNIEnv *, jclass, jint lightTypeId, jfloat intensity) {
    if (lightTypeId >= 0 && lightTypeId < LIGHT_TYPE_COUNT) {
        Renderer::options.perBlockIntensity[lightTypeId] = std::max(0.0f, intensity);
    }
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetAreaLightBlockScale(
    JNIEnv *, jclass, jint lightTypeId, jfloat scale) {
    if (lightTypeId >= 0 && lightTypeId < LIGHT_TYPE_COUNT) {
        Renderer::options.perBlockScale[lightTypeId] = std::max(0.0f, scale);
    }
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetAreaLightBlockYOffset(
    JNIEnv *, jclass, jint lightTypeId, jfloat offset) {
    if (lightTypeId >= 0 && lightTypeId < LIGHT_TYPE_COUNT) {
        Renderer::options.perBlockYOffset[lightTypeId] = offset;
    }
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetAreaLightBlockColor(
    JNIEnv *, jclass, jint lightTypeId, jfloat r, jfloat g, jfloat b) {
    if (lightTypeId >= 0 && lightTypeId < LIGHT_TYPE_COUNT) {
        Renderer::options.perBlockColorR[lightTypeId] = r;
        Renderer::options.perBlockColorG[lightTypeId] = g;
        Renderer::options.perBlockColorB[lightTypeId] = b;
    }
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetBlockLightMode(
    JNIEnv *, jclass, jint lightTypeId, jint mode) {
    if (lightTypeId >= 0 && lightTypeId < LIGHT_TYPE_COUNT) {
        Renderer::options.blockLightMode[lightTypeId] = std::clamp(mode, 0, 2);
    }
}

// --- Per-Block Temperature ---

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetBlockTemperature(
    JNIEnv *, jclass, jint typeId, jfloat kelvin, jboolean write) {
    if (typeId >= 0 && typeId < 50) {
        Renderer::options.perBlockTemperatureK[typeId] = std::clamp(kelvin, 773.15f, 4273.15f);
    }
}

// --- ReSTIR Tuning ---

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetRestirCandidates(
    JNIEnv *, jclass, jint candidates, jboolean write) {
    Renderer::options.restirCandidates = std::clamp(candidates, 8, 64);
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetRestirTemporalMClamp(
    JNIEnv *, jclass, jint clamp, jboolean write) {
    Renderer::options.restirTemporalMClamp = std::clamp(clamp, 5, 50);
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetRestirWClamp(
    JNIEnv *, jclass, jint clamp, jboolean write) {
    Renderer::options.restirWClamp = std::clamp(clamp, 10, 200);
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetRestirSpatialTaps(
    JNIEnv *, jclass, jint taps, jboolean write) {
    Renderer::options.restirSpatialTaps = std::clamp(taps, 1, 10);
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetRestirSpatialRadius(
    JNIEnv *, jclass, jint radius, jboolean write) {
    Renderer::options.restirSpatialRadius = std::clamp(radius, 5, 60);
}

// --- ReSTIR Performance ---

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetRestirSimplifiedBRDF(
    JNIEnv *, jclass, jboolean enabled, jboolean write) {
    Renderer::options.restirSimplifiedBRDF = enabled;
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetRestirSpatialEnabled(
    JNIEnv *, jclass, jboolean enabled, jboolean write) {
    Renderer::options.restirSpatialEnabled = enabled;
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetRestirBounceEnabled(
    JNIEnv *, jclass, jboolean enabled, jboolean write) {
    Renderer::options.restirBounceEnabled = enabled;
}

// --- Parallax Occlusion Mapping ---

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetPOMEnabled(
    JNIEnv *, jclass, jboolean v, jboolean) {
    Renderer::options.pomEnabled = (v == JNI_TRUE);
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetPOMHeightScale(
    JNIEnv *, jclass, jfloat v, jboolean) {
    Renderer::options.pomHeightScale = std::clamp(v, 0.01f, 0.50f);
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetPOMSteps(
    JNIEnv *, jclass, jint v, jboolean) {
    Renderer::options.pomSteps = std::clamp(v, 8, 512);
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetPOMRefinement(
    JNIEnv *, jclass, jint v, jboolean) {
    Renderer::options.pomRefinement = std::clamp(v, 0, 8);
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetPOMFadeDistance(
    JNIEnv *, jclass, jfloat v, jboolean) {
    Renderer::options.pomFadeDistance = std::clamp(v, 8.0f, 256.0f);
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetDisplacementQuality(
    JNIEnv *, jclass, jint v, jboolean) {
    Renderer::options.displacementQuality = std::clamp(v, 0, 4);
    Renderer::options.needRecreate = true;
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetTessMaxLevel(
    JNIEnv *, jclass, jint v, jboolean) {
    Renderer::options.tessMaxLevel = std::clamp(static_cast<uint32_t>(v), 2u, 32u);
    Renderer::options.needRecreate = true;
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetTessNearDist(
    JNIEnv *, jclass, jint v, jboolean) {
    Renderer::options.tessNearDist = std::clamp(static_cast<float>(v), 8.0f, 256.0f);
    Renderer::options.needRecreate = true;
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetTessMidDist(
    JNIEnv *, jclass, jint v, jboolean) {
    Renderer::options.tessMidDist = std::clamp(static_cast<float>(v), 16.0f, 384.0f);
    Renderer::options.needRecreate = true;
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetTessFarDist(
    JNIEnv *, jclass, jint v, jboolean) {
    Renderer::options.tessFarDist = std::clamp(static_cast<float>(v), 32.0f, 512.0f);
    Renderer::options.needRecreate = true;
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetLoggingEnabled(
    JNIEnv *, jclass, jboolean enabled, jboolean) {
    Renderer::options.loggingEnabled = enabled;
    RadianceLogger::setEnabled(enabled, Renderer::folderPath);
}

// --- Offline Accumulation ---

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetOfflineGroundTruth(
    JNIEnv *, jclass, jboolean enabled, jboolean) {
    Renderer::options.offlineGroundTruth = enabled;
    // Ground truth only controls shader quality (Beer's Law, no clamping, physical sun, etc.)
    // It does NOT force denoising mode, native res, or variance reduction settings
    Renderer::accumFrameCount = 0;  // reset accumulation on toggle
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetBeerLawShadows(
    JNIEnv *, jclass, jboolean enabled, jboolean) {
    Renderer::options.beerLawShadows = enabled;
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetNoEmissionClamp(
    JNIEnv *, jclass, jboolean enabled, jboolean) {
    Renderer::options.noEmissionClamp = enabled;
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetPhysicalSunDisk(
    JNIEnv *, jclass, jboolean enabled, jboolean) {
    Renderer::options.physicalSunDisk = enabled;
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetNoHandAmbient(
    JNIEnv *, jclass, jboolean enabled, jboolean) {
    Renderer::options.noHandAmbient = enabled;
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetEntityNormalsEnabled(
    JNIEnv *, jclass, jboolean enabled, jboolean) {
    Renderer::options.entityNormalsEnabled = (enabled == JNI_TRUE);
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetOfflineState(
    JNIEnv *, jclass, jint state, jboolean) {
    Renderer::options.offlineState = static_cast<uint32_t>(std::clamp(state, 0, 2));
    if (state == 2) {
        Renderer::accumFrameCount = 0;  // reset on entering accumulation
        Renderer::dlssEpochFrame = 0;
        Renderer::dlssEpochCount = 0;
    }
    // Accumulation always forces native resolution
    if (state == 2 && !Renderer::options.offlineNativeResActive) {
        Renderer::options.savedUpscalerMode = Renderer::options.upscalerMode;
        Renderer::options.upscalerMode = 3;  // DLAA = 1:1
        Renderer::options.needRecreate = true;
        Renderer::options.offlineNativeResActive = true;
    }
    // FREE: native res if N key toggled on
    if (state == 1 && Renderer::options.offlineNativeRes && !Renderer::options.offlineNativeResActive) {
        Renderer::options.savedUpscalerMode = Renderer::options.upscalerMode;
        Renderer::options.upscalerMode = 3;  // DLAA = 1:1
        Renderer::options.needRecreate = true;
        Renderer::options.offlineNativeResActive = true;
    }
    // Restore upscaler on return to NORMAL
    if (state == 0 && Renderer::options.offlineNativeResActive) {
        Renderer::options.upscalerMode = Renderer::options.savedUpscalerMode;
        Renderer::options.needRecreate = true;
        Renderer::options.offlineNativeResActive = false;
    }
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetOfflineBounces(
    JNIEnv *, jclass, jint bounces, jboolean) {
    Renderer::options.offlineBounces = static_cast<uint32_t>(std::clamp(bounces, 1, 128));
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetOfflineDisableRR(
    JNIEnv *, jclass, jboolean disable, jboolean) {
    Renderer::options.offlineDisableRR = disable;
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetOfflineDisableClamp(
    JNIEnv *, jclass, jboolean disable, jboolean) {
    Renderer::options.offlineDisableClamp = disable;
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetOfflineAperture(
    JNIEnv *, jclass, jfloat aperture, jboolean) {
    Renderer::options.offlineAperture = std::clamp(aperture, 0.0f, 2.0f);
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetOfflineFocalDistance(
    JNIEnv *, jclass, jfloat dist, jboolean) {
    Renderer::options.offlineFocalDistance = std::clamp(dist, 0.5f, 256.0f);
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetDofStrength(
    JNIEnv *, jclass, jfloat strength, jboolean) {
    Renderer::options.dofStrength = std::clamp(strength, 1.0f, 20.0f);
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetOfflineNativeRes(
    JNIEnv *, jclass, jboolean enabled, jboolean) {
    Renderer::options.offlineNativeRes = enabled;
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetOfflineDenoised(
    JNIEnv *, jclass, jint mode, jboolean) {
    // 0=Raw Fast (RR on), 1=Raw Accurate (RR off), 2=Denoised (epoch-based DLSS-RR)
    Renderer::options.offlineDenoised = static_cast<uint32_t>(std::clamp(mode, 0, 2));
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeResetAccumulation(
    JNIEnv *, jclass) {
    Renderer::accumFrameCount = 0;
    Renderer::dlssEpochFrame = 0;
    Renderer::dlssEpochCount = 0;
}

extern "C" JNIEXPORT jint JNICALL Java_com_radiance_client_option_Options_nativeGetAccumFrameCount(
    JNIEnv *, jclass) {
    return static_cast<jint>(Renderer::accumFrameCount);
}

extern "C" JNIEXPORT jint JNICALL Java_com_radiance_client_option_Options_nativeGetDlssEpochCount(
    JNIEnv *, jclass) {
    return static_cast<jint>(Renderer::dlssEpochCount);
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetDlssEpochLength(
    JNIEnv *, jclass, jint length, jboolean) {
    Renderer::options.dlssEpochLength = static_cast<uint32_t>(std::clamp(length, 4, 64));
}

// ═══════════ Frame Generation (DLSS-G) ═══════════

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetFrameGenMode(
    JNIEnv *, jclass, jint mode, jboolean) {
    Renderer::options.frameGenMode = static_cast<uint32_t>(std::clamp(mode, 0, 2));
    Renderer::options.frameGenEnabled = (mode != 0);
    // Trigger swapchain recreation — FrameGenManager::setMode() will be called
    // on the render thread during recreate (via beforeSwapchainRecreate/afterSwapchainRecreate).
    // Don't call slDLSSGSetOptions from the Java thread — it's not thread-safe.
    Renderer::options.needRecreate = true;
    // Recalculate Reflex frame limit (accounts for FG multiplier)
    Renderer::options.reflexDirty = true;
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetFrameGenMultiplier(
    JNIEnv *, jclass, jint multiplier, jboolean) {
    Renderer::options.frameGenMultiplier = static_cast<uint32_t>(std::clamp(multiplier, 1, 5));
    // Changing multiplier while active requires re-calling slDLSSGSetOptions.
    // Trigger recreation to apply on render thread.
    if (Renderer::options.frameGenEnabled) {
        Renderer::options.needRecreate = true;
    }
    // Recalculate Reflex frame limit (accounts for FG multiplier)
    Renderer::options.reflexDirty = true;
}

extern "C" JNIEXPORT jboolean JNICALL Java_com_radiance_client_option_Options_nativeIsFrameGenSupported(
    JNIEnv *, jclass) {
    return FrameGenManager::maxFramesToGenerate() > 0 ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jint JNICALL Java_com_radiance_client_option_Options_nativeGetFrameGenMaxMultiplier(
    JNIEnv *, jclass) {
    return static_cast<jint>(FrameGenManager::maxFramesToGenerate());
}

// --- Volumetric cloud setters ---

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetCloudQuality(
    JNIEnv *, jclass, jint quality, jboolean write) {
    Renderer::options.cloudQuality = static_cast<uint32_t>(std::clamp(quality, 0, 6));
    if (write) Renderer::options.needRecreate = true;
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetCloudDensity(
    JNIEnv *, jclass, jfloat density, jboolean) {
    Renderer::options.cloudDensity = std::clamp(density, 0.1f, 3.0f);
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetCloudCoverage(
    JNIEnv *, jclass, jfloat coverage, jboolean) {
    Renderer::options.cloudCoverage = std::clamp(coverage, 0.0f, 1.0f);
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetCloudType(
    JNIEnv *, jclass, jfloat type, jboolean) {
    Renderer::options.cloudType = std::clamp(type, 0.0f, 1.0f);
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetCloudSpeed(
    JNIEnv *, jclass, jfloat speed, jboolean) {
    Renderer::options.cloudSpeed = std::clamp(speed, 0.0f, 6.0f);
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetCloudAltitude(
    JNIEnv *, jclass, jfloat altitude, jboolean) {
    Renderer::options.cloudAltitude = std::clamp(altitude, 64.0f, 320.0f);
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetCloudThicknessVol(
    JNIEnv *, jclass, jfloat thickness, jboolean) {
    Renderer::options.cloudThickness = std::clamp(thickness, 16.0f, 256.0f);
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetCloudDetailStrength(
    JNIEnv *, jclass, jfloat strength, jboolean) {
    Renderer::options.cloudDetailStrength = std::clamp(strength, 0.0f, 2.0f);
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetCloudScatterOctaves(
    JNIEnv *, jclass, jint octaves, jboolean) {
    Renderer::options.cloudScatterOctaves = static_cast<uint32_t>(std::clamp(octaves, 1, 8));
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetCloudAmbientStrength(
    JNIEnv *, jclass, jfloat strength, jboolean) {
    Renderer::options.cloudAmbientStrength = std::clamp(strength, 0.0f, 2.0f);
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetCloudTemporalBlend(
    JNIEnv *, jclass, jfloat blend, jboolean) {
    Renderer::options.cloudTemporalBlend = blend < 0.0f ? -1.0f : std::clamp(blend, 0.8f, 0.99f);
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetCloudNoiseScale(
    JNIEnv *, jclass, jfloat scale, jboolean) {
    Renderer::options.cloudNoiseScale = std::clamp(scale, 16.0f, 4096.0f);
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetCloudCellFrequency(
    JNIEnv *, jclass, jfloat freq, jboolean) {
    Renderer::options.cloudCellFrequency = std::clamp(freq, 1.0f, 32.0f);
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetCloudAtmosphereFadeDist(
    JNIEnv *, jclass, jfloat dist, jboolean) {
    Renderer::options.cloudAtmosphereFadeDist = std::clamp(dist, 100.0f, 4000.0f);
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetCloudDebugMode(
    JNIEnv *, jclass, jint mode, jboolean) {
    Renderer::options.cloudDebugMode = std::clamp(mode, 0, 8);
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetCloudWindAngle(
    JNIEnv *, jclass, jfloat angle, jboolean) {
    Renderer::options.cloudWindAngle = std::fmod(angle, 6.2831855f);
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetCloudMarchSteps(
    JNIEnv *, jclass, jint steps, jboolean) {
    Renderer::options.cloudMarchStepsOverride = std::clamp(steps, 0, 512);
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetCloudLightSteps(
    JNIEnv *, jclass, jint steps, jboolean) {
    Renderer::options.cloudLightStepsOverride = std::clamp(steps, 0, 16);
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetCloudResDivisor(
    JNIEnv *, jclass, jint div, jboolean needRecreate) {
    Renderer::options.cloudResDivisorOverride = std::clamp(div, 0, 4);
    if (needRecreate && div > 0) Renderer::options.needRecreate = true;
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetCloudNoiseRes(
    JNIEnv *, jclass, jint res, jboolean) {
    // Allowed: 128, 256, 512. Triggers pipeline recreation to reallocate 3D texture.
    uint32_t clamped = static_cast<uint32_t>(std::clamp(res, 128, 512));
    // Snap to nearest power of 2
    if (clamped <= 192) clamped = 128;
    else if (clamped <= 384) clamped = 256;
    else clamped = 512;
    Renderer::options.cloudNoiseRes = clamped;
    Renderer::options.needRecreate = true;
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetWetSurfaceStrength(
    JNIEnv *, jclass, jfloat strength, jboolean) {
    Renderer::options.wetSurfaceStrength = std::clamp(strength, 0.0f, 2.0f);
}

