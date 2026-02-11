#include "com_radiance_client_option_Options.h"

#include "core/all_extern.hpp"
#include "core/render/buffers.hpp"
#include "core/render/chunks.hpp"
#include "core/render/render_framework.hpp"
#include "core/render/renderer.hpp"
#include "core/render/textures.hpp"
#include "core/render/world.hpp"

JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetMaxFps(JNIEnv *,
                                                                               jclass,
                                                                               jint maxFps,
                                                                               jboolean write) {
    Renderer::options.maxFps = maxFps;
}

JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetInactivityFpsLimit(JNIEnv *,
                                                                                           jclass,
                                                                                           jint inactivityFpsLimit,
                                                                                           jboolean write) {
    Renderer::options.inactivityFpsLimit = inactivityFpsLimit;
}

JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetVsync(JNIEnv *,
                                                                              jclass,
                                                                              jboolean vsync,
                                                                              jboolean write) {
    Renderer::options.vsync = vsync;
    if (write) Renderer::options.needRecreate = true;
}

JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetChunkBuildingBatchSize(
    JNIEnv *, jclass, jint chunkBuildingBatchSize, jboolean write) {
    Renderer::options.chunkBuildingBatchSize = chunkBuildingBatchSize;
    if (write) Renderer::instance().world()->chunks()->resetScheduler();
}

JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetChunkBuildingTotalBatches(
    JNIEnv *, jclass, jint chunkBuildingTotalBatches, jboolean write) {
    Renderer::options.chunkBuildingTotalBatches = chunkBuildingTotalBatches;
    if (write) Renderer::instance().world()->chunks()->resetScheduler();
}

JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetTonemappingMode(
    JNIEnv *, jclass, jint mode, jboolean write) {
    Renderer::options.tonemappingMode = mode;
}

JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetRayBounces(
    JNIEnv *, jclass, jint bounces, jboolean write) {
    Renderer::options.rayBounces = bounces;
}

JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetDlssQuality(
    JNIEnv *, jclass, jint quality, jboolean write) {
    Renderer::options.upscalerMode = quality;
    if (write) Renderer::options.needRecreate = true;
}

JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetDlssResOverride(
    JNIEnv *, jclass, jint resOverride, jboolean write) {
    Renderer::options.upscalerResOverride = resOverride;
    if (write) Renderer::options.needRecreate = true;
}

JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetMinExposure(
    JNIEnv *, jclass, jfloat minExposure, jboolean write) {
    Renderer::options.minExposure = minExposure;
}

JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetMaxExposure(
    JNIEnv *, jclass, jint maxExposure, jboolean write) {
    Renderer::options.maxExposure = static_cast<float>(maxExposure);
}

JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetExposureCompensation(
    JNIEnv *, jclass, jfloat ec, jboolean write) {
    Renderer::options.exposureCompensation = ec;
}

JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetMiddleGrey(
    JNIEnv *, jclass, jfloat mg, jboolean write) {
    Renderer::options.middleGrey = mg;
}

JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetLwhite(
    JNIEnv *, jclass, jfloat lw, jboolean write) {
    Renderer::options.Lwhite = lw;
}

JNIEXPORT void JNICALL Java_com_radiance_client_option_Options_nativeSetDlssPreset(
    JNIEnv *, jclass, jint preset, jboolean write) {
    // Clamp to valid DLSS RR preset range (A=0 through G=6)
    Renderer::options.upscalerPreset = static_cast<uint32_t>(std::clamp(preset, 0, 6));
    if (write) Renderer::options.needRecreate = true;
}
