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
