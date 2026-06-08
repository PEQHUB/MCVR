#include "com_radiance_client_proxy_vulkan_PipelineStateProxy_ViewportState.h"

#include "core/render/modules/ui_module.hpp"
#include "core/render/pipeline.hpp"
#include "core/render/render_framework.hpp"
#include "core/render/renderer.hpp"

extern "C" {
JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_PipelineStateProxy_00024ViewportState_setScissorEnabled(
    JNIEnv *, jclass, jboolean enabled) {
    auto framework = Renderer::instance().framework();
    if (framework == nullptr) return;
    auto context = framework->safeAcquireCurrentContext();
    auto pipelineContext = framework->pipeline()->acquirePipelineContext(context);
    pipelineContext->uiModuleContext->setOverlayScissorEnabled(enabled);
}

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_PipelineStateProxy_00024ViewportState_setScissor(
    JNIEnv *, jclass, jint x, jint y, jint width, jint height) {
    auto framework = Renderer::instance().framework();
    if (framework == nullptr) return;
    auto context = framework->safeAcquireCurrentContext();
    auto pipelineContext = framework->pipeline()->acquirePipelineContext(context);
    pipelineContext->uiModuleContext->setOverlayScissor(x, y, width, height);
}

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_PipelineStateProxy_00024ViewportState_setViewport(
    JNIEnv *, jclass, jint x, jint y, jint width, jint height) {
    auto framework = Renderer::instance().framework();
    if (framework == nullptr) return;
    auto context = framework->safeAcquireCurrentContext();
    auto pipelineContext = framework->pipeline()->acquirePipelineContext(context);
    pipelineContext->uiModuleContext->setOverlayViewport(x, y, width, height);
}
}