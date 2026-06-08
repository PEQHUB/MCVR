#include "com_radiance_client_proxy_vulkan_DrawCommandProxy_Overlay.h"

#include "core/render/modules/ui_module.hpp"
#include "core/render/pipeline.hpp"
#include "core/render/render_framework.hpp"
#include "core/render/renderer.hpp"

extern "C" {
JNIEXPORT void JNICALL
Java_com_radiance_client_proxy_vulkan_DrawCommandProxy_00024Overlay_vkCmdClearEntireColorAttachment(JNIEnv *, jclass) {
    auto framework = Renderer::instance().framework();
    if (framework == nullptr) return;
    auto context = framework->safeAcquireCurrentContext();
    auto pipelineContext = framework->pipeline()->acquirePipelineContext(context);
    pipelineContext->uiModuleContext->clearOverlayEntireColorAttachment();
}

JNIEXPORT void JNICALL
Java_com_radiance_client_proxy_vulkan_DrawCommandProxy_00024Overlay_vkCmdClearEntireDepthStencilAttachment(
    JNIEnv *, jclass, jint aspectMask) {
    auto framework = Renderer::instance().framework();
    if (framework == nullptr) return;
    auto context = framework->safeAcquireCurrentContext();
    auto pipelineContext = framework->pipeline()->acquirePipelineContext(context);
    pipelineContext->uiModuleContext->clearOverlayEntireDepthStencilAttachment(aspectMask);
}
}