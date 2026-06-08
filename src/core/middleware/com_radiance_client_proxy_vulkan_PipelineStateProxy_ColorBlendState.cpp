#include "com_radiance_client_proxy_vulkan_PipelineStateProxy_ColorBlendState.h"

#include "core/render/modules/ui_module.hpp"
#include "core/render/pipeline.hpp"
#include "core/render/render_framework.hpp"
#include "core/render/renderer.hpp"

extern "C" {

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_PipelineStateProxy_00024ColorBlendState_setBlendEnable(
    JNIEnv *, jclass, jboolean enable) {
    auto framework = Renderer::instance().framework();
    if (framework == nullptr) return;
    auto context = framework->safeAcquireCurrentContext();
    auto pipelineContext = framework->pipeline()->acquirePipelineContext(context);
    pipelineContext->uiModuleContext->setOverlayBlendEnable(enable);
}

JNIEXPORT void JNICALL
Java_com_radiance_client_proxy_vulkan_PipelineStateProxy_00024ColorBlendState_setColorBlendConstants(
    JNIEnv *, jclass, jfloat const1, jfloat const2, jfloat const3, jfloat const4) {
    auto framework = Renderer::instance().framework();
    if (framework == nullptr) return;
    auto context = framework->safeAcquireCurrentContext();
    auto pipelineContext = framework->pipeline()->acquirePipelineContext(context);
    pipelineContext->uiModuleContext->setOverlayColorBlendConstants(const1, const2, const3, const4);
}

JNIEXPORT void JNICALL
Java_com_radiance_client_proxy_vulkan_PipelineStateProxy_00024ColorBlendState_setColorLogicOpEnable(JNIEnv *,
                                                                                                    jclass,
                                                                                                    jboolean enable) {
    auto framework = Renderer::instance().framework();
    if (framework == nullptr) return;
    auto context = framework->safeAcquireCurrentContext();
    auto pipelineContext = framework->pipeline()->acquirePipelineContext(context);
    pipelineContext->uiModuleContext->setOverlayColorLogicOpEnable(enable);
}

JNIEXPORT void JNICALL
Java_com_radiance_client_proxy_vulkan_PipelineStateProxy_00024ColorBlendState_vkSetBlendFuncSeparate(
    JNIEnv *,
    jclass,
    jint srcColorBlendFactor,
    jint srcAlphaBlendFactor,
    jint dstColorBlendFactor,
    jint dstAlphaBlendFactor) {
    auto framework = Renderer::instance().framework();
    if (framework == nullptr) return;
    auto context = framework->safeAcquireCurrentContext();
    auto pipelineContext = framework->pipeline()->acquirePipelineContext(context);
    pipelineContext->uiModuleContext->setOverlayBlendFuncSeparate(srcColorBlendFactor, srcAlphaBlendFactor,
                                                                  dstColorBlendFactor, dstAlphaBlendFactor);
}

JNIEXPORT void JNICALL
Java_com_radiance_client_proxy_vulkan_PipelineStateProxy_00024ColorBlendState_vkSetBlendOpSeparate(JNIEnv *,
                                                                                                   jclass,
                                                                                                   jint colorBlendOp,
                                                                                                   jint alphaBlendOp) {
    auto framework = Renderer::instance().framework();
    if (framework == nullptr) return;
    auto context = framework->safeAcquireCurrentContext();
    auto pipelineContext = framework->pipeline()->acquirePipelineContext(context);
    pipelineContext->uiModuleContext->setOverlayBlendOpSeparate(colorBlendOp, alphaBlendOp);
}

JNIEXPORT void JNICALL
Java_com_radiance_client_proxy_vulkan_PipelineStateProxy_00024ColorBlendState_vkSetColorWriteMask(JNIEnv *,
                                                                                                  jclass,
                                                                                                  jint colorWriteMask) {
    auto framework = Renderer::instance().framework();
    if (framework == nullptr) return;
    auto context = framework->safeAcquireCurrentContext();
    auto pipelineContext = framework->pipeline()->acquirePipelineContext(context);
    pipelineContext->uiModuleContext->setOverlayColorWriteMask(colorWriteMask);
}

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_PipelineStateProxy_00024ColorBlendState_vkSetColorLogicOp(
    JNIEnv *, jclass, jint colorLogicOp) {
    auto framework = Renderer::instance().framework();
    if (framework == nullptr) return;
    auto context = framework->safeAcquireCurrentContext();
    auto pipelineContext = framework->pipeline()->acquirePipelineContext(context);
    pipelineContext->uiModuleContext->setOverlayColorLogicOp(colorLogicOp);
}
}