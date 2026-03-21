#include "com_radiance_client_proxy_vulkan_UIThreadProxy.h"

#include "core/render/buffers.hpp"
#include "core/render/render_framework.hpp"
#include "core/render/renderer.hpp"
#include "core/render/ui_render_context.hpp"

namespace {

/// Get the UIRenderContext if active. Returns nullptr if unavailable.
inline UIRenderContext *getUICtx() {
    auto framework = Renderer::instance().framework();
    if (!framework) return nullptr;
    return framework->uiRenderContext();
}

} // namespace

// ── Lifecycle ────────────────────────────────────────────────────────────

extern "C" JNIEXPORT jboolean JNICALL
Java_com_radiance_client_proxy_vulkan_UIThreadProxy_createUIRenderContext(JNIEnv *, jclass) {
    auto framework = Renderer::instance().framework();
    if (!framework) return JNI_FALSE;
    return framework->createUIRenderContext() ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT void JNICALL
Java_com_radiance_client_proxy_vulkan_UIThreadProxy_destroyUIRenderContext(JNIEnv *, jclass) {
    auto framework = Renderer::instance().framework();
    if (framework) framework->destroyUIRenderContext();
}

// ── Frame lifecycle ──────────────────────────────────────────────────────

extern "C" JNIEXPORT void JNICALL
Java_com_radiance_client_proxy_vulkan_UIThreadProxy_beginUIFrame(JNIEnv *, jclass) {
    auto *ctx = getUICtx();
    if (ctx) ctx->beginFrame();
}

extern "C" JNIEXPORT void JNICALL
Java_com_radiance_client_proxy_vulkan_UIThreadProxy_endUIFrame(JNIEnv *, jclass) {
    auto *ctx = getUICtx();
    if (ctx) ctx->endFrame();
}

extern "C" JNIEXPORT void JNICALL
Java_com_radiance_client_proxy_vulkan_UIThreadProxy_submitUIFrame(JNIEnv *, jclass) {
    auto *ctx = getUICtx();
    if (ctx) ctx->submitAndPresent();
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_radiance_client_proxy_vulkan_UIThreadProxy_isDecoupledUIActive(JNIEnv *, jclass) {
    auto framework = Renderer::instance().framework();
    return framework ? framework->isDecoupledUIActive() : JNI_FALSE;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_radiance_client_proxy_vulkan_UIThreadProxy_isUIThreadRenderingOverlay(JNIEnv *, jclass) {
    auto framework = Renderer::instance().framework();
    return framework ? framework->isUIThreadRenderingOverlay() : JNI_FALSE;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_radiance_client_proxy_vulkan_UIThreadProxy_isPauseRequested(JNIEnv *, jclass) {
    auto *ctx = getUICtx();
    if (!ctx) return JNI_FALSE;
    return ctx->isPauseRequested() ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_radiance_client_proxy_vulkan_UIThreadProxy_checkPause(JNIEnv *, jclass) {
    auto *ctx = getUICtx();
    if (!ctx) return JNI_TRUE;  // no context — not a stop signal, just inactive
    return ctx->checkPause() ? JNI_TRUE : JNI_FALSE;
}

// ── Draw commands ────────────────────────────────────────────────────────

extern "C" JNIEXPORT void JNICALL
Java_com_radiance_client_proxy_vulkan_UIThreadProxy_drawUIOverlay(
    JNIEnv *, jclass, jint vertexId, jint indexId, jint pipelineType,
    jint indexCount, jint indexType) {
    auto *ctx = getUICtx();
    if (!ctx) return;
    auto vertexBuffer = Renderer::instance().buffers()->getBuffer(vertexId);
    auto indexBuffer = Renderer::instance().buffers()->getBuffer(indexId);
    ctx->drawIndexed(vertexBuffer, indexBuffer,
                     static_cast<OverlayDrawPipelineType>(pipelineType),
                     indexCount,
                     static_cast<VkIndexType>(indexType));
}

// ── Clear commands ───────────────────────────────────────────────────────

extern "C" JNIEXPORT void JNICALL
Java_com_radiance_client_proxy_vulkan_UIThreadProxy_clearUIColor(JNIEnv *, jclass) {
    // Clear is handled by the render pass load op in beginFrame().
    // This is a no-op since we clear on every beginFrame.
}

extern "C" JNIEXPORT void JNICALL
Java_com_radiance_client_proxy_vulkan_UIThreadProxy_clearUIDepthStencil(
    JNIEnv *, jclass, jint /*aspectMask*/) {
    // Depth/stencil clear handled by render pass load op.
}

// ── Viewport / Scissor ──────────────────────────────────────────────────

extern "C" JNIEXPORT void JNICALL
Java_com_radiance_client_proxy_vulkan_UIThreadProxy_setUIViewport(
    JNIEnv *, jclass, jint x, jint y, jint width, jint height) {
    auto *ctx = getUICtx();
    if (ctx) ctx->setViewport(x, y, width, height);
}

extern "C" JNIEXPORT void JNICALL
Java_com_radiance_client_proxy_vulkan_UIThreadProxy_setUIScissor(
    JNIEnv *, jclass, jint x, jint y, jint width, jint height) {
    auto *ctx = getUICtx();
    if (ctx) ctx->setScissor(x, y, width, height);
}

extern "C" JNIEXPORT void JNICALL
Java_com_radiance_client_proxy_vulkan_UIThreadProxy_setUIScissorEnabled(
    JNIEnv *, jclass, jboolean enabled) {
    auto *ctx = getUICtx();
    if (ctx) ctx->setScissorEnabled(enabled);
}

// ── Color blend state ────────────────────────────────────────────────────

extern "C" JNIEXPORT void JNICALL
Java_com_radiance_client_proxy_vulkan_UIThreadProxy_setUIBlendEnable(
    JNIEnv *, jclass, jboolean enable) {
    auto *ctx = getUICtx();
    if (ctx) ctx->setBlendEnable(enable);
}

extern "C" JNIEXPORT void JNICALL
Java_com_radiance_client_proxy_vulkan_UIThreadProxy_setUIBlendFuncSeparate(
    JNIEnv *, jclass, jint srcColor, jint srcAlpha, jint dstColor, jint dstAlpha) {
    auto *ctx = getUICtx();
    if (ctx) ctx->setBlendFuncSeparate(srcColor, srcAlpha, dstColor, dstAlpha);
}

extern "C" JNIEXPORT void JNICALL
Java_com_radiance_client_proxy_vulkan_UIThreadProxy_setUIBlendOpSeparate(
    JNIEnv *, jclass, jint colorOp, jint alphaOp) {
    auto *ctx = getUICtx();
    if (ctx) ctx->setBlendOpSeparate(colorOp, alphaOp);
}

extern "C" JNIEXPORT void JNICALL
Java_com_radiance_client_proxy_vulkan_UIThreadProxy_setUIColorWriteMask(
    JNIEnv *, jclass, jint mask) {
    auto *ctx = getUICtx();
    if (ctx) ctx->setColorWriteMask(mask);
}

extern "C" JNIEXPORT void JNICALL
Java_com_radiance_client_proxy_vulkan_UIThreadProxy_setUIColorLogicOpEnable(
    JNIEnv *, jclass, jboolean enable) {
    auto *ctx = getUICtx();
    if (ctx) ctx->setColorLogicOpEnable(enable);
}

extern "C" JNIEXPORT void JNICALL
Java_com_radiance_client_proxy_vulkan_UIThreadProxy_setUIColorLogicOp(
    JNIEnv *, jclass, jint op) {
    auto *ctx = getUICtx();
    if (ctx) ctx->setColorLogicOp(op);
}

extern "C" JNIEXPORT void JNICALL
Java_com_radiance_client_proxy_vulkan_UIThreadProxy_setUIBlendConstants(
    JNIEnv *, jclass, jfloat c0, jfloat c1, jfloat c2, jfloat c3) {
    auto *ctx = getUICtx();
    if (ctx) ctx->setBlendConstants(c0, c1, c2, c3);
}

// ── Depth / Stencil state ────────────────────────────────────────────────

extern "C" JNIEXPORT void JNICALL
Java_com_radiance_client_proxy_vulkan_UIThreadProxy_setUIDepthTestEnable(
    JNIEnv *, jclass, jboolean enable) {
    auto *ctx = getUICtx();
    if (ctx) ctx->setDepthTestEnable(enable);
}

extern "C" JNIEXPORT void JNICALL
Java_com_radiance_client_proxy_vulkan_UIThreadProxy_setUIDepthWriteEnable(
    JNIEnv *, jclass, jboolean enable) {
    auto *ctx = getUICtx();
    if (ctx) ctx->setDepthWriteEnable(enable);
}

extern "C" JNIEXPORT void JNICALL
Java_com_radiance_client_proxy_vulkan_UIThreadProxy_setUIDepthCompareOp(
    JNIEnv *, jclass, jint op) {
    auto *ctx = getUICtx();
    if (ctx) ctx->setDepthCompareOp(op);
}

extern "C" JNIEXPORT void JNICALL
Java_com_radiance_client_proxy_vulkan_UIThreadProxy_setUIStencilTestEnable(
    JNIEnv *, jclass, jboolean enable) {
    auto *ctx = getUICtx();
    if (ctx) ctx->setStencilTestEnable(enable);
}

extern "C" JNIEXPORT void JNICALL
Java_com_radiance_client_proxy_vulkan_UIThreadProxy_setUIStencilFrontFunc(
    JNIEnv *, jclass, jint compareOp, jint reference, jint compareMask) {
    auto *ctx = getUICtx();
    if (ctx) ctx->setStencilFrontFunc(compareOp, reference, compareMask);
}

extern "C" JNIEXPORT void JNICALL
Java_com_radiance_client_proxy_vulkan_UIThreadProxy_setUIStencilBackFunc(
    JNIEnv *, jclass, jint compareOp, jint reference, jint compareMask) {
    auto *ctx = getUICtx();
    if (ctx) ctx->setStencilBackFunc(compareOp, reference, compareMask);
}

extern "C" JNIEXPORT void JNICALL
Java_com_radiance_client_proxy_vulkan_UIThreadProxy_setUIStencilFrontOp(
    JNIEnv *, jclass, jint failOp, jint depthFailOp, jint passOp) {
    auto *ctx = getUICtx();
    if (ctx) ctx->setStencilFrontOp(failOp, depthFailOp, passOp);
}

extern "C" JNIEXPORT void JNICALL
Java_com_radiance_client_proxy_vulkan_UIThreadProxy_setUIStencilBackOp(
    JNIEnv *, jclass, jint failOp, jint depthFailOp, jint passOp) {
    auto *ctx = getUICtx();
    if (ctx) ctx->setStencilBackOp(failOp, depthFailOp, passOp);
}

extern "C" JNIEXPORT void JNICALL
Java_com_radiance_client_proxy_vulkan_UIThreadProxy_setUIStencilFrontWriteMask(
    JNIEnv *, jclass, jint writeMask) {
    auto *ctx = getUICtx();
    if (ctx) ctx->setStencilFrontWriteMask(writeMask);
}

extern "C" JNIEXPORT void JNICALL
Java_com_radiance_client_proxy_vulkan_UIThreadProxy_setUIStencilBackWriteMask(
    JNIEnv *, jclass, jint writeMask) {
    auto *ctx = getUICtx();
    if (ctx) ctx->setStencilBackWriteMask(writeMask);
}

// ── Rasterization state ──────────────────────────────────────────────────

extern "C" JNIEXPORT void JNICALL
Java_com_radiance_client_proxy_vulkan_UIThreadProxy_setUILineWidth(
    JNIEnv *, jclass, jfloat lineWidth) {
    auto *ctx = getUICtx();
    if (ctx) ctx->setLineWidth(lineWidth);
}

extern "C" JNIEXPORT void JNICALL
Java_com_radiance_client_proxy_vulkan_UIThreadProxy_setUICullMode(
    JNIEnv *, jclass, jint cullMode) {
    auto *ctx = getUICtx();
    if (ctx) ctx->setCullMode(cullMode);
}

extern "C" JNIEXPORT void JNICALL
Java_com_radiance_client_proxy_vulkan_UIThreadProxy_setUIFrontFace(
    JNIEnv *, jclass, jint frontFace) {
    auto *ctx = getUICtx();
    if (ctx) ctx->setFrontFace(frontFace);
}

extern "C" JNIEXPORT void JNICALL
Java_com_radiance_client_proxy_vulkan_UIThreadProxy_setUIPolygonMode(
    JNIEnv *, jclass, jint polygonMode) {
    auto *ctx = getUICtx();
    if (ctx) ctx->setPolygonMode(polygonMode);
}

extern "C" JNIEXPORT void JNICALL
Java_com_radiance_client_proxy_vulkan_UIThreadProxy_setUIDepthBiasEnable(
    JNIEnv *, jclass, jint polygonMode, jboolean enable) {
    auto *ctx = getUICtx();
    if (ctx) ctx->setDepthBiasEnable(polygonMode, enable);
}

extern "C" JNIEXPORT void JNICALL
Java_com_radiance_client_proxy_vulkan_UIThreadProxy_setUIDepthBias(
    JNIEnv *, jclass, jfloat slopeFactor, jfloat constantFactor) {
    auto *ctx = getUICtx();
    if (ctx) ctx->setDepthBias(slopeFactor, constantFactor);
}

// ── Clear state ──────────────────────────────────────────────────────────

extern "C" JNIEXPORT void JNICALL
Java_com_radiance_client_proxy_vulkan_UIThreadProxy_setUIClearColor(
    JNIEnv *, jclass, jfloat r, jfloat g, jfloat b, jfloat a) {
    auto *ctx = getUICtx();
    if (ctx) ctx->clearColor(r, g, b, a);
}

// ── Sync ─────────────────────────────────────────────────────────────────

extern "C" JNIEXPORT void JNICALL
Java_com_radiance_client_proxy_vulkan_UIThreadProxy_syncUIDynamicState(JNIEnv *, jclass) {
    auto *ctx = getUICtx();
    if (ctx) ctx->syncDynamicState();
}
