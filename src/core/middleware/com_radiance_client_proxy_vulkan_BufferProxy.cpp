#include "com_radiance_client_proxy_vulkan_BufferProxy.h"

#include "core/render/buffers.hpp"
#include "core/render/render_framework.hpp"
#include "core/render/renderer.hpp"

JNIEXPORT jint JNICALL Java_com_radiance_client_proxy_vulkan_BufferProxy_allocateBuffer(JNIEnv *, jclass) {
    auto buffers = Renderer::instance().buffers();
    if (buffers == nullptr)
        return 0;
    else
        return buffers->allocateBuffer();
}

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_BufferProxy_initializeBuffer(
    JNIEnv *, jclass, jint id, jint size, jint usageFlags) {
    auto buffers = Renderer::instance().buffers();
    if (buffers == nullptr) return;
    buffers->initializeBuffer(id, size, usageFlags);
}

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_BufferProxy_buildIndexBuffer(
    JNIEnv *, jclass, jint dstId, jint type, jint drawMode, jint vertexCount, jint expectedIndexCount) {
    auto buffers = Renderer::instance().buffers();
    if (buffers == nullptr) return;
    buffers->buildIndexBuffer(dstId, type, drawMode, vertexCount, expectedIndexCount);
}

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_BufferProxy_queueUpload(JNIEnv *env,
                                                                                     jclass,
                                                                                     jlong ptr,
                                                                                     jint dstId) {
    auto buffers = Renderer::instance().buffers();
    if (buffers == nullptr) return;
    buffers->queueOverlayUpload(reinterpret_cast<uint8_t *>(ptr), dstId);
}

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_BufferProxy_performQueuedUpload(JNIEnv *, jclass) {
    auto buffers = Renderer::instance().buffers();
    if (buffers == nullptr) return;
    buffers->performQueuedUpload();
}

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_BufferProxy_updateOverlayDrawUniform(JNIEnv *,
                                                                                                  jclass,
                                                                                                  jlong ptr) {
    auto buffers = Renderer::instance().buffers();
    if (buffers == nullptr) return;
    vk::Data::OverlayUBO *ubo = reinterpret_cast<vk::Data::OverlayUBO *>(ptr);
    buffers->appendOverlayDrawUniform(*ubo);
}

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_BufferProxy_updateOverlayPostUniform(JNIEnv *,
                                                                                                  jclass,
                                                                                                  jlong ptr) {
    auto buffers = Renderer::instance().buffers();
    if (buffers == nullptr) return;
    vk::Data::OverlayPostUBO *ubo = reinterpret_cast<vk::Data::OverlayPostUBO *>(ptr);

    auto extent = Renderer::instance().framework()->swapchain()->vkExtent();
    ubo->inSize = {extent.width, extent.height};
    ubo->outSize = {extent.width, extent.height};

    ubo->blurDir = {1.0, 0.0};
    buffers->appendOverlayPostUniform(*ubo);

    ubo->blurDir = {0.0, 1.0};
    buffers->appendOverlayPostUniform(*ubo);

    ubo->blurDir = {1.0, 0.0};
    ubo->radiusMultiplier = 0.5;
    buffers->appendOverlayPostUniform(*ubo);

    ubo->blurDir = {0.0, 1.0};
    ubo->radiusMultiplier = 0.5;
    buffers->appendOverlayPostUniform(*ubo);

    ubo->blurDir = {1.0, 0.0};
    ubo->radiusMultiplier = 0.25;
    buffers->appendOverlayPostUniform(*ubo);

    ubo->blurDir = {0.0, 1.0};
    ubo->radiusMultiplier = 0.25;
    buffers->appendOverlayPostUniform(*ubo);
}

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_BufferProxy_updateWorldUniform(JNIEnv *,
                                                                                            jclass,
                                                                                            jlong ptr) {
    auto buffers = Renderer::instance().buffers();
    if (buffers == nullptr) return;
    vk::Data::WorldUBO *ubo = reinterpret_cast<vk::Data::WorldUBO *>(ptr);
    buffers->setAndUploadWorldUniformBuffer(*ubo);
}

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_BufferProxy_updateSkyUniform(JNIEnv *, jclass, jlong ptr) {
    auto buffers = Renderer::instance().buffers();
    if (buffers == nullptr) return;
    vk::Data::SkyUBO *ubo = reinterpret_cast<vk::Data::SkyUBO *>(ptr);
    buffers->setAndUploadSkyUniformBuffer(*ubo);
}

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_BufferProxy_updateMapping(JNIEnv *, jclass, jlong ptr) {
    auto buffers = Renderer::instance().buffers();
    if (buffers == nullptr) return;
    vk::Data::TextureMapping *mapping = reinterpret_cast<vk::Data::TextureMapping *>(ptr);
    buffers->setAndUploadTextureMappingBuffer(*mapping);
}

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_BufferProxy_updateLightMapUniform(JNIEnv *,
                                                                                               jclass,
                                                                                               jlong ptr) {
    auto buffers = Renderer::instance().buffers();
    if (buffers == nullptr) return;
    vk::Data::LightMapUBO *lightMapUBO = reinterpret_cast<vk::Data::LightMapUBO *>(ptr);
    buffers->setAndUploadLightMapUniformBuffer(*lightMapUBO);
}