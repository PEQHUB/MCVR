#include "com_radiance_client_proxy_vulkan_TextureProxy.h"

#include "core/render/renderer.hpp"
#include "core/render/textures.hpp"

extern "C" {
JNIEXPORT jint JNICALL Java_com_radiance_client_proxy_vulkan_TextureProxy_generateTextureId(JNIEnv *, jclass) {
    if (!Renderer::is_initialized()) return 0;
    auto textures = Renderer::instance().textures();
    if (textures == nullptr)
        return 0;
    else
        return textures->allocateTexture();
}

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_TextureProxy_prepareImage(
    JNIEnv *, jclass, jint id, jint maxLevel, jint width, jint height, jint format) {
    if (!Renderer::is_initialized()) return;
    auto textures = Renderer::instance().textures();
    if (textures == nullptr) return;
    auto vkFormat = static_cast<VkFormat>(format);
    textures->initializeTexture(id, maxLevel, width, height, vkFormat);
}

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_TextureProxy_setFilter(
    JNIEnv *, jclass, jint id, jint samplingMode, jint mipmapMode) {
    if (!Renderer::is_initialized()) return;
    auto textures = Renderer::instance().textures();
    if (textures == nullptr) return;
    auto vkSamplingMode = static_cast<VkFilter>(samplingMode);
    auto vkMipmapMode = static_cast<VkSamplerMipmapMode>(mipmapMode);
    textures->setSamplingMode(id, vkSamplingMode, vkMipmapMode);
}

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_TextureProxy_setClamp(JNIEnv *,
                                                                                   jclass,
                                                                                   jint id,
                                                                                   jint addressMode) {
    if (!Renderer::is_initialized()) return;
    auto textures = Renderer::instance().textures();
    if (textures == nullptr) return;
    auto vkSamplerAddressMode = static_cast<VkSamplerAddressMode>(addressMode);
    textures->setAddressMode(id, vkSamplerAddressMode);
}

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_TextureProxy_queueUpload(JNIEnv *,
                                                                                      jclass,
                                                                                      jlong srcPointer,
                                                                                      jint srcSizeInBytes,
                                                                                      jint srcRowPixels,
                                                                                      jint dstId,
                                                                                      jint srcOffsetX,
                                                                                      jint srcOffsetY,
                                                                                      jint dstOffsetX,
                                                                                      jint dstOffsetY,
                                                                                      jint width,
                                                                                      jint height,
                                                                                      jint level) {
    if (!Renderer::is_initialized()) return;
    auto textures = Renderer::instance().textures();
    if (textures == nullptr) return;
    textures->queueUpload(reinterpret_cast<uint8_t *>(srcPointer), srcSizeInBytes, srcRowPixels, dstId, srcOffsetX,
                          srcOffsetY, dstOffsetX, dstOffsetY, width, height, level);
}

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_TextureProxy_performQueuedUpload(JNIEnv *, jclass) {
    if (!Renderer::is_initialized()) return;
    auto textures = Renderer::instance().textures();
    if (textures == nullptr) return;
    textures->performQueuedUpload();
}

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_TextureProxy_setTextureAlphaClass(JNIEnv *,
                                                                                                jclass,
                                                                                                jint id,
                                                                                                jint alphaClass) {
    if (!Renderer::is_initialized()) return;
    auto textures = Renderer::instance().textures();
    if (textures == nullptr) return;
    textures->setTextureAlphaClass(id, static_cast<Textures::AlphaClass>(alphaClass));
}

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_TextureProxy_destroyTexture(JNIEnv *,
                                                                                          jclass,
                                                                                          jint id) {
    if (!Renderer::is_initialized()) return;
    auto textures = Renderer::instance().textures();
    if (textures == nullptr) return;
    textures->destroyTexture(static_cast<uint32_t>(id));
}
}