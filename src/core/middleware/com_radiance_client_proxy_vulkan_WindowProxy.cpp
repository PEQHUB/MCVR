#include "com_radiance_client_proxy_vulkan_WindowProxy.h"

#include "core/all_extern.hpp"
#include "core/vulkan/window.hpp"

JNIEXPORT void JNICALL Java_com_radiance_client_proxy_vulkan_WindowProxy_onFramebufferSizeChanged(JNIEnv *,
                                                                                                            jclass) {
    vk::Window::framebufferResized = true;
}