#include "com_radiance_client_proxy_world_PlayerProxy.h"

#include "core/render/chunks.hpp"
#include "core/render/renderer.hpp"

JNIEXPORT void JNICALL
Java_com_radiance_client_proxy_world_PlayerProxy_setCameraPos(JNIEnv *, jclass, jdouble x, jdouble y, jdouble z) {
    auto world = Renderer::instance().world();
    if (world == nullptr) return;
    Renderer::instance().world()->setCameraPos(glm::dvec3{x, y, z});
}