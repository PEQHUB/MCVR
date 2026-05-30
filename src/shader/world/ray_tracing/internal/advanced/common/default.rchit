#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : require

#include "common/shared.hpp"
#include "util/ray.glsl"

layout(location = 0) rayPayloadInEXT MainRay mainRay;

void main() {
    mainRay.hitT = gl_HitTEXT;
    raySetStop(mainRay, true);
}
