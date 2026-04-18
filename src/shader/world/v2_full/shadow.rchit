#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : require

#include "../util/ray_payloads.glsl"

layout(location = 1) rayPayloadInEXT ShadowRay shadowRay;
hitAttributeEXT vec2 attribs;

void main() {
    shadowRay.hitT = gl_HitTEXT;
}