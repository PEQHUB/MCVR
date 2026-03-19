#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : require

#include "../util/ray_payloads.glsl"

layout(location = 1) rayPayloadInEXT ShadowRay shadowRay;
hitAttributeEXT vec2 faceUV;

void main() {
    // The intersection shader already confirmed a hit exists via DDA.
    // For opaque displaced blocks, accept the shadow hit.
    shadowRay.hitT = gl_HitTEXT;
}
