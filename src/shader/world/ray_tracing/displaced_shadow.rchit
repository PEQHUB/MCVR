#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : require

#include "../util/ray_payloads.glsl"

// Shadow closest hit for displaced blocks.
// Currently unused — displaced_block.rchit handles both primary and shadow rays.
// Kept as a stub for future two-hit-group shadow dispatch.

layout(location = 1) rayPayloadInEXT ShadowRay shadowRay;

void main() {
    shadowRay.hitT = gl_HitTEXT;
}
