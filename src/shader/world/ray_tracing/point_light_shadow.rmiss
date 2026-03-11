#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : require

#include "../util/ray_payloads.glsl"

layout(location = 1) rayPayloadInEXT ShadowRay shadowRay;

void main() {
    // Shadow ray reached the light without hitting any geometry.
    // Mark as fully unoccluded.
    shadowRay.throughput = vec3(1.0);
}
