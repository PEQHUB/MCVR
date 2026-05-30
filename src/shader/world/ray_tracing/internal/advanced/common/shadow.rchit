#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : require

#include "util/ray.glsl"

layout(location = 1) rayPayloadInEXT ShadowRay shadowRay;

void main() {}
