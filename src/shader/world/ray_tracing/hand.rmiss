#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : require

#include "common/shared.hpp"
#include "../util/ray_payloads.glsl"
#include "../util/util.glsl"

layout(set = 2, binding = 0) uniform Uniform {
    WorldUBO worldUBO;
};

layout(location = 0) rayPayloadInEXT PrimaryRay mainRay;

void main() {
    mainRay.hitT = INF_DISTANCE;
    mainRay.stop = 1;
}