#version 460
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_GOOGLE_include_directive : require

#include "common/shared.hpp"

#ifndef VPT_POST_STAR_CLOUD_TRANSMITTANCE_BINDING
#define VPT_POST_STAR_CLOUD_TRANSMITTANCE_BINDING 31
#endif

layout(set = 1, binding = 0) uniform WorldUniform {
    WorldUBO worldUBO;
};

layout(set = 1, binding = 1) uniform SkyUniform {
    SkyUBO skyUBO;
};

layout(set = 4, binding = VPT_POST_STAR_CLOUD_TRANSMITTANCE_BINDING) uniform sampler2D postStarCloudTransmittance;

layout(location = 0) in vec3 pos;
layout(location = 1) in vec4 colorLayer;
layout(location = 2) in vec2 screenUv;

layout(location = 0) out vec4 fragColor;

void main() {
    if (worldUBO.skyType != 1) { discard; }

    float progress = skyUBO.rainGradient;
    vec4 color = colorLayer;
    color.a *= 1.0 - progress;
    vec2 transmittanceUv = clamp(screenUv, vec2(0.0), vec2(1.0));
    float cloudTransmittance = clamp(texture(postStarCloudTransmittance, transmittanceUv).r, 0.0, 1.0);
    if (cloudTransmittance < 0.99) { discard; }

    fragColor = color;
    gl_FragDepth = 0.999999;
}
