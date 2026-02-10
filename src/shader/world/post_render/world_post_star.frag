#version 460
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_GOOGLE_include_directive : require

#include "common/shared.hpp"

layout(set = 0, binding = 0) uniform sampler2D textures[];

layout(set = 1, binding = 0) uniform WorldUniform {
    WorldUBO worldUBO;
};

layout(set = 1, binding = 1) uniform SkyUniform {
    SkyUBO skyUBO;
};

layout(set = 2, binding = 0) readonly buffer TextureMappingBuffer {
    TextureMapping mapping;
};

layout(location = 0) in vec3 pos;
layout(location = 1) in vec4 colorLayer;

layout(location = 0) out vec4 fragColor;

void main() {
    if (worldUBO.skyType != 1) { discard; }

    float progress = skyUBO.rainGradient;
    vec4 color = colorLayer;
    color.a *= 1.0 - progress;

    fragColor = color;

    float linearDepth = 0.95;
    gl_FragDepth = linearDepth;
}
