#version 460
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_GOOGLE_include_directive : require

#include "common/shared.hpp"

layout(set = 0, binding = 0) uniform sampler2D textures[];
layout(set = 1, binding = 0) readonly buffer Storage {
    OverlayUBO ubos[];
};

layout(push_constant) uniform Push {
    uint drawId;
};

layout(location = 0) in vec2 texCoord0;
layout(location = 1) in vec4 vertexColor;

layout(location = 0) out vec4 fragColor;

void main() {
    OverlayUBO ubo = ubos[drawId];

    vec4 color = texture(textures[nonuniformEXT(ubo.texIndices[0])], texCoord0) * vertexColor * ubo.colorModulator;
    if (color.a < 0.1) { discard; }
    fragColor = color;
}
