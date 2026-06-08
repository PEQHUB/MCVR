#version 460
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_GOOGLE_include_directive : require

#include "common/shared.hpp"

layout(set = 1, binding = 0) readonly buffer Storage {
    OverlayUBO ubos[];
};

layout(push_constant) uniform Push {
    uint drawId;
};

layout(location = 0) in vec4 vertexColor;

layout(location = 0) out vec4 fragColor;

vec3 srgbToLinear(vec3 c) { return pow(c, vec3(2.2)); }

void main() {
    OverlayUBO ubo = ubos[drawId];

    vec4 color = vec4(srgbToLinear(vertexColor.rgb), vertexColor.a);
    if (color.a == 0.0) { discard; }
    fragColor = color * vec4(srgbToLinear(ubo.colorModulator.rgb), ubo.colorModulator.a);
}
