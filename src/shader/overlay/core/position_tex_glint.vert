#version 460
#extension GL_GOOGLE_include_directive : require

#include "common/shared.hpp"

layout(location = 0) in vec3 Position;
layout(location = 1) in vec2 UV0;

layout(set = 1, binding = 0) readonly buffer Storage {
    OverlayUBO ubos[];
};

layout(push_constant) uniform Push {
    uint drawId;
};

layout(location = 0) out vec2 texCoord0;

void main() {
    OverlayUBO ubo = ubos[drawId];

    gl_Position = ubo.projectionMat * ubo.modelViewMat * vec4(Position, 1.0);

    texCoord0 = (ubo.textureMat * vec4(UV0, 0.0, 1.0)).xy;
}
