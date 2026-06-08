#version 460
#extension GL_GOOGLE_include_directive : require

#include "common/shared.hpp"

layout(set = 1, binding = 0) readonly buffer Storage {
    OverlayUBO ubos[];
};

layout(push_constant) uniform Push {
    uint drawId;
};

layout(location = 0) in vec3 Position;

layout(location = 0) out vec4 vertexColor;

void main() {
    OverlayUBO ubo = ubos[drawId];

    gl_Position = ubo.projectionMat * ubo.modelViewMat * vec4(Position, 1.0);
}
