#version 460
#extension GL_GOOGLE_include_directive : require

#include "common/shared.hpp"

layout(location = 0) out vec2 texCoord;

void main() {
    vec2 positions[3] = vec2[](vec2(-1.0, -1.0), vec2(-1.0, 3.0), vec2(3.0, -1.0));

    vec2 texCoords[3] = vec2[](vec2(0.0, 0.0), vec2(0.0, 2.0), vec2(2.0, 0.0));

    gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
    texCoord = texCoords[gl_VertexIndex].xy;
}
