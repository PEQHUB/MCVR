#version 460
#extension GL_GOOGLE_include_directive : require

#include "common/shared.hpp"

layout(set = 1, binding = 1) readonly buffer Storage {
    OverlayPostUBO ubos[];
};

layout(push_constant) uniform Push {
    uint postId;
};

layout(location = 0) out vec2 texCoord;
layout(location = 1) out vec2 sampleStep;

void main() {
    // 通过顶点索引直接生成全屏三角形的三个顶点
    vec2 positions[3] = vec2[](vec2(-1.0, -1.0), // 左上角
                               vec2(-1.0, 3.0),  // 左下角外
                               vec2(3.0, -1.0)   // 右上角外
    );

    // 对应的纹理坐标
    vec2 texCoords[3] = vec2[](vec2(0.0, 0.0), // 左上角
                               vec2(0.0, 2.0), // 左下角外
                               vec2(2.0, 0.0)  // 右上角外
    );

    gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);

    vec2 oneTexel = 1.0 / ubos[postId].inSize;
    sampleStep = oneTexel * ubos[postId].blurDir;

    texCoord = texCoords[gl_VertexIndex].xy;
}
