#version 460

// Star field vertex shader (V2 engine).
// Stars are submitted as billboard quads in a vertex buffer.
// StarVertex: vec3 pos (location 0) + vec4 color (location 1).
//
// This shader rotates the star sphere so the exclusion zone tracks
// the sun direction, then transforms by viewProj.
// Passes color, visibility, rain gradient, and sky type to fragment.

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec4 inColor;

layout(location = 0) out vec4 outColor;
layout(location = 1) out float outRainGradient;
layout(location = 2) flat out uint outSkyType;

layout(push_constant) uniform PushConstants {
    mat4 viewProj;
    vec3 sunDirection;
    float starVisibility;  // 0 = daytime (hidden), 1 = nighttime (fully visible)
    float rainGradient;    // 0 = clear, 1 = full rain
    uint skyType;          // 1 = overworld
} pc;

// Rodrigues rotation: rotate so that +X aligns with sunDir.
// The star field's exclusion zone (no stars near +X) tracks the sun.
mat3 rotationFromXToDir(vec3 dir) {
    vec3 from = vec3(1.0, 0.0, 0.0);
    vec3 axis = cross(from, dir);
    float cosA = dot(from, dir);
    float sinA = length(axis);

    if (sinA < 1e-6) {
        return (cosA > 0.0) ? mat3(1.0) : mat3(-1.0, 0.0, 0.0,
                                                  0.0, 1.0, 0.0,
                                                  0.0, 0.0, -1.0);
    }
    axis = normalize(axis);
    float k = 1.0 - cosA;
    return mat3(
        cosA + axis.x * axis.x * k,         axis.x * axis.y * k + axis.z * sinA, axis.x * axis.z * k - axis.y * sinA,
        axis.y * axis.x * k - axis.z * sinA, cosA + axis.y * axis.y * k,         axis.y * axis.z * k + axis.x * sinA,
        axis.z * axis.x * k + axis.y * sinA, axis.z * axis.y * k - axis.x * sinA, cosA + axis.z * axis.z * k
    );
}

void main() {
    mat3 sunRot = rotationFromXToDir(pc.sunDirection);
    vec3 rotatedPos = sunRot * inPosition;

    gl_Position = pc.viewProj * vec4(rotatedPos, 1.0);

    outColor = inColor * pc.starVisibility;
    outRainGradient = pc.rainGradient;
    outSkyType = pc.skyType;
}
