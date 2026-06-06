#version 460
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_GOOGLE_include_directive : require

#include "common/shared.hpp"

layout(set = 1, binding = 0) uniform WorldUniform {
    WorldUBO worldUBO;
};

layout(set = 1, binding = 1) uniform SkyUniform {
    SkyUBO skyUBO;
};

layout(location = 0) in vec3 inPos;
layout(location = 4) in vec4 inColorLayer;

layout(location = 0) out vec3 outPos;
layout(location = 1) out vec4 outColorLayer;
layout(location = 2) out vec2 outScreenUv;

void main() {
    vec3 sourceDir = vec3(1.0, 0.0, 0.0);
    vec3 targetDir = normalize(skyUBO.sunDirection);
    float rotationCos = clamp(dot(sourceDir, targetDir), -1.0, 1.0);
    mat3 rotationMatrix = mat3(1.0);
    if (rotationCos < -0.9999) {
        rotationMatrix = mat3(-1, 0, 0, 0, 1, 0, 0, 0, -1);
    } else if (rotationCos <= 0.9999) {
        vec3 rotationAxis = normalize(cross(sourceDir, targetDir));
        float rotationSin = length(cross(sourceDir, targetDir));
        mat3 skewMatrix =
            mat3(0.0, rotationAxis.z, -rotationAxis.y, -rotationAxis.z, 0.0, rotationAxis.x, rotationAxis.y,
                 -rotationAxis.x, 0.0);
        rotationMatrix = mat3(1.0) + skewMatrix * rotationSin + (skewMatrix * skewMatrix) * (1.0 - rotationCos);
    }

    vec3 pos = rotationMatrix * inPos;
    outPos = pos;
    vec4 clipPos = worldUBO.cameraProjMat * worldUBO.cameraEffectedViewMat * vec4(pos, 1.0);
    gl_Position = clipPos;
    outScreenUv = clipPos.xy / max(clipPos.w, 1e-4) * 0.5 + 0.5;

    float vis = smoothstep(0.0, -0.2, skyUBO.sunDirection.y);
    outColorLayer = vec4(inColorLayer.rgb * vis, vis);
}
