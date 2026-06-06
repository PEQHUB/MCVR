#version 460
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_GOOGLE_include_directive : require

#include "common/shared.hpp"

layout(set = 0, binding = 0) uniform sampler2D textures[];

layout(set = 1, binding = 0) uniform WorldUniform {
    WorldUBO worldUBO;
};

layout(location = 0) in vec3 pos;
layout(location = 1) flat in uint useNorm;
layout(location = 2) in vec3 norm;
layout(location = 3) flat in uint useColorLayer;
layout(location = 4) in vec4 colorLayer;
layout(location = 5) flat in uint useTexture;
layout(location = 6) flat in uint useOverlay;
layout(location = 7) in vec2 textureUV;
layout(location = 8) flat in ivec2 overlayUV;
layout(location = 9) flat in uint useGlint;
layout(location = 10) flat in uint textureID;
layout(location = 11) in vec2 glintUV;
layout(location = 12) flat in uint glintTexture;
layout(location = 13) flat in uint useLight;
layout(location = 14) flat in ivec2 lightUV;
layout(location = 15) in vec4 lightMapColor;
layout(location = 16) in vec4 overlayColor;
layout(location = 17) flat in uint postTextMode;

layout(location = 0) out vec4 fragColor;
layout(location = 1) out float outLinearDepth;

void main() {
    vec4 color = vec4(1.0);
    if (postTextMode == 1u || postTextMode == 4u) {
        color = vec4(1.0);
    } else if (postTextMode == 2u || postTextMode == 5u || postTextMode == 7u) {
        color = texture(textures[nonuniformEXT(textureID)], textureUV).rrrr;
    } else if (postTextMode == 3u || postTextMode == 6u || postTextMode == 8u) {
        color = texture(textures[nonuniformEXT(textureID)], textureUV);
    } else if (useTexture > 0) {
        color = texture(textures[nonuniformEXT(textureID)], textureUV);
    }

    if (useColorLayer > 0) { color *= colorLayer; }
    if (useOverlay > 0) { color.rgb = mix(overlayColor.rgb, color.rgb, overlayColor.a); }
    if (color.a < 0.1) { discard; }

    if (useLight == 0)
        fragColor = color;
    else
        fragColor = color * lightMapColor;

    float linearDepth = -(mat4(mat3(worldUBO.cameraEffectedViewMat)) * vec4(pos, 1.0)).z;
    float fragDepth = clamp(linearDepth / 1000.0, 0.0, 1.0);
    if (postTextMode == 7u || postTextMode == 8u) {
        fragDepth = max(fragDepth - 1e-4, 0.0);
    }
    gl_FragDepth = fragDepth;
    outLinearDepth = max(linearDepth, 0.0);
}
