#version 460
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_GOOGLE_include_directive : require

#include "common/shared.hpp"

layout(set = 0, binding = 0) uniform sampler2D textures[];

layout(set = 0, binding = 1) uniform sampler2D lightMap;

layout(set = 1, binding = 0) uniform WorldUniform {
    WorldUBO worldUBO;
};

// Packed PBRTriangle vertex attributes (96 bytes, 12 locations)
layout(location = 0) in vec3 inPos;
layout(location = 1) in uint inFlags;
layout(location = 2) in vec3 inNorm;
layout(location = 3) in float inAlbedoEmission;
layout(location = 4) in vec4 inColorLayer;
layout(location = 5) in vec4 inPostBaseAndEmissive; // postBase.xyz + emissiveBlockType as float bits
layout(location = 6) in vec2 inTextureUV;
layout(location = 7) in vec2 inGlintUV;
layout(location = 8) in uint inTextureID;
layout(location = 9) in uint inGlintTexture;
layout(location = 10) in uint inOverlayPacked;
layout(location = 11) in uint inLightPacked;

layout(location = 0) out vec3 outPos;
layout(location = 1) flat out uint outFlags;
layout(location = 2) out vec3 outNorm;
layout(location = 3) out vec4 outColorLayer;
layout(location = 4) out vec2 outTextureUV;
layout(location = 5) flat out uint outTextureID;
layout(location = 6) out vec2 outGlintUV;
layout(location = 7) flat out uint outGlintTexture;
layout(location = 8) out vec4 lightMapColor;
layout(location = 9) out vec4 overlayColor;

void main() {
    // Unpack flags
    uint coordinate = (inFlags >> PBR_FLAG_COORD_SHIFT) & 0x7u;
    bool useLight   = (inFlags & PBR_FLAG_USE_LIGHT) != 0u;
    bool useOverlay = (inFlags & PBR_FLAG_USE_OVERLAY) != 0u;

    vec3 pos = inPos + inPostBaseAndEmissive.xyz;
    if (coordinate == 0u) {
        pos = pos - worldUBO.cameraViewMatInv[3].xyz;
    } else if (coordinate == 1u) {
        pos = mat3(worldUBO.cameraViewMatInv) * pos;
    } else if (coordinate == 2u) {
        pos = pos;
    }
    outPos = pos;
    outFlags = inFlags;
    if (coordinate == 0u || coordinate == 2u) {
        outNorm = inNorm;
    } else if (coordinate == 1u) {
        outNorm = normalize(mat3(worldUBO.cameraViewMatInv) * inNorm);
    }
    outColorLayer = inColorLayer;
    outTextureUV = inTextureUV;
    outTextureID = inTextureID;
    outGlintUV = inGlintUV;
    outGlintTexture = inGlintTexture;

    gl_Position = worldUBO.cameraProjMat * worldUBO.cameraEffectedViewMat * vec4(pos, 1.0);

    if (useLight) {
        ivec2 lightUV = ivec2(int(inLightPacked & 0xFFFFu), int(inLightPacked >> 16u));
        lightMapColor = texelFetch(lightMap, lightUV / 16, 0);
    } else {
        lightMapColor = vec4(0.0);
    }
    if (useOverlay) {
        ivec2 overlayUV = ivec2(int(inOverlayPacked & 0xFFFFu), int(inOverlayPacked >> 16u));
        overlayColor = texelFetch(textures[nonuniformEXT(worldUBO.overlayTextureID)], overlayUV, 0);
    } else {
        overlayColor = vec4(0.0);
    }
}
