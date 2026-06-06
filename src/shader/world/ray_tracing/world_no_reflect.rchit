#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#include "../util/disney.glsl"
#include "../util/random.glsl"
#include "../util/ray_cone.glsl"
#include "../util/ray_payloads.glsl"
#include "../util/util.glsl"
#include "common/shared.hpp"
#include "../util/colorspace.glsl"
#include "../util/texture_rules.glsl"

layout(set = 0, binding = 0) uniform sampler2D textures[];

#include "../util/sprite_fetch.glsl"

layout(set = 1, binding = 0) uniform accelerationStructureEXT topLevelAS;

layout(set = 1, binding = 1) readonly buffer BLASOffsets {
    uint offsets[];
}
blasOffsets;

layout(set = 1, binding = 2) readonly buffer VertexBufferAddr {
    uint64_t addrs[];
}
vertexBufferAddrs;

layout(set = 1, binding = 3) readonly buffer IndexBufferAddr {
    uint64_t addrs[];
}
indexBufferAddrs;

layout(set = 1, binding = 4) readonly buffer LastVertexBufferAddr {
    uint64_t addrs[];
}
lastVertexBufferAddrs;

layout(set = 1, binding = 5) readonly buffer LastIndexBufferAddr {
    uint64_t addrs[];
}
lastIndexBufferAddrs;

layout(set = 1, binding = 6) readonly buffer LastObjToWorldMat {
    mat4 mat[];
}
lastObjToWorldMats;

layout(set = 1, binding = 7) readonly buffer TextureMappingBuffer {
    TextureMapping mapping;
};

#include "../util/vertex_fetch.glsl"

layout(set = 2, binding = 0) uniform WorldUniform {
    WorldUBO worldUbo;
};

layout(set = 2, binding = 1) uniform LastWorldUniform {
    WorldUBO lastWorldUbo;
};

layout(set = 2, binding = 2) uniform SkyUniform {
    SkyUBO skyUBO;
};

layout(set = 3, binding = 1, rgba8) uniform image2D diffuseAlbedoImage;
layout(set = 3, binding = 2, rgba8) uniform image2D specularAlbedoImage;
layout(set = 3, binding = 3, rgba32f) uniform image2D normalRoughnessImage;
layout(set = 3, binding = 4, rg32f) uniform image2D motionVectorImage;
layout(set = 3, binding = 5, r32f) uniform image2D linearDepthImage;

// VertexBuffer and IndexBuffer declared in vertex_fetch.glsl

layout(location = 0) rayPayloadInEXT PrimaryRay mainRay;
hitAttributeEXT vec2 attribs;

void main() {
    vec3 viewDir = -mainRay.direction;

    uint instanceID = gl_InstanceCustomIndexEXT;
    uint geometryID = gl_GeometryIndexEXT;

    uint blasOffset;
    uint vertexFormat;
    UnpackedVertex v0, v1, v2;
    fetchTriangleVertices(instanceID, geometryID, gl_PrimitiveID, v0, v1, v2, blasOffset, vertexFormat);

    vec3 baryCoords = vec3(1.0 - (attribs.x + attribs.y), attribs.x, attribs.y);
    vec3 worldPos = gl_WorldRayOriginEXT + gl_HitTEXT * gl_WorldRayDirectionEXT;
    uint coordinate = (v0.flags >> PBR_FLAG_COORD_SHIFT) & 0x7u;
    // Compute geometric normal from triangle edges (norm field not stored in compact/lossless)
    vec3 edge1 = v1.pos - v0.pos;
    vec3 edge2 = v2.pos - v0.pos;
    vec3 normal = normalize(cross(edge1, edge2));
    bool isBlockGeometry = (v0.flags & PBR_FLAG_BLOCK_GEOMETRY) != 0u;
    if (coordinate == 1 && !isBlockGeometry) {
        normal = normalize(mat3(worldUbo.cameraViewMatInv) * normal);
    }

    bool useColorLayer = (v0.flags & PBR_FLAG_USE_COLOR_LAYER) != 0u;
    vec3 colorLayer;
    if (useColorLayer) {
        colorLayer = baryCoords.x * v0.colorLayer + baryCoords.y * v1.colorLayer + baryCoords.z * v2.colorLayer;
    } else {
        colorLayer = vec3(1.0);
    }

    bool useTexture = (v0.flags & PBR_FLAG_USE_TEXTURE) != 0u;
    float albedoEmission =
        baryCoords.x * v0.albedoEmission + baryCoords.y * v1.albedoEmission + baryCoords.z * v2.albedoEmission;
    uint textureID = v0.textureID;
    uint materialRuleTextureID = textureID;
    vec4 albedoValue;
    vec4 specularValue;
    vec4 normalValue;
    ivec4 flagValue;
    vec2 textureUV;
    bool foldBlockOverlay = (v0.flags & PBR_FLAG_OVERLAY_ALPHA_MASK) != 0u;
    bool blockOverlayComposited = false;

    if (useTexture) {
        textureUV = baryCoords.x * v0.textureUV + baryCoords.y * v1.textureUV + baryCoords.z * v2.textureUV;

        if (isBlockGeometry) {
            // === BLOCK GEOMETRY: texture array sampling via SpriteRegistry ===
            uint spriteId = textureID;

            float lod = 0;
            albedoValue = fetchBlockAlbedoLod(spriteId, textureUV, worldUbo.animTick, lod);
            specularValue = fetchBlockSpecularLod(spriteId, textureUV, lod);
            normalValue = fetchBlockNormalLod(spriteId, textureUV, lod);
            flagValue = fetchBlockFlagLod(spriteId, textureUV, lod);
            if (foldBlockOverlay) {
                blockOverlayComposited = applyBlockOverlayMaterialLod(
                    albedoValue, specularValue, normalValue, flagValue,
                    spriteId, textureUV, worldUbo.animTick, lod, colorLayer,
                    materialRuleTextureID);
            }
        } else {
            // === ENTITY GEOMETRY: legacy atlas sampling via textures[] ===
            int specularTextureID = mapping.entries[textureID].specular;
            int normalTextureID = mapping.entries[textureID].normal;
            int flagTextureID = mapping.entries[textureID].flag;

            float lod = 0;
            albedoValue = textureLod(textures[nonuniformEXT(textureID)], textureUV, lod);
            if (specularTextureID >= 0) {
                specularValue = textureLod(textures[nonuniformEXT(specularTextureID)], textureUV, lod);
            } else {
                specularValue = vec4(0.0);
            }
            if (normalTextureID >= 0) {
                normalValue = textureLod(textures[nonuniformEXT(normalTextureID)], textureUV, lod);
            } else {
                normalValue = vec4(0.0);
            }
            if (flagTextureID >= 0) {
                vec4 floatFlagValue = textureLod(textures[nonuniformEXT(flagTextureID)], textureUV, ceil(lod));
                flagValue = ivec4(round(floatFlagValue * 255.0));
            } else {
                flagValue = ivec4(0);
            }
        }
    } else {
        albedoValue = vec4(1.0);
        specularValue = vec4(0.0);
        normalValue = vec4(0.0);
        flagValue = ivec4(0);
    }

    vec3 glint = vec3(0.0);
    vec4 overlayColor = vec4(0.0);
    if (!isBlockGeometry) {
        // Entity glint and overlay (blocks don't use these)
        float useGlint = float((v0.flags & PBR_FLAG_USE_GLINT) != 0u);
        uint glintTexture = v0.glintTexture;
        vec2 glintUV = baryCoords.x * v0.glintUV + baryCoords.y * v1.glintUV + baryCoords.z * v2.glintUV;
        glintUV = (worldUbo.textureMat * vec4(glintUV, 0.0, 1.0)).xy;
        glint = useGlint * texture(textures[nonuniformEXT(glintTexture)], glintUV).rgb;
        glint = glint * glint;

        bool useOverlay = (v0.flags & PBR_FLAG_USE_OVERLAY) != 0u;
        if (useOverlay) {
            ivec2 overlayUV = ivec2(int(v0.overlayPacked & 0xFFFFu), int(v0.overlayPacked >> 16u));
            overlayColor = texelFetch(textures[nonuniformEXT(worldUbo.overlayTextureID)], overlayUV, 0);
        }
    }

    vec3 tint;
    if ((v0.flags & PBR_FLAG_USE_OVERLAY) != 0u && !isBlockGeometry) {
        tint = mix(overlayColor.rgb, albedoValue.rgb * colorLayer, overlayColor.a) + glint;
    } else if (isBlockGeometry && blockOverlayComposited) {
        tint = albedoValue.rgb + glint;
    } else {
        tint = albedoValue.rgb * colorLayer + glint;
    }

    tint = CS_BT709_TO_BT2020 * tint;  // BT.709 -> BT.2020 working space
    albedoValue = vec4(tint, albedoValue.a);
    LabPBRMat mat = convertLabPBRMaterial(albedoValue, specularValue, normalValue);
    if (isBlockGeometry) {
        applyTextureRule(materialRuleTextureID, mat);
    }

    // add glowing radiance
    float combinedEmission = max(mat.emission, albedoEmission);
    mainRay.radiance += 12 * tint * combinedEmission * mainRay.throughput;
    mainRay.hitT = gl_HitTEXT;

    mainRay.instanceIndex = instanceID;
    mainRay.geometryIndex = geometryID;
    mainRay.primitiveIndex = gl_PrimitiveID;
    mainRay.baryCoords = baryCoords;
    mainRay.worldPos = worldPos;
    mainRay.normal = vec3(0);
    mainRay.albedoValue = albedoValue;
    // Clear noisy/lobeType, set stop=1 (preserve isHand/insideBoat/emBlockType)
    mainRay.flags = (mainRay.flags & (PR_ISHAND_BIT | PR_INSIDEBOAT_BIT)) | PR_STOP_BIT | (prGetEmBlockType(mainRay) << PR_EMBLOCK_SHIFT);
}
