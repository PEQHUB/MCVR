#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#include "util/disney.glsl"
#include "util/alpha_mode.glsl"
#include "util/height_map.glsl"
#include "util/random.glsl"
#include "util/ray_cone.glsl"
#include "util/ray.glsl"
#include "util/sampling_helpers.glsl"
#include "util/util.glsl"
#include "common/shared.hpp"

layout(set = 0, binding = 0) uniform sampler2D textures[];

#include "util/texture_rules.glsl"
#include "util/sprite_fetch.glsl"

layout(set = 1, binding = 0) uniform accelerationStructureEXT topLevelAS;

layout(set = 1, binding = 1) readonly buffer BLASOffsets {
    uint offsets[];
}
blasOffsets;

layout(set = 1, binding = 6) readonly buffer LastObjToWorldMat {
    mat4 mat[];
}
lastObjToWorldMats;

layout(set = 1, binding = 7) readonly buffer TextureMappingBuffer {
    TextureMapping mapping;
};

layout(set = 2, binding = 0) uniform WorldUniform {
    WorldUBO worldUBO;
};

layout(set = 2, binding = 1) uniform LastWorldUniform {
    WorldUBO lastWorldUbo;
};

layout(set = 2, binding = 2) uniform SkyUniform {
    SkyUBO skyUBO;
};

layout(push_constant) uniform PushConstant {
    int numRayBounces;
    int flags;
    int reservedLighting0;
    float reservedLighting1;
    int reservedLighting2;
    int reservedLighting3;
    int reservedLighting4;
    float preExposure;
    float displacementDepthScale;
    int displacementPrimarySteps;
    int displacementRefinementSteps;
    float displacementFadeDistanceBlocks;
    uint blueNoiseFrame;
    uint rtDebugFlags;
    uint handInstanceCount;
    uint reservedPc0;
    uint64_t sharcHashEntries;
    uint64_t sharcAccumulation;
    uint64_t sharcResolved;
    float sharcCameraX;
    float sharcCameraY;
    float sharcCameraZ;
    float sharcSceneScale;
    uint sharcCapacity;
    float sharcRadianceScale;
    uint sharcFrameIndex;
    float sharcRoughnessThreshold;
    int sharcUpdateBlockSize;
    int sharcUpdateBounces;
    int sharcQueryMode;
    int sharcQueryReserved;
    int offlineFlags;
    int accumFrameCount;
    float aperture;
    float focalDistance;
    uint64_t reservedAddr;
} pc;

#define RT_DEBUG_MATERIAL_MODE_SHIFT 24u
#define RT_DEBUG_MATERIAL_MODE_MASK  15u
#define RT_DEBUG_MATERIAL_MODE       ((pc.rtDebugFlags >> RT_DEBUG_MATERIAL_MODE_SHIFT) & RT_DEBUG_MATERIAL_MODE_MASK)
#define RT_DEBUG_MATERIAL_ID         1u
#define RT_DEBUG_MATERIAL_PAGE_LAYER 2u
#define RT_DEBUG_MATERIAL_DISPLACE_ELIGIBLE 3u
#define RT_DEBUG_MATERIAL_DISPLACE_HEIGHT   4u

layout(set = 3, binding = 1, rgba8) uniform image2D diffuseAlbedoImage;
layout(set = 3, binding = 2, rgba8) uniform image2D specularAlbedoImage;
layout(set = 3, binding = 3, rgba32f) uniform image2D normalRoughnessImage;
layout(set = 3, binding = 4, rg32f) uniform image2D motionVectorImage;
layout(set = 3, binding = 5, r32f) uniform image2D linearDepthImage;

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer IndexBuffer {
    uint indices[];
}
indexBuffer;

#include "util/vertex.glsl"

layout(location = 0) rayPayloadInEXT MainRay mainRay;
hitAttributeEXT vec2 attribs;

vec3 materialDebugHashColor(uint materialId) {
    uint h = materialId * 1664525u + 1013904223u;
    h ^= h >> 16u;
    return vec3(
        float((h >> 0u) & 255u),
        float((h >> 8u) & 255u),
        float((h >> 16u) & 255u)) / 255.0;
}

vec3 materialDebugPageLayer(uint materialId) {
    MaterialEntry material = safeMaterialEntry(materialId);
    uint albedoPage = materialTexturePage(material.albedoPage);
    uint normalPage = materialTexturePage(material.normalPage);
    uint albedoLayer = uint(max(material.albedoLayer, 0));
    return vec3(
        float(albedoPage & 255u) / 255.0,
        float(albedoLayer & 255u) / 255.0,
        float(normalPage & 255u) / 255.0);
}

bool materialDebugHasAuthoredHeight(uint materialId) {
    MaterialEntry material = safeMaterialEntry(materialId);
    return (material.flags & MATERIAL_FLAG_DISPLACEMENT_ELIGIBLE) != 0u &&
           material.displacementPolicy == MATERIAL_DISPLACEMENT_AUTHORED_HEIGHT;
}

vec3 materialDebugDisplacementEligible(uint materialId) {
    return materialDebugHasAuthoredHeight(materialId) ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
}

vec3 materialDebugDisplacementHeight(uint materialId, vec4 normalValue) {
    if (!materialDebugHasAuthoredHeight(materialId)) return vec3(0.0);
    return vec3(clamp(normalValue.a, 0.0, 1.0));
}

vec3 materialDebugColor(uint materialId, vec4 normalValue) {
    uint mode = RT_DEBUG_MATERIAL_MODE;
    if (mode == RT_DEBUG_MATERIAL_ID) return materialDebugHashColor(materialId);
    if (mode == RT_DEBUG_MATERIAL_PAGE_LAYER) return materialDebugPageLayer(materialId);
    if (mode == RT_DEBUG_MATERIAL_DISPLACE_ELIGIBLE) return materialDebugDisplacementEligible(materialId);
    if (mode == RT_DEBUG_MATERIAL_DISPLACE_HEIGHT) return materialDebugDisplacementHeight(materialId, normalValue);
    return vec3(0.0);
}

void main() {
    vec3 viewDir = -mainRay.direction;

    uint instanceID = gl_InstanceCustomIndexEXT;
    uint geometryID = gl_GeometryIndexEXT;

    uint geometryBufferIndex = getGeometryBufferIndex(instanceID, geometryID);

    uint i0;
    uint i1;
    uint i2;
    PositionVertex p0;
    PositionVertex p1;
    PositionVertex p2;
    MaterialVertex m0;
    MaterialVertex m1;
    MaterialVertex m2;
    loadTriangle(geometryBufferIndex, gl_PrimitiveID, i0, i1, i2, p0, p1, p2, m0, m1, m2);

    vec3 baryCoords = vec3(1.0 - (attribs.x + attribs.y), attribs.x, attribs.y);
    vec3 worldPos = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
    uint coordinate = getCoordinate(m0.packedData);
    vec3 normal = baryCoords.x * m0.norm + baryCoords.y * m1.norm + baryCoords.z * m2.norm;
    if (coordinate == 1) {
        normal = normalize(mat3(worldUBO.cameraViewMatInv) * normal);
    } else {
        normal = normalize(normal);
    }

    bool useColorLayer = hasColorLayer(m0.packedData);
    vec4 colorLayerValue;
    if (useColorLayer) {
        colorLayerValue = baryCoords.x * m0.colorLayer + baryCoords.y * m1.colorLayer + baryCoords.z * m2.colorLayer;
    } else {
        colorLayerValue = vec4(1.0);
    }
    vec3 colorLayer = colorLayerValue.rgb;

    bool useTexture = hasTexture(m0.packedData);
    float albedoEmission =
        baryCoords.x * m0.albedoEmission + baryCoords.y * m1.albedoEmission + baryCoords.z * m2.albedoEmission;
    uint textureID = m0.textureID;
    uint materialRuleTextureID = textureID;
    bool isBlockGeometry = (m0.packedData & PBR_FLAG_BLOCK_GEOMETRY) != 0u;
    bool foldBlockOverlay = isBlockGeometry && ((m0.packedData & PBR_FLAG_OVERLAY_ALPHA_MASK) != 0u);
    uint alphaMode = getAlphaMode(m0.packedData);
    vec4 albedoValue;
    vec4 specularValue;
    vec4 normalValue;
    ivec4 flagValue = ivec4(0);
    vec2 textureUV = vec2(0.0);
    bool blockOverlayComposited = false;
    if (useTexture) {
        textureUV = baryCoords.x * m0.textureUV + baryCoords.y * m1.textureUV + baryCoords.z * m2.textureUV;
        vec2 atlasUvMin = isBlockGeometry ? vec2(0.0) : min(m0.textureUV, min(m1.textureUV, m2.textureUV));
        vec2 atlasUvMax = isBlockGeometry ? vec2(1.0) : max(m0.textureUV, max(m1.textureUV, m2.textureUV));

        // ray cone
        float coneRadiusWorld = mainRay.coneWidth + gl_HitTEXT * mainRay.coneSpread;
        vec3 dposdu, dposdv;
        computedposduDv(p0.pos, p1.pos, p2.pos, m0.textureUV, m1.textureUV, m2.textureUV, dposdu, dposdv);
        float lod = isBlockGeometry ? 0.0 : lodWithCone(textures[nonuniformEXT(textureID)], textureUV, coneRadiusWorld, dposdu, dposdv);

        if (isBlockGeometry) {
            albedoValue = fetchBlockAlbedoLod(textureID, textureUV, worldUBO.animTick, lod);
            albedoValue.a = resolveSurfaceAlpha(albedoValue.a * colorLayerValue.a, alphaMode);
            specularValue = fetchBlockSpecularLod(textureID, textureUV, lod);
            normalValue = fetchBlockNormalLod(textureID, textureUV, lod);
            flagValue = fetchBlockFlagLod(textureID, textureUV, lod);
            if (foldBlockOverlay) {
                blockOverlayComposited = applyBlockOverlayMaterialLod(
                    albedoValue, specularValue, normalValue, flagValue,
                    textureID, textureUV, worldUBO.animTick, lod, colorLayer,
                    materialRuleTextureID);
            }
        } else {
            int specularTextureID = mapping.entries[textureID].specular;
            int normalTextureID = mapping.entries[textureID].normal;
            albedoValue = sampleTexture(textures[nonuniformEXT(textureID)], textureUV, lod, false);
            albedoValue.a = resolveSurfaceAlpha(albedoValue.a * colorLayerValue.a, alphaMode);
            if (specularTextureID >= 0) {
                specularValue = sampleTexture(textures[nonuniformEXT(specularTextureID)], textureUV, lod, false);
            } else {
                specularValue = vec4(0.0);
            }
            if (normalTextureID >= 0) {
                normalValue = samplePBRTexture(textures[nonuniformEXT(normalTextureID)], textureUV, atlasUvMin, atlasUvMax,
                                               lod, VPT_PBR_SAMPLING_MODE);
            } else {
                normalValue = vec4(0.0);
            }
        }
    } else {
        albedoValue = vec4(1.0);
        specularValue = vec4(0.0);
        normalValue = vec4(0.0);
    }

    bool useGlint = hasGlint(m0.packedData);
    uint glintTexture = m0.glintTexture;
    vec2 glintUV = baryCoords.x * m0.glintUV + baryCoords.y * m1.glintUV + baryCoords.z * m2.glintUV;
    glintUV = (worldUBO.textureMat * vec4(glintUV, 0.0, 1.0)).xy;
    vec3 glint = useGlint ? sampleTexture(textures[nonuniformEXT(glintTexture)], glintUV, false).rgb : vec3(0.0);
    glint = glint * glint;

    bool useOverlay = hasOverlay(m0.packedData);
    vec3 tint = albedoValue.rgb * colorLayer + glint;
    if (isBlockGeometry && blockOverlayComposited) {
        tint = albedoValue.rgb + glint;
    } else if (useOverlay) {
        ivec2 overlayUV = m0.overlayUV;
        vec4 overlayColor = sampleTexture(textures[nonuniformEXT(worldUBO.overlayTextureID)], overlayUV, 0, false);
        tint = mix(overlayColor.rgb, albedoValue.rgb * colorLayer, overlayColor.a) + glint;
    }

    albedoValue = vec4(tint, albedoValue.a);
    if (isBlockGeometry && RT_DEBUG_MATERIAL_MODE != 0u) {
        albedoValue = vec4(materialDebugColor(materialRuleTextureID, normalValue), 1.0);
        tint = albedoValue.rgb;
    }
    LabPBRMat mat = convertLabPBRMaterial(albedoValue, specularValue, normalValue);
    if (isBlockGeometry) { applyTextureRule(materialRuleTextureID, mat); }

    // add glowing radiance
    mainRay.radiance += 12 * tint * mat.emission * mainRay.throughput;
    mainRay.hitT = gl_HitTEXT;
    mainRay.normal = vec3(0.0);
    mainRay.directLightHitT = 0.0;
    mainRay.instanceIndex = instanceID;
    mainRay.geometryIndex = geometryID;
    mainRay.primitiveIndex = gl_PrimitiveID;
    mainRay.baryCoords = baryCoords;
    mainRay.worldPos = worldPos;
    mainRay.albedoEmission = albedoEmission;
    mainRay.emissiveBlockType = m0.emissiveBlockType & 255u;
    rayStoreMaterial(mainRay, albedoValue, mat.f0, mat.roughness, mat.metallic, mat.transmission, mat.ior, mat.emission);
    raySetNoisy(mainRay, false);
    mainRay.hasPrevScenePos = 0u;
    raySetStop(mainRay, true);
}
