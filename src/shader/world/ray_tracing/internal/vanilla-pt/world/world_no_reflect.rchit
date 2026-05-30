#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#include "util/disney.glsl"
#include "util/alpha_mode.glsl"
#include "util/random.glsl"
#include "util/ray_cone.glsl"
#include "util/ray.glsl"
#include "util/sampling_helpers.glsl"
#include "util/util.glsl"
#include "common/shared.hpp"

layout(set = 0, binding = 0) uniform sampler2D textures[];

layout(set = 1, binding = 0) uniform accelerationStructureEXT topLevelAS;

layout(set = 1, binding = 1) readonly buffer BLASOffsets {
    uint offsets[];
}
blasOffsets;

layout(set = 1, binding = 2) readonly buffer IndexBufferAddr {
    uint64_t addrs[];
}
indexBufferAddrs;

layout(set = 1, binding = 3) readonly buffer LastIndexBufferAddr {
    uint64_t addrs[];
}
lastIndexBufferAddrs;

layout(set = 1, binding = 8) readonly buffer LastObjToWorldMat {
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

layout(set = 3, binding = 1, rgba8) uniform image2D diffuseAlbedoImage;
layout(set = 3, binding = 2, rgba8) uniform image2D specularAlbedoImage;
layout(set = 3, binding = 3, rgba16f) uniform image2D normalRoughnessImage;
layout(set = 3, binding = 4, rg16f) uniform image2D motionVectorImage;
layout(set = 3, binding = 5, r16f) uniform image2D linearDepthImage;

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer IndexBuffer {
    uint indices[];
}
indexBuffer;

#include "util/vertex.glsl"

layout(location = 0) rayPayloadInEXT MainRay mainRay;
hitAttributeEXT vec2 attribs;

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
    uint alphaMode = getAlphaMode(m0.packedData);
    vec4 albedoValue;
    vec4 specularValue;
    vec4 normalValue;
    vec2 textureUV;
    if (useTexture) {
        int specularTextureID = mapping.entries[textureID].specular;
        int normalTextureID = mapping.entries[textureID].normal;
        textureUV = baryCoords.x * m0.textureUV + baryCoords.y * m1.textureUV + baryCoords.z * m2.textureUV;
        vec2 atlasUvMin = min(m0.textureUV, min(m1.textureUV, m2.textureUV));
        vec2 atlasUvMax = max(m0.textureUV, max(m1.textureUV, m2.textureUV));

        // ray cone
        float coneRadiusWorld = mainRay.coneWidth + gl_HitTEXT * mainRay.coneSpread;
        vec3 dposdu, dposdv;
        computedposduDv(p0.pos, p1.pos, p2.pos, m0.textureUV, m1.textureUV, m2.textureUV, dposdu, dposdv);
        float lod = lodWithCone(textures[nonuniformEXT(textureID)], textureUV, coneRadiusWorld, dposdu, dposdv);

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
    if (useOverlay) {
        ivec2 overlayUV = m0.overlayUV;
        vec4 overlayColor = sampleTexture(textures[nonuniformEXT(worldUBO.overlayTextureID)], overlayUV, 0, false);
        tint = mix(overlayColor.rgb, albedoValue.rgb * colorLayer, overlayColor.a) + glint;
    }

    albedoValue = vec4(tint, albedoValue.a);
    LabPBRMat mat = convertLabPBRMaterial(albedoValue, specularValue, normalValue);

    // add glowing radiance
    mainRay.radiance += 12 * tint * mat.emission * mainRay.throughput;
    mainRay.hitT = gl_HitTEXT;
    mainRay.normal = vec3(0.0);
    rayStoreMaterial(mainRay, albedoValue, mat.f0, mat.roughness, mat.metallic, mat.transmission, mat.ior, mat.emission);
    raySetNoisy(mainRay, false);
    mainRay.hasPrevScenePos = 0u;
    raySetStop(mainRay, true);
}
