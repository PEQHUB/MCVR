#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#include "common/shared.hpp"
#include "common/chunk_lookup.glsl"
#include "util/disney.glsl"
#include "util/alpha_mode.glsl"
#include "common/fft_water.glsl"
#include "util/height_map.glsl"
#include "common/parallax_trace.glsl"
#include "util/random.glsl"
#include "util/ray_cone.glsl"
#include "util/ray.glsl"
#include "util/sampling_helpers.glsl"
#include "util/util.glsl"

layout(set = 0, binding = 0) uniform sampler2D textures[];
layout(set = 5, binding = 0) uniform sampler2D transLUT;
layout(set = 5, binding = 2) uniform samplerCube skyFull;

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

layout(set = 1, binding = 6) readonly buffer LastPositionBufferAddr {
    uint64_t addrs[];
}
lastPositionBufferAddrs;

layout(set = 2, binding = 0) uniform WorldUniform {
    WorldUBO worldUBO;
};

layout(set = 2, binding = 2) uniform SkyUniform {
    SkyUBO skyUBO;
};

#include "common/volumetric_cloud.glsl"

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer IndexBuffer {
    uint indices[];
}
indexBuffer;

#include "util/vertex.glsl"
#include "common/constants.glsl"
#include "common/parallax_condition.glsl"

layout(location = 0) rayPayloadInEXT MainRay mainRay;
layout(location = 1) rayPayloadEXT ShadowRay shadowRay;
hitAttributeEXT vec2 attribs;

struct SampledSurface {
    vec2 uv;
    float depth;
    bool edgeWall;
    vec3 worldPos;
    vec3 geometricNormal;
    vec3 shadingNormal;
    vec4 albedoValue;
    vec4 specularValue;
    vec4 normalValue;
    vec3 tint;
    LabPBRMat mat;
};

bool isFiniteFloat(float value) {
    return !isnan(value) && !isinf(value);
}

bool isFiniteVec2(vec2 value) {
    return !any(isnan(value)) && !any(isinf(value));
}

bool isFiniteVec3(vec3 value) {
    return !any(isnan(value)) && !any(isinf(value));
}

bool isValidSampledSurface(SampledSurface surface) {
    return isFiniteVec2(surface.uv) && isFiniteFloat(surface.depth) && isFiniteVec3(surface.worldPos) &&
           isFiniteVec3(surface.geometricNormal) && isFiniteVec3(surface.shadingNormal) &&
           isFiniteFloat(surface.mat.transmission) && isFiniteFloat(surface.mat.roughness) &&
           isFiniteFloat(surface.mat.ior) && isFiniteFloat(surface.mat.emission) &&
           dot(surface.geometricNormal, surface.geometricNormal) > 1e-10 &&
           dot(surface.shadingNormal, surface.shadingNormal) > 1e-10;
}

bool directLightSurfaceEligible(SampledSurface surface) {
    return isValidSampledSurface(surface) &&
           surface.mat.emission <= ADV_DIRECT_LIGHT_EMISSIVE_SURFACE_EPSILON &&
           surface.mat.roughness >= ADV_DIRECT_LIGHT_MIN_SURFACE_ROUGHNESS;
}

void buildSurfaceBasis(vec3 dPdu,
                       vec3 dPdv,
                       vec3 geometricNormal,
                       out vec3 tangent,
                       out vec3 bitangent) {
    tangent = normalizeF(dPdu - geometricNormal * dot(geometricNormal, dPdu),
                         abs(geometricNormal.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0));
    bitangent = normalizeF(cross(geometricNormal, tangent), dPdv);
    tangent = normalizeF(cross(bitangent, geometricNormal), tangent);
}

vec3 applyNormalMapToBasis(vec3 matNormal,
                           vec3 tangent,
                           vec3 bitangent,
                           vec3 geometricNormal,
                           vec3 viewDir) {
    if (any(isnan(matNormal))) { return geometricNormal; }

    vec3 correctedLocalNormal = matNormal;
    correctedLocalNormal.y = -correctedLocalNormal.y;

    vec3 normal = normalizeF(tangent * correctedLocalNormal.x + bitangent * correctedLocalNormal.y +
                                 geometricNormal * correctedLocalNormal.z,
                             geometricNormal);
    float NdotV = dot(normal, viewDir);
    if (NdotV >= 0.99999) { return normal; }

    vec3 edgeNormal = normal - viewDir * NdotV;
    float edgeNormalLen2 = dot(edgeNormal, edgeNormal);
    if (edgeNormalLen2 <= 1e-10) { return geometricNormal; }

    float weight = 1.0 - NdotV;
    weight = sin(min(weight, PI * 0.5));
    weight = clamp(min(max(NdotV, dot(viewDir, geometricNormal)), 1.0 - weight), 0.0, 1.0);

    float tangentWeight2 = max(1.0 - weight * weight, 0.0);
    if (tangentWeight2 <= 1e-10) { return geometricNormal; }

    return viewDir * weight + edgeNormal * inversesqrt(edgeNormalLen2 / tangentWeight2);
}

void sampleSurfaceState(bool useTexture,
                        uint textureID,
                        TextureMapEntry textureMap,
                        vec2 atlasUvMin,
                        vec2 atlasUvMax,
                        vec2 uv,
                        float lod,
                        uint alphaMode,
                        vec4 colorLayerValue,
                        vec3 colorLayer,
                        vec3 glint,
                        bool useOverlay,
                        ivec2 overlayUV,
                        vec3 dPduWorld,
                        vec3 dPdvWorld,
                        vec3 baseGeoNormal,
                        bool hasHeightMap,
                        float maxDepthWorld,
                        HeightMapHit localHit,
                        vec3 worldPos,
                        vec3 viewDir,
                        bool isFftWaterSurface,
                        out SampledSurface surface) {
    vec4 albedoValue = vec4(1.0);
    vec4 specularValue = vec4(0.0);
    vec4 normalValue = vec4(0.0);
    bool useFlatEdgeBand = false;

    if (useTexture) {
        albedoValue = sampleTexture(textures[nonuniformEXT(textureID)], uv, lod, false);
        albedoValue.a = resolveSurfaceAlpha(albedoValue.a * colorLayerValue.a, alphaMode);
        specularValue = textureMap.specular >= 0 ?
                            sampleTexture(textures[nonuniformEXT(textureMap.specular)], uv, lod, false) :
                            vec4(0.0);
        normalValue = textureMap.normal >= 0 ?
                          samplePBRTexture(textures[nonuniformEXT(textureMap.normal)], uv, atlasUvMin, atlasUvMax, lod,
                                           ADV_PBR_SAMPLING_MODE) :
                          vec4(0.0);
        if (hasHeightMap && textureMap.normal >= 0) {
            ivec2 heightMapSize = textureSize(textures[nonuniformEXT(textureMap.normal)], 0);
            useFlatEdgeBand = isEdgeUV(uv, atlasUvMin, atlasUvMax, heightMapSize);
        }
    }

    vec3 tint = albedoValue.rgb * colorLayer + glint;
    if (useOverlay) {
        vec4 overlayColor = sampleTexture(textures[nonuniformEXT(worldUBO.overlayTextureID)], overlayUV, 0, false);
        tint = mix(overlayColor.rgb, albedoValue.rgb * colorLayer, overlayColor.a) + glint;
    }

    albedoValue = vec4(tint, albedoValue.a);
    LabPBRMat mat = convertLabPBRMaterial(albedoValue, specularValue, normalValue);

    vec3 geometricNormal = localHit.sideWall ? localHit.geometricNormal : baseGeoNormal;
    vec3 shadingNormal = geometricNormal;
    if (isFftWaterSurface && !localHit.sideWall) {
        vec3 absWorldPos = worldPos + vec3(worldUBO.cameraPos.xyz);
        vec3 waterCoordNormal = baseGeoNormal.y >= 0.0 ? baseGeoNormal : -baseGeoNormal;
        vec2 waterCoord = fftWaterSurfaceCoord(absWorldPos, dPduWorld, dPdvWorld, waterCoordNormal);
        FftWaterSample waterSample = sampleFftWater(waterCoord, worldUBO.gameTime);
        vec3 tangent, bitangent;
        fftWaterStableBasis(baseGeoNormal, tangent, bitangent);
        vec3 localWaterNormal = waterSample.localNormal;
        vec3 waterTint = vec3(0.95, 0.98, 1.0);
        albedoValue.rgb = waterTint;
        tint = waterTint;
        mat.f0 = vec3(0.02);
        mat.albedo = waterTint;
        mat.roughness = clamp(0.005 + 0.018 * min(length(localWaterNormal.xy), 0.55), 0.005, 0.026);
        mat.metallic = 0.0;
        mat.transmission = 1.0;
        mat.ior = 1.333;
        shadingNormal = applyNormalMapToBasis(localWaterNormal, tangent, bitangent, baseGeoNormal, viewDir);
    } else {
        if (ADV_PBR_SAMPLING_MODE != 0u && hasHeightMap && !localHit.sideWall && textureMap.normal >= 0 &&
            !useFlatEdgeBand) {
            geometricNormal = sampleNormal(textures[nonuniformEXT(textureMap.normal)], uv, atlasUvMin, atlasUvMax,
                                           dPduWorld, dPdvWorld, baseGeoNormal, 0, ADV_PBR_SAMPLING_MODE,
                                           maxDepthWorld, viewDir);
        }

        if (!localHit.sideWall) {
            vec3 tangent, bitangent;
            buildSurfaceBasis(dPduWorld, dPdvWorld, geometricNormal, tangent, bitangent);
            shadingNormal = applyNormalMapToBasis(mat.normal, tangent, bitangent, geometricNormal, viewDir);
        }
    }

    surface.uv = uv;
    surface.depth = localHit.depth;
    surface.edgeWall = localHit.edgeWall;
    surface.worldPos = worldPos;
    surface.geometricNormal = geometricNormal;
    surface.shadingNormal = shadingNormal;
    surface.albedoValue = albedoValue;
    surface.specularValue = specularValue;
    surface.normalValue = normalValue;
    surface.tint = tint;
    surface.mat = mat;
}

vec3 basePlaneWorldPosAtUv(vec2 uv,
                           vec2 referenceUv,
                           vec3 referenceWorldPos,
                           vec3 dPduWorld,
                           vec3 dPdvWorld) {
    vec2 uvOffset = uv - referenceUv;
    return referenceWorldPos + dPduWorld * uvOffset.x + dPdvWorld * uvOffset.y;
}

vec3 heightMapWorldPosAtUvDepth(vec2 uv,
                                float depth,
                                vec2 referenceUv,
                                vec3 referenceWorldPos,
                                vec3 dPduWorld,
                                vec3 dPdvWorld,
                                vec3 baseGeoNormal) {
    return basePlaneWorldPosAtUv(uv, referenceUv, referenceWorldPos, dPduWorld, dPdvWorld) -
           baseGeoNormal * depth;
}

bool traceLocalHeightIntersectionAndExit(int normalTextureID,
                                         vec2 atlasUvMin,
                                         vec2 atlasUvMax,
                                         vec3 dPduWorld,
                                         vec3 dPdvWorld,
                                         vec3 baseGeoNormal,
                                         float maxDepthWorld,
                                         SampledSurface surface,
                                         vec3 worldDir,
                                         int maxTraceSteps,
                                         out HeightMapHit hit,
                                         out vec2 exitUv,
                                         out float exitDepth,
                                         out float exitDistance) {
    initRestirParallaxMiss(surface.uv, surface.depth, baseGeoNormal, hit);
    exitUv = surface.uv;
    exitDepth = surface.depth;
    exitDistance = 0.0;
    if (normalTextureID < 0 || maxDepthWorld <= heightMapMinWorldDepth) { return false; }
    ivec2 heightMapSize = textureSize(textures[nonuniformEXT(normalTextureID)], 0);
    if (heightMapSize.x <= 0 || heightMapSize.y <= 0) { return false; }
    vec2 traceAtlasUvMin = heightMapInnerMinUV(atlasUvMin, atlasUvMax, heightMapSize);
    vec2 traceAtlasUvMax = heightMapInnerMaxUV(atlasUvMin, atlasUvMax, heightMapSize);

    float startBias = 2.0 * heightMapTraceBias;
    vec2 rateUV = directionToRateUv(worldDir, dPduWorld, dPdvWorld);
    float depthRate = dot(worldDir, -baseGeoNormal);
    vec2 uv = surface.uv + rateUV * startBias;
    float depth = surface.depth + depthRate * startBias;
    if (dot(surface.geometricNormal, baseGeoNormal) > 0.5 && depthRate < 0.0) {
        depth = min(depth, surface.depth - startBias);
    }

    bool fallbackToBilinear =
        ADV_PBR_SAMPLING_MODE == 0u && shouldFallbackNearestSecondaryToBilinear(worldDir, baseGeoNormal);
    bool localHit;
    if (depthRate < -1e-6) {
        float maxTraceDistance = max(depth / -depthRate, 0.0);
        localHit = fallbackToBilinear ?
                       traceBilinearHeightMap(textures[nonuniformEXT(normalTextureID)], traceAtlasUvMin, traceAtlasUvMax, uv,
                                              depth, worldDir, dPduWorld, dPdvWorld, baseGeoNormal, maxDepthWorld,
                                              maxTraceDistance, hit) :
                       traceHeightMap(textures[nonuniformEXT(normalTextureID)], traceAtlasUvMin, traceAtlasUvMax, uv, depth,
                                      worldDir, dPduWorld, dPdvWorld, baseGeoNormal, maxDepthWorld,
                                      maxTraceDistance, ADV_PBR_SAMPLING_MODE, hit);
    } else {
        localHit = fallbackToBilinear ?
                       traceRestirBilinearHeightMapCapped(textures[nonuniformEXT(normalTextureID)], traceAtlasUvMin,
                                                          traceAtlasUvMax, uv, depth, worldDir, dPduWorld, dPdvWorld,
                                                          baseGeoNormal, maxDepthWorld, maxTraceSteps, hit) :
                       traceRestirHeightMapCapped(textures[nonuniformEXT(normalTextureID)], traceAtlasUvMin, traceAtlasUvMax,
                                                  uv, depth, worldDir, dPduWorld, dPdvWorld, baseGeoNormal,
                                                  maxDepthWorld, ADV_PBR_SAMPLING_MODE, maxTraceSteps, hit);
    }
    if (localHit) { return true; }

    if (depthRate >= -1e-6) { return true; }

    float exitT = max(depth / -depthRate, 0.0);
    exitUv = clamp(uv + rateUV * exitT, traceAtlasUvMin, traceAtlasUvMax);
    exitDepth = 0.0;
    exitDistance = exitT;
    return false;
}

vec3 sampleSurfaceDirectionalLight(SampledSurface surface,
                                   vec3 viewDir,
                                   vec2 referenceUv,
                                   vec3 referenceWorldPos,
                                   vec2 atlasUvMin,
                                   vec2 atlasUvMax,
                                   vec3 dPduWorld,
                                   vec3 dPdvWorld,
                                   vec3 baseGeoNormal,
                                   bool traceLocalHeight,
                                   int normalTextureID,
                                   float maxDepthWorld,
                                   bool isFftWaterSurface,
                                   bool hasWaterEntry) {
    if (worldUBO.skyType != 1) { return vec3(0.0); }
    if (!directLightSurfaceEligible(surface)) { return vec3(0.0); }

    bool isOpaqueSurface = surface.mat.transmission <= EPS;
    vec3 lightDir = celestialSunDirection();
    if (lightDir.y < 0.0) { lightDir = -lightDir; }

    vec3 sampledLightDir = SampleVMF(mainRay.seed, lightDir, 3000.0);
    if (!isFiniteVec3(sampledLightDir)) { return vec3(0.0); }
    float sampledLightNoL = dot(sampledLightDir, surface.geometricNormal);
    if (isOpaqueSurface && sampledLightNoL <= 0.0) { return vec3(0.0); }

    float lightPdf;
    vec3 lightBRDF = DisneyEval(surface.mat, viewDir, surface.shadingNormal, sampledLightDir, lightPdf,
                                RADIANCE_BRDF_FLAGS);
    if (!isFiniteFloat(lightPdf) || !isFiniteVec3(lightBRDF) || lightPdf <= 1e-6) { return vec3(0.0); }

    vec2 shadowOriginUv = surface.uv;
    float shadowOriginDepth = surface.depth;
    float shadowOriginDistance = 0.0;
    vec3 shadowOriginPos = surface.worldPos;
    vec3 shadowOriginGeomNormal = surface.geometricNormal;
    if (traceLocalHeight) {
        HeightMapHit shadowSelfHit;
        vec2 exitUv;
        float exitDepth;
        float exitDistance;
        if (traceLocalHeightIntersectionAndExit(normalTextureID, atlasUvMin, atlasUvMax, dPduWorld, dPdvWorld,
                                                baseGeoNormal, maxDepthWorld, surface, sampledLightDir,
                                                ADV_PARALLAX_SECONDARY_MAX_STEPS, shadowSelfHit, exitUv,
                                                exitDepth, exitDistance)) {
            return vec3(0.0);
        }
        if (!surface.edgeWall) {
            shadowOriginUv = exitUv;
            shadowOriginDepth = exitDepth;
            shadowOriginDistance = exitDistance;
            shadowOriginPos =
                heightMapWorldPosAtUvDepth(shadowOriginUv, shadowOriginDepth, referenceUv, referenceWorldPos,
                                           dPduWorld, dPdvWorld, baseGeoNormal);
            shadowOriginGeomNormal = baseGeoNormal;
        }
    }

    shadowRay.radiance = vec3(0.0);
    shadowRay.throughput = vec3(1.0);
    shadowRay.insideBoat = rayInsideBoat(mainRay) ? 1u : 0u;
    shadowRay.pad0 = 0u;

    vec3 shadowOrigin = shadowOriginPos - sampledLightDir * 0.0002;
    if (!surface.edgeWall) {
        vec3 visibilityNormal = dot(sampledLightDir, shadowOriginGeomNormal) >= 0.0 ? shadowOriginGeomNormal :
                                                                                  -shadowOriginGeomNormal;
        shadowOrigin = shadowOriginPos + visibilityNormal * 0.0002;
    }
    float shadowLength = max(1000.0 - shadowOriginDistance, 0.0001);
    if (!isFiniteVec3(shadowOrigin) || !isFiniteFloat(shadowLength)) { return vec3(0.0); }
    uint shadowMask = WORLD_MASK | PLAYER_MASK;
    if (ADV_CLOUD_MODE != 2u) { shadowMask |= CLOUD_MASK; }
    traceRayEXT(topLevelAS, gl_RayFlagsNoneEXT, shadowMask, 0, 0, 0, shadowOrigin, 0.0001, sampledLightDir,
                shadowLength, 1);

    float progress = skyUBO.rainGradient;
    vec3 lightRadiance = shadowRay.radiance * mainRay.throughput * lightBRDF;
    if (ADV_CLOUD_MODE == 2u) {
        vec3 absoluteShadowOriginPos = shadowOriginPos + vec3(worldUBO.cameraPos.xyz);
        float cloudVisibility = volumetricCloudLightVisibility(absoluteShadowOriginPos, lightDir,
                                                               max(ADV_VOLUMETRIC_CLOUD_LIGHT_STEPS, 1), 0.175);
        lightRadiance *= cloudVisibility;
    }
    bool applyUnderwaterCaustic = !isFftWaterSurface && (skyUBO.cameraSubmersionType == 1 || hasWaterEntry);
    if (applyUnderwaterCaustic) {
        vec3 absWorldPos = surface.worldPos + vec3(worldUBO.cameraPos.xyz);
        float waterDepth = distance(surface.worldPos, mainRay.origin);
        lightRadiance *= sampleFftWaterCaustic(absWorldPos.xz, worldUBO.gameTime, sampledLightDir, waterDepth);
    }
    return mix(lightRadiance, vec3(0.0), progress);
}

bool loadPreviousScenePos(uint geometryBufferIndex, uint primitiveID, vec3 baryCoords, out vec3 prevScenePos) {
    uint64_t lastPositionAddr = lastPositionBufferAddrs.addrs[geometryBufferIndex];
    uint64_t lastIndexAddr = lastIndexBufferAddrs.addrs[geometryBufferIndex];
    if (lastPositionAddr == 0 || lastIndexAddr == 0) { return false; }

    IndexBuffer lastIndexBuffer = IndexBuffer(lastIndexAddr);
    uint indexBaseID = 3u * primitiveID;
    uint i0 = lastIndexBuffer.indices[indexBaseID];
    uint i1 = lastIndexBuffer.indices[indexBaseID + 1u];
    uint i2 = lastIndexBuffer.indices[indexBaseID + 2u];

    PositionBuffer lastPositionBuffer = PositionBuffer(lastPositionAddr);
    vec3 p0 = lastPositionBuffer.vertices[i0].pos;
    vec3 p1 = lastPositionBuffer.vertices[i1].pos;
    vec3 p2 = lastPositionBuffer.vertices[i2].pos;
    vec3 prevLocalPos = baryCoords.x * p0 + baryCoords.y * p1 + baryCoords.z * p2;
    mat4 lastModelMat = lastObjToWorldMats.mat[gl_InstanceCustomIndexEXT];
    prevScenePos = mat3(lastModelMat) * prevLocalPos + lastModelMat[3].xyz;
    return true;
}

void main() {
    uint instanceID = gl_InstanceCustomIndexEXT;
    uint geometryID = gl_GeometryIndexEXT;
    uint geometryBufferIndex = getGeometryBufferIndex(instanceID, geometryID);

    uint i0, i1, i2;
    MaterialVertex m0, m1, m2;
    loadTriangleIndices(geometryBufferIndex, gl_PrimitiveID, i0, i1, i2);
    loadTriangleMaterial(geometryBufferIndex, i0, i1, i2, m0, m1, m2);
    PositionVertex p0, p1, p2;
    loadTrianglePositions(geometryBufferIndex, i0, i1, i2, p0, p1, p2);

    vec3 baryCoords = vec3(1.0 - (attribs.x + attribs.y), attribs.x, attribs.y);
    vec3 planeHitWorldPos = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
    vec3 baseViewDir = -mainRay.direction;

    uint packedData = m0.packedData;
    bool useColorLayer = hasColorLayer(packedData);
    bool useTexture = hasTexture(packedData);
    bool useGlint = hasGlint(packedData);
    bool useOverlay = hasOverlay(packedData);
    vec4 colorLayerValue = useColorLayer ?
                               baryCoords.x * m0.colorLayer + baryCoords.y * m1.colorLayer + baryCoords.z * m2.colorLayer :
                               vec4(1.0);
    vec3 colorLayer = colorLayerValue.rgb;
    uint bounce = rayBounce(mainRay);

    float albedoEmission =
        baryCoords.x * m0.albedoEmission + baryCoords.y * m1.albedoEmission + baryCoords.z * m2.albedoEmission;
    uint textureID = m0.textureID;
    uint alphaMode = getAlphaMode(packedData);
    uint coordinate = getCoordinate(packedData);

    vec2 textureUV = vec2(0.0);
    TextureMapEntry textureMap = TextureMapEntry(-1, -1, -1);
    vec2 atlasUvMin = vec2(0.0);
    vec2 atlasUvMax = vec2(0.0);
    float lod = 0.0;
    vec3 dposdu = vec3(1.0, 0.0, 0.0);
    vec3 dposdv = vec3(0.0, 1.0, 0.0);
    bool hasHeightMapSurface = false;
    float maxDepthWorld = 0.0;
    mat3 objectToWorld = mat3(gl_ObjectToWorld3x4EXT);
    mat3 normalMatrix = transpose(mat3(gl_WorldToObject3x4EXT));
    vec3 baseGeoNormal =
        normalizeF(normalMatrix * cross(p1.pos - p0.pos, p2.pos - p0.pos), vec3(0.0, 1.0, 0.0));
    vec3 planeGeoNormal = baseGeoNormal;
    vec3 dPduWorld = objectToWorld * dposdu;
    vec3 dPdvWorld = objectToWorld * dposdv;
    bool isWaterMaterial = false;
    bool hasFftWaterSurface = false;
    bool parallaxEnabled = false;
    bool useRealisticWaterSurface = ADV_WATER_SURFACE_MODE == 1u;

    if (useTexture) {
        textureMap = mapping.entries[textureID];
        textureUV = baryCoords.x * m0.textureUV + baryCoords.y * m1.textureUV + baryCoords.z * m2.textureUV;
        atlasUvMin = min(m0.textureUV, min(m1.textureUV, m2.textureUV));
        atlasUvMax = max(m0.textureUV, max(m1.textureUV, m2.textureUV));

        float coneRadiusWorld = mainRay.coneWidth + gl_HitTEXT * mainRay.coneSpread;
        computedposduDv(p0.pos, p1.pos, p2.pos, m0.textureUV, m1.textureUV, m2.textureUV, dposdu, dposdv);
        lod = lodWithCone(textures[nonuniformEXT(textureID)], textureUV, coneRadiusWorld, dposdu, dposdv);

        dPduWorld = objectToWorld * dposdu;
        dPdvWorld = objectToWorld * dposdv;
        planeGeoNormal = normalizeF(cross(dPduWorld, dPdvWorld), baseGeoNormal);
        baseGeoNormal = planeGeoNormal;
        if (dot(baseGeoNormal, baseViewDir) < 0.0) { baseGeoNormal = -baseGeoNormal; }

        if (useRealisticWaterSurface && textureMap.flag >= 0) {
            ivec4 flags = ivec4(round(sampleTexture(textures[nonuniformEXT(textureMap.flag)], textureUV, ceil(lod), false) * 255.0));
            isWaterMaterial = (flags.r & 0x1) > 0;
        }
        hasFftWaterSurface = isWaterMaterial && abs(planeGeoNormal.y) > 0.75;

        if (parallaxEnabled && textureMap.normal >= 0 && coordinate != 1u) {
            maxDepthWorld = heightMapMaxDepthWorld(atlasUvMin, atlasUvMax, dPduWorld, dPdvWorld);
            hasHeightMapSurface =
                maxDepthWorld > heightMapMinWorldDepth && dot(baseViewDir, baseGeoNormal) > ADV_PARALLAX_MIN_VIEW_DOT;
        }
    } else {
        dPduWorld = objectToWorld * dposdu;
        dPdvWorld = objectToWorld * dposdv;
    }
    if (dot(baseGeoNormal, baseViewDir) < 0.0) { baseGeoNormal = -baseGeoNormal; }
    bool traceLocalHeight = hasHeightMapSurface && shouldTraceRestirParallax(lod, planeHitWorldPos) && !hasFftWaterSurface;

    HeightMapHit initialHit;
    initialHit.hit = false;
    initialHit.sideWall = false;
    initialHit.edgeWall = false;
    initialHit.t = 0.0;
    initialHit.uv = textureUV;
    initialHit.depth = 0.0;
    initialHit.geometricNormal = baseGeoNormal;
    if (traceLocalHeight) {
        HeightMapHit tracedInitialHit;
        if (traceRestirHeightMapCapped(textures[nonuniformEXT(textureMap.normal)], atlasUvMin, atlasUvMax, textureUV,
                                       0.0, gl_WorldRayDirectionEXT, dPduWorld, dPdvWorld, baseGeoNormal,
                                       maxDepthWorld, ADV_PBR_SAMPLING_MODE, ADV_PARALLAX_PRIMARY_MAX_STEPS,
                                       tracedInitialHit)) {
            initialHit = tracedInitialHit;
        }
    }

    vec3 hitWorldPos = planeHitWorldPos + gl_WorldRayDirectionEXT * initialHit.t;
    float actualHitT = gl_HitTEXT + initialHit.t;

    mainRay.hitT = actualHitT;
    mainRay.coneWidth += actualHitT * mainRay.coneSpread;
    mainRay.directLightRadiance = vec3(0.0);
    mainRay.hasPrevScenePos = 0u;
    if (bounce == 0u) {
        vec3 prevScenePos;
        if (loadPreviousScenePos(geometryBufferIndex, gl_PrimitiveID, baryCoords, prevScenePos)) {
            mainRay.prevScenePos = prevScenePos;
            mainRay.hasPrevScenePos = 1u;
        }
    }

    vec3 glint = vec3(0.0);
    if (useGlint) {
        vec2 glintUV = baryCoords.x * m0.glintUV + baryCoords.y * m1.glintUV + baryCoords.z * m2.glintUV;
        glintUV = (worldUBO.textureMat * vec4(glintUV, 0.0, 1.0)).xy;
        glint = sampleTexture(textures[nonuniformEXT(m0.glintTexture)], glintUV, false).rgb;
    }
    glint *= glint;

    vec3 viewDir = -gl_WorldRayDirectionEXT;

    SampledSurface surface;
    sampleSurfaceState(useTexture, textureID, textureMap, atlasUvMin, atlasUvMax, initialHit.uv, lod, alphaMode,
                       colorLayerValue, colorLayer, glint, useOverlay, m0.overlayUV, dPduWorld, dPdvWorld,
                       baseGeoNormal, hasHeightMapSurface, maxDepthWorld, initialHit, hitWorldPos, viewDir,
                       hasFftWaterSurface, surface);

    bool incomingWaterEntry = rayLoadAux(mainRay).y > 0.5;
    mainRay.normal = surface.shadingNormal;
    rayStoreMaterial(mainRay, surface.albedoValue, surface.mat.f0, surface.mat.roughness, surface.mat.metallic,
                     surface.mat.transmission, surface.mat.ior, surface.mat.emission);
    rayStoreAux(mainRay, hasFftWaterSurface ? vec2(ADV_FFT_WATER_ORIGIN_BIAS, 1.0) : vec2(0.0));

    SampledSurface currentSurface = surface;
    vec3 currentViewDir = viewDir;
    bool storedLobeType = false;
    for (int localBounce = 0; localBounce < 1; ++localBounce) {
        float emissionFactor =
            (bounce == 0u && localBounce == 0) ? ADV_DIRECT_LIGHT_STRENGTH : ADV_INDIRECT_LIGHT_STRENGTH;
        vec3 emissionRadiance = emissionFactor * currentSurface.tint * currentSurface.mat.emission * mainRay.throughput;
        emissionRadiance += currentSurface.tint * albedoEmission * mainRay.throughput;
        mainRay.radiance += emissionRadiance;

        vec3 directLight =
            sampleSurfaceDirectionalLight(currentSurface, currentViewDir, textureUV, planeHitWorldPos, atlasUvMin,
                                          atlasUvMax, dPduWorld, dPdvWorld, baseGeoNormal,
                                          traceLocalHeight, textureMap.normal, maxDepthWorld,
                                          hasFftWaterSurface, incomingWaterEntry);
        if (localBounce == 0) { mainRay.directLightRadiance = directLight; }
        mainRay.radiance += directLight;

        bool isOpaqueSurface = currentSurface.mat.transmission <= EPS;
        vec3 sampleDir;
        float pdf;
        uint lobeType;
        vec3 bsdf;
        if (hasFftWaterSurface && currentSurface.mat.transmission > EPS) {
            vec3 incident = -currentViewDir;
            vec3 waterNormal = currentSurface.shadingNormal;
            float eta = fftWaterEtaForIncident(incident, waterNormal, currentSurface.mat.ior);
            float fresnel = clamp(DielectricFresnel(abs(dot(currentViewDir, waterNormal)), eta), 0.0, 1.0);
            vec3 refractionDir = refract(incident, waterNormal, eta);
            bool hasRefraction = dot(refractionDir, refractionDir) > 1e-6;
            bool chooseReflection = !hasRefraction || rand(mainRay.seed) < fresnel;
            if (chooseReflection) {
                sampleDir = normalize(reflect(incident, waterNormal));
                pdf = max(fresnel, 1e-4);
                lobeType = 1u;
                bsdf = vec3(fresnel);
            } else {
                sampleDir = normalize(refractionDir);
                pdf = max(1.0 - fresnel, 1e-4);
                lobeType = 2u;
                bsdf = pow(max(currentSurface.albedoValue.rgb, vec3(0.0)), vec3(0.5)) * (1.0 - fresnel);
            }
        } else {
            bsdf = DisneySample(currentSurface.mat, currentViewDir, currentSurface.shadingNormal, sampleDir, pdf,
                                mainRay.seed, lobeType, RADIANCE_BRDF_FLAGS);
        }

        if (!storedLobeType) {
            raySetLobeType(mainRay, lobeType);
            storedLobeType = true;
        }
        raySetNoisy(mainRay, true);

        if (!isFiniteFloat(pdf) || !isFiniteVec3(sampleDir) || !isFiniteVec3(bsdf) || pdf <= 1e-6 ||
            max(bsdf.r, max(bsdf.g, bsdf.b)) <= 1e-6) {
            raySetStop(mainRay, true);
            return;
        }

        if (isOpaqueSurface && dot(sampleDir, currentSurface.geometricNormal) <= 0.0) {
            raySetStop(mainRay, true);
            return;
        }

        mainRay.throughput *= bsdf / max(pdf, 1e-4);

        if (!traceLocalHeight) {
            if (hasFftWaterSurface) {
                vec3 exitNormal = dot(sampleDir, currentSurface.geometricNormal) >= 0.0 ? currentSurface.geometricNormal :
                                                                                           -currentSurface.geometricNormal;
                mainRay.origin = currentSurface.worldPos + exitNormal * ADV_FFT_WATER_ORIGIN_BIAS;
            } else if (hasHeightMapSurface) {
                vec3 exitBasePos =
                    basePlaneWorldPosAtUv(currentSurface.uv, textureUV, planeHitWorldPos, dPduWorld, dPdvWorld);
                vec3 exitNormal = dot(sampleDir, baseGeoNormal) >= 0.0 ? baseGeoNormal : -baseGeoNormal;
                mainRay.origin = exitBasePos + exitNormal * 0.0002;
            } else {
                vec3 exitNormal = dot(sampleDir, currentSurface.geometricNormal) >= 0.0 ? currentSurface.geometricNormal :
                                                                                           -currentSurface.geometricNormal;
                mainRay.origin = currentSurface.worldPos + exitNormal * 0.0002;
            }
            mainRay.direction = sampleDir;
            raySetStop(mainRay, false);
            return;
        }

        HeightMapHit localBounceHit;
        vec2 exitUv;
        float exitDepth;
        float exitDistance;
        bool localBlocked = traceLocalHeightIntersectionAndExit(
            textureMap.normal, atlasUvMin, atlasUvMax, dPduWorld, dPdvWorld, baseGeoNormal, maxDepthWorld,
            currentSurface, sampleDir, ADV_PARALLAX_SECONDARY_MAX_STEPS, localBounceHit, exitUv, exitDepth,
            exitDistance);
        if (!localBlocked) {
            vec3 exitBasePos =
                heightMapWorldPosAtUvDepth(exitUv, exitDepth, textureUV, planeHitWorldPos, dPduWorld, dPdvWorld,
                                           baseGeoNormal);
            vec3 exitNormal = dot(sampleDir, baseGeoNormal) >= 0.0 ? baseGeoNormal : -baseGeoNormal;
            mainRay.origin = exitBasePos + exitNormal * 0.0002;
            mainRay.direction = sampleDir;
            raySetStop(mainRay, false);
            return;
        }

        if (!localBounceHit.hit || (localBounceHit.sideWall && localBounceHit.edgeWall)) {
            raySetStop(mainRay, true);
            return;
        }

        vec3 nextWorldPos = currentSurface.worldPos + sampleDir * localBounceHit.t;
        sampleSurfaceState(useTexture, textureID, textureMap, atlasUvMin, atlasUvMax, localBounceHit.uv, lod, alphaMode,
                           colorLayerValue, colorLayer, glint, useOverlay, m0.overlayUV, dPduWorld, dPdvWorld,
                           baseGeoNormal, hasHeightMapSurface, maxDepthWorld, localBounceHit, nextWorldPos, -sampleDir,
                           hasFftWaterSurface, currentSurface);
        currentViewDir = -sampleDir;
    }

    raySetStop(mainRay, true);
}
