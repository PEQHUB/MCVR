#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#include "common/shared.hpp"
#include "util/surface_bsdf.glsl"
#include "util/alpha_mode.glsl"
#include "common/fft_water.glsl"
#include "util/height_map.glsl"
#include "util/random.glsl"
#include "util/ray_cone.glsl"
#include "util/ray.glsl"
#include "util/sampling_helpers.glsl"
#include "util/util.glsl"
#include "common/raytrace_transforms.glsl"

layout(set = 0, binding = 0) uniform sampler2D textures[];

#include "util/texture_rules.glsl"
#include "util/sprite_fetch.glsl"

layout(set = 1, binding = 0) uniform accelerationStructureEXT topLevelAS;

layout(set = 1, binding = 1) readonly buffer BLASOffsets {
    uint offsets[];
}
blasOffsets;

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

layout(set = 2, binding = 0) uniform WorldUniform {
    WorldUBO worldUBO;
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

#define PHYSICAL_SUN_DISK ((pc.flags & 2048) != 0)
#define RT_DEBUG_MATERIAL_MODE_SHIFT 24u
#define RT_DEBUG_MATERIAL_MODE_MASK  15u
#define RT_DEBUG_MATERIAL_MODE       ((pc.rtDebugFlags >> RT_DEBUG_MATERIAL_MODE_SHIFT) & RT_DEBUG_MATERIAL_MODE_MASK)
#define RT_DEBUG_MATERIAL_ID         1u
#define RT_DEBUG_MATERIAL_PAGE_LAYER 2u
#define RT_DEBUG_MATERIAL_DISPLACE_ELIGIBLE 3u
#define RT_DEBUG_MATERIAL_DISPLACE_HEIGHT   4u

#include "common/celestial.glsl"

layout(set = 5, binding = 0) uniform sampler2D transLUT;
layout(set = 5, binding = 2) uniform samplerCube skyFull;

#include "common/volumetric_cloud.glsl"

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer IndexBuffer {
    uint indices[];
}
indexBuffer;

#include "util/vertex.glsl"
#include "common/constants.glsl"
#include "common/direct_light_fallback.glsl"

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
                        bool isBlockGeometry,
                        TextureMapEntry textureMap,
                        vec2 atlasUvMin,
                        vec2 atlasUvMax,
                        vec2 uv,
                        float lod,
                        uint alphaMode,
                        vec4 colorLayerValue,
                        vec3 colorLayer,
                        bool foldBlockOverlay,
                        vec3 glint,
                        bool useOverlay,
                        ivec2 overlayUV,
                        vec3 dPduWorld,
                        vec3 dPdvWorld,
                        vec3 baseGeoNormal,
                        bool hasHeightMap,
                        float maxDepthWorld,
                        bool localHitHit,
                        bool localHitSideWall,
                        bool localHitEdgeWall,
                        float localHitDepth,
                        vec3 localHitGeometricNormal,
                        vec3 worldPos,
                        vec3 viewDir,
                        bool isWaterMaterial,
                        bool isFftWaterSurface,
                        out SampledSurface surface) {
    vec4 albedoValue = vec4(1.0);
    vec4 specularValue = vec4(0.0);
    vec4 normalValue = vec4(0.0);
    ivec4 flagValue = ivec4(0);
    uint materialRuleTextureID = textureID;
    bool blockOverlayComposited = false;
    bool useFlatEdgeBand = false;

    if (useTexture) {
        if (isBlockGeometry) {
            albedoValue = fetchBlockAlbedoLod(textureID, uv, worldUBO.animTick, lod);
            albedoValue.a = resolveSurfaceAlpha(albedoValue.a * colorLayerValue.a, alphaMode);
            specularValue = fetchBlockSpecularLod(textureID, uv, lod);
            normalValue = fetchBlockNormalLod(textureID, uv, lod);
            flagValue = fetchBlockFlagLod(textureID, uv, lod);
            bool allowBlockOverlayMaterial = foldBlockOverlay && !localHitHit;
            if (allowBlockOverlayMaterial) {
                blockOverlayComposited = applyBlockOverlayMaterialLod(
                    albedoValue, specularValue, normalValue, flagValue,
                    textureID, uv, worldUBO.animTick, lod, colorLayer,
                    materialRuleTextureID);
            }
        } else {
            albedoValue = sampleTexture(textures[nonuniformEXT(textureID)], uv, lod, false);
            albedoValue.a = resolveSurfaceAlpha(albedoValue.a * colorLayerValue.a, alphaMode);
            specularValue = textureMap.specular >= 0 ?
                                sampleTexture(textures[nonuniformEXT(textureMap.specular)], uv, lod, false) :
                                vec4(0.0);
            normalValue = textureMap.normal >= 0 ?
                              samplePBRTexture(textures[nonuniformEXT(textureMap.normal)], uv, atlasUvMin, atlasUvMax, lod,
                                               VPT_PBR_SAMPLING_MODE) :
                              vec4(0.0);
        }
        if (!isBlockGeometry && hasHeightMap && textureMap.normal >= 0) {
            ivec2 heightMapSize = textureSize(textures[nonuniformEXT(textureMap.normal)], 0);
            useFlatEdgeBand = isEdgeUV(uv, atlasUvMin, atlasUvMax, heightMapSize);
        }
    }

    vec3 tint = albedoValue.rgb * colorLayer + glint;
    if (isBlockGeometry && blockOverlayComposited) {
        tint = albedoValue.rgb + glint;
    } else if (useOverlay) {
        vec4 overlayColor = sampleTexture(textures[nonuniformEXT(worldUBO.overlayTextureID)], overlayUV, 0, false);
        tint = mix(overlayColor.rgb, albedoValue.rgb * colorLayer, overlayColor.a) + glint;
    }

    albedoValue = vec4(tint, albedoValue.a);
    LabPBRMat mat = convertLabPBRMaterial(albedoValue, specularValue, normalValue);
    if (isBlockGeometry) { applyTextureRule(materialRuleTextureID, mat); }

    vec3 geometricNormal = (localHitSideWall || (hasHeightMap && localHitHit)) ? localHitGeometricNormal :
                                                                                 baseGeoNormal;
    vec3 shadingNormal = geometricNormal;
    bool useWaterMaterial = isWaterMaterial && !localHitSideWall;
    if (isBlockGeometry && !useWaterMaterial && alphaMode != ALPHA_MODE_TRANSPARENT &&
        mat.metallic < 0.5 && mat.transmission <= EPS) {
        mat.roughness = max(mat.roughness, 0.35);
        mat.coatWeight = min(mat.coatWeight, 0.15);
    }
    if (useWaterMaterial) {
        vec3 waterTint = vec3(0.95, 0.98, 1.0);
        albedoValue.rgb = waterTint;
        tint = waterTint;
        mat.f0 = vec3(0.02);
        mat.albedo = waterTint;
        mat.roughness = 0.01;
        mat.metallic = 0.0;
        mat.transmission = 1.0;
        mat.ior = 1.333;

        if (isFftWaterSurface) {
            vec3 absWorldPos = worldPos + vec3(worldUBO.cameraPos.xyz);
            vec3 waterCoordNormal = baseGeoNormal.y >= 0.0 ? baseGeoNormal : -baseGeoNormal;
            vec2 waterCoord = fftWaterSurfaceCoord(absWorldPos, dPduWorld, dPdvWorld, waterCoordNormal);
            FftWaterSample waterSample = sampleFftWater(waterCoord, worldUBO.gameTime);
            vec3 tangent, bitangent;
            fftWaterStableBasis(baseGeoNormal, tangent, bitangent);
            vec3 localWaterNormal = normalize(vec3(-waterSample.slope.x, waterSample.slope.y, 1.0));
            mat.roughness = clamp(0.005 + 0.012 * min(length(waterSample.slope), 0.45), 0.005, 0.022);
            shadingNormal = applyNormalMapToBasis(localWaterNormal, tangent, bitangent, baseGeoNormal, viewDir);
        }
    }
    if (isBlockGeometry && RT_DEBUG_MATERIAL_MODE != 0u) {
        albedoValue = vec4(materialDebugColor(materialRuleTextureID, normalValue), 1.0);
        tint = albedoValue.rgb;
        mat.albedo = albedoValue.rgb;
    }

    if (!isBlockGeometry && !isFftWaterSurface && VPT_PBR_SAMPLING_MODE != 0u && hasHeightMap &&
        !localHitSideWall && textureMap.normal >= 0 && !useFlatEdgeBand) {
        geometricNormal = sampleNormal(textures[nonuniformEXT(textureMap.normal)], uv, atlasUvMin, atlasUvMax,
                                       dPduWorld, dPdvWorld, baseGeoNormal, 0, VPT_PBR_SAMPLING_MODE,
                                       maxDepthWorld, viewDir);
    }

    if (!isFftWaterSurface && !localHitSideWall) {
        vec3 tangent, bitangent;
        tangent = normalizeF(dPduWorld - geometricNormal * dot(geometricNormal, dPduWorld),
                             abs(geometricNormal.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0));
        bitangent = normalizeF(cross(geometricNormal, tangent), dPdvWorld);
        tangent = normalizeF(cross(bitangent, geometricNormal), tangent);
        shadingNormal = applyNormalMapToBasis(mat.normal, tangent, bitangent, geometricNormal, viewDir);
    }

    surface.uv = uv;
    surface.depth = localHitDepth;
    surface.edgeWall = localHitEdgeWall;
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

void initVanillaParallaxMiss(vec2 uv, float depth, vec3 baseNormal, out HeightMapHit hit) {
    hit.hit = false;
    hit.sideWall = false;
    hit.edgeWall = false;
    hit.t = INF_DISTANCE;
    hit.uv = uv;
    hit.depth = depth;
    hit.geometricNormal = baseNormal;
}

bool traceNearestHeightMapCapped(sampler2D tex,
                                 vec2 minUV,
                                 vec2 maxUV,
                                 vec2 uv,
                                 float depth,
                                 vec3 worldDir,
                                 vec3 dPdu,
                                 vec3 dPdv,
                                 vec3 baseNormal,
                                 float maxDepth,
                                 int maxTraceSteps,
                                 out HeightMapHit hit) {
    initVanillaParallaxMiss(uv, depth, baseNormal, hit);
    if (maxTraceSteps <= 0 || maxDepth <= heightMapMinWorldDepth) { return false; }

    ivec2 size = textureSize(tex, 0);
    if (size.x <= 0 || size.y <= 0) { return false; }

    vec2 boundsMin = heightMapMinUV(minUV, maxUV, size);
    vec2 boundsMax = heightMapMaxUV(minUV, maxUV, size);
    if (!isUvWithinHeightMap(uv, boundsMin, boundsMax)) { return false; }
    float minEdgeDepth = edgeMinDepth(minUV, maxUV, size, maxDepth);

    vec2 rateUV = directionToRateUv(worldDir, dPdu, dPdv);
    float depthRate = dot(worldDir, -baseNormal);

    ivec2 atlasTexelMin = clampTexelCoord(ivec2(floor(min(minUV, maxUV) * vec2(size))), size);
    ivec2 atlasTexelMax = clampTexelCoord(ivec2(ceil(max(minUV, maxUV) * vec2(size)) - vec2(1.0)), size);
    ivec2 texel = clampTexelCoord(ivec2(floor(uv * vec2(size))), size);
    if (any(lessThan(texel, atlasTexelMin)) || any(greaterThan(texel, atlasTexelMax))) { return false; }

    int remainingUSteps = heightMapRemainingSteps(texel.x, atlasTexelMin.x, atlasTexelMax.x, rateUV.x);
    int remainingVSteps = heightMapRemainingSteps(texel.y, atlasTexelMin.y, atlasTexelMax.y, rateUV.y);
    int maxSteps = min(maxTraceSteps, min(heightMapNearestMaxSteps, remainingUSteps + remainingVSteps + 4));
    if (maxSteps <= 0) { return false; }

    float tCurrent = 0.0;
    vec2 uvCurrent = uv;
    float depthCurrent = depth;

    int step = 0;
    while (step < maxSteps) {
        if (depthCurrent < -heightMapTraceBias) { break; }
        if (any(lessThan(texel, atlasTexelMin)) || any(greaterThan(texel, atlasTexelMax))) { break; }

        float surfaceDepth = sampleHeightDepthNearest(tex, texel, atlasTexelMin, atlasTexelMax, uvCurrent, maxDepth);
        if (depthCurrent >= surfaceDepth - heightMapTraceBias) {
            hit.hit = true;
            hit.sideWall = false;
            hit.edgeWall = false;
            hit.t = tCurrent;
            hit.uv = uvCurrent;
            hit.depth = max(depthCurrent, surfaceDepth);
            hit.geometricNormal = baseNormal;
            return true;
        }

        float tTop = INF_DISTANCE;
        if (depthRate > 1e-6 && surfaceDepth > depthCurrent) { tTop = (surfaceDepth - depthCurrent) / depthRate; }

        float tU = INF_DISTANCE;
        float tV = INF_DISTANCE;
        bool stepU = false;
        bool stepV = false;

        if (rateUV.x > 1e-9) {
            float boundary = (float(texel.x + 1)) / float(size.x);
            tU = (boundary - uvCurrent.x) / rateUV.x;
            stepU = true;
        } else if (rateUV.x < -1e-9) {
            float boundary = float(texel.x) / float(size.x);
            tU = (boundary - uvCurrent.x) / rateUV.x;
            stepU = true;
        }

        if (rateUV.y > 1e-9) {
            float boundary = (float(texel.y + 1)) / float(size.y);
            tV = (boundary - uvCurrent.y) / rateUV.y;
            stepV = true;
        } else if (rateUV.y < -1e-9) {
            float boundary = float(texel.y) / float(size.y);
            tV = (boundary - uvCurrent.y) / rateUV.y;
            stepV = true;
        }

        float tAbove = INF_DISTANCE;
        if (depthRate < -1e-6) { tAbove = -depthCurrent / depthRate; }

        float tEdge = min(tU, tV);
        if (tTop <= min(tEdge, tAbove) + heightMapTraceBias && tTop < INF_DISTANCE) {
            tCurrent += max(tTop, 0.0);
            hit.hit = true;
            hit.sideWall = false;
            hit.edgeWall = false;
            hit.t = tCurrent;
            hit.uv = uvCurrent + rateUV * max(tTop, 0.0);
            hit.depth = surfaceDepth;
            hit.geometricNormal = baseNormal;
            return true;
        }

        if (tAbove <= tEdge + heightMapTraceBias) { break; }
        if (tEdge == INF_DISTANCE) { break; }

        float dt = max(tEdge, 0.0);
        tCurrent += dt;

        vec2 edgeUV = uvCurrent + rateUV * dt;
        float edgeDepth = depthCurrent + depthRate * dt;
        bool cornerStep = stepU && stepV && abs(tU - tV) <= heightMapTraceBias;
        if (cornerStep) {
            ivec2 texelU = texel + ivec2(rateUV.x > 0.0 ? 1 : -1, 0);
            ivec2 texelV = texel + ivec2(0, rateUV.y > 0.0 ? 1 : -1);
            ivec2 texelD = texel + ivec2(rateUV.x > 0.0 ? 1 : -1, rateUV.y > 0.0 ? 1 : -1);
            bool inU = all(greaterThanEqual(texelU, atlasTexelMin)) && all(lessThanEqual(texelU, atlasTexelMax));
            bool inV = all(greaterThanEqual(texelV, atlasTexelMin)) && all(lessThanEqual(texelV, atlasTexelMax));
            bool inD = all(greaterThanEqual(texelD, atlasTexelMin)) && all(lessThanEqual(texelD, atlasTexelMax));

            if (inU && inV && inD) {
                float depthUCorner =
                    sampleHeightDepthNearest(tex, texelU, atlasTexelMin, atlasTexelMax, edgeUV, maxDepth);
                float depthVCorner =
                    sampleHeightDepthNearest(tex, texelV, atlasTexelMin, atlasTexelMax, edgeUV, maxDepth);
                float depthDCorner =
                    sampleHeightDepthNearest(tex, texelD, atlasTexelMin, atlasTexelMax, edgeUV, maxDepth);
                bool enteredWallU = enteredTexelWall(surfaceDepth, depthUCorner, edgeDepth) ||
                                    enteredTexelWall(depthVCorner, depthDCorner, edgeDepth);
                bool enteredWallV = enteredTexelWall(surfaceDepth, depthVCorner, edgeDepth) ||
                                    enteredTexelWall(depthUCorner, depthDCorner, edgeDepth);

                if (enteredWallU || enteredWallV) {
                    hit.hit = true;
                    hit.sideWall = true;
                    hit.edgeWall = false;
                    hit.t = tCurrent;
                    hit.uv = edgeUV;
                    hit.depth = edgeDepth;
                    hit.geometricNormal = wallNormal(enteredWallU, enteredWallV, rateUV, dPdu, dPdv, baseNormal);
                    return true;
                }

                texel = texelD;
                uvCurrent = edgeUV;
                depthCurrent = edgeDepth;
                ++step;
                continue;
            }
        }

        bool processU = false;
        if (tU <= tV + heightMapTraceBias) {
            processU = tU < tV - heightMapTraceBias ? true : abs(rateUV.x) >= abs(rateUV.y);
        }

        ivec2 nextTexel = texel;
        bool enteredSideWall = false;
        vec3 enteredWallNormal = baseNormal;
        bool enteredTop = false;

        if (processU) {
            nextTexel += ivec2(rateUV.x > 0.0 ? 1 : -1, 0);
            if (any(lessThan(nextTexel, atlasTexelMin)) || any(greaterThan(nextTexel, atlasTexelMax))) {
                vec3 edgeNormal;
                if (isEdge(edgeDepth, surfaceDepth, minEdgeDepth, edgeUV, boundsMin, boundsMax, rateUV, dPdu, dPdv,
                           baseNormal, edgeNormal)) {
                    hit.hit = true;
                    hit.sideWall = true;
                    hit.edgeWall = true;
                    hit.t = tCurrent;
                    hit.uv = clamp(edgeUV, boundsMin, boundsMax);
                    hit.depth = edgeDepth;
                    hit.geometricNormal = edgeNormal;
                    return true;
                }
                break;
            }

            float nextDepth =
                sampleHeightDepthNearest(tex, nextTexel, atlasTexelMin, atlasTexelMax, edgeUV, maxDepth);
            if (enteredTexelWall(surfaceDepth, nextDepth, edgeDepth)) {
                enteredSideWall = true;
                enteredWallNormal = wallNormal(true, false, rateUV, dPdu, dPdv, baseNormal);
            } else if (shouldContinueTexelTop(surfaceDepth, nextDepth, edgeDepth)) {
                enteredTop = true;
            }
        }

        if (!processU) {
            nextTexel = texel + ivec2(0, rateUV.y > 0.0 ? 1 : -1);
            if (any(lessThan(nextTexel, atlasTexelMin)) || any(greaterThan(nextTexel, atlasTexelMax))) {
                vec3 edgeNormal;
                if (isEdge(edgeDepth, surfaceDepth, minEdgeDepth, edgeUV, boundsMin, boundsMax, rateUV, dPdu, dPdv,
                           baseNormal, edgeNormal)) {
                    hit.hit = true;
                    hit.sideWall = true;
                    hit.edgeWall = true;
                    hit.t = tCurrent;
                    hit.uv = clamp(edgeUV, boundsMin, boundsMax);
                    hit.depth = edgeDepth;
                    hit.geometricNormal = edgeNormal;
                    return true;
                }
                break;
            }

            float nextDepth =
                sampleHeightDepthNearest(tex, nextTexel, atlasTexelMin, atlasTexelMax, edgeUV, maxDepth);
            if (enteredTexelWall(surfaceDepth, nextDepth, edgeDepth)) {
                enteredSideWall = true;
                enteredWallNormal = wallNormal(false, true, rateUV, dPdu, dPdv, baseNormal);
            } else if (shouldContinueTexelTop(surfaceDepth, nextDepth, edgeDepth)) {
                enteredTop = true;
            }
        }

        if (enteredSideWall) {
            hit.hit = true;
            hit.sideWall = true;
            hit.edgeWall = false;
            hit.t = tCurrent;
            hit.uv = edgeUV;
            hit.depth = edgeDepth;
            hit.geometricNormal = enteredWallNormal;
            return true;
        }

        if (enteredTop) {
            hit.hit = true;
            hit.sideWall = false;
            hit.edgeWall = false;
            hit.t = tCurrent;
            hit.uv = edgeUV;
            hit.depth = surfaceDepth;
            hit.geometricNormal = baseNormal;
            return true;
        }

        texel = nextTexel;
        uvCurrent = edgeUV;
        depthCurrent = edgeDepth;
        ++step;
    }

    return false;
}

bool traceBilinearHeightMapCapped(sampler2D tex,
                                  vec2 minUV,
                                  vec2 maxUV,
                                  vec2 uv,
                                  float depth,
                                  vec3 worldDir,
                                  vec3 dPdu,
                                  vec3 dPdv,
                                  vec3 baseNormal,
                                  float maxDepth,
                                  int maxTraceSteps,
                                  out HeightMapHit hit) {
    initVanillaParallaxMiss(uv, depth, baseNormal, hit);
    if (maxTraceSteps <= 0 || maxDepth <= heightMapMinWorldDepth) { return false; }

    ivec2 size = textureSize(tex, 0);
    if (size.x <= 0 || size.y <= 0) { return false; }

    vec2 boundsMin = heightMapMinUV(minUV, maxUV, size);
    vec2 boundsMax = heightMapMaxUV(minUV, maxUV, size);
    if (!isUvWithinHeightMap(uv, boundsMin, boundsMax)) { return false; }
    float minEdgeDepth = edgeMinDepth(minUV, maxUV, size, maxDepth);

    vec2 rateUV = directionToRateUv(worldDir, dPdu, dPdv);
    float depthRate = dot(worldDir, -baseNormal);
    float texelTravel = max(abs(rateUV.x) * float(size.x), abs(rateUV.y) * float(size.y));
    float depthTravel = abs(depthRate) / max(maxDepth, heightMapMinWorldDepth) * float(max(size.x, size.y));
    float stepWorld = 0.5 / max(max(texelTravel, depthTravel), 1e-4);

    float tPrev = 0.0;
    vec2 uvPrev = uv;
    float depthPrev = depth;
    float surfacePrev = sampleHeightDepth(tex, uvPrev, minUV, maxUV, 0, 1u, maxDepth);
    float fPrev = depthPrev - surfacePrev;
    if (fPrev >= -heightMapTraceBias) {
        hit.hit = true;
        hit.t = 0.0;
        hit.uv = uvPrev;
        hit.depth = surfacePrev;
        hit.geometricNormal =
            sampleNormal(tex, uvPrev, minUV, maxUV, dPdu, dPdv, baseNormal, 0, 1u, maxDepth, -worldDir);
        return true;
    }

    int maxSteps = min(maxTraceSteps, heightMapBilinearMaxSteps);
    for (int step = 0; step < maxSteps; ++step) {
        float tCurr = tPrev + stepWorld;
        vec2 uvCurr = uv + rateUV * tCurr;
        float depthCurr = depth + depthRate * tCurr;
        if (!isUvWithinHeightMap(uvCurr, boundsMin, boundsMax)) {
            float tEdge = heightMapBoundaryHitT(tPrev, tCurr, uvPrev, uvCurr, boundsMin, boundsMax);
            vec2 uvClamped = clamp(uv + rateUV * tEdge, boundsMin, boundsMax);
            float depthEdge = depth + depthRate * tEdge;
            vec3 edgeNormal;
            float boundaryDepth = sampleHeightDepth(tex, uvClamped, minUV, maxUV, 0, 1u, maxDepth);
            if (isEdge(depthEdge, boundaryDepth, minEdgeDepth, uvClamped, boundsMin, boundsMax, rateUV, dPdu, dPdv,
                       baseNormal, edgeNormal)) {
                hit.hit = true;
                hit.sideWall = true;
                hit.edgeWall = true;
                hit.t = tEdge;
                hit.uv = uvClamped;
                hit.depth = depthEdge;
                hit.geometricNormal = edgeNormal;
                return true;
            }
            break;
        }
        if (depthCurr < -heightMapTraceBias) { break; }

        float surfaceCurr = sampleHeightDepth(tex, uvCurr, minUV, maxUV, 0, 1u, maxDepth);
        float fCurr = depthCurr - surfaceCurr;
        if (fCurr >= -heightMapTraceBias) {
            float lo = tPrev;
            float hi = tCurr;
            for (int refine = 0; refine < heightMapBilinearBinarySteps; ++refine) {
                float mid = 0.5 * (lo + hi);
                vec2 uvMid = uv + rateUV * mid;
                float depthMid = depth + depthRate * mid;
                float surfaceMid = sampleHeightDepth(tex, uvMid, minUV, maxUV, 0, 1u, maxDepth);
                if (depthMid >= surfaceMid) {
                    hi = mid;
                } else {
                    lo = mid;
                }
            }

            float tHit = hi;
            vec2 uvHit = uv + rateUV * tHit;
            float depthHit = sampleHeightDepth(tex, uvHit, minUV, maxUV, 0, 1u, maxDepth);
            hit.hit = true;
            hit.sideWall = false;
            hit.edgeWall = false;
            hit.t = tHit;
            hit.uv = uvHit;
            hit.depth = depthHit;
            hit.geometricNormal =
                sampleNormal(tex, uvHit, minUV, maxUV, dPdu, dPdv, baseNormal, 0, 1u, maxDepth, -worldDir);
            return true;
        }

        tPrev = tCurr;
        uvPrev = uvCurr;
        depthPrev = depthCurr;
        fPrev = fCurr;
    }

    return false;
}

bool traceHeightMapCapped(sampler2D tex,
                          vec2 minUV,
                          vec2 maxUV,
                          vec2 uv,
                          float depth,
                          vec3 worldDir,
                          vec3 dPdu,
                          vec3 dPdv,
                          vec3 baseNormal,
                          float maxDepth,
                          uint samplingMode,
                          int maxTraceSteps,
                          out HeightMapHit hit) {
    if (samplingMode == 0u) {
        return traceNearestHeightMapCapped(tex, minUV, maxUV, uv, depth, worldDir, dPdu, dPdv, baseNormal, maxDepth,
                                           maxTraceSteps, hit);
    }

    return traceBilinearHeightMapCapped(tex, minUV, maxUV, uv, depth, worldDir, dPdu, dPdv, baseNormal, maxDepth,
                                        maxTraceSteps, hit);
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
    ivec2 heightMapSize = textureSize(textures[nonuniformEXT(normalTextureID)], 0);
    if (normalTextureID < 0 || maxDepthWorld <= heightMapMinWorldDepth) { return false; }
    if (heightMapSize.x <= 0 || heightMapSize.y <= 0) { return false; }
    vec2 traceAtlasUvMin = heightMapInnerMinUV(atlasUvMin, atlasUvMax, heightMapSize);
    vec2 traceAtlasUvMax = heightMapInnerMaxUV(atlasUvMin, atlasUvMax, heightMapSize);
    vec2 traceSurfaceUv = surface.edgeWall ? clamp(surface.uv, traceAtlasUvMin, traceAtlasUvMax) : surface.uv;

    initVanillaParallaxMiss(traceSurfaceUv, surface.depth, baseGeoNormal, hit);
    exitUv = traceSurfaceUv;
    exitDepth = surface.depth;
    exitDistance = 0.0;

    float startBias = 2.0 * heightMapTraceBias;
    vec2 rateUV = directionToRateUv(worldDir, dPduWorld, dPdvWorld);
    float depthRate = dot(worldDir, -baseGeoNormal);
    vec2 uv = traceSurfaceUv + rateUV * startBias;
    float depth = surface.depth + depthRate * startBias;
    if (dot(surface.geometricNormal, baseGeoNormal) > 0.5 && depthRate < 0.0) {
        depth = min(depth, surface.depth - startBias);
    }

    bool fallbackToBilinear =
        VPT_PBR_SAMPLING_MODE == 0u && shouldFallbackNearestSecondaryToBilinear(worldDir, baseGeoNormal);
    bool localHit;
    if (depthRate < -1e-6) {
        float maxTraceDistance = max(depth / -depthRate, 0.0);
        if (fallbackToBilinear) {
            localHit =
                traceBilinearHeightMap(textures[nonuniformEXT(normalTextureID)], traceAtlasUvMin, traceAtlasUvMax, uv, depth,
                                       worldDir, dPduWorld, dPdvWorld, baseGeoNormal, maxDepthWorld,
                                       maxTraceDistance, hit);
        } else {
            localHit =
                traceHeightMap(textures[nonuniformEXT(normalTextureID)], traceAtlasUvMin, traceAtlasUvMax, uv, depth, worldDir,
                               dPduWorld, dPdvWorld, baseGeoNormal, maxDepthWorld, maxTraceDistance,
                               VPT_PBR_SAMPLING_MODE, hit);
        }
    } else if (fallbackToBilinear) {
        localHit = traceBilinearHeightMapCapped(textures[nonuniformEXT(normalTextureID)], traceAtlasUvMin, traceAtlasUvMax, uv,
                                                depth, worldDir, dPduWorld, dPdvWorld, baseGeoNormal, maxDepthWorld,
                                                maxTraceSteps, hit);
    } else {
        localHit = traceHeightMapCapped(textures[nonuniformEXT(normalTextureID)], traceAtlasUvMin, traceAtlasUvMax, uv, depth,
                                        worldDir, dPduWorld, dPdvWorld, baseGeoNormal, maxDepthWorld,
                                        VPT_PBR_SAMPLING_MODE, maxTraceSteps, hit);
    }

    if (localHit) { return true; }

    if (depthRate >= -1e-6) { return true; }

    float exitT = max(depth / -depthRate, 0.0);
    exitUv = clamp(uv + rateUV * exitT, traceAtlasUvMin, traceAtlasUvMax);
    exitDepth = 0.0;
    exitDistance = exitT;
    return false;
}

vec3 sampleSurfaceDirectLight(SampledSurface surface,
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
                              bool isFftWaterSurface) {
    if (worldUBO.skyType != 1) { return vec3(0.0); }

    bool isOpaqueSurface = surface.mat.transmission <= EPS;
    vec3 lightDir = celestialSunDirection();
    bool sunVisible = lightDir.y >= 0.0;
    if (!sunVisible) { lightDir = celestialMoonDirection(); }
    float lightKappa = (sunVisible && PHYSICAL_SUN_DISK) ? 46000.0 : 3000.0;
    vec3 sampledLightDir = SampleVMF(mainRay.seed, lightDir, lightKappa);
    float sampledLightNoL = dot(sampledLightDir, surface.geometricNormal);
    if (isOpaqueSurface && sampledLightNoL <= 0.0) { return vec3(0.0); }

    float lightPdf;
    vec3 lightBRDF = SurfaceBSDFEval(surface.mat, viewDir, surface.shadingNormal, sampledLightDir, lightPdf, pc.flags);
    if (lightPdf <= 1e-6) { return vec3(0.0); }

    vec3 shadowOriginPos = surface.worldPos;
    vec3 shadowOriginGeomNormal = surface.geometricNormal;
    if (traceLocalHeight) {
        HeightMapHit shadowSelfHit;
        vec2 exitUv;
        float exitDepth;
        float exitDistance;
        if (traceLocalHeightIntersectionAndExit(normalTextureID, atlasUvMin, atlasUvMax, dPduWorld, dPdvWorld,
                                                baseGeoNormal, maxDepthWorld, surface, sampledLightDir,
                                                VPT_PARALLAX_SECONDARY_MAX_STEPS, shadowSelfHit, exitUv, exitDepth,
                                                exitDistance)) {
            return vec3(0.0);
        }
        if (!surface.edgeWall) {
            shadowOriginPos =
                basePlaneWorldPosAtUv(exitUv, referenceUv, referenceWorldPos, dPduWorld, dPdvWorld) -
                baseGeoNormal * exitDepth;
            shadowOriginGeomNormal = baseGeoNormal;
        }
    }

    shadowRay.radiance = vec3(0.0);
    shadowRay.throughput = vec3(1.0);
    shadowRay.insideBoat = rayInsideBoat(mainRay) ? 1u : 0u;
    shadowRay.hitT = INF_DISTANCE;
    shadowRay.pad0 = 0u;

    vec3 baseWorldPos = shadowOriginPos;
    vec3 exitGeoNormal = shadowOriginGeomNormal;
    vec3 shadowOrigin = baseWorldPos - sampledLightDir * 0.0002;
    if (!surface.edgeWall) {
        vec3 visibilityNormal = dot(sampledLightDir, exitGeoNormal) >= 0.0 ? exitGeoNormal : -exitGeoNormal;
        shadowOrigin = baseWorldPos + visibilityNormal * 0.0002;
    }
    uint shadowMask = WORLD_MASK | PLAYER_MASK;
    if (VPT_CLOUD_MODE != 2u) { shadowMask |= CLOUD_MASK; }
    traceRayEXT(topLevelAS, gl_RayFlagsNoneEXT, shadowMask, 0, 0, 0,
                shadowOrigin, 0.0001, sampledLightDir, 1000.0, 1);
    if (rayBounce(mainRay) == 0u) { mainRay.directLightHitT = shadowRay.hitT; }

    float progress = skyUBO.rainGradient;
    vec3 shadowRadiance =
        vptRecoverZeroShadowMissRadiance(shadowOrigin, shadowRay.radiance, shadowRay.throughput, shadowRay.hitT);
    vec3 lightRadiance = shadowRadiance * mainRay.throughput * lightBRDF;
    if (VPT_CLOUD_MODE == 2u) {
        vec3 absoluteShadowOriginPos = baseWorldPos + vec3(worldUBO.cameraPos.xyz);
        float cloudVisibility = volumetricCloudLightVisibility(absoluteShadowOriginPos, lightDir,
                                                               max(VPT_VOLUMETRIC_CLOUD_LIGHT_STEPS, 1), 0.175);
        lightRadiance *= cloudVisibility;
    }
    return mix(lightRadiance, vec3(0.0), progress);
}

bool loadPreviousScenePos(uint geometryBufferIndex, uint primitiveID, vec3 baryCoords, out vec3 prevScenePos) {
    prevScenePos = vec3(0.0);
    uint64_t lastIndexBufferAddr = lastIndexBufferAddrs.addrs[geometryBufferIndex];
    uint64_t lastVertexBufferAddr = lastVertexBufferAddrs.addrs[geometryBufferIndex];
    if (lastIndexBufferAddr == 0 || lastVertexBufferAddr == 0) { return false; }

    IndexBuffer indexBuffer = IndexBuffer(lastIndexBufferAddr);
    uint indexBaseID = 3u * primitiveID;
    uint i0 = indexBuffer.indices[indexBaseID];
    uint i1 = indexBuffer.indices[indexBaseID + 1u];
    uint i2 = indexBuffer.indices[indexBaseID + 2u];

    PositionBuffer vertexBuffer = PositionBuffer(lastVertexBufferAddr);
    vec3 p0 = vertexBuffer.vertices[i0].pos;
    vec3 p1 = vertexBuffer.vertices[i1].pos;
    vec3 p2 = vertexBuffer.vertices[i2].pos;
    vec3 prevLocalPos = baryCoords.x * p0 + baryCoords.y * p1 + baryCoords.z * p2;
    uint instanceID = gl_InstanceCustomIndexEXT;
    mat4 lastModelMat = lastObjToWorldMats.mat[instanceID];
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
    bool isBlockGeometry = (packedData & PBR_FLAG_BLOCK_GEOMETRY) != 0u;
    bool isFluidGeometry = (packedData & PBR_FLAG_FLUID_GEOMETRY) != 0u;
    bool isWaterGeometry = (packedData & PBR_FLAG_WATER_GEOMETRY) != 0u;
    bool isLavaGeometry = isFluidGeometry && ((m0.emissiveBlockType & 255u) == 0u);
    bool foldBlockOverlay = isBlockGeometry && ((packedData & PBR_FLAG_OVERLAY_ALPHA_MASK) != 0u);
    uint alphaMode = getAlphaMode(packedData);
    uint coordinate = getCoordinate(packedData);

    vec2 textureUV = vec2(0.0);
    TextureMapEntry textureMap = TextureMapEntry(-1, -1, -1, 0, 0, 0.0, 0.0, 0.0, 0.0);
    vec2 atlasUvMin = vec2(0.0);
    vec2 atlasUvMax = vec2(0.0);
    float lod = 0.0;
    vec3 dposdu = vec3(1.0, 0.0, 0.0);
    vec3 dposdv = vec3(0.0, 1.0, 0.0);
    bool hasHeightMapSurface = false;
    float maxDepthWorld = 0.0;
    vec3 baseGeoNormal =
        normalizeF(vptWorldToObjectVector(cross(p1.pos - p0.pos, p2.pos - p0.pos)), vec3(0.0, 1.0, 0.0));
    bool useVertexNormal = !isBlockGeometry && hasNorm(packedData);
    vec3 vertexGeoNormal = baseGeoNormal;
    if (useVertexNormal) {
        vertexGeoNormal =
            normalizeF(vptWorldToObjectVector(baryCoords.x * m0.norm + baryCoords.y * m1.norm +
                                              baryCoords.z * m2.norm),
                       baseGeoNormal);
        if (dot(vertexGeoNormal, baseViewDir) < 0.0) { vertexGeoNormal = -vertexGeoNormal; }
        baseGeoNormal = vertexGeoNormal;
    }
    vec3 planeGeoNormal = baseGeoNormal;
    vec3 dPduWorld = vptObjectToWorldVector(dposdu);
    vec3 dPdvWorld = vptObjectToWorldVector(dposdv);
    bool isWaterMaterial = false;
    bool hasFftWaterSurface = false;
    bool parallaxEnabled = false;
    bool useRealisticWaterSurface = VPT_WATER_SURFACE_MODE == 1u;

    if (useTexture) {
        textureUV = baryCoords.x * m0.textureUV + baryCoords.y * m1.textureUV + baryCoords.z * m2.textureUV;
        atlasUvMin = isBlockGeometry ? vec2(0.0) : min(m0.textureUV, min(m1.textureUV, m2.textureUV));
        atlasUvMax = isBlockGeometry ? vec2(1.0) : max(m0.textureUV, max(m1.textureUV, m2.textureUV));

        float coneRadiusWorld = mainRay.coneWidth + gl_HitTEXT * mainRay.coneSpread;
        computedposduDv(p0.pos, p1.pos, p2.pos, m0.textureUV, m1.textureUV, m2.textureUV, dposdu, dposdv);
        if (isBlockGeometry) {
            lod = 0.0;
        } else {
            textureMap = mapping.entries[textureID];
            lod = lodWithCone(textures[nonuniformEXT(textureID)], textureUV, coneRadiusWorld, dposdu, dposdv);
        }

        dPduWorld = vptObjectToWorldVector(dposdu);
        dPdvWorld = vptObjectToWorldVector(dposdv);
        planeGeoNormal = normalizeF(cross(dPduWorld, dPdvWorld), baseGeoNormal);
        baseGeoNormal = useVertexNormal ? vertexGeoNormal : planeGeoNormal;
        if (dot(baseGeoNormal, baseViewDir) < 0.0) { baseGeoNormal = -baseGeoNormal; }

        bool labPbrWaterFlag = false;
        bool transmissiveWaterRule = false;
        if (isBlockGeometry) {
            ivec4 flags = fetchBlockFlagLod(textureID, textureUV, lod);
            labPbrWaterFlag = (flags.r & 0x1) > 0;
            transmissiveWaterRule = textureRuleHasTransmission(textureID);
        } else if (textureMap.flag >= 0) {
            ivec4 flags = ivec4(round(sampleTexture(textures[nonuniformEXT(textureMap.flag)], textureUV, ceil(lod), false) * 255.0));
            labPbrWaterFlag = (flags.r & 0x1) > 0;
        }
        isWaterMaterial = isWaterGeometry || (isFluidGeometry && !isLavaGeometry && (labPbrWaterFlag || transmissiveWaterRule));
        hasFftWaterSurface = useRealisticWaterSurface && isWaterMaterial && abs(planeGeoNormal.y) > 0.75;

        if (!isBlockGeometry && !isWaterMaterial && parallaxEnabled && bounce == 0u && textureMap.normal >= 0 &&
            coordinate != 1u) {
            maxDepthWorld = heightMapMaxDepthWorld(atlasUvMin, atlasUvMax, dPduWorld, dPdvWorld);
            bool hasParallaxDepth = maxDepthWorld > heightMapMinWorldDepth;
            bool hasParallaxView = dot(baseViewDir, baseGeoNormal) > VPT_PARALLAX_MIN_VIEW_DOT;
            hasHeightMapSurface = hasParallaxDepth && hasParallaxView;
        }
    } else {
        dPduWorld = vptObjectToWorldVector(dposdu);
        dPdvWorld = vptObjectToWorldVector(dposdv);
    }
    if (dot(baseGeoNormal, baseViewDir) < 0.0) { baseGeoNormal = -baseGeoNormal; }
    vec3 cameraOrigin = vec3(worldUBO.cameraEffectedViewMatInv * vec4(0.0, 0.0, 0.0, 1.0));
    bool traceLocalHeight =
        hasHeightMapSurface && (lod == 0.0 || distance(planeHitWorldPos, cameraOrigin) <= VPT_PARALLAX_CLOSE_DISTANCE) &&
        !hasFftWaterSurface;

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
        if (traceHeightMapCapped(textures[nonuniformEXT(textureMap.normal)], atlasUvMin, atlasUvMax, textureUV, 0.0,
                                 gl_WorldRayDirectionEXT, dPduWorld, dPdvWorld, baseGeoNormal, maxDepthWorld,
                                 VPT_PBR_SAMPLING_MODE, VPT_PARALLAX_PRIMARY_MAX_STEPS, tracedInitialHit)) {
            initialHit = tracedInitialHit;
        }
    }

    vec3 hitWorldPos = planeHitWorldPos + gl_WorldRayDirectionEXT * initialHit.t;
    float actualHitT = gl_HitTEXT + initialHit.t;

    mainRay.hitT = actualHitT;
    mainRay.coneWidth += actualHitT * mainRay.coneSpread;
    mainRay.directLightRadiance = vec3(0.0);
    mainRay.directLightHitT = 0.0;
    mainRay.instanceIndex = instanceID;
    mainRay.geometryIndex = geometryID;
    mainRay.primitiveIndex = gl_PrimitiveID;
    mainRay.baryCoords = baryCoords;
    mainRay.worldPos = hitWorldPos;
    mainRay.albedoEmission = albedoEmission;
    mainRay.emissiveBlockType = m0.emissiveBlockType & 255u;
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
    sampleSurfaceState(useTexture, textureID, isBlockGeometry, textureMap, atlasUvMin, atlasUvMax, initialHit.uv, lod, alphaMode,
                       colorLayerValue, colorLayer, foldBlockOverlay, glint, useOverlay, m0.overlayUV, dPduWorld, dPdvWorld,
                       baseGeoNormal, hasHeightMapSurface, maxDepthWorld,
                       initialHit.hit, initialHit.sideWall, initialHit.edgeWall, initialHit.depth,
                       initialHit.geometricNormal, hitWorldPos, viewDir,
                       isWaterMaterial, hasFftWaterSurface, surface);

    mainRay.normal = surface.shadingNormal;
    rayStoreMaterial(mainRay, surface.albedoValue, surface.mat.f0, surface.mat.roughness, surface.mat.metallic,
                     surface.mat.transmission, surface.mat.ior, surface.mat.emission);
    rayStoreAux(mainRay, hasFftWaterSurface ? vec2(VPT_FFT_WATER_ORIGIN_BIAS, 1.0) : vec2(0.0));

    SampledSurface currentSurface = surface;
    vec3 currentViewDir = viewDir;
    bool storedLobeType = false;
    for (int localBounce = 0; localBounce < 1; ++localBounce) {
        float emissionFactor =
            (bounce == 0u && localBounce == 0) ? VPT_DIRECT_LIGHT_STRENGTH : VPT_INDIRECT_LIGHT_STRENGTH;
        vec3 emissionRadiance = emissionFactor * currentSurface.tint * currentSurface.mat.emission * mainRay.throughput;
        emissionRadiance += currentSurface.tint * albedoEmission * mainRay.throughput;
        mainRay.radiance += emissionRadiance;

        vec3 directLight =
            sampleSurfaceDirectLight(currentSurface, currentViewDir, textureUV, planeHitWorldPos, atlasUvMin, atlasUvMax,
                                     dPduWorld, dPdvWorld, baseGeoNormal, traceLocalHeight,
                                     textureMap.normal, maxDepthWorld, hasFftWaterSurface);
        if (localBounce == 0) { mainRay.directLightRadiance = directLight; }
        mainRay.radiance += directLight;

        bool isOpaqueSurface = currentSurface.mat.transmission <= EPS;
        vec3 sampleDir;
        float pdf;
        uint lobeType;
        vec3 bsdf;
        if (isWaterMaterial && currentSurface.mat.transmission > EPS) {
            vec3 incident = -currentViewDir;
            vec3 waterNormal = currentSurface.shadingNormal;
            vec3 waterUpNormal = waterNormal.y >= 0.0 ? waterNormal : -waterNormal;
            float eta = dot(incident, waterUpNormal) < 0.0 ? (1.0 / currentSurface.mat.ior) : currentSurface.mat.ior;
            float fresnel = clamp(DielectricFresnel(abs(dot(currentViewDir, waterNormal)), eta), 0.0, 1.0);
            vec3 refractionDir = refract(incident, waterNormal, eta);
            bool hasRefraction = dot(refractionDir, refractionDir) > 1e-6;
            bool chooseReflection = !hasRefraction ||
                                    (VPT_TRANSPARENT_SPLIT_MODE != VPT_TRANSPARENT_SPLIT_MODE_DETERMINISTIC &&
                                     rand(mainRay.seed) < fresnel);
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
        } else if (currentSurface.mat.transmission > EPS &&
                   max(currentSurface.mat.roughness,
                       currentSurface.mat.refractionRoughness * currentSurface.mat.transmission) < 0.001) {
            vec3 incident = -currentViewDir;
            vec3 dielectricNormal = currentSurface.shadingNormal;
            bool entering = dot(incident, dielectricNormal) < 0.0;
            vec3 faceNormal = entering ? dielectricNormal : -dielectricNormal;
            float eta = entering ? (1.0 / currentSurface.mat.ior) : currentSurface.mat.ior;
            float fresnel = clamp(DielectricFresnel(abs(dot(currentViewDir, faceNormal)), eta), 0.0, 1.0);
            vec3 refractionDir = refract(incident, faceNormal, eta);
            bool hasRefraction = dot(refractionDir, refractionDir) > 1e-6;
            bool chooseReflection = !hasRefraction ||
                                    (VPT_TRANSPARENT_SPLIT_MODE != VPT_TRANSPARENT_SPLIT_MODE_DETERMINISTIC &&
                                     rand(mainRay.seed) < fresnel);
            if (chooseReflection) {
                sampleDir = normalize(reflect(incident, faceNormal));
                pdf = max(fresnel, 1e-4);
                lobeType = 1u;
                bsdf = vec3(fresnel);
            } else {
                sampleDir = normalize(refractionDir);
                pdf = max(1.0 - fresnel, 1e-4);
                lobeType = 2u;
                bool thinGlass = (currentSurface.mat.materialModeFlags & 0x3u) == 1u;
                float absorptionPath = thinGlass ? currentSurface.mat.thickness
                    : currentSurface.mat.thickness * max(currentSurface.mat.absorptionDistance, 0.01);
                vec3 beerTint = exp(-max(currentSurface.mat.absorption, vec3(0.0)) * absorptionPath);
                bsdf = pow(max(currentSurface.albedoValue.rgb, vec3(0.0)), vec3(0.5)) * beerTint * (1.0 - fresnel);
            }
        } else {
            bsdf = SurfaceBSDFSample(currentSurface.mat, currentViewDir, currentSurface.shadingNormal, sampleDir, pdf,
                                     mainRay.seed, lobeType, pc.flags);
        }

        if (!storedLobeType) {
            raySetLobeType(mainRay, lobeType);
            raySetIndirectVolumetricCloud(mainRay,
                                          vptShouldCaptureIndirectVolumetricCloud(lobeType, currentSurface.mat.roughness));
            storedLobeType = true;
        }
        raySetNoisy(mainRay, true);

        if (pdf <= 1e-6 || max(bsdf.r, max(bsdf.g, bsdf.b)) <= 1e-6) {
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
                mainRay.origin = currentSurface.worldPos + exitNormal * VPT_FFT_WATER_ORIGIN_BIAS;
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
            currentSurface, sampleDir, VPT_PARALLAX_SECONDARY_MAX_STEPS, localBounceHit, exitUv, exitDepth,
            exitDistance);
        if (!localBlocked) {
            vec3 exitBasePos =
                basePlaneWorldPosAtUv(exitUv, textureUV, planeHitWorldPos, dPduWorld, dPdvWorld) -
                baseGeoNormal * exitDepth;
            vec3 exitGeoNormal =
                localBounceHit.sideWall && localBounceHit.edgeWall ? localBounceHit.geometricNormal : baseGeoNormal;
            vec3 exitNormal = dot(sampleDir, exitGeoNormal) >= 0.0 ? exitGeoNormal : -exitGeoNormal;
            mainRay.origin = exitBasePos + exitNormal * 0.0002;
            mainRay.direction = sampleDir;
            raySetStop(mainRay, false);
            return;
        }

        vec3 nextWorldPos = currentSurface.worldPos + sampleDir * localBounceHit.t;
        sampleSurfaceState(useTexture, textureID, isBlockGeometry, textureMap, atlasUvMin, atlasUvMax, localBounceHit.uv,
                           lod, alphaMode, colorLayerValue, colorLayer, foldBlockOverlay, glint, useOverlay, m0.overlayUV, dPduWorld,
                           dPdvWorld, baseGeoNormal, hasHeightMapSurface, maxDepthWorld,
                           localBounceHit.hit, localBounceHit.sideWall, localBounceHit.edgeWall,
                           localBounceHit.depth, localBounceHit.geometricNormal, nextWorldPos,
                           -sampleDir, isWaterMaterial, hasFftWaterSurface, currentSurface);
        currentViewDir = -sampleDir;
    }

    raySetStop(mainRay, true);
}
