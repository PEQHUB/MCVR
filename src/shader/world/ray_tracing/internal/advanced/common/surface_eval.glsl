#include "common/parallax_trace.glsl"
#include "common/constants.glsl"
#include "common/parallax_condition.glsl"

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

bool isFlaggedWaterSurface(int flagTextureID, vec2 uv, float lod) {
    if (flagTextureID < 0) { return false; }
    ivec4 flags = ivec4(round(sampleTexture(textures[nonuniformEXT(flagTextureID)], uv, ceil(lod), false) * 255.0));
    return (flags.r & 0x1) > 0;
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

    initRestirParallaxMiss(traceSurfaceUv, surface.depth, baseGeoNormal, hit);
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
