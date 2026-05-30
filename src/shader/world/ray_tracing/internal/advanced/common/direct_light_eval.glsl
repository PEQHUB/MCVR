#include "common/celestial.glsl"

#ifndef ADV_INDIRECT_LIGHT_STRENGTH
#    define ADV_INDIRECT_LIGHT_STRENGTH 1.0
#endif

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
                                   bool isFftWaterSurface) {
    if (worldUBO.skyType != 1) { return vec3(0.0); }
    if (!isValidSampledSurface(surface)) { return vec3(0.0); }

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
    vec2 rayAux = rayLoadAux(mainRay);
    bool applyUnderwaterCaustic = !isFftWaterSurface && (skyUBO.cameraSubmersionType == 1 || rayAux.y > 0.5);
    if (applyUnderwaterCaustic) {
        vec3 absWorldPos = surface.worldPos + vec3(worldUBO.cameraPos.xyz);
        float waterDepth = distance(surface.worldPos, mainRay.origin);
        lightRadiance *= sampleFftWaterCaustic(absWorldPos.xz, worldUBO.gameTime, sampledLightDir, waterDepth);
    }
    return mix(lightRadiance, vec3(0.0), progress);
}

vec3 shadeStoredAreaLightReservoir(SampledSurface surface,
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
                                   AreaLightReservoir reservoir) {
    if (!(reservoir.valid > 0.5) || reservoir.contributionWeight <= 1e-8) { return vec3(0.0); }

    vec3 transmission = traceAreaLightReservoirTransmission(
        surface, referenceUv, referenceWorldPos, atlasUvMin, atlasUvMax, dPduWorld, dPdvWorld, baseGeoNormal,
        traceLocalHeight, normalTextureID, maxDepthWorld, reservoir.point, reservoir.normal, rayInsideBoat(mainRay));
    if (max(max(transmission.x, transmission.y), transmission.z) <= 1e-8) { return vec3(0.0); }

    float ignoredTargetFunction = 0.0;
    vec3 contribution = evaluateAreaLightSampleContribution(
        surface, viewDir, referenceUv, referenceWorldPos, atlasUvMin, atlasUvMax, dPduWorld, dPdvWorld, baseGeoNormal,
        false, -1, 0.0, reservoir.emission, reservoir.point, reservoir.normal, false,
        ignoredTargetFunction);
    return contribution * transmission * reservoir.contributionWeight;
}
