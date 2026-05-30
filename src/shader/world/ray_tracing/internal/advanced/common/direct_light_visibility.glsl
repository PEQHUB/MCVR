#ifndef ADV_DIRECT_LIGHT_VISIBILITY_GLSL
#define ADV_DIRECT_LIGHT_VISIBILITY_GLSL

#ifndef ADV_AREA_LIGHT_SURFACE_ORIGIN_VALIDATION
#    define ADV_AREA_LIGHT_SURFACE_ORIGIN_VALIDATION 0
#endif
#ifndef ADV_AREA_LIGHT_VISIBILITY_DISTANCE_SHRINK
#    define ADV_AREA_LIGHT_VISIBILITY_DISTANCE_SHRINK 0.002
#endif

vec3 traceAreaLightReservoirTransmission(SampledSurface surface,
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
                                         vec3 sampledPoint,
                                         vec3 sampledNormal,
                                         bool insideBoat) {
    if (!isValidSampledSurface(surface) || !isFiniteVec3(sampledPoint) || !isFiniteVec3(sampledNormal)) {
        return vec3(0.0);
    }

    vec3 sampledPointScene = areaLightWorldToScene(sampledPoint);
    vec3 lightVector = sampledPointScene - surface.worldPos;
    float distance2 = dot(lightVector, lightVector);
    if (!isFiniteFloat(distance2) || distance2 <= 1e-8) { return vec3(0.0); }

    float distanceToLight = sqrt(distance2);
    vec3 sampledLightDir = lightVector / distanceToLight;
    if (!isFiniteVec3(sampledLightDir)) { return vec3(0.0); }
    bool isOpaqueSurface = surface.mat.transmission <= EPS;
    if (isOpaqueSurface && dot(sampledLightDir, surface.geometricNormal) <= 0.0) { return vec3(0.0); }
    if (dot(sampledNormal, -sampledLightDir) <= 1e-6) { return vec3(0.0); }

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
    shadowRay.insideBoat = insideBoat ? 1u : 0u;
    shadowRay.pad0 = 1u;

    vec3 shadowOrigin = shadowOriginPos - sampledLightDir * 0.0002;
    if (!surface.edgeWall) {
        vec3 visibilityNormal = dot(sampledLightDir, shadowOriginGeomNormal) >= 0.0 ? shadowOriginGeomNormal :
                                                                                  -shadowOriginGeomNormal;
        shadowOrigin = shadowOriginPos + visibilityNormal * 0.0002;
    }

    float shadowLength = max(distanceToLight - shadowOriginDistance - ADV_AREA_LIGHT_VISIBILITY_DISTANCE_SHRINK, 0.0001);
    if (!isFiniteVec3(shadowOrigin) || !isFiniteFloat(shadowLength)) { return vec3(0.0); }
    traceRayEXT(topLevelAS, gl_RayFlagsNoneEXT, WORLD_MASK | PLAYER_MASK | CLOUD_MASK, 0, 0, 0, shadowOrigin, 0.0001,
                sampledLightDir, shadowLength, 1);
    return shadowRay.radiance;
}

#endif
