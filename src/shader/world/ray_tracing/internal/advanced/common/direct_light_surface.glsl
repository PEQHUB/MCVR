#include "common/constants.glsl"
#include "common/parallax_condition.glsl"
#ifndef ADV_DIRECT_LIGHT_SURFACE_GLSL
#define ADV_DIRECT_LIGHT_SURFACE_GLSL

#ifndef ADV_EVALUATE_HEIGHT_MAP
#    define ADV_EVALUATE_HEIGHT_MAP 0
#endif

bool hasColorLayer(uint packedData) {
    return (packedData & USE_COLOR_LAYER_BIT) != 0u;
}

bool hasTexture(uint packedData) {
    return (packedData & USE_TEXTURE_BIT) != 0u;
}

bool hasOverlay(uint packedData) {
    return (packedData & USE_OVERLAY_BIT) != 0u;
}

bool hasGlint(uint packedData) {
    return (packedData & USE_GLINT_BIT) != 0u;
}

bool hasNorm(uint packedData) {
    return (packedData & USE_NORM_BIT) != 0u;
}

bool hasLight(uint packedData) {
    return (packedData & USE_LIGHT_BIT) != 0u;
}

bool hasNoHeightSurface(uint packedData) {
    return (packedData & NO_HEIGHT_SURFACE_BIT) != 0u;
}

uint getAlphaMode(uint packedData) {
    return (packedData >> ALPHA_MODE_SHIFT) & 0xFu;
}

uint getCoordinate(uint packedData) {
    return (packedData >> COORDINATE_SHIFT) & 0xFu;
}

bool isFiniteVec4(vec4 value) {
    return !any(isnan(value)) && !any(isinf(value));
}

struct PrimarySurfaceCache {
    bool valid;
    vec3 planeHitWorldPos;
    vec3 dPduWorld;
    vec3 dPdvWorld;
    vec3 baseGeoNormal;
    vec2 textureUV;
    vec2 atlasUvMin;
    vec2 atlasUvMax;
    uint textureID;
    uint packedData;
    vec4 colorLayerValue;
    vec3 glint;
    ivec2 overlayUV;
};

struct DirectLightCameraRay {
    vec3 origin;
    vec3 direction;
    vec3 viewDir;
    float coneSpread;
};

struct DirectLightPreparedSurface {
    SampledSurface surface;
    vec2 referenceUv;
    vec3 referenceWorldPos;
    vec2 atlasUvMin;
    vec2 atlasUvMax;
    vec3 dPduWorld;
    vec3 dPdvWorld;
    vec3 baseGeoNormal;
    bool traceLocalHeight;
    int normalTextureID;
    float maxDepthWorld;
    bool isFftWaterSurface;
};

bool directLightSurfaceEligible(SampledSurface surface) {
    return isValidSampledSurface(surface) &&
           surface.mat.emission <= ADV_DIRECT_LIGHT_EMISSIVE_SURFACE_EPSILON &&
           surface.mat.roughness >= ADV_DIRECT_LIGHT_MIN_SURFACE_ROUGHNESS;
}

bool directLightPreparedSurfaceEligible(DirectLightPreparedSurface prepared) {
    return directLightSurfaceEligible(prepared.surface);
}

PrimarySurfaceCache decodeSurfaceCache(vec4 cache0,
                                       vec4 cache1,
                                       vec4 cache2,
                                       vec4 cache3,
                                       vec4 cache4,
                                       vec4 cache5,
                                       vec4 cache6,
                                       vec4 cache7) {
    PrimarySurfaceCache cache;
    cache.valid = isFiniteVec4(cache0) && isFiniteVec4(cache1) && isFiniteVec4(cache2) && isFiniteVec4(cache3) &&
                  isFiniteVec4(cache4) && isFiniteVec4(cache5) && isFiniteVec4(cache6) && isFiniteVec4(cache7) &&
                  cache7.w == ADV_SURFACE_CACHE_VALID_FLAG;
    cache.planeHitWorldPos = cache0.xyz;
    cache.dPduWorld = cache1.xyz;
    cache.dPdvWorld = cache2.xyz;
    cache.baseGeoNormal = normalizeF(cache3.xyz, vec3(0.0, 1.0, 0.0));
    cache.textureUV = vec2(cache0.w, cache1.w);
    cache.atlasUvMin = vec2(cache2.w, cache3.w);
    cache.atlasUvMax = cache4.xy;
    cache.textureID = decodeFirstHitUint(cache4.zw);
    cache.packedData = decodeFirstHitUint(cache7.yz);
    cache.colorLayerValue = cache5;
    cache.glint = cache6.xyz;
    cache.overlayUV = ivec2(int(round(cache6.w)), int(round(cache7.x)));
    return cache;
}

#ifdef ADV_DIRECT_LIGHT_SURFACE_HAS_SECONDARY_CACHE
PrimarySurfaceCache loadSecondarySurfaceCache(ivec2 pixel) {
    return decodeSurfaceCache(
        loadSecondarySurfaceCacheLayer(pixel, 0), loadSecondarySurfaceCacheLayer(pixel, 1),
        loadSecondarySurfaceCacheLayer(pixel, 2), loadSecondarySurfaceCacheLayer(pixel, 3),
        loadSecondarySurfaceCacheLayer(pixel, 4), loadSecondarySurfaceCacheLayer(pixel, 5),
        loadSecondarySurfaceCacheLayer(pixel, 6), loadSecondarySurfaceCacheLayer(pixel, 7));
}
#endif

DirectLightCameraRay buildPrimaryViewDirectLightCameraRay(ivec2 pixel, vec2 resolution) {
    DirectLightCameraRay cameraRay;

    vec2 pixelCenter = vec2(pixel) + 0.5;
    pixelCenter += worldUBO.cameraJitter;

    float fovY = fovYFromProj(worldUBO.cameraProjMat);
    float fovX = fovXFromProj(worldUBO.cameraProjMat);
    cameraRay.coneSpread = coneSpreadFromFov(fovY, fovX, resolution);

    vec2 ndc = pixelCenter / resolution * 2.0 - 1.0;
    vec4 nearPoint = vec4(ndc, 0.0, 1.0);
    vec4 viewNear = worldUBO.cameraProjMatInv * nearPoint;
    viewNear /= viewNear.w;

    cameraRay.origin = vec3(worldUBO.cameraEffectedViewMatInv * vec4(0.0, 0.0, 0.0, 1.0));
    cameraRay.direction = normalize(vec3(worldUBO.cameraEffectedViewMatInv * vec4(viewNear.xyz, 0.0)));
    cameraRay.viewDir = -cameraRay.direction;
    return cameraRay;
}

DirectLightCameraRay buildStoredDirectLightCameraRay(vec3 origin, vec3 direction, float coneSpread) {
    DirectLightCameraRay cameraRay;
    cameraRay.origin = origin;
    cameraRay.direction = normalizeF(direction, vec3(0.0, 0.0, -1.0));
    cameraRay.viewDir = -cameraRay.direction;
    cameraRay.coneSpread = coneSpread;
    return cameraRay;
}

void prepareDirectLightSurface(PrimarySurfaceCache cache,
                               DirectLightCameraRay cameraRay,
                               out DirectLightPreparedSurface prepared) {
    uint packedData = cache.packedData;
    bool useColorLayer = hasColorLayer(packedData);
    bool useTexture = hasTexture(packedData);
    bool useGlint = hasGlint(packedData);
    bool useOverlay = hasOverlay(packedData);
    uint alphaMode = getAlphaMode(packedData);
    uint coordinate = getCoordinate(packedData);

    vec4 colorLayerValue = useColorLayer ? cache.colorLayerValue : vec4(1.0);
    vec3 colorLayer = colorLayerValue.rgb;

    uint textureID = cache.textureID;
    TextureMapEntry textureMap = TextureMapEntry(-1, -1, -1);
    vec2 textureUV = cache.textureUV;
    vec2 atlasUvMin = cache.atlasUvMin;
    vec2 atlasUvMax = cache.atlasUvMax;
    float lod = 0.0;
    bool hasHeightMapSurface = false;
    float maxDepthWorld = 0.0;
    bool hasFftWaterSurface = false;

    if (useTexture) {
        textureMap = mapping.entries[textureID];
        float coneRadiusWorld = distance(cache.planeHitWorldPos, cameraRay.origin) * cameraRay.coneSpread;
        lod = lodWithCone(textures[nonuniformEXT(textureID)], textureUV, coneRadiusWorld, cache.dPduWorld,
                          cache.dPdvWorld);

        bool isWaterMaterial = ADV_WATER_SURFACE_MODE == 1u && isFlaggedWaterSurface(textureMap.flag, textureUV, lod);
        hasFftWaterSurface = isWaterMaterial && abs(cache.baseGeoNormal.y) > 0.75;

        if (ADV_EVALUATE_HEIGHT_MAP != 0 && ADV_ENABLE_PARALLAX != 0 && !hasNoHeightSurface(packedData) &&
            textureMap.normal >= 0 &&
            coordinate != 1u) {
            maxDepthWorld = heightMapMaxDepthWorld(atlasUvMin, atlasUvMax, cache.dPduWorld, cache.dPdvWorld);
            hasHeightMapSurface = maxDepthWorld > heightMapMinWorldDepth &&
                                  dot(cameraRay.viewDir, cache.baseGeoNormal) > ADV_PARALLAX_MIN_VIEW_DOT &&
                                  !hasFftWaterSurface;
        }
    }

    HeightMapHit initialHit;
    initialHit.hit = false;
    initialHit.sideWall = false;
    initialHit.edgeWall = false;
    initialHit.t = 0.0;
    initialHit.uv = textureUV;
    initialHit.depth = 0.0;
    initialHit.geometricNormal = cache.baseGeoNormal;

    bool traceLocalHeight = ADV_EVALUATE_HEIGHT_MAP != 0 && hasHeightMapSurface &&
                            shouldTraceRestirParallax(lod, cache.planeHitWorldPos);

    if (traceLocalHeight) {
        HeightMapHit tracedInitialHit;
        if (traceRestirHeightMapCapped(textures[nonuniformEXT(textureMap.normal)], atlasUvMin, atlasUvMax, textureUV,
                                       0.0, cameraRay.direction, cache.dPduWorld, cache.dPdvWorld,
                                       cache.baseGeoNormal, maxDepthWorld, ADV_PBR_SAMPLING_MODE,
                                       ADV_PARALLAX_PRIMARY_MAX_STEPS, tracedInitialHit)) {
            initialHit = tracedInitialHit;
        }
    }

    vec3 hitWorldPos = cache.planeHitWorldPos + cameraRay.direction * initialHit.t;
    vec3 glint = useGlint ? cache.glint : vec3(0.0);

    sampleSurfaceState(useTexture, textureID, textureMap, atlasUvMin, atlasUvMax, initialHit.uv, lod, alphaMode,
                       colorLayerValue, colorLayer, glint, useOverlay, cache.overlayUV, cache.dPduWorld,
                       cache.dPdvWorld, cache.baseGeoNormal, hasHeightMapSurface, maxDepthWorld, initialHit,
                       hitWorldPos, cameraRay.viewDir, hasFftWaterSurface, prepared.surface);

    prepared.referenceUv = textureUV;
    prepared.referenceWorldPos = cache.planeHitWorldPos;
    prepared.atlasUvMin = atlasUvMin;
    prepared.atlasUvMax = atlasUvMax;
    prepared.dPduWorld = cache.dPduWorld;
    prepared.dPdvWorld = cache.dPdvWorld;
    prepared.baseGeoNormal = cache.baseGeoNormal;
    prepared.traceLocalHeight = traceLocalHeight;
    prepared.normalTextureID = textureMap.normal;
    prepared.maxDepthWorld = maxDepthWorld;
    prepared.isFftWaterSurface = hasFftWaterSurface;
}

#endif
