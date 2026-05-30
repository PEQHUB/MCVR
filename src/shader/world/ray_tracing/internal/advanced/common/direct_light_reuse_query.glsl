#include "common/constants.glsl"
#ifndef ADV_DIRECT_LIGHT_REUSE_QUERY_GLSL
#define ADV_DIRECT_LIGHT_REUSE_QUERY_GLSL

#ifndef DIRECT_LIGHT_REUSE_QUERY_LOAD_IMAGE1
#    define DIRECT_LIGHT_REUSE_QUERY_LOAD_IMAGE1 directLightReuseQuery1Image
#endif
#ifndef DIRECT_LIGHT_REUSE_QUERY_LOAD_IMAGE2
#    define DIRECT_LIGHT_REUSE_QUERY_LOAD_IMAGE2 directLightReuseQuery2Image
#endif
#ifndef DIRECT_LIGHT_REUSE_QUERY_LOAD_IMAGE3
#    define DIRECT_LIGHT_REUSE_QUERY_LOAD_IMAGE3 directLightReuseQuery3Image
#endif
#ifndef DIRECT_LIGHT_REUSE_QUERY_LOAD_IMAGE4
#    define DIRECT_LIGHT_REUSE_QUERY_LOAD_IMAGE4 directLightReuseQuery4Image
#endif
#ifndef DIRECT_LIGHT_REUSE_QUERY_LOAD_IMAGE5
#    define DIRECT_LIGHT_REUSE_QUERY_LOAD_IMAGE5 directLightReuseQuery5Image
#endif
#ifndef DIRECT_LIGHT_REUSE_QUERY_LOAD_IMAGE6
#    define DIRECT_LIGHT_REUSE_QUERY_LOAD_IMAGE6 directLightReuseQuery6Image
#endif
#ifndef DIRECT_LIGHT_REUSE_QUERY_LOAD_META_IMAGE
#    define DIRECT_LIGHT_REUSE_QUERY_LOAD_META_IMAGE directLightReuseQueryMetaImage
#endif

#ifndef DIRECT_LIGHT_REUSE_QUERY_STORE_IMAGE1
#    define DIRECT_LIGHT_REUSE_QUERY_STORE_IMAGE1 DIRECT_LIGHT_REUSE_QUERY_LOAD_IMAGE1
#endif
#ifndef DIRECT_LIGHT_REUSE_QUERY_STORE_IMAGE2
#    define DIRECT_LIGHT_REUSE_QUERY_STORE_IMAGE2 DIRECT_LIGHT_REUSE_QUERY_LOAD_IMAGE2
#endif
#ifndef DIRECT_LIGHT_REUSE_QUERY_STORE_IMAGE3
#    define DIRECT_LIGHT_REUSE_QUERY_STORE_IMAGE3 DIRECT_LIGHT_REUSE_QUERY_LOAD_IMAGE3
#endif
#ifndef DIRECT_LIGHT_REUSE_QUERY_STORE_IMAGE4
#    define DIRECT_LIGHT_REUSE_QUERY_STORE_IMAGE4 DIRECT_LIGHT_REUSE_QUERY_LOAD_IMAGE4
#endif
#ifndef DIRECT_LIGHT_REUSE_QUERY_STORE_IMAGE5
#    define DIRECT_LIGHT_REUSE_QUERY_STORE_IMAGE5 DIRECT_LIGHT_REUSE_QUERY_LOAD_IMAGE5
#endif
#ifndef DIRECT_LIGHT_REUSE_QUERY_STORE_IMAGE6
#    define DIRECT_LIGHT_REUSE_QUERY_STORE_IMAGE6 DIRECT_LIGHT_REUSE_QUERY_LOAD_IMAGE6
#endif
#ifndef DIRECT_LIGHT_REUSE_QUERY_STORE_META_IMAGE
#    define DIRECT_LIGHT_REUSE_QUERY_STORE_META_IMAGE DIRECT_LIGHT_REUSE_QUERY_LOAD_META_IMAGE
#endif

#ifndef DIRECT_LIGHT_REUSE_QUERY_LOAD_RAW1
#    define DIRECT_LIGHT_REUSE_QUERY_LOAD_RAW1(pixel) imageLoad(DIRECT_LIGHT_REUSE_QUERY_LOAD_IMAGE1, pixel)
#endif
#ifndef DIRECT_LIGHT_REUSE_QUERY_LOAD_RAW2
#    define DIRECT_LIGHT_REUSE_QUERY_LOAD_RAW2(pixel) imageLoad(DIRECT_LIGHT_REUSE_QUERY_LOAD_IMAGE2, pixel)
#endif
#ifndef DIRECT_LIGHT_REUSE_QUERY_LOAD_RAW3
#    define DIRECT_LIGHT_REUSE_QUERY_LOAD_RAW3(pixel) imageLoad(DIRECT_LIGHT_REUSE_QUERY_LOAD_IMAGE3, pixel)
#endif
#ifndef DIRECT_LIGHT_REUSE_QUERY_LOAD_RAW4
#    define DIRECT_LIGHT_REUSE_QUERY_LOAD_RAW4(pixel) imageLoad(DIRECT_LIGHT_REUSE_QUERY_LOAD_IMAGE4, pixel)
#endif
#ifndef DIRECT_LIGHT_REUSE_QUERY_LOAD_RAW5
#    define DIRECT_LIGHT_REUSE_QUERY_LOAD_RAW5(pixel) imageLoad(DIRECT_LIGHT_REUSE_QUERY_LOAD_IMAGE5, pixel)
#endif
#ifndef DIRECT_LIGHT_REUSE_QUERY_LOAD_RAW6
#    define DIRECT_LIGHT_REUSE_QUERY_LOAD_RAW6(pixel) imageLoad(DIRECT_LIGHT_REUSE_QUERY_LOAD_IMAGE6, pixel)
#endif
#ifndef DIRECT_LIGHT_REUSE_QUERY_LOAD_META_RAW
#    define DIRECT_LIGHT_REUSE_QUERY_LOAD_META_RAW(pixel) imageLoad(DIRECT_LIGHT_REUSE_QUERY_LOAD_META_IMAGE, pixel)
#endif

#ifndef DIRECT_LIGHT_REUSE_QUERY_STORE_RAW1
#    define DIRECT_LIGHT_REUSE_QUERY_STORE_RAW1(pixel, value) imageStore(DIRECT_LIGHT_REUSE_QUERY_STORE_IMAGE1, pixel, value)
#endif
#ifndef DIRECT_LIGHT_REUSE_QUERY_STORE_RAW2
#    define DIRECT_LIGHT_REUSE_QUERY_STORE_RAW2(pixel, value) imageStore(DIRECT_LIGHT_REUSE_QUERY_STORE_IMAGE2, pixel, value)
#endif
#ifndef DIRECT_LIGHT_REUSE_QUERY_STORE_RAW3
#    define DIRECT_LIGHT_REUSE_QUERY_STORE_RAW3(pixel, value) imageStore(DIRECT_LIGHT_REUSE_QUERY_STORE_IMAGE3, pixel, value)
#endif
#ifndef DIRECT_LIGHT_REUSE_QUERY_STORE_RAW4
#    define DIRECT_LIGHT_REUSE_QUERY_STORE_RAW4(pixel, value) imageStore(DIRECT_LIGHT_REUSE_QUERY_STORE_IMAGE4, pixel, value)
#endif
#ifndef DIRECT_LIGHT_REUSE_QUERY_STORE_RAW5
#    define DIRECT_LIGHT_REUSE_QUERY_STORE_RAW5(pixel, value) imageStore(DIRECT_LIGHT_REUSE_QUERY_STORE_IMAGE5, pixel, value)
#endif
#ifndef DIRECT_LIGHT_REUSE_QUERY_STORE_RAW6
#    define DIRECT_LIGHT_REUSE_QUERY_STORE_RAW6(pixel, value) imageStore(DIRECT_LIGHT_REUSE_QUERY_STORE_IMAGE6, pixel, value)
#endif
#ifndef DIRECT_LIGHT_REUSE_QUERY_STORE_META_RAW
#    define DIRECT_LIGHT_REUSE_QUERY_STORE_META_RAW(pixel, value) imageStore(DIRECT_LIGHT_REUSE_QUERY_STORE_META_IMAGE, pixel, value)
#endif

struct DirectLightReuseQuery {
    vec3 worldPos;
    float depth;
    vec3 geometricNormal;
    vec2 uv;
    bool edgeWall;
    vec3 shadingNormal;
    vec3 viewDir;
    vec3 albedo;
    float roughness;
    vec3 f0;
    float metallic;
    float subSurface;
    float transmission;
    float ior;
    float maxDepthWorld;
    int normalTextureID;
    bool traceLocalHeight;
    bool valid;
};

DirectLightReuseQuery buildDirectLightReuseQuery(DirectLightPreparedSurface prepared, vec3 viewDir) {
    DirectLightReuseQuery query;
    query.worldPos = vec3(0.0);
    query.depth = 0.0;
    query.geometricNormal = vec3(0.0, 1.0, 0.0);
    query.uv = vec2(0.0);
    query.edgeWall = false;
    query.shadingNormal = vec3(0.0, 1.0, 0.0);
    query.viewDir = vec3(0.0, 0.0, 1.0);
    query.albedo = vec3(0.0);
    query.roughness = 1.0;
    query.f0 = vec3(0.04);
    query.metallic = 0.0;
    query.subSurface = 0.0;
    query.transmission = 0.0;
    query.ior = 1.5;
    query.maxDepthWorld = 0.0;
    query.normalTextureID = -1;
    query.traceLocalHeight = false;
    query.valid = false;
    query.worldPos = prepared.surface.worldPos;
    query.depth = prepared.surface.depth;
    query.geometricNormal = prepared.surface.geometricNormal;
    query.uv = prepared.surface.uv;
    query.edgeWall = prepared.surface.edgeWall;
    query.shadingNormal = prepared.surface.shadingNormal;
    query.viewDir = viewDir;
    query.albedo = prepared.surface.mat.albedo;
    query.roughness = prepared.surface.mat.roughness;
    query.f0 = prepared.surface.mat.f0;
    query.metallic = prepared.surface.mat.metallic;
    query.subSurface = prepared.surface.mat.subSurface;
    query.transmission = prepared.surface.mat.transmission;
    query.ior = prepared.surface.mat.ior;
    query.maxDepthWorld = prepared.maxDepthWorld;
    query.normalTextureID = prepared.normalTextureID;
    query.traceLocalHeight = prepared.traceLocalHeight;
    query.valid = directLightPreparedSurfaceEligible(prepared) && isFiniteVec3(viewDir);
    return query;
}

SampledSurface sampledSurfaceFromReuseQuery(DirectLightReuseQuery query) {
    SampledSurface surface;
    surface.uv = query.uv;
    surface.depth = query.depth;
    surface.edgeWall = query.edgeWall;
    surface.worldPos = query.worldPos;
    surface.geometricNormal = query.geometricNormal;
    surface.shadingNormal = query.shadingNormal;
    surface.albedoValue = vec4(0.0);
    surface.specularValue = vec4(0.0);
    surface.normalValue = vec4(0.0);
    surface.tint = vec3(0.0);

    surface.mat.albedo = query.albedo;
    surface.mat.f0 = query.f0;
    surface.mat.roughness = query.roughness;
    surface.mat.metallic = query.metallic;
    surface.mat.subSurface = query.subSurface;
    surface.mat.transmission = query.transmission;
    surface.mat.ior = query.ior;
    surface.mat.emission = 0.0;
    surface.mat.normal = vec3(0.0, 0.0, 1.0);
    surface.mat.ao = 1.0;
    surface.mat.height = 0.0;
    return surface;
}

float directLightWorldPerPixel(float viewDistance, ivec2 resolution) {
    float fovY = fovYFromProj(worldUBO.cameraProjMat);
    return 2.0 * tan(0.5 * fovY) * max(viewDistance, 1e-3) / float(max(resolution.y, 1));
}

#ifdef ADV_DIRECT_LIGHT_RESERVOIR_GLSL
float evaluateReservoirTargetAtQuery(AreaLightReservoir reservoir,
                                     DirectLightReuseQuery query,
                                     PrimarySurfaceCache cache) {
    if (!(reservoir.valid > 0.5) || !query.valid) { return 0.0; }

    SampledSurface surface = sampledSurfaceFromReuseQuery(query);
    float targetFunction = 0.0;
    evaluateAreaLightSampleContribution(surface, query.viewDir, cache.textureUV, cache.planeHitWorldPos,
                                        cache.atlasUvMin, cache.atlasUvMax, cache.dPduWorld, cache.dPdvWorld,
                                        cache.baseGeoNormal, query.traceLocalHeight, query.normalTextureID,
                                        query.maxDepthWorld, reservoir.emission, reservoir.point, reservoir.normal, true,
                                        targetFunction);
    return targetFunction;
}
#endif

#endif
