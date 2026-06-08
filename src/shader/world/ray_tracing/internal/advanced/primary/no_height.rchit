#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#include "common/shared.hpp"
#include "util/disney.glsl"
#include "util/alpha_mode.glsl"
#include "common/fft_water.glsl"
#include "util/height_map.glsl"
#include "util/random.glsl"
#include "util/ray_cone.glsl"
#include "util/ray.glsl"
#include "util/util.glsl"
#include "common/first_hit_state.glsl"
#include "common/runtime_primary_secondary_arrays.glsl"

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

layout(set = 1, binding = 6) readonly buffer LastPositionBufferAddr {
    uint64_t addrs[];
}
lastPositionBufferAddrs;

layout(set = 2, binding = 0) uniform WorldUniform {
    WorldUBO worldUBO;
};

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer IndexBuffer {
    uint indices[];
}
indexBuffer;

#include "util/vertex.glsl"

layout(location = 0) rayPayloadInEXT MainRay mainRay;
hitAttributeEXT vec2 attribs;

#include "common/surface_eval.glsl"

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
    vec3 viewDir = -gl_WorldRayDirectionEXT;

    uint packedData = m0.packedData;
    bool useColorLayer = hasColorLayer(packedData);
    bool useTexture = hasTexture(packedData);
    bool useGlint = hasGlint(packedData);
    bool useOverlay = hasOverlay(packedData);
    vec4 colorLayerValue = useColorLayer ?
                               baryCoords.x * m0.colorLayer + baryCoords.y * m1.colorLayer + baryCoords.z * m2.colorLayer :
                               vec4(1.0);
    vec3 colorLayer = colorLayerValue.rgb;

    uint textureID = m0.textureID;
    uint alphaMode = getAlphaMode(packedData);
    uint coordinate = getCoordinate(packedData);
    vec2 textureUV = vec2(0.0);
    TextureMapEntry textureMap = TextureMapEntry(-1, -1, -1, 0, -1, 0.0, 0.0, 0.0, 0.0);
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
    vec3 dPduWorld = objectToWorld * dposdu;
    vec3 dPdvWorld = objectToWorld * dposdv;
    bool parallaxEnabled = false;
    bool useRealisticWaterSurface = ADV_WATER_SURFACE_MODE == 1u;
    bool hasFftWaterSurface = false;

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
        baseGeoNormal = normalizeF(cross(dPduWorld, dPdvWorld), baseGeoNormal);
        if (dot(baseGeoNormal, viewDir) < 0.0) { baseGeoNormal = -baseGeoNormal; }

        bool isWaterMaterial = useRealisticWaterSurface && isFlaggedWaterSurface(textureMap.flag, textureUV, lod);
        hasFftWaterSurface = isWaterMaterial && abs(baseGeoNormal.y) > 0.75;

        if (parallaxEnabled && textureMap.normal >= 0 && coordinate != 1u) {
            maxDepthWorld = heightMapMaxDepthWorld(atlasUvMin, atlasUvMax, dPduWorld, dPdvWorld);
            hasHeightMapSurface =
                maxDepthWorld > heightMapMinWorldDepth && dot(viewDir, baseGeoNormal) > ADV_PARALLAX_MIN_VIEW_DOT;
        }
    }
    if (dot(baseGeoNormal, viewDir) < 0.0) { baseGeoNormal = -baseGeoNormal; }
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
    if (rayBounce(mainRay) == 0u) {
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
    uint surfaceCachePackedData = packedData | NO_HEIGHT_SURFACE_BIT;

    if (raySurfaceCacheTargetSecondary(mainRay)) {
        storeSecondarySurfaceCacheLayer(ivec2(gl_LaunchIDEXT.xy), 0, vec4(planeHitWorldPos, textureUV.x));
        storeSecondarySurfaceCacheLayer(ivec2(gl_LaunchIDEXT.xy), 1, vec4(dPduWorld, textureUV.y));
        storeSecondarySurfaceCacheLayer(ivec2(gl_LaunchIDEXT.xy), 2, vec4(dPdvWorld, atlasUvMin.x));
        storeSecondarySurfaceCacheLayer(ivec2(gl_LaunchIDEXT.xy), 3, vec4(baseGeoNormal, atlasUvMin.y));
        storeSecondarySurfaceCacheLayer(ivec2(gl_LaunchIDEXT.xy), 4, vec4(atlasUvMax, encodeFirstHitUint(textureID)));
        storeSecondarySurfaceCacheLayer(ivec2(gl_LaunchIDEXT.xy), 5, colorLayerValue);
        storeSecondarySurfaceCacheLayer(ivec2(gl_LaunchIDEXT.xy), 6, vec4(glint, float(m0.overlayUV.x)));
        storeSecondarySurfaceCacheLayer(ivec2(gl_LaunchIDEXT.xy), 7,
                                        vec4(float(m0.overlayUV.y), encodeFirstHitUint(surfaceCachePackedData),
                                             ADV_SURFACE_CACHE_VALID_FLAG));
    } else {
        storePrimarySurfaceCacheLayer(ivec2(gl_LaunchIDEXT.xy), 0, vec4(planeHitWorldPos, textureUV.x));
        storePrimarySurfaceCacheLayer(ivec2(gl_LaunchIDEXT.xy), 1, vec4(dPduWorld, textureUV.y));
        storePrimarySurfaceCacheLayer(ivec2(gl_LaunchIDEXT.xy), 2, vec4(dPdvWorld, atlasUvMin.x));
        storePrimarySurfaceCacheLayer(ivec2(gl_LaunchIDEXT.xy), 3, vec4(baseGeoNormal, atlasUvMin.y));
        storePrimarySurfaceCacheLayer(ivec2(gl_LaunchIDEXT.xy), 4, vec4(atlasUvMax, encodeFirstHitUint(textureID)));
        storePrimarySurfaceCacheLayer(ivec2(gl_LaunchIDEXT.xy), 5, colorLayerValue);
        storePrimarySurfaceCacheLayer(ivec2(gl_LaunchIDEXT.xy), 6, vec4(glint, float(m0.overlayUV.x)));
        storePrimarySurfaceCacheLayer(ivec2(gl_LaunchIDEXT.xy), 7,
                                      vec4(float(m0.overlayUV.y), encodeFirstHitUint(surfaceCachePackedData),
                                           ADV_SURFACE_CACHE_VALID_FLAG));
    }
    raySetSurfaceCacheWritten(mainRay, true);

    SampledSurface surface;
    sampleSurfaceState(useTexture, textureID, textureMap, atlasUvMin, atlasUvMax, initialHit.uv, lod, alphaMode,
                       colorLayerValue, colorLayer, glint, useOverlay, m0.overlayUV, dPduWorld, dPdvWorld,
                       baseGeoNormal, hasHeightMapSurface, maxDepthWorld, initialHit, hitWorldPos, viewDir,
                       hasFftWaterSurface, surface);

    mainRay.normal = surface.shadingNormal;
    rayStoreMaterial(mainRay, surface.albedoValue, surface.mat.f0, surface.mat.roughness, surface.mat.metallic,
                     surface.mat.transmission, surface.mat.ior, surface.mat.emission);
    rayStoreAux(mainRay, hasFftWaterSurface ? vec2(ADV_FFT_WATER_ORIGIN_BIAS, 1.0) : vec2(0.0));

    if (rayCaptureSurface(mainRay)) {
        raySetCaptureSurface(mainRay, false);
        raySetContinue(mainRay, false);
        raySetStop(mainRay, true);
        return;
    }

    SampledSurface currentSurface = surface;
    vec3 currentViewDir = viewDir;
    bool storedLobeType = false;
    for (int localBounce = 0; localBounce < 1; ++localBounce) {
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
            bool chooseReflection = !hasRefraction;
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
