#include "common/constants.glsl"
#ifndef ADV_DIRECT_LIGHT_RESERVOIR_GLSL
#define ADV_DIRECT_LIGHT_RESERVOIR_GLSL

#include "common/light_neighborhood.glsl"

#ifndef ADV_DIRECT_LIGHT_STRENGTH
#    define ADV_DIRECT_LIGHT_STRENGTH 1.0
#endif

struct AreaLightReservoir {
    vec3 point;
    float valid;
    vec3 normal;
    float targetFunction;
    vec3 emission;
    float contributionWeight;
    float confidence;
    uvec2 sourceID;
    uint sourceLightIndex;
    uint sourceChunkIndex;
};

struct AreaLightSample {
    vec3 point;
    float targetFunction;
    vec3 normal;
    float contributionWeight;
    vec3 emission;
    float confidence;
    uvec2 sourceID;
    uint sourceLightIndex;
    uint sourceChunkIndex;
};

struct AreaLightRisState {
    AreaLightSample sampleOut;
    float weightSum;
    bool hasSampleOut;
};

#ifndef AREA_LIGHT_RESERVOIR_LOAD_IMAGE0
#    define AREA_LIGHT_RESERVOIR_LOAD_IMAGE0 areaLightReservoir0Image
#endif
#ifndef AREA_LIGHT_RESERVOIR_LOAD_IMAGE1
#    define AREA_LIGHT_RESERVOIR_LOAD_IMAGE1 areaLightReservoir1Image
#endif
#ifndef AREA_LIGHT_RESERVOIR_LOAD_IMAGE2
#    define AREA_LIGHT_RESERVOIR_LOAD_IMAGE2 areaLightReservoir2Image
#endif

#ifndef AREA_LIGHT_RESERVOIR_STORE_IMAGE0
#    define AREA_LIGHT_RESERVOIR_STORE_IMAGE0 areaLightReservoir0Image
#endif
#ifndef AREA_LIGHT_RESERVOIR_STORE_IMAGE1
#    define AREA_LIGHT_RESERVOIR_STORE_IMAGE1 areaLightReservoir1Image
#endif
#ifndef AREA_LIGHT_RESERVOIR_STORE_IMAGE2
#    define AREA_LIGHT_RESERVOIR_STORE_IMAGE2 areaLightReservoir2Image
#endif

#ifndef AREA_LIGHT_RESERVOIR_LOAD_RAW0
#    define AREA_LIGHT_RESERVOIR_LOAD_RAW0(pixel) imageLoad(AREA_LIGHT_RESERVOIR_LOAD_IMAGE0, pixel)
#endif
#ifndef AREA_LIGHT_RESERVOIR_LOAD_RAW1
#    define AREA_LIGHT_RESERVOIR_LOAD_RAW1(pixel) imageLoad(AREA_LIGHT_RESERVOIR_LOAD_IMAGE1, pixel)
#endif
#ifndef AREA_LIGHT_RESERVOIR_LOAD_RAW2
#    define AREA_LIGHT_RESERVOIR_LOAD_RAW2(pixel) imageLoad(AREA_LIGHT_RESERVOIR_LOAD_IMAGE2, pixel)
#endif

#ifndef AREA_LIGHT_RESERVOIR_STORE_RAW0
#    define AREA_LIGHT_RESERVOIR_STORE_RAW0(pixel, value) imageStore(AREA_LIGHT_RESERVOIR_STORE_IMAGE0, pixel, value)
#endif
#ifndef AREA_LIGHT_RESERVOIR_STORE_RAW1
#    define AREA_LIGHT_RESERVOIR_STORE_RAW1(pixel, value) imageStore(AREA_LIGHT_RESERVOIR_STORE_IMAGE1, pixel, value)
#endif
#ifndef AREA_LIGHT_RESERVOIR_STORE_RAW2
#    define AREA_LIGHT_RESERVOIR_STORE_RAW2(pixel, value) imageStore(AREA_LIGHT_RESERVOIR_STORE_IMAGE2, pixel, value)
#endif

#ifndef AREA_LIGHT_RESERVOIR_LOAD_SOURCE_RAW
#    define AREA_LIGHT_RESERVOIR_LOAD_SOURCE_RAW(pixel) uvec4(0u)
#endif
#ifndef AREA_LIGHT_RESERVOIR_STORE_SOURCE_RAW
#    define AREA_LIGHT_RESERVOIR_STORE_SOURCE_RAW(pixel, value)
#endif

AreaLightReservoir makeEmptyAreaLightReservoir() {
    AreaLightReservoir reservoir;
    reservoir.point = vec3(0.0);
    reservoir.valid = 0.0;
    reservoir.normal = vec3(0.0, 1.0, 0.0);
    reservoir.targetFunction = 0.0;
    reservoir.emission = vec3(0.0);
    reservoir.contributionWeight = 0.0;
    reservoir.confidence = 0.0;
    reservoir.sourceID = uvec2(0u);
    reservoir.sourceLightIndex = ADV_INVALID_AREA_LIGHT_SOURCE_INDEX;
    reservoir.sourceChunkIndex = ADV_INVALID_AREA_LIGHT_SOURCE_INDEX;
    return reservoir;
}

AreaLightSample makeEmptyAreaLightSample() {
    AreaLightSample lightSample;
    lightSample.point = vec3(0.0);
    lightSample.targetFunction = 0.0;
    lightSample.normal = vec3(0.0, 1.0, 0.0);
    lightSample.contributionWeight = 0.0;
    lightSample.emission = vec3(0.0);
    lightSample.confidence = 0.0;
    lightSample.sourceID = uvec2(0u);
    lightSample.sourceLightIndex = ADV_INVALID_AREA_LIGHT_SOURCE_INDEX;
    lightSample.sourceChunkIndex = ADV_INVALID_AREA_LIGHT_SOURCE_INDEX;
    return lightSample;
}

AreaLightRisState makeAreaLightRisState() {
    AreaLightRisState ris;
    ris.sampleOut = makeEmptyAreaLightSample();
    ris.weightSum = 0.0;
    ris.hasSampleOut = false;
    return ris;
}

void unpackAreaLightReservoirValidityConfidence(float packedValue, out float valid, out float confidence) {
    if (abs(packedValue) <= 1e-8) {
        valid = 0.0;
        confidence = 0.0;
        return;
    }

    valid = packedValue > 0.0 ? 1.0 : 0.0;
    confidence = max(abs(packedValue) - 1.0, 0.0);
}

void storeAreaLightReservoir(ivec2 pixel, AreaLightReservoir reservoir) {
    float packedConfidence = reservoir.confidence + 1.0;
    float packedValidityConfidence = reservoir.valid > 0.5 ? packedConfidence : -packedConfidence;
    AREA_LIGHT_RESERVOIR_STORE_RAW0(pixel, vec4(reservoir.point, packedValidityConfidence));
    AREA_LIGHT_RESERVOIR_STORE_RAW1(pixel, vec4(reservoir.normal, reservoir.targetFunction));
    AREA_LIGHT_RESERVOIR_STORE_RAW2(pixel, vec4(reservoir.emission, reservoir.contributionWeight));
    AREA_LIGHT_RESERVOIR_STORE_SOURCE_RAW(pixel, uvec4(reservoir.sourceID, reservoir.sourceLightIndex,
                                                       reservoir.sourceChunkIndex));
}

AreaLightReservoir decodeAreaLightReservoir(vec4 reservoir0, vec4 reservoir1, vec4 reservoir2, uvec4 reservoirSource) {
    AreaLightReservoir reservoir;
    reservoir.point = reservoir0.xyz;
    unpackAreaLightReservoirValidityConfidence(reservoir0.w, reservoir.valid, reservoir.confidence);
    reservoir.normal = normalizeF(reservoir1.xyz, vec3(0.0, 1.0, 0.0));
    reservoir.targetFunction = reservoir1.w;
    reservoir.emission = reservoir2.xyz;
    reservoir.contributionWeight = reservoir2.w;
    reservoir.sourceID = reservoirSource.xy;
    reservoir.sourceLightIndex = reservoirSource.z;
    reservoir.sourceChunkIndex = reservoirSource.w;
    return reservoir;
}

AreaLightReservoir loadAreaLightReservoir(ivec2 pixel) {
    vec4 reservoir0 = AREA_LIGHT_RESERVOIR_LOAD_RAW0(pixel);
    vec4 reservoir1 = AREA_LIGHT_RESERVOIR_LOAD_RAW1(pixel);
    vec4 reservoir2 = AREA_LIGHT_RESERVOIR_LOAD_RAW2(pixel);
    uvec4 reservoirSource = AREA_LIGHT_RESERVOIR_LOAD_SOURCE_RAW(pixel);
    return decodeAreaLightReservoir(reservoir0, reservoir1, reservoir2, reservoirSource);
}

float areaLightReservoirSampleCount(AreaLightReservoir reservoir) {
    if (!(reservoir.valid > 0.5)) { return 0.0; }
    return reservoir.confidence;
}

AreaLightSample areaLightSampleFromReservoir(AreaLightReservoir reservoir, float targetFunction) {
    AreaLightSample lightSample = makeEmptyAreaLightSample();
    if (!(reservoir.valid > 0.5)) { return lightSample; }

    lightSample.point = reservoir.point;
    lightSample.targetFunction = targetFunction;
    lightSample.normal = reservoir.normal;
    lightSample.contributionWeight = reservoir.contributionWeight;
    lightSample.emission = reservoir.emission;
    lightSample.confidence = areaLightReservoirSampleCount(reservoir);
    lightSample.sourceID = reservoir.sourceID;
    lightSample.sourceLightIndex = reservoir.sourceLightIndex;
    lightSample.sourceChunkIndex = reservoir.sourceChunkIndex;
    return lightSample;
}

float triangleArea(vec3 a, vec3 b, vec3 c) {
    return 0.5 * length(cross(b - a, c - a));
}

vec3 sampleTrianglePoint(vec3 a, vec3 b, vec3 c, vec2 xi) {
    float sqrtXi = sqrt(clamp(xi.x, 0.0, 1.0));
    float bary0 = 1.0 - sqrtXi;
    float bary1 = xi.y * sqrtXi;
    float bary2 = 1.0 - bary0 - bary1;
    return a * bary0 + b * bary1 + c * bary2;
}

vec3 areaLightWorldToScene(vec3 worldPos) {
    return worldPos - vec3(worldUBO.cameraPos.xyz);
}

bool pointOnAreaLightTriangle(vec3 point, vec3 a, vec3 b, vec3 c) {
    vec3 edge0 = b - a;
    vec3 edge1 = c - a;
    vec3 pointVector = point - a;
    vec3 triangleNormal = cross(edge0, edge1);
    float triangleNormalLength2 = dot(triangleNormal, triangleNormal);
    if (triangleNormalLength2 <= 1e-12) { return false; }

    float planeDistance = abs(dot(pointVector, triangleNormal)) * inversesqrt(triangleNormalLength2);
    if (planeDistance > ADV_AREA_LIGHT_RESERVOIR_SOURCE_PLANE_EPSILON) { return false; }

    float dot00 = dot(edge0, edge0);
    float dot01 = dot(edge0, edge1);
    float dot11 = dot(edge1, edge1);
    float dot20 = dot(pointVector, edge0);
    float dot21 = dot(pointVector, edge1);
    float denominator = dot00 * dot11 - dot01 * dot01;
    if (abs(denominator) <= 1e-12) { return false; }

    float v = (dot11 * dot20 - dot01 * dot21) / denominator;
    float w = (dot00 * dot21 - dot01 * dot20) / denominator;
    float u = 1.0 - v - w;
    return u >= -ADV_AREA_LIGHT_RESERVOIR_SOURCE_BARYCENTRIC_EPSILON && v >= -ADV_AREA_LIGHT_RESERVOIR_SOURCE_BARYCENTRIC_EPSILON &&
           w >= -ADV_AREA_LIGHT_RESERVOIR_SOURCE_BARYCENTRIC_EPSILON;
}

bool areaLightMatchesReservoirSource(AreaLightReservoir reservoir, ChunkPackedLight light) {
    vec3 reservoirNormal = normalizeF(reservoir.normal, vec3(0.0, 1.0, 0.0));
    vec3 lightNormal = normalizeF(light.normal.xyz, vec3(0.0, 1.0, 0.0));
    if (dot(lightNormal, reservoirNormal) < ADV_AREA_LIGHT_RESERVOIR_SOURCE_NORMAL_THRESHOLD) { return false; }
    vec3 emissionTolerance = ADV_AREA_LIGHT_RESERVOIR_SOURCE_EMISSION_RELATIVE_EPSILON *
                             max(max(abs(light.radiance.xyz), abs(reservoir.emission)), vec3(1.0));
    if (!all(lessThanEqual(abs(light.radiance.xyz - reservoir.emission), emissionTolerance))) { return false; }

    vec3 p0 = light.p0Area.xyz;
    vec3 p1 = light.p1.xyz;
    vec3 p2 = light.p2.xyz;
    vec3 p3 = light.p3.xyz;
    return pointOnAreaLightTriangle(reservoir.point, p0, p1, p2) || pointOnAreaLightTriangle(reservoir.point, p0, p2, p3);
}

void sampleAreaLightPoint(Light light,
                          inout uint seed,
                          out vec3 sampledPoint,
                          out vec3 lightNormal,
                          out float areaPdf) {
    vec3 p0 = light.p0Area.xyz;
    vec3 p1 = light.p1.xyz;
    vec3 p2 = light.p2.xyz;
    vec3 p3 = light.p3.xyz;

    float triArea0 = triangleArea(p0, p1, p2);
    float triArea1 = triangleArea(p0, p2, p3);
    float totalArea = max(triArea0 + triArea1, light.p0Area.w);
    if (totalArea <= 1e-8) {
        sampledPoint = p0;
        lightNormal = vec3(0.0, 1.0, 0.0);
        areaPdf = 0.0;
        return;
    }

    bool useFirstTriangle = triArea1 <= 1e-8 || rand(seed) * totalArea < triArea0;
    if (useFirstTriangle) {
        sampledPoint = sampleTrianglePoint(p0, p1, p2, vec2(rand(seed), rand(seed)));
        lightNormal = normalizeF(cross(p1 - p0, p2 - p0), vec3(0.0, 1.0, 0.0));
    } else {
        sampledPoint = sampleTrianglePoint(p0, p2, p3, vec2(rand(seed), rand(seed)));
        lightNormal = normalizeF(cross(p2 - p0, p3 - p0), vec3(0.0, 1.0, 0.0));
    }
    areaPdf = 1.0 / totalArea;
}

bool tryGetAreaLightNeighborhoodIndex(vec3 scenePos, out uint centerChunkIndex) {
    centerChunkIndex = ADV_INVALID_AREA_LIGHT_SOURCE_INDEX;
    int resolvedCenterChunkIndex = -1;
    ivec3 centerChunkOrigin = ivec3(0);
    ivec3 sectionCoordinate = ivec3(floor((scenePos + vec3(worldUBO.cameraPos.xyz)) / 16.0));
    if (!tryGetChunkIndexFromSectionCoordinate(sectionCoordinate, worldUBO, resolvedCenterChunkIndex, centerChunkOrigin)) {
        return false;
    }

    ChunkLightData centerChunk = chunkPackedData[resolvedCenterChunkIndex];
    if (!chunkPackedDataMatchesOrigin(centerChunk, centerChunkOrigin)) { return false; }

    centerChunkIndex = uint(resolvedCenterChunkIndex);
    return chunkLightNeighborhoodCount(chunkLightNeighborhoods[centerChunkIndex]) > 0u;
}

bool areaLightReservoirSourceExists(AreaLightReservoir reservoir) {
    if (!(reservoir.valid > 0.5)) { return false; }
    if (reservoir.sourceLightIndex == ADV_INVALID_AREA_LIGHT_SOURCE_INDEX) { return false; }
    if (reservoir.sourceChunkIndex == ADV_INVALID_AREA_LIGHT_SOURCE_INDEX) { return false; }

    ivec3 sectionCoord = ivec3(0);
    ivec3 chunkOrigin = ivec3(0);
    if (!tryGetSectionCoordinateFromChunkIndex(reservoir.sourceChunkIndex, worldUBO, sectionCoord, chunkOrigin)) {
        return false;
    }

    ChunkLightData chunk = chunkPackedData[reservoir.sourceChunkIndex];
    if (!chunkPackedDataMatchesOrigin(chunk, chunkOrigin) || !chunkPackedDataHasLights(chunk)) { return false; }
    if (reservoir.sourceLightIndex >= chunk.lightCount) { return false; }

    ChunkPackedLightBuffer lightBuffer = ChunkPackedLightBuffer(chunk.lightBufferAddress);
    ChunkPackedLight light = lightBuffer.lights[reservoir.sourceLightIndex];
    if (!all(equal(chunkPackedLightSourceID(light), reservoir.sourceID))) { return false; }
    return areaLightMatchesReservoirSource(reservoir, light);
}

vec3 evaluateAreaLightSampleContribution(SampledSurface surface,
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
                                         vec3 lightEmission,
                                         vec3 sampledPoint,
                                         vec3 lightNormal,
                                         bool applyLocalHeightVisibility,
                                         out float targetFunction) {
    targetFunction = 0.0;
    if (!directLightSurfaceEligible(surface) || !isFiniteVec3(lightEmission) || !isFiniteVec3(sampledPoint) ||
        !isFiniteVec3(lightNormal)) {
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
    float sampledLightNoL = dot(sampledLightDir, surface.geometricNormal);
    if (isOpaqueSurface && sampledLightNoL <= 0.0) { return vec3(0.0); }

    float lightNoL = dot(lightNormal, -sampledLightDir);
    if (lightNoL <= 1e-6) { return vec3(0.0); }

    float lightPdf;
    vec3 lightBRDF = DisneyEval(surface.mat, viewDir, surface.shadingNormal, sampledLightDir, lightPdf,
                                RADIANCE_BRDF_FLAGS);
    if (!isFiniteFloat(lightPdf) || !isFiniteVec3(lightBRDF) || lightPdf <= 1e-6) { return vec3(0.0); }

    if (applyLocalHeightVisibility && traceLocalHeight) {
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
    }

    float geometryTerm = lightNoL / distance2;
    vec3 contribution = ADV_DIRECT_LIGHT_STRENGTH * lightEmission * lightBRDF * geometryTerm;
    vec3 clampedContribution = max(contribution, vec3(0.0));
    targetFunction = max(dot(clampedContribution, vec3(0.212671, 0.715160, 0.072169)), 0.0);
    return contribution;
}

bool sampleAreaLightSource(ivec2 pixel,
                           uint baseSeed,
                           uint sampleIndex,
                           uint centerChunkIndex,
                           out ChunkPackedLight chunkLight,
                           out uint sourceChunkIndex,
                           out uint lightIndex,
                           out float lightProbInChunk,
                           out float chunkProbInNeighborhood,
                           out uint candidateSeed) {
    sourceChunkIndex = ADV_INVALID_AREA_LIGHT_SOURCE_INDEX;
    lightIndex = ADV_INVALID_AREA_LIGHT_SOURCE_INDEX;
    lightProbInChunk = 0.0;
    chunkProbInNeighborhood = 0.0;
    candidateSeed = 0u;
    candidateSeed = xxhash32(uvec3(baseSeed ^ sampleIndex, uint(pixel.x), (uint(pixel.y) << 8u) ^ sampleIndex));

    ChunkLightNeighborhood neighborhood = chunkLightNeighborhoods[centerChunkIndex];
    float sampledChunkProb = 0.0;
    int neighborIndex =
        sampleChunkLightNeighborhoodEntry(neighborhood, rand(candidateSeed), rand(candidateSeed), sampledChunkProb);
    if (neighborIndex < 0 || sampledChunkProb <= 1e-8) { return false; }

    ChunkLightNeighborhoodEntry neighborEntry = neighborhood.entries[neighborIndex];
    int sampledChunkIndex = int(neighborEntry.chunkIndex);
    sourceChunkIndex = uint(sampledChunkIndex);
    ChunkLightData sampledChunk = chunkPackedData[sampledChunkIndex];
    chunkProbInNeighborhood = sampledChunkProb;
    if (chunkProbInNeighborhood <= 1e-8 || !chunkPackedDataHasLights(sampledChunk)) { return false; }

    ChunkPackedLightBuffer lightBuffer = ChunkPackedLightBuffer(sampledChunk.lightBufferAddress);

    float scaledLightIndex = clamp(rand(candidateSeed), 0.0, ADV_UNIT_OPEN_UPPER_BOUND) * float(sampledChunk.lightCount);
    lightIndex = min(uint(floor(scaledLightIndex)), sampledChunk.lightCount - 1u);
    chunkLight = lightBuffer.lights[lightIndex];
    lightProbInChunk = 1.0 / float(sampledChunk.lightCount);
    return true;
}

bool areaSampleLights(SampledSurface surface,
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
                      ChunkPackedLight chunkLight,
                      uint sourceChunkIndex,
                      uint lightIndex,
                      float lightProbInChunk,
                      float chunkProbInNeighborhood,
    uint candidateSeed,
    out AreaLightSample lightSample) {
    lightSample = makeEmptyAreaLightSample();
    Light light;
    light.p0Area = chunkLight.p0Area;
    light.p1 = vec4(chunkLight.p1.xyz, 0.0);
    light.p2 = vec4(chunkLight.p2.xyz, 0.0);
    light.p3 = vec4(chunkLight.p3.xyz, 0.0);
    light.emission = vec4(chunkLight.radiance.rgb, chunkLight.p0Area.w);
    light.sampleProb = vec4(lightProbInChunk, lightProbInChunk * chunkProbInNeighborhood, chunkProbInNeighborhood, 0.0);
    if (light.sampleProb.y <= 1e-8 || light.emission.w <= 1e-8) { return false; }

    vec3 sampledPoint;
    vec3 lightNormal;
    float areaPdf;
    sampleAreaLightPoint(light, candidateSeed, sampledPoint, lightNormal, areaPdf);
    if (areaPdf <= 1e-8) { return false; }

    float targetFunction = 0.0;
    evaluateAreaLightSampleContribution(surface, viewDir, referenceUv, referenceWorldPos, atlasUvMin, atlasUvMax,
                                        dPduWorld, dPdvWorld, baseGeoNormal, traceLocalHeight, normalTextureID,
                                        maxDepthWorld, light.emission.rgb, sampledPoint, lightNormal, true,
                                        targetFunction);
    if (targetFunction <= 1e-8) { return false; }

    float proposalPdf = light.sampleProb.y * areaPdf;
    if (proposalPdf <= 1e-8) { return false; }

    lightSample.point = sampledPoint;
    lightSample.normal = lightNormal;
    lightSample.emission = light.emission.rgb;
    lightSample.targetFunction = targetFunction;
    lightSample.contributionWeight = 1.0 / proposalPdf;
    lightSample.confidence = 1.0;
    lightSample.sourceID = chunkPackedLightSourceID(chunkLight);
    lightSample.sourceLightIndex = lightIndex;
    lightSample.sourceChunkIndex = sourceChunkIndex;
    return true;
}

void areaLightRisAddSample(inout AreaLightRisState ris, AreaLightSample lightSample, float weight, inout uint seed) {
    if (!(weight > 1e-8) || !(lightSample.targetFunction > 1e-8) || !(lightSample.contributionWeight > 1e-8) ||
        !(lightSample.confidence > 1e-8)) {
        return;
    }

    ris.weightSum += weight;
    if (!ris.hasSampleOut || rand(seed) * ris.weightSum < weight) {
        ris.sampleOut = lightSample;
        ris.hasSampleOut = true;
    }
}

AreaLightReservoir areaLightReservoirFromRis(AreaLightRisState ris,
                                             float normalizationDenominator,
                                             float outputConfidence) {
    AreaLightReservoir reservoir = makeEmptyAreaLightReservoir();
    if (!ris.hasSampleOut || !(ris.weightSum > 1e-8) || !(ris.sampleOut.targetFunction > 1e-8) ||
        !(normalizationDenominator > 1e-8) || !(outputConfidence > 1e-8)) {
        return reservoir;
    }

    reservoir.point = ris.sampleOut.point;
    reservoir.valid = 1.0;
    reservoir.normal = ris.sampleOut.normal;
    reservoir.confidence = outputConfidence;
    reservoir.targetFunction = ris.sampleOut.targetFunction;
    reservoir.emission = ris.sampleOut.emission;
    reservoir.contributionWeight =
        ris.weightSum / (normalizationDenominator * ris.sampleOut.targetFunction);
    reservoir.sourceID = ris.sampleOut.sourceID;
    reservoir.sourceLightIndex = ris.sampleOut.sourceLightIndex;
    reservoir.sourceChunkIndex = ris.sampleOut.sourceChunkIndex;
    if (!(reservoir.contributionWeight > 1e-8)) {
        return makeEmptyAreaLightReservoir();
    }
    return reservoir;
}

AreaLightReservoir generateInitialAreaLightReservoir(ivec2 pixel,
                                                     uint baseSeed,
                                                     SampledSurface surface,
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
                                                     float maxDepthWorld) {
    AreaLightReservoir reservoir = makeEmptyAreaLightReservoir();
    if (any(isnan(surface.worldPos)) || any(isinf(surface.worldPos))) { return reservoir; }
    uint centerChunkIndex = ADV_INVALID_AREA_LIGHT_SOURCE_INDEX;
    if (!tryGetAreaLightNeighborhoodIndex(surface.worldPos, centerChunkIndex)) { return reservoir; }

    AreaLightRisState ris = makeAreaLightRisState();
    uint risSeed = xxhash32(uvec3(baseSeed, uint(pixel.x), uint(pixel.y)));
    for (uint sampleIndex = 0u; sampleIndex < uint(ADV_INITIAL_SAMPLES); ++sampleIndex) {
        AreaLightSample lightSample;
        ChunkPackedLight chunkLight;
        uint sourceChunkIndex = ADV_INVALID_AREA_LIGHT_SOURCE_INDEX;
        uint lightIndex = ADV_INVALID_AREA_LIGHT_SOURCE_INDEX;
        float lightProbInChunk = 0.0;
        float chunkProbInNeighborhood = 0.0;
        uint candidateSeed = 0u;
        if (!sampleAreaLightSource(pixel, baseSeed, sampleIndex, centerChunkIndex, chunkLight, sourceChunkIndex,
                                   lightIndex, lightProbInChunk, chunkProbInNeighborhood, candidateSeed)) {
            continue;
        }
        if (!areaSampleLights(surface, viewDir, referenceUv, referenceWorldPos, atlasUvMin, atlasUvMax, dPduWorld,
                              dPdvWorld, baseGeoNormal, traceLocalHeight, normalTextureID, maxDepthWorld, chunkLight,
                              sourceChunkIndex, lightIndex, lightProbInChunk, chunkProbInNeighborhood, candidateSeed,
                              lightSample)) {
            continue;
        }

        float risWeight = lightSample.confidence * lightSample.targetFunction * lightSample.contributionWeight;
        areaLightRisAddSample(ris, lightSample, risWeight, risSeed);
    }

    return areaLightReservoirFromRis(ris, float(ADV_INITIAL_SAMPLES), 1.0);
}

#endif
