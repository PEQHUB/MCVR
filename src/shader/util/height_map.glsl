#ifndef HEIGHT_MAP_GLSL
#define HEIGHT_MAP_GLSL

#include "common/shared.hpp"
#include "util/color_space.glsl"
#include "sampling_helpers.glsl"

const float heightMapMinWorldDepth = 1e-5;
const float heightMapDepthScaleInTexture = 0.25;
const float heightMapTraceBias = 2e-4;
const float heightMapNearestSecondaryBilinearFallbackDepthRate = 0.02;
const int heightMapNearestMaxSteps = 1024;
const int heightMapBilinearMaxSteps = 24;
const int heightMapBilinearBinarySteps = 5;

struct HeightMapHit {
    bool hit;
    bool sideWall;
    bool edgeWall;
    float t;
    vec2 uv;
    float depth;
    vec3 geometricNormal;
};

vec3 normalizeF(vec3 dir, vec3 fallback) {
    float len2 = dot(dir, dir);
    if (len2 <= 1e-12) { return fallback; }
    return dir * inversesqrt(len2);
}

vec2 directionToRateUv(vec3 dir, vec3 dPdu, vec3 dPdv) {
    float a00 = dot(dPdu, dPdu);
    float a01 = dot(dPdu, dPdv);
    float a11 = dot(dPdv, dPdv);
    float det = a00 * a11 - a01 * a01;
    if (abs(det) <= 1e-12) { return vec2(0.0); }

    float b0 = dot(dPdu, dir);
    float b1 = dot(dPdv, dir);
    return vec2((b0 * a11 - b1 * a01) / det, (b1 * a00 - b0 * a01) / det);
}

float heightMapMaxDepthWorld(vec2 minUV, vec2 maxUV, vec3 dPdu, vec3 dPdv) {
    vec2 uvSpan = abs(maxUV - minUV);
    if (uvSpan.x <= 1e-8 && uvSpan.y <= 1e-8) { return 0.0; }

    float textureWorldU = length(dPdu) * uvSpan.x;
    float textureWorldV = length(dPdv) * uvSpan.y;
    float textureWorldSize = max(textureWorldU, textureWorldV);
    return textureWorldSize * heightMapDepthScaleInTexture;
}

float edgeDepthWithBand(float edgeBaseDepth, float minEdgeDepth) {
    return max(edgeBaseDepth, minEdgeDepth);
}

bool isEdge(float edgeDepth,
            float edgeBaseDepth,
            float minEdgeDepth,
            vec2 edgeUV,
            vec2 minUV,
            vec2 maxUV,
            vec2 rateUV,
            vec3 dPdu,
            vec3 dPdv,
            vec3 baseNormal,
            out vec3 edgeNormal) {
    float bandDepth = edgeDepthWithBand(edgeBaseDepth, minEdgeDepth);
    if (edgeDepth < -heightMapTraceBias) { return false; }
    if (edgeDepth > bandDepth + heightMapTraceBias) { return false; }

    if (rateUV.x > 0.0 && edgeUV.x >= maxUV.x - heightMapTraceBias) {
        edgeNormal = normalizeF(-dPdu, baseNormal);
        return true;
    }
    if (rateUV.x < 0.0 && edgeUV.x <= minUV.x + heightMapTraceBias) {
        edgeNormal = normalizeF(dPdu, baseNormal);
        return true;
    }
    if (rateUV.y > 0.0 && edgeUV.y >= maxUV.y - heightMapTraceBias) {
        edgeNormal = normalizeF(-dPdv, baseNormal);
        return true;
    }
    if (rateUV.y < 0.0 && edgeUV.y <= minUV.y + heightMapTraceBias) {
        edgeNormal = normalizeF(dPdv, baseNormal);
        return true;
    }
    return false;
}

float edgeMinDepth(vec2 minUV, vec2 maxUV, ivec2 size, float maxDepth) {
    vec2 tilePixels = abs(maxUV - minUV) * vec2(size);
    float tileResolution = max(max(tilePixels.x, tilePixels.y), 1.0);
    return max(2.0 * maxDepth / tileResolution, heightMapTraceBias);
}

vec2 heightMapEdgeBandUv(ivec2 size) {
    vec2 safeSize = max(vec2(size), vec2(1.0));
    return 1.0 / (3.0 * safeSize);
}

vec3 wallNormal(bool steppedU, bool steppedV, vec2 rateUV, vec3 dPdu, vec3 dPdv, vec3 baseNormal) {
    vec3 normal = vec3(0.0);
    if (steppedU) { normal += (rateUV.x > 0.0 ? -1.0 : 1.0) * dPdu; }
    if (steppedV) { normal += (rateUV.y > 0.0 ? -1.0 : 1.0) * dPdv; }
    return normalizeF(normal, baseNormal);
}

bool shouldContinueTexelTop(float currentDepth, float enteredDepth, float rayDepth) {
    return abs(enteredDepth - currentDepth) <= heightMapTraceBias && rayDepth >= enteredDepth - heightMapTraceBias;
}

bool enteredTexelWall(float currentDepth, float enteredDepth, float rayDepth) {
    float wallMinDepth = min(currentDepth, enteredDepth);
    float wallMaxDepth = max(currentDepth, enteredDepth);
    return wallMaxDepth > wallMinDepth + heightMapTraceBias && rayDepth >= wallMinDepth - heightMapTraceBias &&
           rayDepth <= wallMaxDepth + heightMapTraceBias;
}

int heightMapRemainingSteps(int texelCoord, int minCoord, int maxCoord, float rate) {
    if (rate > 1e-9) { return max(maxCoord - texelCoord, 0); }
    if (rate < -1e-9) { return max(texelCoord - minCoord, 0); }
    return 0;
}

bool shouldFallbackNearestSecondaryToBilinear(vec3 worldDir, vec3 baseNormal) {
    float depthRate = dot(worldDir, -baseNormal);
    return depthRate < 0.0 && -depthRate <= heightMapNearestSecondaryBilinearFallbackDepthRate;
}

bool isOuterThirdTexelUV(vec2 uv, ivec2 texel, ivec2 atlasTexelMin, ivec2 atlasTexelMax, ivec2 size) {
    if (size.x <= 0 || size.y <= 0) { return false; }

    vec2 texelSize = 1.0 / vec2(size);
    vec2 texelMin = vec2(texel) * texelSize;
    vec2 texelMax = texelMin + texelSize;
    vec2 band = texelSize / 3.0;

    return (texel.x == atlasTexelMin.x && uv.x <= texelMin.x + band.x) ||
           (texel.x == atlasTexelMax.x && uv.x >= texelMax.x - band.x) ||
           (texel.y == atlasTexelMin.y && uv.y <= texelMin.y + band.y) ||
           (texel.y == atlasTexelMax.y && uv.y >= texelMax.y - band.y);
}

float sampleHeightDepthNearest(
    sampler2D tex, ivec2 texel, ivec2 atlasTexelMin, ivec2 atlasTexelMax, vec2 uv, float maxDepth) {
    if (texel.x < atlasTexelMin.x || texel.x > atlasTexelMax.x || texel.y < atlasTexelMin.y ||
        texel.y > atlasTexelMax.y) {
        return 0.0;
    }
    return (1.0 - clamp(sampleTexture(tex, texel, 0, false).w, 0.0, 1.0)) * maxDepth;
}

bool isEdgeUV(vec2 uv, vec2 minUV, vec2 maxUV, ivec2 size) {
    if (size.x <= 0 || size.y <= 0) { return false; }

    vec2 boundsMin = min(minUV, maxUV);
    vec2 boundsMax = max(minUV, maxUV);
    vec2 texelSize = 1.0 / vec2(size);
    return uv.x <= boundsMin.x + texelSize.x || uv.x >= boundsMax.x - texelSize.x ||
           uv.y <= boundsMin.y + texelSize.y || uv.y >= boundsMax.y - texelSize.y;
}

vec2 heightMapMinUV(vec2 minUV, vec2 maxUV, ivec2 size) {
    return min(minUV, maxUV);
}

vec2 heightMapMaxUV(vec2 minUV, vec2 maxUV, ivec2 size) {
    return max(minUV, maxUV);
}

vec2 heightMapInnerMinUV(vec2 minUV, vec2 maxUV, ivec2 size) {
    vec2 boundsMin = min(minUV, maxUV);
    vec2 boundsMax = max(minUV, maxUV);
    vec2 edgeBand = min(heightMapEdgeBandUv(size), 0.5 * max(boundsMax - boundsMin, vec2(0.0)));
    return boundsMin + edgeBand;
}

vec2 heightMapInnerMaxUV(vec2 minUV, vec2 maxUV, ivec2 size) {
    vec2 boundsMin = min(minUV, maxUV);
    vec2 boundsMax = max(minUV, maxUV);
    vec2 edgeBand = min(heightMapEdgeBandUv(size), 0.5 * max(boundsMax - boundsMin, vec2(0.0)));
    return boundsMax - edgeBand;
}

bool isUvWithinHeightMap(vec2 uv, vec2 minUV, vec2 maxUV) {
    return uv.x >= minUV.x && uv.x <= maxUV.x && uv.y >= minUV.y && uv.y <= maxUV.y;
}

ivec2 clampTexelCoord(ivec2 coord, ivec2 size) {
    return clamp(coord, ivec2(0), max(size - ivec2(1), ivec2(0)));
}

vec2 clampUvToRect(vec2 uv, vec2 minUV, vec2 maxUV, ivec2 size) {
    vec2 boundsMin = min(minUV, maxUV);
    vec2 boundsMax = max(minUV, maxUV);
    vec2 halfTexel = 0.5 / vec2(max(size, ivec2(1)));
    vec2 innerMin = min(boundsMin + halfTexel, boundsMax);
    vec2 innerMax = max(boundsMax - halfTexel, boundsMin);
    return clamp(uv, innerMin, innerMax);
}

vec4 samplePBRTexture(sampler2D tex, vec2 uv, vec2 minUV, vec2 maxUV, float lod, uint samplingMode) {
    int lodLevel = max(int(floor(lod + 0.5)), 0);
    ivec2 size = textureSize(tex, lodLevel);
    if (size.x <= 0 || size.y <= 0) { return vec4(0.0); }

    vec2 clampedUv = clampUvToRect(uv, minUV, maxUV, size);
    return samplingMode == 0u ? sampleNearest(tex, clampedUv, lodLevel, false) :
                                sampleBilinear(tex, clampedUv, lodLevel, false);
}

float sampleHeight(sampler2D tex, vec2 uv, vec2 minUV, vec2 maxUV, int lodLevel, uint samplingMode) {
    ivec2 size = textureSize(tex, lodLevel);
    if (size.x <= 0 || size.y <= 0) { return 1.0; }

    vec2 clampedUv = clampUvToRect(uv, minUV, maxUV, size);
    vec4 pbrValue = samplingMode == 0u ? sampleNearest(tex, clampedUv, lodLevel, false) :
                                         sampleBilinear(tex, clampedUv, lodLevel, false);
    return clamp(pbrValue.w, 0.0, 1.0);
}

float sampleHeightDepth(
    sampler2D tex, vec2 uv, vec2 minUV, vec2 maxUV, int lodLevel, uint samplingMode, float maxDepth) {
    float heightValue = sampleHeight(tex, uv, minUV, maxUV, lodLevel, samplingMode);
    return (1.0 - heightValue) * maxDepth;
}

float heightMapBoundaryHitT(float tPrev,
                            float tCurr,
                            vec2 uvPrev,
                            vec2 uvCurr,
                            vec2 boundsMin,
                            vec2 boundsMax) {
    float alpha = 1.0;
    bool crossed = false;

    if (uvCurr.x < boundsMin.x && uvPrev.x >= boundsMin.x) {
        alpha = min(alpha, (boundsMin.x - uvPrev.x) / (uvCurr.x - uvPrev.x));
        crossed = true;
    }
    if (uvCurr.x > boundsMax.x && uvPrev.x <= boundsMax.x) {
        alpha = min(alpha, (boundsMax.x - uvPrev.x) / (uvCurr.x - uvPrev.x));
        crossed = true;
    }
    if (uvCurr.y < boundsMin.y && uvPrev.y >= boundsMin.y) {
        alpha = min(alpha, (boundsMin.y - uvPrev.y) / (uvCurr.y - uvPrev.y));
        crossed = true;
    }
    if (uvCurr.y > boundsMax.y && uvPrev.y <= boundsMax.y) {
        alpha = min(alpha, (boundsMax.y - uvPrev.y) / (uvCurr.y - uvPrev.y));
        crossed = true;
    }

    return crossed ? mix(tPrev, tCurr, clamp(alpha, 0.0, 1.0)) : tCurr;
}

vec3 sampleNormal(sampler2D tex,
                  vec2 uv,
                  vec2 minUV,
                  vec2 maxUV,
                  vec3 dPdu,
                  vec3 dPdv,
                  vec3 fallbackNormal,
                  int lodLevel,
                  uint samplingMode,
                  float maxDepth,
                  vec3 viewDir) {
    ivec2 size = textureSize(tex, lodLevel);
    if (size.x <= 0 || size.y <= 0 || maxDepth <= heightMapMinWorldDepth) { return fallbackNormal; }

    vec2 texelSize = 1.0 / vec2(size);
    float center = sampleHeightDepth(tex, uv, minUV, maxUV, lodLevel, samplingMode, maxDepth);
    float depthU = sampleHeightDepth(tex, uv + vec2(texelSize.x, 0.0), minUV, maxUV, lodLevel, samplingMode, maxDepth);
    float depthV = sampleHeightDepth(tex, uv + vec2(0.0, texelSize.y), minUV, maxUV, lodLevel, samplingMode, maxDepth);

    vec3 displacedDu = dPdu - fallbackNormal * ((depthU - center) / texelSize.x);
    vec3 displacedDv = dPdv - fallbackNormal * ((depthV - center) / texelSize.y);
    vec3 normal = normalizeF(cross(displacedDu, displacedDv), fallbackNormal);
    if (dot(normal, viewDir) < 0.0) { normal = -normal; }
    return normal;
}

bool traceNearestHeightMap(sampler2D tex,
                           vec2 minUV,
                           vec2 maxUV,
                           vec2 uv,
                           float depth,
                           vec3 worldDir,
                           vec3 dPdu,
                           vec3 dPdv,
                           vec3 baseNormal,
                           float maxDepth,
                           float maxDistance,
                           out HeightMapHit hit) {
    hit.hit = false;
    hit.sideWall = false;
    hit.edgeWall = false;
    hit.t = INF_DISTANCE;
    hit.uv = uv;
    hit.depth = depth;
    hit.geometricNormal = baseNormal;

    if (maxDepth <= heightMapMinWorldDepth) { return false; }

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
    int maxSteps = min(heightMapNearestMaxSteps, remainingUSteps + remainingVSteps + 4);

    float tCurrent = 0.0;
    vec2 uvCurrent = uv;
    float depthCurrent = depth;

    int step = 0;
    while (step < maxSteps) {
        if (tCurrent > maxDistance) { break; }
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
        if (tCurrent > maxDistance) { break; }

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
            if (tU < tV - heightMapTraceBias) {
                processU = true;
            } else {
                processU = abs(rateUV.x) >= abs(rateUV.y);
            }
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

bool traceBilinearHeightMap(sampler2D tex,
                            vec2 minUV,
                            vec2 maxUV,
                            vec2 uv,
                            float depth,
                            vec3 worldDir,
                            vec3 dPdu,
                            vec3 dPdv,
                            vec3 baseNormal,
                            float maxDepth,
                            float maxDistance,
                            out HeightMapHit hit) {
    hit.hit = false;
    hit.sideWall = false;
    hit.edgeWall = false;
    hit.t = INF_DISTANCE;
    hit.uv = uv;
    hit.depth = depth;
    hit.geometricNormal = baseNormal;

    if (maxDepth <= heightMapMinWorldDepth) { return false; }

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

    for (int step = 0; step < heightMapBilinearMaxSteps; ++step) {
        float tCurr = min(tPrev + stepWorld, maxDistance);
        vec2 uvCurr = uv + rateUV * tCurr;
        float depthCurr = depth + depthRate * tCurr;
        if (!isUvWithinHeightMap(uvCurr, boundsMin, boundsMax)) {
            float tEdge = heightMapBoundaryHitT(tPrev, tCurr, uvPrev, uvCurr, boundsMin, boundsMax);
            vec2 uvClamped = clamp(uv + rateUV * tEdge, boundsMin, boundsMax);
            float depthEdge = depth + depthRate * tEdge;
            vec3 edgeNormal;
            float boundaryDepth = max(surfacePrev, sampleHeightDepth(tex, uvClamped, minUV, maxUV, 0, 1u, maxDepth));
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

        if (tCurr >= maxDistance - 1e-6) { break; }
        tPrev = tCurr;
        uvPrev = uvCurr;
        depthPrev = depthCurr;
        fPrev = fCurr;
    }

    return false;
}

bool traceHeightMap(sampler2D tex,
                    vec2 minUV,
                    vec2 maxUV,
                    vec2 uv,
                    float depth,
                    vec3 worldDir,
                    vec3 dPdu,
                    vec3 dPdv,
                    vec3 baseNormal,
                    float maxDepth,
                    float maxDistance,
                    uint samplingMode,
                    out HeightMapHit hit) {
    if (samplingMode == 0u) {
        return traceNearestHeightMap(tex, minUV, maxUV, uv, depth, worldDir, dPdu, dPdv, baseNormal, maxDepth,
                                     maxDistance, hit);
    }

    return traceBilinearHeightMap(tex, minUV, maxUV, uv, depth, worldDir, dPdu, dPdv, baseNormal, maxDepth,
                                  maxDistance, hit);
}

#endif
