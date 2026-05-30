#include "common/constants.glsl"
#ifndef ADV_RESTIR_PARALLAX_TRACE_GLSL
#define ADV_RESTIR_PARALLAX_TRACE_GLSL

void initRestirParallaxMiss(vec2 uv, float depth, vec3 baseNormal, out HeightMapHit hit) {
    hit.hit = false;
    hit.sideWall = false;
    hit.edgeWall = false;
    hit.t = INF_DISTANCE;
    hit.uv = uv;
    hit.depth = depth;
    hit.geometricNormal = baseNormal;
}

bool traceRestirNearestHeightMapCapped(sampler2D tex,
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
    initRestirParallaxMiss(uv, depth, baseNormal, hit);
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

bool traceRestirBilinearHeightMapCapped(sampler2D tex,
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
    initRestirParallaxMiss(uv, depth, baseNormal, hit);
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

bool traceRestirHeightMapCapped(sampler2D tex,
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
        return traceRestirNearestHeightMapCapped(tex, minUV, maxUV, uv, depth, worldDir, dPdu, dPdv, baseNormal,
                                                maxDepth, maxTraceSteps, hit);
    }

    return traceRestirBilinearHeightMapCapped(tex, minUV, maxUV, uv, depth, worldDir, dPdu, dPdv, baseNormal,
                                             maxDepth, maxTraceSteps, hit);
}

#endif
