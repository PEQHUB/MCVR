#ifndef RARSER_DISPLACEMENT_GLSL
#define RARSER_DISPLACEMENT_GLSL

const float DISPLACEMENT_MIN_DEPTH = 1e-5;
const float DISPLACEMENT_TRACE_BIAS = 2e-4;
const float DISPLACEMENT_WORLD_OFFSET_SCALE = 1.0;
const uint DISPLACEMENT_SOURCE_FLAT = 0u;
const uint DISPLACEMENT_SOURCE_AUTHORED_NORMAL_ALPHA = 1u;
const float DISPLACEMENT_INF = 1e20;

struct DisplacementSource {
    uint mode;
    bool isBlock;
    uint textureID;
    int normalTextureID;
    uint animTick;
    vec2 uvMin;
    vec2 uvMax;
    float maxDepth;
};

struct DisplacementHit {
    bool hit;
    bool sideWall;
    bool edgeWall;
    float rayT;
    vec2 uv;
    float depth;
    vec3 worldPos;
    vec3 geometricNormal;
};

struct DisplacementOutgoingRayStart {
    bool visible;
    vec3 origin;
    vec3 biasNormal;
};

void displacementInitSource(out DisplacementSource src) {
    src.mode = DISPLACEMENT_SOURCE_FLAT;
    src.isBlock = false;
    src.textureID = 0u;
    src.normalTextureID = -1;
    src.animTick = 0u;
    src.uvMin = vec2(0.0);
    src.uvMax = vec2(1.0);
    src.maxDepth = 0.0;
}

vec3 displacementNormalize(vec3 value, vec3 fallback) {
    float len2 = dot(value, value);
    return len2 > 1e-12 ? value * inversesqrt(len2) : fallback;
}

bool displacementBuildBasis(vec3 p0,
                            vec3 p1,
                            vec3 p2,
                            vec2 uv0,
                            vec2 uv1,
                            vec2 uv2,
                            out vec3 dPdu,
                            out vec3 dPdv,
                            out vec3 baseNormal) {
    vec3 e1 = p1 - p0;
    vec3 e2 = p2 - p0;
    baseNormal = displacementNormalize(cross(e1, e2), vec3(0.0, 1.0, 0.0));

    vec2 duv1 = uv1 - uv0;
    vec2 duv2 = uv2 - uv0;
    float det = duv1.x * duv2.y - duv2.x * duv1.y;
    if (abs(det) <= 1e-8) {
        dPdu = vec3(0.0);
        dPdv = vec3(0.0);
        return false;
    }

    float invDet = 1.0 / det;
    dPdu = (e1 * duv2.y - e2 * duv1.y) * invDet;
    dPdv = (e2 * duv1.x - e1 * duv2.x) * invDet;
    return dot(dPdu, dPdu) > 1e-10 && dot(dPdv, dPdv) > 1e-10;
}

vec2 displacementDirectionToRateUv(vec3 dir, vec3 dPdu, vec3 dPdv) {
    float a00 = dot(dPdu, dPdu);
    float a01 = dot(dPdu, dPdv);
    float a11 = dot(dPdv, dPdv);
    float det = a00 * a11 - a01 * a01;
    if (abs(det) <= 1e-12) return vec2(0.0);

    float b0 = dot(dPdu, dir);
    float b1 = dot(dPdv, dir);
    return vec2((b0 * a11 - b1 * a01) / det,
                (b1 * a00 - b0 * a01) / det);
}

vec2 displacementWrappedBlockUV(vec2 uv) {
    return fract(uv);
}

vec2 displacementBoundsMin(DisplacementSource src) {
    return min(src.uvMin, src.uvMax);
}

vec2 displacementBoundsMax(DisplacementSource src) {
    return max(src.uvMin, src.uvMax);
}

bool displacementUvInside(vec2 uv, vec2 boundsMin, vec2 boundsMax) {
    return uv.x >= boundsMin.x - DISPLACEMENT_TRACE_BIAS &&
           uv.x <= boundsMax.x + DISPLACEMENT_TRACE_BIAS &&
           uv.y >= boundsMin.y - DISPLACEMENT_TRACE_BIAS &&
           uv.y <= boundsMax.y + DISPLACEMENT_TRACE_BIAS;
}

bool displacementBlockHasAuthoredHeight(uint spriteId) {
    SpriteEntry se = safeSpriteEntry(spriteId);
    if (se.normalLayer < 0) return false;
    if (se.maskLayer < 0) return false;
    if ((se.flags & SPRITE_FLAG_HAS_HEIGHT) == 0u) return false;

    uint normalSource = (se.flags >> SPRITE_FLAG_NORMAL_SOURCE_SHIFT) & SPRITE_FLAG_SOURCE_MASK;
    if (normalSource != SPRITE_SOURCE_PACK_AUTHORED && normalSource != SPRITE_SOURCE_USER_CUSTOM) return false;

    ivec3 normalSize = textureSize(blockNormal, 0);
    return normalSize.x > 0 && normalSize.y > 0 && se.normalLayer < normalSize.z;
}

ivec2 displacementTextureSize(DisplacementSource src) {
    if (src.isBlock && src.mode == DISPLACEMENT_SOURCE_AUTHORED_NORMAL_ALPHA &&
        displacementBlockHasAuthoredHeight(src.textureID)) {
        return max(textureSize(blockNormal, 0).xy, ivec2(1));
    }
    return ivec2(0);
}

ivec2 displacementWrapTexel(ivec2 p, ivec2 size) {
    return ivec2((p.x % size.x + size.x) % size.x,
                 (p.y % size.y + size.y) % size.y);
}

vec2 displacementUvToUnit(DisplacementSource src, vec2 uv) {
    vec2 boundsMin = displacementBoundsMin(src);
    vec2 boundsMax = displacementBoundsMax(src);
    vec2 span = max(boundsMax - boundsMin, vec2(1e-6));
    return clamp((uv - boundsMin) / span, vec2(0.0), vec2(0.999999));
}

ivec2 displacementTexelFromUv(DisplacementSource src, vec2 uv, ivec2 size) {
    return clamp(ivec2(floor(displacementUvToUnit(src, uv) * vec2(size))), ivec2(0), size - ivec2(1));
}

float displacementSampleHeight01Texel(DisplacementSource src, ivec2 texel, ivec2 size) {
    if (!src.isBlock || src.mode != DISPLACEMENT_SOURCE_AUTHORED_NORMAL_ALPHA) return 1.0;
    SpriteEntry se = safeSpriteEntry(src.textureID);
    if (!displacementBlockHasAuthoredHeight(src.textureID)) return 1.0;

    ivec2 p = displacementWrapTexel(texel, size);
    float h = clamp(texelFetch(blockNormal, ivec3(p, se.normalLayer), 0).a, 0.0, 1.0);
    uint packedRange = uint(se.maskLayer);
    float hMin = float(packedRange & 0xFFu) / 255.0;
    float hMax = float((packedRange >> 8u) & 0xFFu) / 255.0;
    if (hMax > hMin + (1.0 / 255.0)) {
        h = clamp((h - hMin) / (hMax - hMin), 0.0, 1.0);
    }
    return h;
}

float displacementSampleDepthTexel(DisplacementSource src, ivec2 texel, ivec2 size) {
    return (1.0 - displacementSampleHeight01Texel(src, texel, size)) * src.maxDepth;
}

float displacementSampleDepth(DisplacementSource src, vec2 uv, float lod) {
    ivec2 size = displacementTextureSize(src);
    if (size.x <= 0 || size.y <= 0 || src.maxDepth <= DISPLACEMENT_MIN_DEPTH) return 0.0;
    return displacementSampleDepthTexel(src, displacementTexelFromUv(src, uv, size), size);
}

vec3 displacementHeightfieldNormalTexel(DisplacementSource src,
                                        ivec2 texel,
                                        ivec2 size,
                                        vec3 dPdu,
                                        vec3 dPdv,
                                        vec3 baseNormal,
                                        vec3 viewDir) {
    if (size.x <= 0 || size.y <= 0 || src.maxDepth <= DISPLACEMENT_MIN_DEPTH) {
        return baseNormal;
    }

    vec2 boundsSpan = max(displacementBoundsMax(src) - displacementBoundsMin(src), vec2(1e-6));
    vec2 cellSize = boundsSpan / vec2(size);

    float depthCenter = displacementSampleDepthTexel(src, texel, size);
    float depthU = depthCenter;
    float depthV = depthCenter;
    if (size.x > 1) {
        depthU = displacementSampleDepthTexel(src, texel + ivec2(1, 0), size);
    }
    if (size.y > 1) {
        depthV = displacementSampleDepthTexel(src, texel + ivec2(0, 1), size);
    }

    vec3 displacedDu = dPdu - baseNormal * ((depthU - depthCenter) / max(cellSize.x, 1e-6));
    vec3 displacedDv = dPdv - baseNormal * ((depthV - depthCenter) / max(cellSize.y, 1e-6));
    vec3 n = displacementNormalize(cross(displacedDu, displacedDv), baseNormal);
    if (dot(n, viewDir) < 0.0) n = -n;
    if (dot(n, baseNormal) < 0.0) n = baseNormal;
    return n;
}

vec3 displacementWorldPosAt(vec2 uv,
                            float depth,
                            vec2 referenceUV,
                            vec3 referenceWorldPos,
                            vec3 dPdu,
                            vec3 dPdv,
                            vec3 baseNormal) {
    vec2 duv = uv - referenceUV;
    return referenceWorldPos + dPdu * duv.x + dPdv * duv.y - baseNormal * depth;
}

bool displacementWallInterval(float depthAtWall, float depthA, float depthB) {
    float lo = min(depthA, depthB);
    float hi = max(depthA, depthB);
    return hi > lo + DISPLACEMENT_TRACE_BIAS &&
           depthAtWall >= lo - DISPLACEMENT_TRACE_BIAS &&
           depthAtWall <= hi + DISPLACEMENT_TRACE_BIAS;
}

vec3 displacementWallNormal(bool steppedU,
                            bool steppedV,
                            vec2 rateUV,
                            vec3 dPdu,
                            vec3 dPdv,
                            vec3 baseNormal) {
    vec3 n = vec3(0.0);
    float dPduLen2 = max(dot(dPdu, dPdu), 1e-12);
    float dPdvLen2 = max(dot(dPdv, dPdv), 1e-12);
    vec3 uWallNormal = dPdu - dPdv * (dot(dPdu, dPdv) / dPdvLen2);
    vec3 vWallNormal = dPdv - dPdu * (dot(dPdv, dPdu) / dPduLen2);
    if (steppedU) n += (rateUV.x > 0.0 ? -1.0 : 1.0) * uWallNormal;
    if (steppedV) n += (rateUV.y > 0.0 ? -1.0 : 1.0) * vWallNormal;
    return displacementNormalize(n, baseNormal);
}

float displacementNextBoundaryT(float originUv,
                                float rate,
                                int texel,
                                int axisSize,
                                float boundsMin,
                                float cellSize) {
    if (abs(rate) <= 1e-9) return DISPLACEMENT_INF;
    float edge = rate > 0.0
        ? boundsMin + float(texel + 1) * cellSize
        : boundsMin + float(texel) * cellSize;
    float t = (edge - originUv) / rate;
    return t > DISPLACEMENT_TRACE_BIAS ? t : DISPLACEMENT_INF;
}

void displacementClearHit(out DisplacementHit hit) {
    hit.hit = false;
    hit.sideWall = false;
    hit.edgeWall = false;
    hit.rayT = 0.0;
    hit.uv = vec2(0.0);
    hit.depth = 0.0;
    hit.worldPos = vec3(0.0);
    hit.geometricNormal = vec3(0.0, 1.0, 0.0);
}

bool displacementTracePrimary(DisplacementSource src,
                              vec2 startUV,
                              vec3 planeWorldPos,
                              vec3 rayDir,
                              vec3 viewDir,
                              vec3 dPdu,
                              vec3 dPdv,
                              vec3 baseNormal,
                              int primarySteps,
                              int refinementSteps,
                              out DisplacementHit hit) {
    displacementClearHit(hit);
    hit.uv = startUV;
    hit.worldPos = planeWorldPos;
    hit.geometricNormal = baseNormal;

    if (src.mode != DISPLACEMENT_SOURCE_AUTHORED_NORMAL_ALPHA ||
        src.maxDepth <= DISPLACEMENT_MIN_DEPTH) {
        return false;
    }

    ivec2 size = displacementTextureSize(src);
    if (size.x <= 0 || size.y <= 0) return false;

    vec2 boundsMin = displacementBoundsMin(src);
    vec2 boundsMax = displacementBoundsMax(src);
    if (!displacementUvInside(startUV, boundsMin, boundsMax)) return false;

    vec2 boundsSpan = max(boundsMax - boundsMin, vec2(1e-6));
    vec2 cellSize = boundsSpan / vec2(size);
    vec2 rateUV = displacementDirectionToRateUv(rayDir, dPdu, dPdv);
    float depthRate = -dot(rayDir, baseNormal);
    if (depthRate <= 1e-8) return false;

    ivec2 texel = displacementTexelFromUv(src, startUV, size);
    float tEntry = 0.0;
    int maxSteps = clamp(primarySteps, 1, 512);
    ivec2 stepSign = ivec2(rateUV.x > 0.0 ? 1 : -1,
                          rateUV.y > 0.0 ? 1 : -1);

    for (int step = 0; step < maxSteps; ++step) {
        float currentDepth = displacementSampleDepthTexel(src, texel, size);
        float tTop = currentDepth / depthRate;
        float tU = displacementNextBoundaryT(startUV.x, rateUV.x, texel.x, size.x, boundsMin.x, cellSize.x);
        float tV = displacementNextBoundaryT(startUV.y, rateUV.y, texel.y, size.y, boundsMin.y, cellSize.y);
        float tExit = min(tU, tV);

        if (tTop >= tEntry - DISPLACEMENT_TRACE_BIAS && tTop <= tExit + DISPLACEMENT_TRACE_BIAS) {
            vec2 hitUV = startUV + rateUV * tTop;
            hit.hit = true;
            hit.sideWall = false;
            hit.edgeWall = false;
            hit.rayT = tTop;
            hit.uv = hitUV;
            hit.depth = currentDepth;
            hit.worldPos = displacementWorldPosAt(hitUV, currentDepth, startUV, planeWorldPos,
                                                  dPdu, dPdv, baseNormal);
            hit.geometricNormal = displacementHeightfieldNormalTexel(src, texel, size, dPdu, dPdv,
                                                                     baseNormal, viewDir);
            return true;
        }

        if (tExit >= DISPLACEMENT_INF * 0.5) return false;

        bool stepU = tU <= tV + DISPLACEMENT_TRACE_BIAS;
        bool stepV = tV <= tU + DISPLACEMENT_TRACE_BIAS;
        vec2 edgeUV = startUV + rateUV * tExit;
        float edgeDepth = depthRate * tExit;

        ivec2 nextTexel = texel + ivec2(stepU ? stepSign.x : 0, stepV ? stepSign.y : 0);
        bool outside = nextTexel.x < 0 || nextTexel.y < 0 ||
                       nextTexel.x >= size.x || nextTexel.y >= size.y;

        if (outside) {
            if (displacementWallInterval(edgeDepth, currentDepth, 0.0)) {
                hit.hit = true;
                hit.sideWall = true;
                hit.edgeWall = true;
                hit.rayT = tExit;
                hit.uv = clamp(edgeUV, boundsMin, boundsMax);
                hit.depth = edgeDepth;
                hit.worldPos = displacementWorldPosAt(hit.uv, edgeDepth, startUV, planeWorldPos,
                                                      dPdu, dPdv, baseNormal);
                hit.geometricNormal = displacementWallNormal(stepU, stepV, rateUV, dPdu, dPdv, baseNormal);
                return true;
            }
            return false;
        }

        float nextDepth = displacementSampleDepthTexel(src, nextTexel, size);
        if (displacementWallInterval(edgeDepth, currentDepth, nextDepth)) {
            hit.hit = true;
            hit.sideWall = true;
            hit.edgeWall = false;
            hit.rayT = tExit;
            hit.uv = clamp(edgeUV, boundsMin, boundsMax);
            hit.depth = edgeDepth;
            hit.worldPos = displacementWorldPosAt(hit.uv, edgeDepth, startUV, planeWorldPos,
                                                  dPdu, dPdv, baseNormal);
            hit.geometricNormal = displacementWallNormal(stepU, stepV, rateUV, dPdu, dPdv, baseNormal);
            return true;
        }

        texel = nextTexel;
        tEntry = tExit;
    }

    return false;
}

bool displacementTraceExit(DisplacementSource src,
                           vec2 startUV,
                           float startDepth,
                           vec3 planeWorldPos,
                           vec3 rayDir,
                           vec3 dPdu,
                           vec3 dPdv,
                           vec3 baseNormal,
                           int maxSteps,
                           out vec3 exitPos,
                           out vec3 exitNormal) {
    exitPos = displacementWorldPosAt(startUV, startDepth, startUV, planeWorldPos,
                                     dPdu, dPdv, baseNormal);
    exitNormal = baseNormal;
    if (src.mode != DISPLACEMENT_SOURCE_AUTHORED_NORMAL_ALPHA) return false;

    vec2 rateUV = displacementDirectionToRateUv(rayDir, dPdu, dPdv);
    float depthRate = -dot(rayDir, baseNormal);
    if (depthRate < -1e-8) {
        float t = max((startDepth + DISPLACEMENT_TRACE_BIAS) / -depthRate, DISPLACEMENT_TRACE_BIAS);
        vec2 uv = startUV + rateUV * t;
        exitPos = displacementWorldPosAt(uv, 0.0, startUV, planeWorldPos, dPdu, dPdv, baseNormal);
        exitNormal = baseNormal;
        return true;
    }

    ivec2 size = displacementTextureSize(src);
    if (size.x <= 0 || size.y <= 0) return false;
    vec2 boundsMin = displacementBoundsMin(src);
    vec2 boundsMax = displacementBoundsMax(src);
    vec2 boundsSpan = max(boundsMax - boundsMin, vec2(1e-6));
    vec2 cellSize = boundsSpan / vec2(size);
    ivec2 texel = displacementTexelFromUv(src, startUV, size);
    ivec2 stepSign = ivec2(rateUV.x > 0.0 ? 1 : -1,
                          rateUV.y > 0.0 ? 1 : -1);

    float tEntry = DISPLACEMENT_TRACE_BIAS;
    int steps = clamp(maxSteps, 1, 512);
    for (int step = 0; step < steps; ++step) {
        float tU = displacementNextBoundaryT(startUV.x, rateUV.x, texel.x, size.x, boundsMin.x, cellSize.x);
        float tV = displacementNextBoundaryT(startUV.y, rateUV.y, texel.y, size.y, boundsMin.y, cellSize.y);
        float tExit = min(tU, tV);
        if (tExit >= DISPLACEMENT_INF * 0.5) return false;

        bool stepU = tU <= tV + DISPLACEMENT_TRACE_BIAS;
        bool stepV = tV <= tU + DISPLACEMENT_TRACE_BIAS;
        ivec2 nextTexel = texel + ivec2(stepU ? stepSign.x : 0, stepV ? stepSign.y : 0);
        if (nextTexel.x < 0 || nextTexel.y < 0 || nextTexel.x >= size.x || nextTexel.y >= size.y) {
            float d = max(startDepth + depthRate * tExit, 0.0);
            vec2 uv = clamp(startUV + rateUV * tExit, boundsMin, boundsMax);
            exitPos = displacementWorldPosAt(uv, d, startUV, planeWorldPos, dPdu, dPdv, baseNormal);
            exitNormal = displacementWallNormal(stepU, stepV, rateUV, dPdu, dPdv, baseNormal);
            return true;
        }
        texel = nextTexel;
        tEntry = tExit;
    }

    return false;
}

void displacementInitOutgoingRayStart(vec3 worldPos,
                                      vec3 rayDir,
                                      vec3 fallbackNormal,
                                      out DisplacementOutgoingRayStart rayStart) {
    rayStart.visible = true;
    rayStart.origin = worldPos;
    rayStart.biasNormal = dot(rayDir, fallbackNormal) > 0.0 ? fallbackNormal : -fallbackNormal;
}

bool displacementResolveOutgoingRayStart(DisplacementSource src,
                                         vec2 startUV,
                                         float startDepth,
                                         vec3 startWorldPos,
                                         vec3 planeWorldPos,
                                         vec3 rayDir,
                                         vec3 dPdu,
                                         vec3 dPdv,
                                         vec3 baseNormal,
                                         vec3 fallbackNormal,
                                         int maxSteps,
                                         out DisplacementOutgoingRayStart rayStart) {
    displacementInitOutgoingRayStart(startWorldPos, rayDir, fallbackNormal, rayStart);
    if (src.mode != DISPLACEMENT_SOURCE_AUTHORED_NORMAL_ALPHA ||
        src.maxDepth <= DISPLACEMENT_MIN_DEPTH) {
        return true;
    }

    vec3 exitPos;
    vec3 exitNormal;
    if (displacementTraceExit(src, startUV, startDepth, planeWorldPos, rayDir,
                              dPdu, dPdv, baseNormal, maxSteps, exitPos, exitNormal)) {
        rayStart.origin = exitPos;
        rayStart.biasNormal = dot(rayDir, exitNormal) > 0.0 ? exitNormal : -exitNormal;
        return true;
    }

    rayStart.visible = false;
    return false;
}

#endif // RARSER_DISPLACEMENT_GLSL
