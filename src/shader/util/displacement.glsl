#ifndef RARSER_DISPLACEMENT_GLSL
#define RARSER_DISPLACEMENT_GLSL

const float DISPLACEMENT_MIN_DEPTH = 1e-5;
const float DISPLACEMENT_TRACE_BIAS = 2e-5;
const float DISPLACEMENT_WORLD_OFFSET_SCALE = 1.0;
const float DISPLACEMENT_INF = 1e20;
const int DISPLACEMENT_MAX_TRACE_STEPS = 768;

const uint DISPLACEMENT_SOURCE_FLAT = 0u;
const uint DISPLACEMENT_SOURCE_AUTHORED_NORMAL_ALPHA = 1u;

struct DisplacementSource {
    uint mode;
    bool isBlock;
    uint textureID;
    int normalTextureID;
    uint animTick;
    vec2 uvMin;
    vec2 uvMax;
    float maxDepth;
    bool boundaryWalls;
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
    src.boundaryWalls = true;
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

void displacementInitOutgoingRayStart(vec3 worldPos,
                                      vec3 rayDir,
                                      vec3 fallbackNormal,
                                      out DisplacementOutgoingRayStart rayStart) {
    rayStart.visible = true;
    rayStart.origin = worldPos;
    rayStart.biasNormal = dot(rayDir, fallbackNormal) > 0.0 ? fallbackNormal : -fallbackNormal;
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

vec2 displacementBoundsSpan(DisplacementSource src) {
    return max(displacementBoundsMax(src) - displacementBoundsMin(src), vec2(1e-6));
}

bool displacementUvInside(vec2 uv, vec2 boundsMin, vec2 boundsMax) {
    return uv.x >= boundsMin.x - DISPLACEMENT_TRACE_BIAS &&
           uv.x <= boundsMax.x + DISPLACEMENT_TRACE_BIAS &&
           uv.y >= boundsMin.y - DISPLACEMENT_TRACE_BIAS &&
           uv.y <= boundsMax.y + DISPLACEMENT_TRACE_BIAS;
}

int displacementNormalLayer(uint materialId) {
    SpriteEntry se = safeMaterialSpriteEntry(materialId);
    return materialNormalLayer(materialId, se);
}

int displacementHeightRangePacked(uint materialId) {
    MaterialEntry material = safeMaterialEntry(materialId);
    if (material.heightRangePacked >= 0) return material.heightRangePacked;
    SpriteEntry se = safeMaterialSpriteEntry(materialId);
    return se.maskLayer;
}

bool displacementBlockHasAuthoredHeight(uint materialId) {
    MaterialEntry material = safeMaterialEntry(materialId);
    if ((material.flags & MATERIAL_FLAG_DISPLACEMENT_ELIGIBLE) == 0u) return false;
    if (material.displacementPolicy != MATERIAL_DISPLACEMENT_AUTHORED_HEIGHT) return false;

    SpriteEntry se = safeMaterialSpriteEntry(materialId);
    int normalLayer = displacementNormalLayer(materialId);
    int heightRange = displacementHeightRangePacked(materialId);
    if (normalLayer < 0) return false;
    if (heightRange < 0) return false;
    if ((se.flags & SPRITE_FLAG_HAS_HEIGHT) == 0u) return false;

    uint normalSource = (se.flags >> SPRITE_FLAG_NORMAL_SOURCE_SHIFT) & SPRITE_FLAG_SOURCE_MASK;
    if (normalSource != SPRITE_SOURCE_PACK_AUTHORED && normalSource != SPRITE_SOURCE_USER_CUSTOM) {
        return false;
    }

    ivec3 normalSize = textureSize(blockNormal, 0);
    return normalSize.x > 0 && normalSize.y > 0 && normalLayer < normalSize.z;
}

bool displacementSourceValid(DisplacementSource src) {
    return src.isBlock &&
           src.mode == DISPLACEMENT_SOURCE_AUTHORED_NORMAL_ALPHA &&
           src.maxDepth > DISPLACEMENT_MIN_DEPTH &&
           displacementBlockHasAuthoredHeight(src.textureID);
}

ivec2 displacementTextureSize(DisplacementSource src) {
    if (!displacementSourceValid(src)) return ivec2(0);
    return max(textureSize(blockNormal, 0).xy, ivec2(1));
}

ivec2 displacementWrapTexel(ivec2 p, ivec2 size) {
    return ivec2((p.x % size.x + size.x) % size.x,
                 (p.y % size.y + size.y) % size.y);
}

vec2 displacementUvToUnit(DisplacementSource src, vec2 uv) {
    return (uv - displacementBoundsMin(src)) / displacementBoundsSpan(src);
}

vec2 displacementSampleUv(DisplacementSource src, vec2 uv) {
    return materialRuleUv(src.textureID, fract(displacementUvToUnit(src, uv)));
}

int displacementNormalMipCount() {
    return max(textureQueryLevels(blockNormal), 1);
}

float displacementClampedRuleLod(uint spriteId, float lod) {
    int mipCount = displacementNormalMipCount();
    float maxMip = float(max(mipCount - 1, 0));
    return clamp(materialRuleLod(spriteId, lod), 0.0, maxMip);
}

float displacementFetchNormalAlpha(int layer, ivec2 texel, int lodLevel) {
    int level = clamp(lodLevel, 0, displacementNormalMipCount() - 1);
    ivec3 dims = textureSize(blockNormal, level);
    ivec2 size = max(dims.xy, ivec2(1));
    ivec2 p = displacementWrapTexel(texel, size);
    return clamp(texelFetch(blockNormal, ivec3(p, layer), level).a, 0.0, 1.0);
}

float displacementSampleNormalAlphaBilinear(int layer, vec2 uv, int lodLevel) {
    int level = clamp(lodLevel, 0, displacementNormalMipCount() - 1);
    ivec2 size = max(textureSize(blockNormal, level).xy, ivec2(1));
    vec2 texel = fract(uv) * vec2(size) - vec2(0.5);
    ivec2 p0 = ivec2(floor(texel));
    vec2 f = fract(texel);

    float a00 = displacementFetchNormalAlpha(layer, p0, level);
    float a10 = displacementFetchNormalAlpha(layer, p0 + ivec2(1, 0), level);
    float a01 = displacementFetchNormalAlpha(layer, p0 + ivec2(0, 1), level);
    float a11 = displacementFetchNormalAlpha(layer, p0 + ivec2(1, 1), level);

    float ax0 = mix(a00, a10, f.x);
    float ax1 = mix(a01, a11, f.x);
    return mix(ax0, ax1, f.y);
}

float displacementSampleNormalAlphaTrilinear(int layer, vec2 uv, float lod) {
    float clampedLod = clamp(lod, 0.0, float(max(displacementNormalMipCount() - 1, 0)));
    int level0 = int(floor(clampedLod));
    int level1 = min(level0 + 1, displacementNormalMipCount() - 1);
    float alpha0 = displacementSampleNormalAlphaBilinear(layer, uv, level0);
    float alpha1 = displacementSampleNormalAlphaBilinear(layer, uv, level1);
    return mix(alpha0, alpha1, clampedLod - float(level0));
}

float displacementSampleHeight01(DisplacementSource src, vec2 uv, float lod) {
    if (!displacementSourceValid(src)) return 1.0;

    int normalLayer = displacementNormalLayer(src.textureID);
    if (normalLayer < 0) return 1.0;
    vec2 sampleUv = displacementSampleUv(src, uv);
    float sampleLod = displacementClampedRuleLod(src.textureID, lod);
    return displacementSampleNormalAlphaTrilinear(normalLayer, sampleUv, sampleLod);
}

float displacementSampleDepthAt(DisplacementSource src, vec2 uv, float lod) {
    return (1.0 - displacementSampleHeight01(src, uv, lod)) * src.maxDepth;
}

float displacementSampleDepth(DisplacementSource src, vec2 uv, float lod) {
    if (!displacementSourceValid(src)) return 0.0;
    return displacementSampleDepthAt(src, uv, lod);
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

vec3 displacementHeightfieldNormal(DisplacementSource src,
                                   vec2 uv,
                                   ivec2 textureSize0,
                                   vec3 dPdu,
                                   vec3 dPdv,
                                   vec3 baseNormal,
                                   vec3 viewDir) {
    if (!displacementSourceValid(src) || textureSize0.x <= 0 || textureSize0.y <= 0) {
        return baseNormal;
    }

    vec2 cellSize = displacementBoundsSpan(src) / vec2(max(textureSize0, ivec2(1)));
    float depthU0 = displacementSampleDepthAt(src, uv - vec2(cellSize.x, 0.0), 0.0);
    float depthU1 = displacementSampleDepthAt(src, uv + vec2(cellSize.x, 0.0), 0.0);
    float depthV0 = displacementSampleDepthAt(src, uv - vec2(0.0, cellSize.y), 0.0);
    float depthV1 = displacementSampleDepthAt(src, uv + vec2(0.0, cellSize.y), 0.0);

    vec3 displacedDu = dPdu - baseNormal * ((depthU1 - depthU0) / max(2.0 * cellSize.x, 1e-6));
    vec3 displacedDv = dPdv - baseNormal * ((depthV1 - depthV0) / max(2.0 * cellSize.y, 1e-6));
    vec3 n = displacementNormalize(cross(displacedDu, displacedDv), baseNormal);
    if (dot(n, viewDir) < 0.0) n = -n;
    if (dot(n, baseNormal) < 0.0) n = baseNormal;
    return n;
}

float displacementDominantAxisSignedOffset(vec3 blockBase, vec3 axis) {
    vec3 a = abs(axis);
    if (a.x >= a.y && a.x >= a.z) return blockBase.x * (axis.x < 0.0 ? -1.0 : 1.0);
    if (a.y >= a.z) return blockBase.y * (axis.y < 0.0 ? -1.0 : 1.0);
    return blockBase.z * (axis.z < 0.0 ? -1.0 : 1.0);
}

bool displacementAxisNearlyCardinal(vec3 axis) {
    float len2 = dot(axis, axis);
    if (len2 <= 1e-12) return false;
    vec3 a = abs(axis * inversesqrt(len2));
    return max(max(a.x, a.y), a.z) > 0.999;
}

bool displacementBuildBlockChart(vec3 blockBase,
                                 vec3 dPdu,
                                 vec3 dPdv,
                                 out vec2 chartOffset) {
    chartOffset = vec2(0.0);
    if (!displacementAxisNearlyCardinal(dPdu) || !displacementAxisNearlyCardinal(dPdv)) {
        return false;
    }

    chartOffset = vec2(displacementDominantAxisSignedOffset(blockBase, dPdu),
                       displacementDominantAxisSignedOffset(blockBase, dPdv));
    return true;
}

float displacementMaterialDepthScale(uint materialId) {
    MaterialEntry material = safeMaterialEntry(materialId);
    TextureRuleEntry rule = safeTextureRuleEntry(materialId);
    float materialScale = max(material.displacementScale, 0.0);
    float ruleScale = ((rule.flags & TEXTURE_RULE_ENABLED) != 0u &&
            (rule.flags & TEXTURE_RULE_DISPLACEMENT_SCALE) != 0u)
        ? clamp(rule.displacementScale, 0.0, 4.0)
        : 1.0;
    return materialScale * ruleScale;
}

bool displacementPrepareAuthoredBlockSource(uint packedFlags,
                                            uint packedBlockType,
                                            uint textureID,
                                            vec3 wp0,
                                            vec3 wp1,
                                            vec3 wp2,
                                            vec2 uv0,
                                            vec2 uv1,
                                            vec2 uv2,
                                            vec3 blockBase,
                                            uint animTick,
                                            float hitT,
                                            float depthScale,
                                            float fadeDistanceBlocks,
                                            vec3 viewDir,
                                            bool rejectFluidGeometry,
                                            bool rejectThinCutoutPlant,
                                            out DisplacementSource src,
                                            out vec3 dPdu,
                                            out vec3 dPdv,
                                            out vec3 baseNormal,
                                            out vec2 chartOffset) {
    displacementInitSource(src);
    dPdu = vec3(0.0);
    dPdv = vec3(0.0);
    baseNormal = vec3(0.0, 1.0, 0.0);
    chartOffset = vec2(0.0);

    if (depthScale <= DISPLACEMENT_MIN_DEPTH ||
        fadeDistanceBlocks <= DISPLACEMENT_MIN_DEPTH ||
        hitT >= fadeDistanceBlocks ||
        (packedFlags & PBR_FLAG_USE_TEXTURE) == 0u ||
        (packedFlags & PBR_FLAG_BLOCK_GEOMETRY) == 0u ||
        (rejectFluidGeometry && (packedFlags & PBR_FLAG_FLUID_GEOMETRY) != 0u) ||
        (rejectThinCutoutPlant && (packedBlockType & PBR_PACKED_THIN_CUTOUT_PLANT) != 0u) ||
        !displacementBlockHasAuthoredHeight(textureID)) {
        return false;
    }

    uint coordinateMode = (packedFlags & PBR_FLAG_COORD_MASK) >> PBR_FLAG_COORD_SHIFT;
    if (coordinateMode != 0u) return false;

    if (!displacementBuildBasis(wp0, wp1, wp2, uv0, uv1, uv2, dPdu, dPdv, baseNormal)) {
        return false;
    }
    if (dot(baseNormal, viewDir) < 0.0) {
        baseNormal = -baseNormal;
    }

    displacementBuildBlockChart(blockBase, dPdu, dPdv, chartOffset);

    float fade = 1.0 - smoothstep(fadeDistanceBlocks * 0.75, fadeDistanceBlocks, hitT);
    src.isBlock = true;
    src.textureID = textureID;
    src.normalTextureID = -1;
    src.animTick = animTick;
    src.uvMin = min(min(uv0, uv1), uv2) + chartOffset;
    src.uvMax = max(max(uv0, uv1), uv2) + chartOffset;
    src.maxDepth = max(depthScale, 0.0) * fade * displacementMaterialDepthScale(textureID);
    src.mode = DISPLACEMENT_SOURCE_AUTHORED_NORMAL_ALPHA;
    src.boundaryWalls = false;
    return src.maxDepth > DISPLACEMENT_MIN_DEPTH;
}

float displacementAxisExitT(float origin, float rate, float boundsMin, float boundsMax) {
    if (rate > 1e-9) return (boundsMax - origin) / rate;
    if (rate < -1e-9) return (boundsMin - origin) / rate;
    return DISPLACEMENT_INF;
}

float displacementUvExitT(vec2 uv, vec2 rateUV, vec2 boundsMin, vec2 boundsMax) {
    float tx = displacementAxisExitT(uv.x, rateUV.x, boundsMin.x, boundsMax.x);
    float ty = displacementAxisExitT(uv.y, rateUV.y, boundsMin.y, boundsMax.y);
    return min(tx, ty);
}

int displacementTraceStepCount(DisplacementSource src,
                               vec2 rateUV,
                               float tLimit,
                               ivec2 textureSize0,
                               int requestedSteps) {
    float texelTravel = length(rateUV * tLimit * vec2(max(textureSize0, ivec2(1))));
    int adaptiveSteps = int(ceil(texelTravel * 1.5 + 4.0));
    return clamp(max(requestedSteps, adaptiveSteps), 1, DISPLACEMENT_MAX_TRACE_STEPS);
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

    if (!displacementSourceValid(src)) return false;

    ivec2 textureSize0 = displacementTextureSize(src);
    if (textureSize0.x <= 0 || textureSize0.y <= 0) return false;

    vec2 boundsMin = displacementBoundsMin(src);
    vec2 boundsMax = displacementBoundsMax(src);
    if (!displacementUvInside(startUV, boundsMin, boundsMax)) return false;

    vec2 rateUV = displacementDirectionToRateUv(rayDir, dPdu, dPdv);
    float depthRate = -dot(rayDir, baseNormal);
    if (depthRate <= 1e-8) return false;

    float uvExitT = displacementUvExitT(startUV, rateUV, boundsMin, boundsMax);
    float depthExitT = src.maxDepth / depthRate + DISPLACEMENT_TRACE_BIAS;
    float tLimit = min(depthExitT, uvExitT);
    if (tLimit <= 1e-8) return false;

    float surfacePrev = displacementSampleDepthAt(src, startUV, 0.0);
    float fPrev = -surfacePrev;
    if (fPrev >= -DISPLACEMENT_TRACE_BIAS) {
        hit.hit = true;
        hit.rayT = 0.0;
        hit.uv = startUV;
        hit.depth = surfacePrev;
        hit.worldPos = displacementWorldPosAt(startUV, surfacePrev, startUV, planeWorldPos,
                                              dPdu, dPdv, baseNormal);
        hit.geometricNormal = displacementHeightfieldNormal(src, startUV, textureSize0, dPdu, dPdv,
                                                            baseNormal, viewDir);
        return true;
    }

    float tPrev = 0.0;
    int maxSteps = displacementTraceStepCount(src, rateUV, tLimit, textureSize0, clamp(primarySteps, 1, 512));
    for (int step = 1; step <= maxSteps; ++step) {
        float tCurr = tLimit * (float(step) / float(maxSteps));
        vec2 uvCurr = startUV + rateUV * tCurr;
        if (!displacementUvInside(uvCurr, boundsMin, boundsMax)) return false;

        float surfaceCurr = displacementSampleDepthAt(src, uvCurr, 0.0);
        float fCurr = depthRate * tCurr - surfaceCurr;
        if (fCurr >= -DISPLACEMENT_TRACE_BIAS) {
            float tLo = tPrev;
            float tHi = tCurr;
            int refineCount = clamp(refinementSteps, 0, 10);
            for (int refine = 0; refine < refineCount; ++refine) {
                float tMid = 0.5 * (tLo + tHi);
                vec2 uvMid = startUV + rateUV * tMid;
                float fMid = depthRate * tMid - displacementSampleDepthAt(src, uvMid, 0.0);
                if (fMid >= -DISPLACEMENT_TRACE_BIAS) {
                    tHi = tMid;
                } else {
                    tLo = tMid;
                }
            }

            vec2 hitUV = startUV + rateUV * tHi;
            float hitDepth = displacementSampleDepthAt(src, hitUV, 0.0);
            hit.hit = true;
            hit.rayT = tHi;
            hit.uv = hitUV;
            hit.depth = hitDepth;
            hit.worldPos = displacementWorldPosAt(hitUV, hitDepth, startUV, planeWorldPos,
                                                  dPdu, dPdv, baseNormal);
            hit.geometricNormal = displacementHeightfieldNormal(src, hitUV, textureSize0, dPdu, dPdv,
                                                                baseNormal, viewDir);
            return true;
        }

        tPrev = tCurr;
        fPrev = fCurr;
    }

    return false;
}

bool displacementTraceOutwardOccluded(DisplacementSource src,
                                      vec2 startUV,
                                      float startDepth,
                                      vec3 rayDir,
                                      vec3 dPdu,
                                      vec3 dPdv,
                                      vec3 baseNormal,
                                      int maxSteps) {
    if (!displacementSourceValid(src)) return false;

    float depthRate = -dot(rayDir, baseNormal);
    if (depthRate >= -1e-8) return false;

    ivec2 textureSize0 = displacementTextureSize(src);
    if (textureSize0.x <= 0 || textureSize0.y <= 0) return false;

    vec2 boundsMin = displacementBoundsMin(src);
    vec2 boundsMax = displacementBoundsMax(src);
    if (!displacementUvInside(startUV, boundsMin, boundsMax)) return false;

    vec2 rateUV = displacementDirectionToRateUv(rayDir, dPdu, dPdv);
    float planeExitT = max(startDepth / -depthRate, DISPLACEMENT_TRACE_BIAS);
    float uvExitT = displacementUvExitT(startUV, rateUV, boundsMin, boundsMax);
    float tLimit = min(planeExitT, uvExitT);
    if (tLimit <= DISPLACEMENT_TRACE_BIAS) return false;

    int steps = displacementTraceStepCount(src, rateUV, tLimit, textureSize0, clamp(maxSteps, 1, 512));
    for (int step = 1; step <= steps; ++step) {
        float t = tLimit * (float(step) / float(steps));
        vec2 uv = startUV + rateUV * t;
        if (!displacementUvInside(uv, boundsMin, boundsMax)) return false;

        float rayDepth = startDepth + depthRate * t;
        float surfaceDepth = displacementSampleDepthAt(src, uv, 0.0);
        if (rayDepth > surfaceDepth + DISPLACEMENT_TRACE_BIAS) {
            return true;
        }
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
    if (!displacementSourceValid(src)) return false;

    vec2 rateUV = displacementDirectionToRateUv(rayDir, dPdu, dPdv);
    float depthRate = -dot(rayDir, baseNormal);
    if (depthRate >= -1e-8) return false;

    if (displacementTraceOutwardOccluded(src, startUV, startDepth, rayDir,
                                         dPdu, dPdv, baseNormal, maxSteps)) {
        return false;
    }

    vec2 boundsMin = displacementBoundsMin(src);
    vec2 boundsMax = displacementBoundsMax(src);
    float planeExitT = max((startDepth + DISPLACEMENT_TRACE_BIAS) / -depthRate, DISPLACEMENT_TRACE_BIAS);
    float uvExitT = displacementUvExitT(startUV, rateUV, boundsMin, boundsMax);
    float exitT = min(planeExitT, uvExitT);
    if (exitT <= 1e-8 || exitT >= DISPLACEMENT_INF * 0.5) return false;

    vec2 exitUV = startUV + rateUV * exitT;
    float exitDepth = max(startDepth + depthRate * exitT, 0.0);
    exitPos = displacementWorldPosAt(exitUV, exitDepth, startUV, planeWorldPos, dPdu, dPdv, baseNormal);
    exitNormal = baseNormal;
    return true;
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
    if (!displacementSourceValid(src)) return true;

    vec3 exitPos;
    vec3 exitNormal;
    DisplacementSource exitSource = src;
    if (exitSource.isBlock) {
        exitSource.boundaryWalls = true;
    }
    if (displacementTraceExit(exitSource, startUV, startDepth, planeWorldPos, rayDir,
                              dPdu, dPdv, baseNormal, maxSteps, exitPos, exitNormal)) {
        rayStart.origin = exitPos;
        rayStart.biasNormal = dot(rayDir, exitNormal) > 0.0 ? exitNormal : -exitNormal;
        return true;
    }

    rayStart.visible = false;
    return false;
}

#endif // RARSER_DISPLACEMENT_GLSL
