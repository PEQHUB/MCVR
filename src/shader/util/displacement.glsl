#ifndef RARSER_DISPLACEMENT_GLSL
#define RARSER_DISPLACEMENT_GLSL

const float DISPLACEMENT_MIN_DEPTH = 1e-5;
const float DISPLACEMENT_TRACE_BIAS = 2e-4;
const float DISPLACEMENT_WORLD_OFFSET_SCALE = 0.0;
const uint DISPLACEMENT_SOURCE_FLAT = 0u;
const uint DISPLACEMENT_SOURCE_NORMAL_ALPHA = 1u;
const uint DISPLACEMENT_SOURCE_AUTOPBR_ALBEDO = 2u;

struct DisplacementSource {
    uint mode;
    bool isBlock;
    uint textureID;
    int normalTextureID;
    uint animTick;
    vec2 uvMin;
    vec2 uvMax;
    float maxDepth;
    uint pomPacked0;
    uint pomPacked1;
    uint pomPacked2;
    uint flags;
    float lumMin;
    float lumMax;
    uint autoPBRPacked1;
};

struct DisplacementHit {
    bool hit;
    float rayT;
    vec2 uv;
    float depth;
    vec3 worldPos;
    vec3 geometricNormal;
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
    src.pomPacked0 = 0u;
    src.pomPacked1 = 0u;
    src.pomPacked2 = 0u;
    src.flags = 0u;
    src.lumMin = 0.0;
    src.lumMax = 1.0;
    src.autoPBRPacked1 = 100u;
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

uint displacementFilterMode(DisplacementSource src) {
    return src.pomPacked0 & 0x7u; // 0=nearest, 1=bilinear, 2+=smooth/bicubic-ish
}

int displacementMipLevel(float lod) {
    return max(int(floor(lod + 0.5)), 0);
}

ivec2 displacementWrapTexel(ivec2 p, ivec2 size) {
    return ivec2((p.x % size.x + size.x) % size.x,
                 (p.y % size.y + size.y) % size.y);
}

vec2 displacementWrappedBlockUV(vec2 uv) {
    return fract(uv);
}

vec2 displacementEntityUV(DisplacementSource src, vec2 uv) {
    return clamp(uv, src.uvMin, src.uvMax);
}

vec4 displacementSampleBlockAlbedo(DisplacementSource src, vec2 uv, float lod) {
    SpriteEntry se = safeSpriteEntry(src.textureID);
    uint layer = spriteAnimLayer(se, src.animTick);
    int level = displacementMipLevel(lod);
    ivec2 size = max(textureSize(blockAlbedo, level).xy, ivec2(1));
    vec2 st = displacementWrappedBlockUV(uv);
    uint filterMode = displacementFilterMode(src);

    if (filterMode == 0u) {
        ivec2 p = clamp(ivec2(floor(st * vec2(size))), ivec2(0), size - ivec2(1));
        return texelFetch(blockAlbedo, ivec3(p, int(layer)), level);
    }

    vec2 fp = st * vec2(size) - 0.5;
    ivec2 p0 = ivec2(floor(fp));
    vec2 f = fract(fp);
    if (filterMode >= 2u) {
        f = f * f * (3.0 - 2.0 * f);
    }

    vec4 s00 = texelFetch(blockAlbedo, ivec3(displacementWrapTexel(p0, size), int(layer)), level);
    vec4 s10 = texelFetch(blockAlbedo, ivec3(displacementWrapTexel(p0 + ivec2(1, 0), size), int(layer)), level);
    vec4 s01 = texelFetch(blockAlbedo, ivec3(displacementWrapTexel(p0 + ivec2(0, 1), size), int(layer)), level);
    vec4 s11 = texelFetch(blockAlbedo, ivec3(displacementWrapTexel(p0 + ivec2(1, 1), size), int(layer)), level);
    return mix(mix(s00, s10, f.x), mix(s01, s11, f.x), f.y);
}

vec4 displacementSampleBlockNormal(DisplacementSource src, vec2 uv, float lod) {
    SpriteEntry se = safeSpriteEntry(src.textureID);
    if (se.normalLayer < 0) return vec4(0.5, 0.5, 1.0, 1.0);

    int level = displacementMipLevel(lod);
    ivec2 size = max(textureSize(blockNormal, level).xy, ivec2(1));
    vec2 st = displacementWrappedBlockUV(uv);
    uint filterMode = displacementFilterMode(src);

    if (filterMode == 0u) {
        ivec2 p = clamp(ivec2(floor(st * vec2(size))), ivec2(0), size - ivec2(1));
        return texelFetch(blockNormal, ivec3(p, se.normalLayer), level);
    }

    vec2 fp = st * vec2(size) - 0.5;
    ivec2 p0 = ivec2(floor(fp));
    vec2 f = fract(fp);
    if (filterMode >= 2u) {
        f = f * f * (3.0 - 2.0 * f);
    }

    vec4 s00 = texelFetch(blockNormal, ivec3(displacementWrapTexel(p0, size), se.normalLayer), level);
    vec4 s10 = texelFetch(blockNormal, ivec3(displacementWrapTexel(p0 + ivec2(1, 0), size), se.normalLayer), level);
    vec4 s01 = texelFetch(blockNormal, ivec3(displacementWrapTexel(p0 + ivec2(0, 1), size), se.normalLayer), level);
    vec4 s11 = texelFetch(blockNormal, ivec3(displacementWrapTexel(p0 + ivec2(1, 1), size), se.normalLayer), level);
    return mix(mix(s00, s10, f.x), mix(s01, s11, f.x), f.y);
}

vec4 displacementSampleEntityTexture(uint textureID, DisplacementSource src, vec2 uv, float lod) {
    int level = displacementMipLevel(lod);
    ivec2 size = max(textureSize(textures[nonuniformEXT(textureID)], level), ivec2(1));
    vec2 st = displacementEntityUV(src, uv);
    uint filterMode = displacementFilterMode(src);

    if (filterMode == 0u) {
        ivec2 p = clamp(ivec2(floor(st * vec2(size))), ivec2(0), size - ivec2(1));
        return texelFetch(textures[nonuniformEXT(textureID)], p, level);
    }

    vec2 fp = st * vec2(size) - 0.5;
    ivec2 p0 = ivec2(floor(fp));
    ivec2 p1 = p0 + ivec2(1);
    vec2 f = fract(fp);
    if (filterMode >= 2u) {
        f = f * f * (3.0 - 2.0 * f);
    }

    ivec2 c00 = clamp(p0, ivec2(0), size - ivec2(1));
    ivec2 c10 = clamp(ivec2(p1.x, p0.y), ivec2(0), size - ivec2(1));
    ivec2 c01 = clamp(ivec2(p0.x, p1.y), ivec2(0), size - ivec2(1));
    ivec2 c11 = clamp(p1, ivec2(0), size - ivec2(1));
    vec4 s00 = texelFetch(textures[nonuniformEXT(textureID)], c00, level);
    vec4 s10 = texelFetch(textures[nonuniformEXT(textureID)], c10, level);
    vec4 s01 = texelFetch(textures[nonuniformEXT(textureID)], c01, level);
    vec4 s11 = texelFetch(textures[nonuniformEXT(textureID)], c11, level);
    return mix(mix(s00, s10, f.x), mix(s01, s11, f.x), f.y);
}

float displacementRawAlbedo(DisplacementSource src, vec2 uv, float lod) {
    vec4 s;
    if (src.isBlock) {
        s = displacementSampleBlockAlbedo(src, uv, lod);
    } else {
        s = displacementSampleEntityTexture(src.textureID, src, uv, lod);
    }

    int heightSourceMode = int((src.pomPacked0 >> 6u) & 0x7u);
    if (heightSourceMode == 1) return s.r;
    if (heightSourceMode == 2) return s.g;
    if (heightSourceMode == 3) return s.b;
    if (heightSourceMode == 4) return s.a;
    if (heightSourceMode == 5) return max(s.r, max(s.g, s.b));
    if (heightSourceMode == 6) return min(s.r, min(s.g, s.b));
    return dot(s.rgb, vec3(0.2126, 0.7152, 0.0722));
}

float displacementProcessAutoPBR(DisplacementSource src, float raw) {
    float lumMin = src.lumMin;
    float lumMax = src.lumMax;
    if (lumMax <= lumMin + 1e-5) {
        lumMin = 0.0;
        lumMax = 1.0;
    }
    float lumSpan = max(lumMax - lumMin, 1e-5);
    float h = clamp((raw - lumMin) / lumSpan, 0.0, 1.0);

    bool invertHeight = (src.flags & 0x40u) != 0u;
    if (invertHeight) h = 1.0 - h;

    float heightGamma = max(float(src.autoPBRPacked1 & 0xFFFFu) / 100.0, 0.01);
    uint pp1 = src.pomPacked1;
    uint pp2 = src.pomPacked2;
    float heightContrast = max(float((pp1 >> 24u) & 0xFFu) / 10.0, 0.01);
    float remapMin = float(pp2 & 0xFFu) / 100.0;
    float remapMax = float((pp2 >> 8u) & 0xFFu) / 100.0;
    float offset = (float((pp2 >> 16u) & 0xFFu) - 100.0) / 100.0;

    h = mix(remapMin, remapMax, h);
    if (heightGamma != 1.0) h = pow(clamp(h, 0.001, 1.0), heightGamma);
    if (heightContrast != 1.0) h = pow(clamp(h, 0.001, 1.0), heightContrast);
    return clamp(h + offset, 0.0, 1.0);
}

float displacementSampleHeight01(DisplacementSource src, vec2 uv, float lod) {
    if (src.mode == DISPLACEMENT_SOURCE_NORMAL_ALPHA) {
        if (src.isBlock) {
            SpriteEntry se = safeSpriteEntry(src.textureID);
            if (se.normalLayer < 0) return 1.0;
            float h = clamp(displacementSampleBlockNormal(src, uv, lod).a, 0.0, 1.0);
            if (se.maskLayer >= 0) {
                uint packedRange = uint(se.maskLayer);
                float hMin = float(packedRange & 0xFFu) / 255.0;
                float hMax = float((packedRange >> 8u) & 0xFFu) / 255.0;
                if (hMax > hMin + (1.0 / 255.0)) {
                    h = clamp((h - hMin) / (hMax - hMin), 0.0, 1.0);
                }
            }
            return h;
        }
        if (src.normalTextureID < 0) return 1.0;
        return clamp(displacementSampleEntityTexture(uint(src.normalTextureID), src, uv, lod).a, 0.0, 1.0);
    }

    if (src.mode == DISPLACEMENT_SOURCE_AUTOPBR_ALBEDO) {
        return displacementProcessAutoPBR(src, displacementRawAlbedo(src, uv, lod));
    }

    return 1.0;
}

vec2 displacementTexelSpan(DisplacementSource src) {
    vec2 uvSpan = max(abs(src.uvMax - src.uvMin), vec2(1e-5));
    if (src.mode == DISPLACEMENT_SOURCE_NORMAL_ALPHA) {
        if (src.isBlock) {
            return 1.0 / max(vec2(textureSize(blockNormal, 0).xy), vec2(1.0));
        }
        if (src.normalTextureID >= 0) {
            return 1.0 / max(vec2(textureSize(textures[nonuniformEXT(src.normalTextureID)], 0)), vec2(1.0));
        }
    } else if (src.mode == DISPLACEMENT_SOURCE_AUTOPBR_ALBEDO) {
        if (src.isBlock) {
            return 1.0 / max(vec2(textureSize(blockAlbedo, 0).xy), vec2(1.0));
        }
        return 1.0 / max(vec2(textureSize(textures[nonuniformEXT(src.textureID)], 0)), vec2(1.0));
    }
    return uvSpan / 64.0;
}

float displacementEdgeFade(DisplacementSource src, vec2 uv) {
    if (src.isBlock) return 1.0;
    vec2 uvSpan = max(abs(src.uvMax - src.uvMin), vec2(1e-5));
    vec2 texel = displacementTexelSpan(src);
    vec2 fadeWidth = max(texel * 1.5, uvSpan * 0.015);
    vec2 edgeDist = min(uv - src.uvMin, src.uvMax - uv);
    vec2 fade = smoothstep(vec2(0.0), fadeWidth, edgeDist);
    return clamp(fade.x * fade.y, 0.0, 1.0);
}

float displacementSampleDepth(DisplacementSource src, vec2 uv, float lod) {
    return (1.0 - displacementSampleHeight01(src, uv, lod)) *
           src.maxDepth *
           displacementEdgeFade(src, uv);
}

DisplacementSource displacementTraceSource(DisplacementSource source) {
    DisplacementSource src = source;
    src.pomPacked0 = src.pomPacked0 & ~0x7u;
    return src;
}

float displacementSampleTraceDepth(DisplacementSource src, vec2 uv, float lod) {
    return displacementSampleDepth(displacementTraceSource(src), uv, lod);
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

vec3 displacementNormal(DisplacementSource src,
                        vec2 uv,
                        vec3 dPdu,
                        vec3 dPdv,
                        vec3 baseNormal,
                        vec3 viewDir) {
    vec2 uvSpan = max(abs(src.uvMax - src.uvMin), vec2(1e-5));
    vec2 texel = displacementTexelSpan(src);
    vec2 stepUv = max(texel, uvSpan / 1024.0);
    float center = displacementSampleDepth(src, uv, 0.0);
    float depthU = displacementSampleDepth(src, uv + vec2(stepUv.x, 0.0), 0.0);
    float depthV = displacementSampleDepth(src, uv + vec2(0.0, stepUv.y), 0.0);
    vec3 displacedDu = dPdu - baseNormal * ((depthU - center) / max(stepUv.x, 1e-6));
    vec3 displacedDv = dPdv - baseNormal * ((depthV - center) / max(stepUv.y, 1e-6));
    vec3 n = displacementNormalize(cross(displacedDu, displacedDv), baseNormal);
    if (dot(n, viewDir) < 0.0) n = -n;
    return n;
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
    hit.hit = false;
    hit.rayT = 0.0;
    hit.uv = startUV;
    hit.depth = 0.0;
    hit.worldPos = planeWorldPos;
    hit.geometricNormal = baseNormal;

    if (src.mode == DISPLACEMENT_SOURCE_FLAT || src.maxDepth <= DISPLACEMENT_MIN_DEPTH) return false;

    float depthRate = dot(rayDir, -baseNormal);
    if (depthRate <= 1e-5) return false;

    vec2 rateUV = displacementDirectionToRateUv(rayDir, dPdu, dPdv);
    float maxT = src.maxDepth / depthRate;
    int steps = clamp(primarySteps, 1, 512);

    float tPrev = 0.0;
    vec2 uvPrev = src.isBlock ? startUV : clamp(startUV, src.uvMin, src.uvMax);
    float fPrev = -displacementSampleTraceDepth(src, uvPrev, 0.0);
    if (fPrev >= -DISPLACEMENT_TRACE_BIAS) {
        hit.hit = true;
        hit.uv = uvPrev;
        hit.depth = 0.0;
        hit.worldPos = planeWorldPos;
        hit.geometricNormal = displacementNormal(src, uvPrev, dPdu, dPdv, baseNormal, viewDir);
        return true;
    }

    float tHitLo = 0.0;
    float tHitHi = 0.0;
    bool bracketed = false;
    for (int i = 1; i <= steps; ++i) {
        float tCurr = maxT * (float(i) / float(steps));
        vec2 uvCurr = startUV + rateUV * tCurr;
        if (!src.isBlock && (any(lessThan(uvCurr, src.uvMin)) || any(greaterThan(uvCurr, src.uvMax)))) break;

        float depthCurr = depthRate * tCurr;
        float surfaceDepth = displacementSampleTraceDepth(src, uvCurr, 0.0);
        float fCurr = depthCurr - surfaceDepth;
        if (fCurr >= -DISPLACEMENT_TRACE_BIAS) {
            tHitLo = tPrev;
            tHitHi = tCurr;
            bracketed = true;
            break;
        }

        tPrev = tCurr;
        fPrev = fCurr;
    }

    if (!bracketed) return false;

    int refine = clamp(refinementSteps, 0, 8);
    for (int r = 0; r < refine; ++r) {
        float tMid = 0.5 * (tHitLo + tHitHi);
        vec2 uvMid = startUV + rateUV * tMid;
        float depthMid = depthRate * tMid;
        float surfaceDepth = displacementSampleTraceDepth(src, uvMid, 0.0);
        if (depthMid - surfaceDepth >= -DISPLACEMENT_TRACE_BIAS) {
            tHitHi = tMid;
        } else {
            tHitLo = tMid;
        }
    }

    float t = tHitHi;
    vec2 uv = src.isBlock ? displacementWrappedBlockUV(startUV + rateUV * t)
                          : clamp(startUV + rateUV * t, src.uvMin, src.uvMax);
    float depth = displacementSampleDepth(src, uv, 0.0);
    hit.hit = true;
    hit.rayT = t;
    hit.uv = uv;
    hit.depth = depth;
    hit.worldPos = planeWorldPos + rayDir * t;
    hit.geometricNormal = displacementNormal(src, uv, dPdu, dPdv, baseNormal, viewDir);
    return true;
}

bool displacementTraceExit(DisplacementSource src,
                           vec2 startUV,
                           float startDepth,
                           vec3 planeWorldPos,
                           vec3 rayDir,
                           vec3 dPdu,
                           vec3 dPdv,
                           vec3 baseNormal,
                           int secondarySteps,
                           out vec3 exitWorldPos,
                           out float exitDistance) {
    exitWorldPos = displacementWorldPosAt(startUV, startDepth, startUV, planeWorldPos, dPdu, dPdv, baseNormal);
    exitDistance = 0.0;
    if (src.mode == DISPLACEMENT_SOURCE_FLAT || src.maxDepth <= DISPLACEMENT_MIN_DEPTH) return false;

    vec2 rateUV = displacementDirectionToRateUv(rayDir, dPdu, dPdv);
    float depthRate = dot(rayDir, -baseNormal);
    float maxT = src.maxDepth / max(abs(depthRate), 1e-4);
    int steps = clamp(secondarySteps, 1, 256);
    float bias = 2.0 * DISPLACEMENT_TRACE_BIAS;

    for (int i = 1; i <= steps; ++i) {
        float t = bias + maxT * (float(i) / float(steps));
        vec2 uv = startUV + rateUV * t;
        float depth = startDepth + depthRate * t;
        if ((!src.isBlock && (any(lessThan(uv, src.uvMin)) || any(greaterThan(uv, src.uvMax)))) ||
            depth <= 0.0 || depth >= src.maxDepth) {
            vec2 clampedUV = src.isBlock ? displacementWrappedBlockUV(uv) : clamp(uv, src.uvMin, src.uvMax);
            float clampedDepth = clamp(depth, 0.0, src.maxDepth);
            exitWorldPos = displacementWorldPosAt(clampedUV, clampedDepth, startUV, planeWorldPos,
                                                  dPdu, dPdv, baseNormal);
            exitDistance = t;
            return true;
        }
    }
    return false;
}

float displacementSelfShadow(DisplacementSource src,
                             vec2 startUV,
                             float startDepth,
                             vec3 lightDir,
                             vec3 dPdu,
                             vec3 dPdv,
                             vec3 baseNormal,
                             int shadowSteps) {
    if (src.mode == DISPLACEMENT_SOURCE_FLAT || src.maxDepth <= DISPLACEMENT_MIN_DEPTH) return 1.0;

    vec2 rateUV = displacementDirectionToRateUv(lightDir, dPdu, dPdv);
    float depthRate = dot(lightDir, -baseNormal);
    int steps = clamp(shadowSteps, 1, 128);
    float maxT = src.maxDepth / max(abs(depthRate), 1e-4);

    for (int i = 1; i <= steps; ++i) {
        float t = DISPLACEMENT_TRACE_BIAS + maxT * (float(i) / float(steps));
        vec2 uv = startUV + rateUV * t;
        float rayDepth = startDepth + depthRate * t;

        if ((!src.isBlock && (any(lessThan(uv, src.uvMin)) || any(greaterThan(uv, src.uvMax)))) ||
            rayDepth <= 0.0) {
            return 1.0;
        }
        if (rayDepth >= src.maxDepth) return 0.0;

        float surfaceDepth = displacementSampleTraceDepth(src, uv, 0.0);
        if (rayDepth >= surfaceDepth + DISPLACEMENT_TRACE_BIAS) return 0.0;
    }
    return 1.0;
}

#endif // RARSER_DISPLACEMENT_GLSL
