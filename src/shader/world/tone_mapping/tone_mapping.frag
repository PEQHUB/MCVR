#version 460
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_GOOGLE_include_directive : require

#include "common/shared.hpp"

layout(set = 0, binding = 0) uniform sampler2D HDR;

layout(set = 0, binding = 2) readonly buffer ExposureBuffer {
    float exposure;
    float avgLogLum;
    float tonemapMode;
    float Lwhite;
    float exposureCompensation;
    // HDR fields (appended)
    float hdrPipelineEnabled;  // 0.0 = SDR pipeline, 1.0 = HDR pipeline behavior
    float hdr10OutputEnabled;  // 0.0 = SDR output, 1.0 = HDR10 output encoding
    float peakNits;         // Display peak brightness (e.g. 1000.0)
    float paperWhiteNits;   // ITU-R BT.2408 reference white (e.g. 203.0)
    float saturation;       // Saturation boost (1.0 = neutral)
    float sdrTransferFunction; // 0.0 = Gamma 2.2, 1.0 = sRGB
    float capExposureSmoothed; // internal: kept for layout parity
    float manualExposureEnabled; // 0.0 = auto exposure, 1.0 = manual exposure
    float manualExposure; // direct exposure multiplier when manual is enabled
    // PsychoV tonemapper parameters
    float psychoEnabled;
    float psychoHighlights;
    float psychoShadows;
    float psychoContrast;
    float psychoPurity;
    float psychoBleaching;
    float psychoClipPoint;
    float psychoHueRestore;
    float psychoAdaptContrast;
    float psychoWhiteCurve;      // 0.0 = Neutwo, 1.0 = Naka-Rushton
    float psychoConeExponent;
    float bootTimer;             // layout parity (not read by tone mapper)
}
gExposure;

layout(location = 0) in vec2 texCoord;

layout(location = 0) out vec4 fragColor;

// ============================================================================
// PQ (Perceptual Quantizer) OETF — linear to PQ (ST 2084)
// ============================================================================
vec3 PQ_OETF(vec3 L) {
    const float m1 = 0.1593017578125;    // 2610 / 16384
    const float m2 = 78.84375;            // 2523 / 32 * 128
    const float c1 = 0.8359375;           // 3424 / 4096
    const float c2 = 18.8515625;          // 2413 / 128
    const float c3 = 18.6875;             // 2392 / 128

    vec3 Lm1 = pow(max(L, vec3(0.0)), vec3(m1));
    return pow((c1 + c2 * Lm1) / (1.0 + c3 * Lm1), vec3(m2));
}

// PQ EOTF — PQ to linear
vec3 PQ_EOTF(vec3 N) {
    const float m1 = 0.1593017578125;
    const float m2 = 78.84375;
    const float c1 = 0.8359375;
    const float c2 = 18.8515625;
    const float c3 = 18.6875;

    vec3 Nm2inv = pow(max(N, vec3(0.0)), vec3(1.0 / m2));
    return pow(max(Nm2inv - c1, vec3(0.0)) / (c2 - c3 * Nm2inv), vec3(1.0 / m1));
}

// ============================================================================
// Oklab color space (Björn Ottosson, 2020)
// Perceptually uniform — no luminance-dependent hue shifts like ICtCp/PQ.
// Uses cube root (similar to CIE L*) instead of PQ, so chroma scaling works
// uniformly across all luminance levels including bright emissives (lava).
// ============================================================================

// BT.709 linear → Oklab
vec3 bt709ToOklab(vec3 rgb) {
    // BT.709 → LMS (Oklab specific, column-major)
    const mat3 bt709_to_lms = mat3(
        0.4122214708, 0.2119034982, 0.0883024619,
        0.5363325363, 0.6806995451, 0.2817188376,
        0.0514459929, 0.1073969566, 0.6299787005);
    // LMS^(1/3) → Oklab (column-major)
    const mat3 lms_to_lab = mat3(
        0.2104542553,  1.9779984951,  0.0259040371,
        0.7936177850, -2.4285922050,  0.7827717662,
       -0.0040720468,  0.4505937099, -0.8086757660);

    vec3 lms = bt709_to_lms * rgb;
    // Cube root (handles negatives via sign preservation)
    vec3 lms_g = sign(lms) * pow(abs(lms), vec3(1.0 / 3.0));
    return lms_to_lab * lms_g;
}

// Oklab → BT.709 linear
vec3 oklabToBt709(vec3 lab) {
    // Oklab → LMS^(1/3) (column-major, inverse of lms_to_lab)
    const mat3 lab_to_lms = mat3(
        1.0,           1.0,           1.0,
        0.3963377774, -0.1055613458, -0.0894841775,
        0.2158037573, -0.0638541728, -1.2914855480);
    // LMS → BT.709 (column-major, inverse of bt709_to_lms)
    const mat3 lms_to_bt709 = mat3(
         4.0767416621, -1.2684380046, -0.0041960863,
        -3.3077115913,  2.6097574011, -0.7034186147,
         0.2309699292, -0.3413193965,  1.7076147010);

    vec3 lms_g = lab_to_lms * lab;
    vec3 lms = lms_g * lms_g * lms_g; // cube
    return lms_to_bt709 * lms;
}

// BT.2020 linear → Oklab (direct combined matrix, no intermediate BT.709)
vec3 bt2020ToOklab(vec3 rgb) {
    // BT.2020 → LMS (Oklab): bt709_to_oklab_lms × BT2020_TO_BT709 (column-major)
    const mat3 bt2020_to_lms = mat3(
        0.6167557871, 0.2651330639, 0.1001026342,
        0.3601983994, 0.6358393640, 0.2039065193,
        0.0230458134, 0.0990275718, 0.6959908464);
    // LMS^(1/3) → Oklab (same as BT.709 — gamut-independent)
    const mat3 lms_to_lab = mat3(
        0.2104542553,  1.9779984951,  0.0259040371,
        0.7936177850, -2.4285922050,  0.7827717662,
       -0.0040720468,  0.4505937099, -0.8086757660);

    vec3 lms = bt2020_to_lms * rgb;
    vec3 lms_g = sign(lms) * pow(abs(lms), vec3(1.0 / 3.0));
    return lms_to_lab * lms_g;
}

// Oklab → BT.2020 linear (direct combined matrix, no intermediate BT.709)
vec3 oklabToBt2020(vec3 lab) {
    // Oklab → LMS^(1/3) (same as BT.709 — gamut-independent)
    const mat3 lab_to_lms = mat3(
        1.0,           1.0,           1.0,
        0.3963377774, -0.1055613458, -0.0894841775,
        0.2158037573, -0.0638541728, -1.2914855480);
    // LMS → BT.2020 (column-major, inverse of bt2020_to_lms)
    const mat3 lms_to_bt2020 = mat3(
         2.1399067359, -0.8847358624, -0.0485737581,
        -1.2463895090,  2.1632309822, -0.4545031427,
         0.1064827729, -0.2784951194,  1.5030769008);

    vec3 lms_g = lab_to_lms * lab;
    vec3 lms = lms_g * lms_g * lms_g; // cube
    return lms_to_bt2020 * lms;
}

// Luma weight constants
const vec3 LUMA_BT709  = vec3(0.2126, 0.7152, 0.0722);
const vec3 LUMA_BT2020 = vec3(0.2627, 0.6780, 0.0593);

// ============================================================================
// HDR Mode 0: Hermite Spline Reinhard (BT.2390-aligned)
// Luminance-based Reinhard with cubic Hermite spline knee for smooth
// highlight rolloff. Preserves chrominance ratios.
// References: BT.2390-7, Reinhard et al. 2002
// ============================================================================
vec3 HermiteSplineReinhardToneMap(vec3 color, float Lw, vec3 luma) {
    float L = dot(color, luma);
    if (L < 1e-6) return color;

    // Basic Reinhard with white point
    float Lw2 = Lw * Lw;
    float Lr = L * (1.0 + L / Lw2) / (1.0 + L);

    // Hermite spline knee: smooth transition in highlight region
    // Knee region: [kneeStart, kneeEnd] gets smooth cubic interpolation
    float kneeStart = 0.5 * Lw;
    float kneeEnd = Lw;

    if (L > kneeStart && L < kneeEnd) {
        float t = (L - kneeStart) / (kneeEnd - kneeStart);
        float t2 = t * t;
        float t3 = t2 * t;

        // Hermite basis functions
        float h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
        float h10 = t3 - 2.0 * t2 + t;
        float h01 = -2.0 * t3 + 3.0 * t2;
        float h11 = t3 - t2;

        // Endpoint values from Reinhard formula
        float p0 = kneeStart * (1.0 + kneeStart / Lw2) / (1.0 + kneeStart);
        float p1 = kneeEnd * (1.0 + kneeEnd / Lw2) / (1.0 + kneeEnd);

        // Tangents: derivative of Reinhard at endpoints, scaled by interval width
        float span = kneeEnd - kneeStart;
        float m0 = (1.0 + 2.0 * kneeStart / Lw2) / ((1.0 + kneeStart) * (1.0 + kneeStart)) * span;
        float m1 = (1.0 + 2.0 * kneeEnd / Lw2) / ((1.0 + kneeEnd) * (1.0 + kneeEnd)) * span;

        Lr = h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1;
    }

    return color * (Lr / L);
}

// ============================================================================
// HDR Mode 2: BT.2390 EETF (ITU-R BT.2390-7)
// Electrical-Electrical Transfer Function for HDR display adaptation.
// Maps scene luminance to display luminance using Hermite spline knee.
// Input: exposed linear RGB. Output: linear RGB scaled for HDR headroom.
// ============================================================================
vec3 BT2390EETF(vec3 color, float maxLum, vec3 luma) {
    float L = dot(color, luma);
    if (L < 1e-6) return color;

    // Normalize luminance to [0, maxLum] range
    float Ln = min(L / maxLum, 1.0);

    // BT.2390 knee parameters
    // ks = knee start point (normalized), typically 1.5*maxLum - 0.5 mapped to [0,1]
    float ks = max(1.5 * maxLum - 0.5, 0.0) / maxLum;
    ks = clamp(ks, 0.0, 0.99);

    float b = Ln; // Default: passthrough below knee

    if (Ln >= ks) {
        // Hermite spline compression above knee
        float t = (Ln - ks) / (1.0 - ks);
        float t2 = t * t;
        float t3 = t2 * t;

        // Spline from (ks, ks) to (1.0, 1.0) with controlled tangents
        float p0 = ks;
        float p1 = 1.0;
        float m0 = 1.0 * (1.0 - ks);  // tangent matches linear below knee
        float m1 = 0.0;                 // zero tangent at peak (soft rolloff)

        b = (2.0 * t3 - 3.0 * t2 + 1.0) * p0
          + (t3 - 2.0 * t2 + t) * m0
          + (-2.0 * t3 + 3.0 * t2) * p1
          + (t3 - t2) * m1;
    }

    // Scale back and apply ratio to preserve chrominance
    float Lout = b * maxLum;
    return color * (Lout / L);
}

// ============================================================================
// PsychoV Tonemapper (RenoDX psycho_test11)
// Perceptual tonemapper operating in Stockman-Sharp LMS cone space with
// MacLeod-Boynton chromaticity, Weber-Fechner adaptation, and hue-preserving
// gamut compression. Ported from HLSL to GLSL.
// ============================================================================

// Color space matrices (GLSL column-major = transposed from HLSL row-major)
const mat3 PSYCHO_BT709_TO_XYZ = mat3(
    0.4123907993, 0.2126390059, 0.0193308187,
    0.3575843394, 0.7151686788, 0.1191947798,
    0.1804807884, 0.0721923154, 0.9505321522);

const mat3 PSYCHO_XYZ_TO_BT709 = mat3(
     3.2409699419, -0.9692436363,  0.0556300797,
    -1.5373831776,  1.8759675015, -0.2039769589,
    -0.4986107603,  0.0415550574,  1.0569715142);

const mat3 PSYCHO_BT2020_TO_XYZ = mat3(
    0.6369580483, 0.2627002120, 0.0000000000,
    0.1446169036, 0.6779980715, 0.0280726930,
    0.1688809752, 0.0593017165, 1.0609850577);

const mat3 PSYCHO_XYZ_TO_BT2020 = mat3(
     1.7166511880, -0.6666843518,  0.0176398574,
    -0.3556707838,  1.6164812366, -0.0427706133,
    -0.2533662814,  0.0157685458,  0.9421031212);

const mat3 PSYCHO_XYZ_TO_LMS = mat3(
    0.2670502842655792,  -0.38706882411220156, 0.026727793989083093,
    0.8471990148492798,   1.165429935890458,   -0.02729131667566509,
   -0.03470416612462053,  0.10302286696614202,  0.5333267257603284);

const vec3 PSYCHO_MB_WEIGHTS = vec3(0.68990272, 0.34832189, 0.0371597);
const vec2 PSYCHO_WHITE_D65 = vec2(0.31272, 0.32903);

float psychoDivideSafe(float a, float b, float fallback) {
    return b == 0.0 ? fallback : a / b;
}

vec3 psychoSignPow(vec3 x, vec3 e) {
    return vec3(
        (x.x < 0.0 ? -1.0 : 1.0) * pow(abs(x.x), e.x),
        (x.y < 0.0 ? -1.0 : 1.0) * pow(abs(x.y), e.y),
        (x.z < 0.0 ? -1.0 : 1.0) * pow(abs(x.z), e.z));
}

mat3 psychoInvert3x3(mat3 m) {
    // GLSL mat3 is column-major: m[col][row]
    float a = m[0][0], b = m[1][0], c = m[2][0];
    float d = m[0][1], e = m[1][1], f = m[2][1];
    float g = m[0][2], h = m[1][2], i = m[2][2];

    float A = e * i - f * h;
    float B = -(d * i - f * g);
    float C = d * h - e * g;
    float D = -(b * i - c * h);
    float E = a * i - c * g;
    float F = -(a * h - b * g);
    float G = b * f - c * e;
    float H = -(a * f - c * d);
    float I = a * e - b * d;

    float det = a * A + b * B + c * C;
    float inv_det = psychoDivideSafe(1.0, det, 0.0);

    // Result is column-major
    return mat3(
        A * inv_det, B * inv_det, C * inv_det,
        D * inv_det, E * inv_det, F * inv_det,
        G * inv_det, H * inv_det, I * inv_det);
}

vec3 psychoBT2020FromBT709(vec3 bt709) {
    return PSYCHO_XYZ_TO_BT2020 * (PSYCHO_BT709_TO_XYZ * bt709);
}

vec3 psychoBT709FromBT2020(vec3 bt2020) {
    return PSYCHO_XYZ_TO_BT709 * (PSYCHO_BT2020_TO_XYZ * bt2020);
}

vec3 psychoLMSFromBT2020(vec3 bt2020) {
    return PSYCHO_XYZ_TO_LMS * (PSYCHO_BT2020_TO_XYZ * bt2020);
}

vec3 psychoBT2020FromLMS(vec3 lms_abs) {
    mat3 lms_to_xyz = psychoInvert3x3(PSYCHO_XYZ_TO_LMS);
    return PSYCHO_XYZ_TO_BT2020 * (lms_to_xyz * lms_abs);
}

float psychoLuminanceFromLMS(vec3 lms) {
    return dot(lms, vec3(0.68990272, 0.34832189, 0.0));
}

float psychoMBYFromLMS(vec3 lms) {
    return PSYCHO_MB_WEIGHTS.x * lms.x + PSYCHO_MB_WEIGHTS.y * lms.y;
}

vec3 psychoMB2FromLMS(vec3 lms) {
    const float eps = 1e-12;
    float wl = PSYCHO_MB_WEIGHTS.x * lms.x;
    float wm = PSYCHO_MB_WEIGHTS.y * lms.y;
    float y_mb = wl + wm;
    if (y_mb <= eps) return vec3(0.0);
    float inv = psychoDivideSafe(1.0, y_mb, 0.0);
    return vec3(wl * inv, PSYCHO_MB_WEIGHTS.z * lms.z * inv, y_mb);
}

vec3 psychoLMSFromMB2(vec3 mb2) {
    float l = mb2.x, s = mb2.y, y = max(mb2.z, 0.0);
    float L = psychoDivideSafe(l * y, PSYCHO_MB_WEIGHTS.x, 0.0);
    float M = psychoDivideSafe((1.0 - l) * y, PSYCHO_MB_WEIGHTS.y, 0.0);
    float S = psychoDivideSafe(s * y, PSYCHO_MB_WEIGHTS.z, 0.0);
    return vec3(L, M, S);
}

vec3 psychoXYZFromxyY(vec3 xyY) {
    vec3 xyz;
    xyz.xz = vec2(xyY.x, 1.0 - xyY.x - xyY.y) / xyY.y * xyY.z;
    xyz.y = xyY.z;
    return xyz;
}

vec2 psychoWhiteD65Chromaticity() {
    vec3 d65_xyz = psychoXYZFromxyY(vec3(PSYCHO_WHITE_D65, 1.0));
    vec3 d65_lms = PSYCHO_XYZ_TO_LMS * d65_xyz;
    return psychoMB2FromLMS(d65_lms).xy;
}

vec2 psychoMBFromLMS(vec3 lms) {
    float y_mb = psychoMBYFromLMS(lms);
    if (y_mb <= 0.0) return vec2(0.0);
    return vec2(
        psychoDivideSafe(PSYCHO_MB_WEIGHTS.x * lms.x, y_mb, 0.0),
        psychoDivideSafe(PSYCHO_MB_WEIGHTS.z * lms.z, y_mb, 0.0));
}

vec2 psychoMBFromBT2020Primary(vec3 primary_rgb) {
    vec3 xyz = PSYCHO_BT2020_TO_XYZ * primary_rgb;
    vec3 lms = PSYCHO_XYZ_TO_LMS * xyz;
    return psychoMBFromLMS(lms);
}

float psychoCross2(vec2 a, vec2 b) {
    return a.x * b.y - a.y * b.x;
}

bool psychoRaySegmentHit2D(vec2 origin, vec2 dir, vec2 a, vec2 b, out float t_hit) {
    const float eps = 1e-20;
    t_hit = 0.0;
    vec2 e = b - a;
    float denom = psychoCross2(dir, e);
    if (abs(denom) <= eps) return false;
    vec2 ao = a - origin;
    float t = psychoCross2(ao, e) / denom;
    float u = psychoCross2(ao, dir) / denom;
    if (t < 0.0 || u < 0.0 || u > 1.0) return false;
    t_hit = t;
    return true;
}

float psychoRayMaxT_BT2020(vec2 origin, vec2 direction, out bool has_solution) {
    const float interval_max = 1e30;
    const float eps = 1e-14;
    has_solution = false;
    if (dot(direction, direction) <= eps) return 0.0;

    vec2 r = psychoMBFromBT2020Primary(vec3(1.0, 0.0, 0.0));
    vec2 g = psychoMBFromBT2020Primary(vec3(0.0, 1.0, 0.0));
    vec2 b = psychoMBFromBT2020Primary(vec3(0.0, 0.0, 1.0));

    float t_best = interval_max;
    float t;
    bool hit_any = false;

    if (psychoRaySegmentHit2D(origin, direction, r, g, t)) { t_best = min(t_best, t); hit_any = true; }
    if (psychoRaySegmentHit2D(origin, direction, g, b, t)) { t_best = min(t_best, t); hit_any = true; }
    if (psychoRaySegmentHit2D(origin, direction, b, r, t)) { t_best = min(t_best, t); hit_any = true; }

    has_solution = hit_any;
    return hit_any ? max(t_best, 0.0) : 0.0;
}

vec3 psychoGamutCompress(vec3 lms) {
    const float eps = 1e-14;
    float y_mb = psychoMBYFromLMS(abs(lms));
    vec2 white = psychoWhiteD65Chromaticity();
    vec2 mb0 = psychoMBFromLMS(lms);
    vec2 direction = mb0 - white;
    if (dot(direction, direction) < eps) return lms;

    bool has_solution;
    float t_max = psychoRayMaxT_BT2020(white, direction, has_solution);
    if (!has_solution) return lms;

    float white_ratio = max(psychoDivideSafe(1.0 - t_max, t_max, 0.0), 0.0);
    float white_add = y_mb * white_ratio;
    vec3 white_unit_lms = psychoLMSFromMB2(vec3(white, 1.0));
    return lms + white_unit_lms * white_add;
}

vec3 psychoRestoreHueMB2(vec3 lms_source, vec3 lms_target, float amount) {
    const float eps = 1e-6;
    float restore = clamp(amount, 0.0, 1.0);
    if (restore <= 0.0) return lms_target;

    vec3 mb_source = psychoMB2FromLMS(lms_source);
    vec3 mb_target = psychoMB2FromLMS(lms_target);
    vec2 mb_white = psychoWhiteD65Chromaticity();

    vec2 source_offset = mb_source.xy - mb_white;
    vec2 target_offset = mb_target.xy - mb_white;
    float src2 = dot(source_offset, source_offset);
    float tgt2 = dot(target_offset, target_offset);
    if (src2 <= eps || tgt2 <= eps) return lms_target;

    vec2 source_dir = source_offset * inversesqrt(src2);
    vec2 target_dir = target_offset * inversesqrt(tgt2);
    vec2 blended_dir = mix(target_dir, source_dir, restore);
    float blended_len2 = dot(blended_dir, blended_dir);
    if (blended_len2 <= eps) {
        blended_dir = target_dir;
    } else {
        blended_dir *= inversesqrt(blended_len2);
    }

    float target_radius = sqrt(tgt2);
    vec2 mb_restored_xy = mb_white + blended_dir * target_radius;
    return psychoLMSFromMB2(vec3(mb_restored_xy, mb_target.z));
}

vec3 psychoScalePurityMB2(vec3 lms, float purity_scale) {
    const float eps = 1e-6;
    if (abs(purity_scale - 1.0) <= eps) return lms;
    vec3 mb = psychoMB2FromLMS(lms);
    vec2 mb_white = psychoWhiteD65Chromaticity();
    vec2 mb_offset = mb.xy - mb_white;
    vec2 direction = mb_offset;
    float offsetLen = length(direction);
    if (offsetLen < eps) return lms;

    // Clamp scaled offset to BT.2020 gamut boundary to prevent negative LMS
    float desiredLen = offsetLen * max(purity_scale, 0.0);
    bool has_solution;
    float t_max = psychoRayMaxT_BT2020(mb_white, direction / offsetLen, has_solution);
    float maxLen = has_solution ? t_max : offsetLen;
    float clampedLen = min(desiredLen, maxLen * 0.98); // 2% margin from boundary

    vec2 mb_scaled = mb_white + (direction / offsetLen) * clampedLen;
    return psychoLMSFromMB2(vec3(mb_scaled, mb.z));
}

float psychoContrastSafe(float x, float contrast, float mid_gray) {
    float ratio = x / mid_gray;
    float signed_pow = (ratio < 0.0 ? -1.0 : 1.0) * pow(abs(ratio), contrast);
    return signed_pow * mid_gray;
}

float psychoHighlightsFn(float x, float highlights, float mid_gray) {
    if (highlights > 1.0) {
        return max(x, mix(x, mid_gray * pow(x / mid_gray, highlights), x));
    }
    if (highlights < 1.0) {
        return min(x, x / (1.0 + mid_gray * pow(x / mid_gray, 2.0 - highlights) - x));
    }
    return x;
}

float psychoShadowsFn(float x, float shadows, float mid_gray) {
    if (shadows > 1.0) {
        return max(x, x * (1.0 + (x * mid_gray / pow(x / mid_gray, shadows))));
    }
    if (shadows < 1.0) {
        return clamp(x * (1.0 - (x * mid_gray / pow(x / mid_gray, 2.0 - shadows))), 0.0, x);
    }
    return x;
}

float psychoNeutwo(float x, float peak) {
    return (peak * x) * inversesqrt(x * x + peak * peak);
}

float psychoNeutwoClip(float x, float peak, float clip) {
    float cc = clip * clip;
    float pp = peak * peak;
    float xx = x * x;
    float numerator = clip * peak * x;
    float denom_sq = xx * (cc - pp) + cc * pp;
    return numerator * inversesqrt(denom_sq);
}

vec3 psychoNeutwoPerChannel(vec3 color, vec3 peak) {
    return vec3(
        psychoNeutwo(color.r, peak.r),
        psychoNeutwo(color.g, peak.g),
        psychoNeutwo(color.b, peak.b));
}

vec3 psychoNeutwoClipPerChannel(vec3 color, vec3 peak, vec3 clip) {
    return vec3(
        psychoNeutwoClip(color.r, peak.r, clip.r),
        psychoNeutwoClip(color.g, peak.g, clip.g),
        psychoNeutwoClip(color.b, peak.b, clip.b));
}

vec3 psychoNakaRushton(vec3 x, vec3 peak, vec3 gray, float cone_exp) {
    vec3 n = cone_exp * peak / (peak - gray);
    vec3 x_n = psychoSignPow(x, n);
    vec3 num = peak * x_n;
    vec3 den = pow(gray, n - 1.0) * (peak - gray) + x_n;
    return num / den;
}

// Main PsychoV tonemapping function
// wideGamutInput: if true, input/output are BT.2020 (skip BT.709↔BT.2020 conversions)
vec3 psychoTonemap(vec3 inputColor, bool wideGamutInput,
    float peak_value, float highlights, float shadows, float contrast,
    float purity_scale, float bleaching_intensity, float clip_point,
    float hue_restore, float adaptation_contrast, float white_curve_mode,
    float cone_response_exponent)
{
    const float kEps = 1e-6;
    const float kHalfBleachTrolands = 20000.0;
    const vec3 lms_midgray_raw = psychoLMSFromBT2020(vec3(0.18));
    const float lum_midgray = psychoLuminanceFromLMS(lms_midgray_raw);

    vec3 bt2020 = wideGamutInput ? inputColor : psychoBT2020FromBT709(inputColor);
    vec3 lms_color_raw = psychoLMSFromBT2020(bt2020);

    float lum_current = psychoLuminanceFromLMS(lms_color_raw);
    float lum_target = lum_current;

    if (highlights != 1.0) lum_target = psychoHighlightsFn(lum_target, highlights, lum_midgray);
    if (shadows != 1.0) lum_target = psychoShadowsFn(lum_target, shadows, lum_midgray);
    if (contrast != 1.0) lum_target = psychoContrastSafe(lum_target, contrast, lum_midgray);

    float lum_scale = psychoDivideSafe(lum_target, lum_current, 1.0);
    clip_point *= lum_scale;

    vec3 lms_scene_unit = lms_color_raw * lum_scale;
    vec3 lms_midgray_unit = lms_midgray_raw;

    if (purity_scale != 1.0) {
        lms_scene_unit = psychoScalePurityMB2(lms_scene_unit, purity_scale);
    }

    vec3 lms_scene_unit_source = lms_scene_unit;
    if (adaptation_contrast != 1.0) {
        vec3 lms_sigma_unit = max(lms_midgray_unit, vec3(kEps));
        float exponent = max(adaptation_contrast, kEps);

        vec3 ax = abs(lms_scene_unit);
        vec3 ax_n = pow(ax, vec3(exponent));
        vec3 s_n = pow(lms_sigma_unit, vec3(exponent));
        vec3 response_target = ax_n / max(ax_n + s_n, vec3(kEps));
        vec3 response_baseline = ax / max(ax + lms_sigma_unit, vec3(kEps));
        vec3 gain = response_target / max(response_baseline, vec3(kEps));
        vec3 sign_raw = vec3(
            lms_scene_unit.x < 0.0 ? -1.0 : 1.0,
            lms_scene_unit.y < 0.0 ? -1.0 : 1.0,
            lms_scene_unit.z < 0.0 ? -1.0 : 1.0);
        lms_scene_unit = sign_raw * (ax * gain);

        if (hue_restore > 0.0) {
            lms_scene_unit = psychoRestoreHueMB2(lms_scene_unit_source, lms_scene_unit, hue_restore);
        }
    }

    vec3 lms_unit = lms_scene_unit;
    if (bleaching_intensity != 0.0) {
        float blend_weight = clamp(bleaching_intensity, 0.0, 1.0);
        float adapted_lum = max(psychoLuminanceFromLMS(max(lms_unit, vec3(0.0))), 0.18);
        vec3 adapted_bt2020 = vec3(adapted_lum);
        vec3 lms_adapted_unit = psychoLMSFromBT2020(adapted_bt2020);

        vec3 stimulus_nits = max(lms_adapted_unit, vec3(0.0)) * 100.0;
        vec3 stimulus_trolands = stimulus_nits * 4.0;
        vec3 availability_raw = 1.0 / (1.0 + stimulus_trolands / max(kHalfBleachTrolands, kEps));
        vec3 availability = mix(vec3(1.0), availability_raw, blend_weight);
        lms_unit = lms_unit * max(availability, vec3(0.0));
    }

    vec3 lms_peak_unit = psychoLMSFromBT2020(vec3(peak_value));
    vec3 lms_toned_unit;
    if (white_curve_mode > 0.5) {
        // Naka-Rushton
        lms_toned_unit = psychoNakaRushton(lms_unit, lms_peak_unit, lms_midgray_unit, cone_response_exponent);
    } else if (clip_point > peak_value) {
        // Neutwo with clip
        vec3 lms_clip_unit = psychoLMSFromBT2020(vec3(clip_point));
        lms_toned_unit = psychoNeutwoClipPerChannel(lms_unit, lms_peak_unit, lms_clip_unit);
    } else {
        // Neutwo without clip
        lms_toned_unit = psychoNeutwoPerChannel(lms_unit, lms_peak_unit);
    }

    if (hue_restore > 0.0) {
        lms_toned_unit = psychoRestoreHueMB2(lms_unit, lms_toned_unit, hue_restore);
    }

    lms_toned_unit = psychoGamutCompress(lms_toned_unit);
    vec3 bt2020_toned = psychoBT2020FromLMS(lms_toned_unit);
    return wideGamutInput ? bt2020_toned : psychoBT709FromBT2020(bt2020_toned);
}

// ============================================================================
// sRGB OETF (IEC 61966-2-1)
// ============================================================================
vec3 linearToSRGB(vec3 c) {
    vec3 lo = c * 12.92;
    vec3 hi = 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055;
    return mix(lo, hi, step(vec3(0.0031308), c));
}

// ============================================================================
// HDR10 Output Support (ST.2084 PQ + BT.2020)
// ============================================================================

// Dedicated PQ OETF for HDR10 output — input is normalized [0,1] where 1.0 = 10000 nits
// (Different from the existing PQ_OETF which is used for ICtCp with /100 normalization)
vec3 PQ_OETF_HDR10(vec3 L) {
    const float m1 = 0.1593017578125;    // 2610 / 16384
    const float m2 = 78.84375;            // 2523 / 32 * 128
    const float c1 = 0.8359375;           // 3424 / 4096
    const float c2 = 18.8515625;          // 2413 / 128
    const float c3 = 18.6875;             // 2392 / 128

    vec3 Lm1 = pow(max(L, vec3(0.0)), vec3(m1));
    return pow((c1 + c2 * Lm1) / (1.0 + c3 * Lm1), vec3(m2));
}

void main() {
    vec3 hdr = texture(HDR, texCoord).rgb;
    vec3 expColor = hdr * gExposure.exposure;

    bool hdr10Output = gExposure.hdr10OutputEnabled > 0.5;
    float paperWhite = gExposure.paperWhiteNits;
    float peak = gExposure.peakNits;

    if (hdr10Output) {
        // ====================================================================
        // HDR10 WIDE-GAMUT PIPELINE — works in BT.2020 throughout
        // RT pipeline outputs native BT.2020 — no conversion needed.
        // ====================================================================
        vec3 workingColor = expColor;  // Already BT.2020 from RT pipeline

        // Saturation in Oklab (BT.2020 matrices — direct, no intermediate BT.709)
        if (gExposure.saturation != 1.0) {
            vec3 lab = bt2020ToOklab(max(workingColor, vec3(0.0)));
            lab.yz *= gExposure.saturation;
            vec3 boosted = oklabToBt2020(lab);

            // Gamut clamp: if saturation boost pushed color outside BT.2020 gamut,
            // reduce chroma toward achromatic point until all components >= 0.
            // Prevents purple hue shifts from negative-component clipping on
            // highly saturated BT.2020 colors (blackbody fire/lava).
            float minC = min(boosted.r, min(boosted.g, boosted.b));
            if (minC < 0.0) {
                vec3 gray = oklabToBt2020(vec3(lab.x, 0.0, 0.0));
                // Find max t in [0,1] such that gray + t*(boosted - gray) >= 0
                vec3 d = gray - boosted;
                float t = 1.0;
                if (d.r > 1e-6 && boosted.r < 0.0) t = min(t, gray.r / d.r);
                if (d.g > 1e-6 && boosted.g < 0.0) t = min(t, gray.g / d.g);
                if (d.b > 1e-6 && boosted.b < 0.0) t = min(t, gray.b / d.b);
                boosted = mix(gray, boosted, max(t, 0.0));
            }
            workingColor = boosted;
        }

        float hdrHeadroom = peak / paperWhite;

        vec3 mapped;
        if (gExposure.psychoEnabled > 0.5) {
            // PsychoV: input is BT.2020, output stays BT.2020 (wideGamutInput=true)
            float psychoPeak = peak / paperWhite;
            mapped = psychoTonemap(workingColor, true,
                psychoPeak,
                gExposure.psychoHighlights,
                gExposure.psychoShadows,
                gExposure.psychoContrast,
                gExposure.psychoPurity,
                gExposure.psychoBleaching,
                gExposure.psychoClipPoint,
                gExposure.psychoHueRestore,
                gExposure.psychoAdaptContrast,
                gExposure.psychoWhiteCurve,
                gExposure.psychoConeExponent);
        } else {
            // BT.2390 EETF with BT.2020 luma weights
            mapped = BT2390EETF(workingColor, hdrHeadroom, LUMA_BT2020);
        }

        // Output is already BT.2020 — no gamut conversion needed
        vec3 nits = mapped * paperWhite;
        vec3 bt2020 = clamp(nits, vec3(0.0), vec3(peak));
        vec3 pq = PQ_OETF_HDR10(bt2020 / 10000.0);
        fragColor = vec4(pq, 1.0);

    } else {
        // ====================================================================
        // SDR PIPELINE — process in BT.2020 (identical to HDR), convert to
        // BT.709 only after tonemapping for maximum color fidelity.
        // ====================================================================
        vec3 workingColor = expColor;  // Already BT.2020 from RT pipeline

        // Saturation in Oklab (BT.2020 — identical to HDR path)
        if (gExposure.saturation != 1.0) {
            vec3 lab = bt2020ToOklab(max(workingColor, vec3(0.0)));
            lab.yz *= gExposure.saturation;
            vec3 boosted = oklabToBt2020(lab);

            // Gamut clamp (identical to HDR path)
            float minC = min(boosted.r, min(boosted.g, boosted.b));
            if (minC < 0.0) {
                vec3 gray = oklabToBt2020(vec3(lab.x, 0.0, 0.0));
                vec3 d = gray - boosted;
                float t = 1.0;
                if (d.r > 1e-6 && boosted.r < 0.0) t = min(t, gray.r / d.r);
                if (d.g > 1e-6 && boosted.g < 0.0) t = min(t, gray.g / d.g);
                if (d.b > 1e-6 && boosted.b < 0.0) t = min(t, gray.b / d.b);
                boosted = mix(gray, boosted, max(t, 0.0));
            }
            workingColor = boosted;
        }

        // SDR has no headroom above paper white — 1.0 IS the display peak.
        // PsychoV targets [0,1] with smooth rolloff; BT.2390 hard-clips at 1.0
        // (physically correct for SDR — no super-whites available).
        float sdrHeadroom = 1.0;

        vec3 mapped;
        if (gExposure.psychoEnabled > 0.5) {
            // PsychoV in BT.2020 (wideGamutInput=true, peak=1.0 for SDR)
            mapped = psychoTonemap(workingColor, true,
                sdrHeadroom,
                gExposure.psychoHighlights,
                gExposure.psychoShadows,
                gExposure.psychoContrast,
                gExposure.psychoPurity,
                gExposure.psychoBleaching,
                gExposure.psychoClipPoint,
                gExposure.psychoHueRestore,
                gExposure.psychoAdaptContrast,
                gExposure.psychoWhiteCurve,
                gExposure.psychoConeExponent);
        } else {
            // BT.2390 with sdrHeadroom=1.0: hard-clips at 1.0 (correct for SDR)
            mapped = BT2390EETF(workingColor, sdrHeadroom, LUMA_BT2020);
        }

        // Convert BT.2020 → BT.709 AFTER tonemapping (preserves wide-gamut processing)
        const mat3 BT2020_TO_BT709 = mat3(
             1.6604910, -0.1245505, -0.0181508,
            -0.5876411,  1.1328999, -0.1005789,
            -0.0728499, -0.0083494,  1.1187297);
        mapped = BT2020_TO_BT709 * mapped;
        mapped = clamp(mapped, 0.0, 1.0);

        bool useSrgb = gExposure.sdrTransferFunction > 0.5;
        vec3 encoded = useSrgb ? linearToSRGB(mapped) : pow(mapped, vec3(1.0 / 2.2));

        // Triangular dithering: breaks 8-bit quantization banding (standard in UE5/Frostbite).
        float n1 = fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))));
        float n2 = fract(52.9829189 * fract(dot(gl_FragCoord.xy + vec2(0.5, 0.5), vec2(0.06711056, 0.00583715))));
        vec3 dither = vec3(n1 + n2 - 1.0) / 255.0;
        encoded += dither;

        fragColor = vec4(encoded, 1.0);
    }
}
