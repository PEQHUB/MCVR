#version 460
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_GOOGLE_include_directive : require

#include "common/shared.hpp"
#include "color_pipeline.glsl"

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
    float saturationAdaptive;    // 0.0 = linear chroma multiply, 1.0 = adaptive
    float tonemapParam0;
    float tonemapParam1;
    float tonemapParam2;
    float tonemapParam3;
    float tonemapParam4;
    float tonemapParam5;
    float tonemapParam6;
    float tonemapParam7;
    float bootTimer;             // shader-internal (not used by frag, layout parity with exposure.comp)
}
gExposure;

layout(location = 0) in vec2 texCoord;

layout(location = 0) out vec4 fragColor;

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
    float kneeStart = 0.5 * Lw;
    float kneeEnd = Lw;

    if (L > kneeStart && L < kneeEnd) {
        float t = (L - kneeStart) / (kneeEnd - kneeStart);
        float t2 = t * t;
        float t3 = t2 * t;

        float h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
        float h10 = t3 - 2.0 * t2 + t;
        float h01 = -2.0 * t3 + 3.0 * t2;
        float h11 = t3 - t2;

        float p0 = kneeStart * (1.0 + kneeStart / Lw2) / (1.0 + kneeStart);
        float p1 = kneeEnd * (1.0 + kneeEnd / Lw2) / (1.0 + kneeEnd);

        float span = kneeEnd - kneeStart;
        float m0 = (1.0 + 2.0 * kneeStart / Lw2) / ((1.0 + kneeStart) * (1.0 + kneeStart)) * span;
        float m1 = (1.0 + 2.0 * kneeEnd / Lw2) / ((1.0 + kneeEnd) * (1.0 + kneeEnd)) * span;

        Lr = h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1;
    }

    return color * (Lr / L);
}

// ============================================================================
// HDR Mode 1: ITU EETF (Hermite spline) — BT.2390 Method B / BT.2408/BT.2446
// ============================================================================
vec3 BT2390EETF(vec3 color, float maxLum, vec3 luma) {
    float L = dot(color, luma);
    if (L < 1e-6) return color;

    float Ln = min(L / maxLum, 1.0);

    float ks = max(1.5 * maxLum - 0.5, 0.0) / maxLum;
    ks = clamp(ks, 0.0, 0.99);

    float b = Ln;

    if (Ln >= ks) {
        float t = (Ln - ks) / (1.0 - ks);
        float t2 = t * t;
        float t3 = t2 * t;

        float p0 = ks;
        float p1 = 1.0;
        float m0 = 1.0 * (1.0 - ks);
        float m1 = 0.0;

        b = (2.0 * t3 - 3.0 * t2 + 1.0) * p0
          + (t3 - 2.0 * t2 + t) * m0
          + (-2.0 * t3 + 3.0 * t2) * p1
          + (t3 - t2) * m1;
    }

    float Lout = b * maxLum;
    return color * (Lout / L);
}

// ============================================================================
// PsychoV Tonemapper (RenoDX psycho_test11)
// Perceptual tonemapper operating in Stockman-Sharp LMS cone space with
// MacLeod-Boynton chromaticity. USES STOCKMAN-SHARP LMS — NOT Oklab LMS.
// ============================================================================

// Stockman-Sharp LMS matrices (physiological basis — ONLY for PsychoV)
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

    float desiredLen = offsetLen * max(purity_scale, 0.0);
    bool has_solution;
    float t_max = psychoRayMaxT_BT2020(mb_white, direction / offsetLen, has_solution);
    float maxLen = has_solution ? t_max : offsetLen;
    float clampedLen = min(desiredLen, maxLen * 0.98);
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
        lms_toned_unit = psychoNakaRushton(lms_unit, lms_peak_unit, lms_midgray_unit, cone_response_exponent);
    } else if (clip_point > peak_value) {
        vec3 lms_clip_unit = psychoLMSFromBT2020(vec3(clip_point));
        lms_toned_unit = psychoNeutwoClipPerChannel(lms_unit, lms_peak_unit, lms_clip_unit);
    } else {
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
// SDR Tonemappers — consolidated to 5 modes
// All receive BT.2020 linear input, output BT.2020 linear [0,1].
// Mode 0: PBR Neutral, 1: AgX (Eary Chroma), 2: ACES Hill, 3: Reinhard, 4: PsychoV
// ============================================================================

// Mode 0: Khronos PBR Neutral (default)
vec3 PBRNeutralToneMap(vec3 color) {
    float startCompression = gExposure.tonemapParam0 > 0.0 ? gExposure.tonemapParam0 : 0.76;
    float desaturation = gExposure.tonemapParam1 > 0.0 ? gExposure.tonemapParam1 : 0.15;

    float x = min(color.r, min(color.g, color.b));
    float offset = x < 0.08 ? x - 6.25 * x * x : 0.04;
    color -= offset;

    float peak = max(color.r, max(color.g, color.b));
    if (peak < startCompression) return color;

    float d = 1.0 - startCompression;
    float newPeak = 1.0 - d * d / (peak + d - startCompression);
    color *= newPeak / peak;

    float g = 1.0 - 1.0 / (desaturation * (peak - newPeak) + 1.0);
    return mix(color, vec3(newPeak), g);
}

// Mode 1: AgX Eary Chroma (Troy Sobotka + Eary Chow improvements)
// Upgraded with CIE XYZ E-white reference and look presets.
vec3 agxDefaultContrastApprox(vec3 x) {
    vec3 x2 = x * x;
    vec3 x4 = x2 * x2;
    return + 15.5     * x4 * x2
           - 40.14    * x4 * x
           + 31.96    * x4
           - 6.868    * x2 * x
           + 0.4298   * x2
           + 0.1191   * x
           - 0.00232;
}

vec3 AgXToneMap(vec3 bt2020) {
    // BT.2020 -> BT.709 (AgX designed for Rec.709)
    vec3 color = max(CP_BT2020_TO_BT709 * bt2020, vec3(1e-10));

    // AgX inset matrix (Rec.709 -> AgX log space)
    const mat3 AgX_INSET = mat3(
        0.842479062253094,  0.0423282422610123, 0.0423756549057051,
        0.0784335999999992, 0.878468636469772,  0.0784336,
        0.0792237451477643, 0.0791661274605434, 0.879142973793104);
    color = AgX_INSET * color;

    // Log2 encoding
    const float minEv = -12.47393;
    const float maxEv = 4.026069;
    color = clamp(log2(color), minEv, maxEv);
    color = (color - minEv) / (maxEv - minEv);

    // Sigmoid approximation
    color = agxDefaultContrastApprox(color);

    // AgX Look presets via tonemapParam0:
    // 0 = Base (neutral), 1 = Punchy (vivid), 2 = Golden (warm)
    float look = gExposure.tonemapParam0;

    if (look > 1.5) {
        // Golden: warm cinematic look
        color *= vec3(1.0, 0.9, 0.5);
        color = pow(max(color, vec3(0.0)), vec3(0.8));
        vec3 luma = vec3(dot(color, vec3(0.2126, 0.7152, 0.0722)));
        color = luma + (color - luma) * 0.8;
        color = clamp(color, 0.0, 1.0);
    } else if (look > 0.5) {
        // Punchy: Blender's most popular look (power=1.35, saturation=1.4)
        vec3 pivot = vec3(0.39);
        color = pivot + (color - pivot) * 1.35;
        color = clamp(color, 0.0, 1.0);
        vec3 luma = vec3(dot(color, vec3(0.2126, 0.7152, 0.0722)));
        color = luma + (color - luma) * 1.4;
        color = clamp(color, 0.0, 1.0);
    }
    // else: Base (neutral) — no look applied

    // Extra contrast/saturation fine-tuning (tonemapParam1/2)
    float extraContrast = gExposure.tonemapParam1;
    float extraSaturation = gExposure.tonemapParam2;
    if (extraContrast > 0.0 && extraContrast != 1.0) {
        vec3 pivot = vec3(0.39);
        color = pivot + (color - pivot) * extraContrast;
        color = clamp(color, 0.0, 1.0);
    }
    if (extraSaturation > 0.0 && extraSaturation != 1.0) {
        vec3 luma = vec3(dot(color, vec3(0.2126, 0.7152, 0.0722)));
        color = luma + (color - luma) * extraSaturation;
        color = clamp(color, 0.0, 1.0);
    }

    // AgX outset matrix (AgX -> Rec.709 display)
    const mat3 AgX_OUTSET = mat3(
         1.19687900512017,   -0.0528968517574562, -0.0529716355144438,
        -0.0980208811401368,  1.15190312990417,   -0.0980434066391996,
        -0.0990297440797205, -0.0989611768448433,  1.15107367264116);
    color = max(AgX_OUTSET * color, vec3(0.0));

    // BT.709 -> BT.2020 (returned to pipeline working space)
    return max(CP_BT709_TO_BT2020 * color, vec3(0.0));
}

// Mode 2: ACES Hill fit (Stephen Hill's RRT+ODT approximation)
vec3 ACESHillToneMap(vec3 color) {
    float preExp = gExposure.tonemapParam0 > 0.0 ? gExposure.tonemapParam0 : 1.0;
    color *= preExp;
    float L = dot(color, LUMA_BT2020);
    if (L < 1e-6) return color;
    float Lm = (L * (2.51 * L + 0.03)) / (L * (2.43 * L + 0.59) + 0.14);
    Lm = clamp(Lm, 0.0, 1.0);
    return color * (Lm / L);
}

// Mode 4: Lottes (Timothy Lottes, GDC 2016)
vec3 LottesToneMap(vec3 color) {
    float L = dot(color, LUMA_BT2020);
    if (L < 1e-6) return color;

    float a = gExposure.tonemapParam0 > 0.0 ? gExposure.tonemapParam0 : 2.0;
    float d = gExposure.tonemapParam1 > 0.0 ? gExposure.tonemapParam1 : 1.0;
    float hdrMax = gExposure.tonemapParam2 > 0.0 ? gExposure.tonemapParam2 : 16.0;
    float midIn = gExposure.tonemapParam3 > 0.0 ? gExposure.tonemapParam3 : 0.18;
    float midOut = gExposure.tonemapParam4 > 0.0 ? gExposure.tonemapParam4 : 0.18;

    float b = (-pow(midIn, a) + pow(hdrMax, a) * midOut) /
              ((pow(hdrMax, a * d) - pow(midIn, a * d)) * midOut);
    float c = (pow(hdrMax, a * d) * pow(midIn, a) - pow(hdrMax, a) * pow(midIn, a * d) * midOut) /
              ((pow(hdrMax, a * d) - pow(midIn, a * d)) * midOut);

    float Lm = pow(L, a) / (pow(L, a * d) * b + c);
    Lm = clamp(Lm, 0.0, 1.0);
    return color * (Lm / L);
}

// Mode 5: Frostbite (Sebastien Hillaire, SIGGRAPH 2014/2017)
vec3 FrostbiteToneMap(vec3 color) {
    float L = dot(color, LUMA_BT2020);
    if (L < 1e-6) return color;

    float linearEnd = gExposure.tonemapParam0 > 0.0 ? gExposure.tonemapParam0 : 0.25;
    float shoulderStr = gExposure.tonemapParam1 > 0.0 ? gExposure.tonemapParam1 : 2.0;

    float Lm;
    if (L <= linearEnd) {
        Lm = L;
    } else {
        float x = L - linearEnd;
        float shoulder = (1.0 - linearEnd) * (1.0 - exp(-x * shoulderStr));
        Lm = linearEnd + shoulder;
    }
    Lm = clamp(Lm, 0.0, 1.0);
    return color * (Lm / L);
}

// Mode 6: Uncharted 2 (John Hable's filmic curve)
float hablePartialP(float x, float A, float B, float C, float D, float E, float F) {
    return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
}

vec3 Uncharted2ToneMap(vec3 color) {
    float L = dot(color, LUMA_BT2020);
    if (L < 1e-6) return color;
    float A = gExposure.tonemapParam0 > 0.0 ? gExposure.tonemapParam0 : 0.15;
    float B = gExposure.tonemapParam1 > 0.0 ? gExposure.tonemapParam1 : 0.50;
    float C = gExposure.tonemapParam2 > 0.0 ? gExposure.tonemapParam2 : 0.10;
    float D = gExposure.tonemapParam3 > 0.0 ? gExposure.tonemapParam3 : 0.20;
    float E = gExposure.tonemapParam4 > 0.0 ? gExposure.tonemapParam4 : 0.02;
    float F = gExposure.tonemapParam5 > 0.0 ? gExposure.tonemapParam5 : 0.30;
    float W = gExposure.tonemapParam6 > 0.0 ? gExposure.tonemapParam6 : 11.2;
    float Lm = hablePartialP(L, A, B, C, D, E, F) / hablePartialP(W, A, B, C, D, E, F);
    Lm = clamp(Lm, 0.0, 1.0);
    return color * (Lm / L);
}

// Mode 7: GT Tonemap (Hajime Uchimura, Gran Turismo)
vec3 GTToneMap(vec3 color) {
    float L = dot(color, LUMA_BT2020);
    if (L < 1e-6) return color;

    float P = 1.0;
    float a = gExposure.tonemapParam0 > 0.0 ? gExposure.tonemapParam0 : 1.0;
    float m = gExposure.tonemapParam1 > 0.0 ? gExposure.tonemapParam1 : 0.22;
    float l = gExposure.tonemapParam2 > 0.0 ? gExposure.tonemapParam2 : 0.4;
    float c = gExposure.tonemapParam3 > 0.0 ? gExposure.tonemapParam3 : 1.33;
    float b = gExposure.tonemapParam4;

    float l0 = ((P - m) * l) / a;
    float S0 = m + l0;
    float S1 = m + a * l0;
    float C2 = (a * P) / (P - S1);
    float CP = -C2 / P;

    float w0 = 1.0 - smoothstep(0.0, m, L);
    float w2 = step(m + l0, L);
    float w1 = 1.0 - w0 - w2;

    float T = m * pow(L / m, c) + b;
    float S = P - (P - S1) * exp(CP * (L - S0));
    float Lm = T * w0 + L * w1 + S * w2;
    Lm = clamp(Lm, 0.0, 1.0);

    return color * (Lm / L);
}

// Mode 1: Reinhard Extended (luminance-based, Lwhite configurable)
vec3 ReinhardExtToneMap(vec3 color) {
    float Lwhite = gExposure.tonemapParam0 > 0.0 ? gExposure.tonemapParam0 : gExposure.Lwhite;
    float L = dot(color, LUMA_BT2020);
    if (L < 1e-6) return color;
    float Lw2 = Lwhite * Lwhite;
    float Lm = L * (1.0 + L / Lw2) / (1.0 + L);
    Lm = clamp(Lm, 0.0, 1.0);
    return color * (Lm / L);
}

// ============================================================================
// HDR10 Output Support (ST.2084 PQ + BT.2020)
// ============================================================================

void main() {
    vec3 hdr = texture(HDR, texCoord).rgb;
    vec3 expColor = hdr * gExposure.exposure;

    // hdr10OutputEnabled: 0.0 = SDR, 1.0 = HDR10 (PQ), 2.0 = scRGB (linear)
    bool hdr10Output = gExposure.hdr10OutputEnabled > 0.5 && gExposure.hdr10OutputEnabled < 1.5;
    bool scrgbOutput = gExposure.hdr10OutputEnabled > 1.5;
    bool anyHdrOutput = hdr10Output || scrgbOutput;
    float paperWhite = gExposure.paperWhiteNits;
    float peak = gExposure.peakNits;

    if (anyHdrOutput) {
        // ====================================================================
        // HDR WIDE-GAMUT PIPELINE — works in BT.2020 throughout
        // Output: HDR10 (PQ-encoded BT.2020) or scRGB (linear BT.709)
        // ====================================================================
        vec3 workingColor = expColor;

        // Saturation in Oklab (BT.2020 matrices — direct, no intermediate BT.709)
        if (gExposure.saturation != 1.0) {
            vec3 lab = cpBt2020ToOklab(max(workingColor, vec3(0.0)));

            float sat = gExposure.saturation;
            if (sat <= 1.0) {
                lab.yz *= sat;
            } else if (gExposure.saturationAdaptive > 0.5) {
                float chroma = length(lab.yz);
                float L = lab.x;
                float adaptAmount = (1.0 - exp2(-4.0 * chroma * chroma))
                                  * (1.0 - exp2(-4.0 * L * L));
                lab.yz *= 1.0 + adaptAmount * (sat - 1.0);
            } else {
                float boostAmount = 1.0 - exp2(-4.0 * dot(lab.yz, lab.yz));
                lab.yz *= 1.0 + boostAmount * (sat - 1.0);
            }

            workingColor = max(cpOklabToBt2020(lab), vec3(0.0));
        }

        float hdrHeadroom = peak / paperWhite;

        // HDR tonemapper: 0 = PsychoVisual, 1 = BT.2390 EETF (industry standard)
        vec3 mapped;
        float hdrMode = gExposure.psychoEnabled;
        if (hdrMode < 0.5) {
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
            mapped = BT2390EETF(workingColor, hdrHeadroom, LUMA_BT2020);
        }

        // Gamut clip BT.2020 (remove negative values gracefully)
        mapped = cpGamutClipBt2020(mapped);

        if (scrgbOutput) {
            // scRGB: output linear BT.709 values, SDR white = 1.0, HDR peak = peakNits/80
            vec3 bt709 = CP_BT2020_TO_BT709 * mapped;
            float scrgbScale = paperWhite / 80.0; // 80 nits = scRGB reference white
            fragColor = vec4(bt709 * scrgbScale, 1.0);
        } else {
            // HDR10: PQ-encoded BT.2020
            vec3 nits = mapped * paperWhite;
            vec3 bt2020 = clamp(nits, vec3(0.0), vec3(peak));
            vec3 pq = cpPQ_OETF(bt2020 / 10000.0);
            fragColor = vec4(pq, 1.0);
        }

    } else {
        // ====================================================================
        // SDR PIPELINE — process in BT.2020, convert to BT.709 after tonemapping
        // ====================================================================
        vec3 workingColor = expColor;

        // Pre-tonemap saturation in Oklab (BT.2020)
        if (gExposure.saturation != 1.0) {
            vec3 lab = cpBt2020ToOklab(max(workingColor, vec3(0.0)));

            if (gExposure.saturationAdaptive > 0.5) {
                float chroma = length(lab.yz);
                float L = lab.x;
                float adaptAmount = (1.0 - exp2(-4.0 * chroma * chroma))
                                  * (1.0 - exp2(-4.0 * gExposure.saturation * L * L));
                lab.yz *= 1.0 + adaptAmount * (gExposure.saturation - 1.0);
            } else {
                float boostAmount = 1.0 - exp2(-4.0 * gExposure.saturation * dot(lab.yz, lab.yz));
                lab.yz *= 1.0 + boostAmount * (gExposure.saturation - 1.0);
            }

            workingColor = max(cpOklabToBt2020(lab), vec3(0.0));
        }

        // SDR tonemapping — 9 modes (original numbering preserved)
        vec3 mapped;
        float mode = gExposure.tonemapMode;
        if      (mode < 0.5) mapped = PBRNeutralToneMap(workingColor);       // 0: PBR Neutral
        else if (mode < 1.5) mapped = ReinhardExtToneMap(workingColor);      // 1: Reinhard
        else if (mode < 2.5) mapped = ACESHillToneMap(workingColor);         // 2: ACES
        else if (mode < 3.5) mapped = AgXToneMap(workingColor);              // 3: AgX
        else if (mode < 4.5) mapped = LottesToneMap(workingColor);           // 4: Lottes
        else if (mode < 5.5) mapped = FrostbiteToneMap(workingColor);        // 5: Frostbite
        else if (mode < 6.5) mapped = Uncharted2ToneMap(workingColor);       // 6: Uncharted 2
        else if (mode < 7.5) mapped = GTToneMap(workingColor);               // 7: GT
        else                 mapped = psychoTonemap(workingColor, true,      // 8: PsychoVisual
            1.0,
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

        // Post-tonemap gamut mapping: BT.2020 -> BT.709 with Oklab perceptual soft-clip
        // Replaces hard clamp — preserves hue and adapts lightness for out-of-gamut colors
        mapped = cpGamutMapToSrgb(mapped);

        bool useSrgb = gExposure.sdrTransferFunction > 0.5;
        vec3 encoded = useSrgb ? cpLinearToSRGB(mapped) : pow(max(mapped, vec3(0.0)), vec3(1.0 / 2.2));

        // Triangular dithering: breaks 8-bit quantization banding
        float n1 = fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))));
        float n2 = fract(52.9829189 * fract(dot(gl_FragCoord.xy + vec2(0.5, 0.5), vec2(0.06711056, 0.00583715))));
        vec3 dither = vec3(n1 + n2 - 1.0) / 255.0;
        encoded += dither;

        fragColor = vec4(encoded, 1.0);
    }
}
