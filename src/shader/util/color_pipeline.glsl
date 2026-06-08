// color_pipeline.glsl — Unified color space framework for RadSER
//
// CENTRALIZED DEFINITIONS: All color space matrices, transfer functions, and
// gamut mapping utilities live here. No other file should define its own matrices.
//
// TWO LMS SPACES — NEVER MIX THEM:
//   1. Oklab LMS (Ottosson basis) — for perceptual operations (saturation, gamut mapping)
//   2. Stockman-Sharp LMS (physiological) — ONLY used by PsychoV tonemapper
//
// WORKING SPACE: BT.2020 linear (D65) throughout the RT pipeline.
// DISPLAY SPACES: BT.709/sRGB (SDR), BT.2020/PQ (HDR10), BT.709/scRGB (HDR).

#ifndef COLOR_PIPELINE_GLSL
#define COLOR_PIPELINE_GLSL

// ============================================================================
// Luminance coefficients
// ============================================================================
const vec3 LUMA_BT709  = vec3(0.2126, 0.7152, 0.0722);
const vec3 LUMA_BT2020 = vec3(0.2627, 0.6780, 0.0593);

// ============================================================================
// BT.709 <-> BT.2020 gamut conversion (D65 white point, column-major GLSL)
// ============================================================================
const mat3 CP_BT709_TO_BT2020 = mat3(
    0.6274038960, 0.0690972894, 0.0163914389,
    0.3292830383, 0.9195403952, 0.0880133079,
    0.0433130656, 0.0113623156, 0.8955952532);

const mat3 CP_BT2020_TO_BT709 = mat3(
     1.6604910, -0.1245505, -0.0181508,
    -0.5876411,  1.1328999, -0.1005789,
    -0.0728499, -0.0083494,  1.1187297);

// ============================================================================
// Oklab (Bjorn Ottosson, 2020) — perceptual uniform color space
// Uses Ottosson's custom LMS basis (NOT Stockman-Sharp)
// ============================================================================

// --- BT.709 path ---
const mat3 CP_BT709_TO_OKLAB_LMS = mat3(
    0.4122214708, 0.2119034982, 0.0883024619,
    0.5363325363, 0.6806995451, 0.2817188376,
    0.0514459929, 0.1073969566, 0.6299787005);

const mat3 CP_OKLAB_LMS_TO_BT709 = mat3(
     4.0767416621, -1.2684380046, -0.0041960863,
    -3.3077115913,  2.6097574011, -0.7034186147,
     0.2309699292, -0.3413193965,  1.7076147010);

// --- BT.2020 path (direct combined, no intermediate BT.709) ---
const mat3 CP_BT2020_TO_OKLAB_LMS = mat3(
    0.6167557871, 0.2651330639, 0.1001026342,
    0.3601983994, 0.6358393640, 0.2039065193,
    0.0230458134, 0.0990275718, 0.6959908464);

const mat3 CP_OKLAB_LMS_TO_BT2020 = mat3(
     2.1399067359, -0.8847358624, -0.0485737581,
    -1.2463895090,  2.1632309822, -0.4545031427,
     0.1064827729, -0.2784951194,  1.5030769008);

// --- Lab <-> LMS (gamut-independent) ---
const mat3 CP_OKLAB_LMS_TO_LAB = mat3(
    0.2104542553,  1.9779984951,  0.0259040371,
    0.7936177850, -2.4285922050,  0.7827717662,
   -0.0040720468,  0.4505937099, -0.8086757660);

const mat3 CP_LAB_TO_OKLAB_LMS = mat3(
    1.0,           1.0,           1.0,
    0.3963377774, -0.1055613458, -0.0894841775,
    0.2158037573, -0.0638541728, -1.2914855480);

// --- Conversion functions ---

vec3 cpBt709ToOklab(vec3 rgb) {
    vec3 lms = CP_BT709_TO_OKLAB_LMS * max(rgb, vec3(0.0));
    vec3 lms_g = sign(lms) * pow(abs(lms), vec3(1.0 / 3.0));
    return CP_OKLAB_LMS_TO_LAB * lms_g;
}

vec3 cpOklabToBt709(vec3 lab) {
    vec3 lms_g = CP_LAB_TO_OKLAB_LMS * lab;
    vec3 lms = lms_g * lms_g * lms_g;
    return CP_OKLAB_LMS_TO_BT709 * lms;
}

vec3 cpBt2020ToOklab(vec3 rgb) {
    vec3 lms = CP_BT2020_TO_OKLAB_LMS * max(rgb, vec3(0.0));
    vec3 lms_g = sign(lms) * pow(abs(lms), vec3(1.0 / 3.0));
    return CP_OKLAB_LMS_TO_LAB * lms_g;
}

vec3 cpOklabToBt2020(vec3 lab) {
    vec3 lms_g = CP_LAB_TO_OKLAB_LMS * lab;
    vec3 lms = lms_g * lms_g * lms_g;
    return CP_OKLAB_LMS_TO_BT2020 * lms;
}

// ============================================================================
// Oklab Gamut Mapping — Ottosson Method 5 (adaptive L0, hue-dependent)
// Perceptually clips out-of-gamut colors preserving hue and adapting lightness.
// ============================================================================

// Compute maximum sRGB chroma at given Oklab L and hue angle h.
// Uses analytical cubic sRGB boundary intersection (simplified for real-time).
float cpOklabMaxChromaSrgb(float L, float h) {
    // For each sRGB primary/secondary edge, find max chroma at (L, h).
    // The sRGB boundary in Oklab is piecewise cubic, but for real-time
    // we use a conservative estimate via binary search in 8 steps.
    vec2 ab_dir = vec2(cos(h), sin(h));
    float lo = 0.0, hi = 0.5;
    for (int i = 0; i < 8; i++) {
        float mid = (lo + hi) * 0.5;
        vec3 lab = vec3(L, ab_dir * mid);
        vec3 rgb = cpOklabToBt709(lab);
        if (rgb.r >= -0.001 && rgb.r <= 1.001 &&
            rgb.g >= -0.001 && rgb.g <= 1.001 &&
            rgb.b >= -0.001 && rgb.b <= 1.001) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return lo;
}

// Soft-clip BT.2020 tone-mapped result into sRGB gamut via Oklab perceptual mapping.
// Replaces hard clamp. Fast path for in-gamut pixels (majority of screen).
vec3 cpGamutMapToSrgb(vec3 bt2020) {
    vec3 bt709 = CP_BT2020_TO_BT709 * bt2020;

    // Fast path: already in gamut (vast majority of pixels)
    float minC = min(bt709.r, min(bt709.g, bt709.b));
    float maxC = max(bt709.r, max(bt709.g, bt709.b));
    if (minC >= 0.0 && maxC <= 1.0) return bt709;

    // Convert to Oklab for perceptual clipping
    vec3 lab = cpBt709ToOklab(bt709);
    float L = lab.x;
    float C = length(lab.yz);
    float h = atan(lab.z, lab.y);

    // Clamp lightness to [0, 1]
    L = clamp(L, 0.0, 1.0);

    if (C < 1e-6) return vec3(clamp(L, 0.0, 1.0));

    // Adaptive L0 (Method 5) — project toward cusp-adapted lightness
    // Uses L=0.5 as hue-independent approximation for speed
    float L_target = 0.5;
    float Ld = L - L_target;
    float alpha = 0.05;
    float e1 = 0.5 + abs(Ld) + alpha * C;
    float L0 = (1.0 + sign(Ld) * (e1 - sqrt(max(e1 * e1 - 2.0 * abs(Ld), 0.0)))) / 2.0;

    // Find max in-gamut chroma at this L and hue
    float C_max = cpOklabMaxChromaSrgb(L, h);

    if (C_max < 1e-6) {
        // No chroma possible at this lightness — return achromatic
        return vec3(clamp(L, 0.0, 1.0));
    }

    if (C <= C_max) {
        // In gamut at this lightness — just rebuild
        lab.x = L;
        return clamp(cpOklabToBt709(lab), 0.0, 1.0);
    }

    // Out of gamut: compress chroma with smooth Reinhard-style rolloff
    // Preserves hue direction, adapts lightness toward L0
    float t = C / C_max;
    float C_new = C_max * (1.0 - 1.0 / (t * 0.5 + 0.5));

    // Adapt lightness: blend toward L0 proportional to compression amount
    float compress_ratio = 1.0 - C_new / C;
    float L_adapted = mix(L, L0, compress_ratio * 0.3);

    vec2 ab_dir = lab.yz / C;
    vec3 clipped_lab = vec3(L_adapted, ab_dir * C_new);
    return clamp(cpOklabToBt709(clipped_lab), 0.0, 1.0);
}

// Simple BT.2020 negative-clip (for HDR path — just remove negative values gracefully)
vec3 cpGamutClipBt2020(vec3 bt2020) {
    float minC = min(bt2020.r, min(bt2020.g, bt2020.b));
    if (minC >= 0.0) return bt2020;
    float luma = dot(bt2020, LUMA_BT2020);
    if (luma <= 0.0) return vec3(0.0);
    float t = luma / (luma - minC);
    return mix(vec3(luma), bt2020, t);
}

// ============================================================================
// Transfer functions
// ============================================================================

// sRGB OETF (IEC 61966-2-1)
vec3 cpLinearToSRGB(vec3 c) {
    vec3 lo = c * 12.92;
    vec3 hi = 1.055 * pow(max(c, vec3(0.0)), vec3(1.0 / 2.4)) - 0.055;
    return mix(lo, hi, step(vec3(0.0031308), c));
}

// PQ OETF — linear to PQ (ST 2084), input normalized [0,1] where 1.0 = 10000 nits
vec3 cpPQ_OETF(vec3 L) {
    const float m1 = 0.1593017578125;
    const float m2 = 78.84375;
    const float c1 = 0.8359375;
    const float c2 = 18.8515625;
    const float c3 = 18.6875;
    vec3 Lm1 = pow(max(L, vec3(0.0)), vec3(m1));
    return pow((c1 + c2 * Lm1) / (1.0 + c3 * Lm1), vec3(m2));
}

// PQ EOTF — PQ to linear, output normalized [0,1] where 1.0 = 10000 nits
vec3 cpPQ_EOTF(vec3 N) {
    const float m1 = 0.1593017578125;
    const float m2 = 78.84375;
    const float c1 = 0.8359375;
    const float c2 = 18.8515625;
    const float c3 = 18.6875;
    vec3 Nm2inv = pow(max(N, vec3(0.0)), vec3(1.0 / m2));
    return pow(max(Nm2inv - c1, vec3(0.0)) / (c2 - c3 * Nm2inv), vec3(1.0 / m1));
}

// ============================================================================
// Gamut Boost (Oklab chroma scaling, Rec.709 input -> BT.2020 output)
// Used by material system for per-block/per-material chroma expansion.
// mode 0 = uniform, mode 1 = saturation-based (self-limiting sigmoid).
// ============================================================================
vec3 cpApplyGamutBoost(vec3 colorRec709, float boostFactor, int mode) {
    // Forward: Rec.709 -> Oklab LMS (MATCHED pair with inverse)
    vec3 lms = CP_BT709_TO_OKLAB_LMS * max(colorRec709, vec3(0.0));
    vec3 lms_g = sign(lms) * pow(abs(lms), vec3(1.0 / 3.0));
    vec3 lab = CP_OKLAB_LMS_TO_LAB * lms_g;

    if (mode == 0) {
        lab.yz *= boostFactor;
    } else {
        float boostAmount = 1.0 - exp2(-4.0 * boostFactor * dot(lab.yz, lab.yz));
        lab.yz *= 1.0 + boostAmount * (boostFactor - 1.0);
    }

    // Inverse: Oklab -> LMS -> BT.2020 (intentional gamut expansion)
    vec3 lms_g2 = CP_LAB_TO_OKLAB_LMS * lab;
    vec3 lms2 = lms_g2 * lms_g2 * lms_g2;
    vec3 result = CP_OKLAB_LMS_TO_BT2020 * lms2;

    // Gamut clamp: desaturate toward grey if outside BT.2020
    float minVal = min(result.r, min(result.g, result.b));
    if (minVal < 0.0) {
        float luma = dot(result, LUMA_BT2020);
        float t = luma / (luma - minVal);
        result = mix(vec3(luma), result, t);
    }
    return result;
}

// Overload without mode parameter (defaults to saturation-based)
vec3 cpApplyGamutBoost(vec3 colorRec709, float boostFactor) {
    return cpApplyGamutBoost(colorRec709, boostFactor, 1);
}

#endif // COLOR_PIPELINE_GLSL
