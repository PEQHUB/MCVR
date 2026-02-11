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
}
gExposure;

layout(location = 0) in vec2 texCoord;

layout(location = 0) out vec4 fragColor;

// ============================================================================
// Mode 0: PBR Neutral Tone Mapping (Khronos PBR Neutral specification)
// ============================================================================
vec3 PBRNeutralToneMap(vec3 color) {
    float startCompression = 0.76;
    float desaturation = 0.01;

    float x = min(color.r, min(color.g, color.b));
    float offset = (x < 0.08) ? (x - 6.25 * x * x) : 0.04;
    color -= offset;

    float peak = max(color.r, max(color.g, color.b));
    if (peak < startCompression) return color;

    float d = 1.0 - startCompression;
    float newPeak = 1.0 - d * d / (peak + d - startCompression);
    color *= newPeak / peak;

    float g = 1.0 - 1.0 / (desaturation * (peak - newPeak) + 1.0);
    return mix(color, newPeak * vec3(1.0), g);
}

// ============================================================================
// Mode 1: Reinhard Extended Tone Mapping (luminance-based)
// Reinhard et al. 2002, "Photographic Tone Reproduction for Digital Images", eq. 4
// Lwhite is the smallest luminance that maps to pure white
// ============================================================================
vec3 ReinhardExtendedToneMap(vec3 color, float Lw) {
    const vec3 LUMA = vec3(0.2126, 0.7152, 0.0722); // Rec. ITU-R BT.709
    float L_old = dot(color, LUMA);
    if (L_old < 1e-6) return color;
    float Lw2 = Lw * Lw;
    float L_new = L_old * (1.0 + L_old / Lw2) / (1.0 + L_old);
    return color * (L_new / L_old);
}

// ============================================================================
// Mode 2: ACES Filmic Tone Mapping (Narkowicz 2015)
// Stephen Hill's fit of the ACES RRT+ODT
// ============================================================================
vec3 ACESToneMap(vec3 x) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

// ============================================================================
// Mode 3: AgX Tone Mapping (Troy Sobotka, Blender 4.x)
// ============================================================================
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

// AgX "Punchy" look transform (from Blender 4.x)
vec3 agxLook(vec3 val) {
    const vec3 lw = vec3(0.2126, 0.7152, 0.0722);
    float luma = dot(val, lw);

    // Punchy look settings (from Blender)
    vec3 offset = vec3(0.0);
    vec3 slope = vec3(1.0);
    vec3 power = vec3(1.35);
    float sat = 1.4;

    // ASC CDL
    val = pow(val * slope + offset, power);
    return luma + sat * (val - luma);
}

vec3 AgXToneMap(vec3 color) {
    // AgX input transform (BT.709 adjusted)
    const mat3 agxTransform = mat3(
        0.842479062253094,  0.0423282422610123, 0.0423756549057051,
        0.0784335999999992, 0.878468636469772,  0.0784336,
        0.0792237451477643, 0.0791661274605434, 0.879142973793104);
    // AgX inverse transform
    const mat3 agxTransformInv = mat3(
        1.19687900512017,  -0.0528968517574562, -0.0529716355144438,
       -0.0980208811401368, 1.15190312990417,   -0.0980434501171241,
       -0.0990297440797205,-0.0989611768448433,  1.15107367264116);

    const float minEv = -12.47393;
    const float maxEv = 4.026069;

    color = agxTransform * color;
    color = clamp(log2(color), minEv, maxEv);
    color = (color - minEv) / (maxEv - minEv);
    color = agxDefaultContrastApprox(color);

    // Apply punchy look (contrast + saturation)
    color = agxLook(color);

    // Inverse transform
    color = agxTransformInv * color;

    return color;
}

// ============================================================================
// Mode 4: Lottes Tone Mapping (AMD, GDC 2016)
// Timothy Lottes, "Advanced Techniques and Optimization of HDR Color Pipelines"
// ============================================================================
vec3 LottesToneMap(vec3 x) {
    const vec3 a      = vec3(1.6);
    const vec3 d      = vec3(0.977);
    const vec3 hdrMax = vec3(8.0);
    const vec3 midIn  = vec3(0.18);
    const vec3 midOut = vec3(0.267);

    const vec3 b = (-pow(midIn, a) + pow(hdrMax, a) * midOut)
                 / ((pow(hdrMax, a * d) - pow(midIn, a * d)) * midOut);
    const vec3 c = (pow(hdrMax, a * d) * pow(midIn, a) - pow(hdrMax, a) * pow(midIn, a * d) * midOut)
                 / ((pow(hdrMax, a * d) - pow(midIn, a * d)) * midOut);

    return pow(x, a) / (pow(x, a * d) * b + c);
}

// ============================================================================
// Mode 5: Frostbite Tone Mapping (EA DICE)
// "High Dynamic Range Color Grading and Display in Frostbite"
// Uses ICtCp perceptual color space for adaptive desaturation and hue preservation
// Port from renodx HLSL implementation to GLSL
// ============================================================================

// PQ (Perceptual Quantizer) OETF — linear to PQ (ST 2084)
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

// BT.709 to ICtCp conversion (using RenoDX/BT.2100 combined matrices)
// PQ scaling = 100 nits (appropriate for SDR game content)
vec3 bt709ToICtCp(vec3 rgb) {
    // Combined BT.709 → BT.2020 → LMS (GLSL column-major)
    // Source: RenoDX BT709_TO_ICTCP_LMS_MAT, transposed for GLSL
    const mat3 bt709_to_lms = mat3(
        0.295764088,  0.156191974,  0.0351022854,
        0.623072445,  0.727251648,  0.156589955,
        0.0811667516, 0.116557933,  0.808302998);
    // LMS (PQ) to ICtCp — BT.2100 spec (GLSL column-major)
    const mat3 lms_to_ictcp = mat3(
         2048.0 / 4096.0,   6610.0 / 4096.0,  17933.0 / 4096.0,
         2048.0 / 4096.0,  -13613.0 / 4096.0, -17390.0 / 4096.0,
            0.0 / 4096.0,   7003.0 / 4096.0,   -543.0 / 4096.0);

    vec3 lms = bt709_to_lms * max(rgb, vec3(0.0));
    vec3 lmsPQ = PQ_OETF(lms / 100.0);
    return lms_to_ictcp * lmsPQ;
}

// ICtCp to BT.709 conversion (using RenoDX/BT.2100 combined matrices)
vec3 ictcpToBt709(vec3 ictcp) {
    // ICtCp to LMS (PQ) — inverse of BT.2100 LMS→ICtCp (GLSL column-major)
    const mat3 ictcp_to_lms = mat3(
        1.0,                1.0,               1.0,
        0.008609037,       -0.008609037,        0.560031335,
        0.111029625,       -0.111029625,       -0.320627174);
    // Combined LMS → BT.2020 → BT.709 (GLSL column-major)
    // Source: RenoDX ICTCP_LMS_TO_BT709_MAT, transposed for GLSL
    const mat3 lms_to_bt709 = mat3(
         6.17353248, -1.32403194, -0.0115983877,
        -5.32089900,  2.56026983, -0.264921456,
         0.147354885,-0.236238613, 1.27652633);

    vec3 lmsPQ = ictcp_to_lms * ictcp;
    vec3 lms = PQ_EOTF(lmsPQ) * 100.0;
    return lms_to_bt709 * lms;
}

float frostbiteRangeCompress(float x) {
    return 1.0 - exp(-x);
}

float frostbiteRangeCompressThreshold(float val, float threshold) {
    if (val < threshold) {
        return val;
    }
    float range = 1.0 - threshold;
    return threshold + range * frostbiteRangeCompress((val - threshold) / range);
}

vec3 FrostbiteToneMap(vec3 col) {
    const float rolloffStart = 0.25;
    const float satBoostAmount = 0.3;
    const float hueCorrectAmount = 0.6;

    // Convert to ICtCp for perceptual operations
    vec3 ictcp = bt709ToICtCp(col);

    // Adaptive desaturation based on intensity
    float satAmount = pow(smoothstep(1.0, 0.3, ictcp.x), 1.3);
    col = ictcpToBt709(ictcp * vec3(1.0, satAmount, satAmount));

    // Hue-preserving range compression
    float maxCol = max(col.r, max(col.g, col.b));
    float mappedMax = frostbiteRangeCompressThreshold(maxCol, rolloffStart);
    vec3 compressedHP = col * (mappedMax / max(maxCol, 1e-6));

    // Per-channel compression
    vec3 compressedPC = vec3(
        frostbiteRangeCompressThreshold(col.r, rolloffStart),
        frostbiteRangeCompressThreshold(col.g, rolloffStart),
        frostbiteRangeCompressThreshold(col.b, rolloffStart));

    // Blend hue-preserving and per-channel
    col = mix(compressedPC, compressedHP, hueCorrectAmount);

    // Post-compression saturation boost
    vec3 ictcpMapped = bt709ToICtCp(col);
    float postBoost = satBoostAmount * smoothstep(1.0, 0.5, ictcp.x);
    float lumRatio = ictcpMapped.x / max(ictcp.x, 1e-3);
    ictcpMapped.yz = mix(ictcpMapped.yz, ictcp.yz * lumRatio, postBoost);

    col = ictcpToBt709(ictcpMapped);
    return col;
}

// ============================================================================
// Mode 6: Uncharted 2 / Hable Tone Mapping (Naughty Dog, GDC 2010)
// John Hable, "Filmic Tonemapping for Real-time Rendering"
// ============================================================================
vec3 Uncharted2Helper(vec3 x) {
    const float A = 0.15; // Shoulder Strength
    const float B = 0.50; // Linear Strength
    const float C = 0.10; // Linear Angle
    const float D = 0.20; // Toe Strength
    const float E = 0.02; // Toe Numerator
    const float F = 0.30; // Toe Denominator
    return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
}

vec3 Uncharted2ToneMap(vec3 color) {
    const float W = 11.2; // Linear White Point
    const float exposureBias = 2.0;
    vec3 curr = Uncharted2Helper(color * exposureBias);
    vec3 whiteScale = vec3(1.0) / Uncharted2Helper(vec3(W));
    return curr * whiteScale;
}

// ============================================================================
// Mode 7: Gran Turismo Tonemapper (Uchimura 2017, Polyphony Digital)
// Hajime Uchimura, "HDR Theory and Practice", CEDEC 2017
// Three-segment piecewise curve: toe (power) + linear + shoulder (exponential)
// ============================================================================
float GTTonemapChannel(float x) {
    const float P = 1.0;    // Max brightness
    const float a = 1.0;    // Contrast (linear slope)
    const float m = 0.22;   // Linear section start
    const float l = 0.4;    // Linear section length
    const float c = 1.33;   // Black tightness (toe curvature)
    const float b = 0.0;    // Pedestal (black lift)

    float l0 = ((P - m) * l) / a;
    float S0 = m + l0;
    float S1 = m + a * l0;
    float C2 = (a * P) / (P - S1);
    float CP = -C2 / P;

    float w0 = 1.0 - smoothstep(0.0, m, x);
    float w2 = step(m + l0, x);
    float w1 = 1.0 - w0 - w2;

    float T = m * pow(x / m, c) + b;
    float L = m + a * (x - m);
    float S = P - (P - S1) * exp(CP * (x - S0));

    return T * w0 + L * w1 + S * w2;
}

vec3 GTToneMap(vec3 color) {
    return vec3(
        GTTonemapChannel(color.r),
        GTTonemapChannel(color.g),
        GTTonemapChannel(color.b)
    );
}

// ============================================================================
// sRGB OETF (IEC 61966-2-1)
// ============================================================================
vec3 linearToSRGB(vec3 c) {
    vec3 lo = c * 12.92;
    vec3 hi = 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055;
    return mix(lo, hi, step(vec3(0.0031308), c));
}

void main() {
    vec3 hdr = texture(HDR, texCoord).rgb;
    vec3 expColor = hdr * gExposure.exposure;

    vec3 mapped;
    int mode = int(gExposure.tonemapMode + 0.5);
    bool isDisplayReferred = false; // true if tonemapper outputs sRGB-like (skip linearToSRGB)

    if (mode == 0)       mapped = PBRNeutralToneMap(expColor);
    else if (mode == 1)  mapped = ReinhardExtendedToneMap(expColor, gExposure.Lwhite);
    else if (mode == 2)  mapped = ACESToneMap(expColor);
    else if (mode == 3) { mapped = AgXToneMap(expColor); isDisplayReferred = true; }
    else if (mode == 4)  mapped = LottesToneMap(expColor);
    else if (mode == 5)  mapped = FrostbiteToneMap(expColor);
    else if (mode == 6)  mapped = Uncharted2ToneMap(expColor);
    else if (mode == 7)  mapped = GTToneMap(expColor);
    else                 mapped = ReinhardExtendedToneMap(expColor, gExposure.Lwhite);

    mapped = clamp(mapped, 0.0, 1.0);

    // AgX outputs display-referred (sRGB gamma baked in), others output scene-linear
    if (!isDisplayReferred) {
        mapped = pow(mapped, vec3(1.0 / 2.2));
    }

    fragColor = vec4(mapped, 1.0);
}
