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
// HDR Mode 0: Hermite Spline Reinhard (BT.2390-aligned)
// Luminance-based Reinhard with cubic Hermite spline knee for smooth
// highlight rolloff. Preserves chrominance ratios.
// References: BT.2390-7, Reinhard et al. 2002
// ============================================================================
vec3 HermiteSplineReinhardToneMap(vec3 color, float Lw) {
    const vec3 LUMA = vec3(0.2126, 0.7152, 0.0722); // BT.709
    float L = dot(color, LUMA);
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
vec3 BT2390EETF(vec3 color, float maxLum) {
    const vec3 LUMA = vec3(0.2126, 0.7152, 0.0722); // BT.709
    float L = dot(color, LUMA);
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

// BT.709 → BT.2020 color matrix (ITU-R BT.2087-0, column-major for GLSL)
const mat3 BT709_TO_BT2020 = mat3(
    0.6274, 0.0691, 0.0164,
    0.3293, 0.9195, 0.0880,
    0.0433, 0.0113, 0.8956
);

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

// Wang hash: fast integer hash for spatial dithering (self-contained, no buffers needed)
uint wangHash(uint seed) {
    seed = (seed ^ 61u) ^ (seed >> 16u);
    seed *= 9u;
    seed = seed ^ (seed >> 4u);
    seed *= 0x27d4eb2du;
    seed = seed ^ (seed >> 15u);
    return seed;
}

void main() {
    vec3 hdr = texture(HDR, texCoord).rgb;
    vec3 expColor = hdr * gExposure.exposure;

    bool hdrPipeline = gExposure.hdrPipelineEnabled > 0.5;
    bool hdr10Output = gExposure.hdr10OutputEnabled > 0.5;

    // Apply Saturation/Vibrance boost (HDR-pipeline only, controlled by slider)
    if (hdrPipeline && gExposure.saturation != 1.0) {
        const vec3 lumaWeight = vec3(0.2126, 0.7152, 0.0722);
        float luma = dot(expColor, lumaWeight);
        expColor = mix(vec3(luma), expColor, gExposure.saturation);
    }

    vec3 mapped;
    int mode = int(gExposure.tonemapMode + 0.5);
    bool isDisplayReferred = false;

    float paperWhite = gExposure.paperWhiteNits;
    float peak = gExposure.peakNits;

    // HDR headroom: highlights can be this many times brighter than paper white
    float hdrHeadroom = hdrPipeline ? (peak / paperWhite) : 1.0;

    // Reinhard Extended in HDR: widen Lwhite so output exceeds 1.0 for highlights
    float effectiveLwhite = hdrPipeline ? max(gExposure.Lwhite, hdrHeadroom) : gExposure.Lwhite;

    if (hdrPipeline) {
        // HDR pipeline behavior: use BT.2390 rolloff in scene-linear space.
        mapped = BT2390EETF(expColor, hdrHeadroom);
    } else {
        // SDR mode: all 8 original operators unchanged
        if (mode == 0)       mapped = PBRNeutralToneMap(expColor);
        else if (mode == 1)  mapped = ReinhardExtendedToneMap(expColor, effectiveLwhite);
        else if (mode == 2)  mapped = ACESToneMap(expColor);
        else if (mode == 3) { mapped = AgXToneMap(expColor); isDisplayReferred = true; }
        else if (mode == 4)  mapped = LottesToneMap(expColor);
        else if (mode == 5)  mapped = FrostbiteToneMap(expColor);
        else if (mode == 6)  mapped = Uncharted2ToneMap(expColor);
        else if (mode == 7)  mapped = GTToneMap(expColor);
        else                 mapped = ReinhardExtendedToneMap(expColor, effectiveLwhite);
    }

    if (hdr10Output) {
        // ═══════════ HDR10 output path ═══════════
        mapped = max(mapped, vec3(0.0));

        if (isDisplayReferred) {
            mapped = pow(mapped, vec3(2.2));  // AgX: undo baked sRGB gamma → linear
        }

        vec3 nits = mapped * paperWhite;

        // Clamp to display peak
        nits = clamp(nits, vec3(0.0), vec3(peak));

        // BT.709 → BT.2020 gamut conversion (in linear light)
        vec3 bt2020 = max(BT709_TO_BT2020 * nits, vec3(0.0));

        // PQ encode: normalize to [0,1] where 1.0 = 10000 nits
        vec3 pq = PQ_OETF_HDR10(bt2020 / 10000.0);

        // ── Anti-banding dither (in PQ space = perceptually uniform) ──
        // Wang hash per-pixel: spatial variation. Temporal accumulation upstream
        // (TAA/SVGF jitter) provides frame-to-frame smoothing automatically.
        uint px = uint(gl_FragCoord.x);
        uint py = uint(gl_FragCoord.y);
        uint seed = px + py * 8192u;
        // 3 independent dither values (one per channel) to decorrelate RGB
        float d0 = float(wangHash(seed))       / 4294967296.0 - 0.5;
        float d1 = float(wangHash(seed + 1u))  / 4294967296.0 - 0.5;
        float d2 = float(wangHash(seed + 2u))  / 4294967296.0 - 0.5;
        // 10-bit step size in [0,1] PQ range = 1/1024
        pq += vec3(d0, d1, d2) * (1.0 / 1024.0);
        pq = clamp(pq, 0.0, 1.0);

        fragColor = vec4(pq, 1.0);
    } else {
        // ═══════════ SDR output path ═══════════
        // Note: When hdrPipeline is enabled but hdr10Output is not, values above 1.0
        // (i.e., above paper white) cannot be represented in SDR and will clip.
        mapped = clamp(mapped, 0.0, 1.0);

        // Encode to sRGB for UNORM output.
        vec3 encoded;
        if (isDisplayReferred) {
            // AgX outputs display-referred (sRGB-like gamma baked in)
            encoded = mapped;
        } else {
            bool useSrgb = gExposure.sdrTransferFunction > 0.5;
            if (useSrgb) {
                encoded = linearToSRGB(mapped);
            } else {
                encoded = pow(mapped, vec3(1.0 / 2.2));
            }
        }

        // Anti-banding dither in encoded (post-OETF) space.
        // 8-bit UNORM step size = 1/255.
        uint px = uint(gl_FragCoord.x);
        uint py = uint(gl_FragCoord.y);
        uint seed = px + py * 8192u;
        float d0 = float(wangHash(seed))       / 4294967296.0 - 0.5;
        float d1 = float(wangHash(seed + 1u))  / 4294967296.0 - 0.5;
        float d2 = float(wangHash(seed + 2u))  / 4294967296.0 - 0.5;
        encoded += vec3(d0, d1, d2) * (1.0 / 255.0);
        encoded = clamp(encoded, 0.0, 1.0);

        fragColor = vec4(encoded, 1.0);
    }
}
