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

    // No floor here: negative BT.709 values represent valid BT.2020-gamut colors
    // produced by ICtCp saturation expansion. They are handled by the downstream
    // BT709_TO_BT2020 matrix in the HDR10 output path.
    vec3 lms = bt709_to_lms * rgb;
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

// ============================================================================
// HDR Mode 0: Hermite Spline Reinhard (BT.2390-aligned)
// Luminance-based Reinhard with cubic Hermite spline knee for smooth
// highlight rolloff. Preserves chrominance ratios.
// References: BT.2390-7, Reinhard et al. 2002
// ============================================================================
vec3 HermiteSplineReinhardToneMap(vec3 color, float Lw) {
    const vec3 LUMA = vec3(0.2627, 0.6780, 0.0593); // BT.2020 (ITU-R BT.2020)
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
    const vec3 LUMA = vec3(0.2627, 0.6780, 0.0593); // BT.2020 (ITU-R BT.2020)
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

void main() {
    vec3 hdr = texture(HDR, texCoord).rgb;
    vec3 expColor = hdr * gExposure.exposure;

    bool hdr10Output = gExposure.hdr10OutputEnabled > 0.5;

    // Expand color volume in ICtCp space (perceptually uniform chroma).
    // Scaling Ct/Cp pushes colors into BT.2020 gamut without hue distortion.
    // Out-of-BT.709-gamut values produced here are correctly handled by the
    // downstream BT709_TO_BT2020 matrix in the HDR10 output path.
    if (gExposure.saturation != 1.0) {
        vec3 ictcp = bt709ToICtCp(max(expColor, vec3(0.0)));
        ictcp.yz *= gExposure.saturation;
        expColor = ictcpToBt709(ictcp);
    }

    float paperWhite = gExposure.paperWhiteNits;
    float peak = gExposure.peakNits;

    // HDR headroom: highlights can be this many times brighter than paper white
    float hdrHeadroom = peak / paperWhite;

    // BT.2390 EETF tone mapper — maps scene luminance to display luminance
    vec3 mapped = BT2390EETF(expColor, hdrHeadroom);

    if (hdr10Output) {
        // ═══════════ HDR10 output path ═══════════
        vec3 nits = mapped * paperWhite;

        // BT.709 → BT.2020 gamut conversion (in linear light, before peak clamp).
        // Performing the matrix first allows BT.709-negative-but-BT.2020-valid values
        // (produced by the ICtCp saturation expansion above) to survive correctly.
        vec3 bt2020 = BT709_TO_BT2020 * nits;

        // Clamp to display peak in BT.2020 space
        bt2020 = clamp(bt2020, vec3(0.0), vec3(peak));

        // PQ encode: normalize to [0,1] where 1.0 = 10000 nits
        vec3 pq = PQ_OETF_HDR10(bt2020 / 10000.0);

        fragColor = vec4(pq, 1.0);
    } else {
        // ═══════════ SDR output path ═══════════
        // Note: values above 1.0 (above paper white) cannot be represented in SDR and will clip.
        mapped = clamp(mapped, 0.0, 1.0);

        bool useSrgb = gExposure.sdrTransferFunction > 0.5;
        vec3 encoded = useSrgb ? linearToSRGB(mapped) : pow(mapped, vec3(1.0 / 2.2));

        fragColor = vec4(encoded, 1.0);
    }
}
