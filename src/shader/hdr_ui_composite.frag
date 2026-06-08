#version 460

layout(set = 0, binding = 0) uniform sampler2D worldHDR;   // World (A2B10G10R10, PQ-encoded)
layout(set = 0, binding = 1) uniform sampler2D overlayUI;   // UI overlay (R8G8B8A8_SRGB, alpha)

layout(push_constant) uniform PC {
    float uiBrightnessNits;
} pc;

layout(location = 0) in vec2 texCoord;
layout(location = 0) out vec4 fragColor;

// ST.2084 PQ OETF: linear [0,1] (where 1.0 = 10000 nits) → PQ code value
vec3 PQ_OETF(vec3 L) {
    const float m1 = 0.1593017578125;
    const float m2 = 78.84375;
    const float c1 = 0.8359375;
    const float c2 = 18.8515625;
    const float c3 = 18.6875;
    vec3 Lm1 = pow(max(L, vec3(0.0)), vec3(m1));
    return pow((c1 + c2 * Lm1) / (1.0 + c3 * Lm1), vec3(m2));
}

// ST.2084 PQ EOTF: PQ code value → linear [0,1] (where 1.0 = 10000 nits)
vec3 PQ_EOTF(vec3 N) {
    const float m1 = 0.1593017578125;
    const float m2 = 78.84375;
    const float c1 = 0.8359375;
    const float c2 = 18.8515625;
    const float c3 = 18.6875;
    vec3 Nm2inv = pow(max(N, vec3(0.0)), vec3(1.0 / m2));
    return pow(max(Nm2inv - c1, vec3(0.0)) / (c2 - c3 * Nm2inv), vec3(1.0 / m1));
}

// BT.709 → BT.2020 (ITU-R BT.2087-0, column-major for GLSL)
const mat3 BT709_TO_BT2020 = mat3(
    0.6274, 0.0691, 0.0164,
    0.3293, 0.9195, 0.0880,
    0.0433, 0.0113, 0.8956
);

// BT.2020 → BT.709 (inverse of above, column-major for GLSL)
const mat3 BT2020_TO_BT709 = mat3(
    1.6605, -0.1246, -0.0182,
   -0.5877,  1.1330, -0.1006,
   -0.0728, -0.0084,  1.1187
);

void main() {
    vec4 uiSample = texture(overlayUI, texCoord);  // sRGB-decoded by sampler (linear BT.709)
    float alpha = uiSample.a;

    if (alpha < 0.001) {
        // No UI pixel: passthrough world PQ.
        fragColor = vec4(texture(worldHDR, texCoord).rgb, 1.0);
        return;
    }

    // Decode world from PQ to linear nits/10000 (BT.2020 space)
    vec3 worldPQ = texture(worldHDR, texCoord).rgb;
    vec3 worldLinear2020 = PQ_EOTF(worldPQ);  // linear [0,1] where 1.0 = 10000 nits

    // Convert world from BT.2020 to BT.709 for blending
    vec3 worldBt709 = BT2020_TO_BT709 * worldLinear2020;

    // UI is already linearized by the SRGB sampler (BT.709 space)
    // Scale UI linear values to nit level, normalize to PQ range
    vec3 uiLinear = uiSample.rgb;
    vec3 uiNits = uiLinear * (pc.uiBrightnessNits / 10000.0);

    // Alpha blend in BT.709 space (both sources now in same color space)
    // This avoids the BT.709→BT.2020 matrix amplifying 8-bit quantization errors
    vec3 blendedBt709 = mix(worldBt709, uiNits, alpha);

    // Convert blended result back to BT.2020 for PQ encoding
    vec3 blendedBt2020 = BT709_TO_BT2020 * blendedBt709;

    // PQ encode the final composite
    vec3 pq = PQ_OETF(max(blendedBt2020, vec3(0.0)));

    fragColor = vec4(pq, 1.0);
}
