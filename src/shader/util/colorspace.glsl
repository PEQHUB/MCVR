#ifndef COLORSPACE_GLSL
#define COLORSPACE_GLSL

// CIE XYZ -> BT.2020 (D65) conversion matrix, column-major for GLSL mat3.
// Derived from BT.2020 primaries and D65 white point (IEC 61966-2-4).
const mat3 CS_XYZ_TO_BT2020 = mat3(
     1.7166511880,  -0.6666843518,  0.0176398574,
    -0.3556707838,   1.6164812366, -0.0427706133,
    -0.2533662814,   0.0157685458,  0.9421031212);

// Blackbody chromaticity approximation (Krystek 1985 / Kim et al. 2012).
// Returns BT.2020 linear color, Y-normalized (max component ≈ 1.0).
// Valid range: 800 K – 10000 K.
vec3 blackbodyBT2020(float T) {
    T = clamp(T, 800.0, 10000.0);
    float x, y;
    if (T <= 4000.0) {
        x = -0.2661239e9/(T*T*T) - 0.2343580e6/(T*T) + 0.8776956e3/T + 0.179910;
        y = -1.1063814*x*x*x - 1.34811020*x*x + 2.18555832*x - 0.20219683;
    } else {
        x = -3.0258469e9/(T*T*T) + 2.1070379e6/(T*T) + 0.2226347e3/T + 0.240390;
        y =  3.0817580*x*x*x - 5.87338670*x*x + 3.75112997*x - 0.37001483;
    }
    float Y = 1.0;
    float X = (y > 1e-6) ? (x / y) : 0.0;
    float Z = (y > 1e-6) ? ((1.0 - x - y) / y) : 0.0;
    vec3 bt2020 = CS_XYZ_TO_BT2020 * vec3(X, Y, Z);
    bt2020 = max(bt2020, vec3(0.0));
    float maxComp = max(max(bt2020.r, bt2020.g), bt2020.b);
    return (maxComp > 1e-6) ? (bt2020 / maxComp) : vec3(1.0);
}

// Full-precision BT.709 -> BT.2020 conversion matrix
// Column-major for GLSL mat3 constructor (matches tone_mapping.frag precision)
const mat3 CS_BT709_TO_BT2020 = mat3(
    0.6274038960, 0.0690972894, 0.0163914389,
    0.3292830383, 0.9195403952, 0.0880133079,
    0.0433130656, 0.0113623156, 0.8955952532);

// BT.2020 luminance coefficients (ITU-R BT.2100)
float luminanceBT2020(vec3 c) {
    return dot(c, vec3(0.2627, 0.6780, 0.0593));
}

// Spectral uplift: boost chroma in BT.2020 space
// Input and output are both BT.2020.
// purity = 0.0: no change. purity = 0.5: 50% chroma boost.
vec3 spectralUplift(vec3 bt2020, float purity) {
    float lum = luminanceBT2020(bt2020);
    vec3 chroma = bt2020 - vec3(lum);
    return vec3(lum) + chroma * (1.0 + purity);
}

#endif
