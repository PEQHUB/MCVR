#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <algorithm>
#include <cmath>

namespace colorspace {

// ============================================================================
// CIE 1931 2-degree Standard Observer Color Matching Functions
// 5nm intervals, 380nm to 780nm (81 entries)
// Source: CIE 15:2004, Table T.1 (standard colorimetric observer)
// ============================================================================

static constexpr float CIE_X[81] = {
    0.001368f, 0.002236f, 0.004243f, 0.007650f, 0.014310f,  // 380-400
    0.023190f, 0.043510f, 0.077630f, 0.134380f, 0.214770f,  // 405-425
    0.283900f, 0.328500f, 0.348280f, 0.348060f, 0.336200f,  // 430-450
    0.318700f, 0.290800f, 0.251100f, 0.195360f, 0.142100f,  // 455-475
    0.095640f, 0.057950f, 0.032010f, 0.014700f, 0.004900f,  // 480-500
    0.002400f, 0.009300f, 0.029100f, 0.063270f, 0.109600f,  // 505-525
    0.165500f, 0.225750f, 0.290400f, 0.359700f, 0.433450f,  // 530-550
    0.512050f, 0.594500f, 0.678400f, 0.762100f, 0.842500f,  // 555-575
    0.916300f, 0.978600f, 1.026300f, 1.056700f, 1.062200f,  // 580-600
    1.045600f, 1.002600f, 0.938400f, 0.854450f, 0.751400f,  // 605-625
    0.642400f, 0.541900f, 0.447900f, 0.360800f, 0.283500f,  // 630-650
    0.218700f, 0.164900f, 0.121200f, 0.087400f, 0.063600f,  // 655-675
    0.046770f, 0.032900f, 0.022700f, 0.015840f, 0.011359f,  // 680-700
    0.008111f, 0.005790f, 0.004109f, 0.002899f, 0.002049f,  // 705-725
    0.001440f, 0.001000f, 0.000690f, 0.000476f, 0.000332f,  // 730-750
    0.000235f, 0.000166f, 0.000117f, 0.000083f, 0.000059f,  // 755-775
    0.000042f                                                 // 780
};

static constexpr float CIE_Y[81] = {
    0.000039f, 0.000064f, 0.000120f, 0.000217f, 0.000396f,  // 380-400
    0.000640f, 0.001210f, 0.002180f, 0.004000f, 0.007300f,  // 405-425
    0.011600f, 0.016840f, 0.023000f, 0.029800f, 0.038000f,  // 430-450
    0.048000f, 0.060000f, 0.073900f, 0.090980f, 0.112600f,  // 455-475
    0.139020f, 0.169300f, 0.208020f, 0.258600f, 0.323000f,  // 480-500
    0.407300f, 0.503000f, 0.608200f, 0.710000f, 0.793200f,  // 505-525
    0.862000f, 0.914850f, 0.954000f, 0.980300f, 0.994950f,  // 530-550
    1.000000f, 0.995000f, 0.978600f, 0.952000f, 0.915400f,  // 555-575
    0.870000f, 0.816300f, 0.757000f, 0.694900f, 0.631000f,  // 580-600
    0.566800f, 0.503000f, 0.441200f, 0.381000f, 0.321000f,  // 605-625
    0.265000f, 0.217000f, 0.175000f, 0.138200f, 0.107000f,  // 630-650
    0.081600f, 0.061000f, 0.044580f, 0.032000f, 0.023200f,  // 655-675
    0.017000f, 0.011920f, 0.008210f, 0.005723f, 0.004102f,  // 680-700
    0.002929f, 0.002091f, 0.001484f, 0.001047f, 0.000740f,  // 705-725
    0.000520f, 0.000361f, 0.000249f, 0.000172f, 0.000120f,  // 730-750
    0.000085f, 0.000060f, 0.000042f, 0.000030f, 0.000021f,  // 755-775
    0.000015f                                                 // 780
};

static constexpr float CIE_Z[81] = {
    0.006450f, 0.010550f, 0.020050f, 0.036210f, 0.067850f,  // 380-400
    0.110200f, 0.207400f, 0.371300f, 0.645600f, 1.039050f,  // 405-425
    1.385600f, 1.622960f, 1.747060f, 1.782600f, 1.772110f,  // 430-450
    1.744100f, 1.669200f, 1.528100f, 1.287640f, 1.041900f,  // 455-475
    0.812950f, 0.616200f, 0.465180f, 0.353300f, 0.272000f,  // 480-500
    0.212300f, 0.158200f, 0.111700f, 0.078250f, 0.057250f,  // 505-525
    0.042160f, 0.029840f, 0.020300f, 0.013400f, 0.008750f,  // 530-550
    0.005750f, 0.003900f, 0.002750f, 0.002100f, 0.001800f,  // 555-575
    0.001650f, 0.001400f, 0.001100f, 0.001000f, 0.000800f,  // 580-600
    0.000600f, 0.000340f, 0.000240f, 0.000190f, 0.000100f,  // 605-625
    0.000050f, 0.000030f, 0.000020f, 0.000010f, 0.000000f,  // 630-650
    0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f,  // 655-675
    0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f,  // 680-700
    0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f,  // 705-725
    0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f,  // 730-750
    0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f,  // 755-775
    0.000000f                                                 // 780
};

// ============================================================================
// Color Space Matrices
// ============================================================================

// XYZ to BT.2020 linear RGB (ITU-R BT.2020)
// Derived from BT.2020 primaries and D65 white point
// Column-major storage (glm::mat3 constructor is column-major)
inline const glm::mat3 XYZ_TO_BT2020 = glm::mat3(
     1.7166511880f, -0.6666843518f,  0.0176398574f,  // col 0
    -0.3556707838f,  1.6164812366f, -0.0427706133f,  // col 1
    -0.2533662814f,  0.0157685458f,  0.9421031212f   // col 2
);

// BT.709 to BT.2020 (full precision, matches tone_mapping.frag)
// Column-major storage (glm::mat3 constructor is column-major)
inline const glm::mat3 BT709_TO_BT2020 = glm::mat3(
    0.6274038960f, 0.0690972894f, 0.0163914389f,
    0.3292830383f, 0.9195403952f, 0.0880133079f,
    0.0433130656f, 0.0113623156f, 0.8955952532f
);

// BT.2020 to BT.709
inline const glm::mat3 BT2020_TO_BT709 = glm::mat3(
     1.6604910f, -0.1245505f, -0.0181508f,
    -0.5876411f,  1.1328999f, -0.1005789f,
    -0.0728499f, -0.0083494f,  1.1187297f
);

// ============================================================================
// Blackbody Radiation via CIE Spectral Integration
// ============================================================================

// Planck spectral radiance integrated against CIE 1931 2-degree observer CMFs.
// Returns Y-normalized XYZ (Y = 1.0).
inline glm::vec3 blackbodyXYZ(float tempK) {
    // Second radiation constant c2 = hc/k (CODATA 2018)
    const float c2 = 14388.0f; // micrometers * Kelvin

    float X = 0.0f, Y = 0.0f, Z = 0.0f;
    for (int i = 0; i < 81; i++) {
        float lambda_nm = 380.0f + static_cast<float>(i) * 5.0f;
        float lambda_um = lambda_nm * 0.001f; // convert nm to micrometers
        float x = c2 / (lambda_um * tempK);

        // Planck's law (relative spectral radiance, dropping constant prefactors
        // that cancel in the Y-normalization)
        float planck;
        if (x > 40.0f) {
            planck = 0.0f; // Avoid overflow in exp()
        } else {
            planck = 1.0f / (std::pow(lambda_um, 5.0f) * std::expm1(x));
        }

        X += planck * CIE_X[i];
        Y += planck * CIE_Y[i];
        Z += planck * CIE_Z[i];
    }

    // Y-normalize: scale so Y = 1.0 (preserves chromaticity)
    float scale = 1.0f / std::max(Y, 1e-10f);
    return glm::vec3(X * scale, 1.0f, Z * scale);
}

// Convert CIE XYZ to BT.2020 linear RGB
inline glm::vec3 xyzToBT2020(glm::vec3 xyz) {
    return XYZ_TO_BT2020 * xyz;
}

// ============================================================================
// Spectral Uplift (Chroma Boost in BT.2020 Space)
// ============================================================================

// Boost chroma in BT.2020 space. Input and output are both BT.2020.
// purity = 0.0: no change. purity = 0.5: 50% chroma boost.
inline glm::vec3 spectralUplift(glm::vec3 bt2020, float purity) {
    // BT.2020 luminance (ITU-R BT.2100)
    float lum = 0.2627f * bt2020.r + 0.6780f * bt2020.g + 0.0593f * bt2020.b;
    glm::vec3 chroma = bt2020 - glm::vec3(lum);
    return glm::vec3(lum) + chroma * (1.0f + purity);
}

// ============================================================================
// Main Entry: Compute Emission Color (returns BT.2020)
// ============================================================================

// Compute emission color in BT.2020.
//   tempK > 0: Blackbody at given temperature, XYZ -> BT.2020 directly
//   tempK == 0: Convert fallbackBt709 (artist BT.709 color) to BT.2020
//   purity > 0: Apply spectral uplift (chroma boost in BT.2020)
inline glm::vec3 computeEmissionColor(float tempK, glm::vec3 fallbackBt709, float purity) {
    glm::vec3 bt2020;

    if (tempK > 0.0f) {
        // Physically accurate blackbody: Planck x CIE CMF -> XYZ -> BT.2020
        glm::vec3 xyz = blackbodyXYZ(tempK);
        bt2020 = xyzToBT2020(xyz);

        // Luminance-match: scale so perceived brightness matches the fallback
        // artist color. blackbodyXYZ is Y-normalized (luminance ≈ 1.0), but
        // the intensity values in lights.hpp were tuned for the old artist colors.
        // Components may exceed 1.0 — that's correct for BT.2020 wide gamut.
        float bbLum = 0.2627f * bt2020.r + 0.6780f * bt2020.g + 0.0593f * bt2020.b;
        glm::vec3 fb2020 = BT709_TO_BT2020 * fallbackBt709;
        float fbLum = 0.2627f * fb2020.r + 0.6780f * fb2020.g + 0.0593f * fb2020.b;
        if (bbLum > 0.0f && fbLum > 0.0f) {
            bt2020 *= fbLum / bbLum;
        }
    } else {
        // Non-thermal: convert artist BT.709 color to BT.2020
        bt2020 = BT709_TO_BT2020 * fallbackBt709;
    }

    // Apply spectral uplift if requested
    if (purity > 0.0f) {
        bt2020 = spectralUplift(bt2020, purity);
    }

    return bt2020;
}

} // namespace colorspace
