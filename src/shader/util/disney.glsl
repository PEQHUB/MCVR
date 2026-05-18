/*
 * MIT License
 *
 * Copyright(c) 2019 Asif Ali
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/* References:
 * [1] [Physically Based Shading at Disney]
 * https://media.disneyanimation.com/uploads/production/publication_asset/48/asset/s2012_pbs_disney_brdf_notes_v3.pdf
 * [2] [Extending the Disney BRDF to a BSDF with Integrated Subsurface Scattering]
 * https://blog.selfshadow.com/publications/s2015-shading-course/burley/s2015_pbs_disney_bsdf_notes.pdf [3] [The Disney
 * BRDF Explorer] https://github.com/wdas/brdf/blob/main/src/brdfs/disney.brdf [4] [Miles Macklin's implementation]
 * https://github.com/mmacklin/tinsel/blob/master/src/disney.h [5] [Simon Kallweit's project report]
 * http://simon-kallweit.me/rendercompo2015/report/ [6] [Microfacet Models for Refraction through Rough Surfaces]
 * https://www.cs.cornell.edu/~srm/publications/EGSR07-btdf.pdf [7] [Sampling the GGX Distribution of Visible Normals]
 * https://jcgt.org/published/0007/04/01/paper.pdf [8] [Pixar's Foundation for Materials]
 * https://graphics.pixar.com/library/PxrMaterialsCourse2017/paper.pdf [9] [Mitsuba 3]
 * https://github.com/mitsuba-renderer/mitsuba3
 * [10] [Kulla & Conty 2017 — Revisiting Physically Based Shading at Imageworks]
 * https://blog.selfshadow.com/publications/s2017-shading-course/imageworks/s2017_pbs_imageworks_slides_v2.pdf
 * [11] [EON — Energy-Preserving Rough Diffuse BRDF (Portsmouth, Kutz, Hill 2024)]
 * https://arxiv.org/html/2410.18026v2
 */

/*
 * Modifications:
 * - Copyright (c) 2026 Radiance
 * - Changes: Applied to LabPBR materials, added Kulla-Conty multi-scatter GGX
 *   compensation [10] and EON energy-preserving diffuse [11] via precomputed LUT.
 *
 * Note: Original license notice and permission notice are retained per MIT License.
 */

#include "labpbr.glsl"
#include "random.glsl"
#include "common/shared.hpp"

#ifndef DISNEY_GLSL
#    define DISNEY_GLSL

// BRDF feature flag bits (from push constant flags field)
#define BRDF_FLAG_MULTISCATTER_GGX 128   // bit 7
#define BRDF_FLAG_EON_DIFFUSE      256   // bit 8

// Energy compensation LUT (64x64 RGBA16F, baked at init)
// R = GGX E(NdotV, alpha), G = FON E(NdotV, r), B = GGX E_avg, A = FON E_avg
// UV: u = NdotV [0,1], v = perceptual roughness r [0,1] (alpha = r*r)
layout(set = 2, binding = 4) uniform sampler2D energyLUT;

vec3 ToWorld(vec3 X, vec3 Y, vec3 Z, vec3 V) {
    return V.x * X + V.y * Y + V.z * Z;
}

vec3 ToLocal(vec3 X, vec3 Y, vec3 Z, vec3 V) {
    return vec3(dot(V, X), dot(V, Y), dot(V, Z));
}

void Onb(in vec3 N, inout vec3 T, inout vec3 B) {
    vec3 up = abs(N.z) < 0.9999999 ? vec3(0, 0, 1) : vec3(1, 0, 0);
    T = normalize(cross(up, N));
    B = cross(N, T);
}

float SchlickWeight(float u) {
    float m = clamp(1.0 - u, 0.0, 1.0);
    float m2 = m * m;
    return m2 * m2 * m;
}

vec3 SampleGGXVNDF(vec3 V, float ax, float ay, float r1, float r2) {
    vec3 Vh = normalize(vec3(ax * V.x, ay * V.y, V.z));

    float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
    vec3 T1 = lensq > 0 ? vec3(-Vh.y, Vh.x, 0) * inversesqrt(lensq) : vec3(1, 0, 0);
    vec3 T2 = cross(Vh, T1);

    float r = sqrt(r1);
    float phi = 2.0 * PI * r2;
    float t1 = r * cos(phi);
    float t2 = r * sin(phi);
    float s = 0.5 * (1.0 + Vh.z);
    t2 = (1.0 - s) * sqrt(1.0 - t1 * t1) + s * t2;

    vec3 Nh = t1 * T1 + t2 * T2 + sqrt(max(0.0, 1.0 - t1 * t1 - t2 * t2)) * Vh;

    return normalize(vec3(ax * Nh.x, ay * Nh.y, max(0.0, Nh.z)));
}

float GTR2Aniso(float NDotH, float HDotX, float HDotY, float ax, float ay) {
    float a = HDotX / ax;
    float b = HDotY / ay;
    float c = a * a + b * b + NDotH * NDotH;
    return 1.0 / (PI * ax * ay * c * c);
}

float SmithGAniso(float NDotV, float VDotX, float VDotY, float ax, float ay) {
    float a = VDotX * ax;
    float b = VDotY * ay;
    float c = NDotV;
    return (2.0 * NDotV) / (NDotV + sqrt(a * a + b * b + c * c));
}

vec3 SampleVMF(inout uint seed, vec3 mu, float kappa, vec2 xi) {
    kappa = max(kappa, 0.1);
    bool useBN = (xi.x >= 0.0);

    float u1 = useBN ? xi.x : rand(seed);
    float expNeg2K = exp(-2.0 * kappa);
    float w = 1.0 + (1.0 / kappa) * log(u1 + (1.0 - u1) * expNeg2K);

    w = clamp(w, -1.0, 1.0);

    float u2 = useBN ? xi.y : rand(seed);
    float phi = 2.0 * PI * u2;
    float vx = cos(phi);
    float vy = sin(phi);

    // Robust tangent basis. The previous basis construction degenerates for mu ~= (0, +/-1, 0)
    // (cross products collapse), which caused moon/sun overhead to intermittently produce NaNs.
    vec3 u_basis;
    if (abs(mu.y) > 0.99) {
        // mu is near +/-Y; use Z as the helper axis
        u_basis = normalize(cross(vec3(0.0, 0.0, 1.0), mu));
    } else {
        // general case; use Y as the helper axis
        u_basis = normalize(cross(vec3(0.0, 1.0, 0.0), mu));
    }
    vec3 v_basis = normalize(cross(mu, u_basis));

    float sinTheta = sqrt(max(0.0, 1.0 - w * w));
    return normalize(w * mu + sinTheta * (vx * u_basis + vy * v_basis));
}

// PCG fallback overload
vec3 SampleVMF(inout uint seed, vec3 mu, float kappa) {
    return SampleVMF(seed, mu, kappa, vec2(-1.0));
}

float Luminance(vec3 c) {
    return 0.2627 * c.x + 0.6780 * c.y + 0.0593 * c.z;
}

float DielectricFresnel(float cosThetaI, float eta) {
    float sinThetaTSq = eta * eta * (1.0f - cosThetaI * cosThetaI);

    // Total internal reflection
    if (sinThetaTSq > 1.0) return 1.0;

    float cosThetaT = sqrt(max(1.0 - sinThetaTSq, 0.0));

    float rs = (eta * cosThetaT - cosThetaI) / (eta * cosThetaT + cosThetaI);
    float rp = (eta * cosThetaI - cosThetaT) / (eta * cosThetaI + cosThetaT);

    return 0.5f * (rs * rs + rp * rp);
}

vec3 CosineSampleHemisphere(float r1, float r2) {
    vec3 dir;
    float r = sqrt(r1);
    float phi = TWO_PI * r2;
    dir.x = r * cos(phi);
    dir.y = r * sin(phi);
    dir.z = sqrt(max(0.0, 1.0 - dir.x * dir.x - dir.y * dir.y));
    return dir;
}

// ============================================================================
// Energy LUT sampling helpers
// LUT UV: u = NdotV, v = perceptual roughness r (alpha = r*r)
// ============================================================================

// Sample GGX directional albedo E(mu, alpha) and average E_avg(alpha)
void sampleGGXEnergy(float NdotV, float alpha, out float E, out float E_avg) {
    float r = sqrt(max(alpha, 0.0)); // alpha -> perceptual roughness
    vec4 lut = texture(energyLUT, vec2(NdotV, r));
    E = lut.r;
    E_avg = lut.b;
}

// Sample FON directional albedo E_F(mu, r) and average E_F_avg(r)
void sampleFONEnergy(float NdotV, float r, out float E, out float E_avg) {
    vec4 lut = texture(energyLUT, vec2(NdotV, r));
    E = lut.g;
    E_avg = lut.a;
}

// ============================================================================
// DisneyEval — Main BRDF evaluation
// brdfFlags: push constant flags (0 = original behavior, no enhancements)
// ============================================================================

vec3 DisneyEval(LabPBRMat mat, vec3 V, vec3 N, vec3 L, out float pdf, int brdfFlags) {
    pdf = 0.0;
    vec3 f = vec3(0.0);

    vec3 T, B;
    Onb(N, T, B);

    vec3 localV = ToLocal(T, B, N, V);
    vec3 localL = ToLocal(T, B, N, L);

    // in / out
    float eta = (dot(V, N) > 0.0) ? (1.0 / mat.ior) : mat.ior;

    vec3 localH;
    if (localL.z > 0.0)
        localH = normalize(localL + localV);
    else
        localH = normalize(localL + localV * eta);

    if (localH.z < 0.0) localH = -localH;

    // Anisotropic roughness (Disney 2015 parameterization)
    float a = max(mat.roughness, 1e-4);
    float aspect = sqrt(1.0 - 0.9 * mat.anisotropic);
    float ax = max(a / aspect, 1e-4);
    float ay = max(a * aspect, 1e-4);

    // Model weights
    float dielectricWeight = (1.0 - mat.metallic) * (1.0 - mat.transmission);
    float metalWeight = mat.metallic;
    float glassWeight = (1.0 - mat.metallic) * mat.transmission;

    // Lobe probabilities
    float schlickWeight = SchlickWeight(abs(localV.z));

    float diffPr = dielectricWeight * Luminance(mat.albedo);
    float dielectricPr = dielectricWeight * Luminance(mix(mat.f0, vec3(1.0), schlickWeight));
    float metalPr = metalWeight * Luminance(mix(mat.albedo, vec3(1.0), schlickWeight));
    float glassPr = glassWeight;
    float coatPr = mat.coatWeight * 0.25;

    float invTotalWeight = 1.0 / (diffPr + dielectricPr + metalPr + glassPr + coatPr + 1e-5);
    diffPr *= invTotalWeight;
    dielectricPr *= invTotalWeight;
    metalPr *= invTotalWeight;
    glassPr *= invTotalWeight;
    coatPr *= invTotalWeight;

    bool reflect = localL.z * localV.z > 0.0;
    float tmpPdf = 0.0;
    float VDotH = abs(dot(localV, localH));

    // Multi-scatter GGX precomputation (shared across specular lobes)
    bool msGGX = (brdfFlags & BRDF_FLAG_MULTISCATTER_GGX) != 0;
    float msAlphaEff = sqrt(ax * ay); // effective isotropic alpha for anisotropic
    float msE_o = 0.0, msE_i = 0.0, msE_avg = 0.0, ms_fms = 0.0;
    if (msGGX && reflect) {
        float dummy;
        sampleGGXEnergy(abs(localV.z), msAlphaEff, msE_o, msE_avg);
        sampleGGXEnergy(abs(localL.z), msAlphaEff, msE_i, dummy);
        ms_fms = (1.0 - msE_o) * (1.0 - msE_i) / max(PI * (1.0 - msE_avg), 1e-5);
    }

    // Diffuse (+ sheen, which rides on diffuse sampling)
    if (diffPr > 0.0 && reflect) {
        bool eonEnabled = (brdfFlags & BRDF_FLAG_EON_DIFFUSE) != 0;

        if (eonEnabled) {
            // EON energy-preserving rough diffuse [11]
            float r = sqrt(max(mat.roughness, 0.0)); // perceptual roughness

            // FON base (Fujii Oren-Nayar)
            float A_F = 1.0 / (1.0 + (0.5 - 2.0 / (3.0 * PI)) * r);
            float B_F = r * A_F;

            float mu_i = abs(localL.z);
            float mu_o = abs(localV.z);
            float s = dot(localL, localV) - mu_i * mu_o;
            float t = (s > 0.0) ? max(mu_i, mu_o) : 1.0;
            float f_FON_value = INV_PI * (A_F + B_F * s / t);

            // Multi-scatter compensation term (LUT-based)
            float E_o, E_avg_d_o;
            sampleFONEnergy(mu_o, r, E_o, E_avg_d_o);
            float E_i, E_avg_d_i;
            sampleFONEnergy(mu_i, r, E_i, E_avg_d_i);
            float E_avg_d = E_avg_d_o; // same for both (same roughness row)

            float rho = Luminance(mat.albedo);
            float f_ms_d = INV_PI * (rho * rho * E_avg_d)
                         / max(1.0 - rho * (1.0 - E_avg_d), 1e-5)
                         * (1.0 - E_o) * (1.0 - E_i)
                         / max(1.0 - E_avg_d, 1e-5);

            vec3 diffuseColor = mat.albedo * (f_FON_value + f_ms_d);

            // Diffuse-specular energy coupling: attenuate by F0-weighted specular E
            if (msGGX) {
                float f0_scalar = Luminance(mat.f0);
                diffuseColor *= (1.0 - f0_scalar * msE_o) * (1.0 - f0_scalar * msE_i);
            }

            f += diffuseColor * dielectricWeight;
        } else {
            // Original Disney diffuse
            float LDotH = dot(localL, localH);
            float Rr = 2.0 * mat.roughness * LDotH * LDotH;
            float FL = SchlickWeight(localL.z);
            float FV = SchlickWeight(localV.z);
            float Fretro = Rr * (FL + FV + FL * FV * (Rr - 1.0));
            float Fd = (1.0 - 0.5 * FL) * (1.0 - 0.5 * FV);

            // Fake subsurface
            float Fss90 = 0.5 * Rr;
            float Fss = mix(1.0, Fss90, FL) * mix(1.0, Fss90, FV);
            float denom = localL.z + localV.z;
            float ss = (denom > 1e-4) ? 1.25 * (Fss * (1.0 / denom - 0.5) + 0.5) : 1.0;

            vec3 diffuseColor = INV_PI * mat.albedo * mix(Fd + Fretro, ss, mat.subSurface);

            f += diffuseColor * dielectricWeight;
        }

        pdf += (localL.z * INV_PI) * diffPr; // Cosine weighted PDF

        // Sheen lobe (Disney 2015 — retroreflective fabric sheen)
        if (mat.sheenWeight > 0.0) {
            float LDotH_sheen = dot(localL, localH);
            float FH = SchlickWeight(abs(LDotH_sheen));
            float lum = Luminance(mat.albedo);
            vec3 Ctint = lum > 0.0 ? mat.albedo / lum : vec3(1.0);
            vec3 Csheen = mix(vec3(1.0), Ctint, mat.sheenTint);
            f += mat.sheenWeight * Csheen * FH * dielectricWeight;
        }
    }

    // Dielectric Reflection (anisotropic GGX)
    if (dielectricPr > 0.0 && reflect) {
        float F = DielectricFresnel(VDotH, 1.0 / mat.ior);

        float D = GTR2Aniso(localH.z, localH.x, localH.y, ax, ay);
        float G1 = SmithGAniso(abs(localV.z), localV.x, localV.y, ax, ay);
        float G2 = G1 * SmithGAniso(abs(localL.z), localL.x, localL.y, ax, ay);

        tmpPdf = G1 * D / (4.0 * localV.z);
        vec3 specColor = vec3(F) * D * G2 / (4.0 * localL.z * localV.z);

        // Kulla-Conty multi-scatter compensation for dielectrics
        if (msGGX) {
            // F_avg uses F0 (normal-incidence), NOT the angle-dependent F(theta)
            float f0_dielectric = pow((1.0 - mat.ior) / (1.0 + mat.ior), 2.0);
            float F_avg_d = (20.0 / 21.0) * f0_dielectric + (1.0 / 21.0);
            float F_ms_d = (F_avg_d * msE_avg) / max(1.0 - F_avg_d * (1.0 - msE_avg), 1e-5);
            specColor += vec3(F_ms_d * ms_fms);
        }

        f += specColor * dielectricWeight;
        pdf += tmpPdf * dielectricPr;
    }

    // Metallic Reflection (anisotropic GGX)
    if (metalPr > 0.0 && reflect) {
        vec3 FMetal = mix(mat.albedo, vec3(1.0), SchlickWeight(VDotH));

        float D = GTR2Aniso(localH.z, localH.x, localH.y, ax, ay);
        float G1 = SmithGAniso(abs(localV.z), localV.x, localV.y, ax, ay);
        float G2 = G1 * SmithGAniso(abs(localL.z), localL.x, localL.y, ax, ay);

        tmpPdf = G1 * D / (4.0 * localV.z);
        vec3 specColor = FMetal * D * G2 / (4.0 * localL.z * localV.z);

        // Kulla-Conty multi-scatter compensation for conductors (colored F_ms)
        if (msGGX) {
            vec3 F_avg = (20.0 / 21.0) * mat.f0 + vec3(1.0 / 21.0);
            vec3 F_ms = (F_avg * msE_avg) / max(vec3(1.0) - F_avg * (1.0 - msE_avg), vec3(1e-5));
            specColor += F_ms * ms_fms;
        }

        f += specColor * metalWeight;
        pdf += tmpPdf * metalPr;
    }

    // Glass / Specular BSDF (anisotropic GGX)
    if (glassPr > 0.0) {
        float F = DielectricFresnel(VDotH, eta);
        float D = GTR2Aniso(localH.z, localH.x, localH.y, ax, ay);
        float G1 = SmithGAniso(abs(localV.z), localV.x, localV.y, ax, ay);
        float G2 = G1 * SmithGAniso(abs(localL.z), localL.x, localL.y, ax, ay);

        if (reflect) {
            tmpPdf = G1 * D / (4.0 * localV.z);
            vec3 glassSpec = vec3(F) * D * G2 / (4.0 * localL.z * localV.z);

            // Multi-scatter compensation on glass reflection side (uses F0, not F(theta))
            if (msGGX) {
                float f0_glass = pow((1.0 - mat.ior) / (1.0 + mat.ior), 2.0);
                float F_avg_g = (20.0 / 21.0) * f0_glass + (1.0 / 21.0);
                float F_ms_g = (F_avg_g * msE_avg) / max(1.0 - F_avg_g * (1.0 - msE_avg), 1e-5);
                glassSpec += vec3(F_ms_g * ms_fms);
            }

            f += glassSpec * glassWeight;
            pdf += tmpPdf * glassPr * F;
        } else {
            float denom = dot(localL, localH) + dot(localV, localH) * eta;
            denom *= denom;
            float jacobian = abs(dot(localL, localH)) / denom;

            tmpPdf = G1 * max(0.0, VDotH) * D * jacobian / localV.z;

            vec3 transColor = pow(mat.albedo, vec3(0.5)) * (1.0 - F) * D * G2 * abs(dot(localV, localH)) * jacobian *
                              (eta * eta) / abs(localL.z * localV.z);

            f += transColor * glassWeight;
            pdf += tmpPdf * glassPr * (1.0 - F);
        }
    }

    // Coat lobe (GGX clearcoat, fixed IOR 1.5 -> F0 = 0.04, isotropic)
    if (coatPr > 0.0 && reflect) {
        float ca = max(mat.coatRoughness * mat.coatRoughness, 1e-4);
        float D = GTR2Aniso(localH.z, localH.x, localH.y, ca, ca);
        float G1 = SmithGAniso(abs(localV.z), localV.x, localV.y, ca, ca);
        float G2 = G1 * SmithGAniso(abs(localL.z), localL.x, localL.y, ca, ca);
        float FH = SchlickWeight(VDotH);
        float F = mix(0.04, 1.0, FH);

        vec3 coatSpec = vec3(mat.coatWeight * F * D * G2 / (4.0 * abs(localL.z) * abs(localV.z)));

        // Multi-scatter compensation for coat (achromatic, F0=0.04)
        if (msGGX) {
            float coatE_o, coatE_avg;
            sampleGGXEnergy(abs(localV.z), ca, coatE_o, coatE_avg);
            float coatE_i, dummy;
            sampleGGXEnergy(abs(localL.z), ca, coatE_i, dummy);
            float coatF_avg = (20.0 / 21.0) * 0.04 + (1.0 / 21.0);
            float coatF_ms = (coatF_avg * coatE_avg) / max(1.0 - coatF_avg * (1.0 - coatE_avg), 1e-5);
            float coat_fms = (1.0 - coatE_o) * (1.0 - coatE_i) / max(PI * (1.0 - coatE_avg), 1e-5);
            coatSpec += vec3(mat.coatWeight * coatF_ms * coat_fms);
        }

        f += coatSpec;
        tmpPdf = G1 * D / (4.0 * localV.z);
        pdf += tmpPdf * coatPr;
    }

    return f * abs(localL.z); // Cosine term applied
}

// Backward-compatible overload (no flags = original Disney behavior)
vec3 DisneyEval(LabPBRMat mat, vec3 V, vec3 N, vec3 L, out float pdf) {
    return DisneyEval(mat, V, N, L, pdf, 0);
}

vec3 DisneySample(LabPBRMat mat, vec3 V, vec3 N, out vec3 L, out float pdf, inout uint seed, out uint lobeType, int brdfFlags, vec3 xi) {
    pdf = 0.0;
    vec3 T, B;
    Onb(N, T, B);

    vec3 localV = ToLocal(T, B, N, V);
    // When xi >= 0, use pre-generated samples (blue noise on primary hit); otherwise PCG
    float r1 = (xi.x >= 0.0) ? xi.x : rand(seed);
    float r2 = (xi.y >= 0.0) ? xi.y : rand(seed);
    float r3 = (xi.z >= 0.0) ? xi.z : rand(seed);

    // Anisotropic roughness
    float a = max(mat.roughness, 1e-4);
    float aspect = sqrt(1.0 - 0.9 * mat.anisotropic);
    float ax = max(a / aspect, 1e-4);
    float ay = max(a * aspect, 1e-4);

    float dielectricWeight = (1.0 - mat.metallic) * (1.0 - mat.transmission);
    float metalWeight = mat.metallic;
    float glassWeight = (1.0 - mat.metallic) * mat.transmission;
    float schlickWeight = SchlickWeight(abs(localV.z));

    float diffPr = dielectricWeight * Luminance(mat.albedo);
    float dielectricPr = dielectricWeight * Luminance(mix(mat.f0, vec3(1.0), schlickWeight));
    float metalPr = metalWeight * Luminance(mix(mat.albedo, vec3(1.0), schlickWeight));
    float glassPr = glassWeight;
    float coatPr = mat.coatWeight * 0.25;

    float invTotalWeight = 1.0 / (diffPr + dielectricPr + metalPr + glassPr + coatPr + 1e-5);
    diffPr *= invTotalWeight;
    dielectricPr *= invTotalWeight;
    metalPr *= invTotalWeight;
    glassPr *= invTotalWeight;
    coatPr *= invTotalWeight;

    float cdf0 = diffPr;
    float cdf1 = cdf0 + dielectricPr;
    float cdf2 = cdf1 + metalPr;
    float cdf3 = cdf2 + glassPr;

    vec3 localL;

    if (r3 < cdf0) { // Diffuse (+ sheen evaluated in DisneyEval)
        lobeType = 0;
        localL = CosineSampleHemisphere(r1, r2);
    } else if (r3 < cdf2) {                      // Dielectric + Metallic Reflection
        lobeType = 1;
        vec3 localH = SampleGGXVNDF(localV, ax, ay, r1, r2);
        if (localH.z < 0.0) localH = -localH;
        localL = normalize(reflect(-localV, localH));
    } else if (r3 < cdf3) {                     // Glass
        lobeType = 2;
        vec3 localH = SampleGGXVNDF(localV, ax, ay, r1, r2);
        if (localH.z < 0.0) localH = -localH;

        float eta = (localV.z > 0.0) ? (1.0 / mat.ior) : mat.ior;
        float F = DielectricFresnel(abs(dot(localV, localH)), eta);

        // Rescale random number for reuse
        float r_glass = (r3 - cdf2) / (cdf3 - cdf2 + 1e-5);

        if (r_glass < F) {
            localL = normalize(reflect(-localV, localH));
        } else {
            localL = normalize(refract(-localV, localH, eta));
        }
    } else {                                     // Coat
        lobeType = 3;
        float ca = max(mat.coatRoughness * mat.coatRoughness, 1e-4);
        vec3 localH = SampleGGXVNDF(localV, ca, ca, r1, r2);
        if (localH.z < 0.0) localH = -localH;
        localL = normalize(reflect(-localV, localH));
    }

    L = ToWorld(T, B, N, localL);
    V = ToWorld(T, B, N, localV);

    return DisneyEval(mat, V, N, L, pdf, brdfFlags);
}

// Overload: flags but no blue noise (PCG fallback via negative xi)
vec3 DisneySample(LabPBRMat mat, vec3 V, vec3 N, out vec3 L, out float pdf, inout uint seed, out uint lobeType, int brdfFlags) {
    return DisneySample(mat, V, N, L, pdf, seed, lobeType, brdfFlags, vec3(-1.0));
}

// Overload: no flags, no blue noise
vec3 DisneySample(LabPBRMat mat, vec3 V, vec3 N, out vec3 L, out float pdf, inout uint seed, out uint lobeType) {
    return DisneySample(mat, V, N, L, pdf, seed, lobeType, 0, vec3(-1.0));
}

#endif
