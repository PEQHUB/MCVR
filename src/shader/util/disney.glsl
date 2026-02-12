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
 */

/*
 * Modifications:
 * - Copyright (c) 2026 Radiance
 * - Changes: Applied to LabPBR materials
 *
 * Note: Original license notice and permission notice are retained per MIT License.
 */

#include "labpbr.glsl"
#include "random.glsl"
#include "common/shared.hpp"

#ifndef DISNEY_GLSL
#    define DISNEY_GLSL

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

vec3 SampleVMF(inout uint seed, vec3 mu, float kappa) {
    kappa = max(kappa, 0.1);

    float u1 = rand(seed);
    float expNeg2K = exp(-2.0 * kappa);
    float w = 1.0 + (1.0 / kappa) * log(u1 + (1.0 - u1) * expNeg2K);

    w = clamp(w, -1.0, 1.0);

    float u2 = rand(seed);
    float phi = 2.0 * PI * u2;
    float vx = cos(phi);
    float vy = sin(phi);

    vec3 u_basis;
    if (abs(mu.x) > 0.99) {
        u_basis = normalize(vec3(mu.y, -mu.x, 0.0));
    } else {
        u_basis = normalize(vec3(-mu.z, 0.0, mu.x));
    }
    vec3 v_basis = normalize(cross(mu, u_basis));

    float sinTheta = sqrt(max(0.0, 1.0 - w * w));
    return normalize(w * mu + sinTheta * (vx * u_basis + vy * v_basis));
}

float Luminance(vec3 c) {
    return 0.212671 * c.x + 0.715160 * c.y + 0.072169 * c.z;
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

vec3 DisneyEval(LabPBRMat mat, vec3 V, vec3 N, vec3 L, out float pdf) {
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

    float invTotalWeight = 1.0 / (diffPr + dielectricPr + metalPr + glassPr + 1e-5);
    diffPr *= invTotalWeight;
    dielectricPr *= invTotalWeight;
    metalPr *= invTotalWeight;
    glassPr *= invTotalWeight;

    bool reflect = localL.z * localV.z > 0.0;
    float tmpPdf = 0.0;
    float VDotH = abs(dot(localV, localH));

    // Diffuse
    if (diffPr > 0.0 && reflect) {
        float LDotH = dot(localL, localH);
        float Rr = 2.0 * mat.roughness * LDotH * LDotH;
        float FL = SchlickWeight(localL.z);
        float FV = SchlickWeight(localV.z);
        float Fretro = Rr * (FL + FV + FL * FV * (Rr - 1.0));
        float Fd = (1.0 - 0.5 * FL) * (1.0 - 0.5 * FV);

        // Fake subsurface
        float Fss90 = 0.5 * Rr;
        float Fss = mix(1.0, Fss90, FL) * mix(1.0, Fss90, FV);
        float ss = 1.25 * (Fss * (1.0 / (localL.z + localV.z) - 0.5) + 0.5);

        vec3 diffuseColor = INV_PI * mat.albedo * mix(Fd + Fretro, ss, mat.subSurface);

        f += diffuseColor * dielectricWeight;
        pdf += (localL.z * INV_PI) * diffPr; // Cosine weighted PDF
    }

    // Dielectric Reflection
    if (dielectricPr > 0.0 && reflect) {
        float F = DielectricFresnel(VDotH, 1.0 / mat.ior);

        float a = max(mat.roughness, 1e-4);
        float D = GTR2Aniso(localH.z, localH.x, localH.y, a, a);
        float G1 = SmithGAniso(abs(localV.z), localV.x, localV.y, a, a);
        float G2 = G1 * SmithGAniso(abs(localL.z), localL.x, localL.y, a, a);

        tmpPdf = G1 * D / (4.0 * localV.z);
        vec3 specColor = vec3(F) * D * G2 / (4.0 * localL.z * localV.z);

        f += specColor * dielectricWeight;
        pdf += tmpPdf * dielectricPr;
    }

    // Metallic Reflection
    if (metalPr > 0.0 && reflect) {
        vec3 FMetal = mix(mat.albedo, vec3(1.0), SchlickWeight(VDotH));

        float a = max(mat.roughness, 1e-4);
        float D = GTR2Aniso(localH.z, localH.x, localH.y, a, a);
        float G1 = SmithGAniso(abs(localV.z), localV.x, localV.y, a, a);
        float G2 = G1 * SmithGAniso(abs(localL.z), localL.x, localL.y, a, a);

        tmpPdf = G1 * D / (4.0 * localV.z);
        vec3 specColor = FMetal * D * G2 / (4.0 * localL.z * localV.z);

        f += specColor * metalWeight;
        pdf += tmpPdf * metalPr;
    }

    // Glass / Specular BSDF
    if (glassPr > 0.0) {
        float F = DielectricFresnel(VDotH, eta);
        float a = max(mat.roughness, 1e-4);
        float D = GTR2Aniso(localH.z, localH.x, localH.y, a, a);
        float G1 = SmithGAniso(abs(localV.z), localV.x, localV.y, a, a);
        float G2 = G1 * SmithGAniso(abs(localL.z), localL.x, localL.y, a, a);

        if (reflect) {
            tmpPdf = G1 * D / (4.0 * localV.z);
            f += vec3(F) * D * G2 / (4.0 * localL.z * localV.z) * glassWeight;
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

    return f * abs(localL.z); // Cosine term applied
}

vec3 DisneySample(LabPBRMat mat, vec3 V, vec3 N, out vec3 L, out float pdf, inout uint seed, out uint lobeType) {
    pdf = 0.0;
    vec3 T, B;
    Onb(N, T, B);

    vec3 localV = ToLocal(T, B, N, V);
    float r1 = rand(seed);
    float r2 = rand(seed);
    float r3 = rand(seed);

    float dielectricWeight = (1.0 - mat.metallic) * (1.0 - mat.transmission);
    float metalWeight = mat.metallic;
    float glassWeight = (1.0 - mat.metallic) * mat.transmission;
    float schlickWeight = SchlickWeight(abs(localV.z));

    float diffPr = dielectricWeight * Luminance(mat.albedo);
    float dielectricPr = dielectricWeight * Luminance(mix(mat.f0, vec3(1.0), schlickWeight));
    float metalPr = metalWeight * Luminance(mix(mat.albedo, vec3(1.0), schlickWeight));
    float glassPr = glassWeight;

    float invTotalWeight = 1.0 / (diffPr + dielectricPr + metalPr + glassPr + 1e-5);
    diffPr *= invTotalWeight;
    dielectricPr *= invTotalWeight;
    metalPr *= invTotalWeight;
    glassPr *= invTotalWeight;

    float cdf0 = diffPr;
    float cdf1 = cdf0 + dielectricPr;
    float cdf2 = cdf1 + metalPr;

    vec3 localL;

    if (r3 < cdf0) { // Diffuse
        lobeType = 0;
        localL = CosineSampleHemisphere(r1, r2);
    } else if (r3 < cdf2) {                      // Dielectric + Metallic Reflection
        lobeType = 1;
        float a = max(mat.roughness, 1e-4);
        vec3 localH = SampleGGXVNDF(localV, a, a, r1, r2);
        if (localH.z < 0.0) localH = -localH;
        localL = normalize(reflect(-localV, localH));
    } else {                                     // Glass
        lobeType = 2;
        float a = max(mat.roughness, 1e-4);
        vec3 localH = SampleGGXVNDF(localV, a, a, r1, r2);
        if (localH.z < 0.0) localH = -localH;

        float eta = (localV.z > 0.0) ? (1.0 / mat.ior) : mat.ior;
        float F = DielectricFresnel(abs(dot(localV, localH)), eta);

        // Rescale random number for reuse
        float r_glass = (r3 - cdf2) / (1.0 - cdf2 + 1e-5);

        if (r_glass < F) {
            localL = normalize(reflect(-localV, localH));
        } else {
            localL = normalize(refract(-localV, localH, eta));
        }
    }

    L = ToWorld(T, B, N, localL);
    V = ToWorld(T, B, N, localV);

    return DisneyEval(mat, V, N, L, pdf);
}

#endif
