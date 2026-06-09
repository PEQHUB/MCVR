#ifndef VPT_FFT_WATER_GLSL
#define VPT_FFT_WATER_GLSL

struct FftWaterSample {
    float height;
    vec2 slope;
    vec2 displacement;
    vec2 curvature;
    vec3 normal;
};

layout(set = 5, binding = 27) uniform sampler2D fftWaterFlowTexture;
layout(set = 5, binding = 28) uniform sampler2D fftWaterNormalATexture;
layout(set = 5, binding = 29) uniform sampler2D fftWaterNormalBTexture;

const float FFT_WATER_MACRO_SPATIAL_SCALE = 0.65;
const float FFT_WATER_DETAIL_SPATIAL_SCALE = 0.55;
const float FFT_WATER_SLOPE_STRENGTH = 0.72;
const float FFT_WATER_DISPLACEMENT_STRENGTH = 0.55;
const float FFT_WATER_CAUSTIC_CONTRAST = 0.38;
const float FFT_WATER_CAUSTIC_MAX = 2.2;

mat2 fftWaterRotation(float angle) {
    float s = sin(angle);
    float c = cos(angle);
    return mat2(c, -s, s, c);
}

void fftWaterStableBasis(vec3 geometricNormal, out vec3 tangent, out vec3 bitangent) {
    vec3 upNormal = geometricNormal.y >= 0.0 ? geometricNormal : -geometricNormal;
    tangent = vec3(1.0, 0.0, 0.0) - upNormal * upNormal.x;
    if (dot(tangent, tangent) < 1e-10) {
        tangent = vec3(0.0, 0.0, 1.0) - upNormal * upNormal.z;
    }
    tangent = normalize(tangent);
    bitangent = normalize(cross(tangent, upNormal));
}

vec2 fftWaterSurfaceCoord(vec3 absWorldPos, vec3 dPduWorld, vec3 dPdvWorld, vec3 geometricNormal) {
    vec3 tangent, bitangent;
    fftWaterStableBasis(geometricNormal, tangent, bitangent);
    return vec2(dot(absWorldPos, tangent), dot(absWorldPos, bitangent));
}

void fftWaterAddWave(vec2 p,
                     float t,
                     vec2 dir,
                     float freq,
                     float amp,
                     float speed,
                     inout float height,
                     inout vec2 slope,
                     inout vec2 displacement,
                     inout vec2 curvature) {
    float phase = dot(p, dir) * freq + t * speed;
    float s = sin(phase);
    float c = cos(phase);
    height += amp * s;
    slope += dir * (amp * freq * c);
    displacement += dir * (amp * 0.42 * s);
    curvature += (dir * dir) * (-amp * freq * freq * s);
}

vec2 fftWaterTextureNormalSlope(sampler2D normalTexture, vec2 uv, float strength) {
    vec3 normalSample = textureLod(normalTexture, uv, 0.0).xyz * 2.0 - 1.0;
    if (dot(normalSample, normalSample) <= 1e-8 || any(isnan(normalSample)) || any(isinf(normalSample))) {
        return vec2(0.0);
    }
    normalSample = normalize(normalSample);
    return normalSample.xy * strength;
}

FftWaterSample sampleFftWater(vec2 worldXZ, float gameTime) {
    float t = gameTime * 24000.0 * 0.05;
    vec2 macroXZ = worldXZ * FFT_WATER_MACRO_SPATIAL_SCALE;
    vec2 detailXZ = worldXZ * FFT_WATER_DETAIL_SPATIAL_SCALE;
    vec2 textureFlowUv = macroXZ * 0.004 + vec2(t * 0.004, -t * 0.003);
    vec2 textureFlow = textureLod(fftWaterFlowTexture, textureFlowUv, 0.0).rg * 2.0 - 1.0;
    if (any(isnan(textureFlow)) || any(isinf(textureFlow))) { textureFlow = vec2(0.0); }
    vec2 lowFlow = vec2(sin(dot(macroXZ, vec2(0.023, -0.019)) + t * 0.61),
                        cos(dot(macroXZ, vec2(-0.017, 0.021)) - t * 0.54));
    vec2 highFlow = vec2(cos(dot(detailXZ, vec2(0.041, 0.037)) - t * 0.88 + 1.9),
                         sin(dot(detailXZ, vec2(-0.036, 0.044)) + t * 0.79 - 0.7));
    vec2 flow = lowFlow * 0.55 + highFlow * 0.25 + textureFlow * 0.35;
    vec2 p0 = macroXZ * 1.14 + flow * 0.75;
    vec2 p1 = fftWaterRotation(0.73) * (detailXZ * 1.40) + fftWaterRotation(1.11) * flow * 0.58;

    float height = 0.0;
    vec2 slope = vec2(0.0);
    vec2 displacement = flow * 0.10;
    vec2 curvature = vec2(0.0);

    fftWaterAddWave(p0, t, normalize(vec2(1.00, 0.18)), 0.27, 0.120, 0.72, height, slope, displacement, curvature);
    fftWaterAddWave(p0, t, normalize(vec2(-0.61, 0.79)), 0.41, 0.090, -1.03, height, slope, displacement, curvature);
    fftWaterAddWave(p0, t, normalize(vec2(0.19, 0.98)), 0.61, 0.060, 1.41, height, slope, displacement, curvature);
    fftWaterAddWave(p0, t, normalize(vec2(-0.93, -0.36)), 0.92, 0.035, -1.92, height, slope, displacement, curvature);
    fftWaterAddWave(p1, t, normalize(vec2(0.71, -0.70)), 1.31, 0.020, 2.67, height, slope, displacement, curvature);
    fftWaterAddWave(p1, t, normalize(vec2(-0.24, -0.97)), 1.78, 0.014, -3.21, height, slope, displacement, curvature);
    fftWaterAddWave(p1, t, normalize(vec2(0.89, 0.46)), 2.34, 0.010, 4.10, height, slope, displacement, curvature);
    fftWaterAddWave(p1, t, normalize(vec2(-0.78, 0.62)), 2.92, 0.007, -4.63, height, slope, displacement, curvature);

    vec2 normalUvA = detailXZ * 0.020 + flow * 0.035 + vec2(t * 0.012, -t * 0.008);
    vec2 normalUvB = fftWaterRotation(0.82) * (detailXZ * 0.045 + flow * 0.025) + vec2(-t * 0.019, t * 0.014);
    vec2 textureSlope = fftWaterTextureNormalSlope(fftWaterNormalATexture, normalUvA, 0.23) +
                        fftWaterTextureNormalSlope(fftWaterNormalBTexture, normalUvB, 0.12);
    slope += textureSlope;
    displacement += textureSlope * 0.025;

    slope *= FFT_WATER_SLOPE_STRENGTH;
    displacement *= FFT_WATER_DISPLACEMENT_STRENGTH;
    curvature *= 18.0;

    FftWaterSample waterSample;
    waterSample.height = height;
    waterSample.slope = slope;
    waterSample.displacement = displacement;
    waterSample.curvature = curvature;
    waterSample.normal = normalize(vec3(-slope.x, 1.0, -slope.y));
    return waterSample;
}

float sampleFftWaterCaustic(vec2 worldXZ, float gameTime, vec3 toLightDir, float waterDepth) {
    if (toLightDir.y <= 0.01) { return 1.0; }

    vec3 airDir = normalize(-toLightDir);
    float offset = 0.10;
    float focusDepth = waterDepth;

    FftWaterSample center = sampleFftWater(worldXZ, gameTime);
    FftWaterSample sampleX = sampleFftWater(worldXZ + vec2(offset, 0.0), gameTime);
    FftWaterSample sampleZ = sampleFftWater(worldXZ + vec2(0.0, offset), gameTime);

    vec3 refractedCenter = refract(airDir, center.normal, 1.0 / 1.333);
    vec3 refractedX = refract(airDir, sampleX.normal, 1.0 / 1.333);
    vec3 refractedZ = refract(airDir, sampleZ.normal, 1.0 / 1.333);

    if (dot(refractedCenter, refractedCenter) <= 1e-6 || dot(refractedX, refractedX) <= 1e-6 ||
        dot(refractedZ, refractedZ) <= 1e-6) {
        return 1.0;
    }
    if (refractedCenter.y >= -0.1 || refractedX.y >= -1e-3 || refractedZ.y >= -1e-3) { return 1.0; }

    vec2 projectedCenter = center.displacement + refractedCenter.xz * (focusDepth / -refractedCenter.y);
    vec2 projectedX = vec2(offset, 0.0) + sampleX.displacement + refractedX.xz * (focusDepth / -refractedX.y);
    vec2 projectedZ = vec2(0.0, offset) + sampleZ.displacement + refractedZ.xz * (focusDepth / -refractedZ.y);

    vec2 dx = projectedX - projectedCenter;
    vec2 dz = projectedZ - projectedCenter;
    float area = abs(dx.x * dz.y - dx.y * dz.x);
    float jacobian = area / (offset * offset);
    float softening = 0.18 + waterDepth * 0.08;
    float strength = 1.0 / (jacobian + softening);
    strength *= 1.0 + softening;
    float contrast = strength - 1.0;
    float attenuation = 1.0 / (1.0 + waterDepth * waterDepth * 0.045);
    float finalEffect = 1.0 + contrast * attenuation * FFT_WATER_CAUSTIC_CONTRAST;
    float sunWeight = clamp(toLightDir.y * 1.5, 0.0, 1.0);
    return clamp(mix(1.0, finalEffect, sunWeight), 0.25, FFT_WATER_CAUSTIC_MAX);
}

#endif
