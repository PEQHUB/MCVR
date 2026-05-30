layout(set = 5, binding = 11) uniform sampler2D fftWaterNormalAImage;
layout(set = 5, binding = 12) uniform sampler2D fftWaterNormalBImage;
layout(set = 5, binding = 13) uniform sampler2D fftWaterFlowImage;

struct FftWaterSample {
    vec3 localNormal;
};

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

vec3 fftWaterDecodeTextureNormal(vec3 encodedNormal) {
    vec3 normal = encodedNormal * 2.0 - 1.0;
    // Texture normals are encoded as tangent/up/bitangent; applyNormalMapToBasis flips local Y.
    return normalize(vec3(normal.x, -normal.z, max(normal.y, 1e-4)));
}

float fftWaterEtaForIncident(vec3 incident, vec3 waterNormal, float ior) {
    vec3 waterUpNormal = waterNormal.y >= 0.0 ? waterNormal : -waterNormal;
    return dot(incident, waterUpNormal) < 0.0 ? (1.0 / ior) : ior;
}

vec2 fftWaterFlowSample(vec2 worldXZ, float t) {
    vec2 flowUv = worldXZ * (1.0 / 48.0) + vec2(t * 0.0031, -t * 0.0027);
    return texture(fftWaterFlowImage, fract(flowUv)).rg * 2.0 - 1.0;
}

FftWaterSample sampleFftWater(vec2 worldXZ, float gameTime) {
    const float waterTimeScale = 24000.0 * 0.35;
    const float normalStrengthA = 2.2;
    const float normalStrengthB = 0.75;
    const float normalUpBias = 1.15;

    float t = gameTime * waterTimeScale;
    vec2 uv = worldXZ * (1.0 / 24.0);

    vec2 flowA = fftWaterFlowSample(worldXZ, t);
    vec2 flowB = fftWaterFlowSample(worldXZ * 0.73 + vec2(19.7, -11.3), t * 0.89);
    vec2 distortion = 0.036 * flowA + 0.018 * flowB;

    vec2 normalUvA = fract(uv + distortion + vec2(t * 0.0034, t * 0.0028));
    vec2 normalUvB = fract(uv * 1.35 - distortion * 0.45 + vec2(-t * 0.0038, t * 0.0033));

    vec3 normalA = fftWaterDecodeTextureNormal(texture(fftWaterNormalAImage, normalUvA).rgb);
    vec3 normalB = fftWaterDecodeTextureNormal(texture(fftWaterNormalBImage, normalUvB).rgb);

    FftWaterSample waterSample;
    waterSample.localNormal = normalize(vec3(normalA.xy * normalStrengthA + normalB.xy * normalStrengthB,
                                             normalA.z + normalB.z * 0.9 + normalUpBias));
    return waterSample;
}

float sampleFftWaterCaustic(vec2 worldXZ, float gameTime, vec3 toLightDir, float waterDepth) {
    if (waterDepth <= 1e-4 || toLightDir.y <= 0.01) { return 1.0; }

    vec3 airDir = normalize(-toLightDir);
    const float offset = 0.08;
    float focusDepth = waterDepth;

    FftWaterSample center = sampleFftWater(worldXZ, gameTime);
    FftWaterSample sampleX = sampleFftWater(worldXZ + vec2(offset, 0.0), gameTime);
    FftWaterSample sampleZ = sampleFftWater(worldXZ + vec2(0.0, offset), gameTime);

    vec3 refractedCenter = refract(airDir, center.localNormal, 1.0 / 1.333);
    vec3 refractedX = refract(airDir, sampleX.localNormal, 1.0 / 1.333);
    vec3 refractedZ = refract(airDir, sampleZ.localNormal, 1.0 / 1.333);

    if (dot(refractedCenter, refractedCenter) <= 1e-6 || dot(refractedX, refractedX) <= 1e-6 ||
        dot(refractedZ, refractedZ) <= 1e-6) {
        return 1.0;
    }
    if (refractedCenter.y >= -0.1 || refractedX.y >= -1e-3 || refractedZ.y >= -1e-3) { return 1.0; }

    vec2 projectedCenter = refractedCenter.xz * (focusDepth / -refractedCenter.y);
    vec2 projectedX = vec2(offset, 0.0) + refractedX.xz * (focusDepth / -refractedX.y);
    vec2 projectedZ = vec2(0.0, offset) + refractedZ.xz * (focusDepth / -refractedZ.y);

    vec2 dx = projectedX - projectedCenter;
    vec2 dz = projectedZ - projectedCenter;
    float area = abs(dx.x * dz.y - dx.y * dz.x);
    float jacobian = area / (offset * offset);
    float softening = 0.08 + waterDepth * 0.08;
    float strength = 1.0 / (jacobian + softening);
    strength *= 1.0 + softening;
    float contrast = strength - 1.0;
    contrast /= 1.0 + abs(contrast) * 1.4;
    float attenuation = 1.0 / (1.0 + waterDepth * waterDepth * 0.06);
    float finalEffect = 1.0 + contrast * attenuation * 0.45;
    float sunWeight = clamp(toLightDir.y * 1.2, 0.0, 1.0);
    return clamp(mix(1.0, finalEffect, sunWeight), 0.6, 2.1);
}
