#ifndef ADV_VOLUMETRIC_CLOUD_GLSL
#define ADV_VOLUMETRIC_CLOUD_GLSL

#include "common/celestial.glsl"

#ifndef ADV_CLOUD_MODE
#    define ADV_CLOUD_MODE 1u
#endif
#ifndef ADV_VOLUMETRIC_CLOUD_BOTTOM_HEIGHT
#    define ADV_VOLUMETRIC_CLOUD_BOTTOM_HEIGHT 300.0
#endif
#ifndef ADV_VOLUMETRIC_CLOUD_TOP_HEIGHT
#    define ADV_VOLUMETRIC_CLOUD_TOP_HEIGHT 600.0
#endif
#ifndef ADV_VOLUMETRIC_CLOUD_BASE_SCALE
#    define ADV_VOLUMETRIC_CLOUD_BASE_SCALE 0.30
#endif
#ifndef ADV_VOLUMETRIC_CLOUD_DETAIL_SCALE
#    define ADV_VOLUMETRIC_CLOUD_DETAIL_SCALE 0.60
#endif
#ifndef ADV_VOLUMETRIC_CLOUD_COVERAGE
#    define ADV_VOLUMETRIC_CLOUD_COVERAGE 0.50
#endif
#ifndef ADV_VOLUMETRIC_CLOUD_CLEAR_AMOUNT
#    define ADV_VOLUMETRIC_CLOUD_CLEAR_AMOUNT 1u
#endif
#ifndef ADV_VOLUMETRIC_CLOUD_DENSITY
#    define ADV_VOLUMETRIC_CLOUD_DENSITY 1.00
#endif
#ifndef ADV_VOLUMETRIC_CLOUD_VIEW_STEPS
#    define ADV_VOLUMETRIC_CLOUD_VIEW_STEPS 128
#endif
#ifndef ADV_VOLUMETRIC_CLOUD_LIGHT_STEPS
#    define ADV_VOLUMETRIC_CLOUD_LIGHT_STEPS 12
#endif
#ifndef ADV_VOLUMETRIC_CLOUD_CAST_SHADOW
#    define ADV_VOLUMETRIC_CLOUD_CAST_SHADOW 1
#endif
#ifndef ADV_VOLUMETRIC_CLOUD_AMBIENT_STEPS
#    define ADV_VOLUMETRIC_CLOUD_AMBIENT_STEPS 8
#endif
#ifndef ADV_VOLUMETRIC_CLOUD_AMBIENT_STRENGTH
#    define ADV_VOLUMETRIC_CLOUD_AMBIENT_STRENGTH 1.0
#endif
#ifndef ADV_VOLUMETRIC_CLOUD_POWDER_STRENGTH
#    define ADV_VOLUMETRIC_CLOUD_POWDER_STRENGTH 1.0
#endif
#ifndef ADV_VOLUMETRIC_CLOUD_WEATHER_SCALE
#    define ADV_VOLUMETRIC_CLOUD_WEATHER_SCALE 0.020
#endif
#ifndef ADV_VOLUMETRIC_CLOUD_SHADOW_STRENGTH
#    define ADV_VOLUMETRIC_CLOUD_SHADOW_STRENGTH 0.80
#endif
#ifndef ADV_VOLUMETRIC_CLOUD_SHADOW_SOFTNESS
#    define ADV_VOLUMETRIC_CLOUD_SHADOW_SOFTNESS 0.50
#endif
#ifndef ADV_INDIRECT_VOLUMETRIC_CLOUD_VIEW_STEPS
#    define ADV_INDIRECT_VOLUMETRIC_CLOUD_VIEW_STEPS 20
#endif
#ifndef ADV_INDIRECT_VOLUMETRIC_CLOUD_LIGHT_STEPS
#    define ADV_INDIRECT_VOLUMETRIC_CLOUD_LIGHT_STEPS 5
#endif
#ifndef ADV_INDIRECT_VOLUMETRIC_CLOUD_AMBIENT_STEPS
#    define ADV_INDIRECT_VOLUMETRIC_CLOUD_AMBIENT_STEPS 3
#endif

#define ADV_VOLUMETRIC_CLOUD_CLEAR_AMOUNT_FEW 0u
#define ADV_VOLUMETRIC_CLOUD_CLEAR_AMOUNT_MEDIUM 1u
#define ADV_VOLUMETRIC_CLOUD_CLEAR_AMOUNT_MANY 2u

layout(set = 5, binding = 22) uniform sampler3D volumetricCloudBasicNoiseTexture;
layout(set = 5, binding = 23) uniform sampler3D volumetricCloudDetailNoiseTexture;
layout(set = 5, binding = 24) uniform sampler2D volumetricCloudWeatherTexture;
layout(set = 5, binding = 25) uniform sampler2D volumetricCloudCurlTexture;
layout(set = 5, binding = 26) uniform sampler2D volumetricCloudCoverageNoiseTexture;
struct VolumetricCloudResult {
    vec3 color;
    float transmittance;
    float hit;
};

struct VolumetricCloudAtmosphereSegmentResult {
    vec3 scatteredLight;
    vec3 transmittance;
};

struct VolumetricCloudSampleContext {
    vec3 posKm;
    vec3 windOffset;
    float coverage;
    float gradientShape;
    float normalizedHeight;
    float layerHeight01;
    int layerIndex;
};

float cloudSaturate(float value) {
    return clamp(value, 0.0, 1.0);
}

float cloudRemapClamped(float value, float inMin, float inMax, float outMin, float outMax) {
    float t = clamp((value - inMin) / max(inMax - inMin, 1e-5), 0.0, 1.0);
    return mix(outMin, outMax, t);
}

vec3 cloudSafeNormalize(vec3 value, vec3 fallback) {
    float len2 = dot(value, value);
    if (len2 <= 1e-8 || any(isnan(value)) || any(isinf(value))) { return fallback; }
    return value * inversesqrt(len2);
}

float cloudHash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

bool cloudIntersectSphere(vec3 ro, vec3 rd, float radius, out float tNear, out float tFar) {
    float b = dot(ro, rd);
    float c = dot(ro, ro) - radius * radius;
    float h = b * b - c;
    if (h < 0.0) { return false; }
    h = sqrt(h);
    tNear = -b - h;
    tFar = -b + h;
    return true;
}

vec3 cloudAbsoluteWorldPos(vec3 relativeWorldPos) {
    return relativeWorldPos + vec3(worldUBO.cameraPos.xyz);
}

vec3 cloudPlanetPos(vec3 relativeWorldPos) {
    vec3 absoluteWorldPos = cloudAbsoluteWorldPos(relativeWorldPos);
    return vec3(absoluteWorldPos.x, ADV_ATMOSPHERE_RG + absoluteWorldPos.y, absoluteWorldPos.z);
}

float cloudBottomRadius() {
    return ADV_ATMOSPHERE_RG + ADV_VOLUMETRIC_CLOUD_BOTTOM_HEIGHT;
}

float cloudTopRadius() {
    return ADV_ATMOSPHERE_RG + ADV_VOLUMETRIC_CLOUD_TOP_HEIGHT;
}

float cloudHeight01(vec3 planetPos) {
    float r = length(planetPos);
    return clamp((r - cloudBottomRadius()) / max(cloudTopRadius() - cloudBottomRadius(), 1e-3), 0.0, 1.0);
}

float cloudTime() {
    return worldUBO.gameTime * 24000.0 / 200;
}

vec3 cloudWindDirection() {
    return cloudSafeNormalize(vec3(0.8, 0.0, 0.4), vec3(1.0, 0.0, 0.0));
}

bool intersectCloudLayer(vec3 rayOrigin, vec3 rayDir, out float tEnter, out float tExit) {
    vec3 originPlanet = cloudPlanetPos(rayOrigin);
    float outerNear, outerFar;
    if (!cloudIntersectSphere(originPlanet, rayDir, cloudTopRadius(), outerNear, outerFar)) { return false; }

    tEnter = max(outerNear, 0.0);
    tExit = max(outerFar, 0.0);
    if (tExit <= tEnter) { return false; }

    float innerNear, innerFar;
    if (!cloudIntersectSphere(originPlanet, rayDir, cloudBottomRadius(), innerNear, innerFar)) {
        return tExit > tEnter;
    }

    if (innerFar > 0.0) {
        if (innerNear > 0.0) {
            if (tEnter < innerNear) {
                tExit = min(tExit, innerNear);
            } else {
                tEnter = max(tEnter, innerFar);
            }
        } else {
            tEnter = max(tEnter, innerFar);
        }
    }

    float groundNear, groundFar;
    if (cloudIntersectSphere(originPlanet, rayDir, ADV_ATMOSPHERE_RG, groundNear, groundFar)) {
        float groundHit = groundNear > 0.0 ? groundNear : groundFar;
        if (groundHit > 0.0) { tExit = min(tExit, groundHit); }
    }

    return tExit > tEnter;
}

float volumetricCloudSegmentLength(vec3 rayOrigin, vec3 rayDir) {
    float tEnter, tExit;
    if (!intersectCloudLayer(rayOrigin, rayDir, tEnter, tExit)) { return 0.0; }
    return max(tExit - tEnter, 0.0);
}

bool volumetricCloudTraceSegment(vec3 rayOrigin, vec3 rayDir, out vec3 segmentOrigin, out float segmentLength) {
    float tEnter, tExit;
    if (!intersectCloudLayer(rayOrigin, rayDir, tEnter, tExit)) {
        segmentOrigin = rayOrigin;
        segmentLength = 0.0;
        return false;
    }

    segmentOrigin = rayOrigin + rayDir * tEnter;
    segmentLength = max(tExit - tEnter, 0.0);
    return segmentLength > 1e-3;
}

vec2 transmittanceUv(float r, float mu) {
    float u = clamp(mu * 0.5 + 0.5, 0.0, 1.0);
    float v = clamp((r - ADV_ATMOSPHERE_RG) / max(ADV_ATMOSPHERE_RT - ADV_ATMOSPHERE_RG, 1.0), 0.0, 1.0);
    return vec2(u, v);
}

vec3 sampleCloudAtmosphereTransmittance(float r, float mu) {
    return sampleTexture(transLUT, transmittanceUv(r, mu), false).rgb;
}

float volumetricCloudRainBlend() {
    return skyUBO.rainGradient;
}

float volumetricCloudOvercastBlend() {
    return smoothstep(0.02, 0.20, volumetricCloudRainBlend());
}

float volumetricCloudClearAmountScale() {
    if (ADV_VOLUMETRIC_CLOUD_CLEAR_AMOUNT == ADV_VOLUMETRIC_CLOUD_CLEAR_AMOUNT_FEW) { return 0.82; }
    if (ADV_VOLUMETRIC_CLOUD_CLEAR_AMOUNT == ADV_VOLUMETRIC_CLOUD_CLEAR_AMOUNT_MANY) { return 1.22; }
    return 1.0;
}

float volumetricCloudClearCoverageDrift() {
    float t = cloudTime();
    float wave =
        0.50 * sin(t * 0.00013) +
        0.35 * sin(t * 0.000057 + 1.7) +
        0.15 * sin(t * 0.000031 - 0.8);
    return 1.0 + 0.08 * wave;
}

float volumetricCloudMergedCoverageNoise(vec2 animatedWorldXZ, float scale, vec2 offset) {
    float coarse = texture(volumetricCloudCoverageNoiseTexture, animatedWorldXZ * scale + offset).x;
    float merged = texture(volumetricCloudCoverageNoiseTexture, animatedWorldXZ * (scale * 0.57) + offset + vec2(0.173, -0.121)).x;
    return mix(coarse, merged, 0.55);
}

float volumetricCloudCoverageBase(float clearBase, float rainyBase) {
    return cloudSaturate(mix(clearBase, rainyBase, volumetricCloudOvercastBlend()));
}

float volumetricCloudWeatherSignal(float weatherValue, float clearScale, float clearBias, float rainyFloor) {
    float clearSignal = cloudSaturate(weatherValue * clearScale + clearBias);
    return cloudSaturate(mix(clearSignal, max(clearSignal, rainyFloor), volumetricCloudOvercastBlend()));
}

vec3 volumetricCloudSkyRadiance(vec3 sampleDir) {
    vec3 sunDir = celestialSunDirection();
    float progress = volumetricCloudRainBlend();
    vec3 rainyRadiance = mix(vec3(0.03, 0.035, 0.04), vec3(0.12, 0.13, 0.14), smoothstep(-0.3, 0.3, sunDir.y));
    return mix(texture(skyFull, sampleDir).rgb, rainyRadiance, progress);
}

float volumetricCloudSunDirectScale() {
    return smoothstep(0.03, 0.14, celestialSunDirection().y);
}

float volumetricCloudMoonDirectScale() {
    return smoothstep(0.03, 0.14, celestialMoonDirection().y);
}

vec3 volumetricCloudPrimaryLightDir() {
    vec3 sunDir = celestialSunDirection();
    vec3 moonDir = celestialMoonDirection();
    float sunScale = volumetricCloudSunDirectScale();
    float moonScale = volumetricCloudMoonDirectScale();
    float strongestDirectScale = max(sunScale, moonScale);
    if (strongestDirectScale <= 1e-4) { return sunDir.y >= 0.0 ? sunDir : moonDir; }
    if (sunScale >= moonScale) { return sunDir; }
    return moonDir;
}

vec3 volumetricCloudPrimaryLightRadiance() {
    float sunScale = volumetricCloudSunDirectScale();
    float moonScale = volumetricCloudMoonDirectScale();
    float rainAttenuation = mix(1.0, 0.35, volumetricCloudRainBlend());
    if (sunScale >= moonScale) {
        return ADV_SUN_RADIANCE * sunScale * rainAttenuation;
    }
    return ADV_MOON_RADIANCE * moonScale * rainAttenuation;
}

float atmosphereDensityExp(float height, float scaleHeight) {
    return exp(-max(height, 0.0) / max(scaleHeight, 1e-3));
}

float phaseRayleigh(float cosTheta) {
    return 3.0 / (16.0 * PI) * (1.0 + cosTheta * cosTheta);
}

float phaseMieHG(float cosTheta, float g) {
    cosTheta = clamp(cosTheta, -1.0, 1.0);
    g = clamp(g, -0.999, 0.999);
    float g2 = g * g;
    float d = 1.0 + g2 - 2.0 * g * cosTheta;
    return (1.0 - g2) / (4.0 * PI * pow(max(d, 1e-6), 1.5));
}

VolumetricCloudAtmosphereSegmentResult integrateAtmosphereSegment(vec3 rayOrigin, vec3 rayDir, float maxDistance) {
    VolumetricCloudAtmosphereSegmentResult result;
    result.scatteredLight = vec3(0.0);
    result.transmittance = vec3(1.0);

    vec3 originPlanet = cloudPlanetPos(rayOrigin);
    float topNear, topFar;
    if (!cloudIntersectSphere(originPlanet, rayDir, ADV_ATMOSPHERE_RT, topNear, topFar)) { return result; }

    float tEnter = max(topNear, 0.0);
    float tExit = max(topFar, 0.0);

    float groundNear, groundFar;
    if (cloudIntersectSphere(originPlanet, rayDir, ADV_ATMOSPHERE_RG, groundNear, groundFar)) {
        float groundHit = groundNear > 0.0 ? groundNear : groundFar;
        if (groundHit > 0.0) { tExit = min(tExit, groundHit); }
    }

    if (maxDistance > 0.0) { tExit = min(tExit, maxDistance); }
    if (tExit <= tEnter) { return result; }

    vec3 lightDir = volumetricCloudPrimaryLightDir();
    vec3 lightRadiance = volumetricCloudPrimaryLightRadiance();
    float cosTheta = dot(lightDir, rayDir);
    float rayleighPhase = phaseRayleigh(cosTheta);
    float miePhase = phaseMieHG(cosTheta, ADV_ATMOSPHERE_MIE_G);
    int stepCount = 8;
    float stepLength = (tExit - tEnter) / max(float(stepCount), 1.0);

    for (int i = 0; i < stepCount; ++i) {
        float sampleT = tEnter + (float(i) + 0.5) * stepLength;
        vec3 samplePos = originPlanet + rayDir * sampleT;
        float sampleRadius = length(samplePos);
        float sampleHeight = sampleRadius - ADV_ATMOSPHERE_RG;
        vec3 up = samplePos / max(sampleRadius, 1e-4);
        float muS = dot(up, lightDir);

        float densityRayleigh = atmosphereDensityExp(sampleHeight, ADV_ATMOSPHERE_HR);
        float densityMie = atmosphereDensityExp(sampleHeight, ADV_ATMOSPHERE_HM);
        vec3 sigmaSR = ADV_ATMOSPHERE_BETA_R * densityRayleigh;
        vec3 sigmaSM = ADV_ATMOSPHERE_BETA_M * densityMie;
        vec3 sigmaT = sigmaSR + sigmaSM;
        vec3 sunTransmittance = sampleCloudAtmosphereTransmittance(sampleRadius, muS);
        vec3 scattering = sigmaSR * rayleighPhase + sigmaSM * miePhase;

        result.scatteredLight += result.transmittance * (sunTransmittance * scattering * lightRadiance) * stepLength;
        result.transmittance *= exp(-sigmaT * stepLength);
    }

    return result;
}

float hgPhase(float g, float cosTheta) {
    float numer = 1.0 - g * g;
    float denom = 1.0 + g * g + 2.0 * g * cosTheta;
    return numer / (4.0 * PI * denom * sqrt(max(denom, 1e-5)));
}

float dualLobePhase(float g0, float g1, float w, float cosTheta) {
    return mix(hgPhase(g0, cosTheta), hgPhase(g1, cosTheta), w);
}

float powderEffectNew(float depth, float height, float viewToLight) {
    float r = -abs(viewToLight) * 0.5 + 0.5;
    r = r * r;
    height = height * (1.0 - r) + r;
    return depth * height;
}

VolumetricCloudSampleContext invalidCloudSampleContext() {
    VolumetricCloudSampleContext context;
    context.posKm = vec3(0.0);
    context.windOffset = vec3(0.0);
    context.coverage = 0.0;
    context.gradientShape = 0.0;
    context.normalizedHeight = 0.0;
    context.layerHeight01 = 0.0;
    context.layerIndex = -1;
    return context;
}

VolumetricCloudSampleContext prepareCloudLayer0(vec3 sampleWorldPos, float normalizedHeight, float layerHeight01) {
    VolumetricCloudSampleContext context = invalidCloudSampleContext();
    vec3 posMeter = cloudAbsoluteWorldPos(sampleWorldPos);
    vec3 windDirection = cloudWindDirection();
    float time = cloudTime();
    float cloudSpeed = 0.05;
    float overcastBlend = volumetricCloudOvercastBlend();
    float clearCoverageDrift = mix(volumetricCloudClearCoverageDrift(), 1.0, overcastBlend);
    float clearAmountScale = volumetricCloudClearAmountScale();
    float coverageBase =
        volumetricCloudCoverageBase(ADV_VOLUMETRIC_CLOUD_COVERAGE * 0.92 * clearCoverageDrift * clearAmountScale,
                                    max(ADV_VOLUMETRIC_CLOUD_COVERAGE + 0.12, 0.72));

    posMeter += windDirection * layerHeight01 * 500.0;
    context.posKm = posMeter * 0.001;

    vec3 curl = texture(volumetricCloudCurlTexture, (time * cloudSpeed * 50.0 + posMeter.xz) * 0.0000008 + 0.7).xyz;
    curl = curl * 2.0 - 1.0;
    context.posKm += curl * 2.0;

    context.windOffset = (windDirection + vec3(0.0, 0.1, 0.0)) * time * cloudSpeed;
    float rainMotionScale = mix(1.0, 0.10, overcastBlend);
    vec2 stableKmXZ = posMeter.xz * 0.001;
    vec2 weatherKmXZ = mix(context.posKm.xz, stableKmXZ, overcastBlend * 0.85);
    vec2 sampleUv = weatherKmXZ * ADV_VOLUMETRIC_CLOUD_WEATHER_SCALE;
    vec4 weatherValue = texture(volumetricCloudWeatherTexture, sampleUv);
    float weatherCoverage = volumetricCloudWeatherSignal(weatherValue.x, 0.72, -0.04, 0.58);

    vec2 animatedWorldXZ = posMeter.xz + vec2(time * cloudSpeed * 50.0 * rainMotionScale);
    float localCoverage = volumetricCloudMergedCoverageNoise(animatedWorldXZ, 0.00000060, vec2(0.50));
    localCoverage = cloudRemapClamped(localCoverage, 0.34, 0.76, 0.0, 1.0) * mix(0.18, 0.22, overcastBlend);

    float combinedCoverage = max(cloudSaturate(localCoverage + weatherCoverage), overcastBlend * 0.60);
    context.coverage = cloudSaturate(coverageBase * combinedCoverage);
    context.gradientShape =
        cloudRemapClamped(layerHeight01, 0.10, 0.80, coverageBase * 1.9, 0.2) *
        cloudRemapClamped(layerHeight01, 0.00, 0.10, 0.5, 1.0);
    context.normalizedHeight = normalizedHeight;
    context.layerHeight01 = layerHeight01;
    context.layerIndex = 0;
    return context;
}

VolumetricCloudSampleContext prepareCloudLayer1(vec3 sampleWorldPos, float normalizedHeight, float layerHeight01) {
    VolumetricCloudSampleContext context = invalidCloudSampleContext();
    vec3 posMeter = cloudAbsoluteWorldPos(sampleWorldPos);
    vec3 windDirection = cloudWindDirection();
    float time = cloudTime();
    float cloudSpeed = 0.05;
    float overcastBlend = volumetricCloudOvercastBlend();
    float clearCoverageDrift = mix(volumetricCloudClearCoverageDrift(), 1.0, overcastBlend);
    float clearAmountScale = volumetricCloudClearAmountScale();
    float coverageBase =
        volumetricCloudCoverageBase(ADV_VOLUMETRIC_CLOUD_COVERAGE * 0.60 * clearCoverageDrift * clearAmountScale,
                                    max(ADV_VOLUMETRIC_CLOUD_COVERAGE + 0.08, 0.54));

    posMeter += windDirection * layerHeight01 * 500.0;
    context.posKm = posMeter * 0.001;

    vec3 curl = texture(volumetricCloudCurlTexture, (time * cloudSpeed * 50.0 + posMeter.xz) * 0.000001 - 0.3).xyz;
    curl = curl * 2.0 - 1.0;
    context.posKm += curl * 5.0;

    context.windOffset = (windDirection + vec3(0.0, 0.1, 0.0)) * time * cloudSpeed;
    float rainMotionScale = mix(1.0, 0.10, overcastBlend);
    vec2 stableKmXZ = posMeter.xz * 0.001;
    vec2 weatherKmXZ = mix(context.posKm.xz, stableKmXZ, overcastBlend * 0.85);
    vec2 sampleUv = weatherKmXZ * ADV_VOLUMETRIC_CLOUD_WEATHER_SCALE * 0.5 + 0.39;
    sampleUv.y *= 2.0;
    vec4 weatherValue = texture(volumetricCloudWeatherTexture, sampleUv);
    float weatherCoverage = volumetricCloudWeatherSignal(weatherValue.x, 0.56, -0.12, 0.34);

    vec2 animatedWorldXZ = posMeter.xz + vec2(time * cloudSpeed * 50.0 * rainMotionScale);
    float localCoverage = volumetricCloudMergedCoverageNoise(animatedWorldXZ, 0.00000072, vec2(-0.11, -0.11));
    localCoverage = cloudRemapClamped(localCoverage, 0.38, 0.78, 0.0, 1.0) * mix(0.24, 0.28, overcastBlend);

    float combinedCoverage = max(cloudSaturate(localCoverage + weatherCoverage), overcastBlend * 0.36);
    context.coverage = cloudSaturate(coverageBase * combinedCoverage);
    context.gradientShape =
        cloudRemapClamped(layerHeight01, 0.00, 0.01, 0.1, 1.0) *
        cloudRemapClamped(layerHeight01, 0.10, 0.80, 0.7, 0.2);
    context.normalizedHeight = normalizedHeight;
    context.layerHeight01 = layerHeight01;
    context.layerIndex = 1;
    return context;
}

VolumetricCloudSampleContext prepareCloudLayer2(vec3 sampleWorldPos, float normalizedHeight, float layerHeight01) {
    VolumetricCloudSampleContext context = invalidCloudSampleContext();
    vec3 posMeter = cloudAbsoluteWorldPos(sampleWorldPos);
    vec3 windDirection = cloudWindDirection();
    float time = cloudTime();
    float cloudSpeed = 0.05;
    float overcastBlend = volumetricCloudOvercastBlend();
    float clearCoverageDrift = mix(volumetricCloudClearCoverageDrift(), 1.0, overcastBlend);
    float clearAmountScale = volumetricCloudClearAmountScale();
    float coverageBase =
        volumetricCloudCoverageBase(ADV_VOLUMETRIC_CLOUD_COVERAGE * 0.46 * clearCoverageDrift * clearAmountScale,
                                    ADV_VOLUMETRIC_CLOUD_COVERAGE * 0.22);

    posMeter += windDirection * layerHeight01 * 500.0;
    context.posKm = posMeter * 0.001;

    vec3 curl =
        texture(volumetricCloudCurlTexture, (time * cloudSpeed * 50.0 + posMeter.xz) * 0.00000125 + 0.7).xyz;
    curl = curl * 2.0 - 1.0;
    context.posKm += curl * 10.0;

    context.windOffset = (windDirection + vec3(0.0, 0.1, 0.0)) * time * cloudSpeed;
    vec2 stableKmXZ = posMeter.xz * 0.001;
    vec2 weatherKmXZ = mix(context.posKm.xz, stableKmXZ, overcastBlend * 0.85);
    vec2 sampleUv = weatherKmXZ * ADV_VOLUMETRIC_CLOUD_WEATHER_SCALE * 0.6 + 0.739;
    sampleUv.y *= 6.0;
    vec4 weatherValue = texture(volumetricCloudWeatherTexture, sampleUv);
    float weatherCoverage = volumetricCloudWeatherSignal(weatherValue.x, 0.35, -0.30, 0.0);

    float localCoverage =
        texture(volumetricCloudCoverageNoiseTexture, (time * cloudSpeed * 50.0 + posMeter.xz) * 0.000001 - 0.39).x;
    localCoverage = cloudSaturate(1.0 - pow(localCoverage, 8.0)) * mix(0.26, 0.12, overcastBlend);

    context.coverage = cloudSaturate(coverageBase * (localCoverage + weatherCoverage));
    context.gradientShape =
        cloudRemapClamped(layerHeight01, 0.00, 0.01, 0.1, 1.0) *
        cloudRemapClamped(layerHeight01, 0.10, 0.20, 0.8, 0.5);
    context.normalizedHeight = normalizedHeight;
    context.layerHeight01 = layerHeight01;
    context.layerIndex = 2;
    return context;
}

VolumetricCloudSampleContext sampleCloudContext(vec3 sampleWorldPos) {
    vec3 planetPos = cloudPlanetPos(sampleWorldPos);
    float normalizedHeight = cloudHeight01(planetPos);
    if (normalizedHeight <= 0.0 || normalizedHeight >= 1.0) { return invalidCloudSampleContext(); }

    if (normalizedHeight < 0.4) {
        float layerHeight01 = normalizedHeight / 0.4;
        return prepareCloudLayer0(sampleWorldPos, normalizedHeight, layerHeight01);
    }
    if (normalizedHeight < 0.8) {
        float layerHeight01 = (normalizedHeight - 0.4) / 0.4;
        return prepareCloudLayer1(sampleWorldPos, normalizedHeight, layerHeight01);
    }

    float layerHeight01 = (normalizedHeight - 0.8) / 0.2;
    return prepareCloudLayer2(sampleWorldPos, normalizedHeight, layerHeight01);
}

float sampleCloudMacroOccupancyFromContext(VolumetricCloudSampleContext context) {
    if (context.layerIndex < 0) { return 0.0; }
    return cloudSaturate(context.coverage * max(cloudSaturate(context.gradientShape), 0.35));
}

float sampleCloudDensityFromContext(VolumetricCloudSampleContext context) {
    if (context.layerIndex < 0) { return 0.0; }

    if (context.layerIndex == 0) {
        float densityBase = ADV_VOLUMETRIC_CLOUD_DENSITY * mix(1.60, 2.10, volumetricCloudOvercastBlend());
        float basicNoise =
            texture(volumetricCloudBasicNoiseTexture, fract((context.posKm + context.windOffset) * ADV_VOLUMETRIC_CLOUD_BASE_SCALE)).r;
        float basicCloudNoise = context.gradientShape * basicNoise;
        float basicCloudWithCoverage =
            context.coverage * cloudRemapClamped(basicCloudNoise, 1.0 - context.coverage, 1.0, 0.0, 1.0);

        vec3 sampleDetailNoise = context.posKm - context.windOffset * 0.15;
        float detailNoiseComposite =
            texture(volumetricCloudDetailNoiseTexture, fract(sampleDetailNoise * ADV_VOLUMETRIC_CLOUD_DETAIL_SCALE)).r;
        float detailNoiseMixByHeight =
            0.2 * mix(detailNoiseComposite, 1.0 - detailNoiseComposite, cloudSaturate(context.layerHeight01 * 10.0));

        float densityShape =
            cloudSaturate(0.01 + (1.0 - context.layerHeight01) * 0.5) * 0.25 *
            cloudRemapClamped(context.layerHeight01, 0.0, 0.3, 0.0, 1.0) *
            cloudRemapClamped(context.layerHeight01, 0.7, 1.0, 1.0, 0.0);

        float cloudDensity =
            densityShape * cloudRemapClamped(basicCloudWithCoverage, detailNoiseMixByHeight, 1.0, 0.0, 1.0);
        cloudDensity =
            pow(cloudDensity, cloudSaturate(1.0 - context.layerHeight01) * 0.4 + 0.1) * densityBase * 0.1;
        return cloudSaturate(cloudDensity);
    }

    if (context.layerIndex == 1) {
        float densityBase = ADV_VOLUMETRIC_CLOUD_DENSITY * mix(0.26, 0.40, volumetricCloudOvercastBlend());
        float basicNoise = texture(volumetricCloudBasicNoiseTexture,
                                   fract(2.0 * (context.posKm + context.windOffset) * ADV_VOLUMETRIC_CLOUD_BASE_SCALE))
                               .r;
        float basicCloudNoise = context.gradientShape * basicNoise;
        float basicCloudWithCoverage =
            context.coverage * cloudRemapClamped(basicCloudNoise, 1.0 - context.coverage, 1.0, 0.0, 1.0);

        vec3 sampleDetailNoise = context.posKm - context.windOffset * 0.15;
        float detailNoiseComposite = texture(
                                         volumetricCloudDetailNoiseTexture,
                                         fract(2.0 * sampleDetailNoise * ADV_VOLUMETRIC_CLOUD_DETAIL_SCALE))
                                         .r;
        float detailNoiseMixByHeight =
            0.2 * mix(detailNoiseComposite, 1.0 - detailNoiseComposite, cloudSaturate(context.layerHeight01 * 10.0));

        float densityShape =
            cloudSaturate(0.01 + (1.0 - context.layerHeight01) * 0.5) * 0.1 *
            cloudRemapClamped(context.layerHeight01, 0.0, 0.3, 0.0, 1.0) *
            cloudRemapClamped(context.layerHeight01, 0.7, 1.0, 1.0, 0.0);

        float cloudDensity =
            densityShape * cloudRemapClamped(basicCloudWithCoverage, detailNoiseMixByHeight, 1.0, 0.0, 1.0);
        cloudDensity =
            pow(cloudDensity, cloudSaturate(1.0 - context.layerHeight01) * 0.4 + 0.1) * densityBase * 0.1;
        return cloudSaturate(cloudDensity);
    }

    float densityBase = ADV_VOLUMETRIC_CLOUD_DENSITY * mix(0.12, 0.09, volumetricCloudOvercastBlend());
    vec3 posS = 3.0 * (context.posKm + context.windOffset + 0.39) * ADV_VOLUMETRIC_CLOUD_BASE_SCALE;
    float basicNoise = texture(volumetricCloudBasicNoiseTexture, fract(posS)).r;
    float basicCloudNoise = context.gradientShape * basicNoise;
    float basicCloudWithCoverage =
        context.coverage * cloudRemapClamped(basicCloudNoise, 1.0 - context.coverage, 1.0, 0.0, 1.0);

    float densityShape =
        cloudSaturate(0.01 + (1.0 - context.layerHeight01) * 0.5) * 0.1 *
        cloudRemapClamped(context.layerHeight01, 0.0, 0.3, 0.0, 1.0) *
        cloudRemapClamped(context.layerHeight01, 0.7, 1.0, 1.0, 0.0);

    return cloudSaturate(densityShape * basicCloudWithCoverage * densityBase);
}

int cloudSkipStepCount(float macroOccupancy, bool beforeFirstHit) {
    if (macroOccupancy < 0.004) { return beforeFirstHit ? 6 : 3; }
    if (macroOccupancy < 0.012) { return beforeFirstHit ? 4 : 2; }
    if (macroOccupancy < 0.025) { return 2; }
    return 1;
}

vec2 getParticipatingMediaPhase(float basePhase) {
    float uniformPhase = 1.0 / (4.0 * PI);
    return vec2(basePhase, mix(uniformPhase, basePhase, 0.5));
}

vec2 marchCloudLightTransport(vec3 rayOrigin, vec3 rayDir, float maxDistance, int minimumSteps, float multiScatterExtinction) {
    if (maxDistance <= 1e-3 || minimumSteps <= 0) { return vec2(1.0); }

    int stepCount = max(minimumSteps, 1);
    float stepLength = maxDistance / max(float(stepCount), 1.0);
    float extinctionAcc = 0.0;

    for (int i = 0; i < stepCount; ++i) {
        vec3 samplePos = rayOrigin + rayDir * (stepLength * (float(i) + 0.5));
        VolumetricCloudSampleContext context = sampleCloudContext(samplePos);
        float macroOccupancy = sampleCloudMacroOccupancyFromContext(context);
        int skipCount = cloudSkipStepCount(macroOccupancy, false);
        if (skipCount > 1) {
            i += min(skipCount - 1, stepCount - 1 - i);
            continue;
        }

        float density = sampleCloudDensityFromContext(context);
        if (density > 1e-5) { extinctionAcc += density * stepLength; }
    }

    float singleScatter = exp(-extinctionAcc);
    float multiScatter = exp(-extinctionAcc * multiScatterExtinction);
    return vec2(singleScatter, multiScatter);
}

float volumetricCloudLightVisibility(vec3 absoluteWorldPos, vec3 lightDir, int minimumSteps, float multiScatterExtinction) {
    if (ADV_CLOUD_MODE != 2u || ADV_VOLUMETRIC_CLOUD_CAST_SHADOW == 0) { return 1.0; }

    float horizonShadowWeight = smoothstep(0.06, 0.24, max(lightDir.y, 0.0));
    float cloudShadowWeight = horizonShadowWeight * clamp(ADV_VOLUMETRIC_CLOUD_SHADOW_STRENGTH, 0.0, 1.0);
    float cloudShadowSoftness = clamp(ADV_VOLUMETRIC_CLOUD_SHADOW_SOFTNESS, 0.0, 1.0);
    if (cloudShadowWeight <= 1e-4) { return 1.0; }

    vec3 rayOrigin = absoluteWorldPos - vec3(worldUBO.cameraPos.xyz);
    vec3 traceOrigin = rayOrigin + lightDir * 5.0;
    vec3 cloudTraceOrigin;
    float lightDistance;
    if (!volumetricCloudTraceSegment(traceOrigin, lightDir, cloudTraceOrigin, lightDistance)) { return 1.0; }
    traceOrigin = cloudTraceOrigin;

    float cloudLightVisibility =
        marchCloudLightTransport(traceOrigin, lightDir, lightDistance, max(minimumSteps, 1), multiScatterExtinction).x;
    if (cloudShadowSoftness > 1e-4) {
        vec3 shadowTangent = cloudSafeNormalize(cross(lightDir, vec3(0.0, 1.0, 0.0)), vec3(1.0, 0.0, 0.0));
        vec3 shadowBitangent = cloudSafeNormalize(cross(lightDir, shadowTangent), vec3(0.0, 0.0, 1.0));
        vec3 shadowOffsetDir = cloudSafeNormalize(shadowTangent + shadowBitangent * 0.5, shadowTangent);
        float shadowSampleRadius = clamp(lightDistance * 0.04, 40.0, 450.0) * cloudShadowSoftness;
        int offsetStepCount = max(minimumSteps / 2, 1);

        vec3 positiveTraceOrigin = traceOrigin + shadowOffsetDir * shadowSampleRadius;
        vec3 positiveCloudTraceOrigin;
        float positiveLightDistance;
        float positiveCloudLightVisibility =
            !volumetricCloudTraceSegment(positiveTraceOrigin, lightDir, positiveCloudTraceOrigin,
                                         positiveLightDistance) ?
                1.0 :
                marchCloudLightTransport(positiveCloudTraceOrigin, lightDir, positiveLightDistance, offsetStepCount,
                                         multiScatterExtinction)
                    .x;

        vec3 negativeTraceOrigin = traceOrigin - shadowOffsetDir * shadowSampleRadius;
        vec3 negativeCloudTraceOrigin;
        float negativeLightDistance;
        float negativeCloudLightVisibility =
            !volumetricCloudTraceSegment(negativeTraceOrigin, lightDir, negativeCloudTraceOrigin,
                                         negativeLightDistance) ?
                1.0 :
                marchCloudLightTransport(negativeCloudTraceOrigin, lightDir, negativeLightDistance, offsetStepCount,
                                         multiScatterExtinction)
                    .x;

        float filteredCloudLightVisibility =
            cloudLightVisibility * 0.50 + (positiveCloudLightVisibility + negativeCloudLightVisibility) * 0.25;
        cloudLightVisibility = mix(cloudLightVisibility, filteredCloudLightVisibility, cloudShadowSoftness);
    }
    return mix(1.0, cloudLightVisibility, cloudShadowWeight);
}

vec3 volumetricCloudSkyFullHorizonBase(vec3 rayDir);
vec3 volumetricCloudSkyFullBelowHorizonBase(vec3 rayDir);

vec3 volumetricCloudRainOvercastBase() {
    vec3 sunDir = celestialSunDirection();
    return mix(vec3(0.045, 0.050, 0.055), vec3(0.12, 0.13, 0.14), smoothstep(-0.3, 0.3, sunDir.y));
}

vec3 volumetricCloudRainFilter(vec3 clearSky, float darkness) {
    float rainBlend = volumetricCloudRainBlend();
    float lum = dot(max(clearSky, vec3(0.0)), vec3(0.2126, 0.7152, 0.0722));
    vec3 desaturatedSky = mix(clearSky, vec3(lum), 0.72);
    vec3 rainyBase = volumetricCloudRainOvercastBase() * darkness;
    vec3 rainySky = mix(desaturatedSky, rainyBase, 0.85);
    return mix(clearSky, rainySky, rainBlend);
}

vec3 sampleCloudAmbientIrradiance(vec3 up) {
    vec3 horizonX = volumetricCloudSkyFullHorizonBase(normalize(vec3(1.0, max(up.y, 0.0), 0.0)));
    vec3 horizonZ = volumetricCloudSkyFullHorizonBase(normalize(vec3(0.0, max(up.y, 0.0), 1.0)));
    vec3 skyUp = volumetricCloudSkyRadiance(up);
    vec3 ambient = skyUp * 0.55 + 0.225 * (horizonX + horizonZ);
    return ambient * ADV_VOLUMETRIC_CLOUD_AMBIENT_STRENGTH;
}

vec3 sampleCloudAmbientIrradianceUp() {
    return sampleCloudAmbientIrradiance(vec3(0.0, 1.0, 0.0));
}

vec3 volumetricCloudGroundToCloudTransfertIsoScatter(vec3 up, float transmittance, float powderEffect) {
    vec3 lowerHorizon = volumetricCloudSkyFullBelowHorizonBase(-up);
    vec3 horizon = volumetricCloudSkyFullHorizonBase(vec3(up.x, 0.02, up.z));
    vec3 groundBounce = mix(lowerHorizon, horizon, 0.35);
    float lift = 1.0 - dot(volumetricCloudPrimaryLightDir(), vec3(0.0, 1.0, 0.0));
    lift *= 1.0 + lift;
    return groundBounce * powderEffect * clamp(lift, 0.0, 1.0) *
           mix(vec3(1.0), vec3(1.35), cloudSaturate(1.0 - transmittance));
}

float volumetricCloudLuminance(vec3 color) {
    return dot(color, vec3(0.2126, 0.7152, 0.0722));
}

vec3 volumetricCloudSkyFullHorizonBase(vec3 rayDir) {
    float blend = smoothstep(-0.05, 0.02, rayDir.y);
    vec3 clampedDir = rayDir;
    clampedDir.y = max(clampedDir.y, ADV_ATMOSPHERE_MIN_VIEW_COS);
    clampedDir = normalize(mix(vec3(rayDir.x, ADV_ATMOSPHERE_MIN_VIEW_COS, rayDir.z), clampedDir, blend));
    return volumetricCloudRainFilter(texture(skyFull, clampedDir).rgb, 0.95);
}

vec3 volumetricCloudSkyFullBelowHorizonBase(vec3 rayDir) {
    const float belowHorizonCos = ADV_ATMOSPHERE_MIN_VIEW_COS - 0.035;
    float blend = smoothstep(-0.06, 0.03, rayDir.y);
    vec3 biasedDir = rayDir;
    biasedDir.y = max(biasedDir.y, belowHorizonCos);
    biasedDir = normalize(mix(vec3(rayDir.x, belowHorizonCos, rayDir.z), biasedDir, blend));
    return volumetricCloudRainFilter(texture(skyFull, biasedDir).rgb, 0.70);
}

float volumetricCloudHorizonFade(vec3 rayDir, vec3 cloudWorldPos, float cloudDistance) {
    vec3 planetPos = cloudPlanetPos(cloudWorldPos);
    vec3 up = cloudSafeNormalize(planetPos, vec3(0.0, 1.0, 0.0));

    float horizonView = 1.0 - smoothstep(0.015, 0.070, dot(rayDir, up));
    float farFade = smoothstep(24000.0, 90000.0, cloudDistance);

    float height01 = cloudHeight01(planetPos);
    float heightFade = 1.0 - smoothstep(0.22, 0.65, height01);

    return clamp(horizonView * farFade * heightFade, 0.0, 1.0);
}

float volumetricCloudHorizonMatch(vec3 rayDir) {
    return 1.0 - smoothstep(0.035, 0.16, rayDir.y);
}

float volumetricCloudHorizonMissMix(vec3 rayDir) {
    return 1.0 - smoothstep(0.0, 0.08, rayDir.y);
}

VolumetricCloudResult applyVolumetricCloudBudgeted(
    vec3 rayOrigin, vec3 rayDir, vec3 skyBackgroundRadiance, int viewStepCount, int lightStepCount, int ambientStepCount) {
    VolumetricCloudResult result;
    result.color = skyBackgroundRadiance;
    result.transmittance = 1.0;
    result.hit = 0.0;

    if (ADV_CLOUD_MODE != 2u) { return result; }

    float tEnter, tExit;
    if (!intersectCloudLayer(rayOrigin, rayDir, tEnter, tExit)) { return result; }

    float totalLength = tExit - tEnter;
    if (totalLength <= 1e-3) { return result; }

    vec3 dominantLightDir = volumetricCloudPrimaryLightDir();
    vec3 primaryLightRadiance = volumetricCloudPrimaryLightRadiance();
    vec3 horizonBase = volumetricCloudSkyFullHorizonBase(rayDir);
    vec3 belowHorizonBase = volumetricCloudSkyFullBelowHorizonBase(rayDir);
    float horizonMatch = volumetricCloudHorizonMatch(rayDir);
    float viewToLight = clamp(dot(rayDir, dominantLightDir), -1.0, 1.0);
    float basePhase = dualLobePhase(0.5, -0.5, 0.5, -viewToLight);
    vec2 phases = getParticipatingMediaPhase(basePhase);

    viewStepCount = max(viewStepCount, 1);
    lightStepCount = max(lightStepCount, 1);
    ambientStepCount = max(ambientStepCount, 1);
    int lightTransportReuseSteps = clamp(viewStepCount / 32, 2, 6);
    int ambientTransportReuseSteps = clamp(viewStepCount / 24, 2, 8);
    float stepLength = totalLength / float(viewStepCount);
    float transmittance = 1.0;
    vec3 scattering = vec3(0.0);
    vec3 weightedWorldPos = vec3(0.0);
    float weightedWorldPosWeight = 0.0;
    float sunsetScale = 1.0 + cloudSaturate(1.0 - dominantLightDir.y * 2.0);
    float horizonShadowWeight = smoothstep(0.06, 0.24, max(dominantLightDir.y, 0.0));
    float cloudShadowWeight = horizonShadowWeight * clamp(ADV_VOLUMETRIC_CLOUD_SHADOW_STRENGTH, 0.0, 1.0);
    float cloudShadowSoftness = clamp(ADV_VOLUMETRIC_CLOUD_SHADOW_SOFTNESS, 0.0, 1.0);
    float rayJitter = cloudHash12(vec2(gl_LaunchIDEXT.xy) + vec2(float(worldUBO.seed & 255u), 17.0));
    vec2 cachedSunTransport = vec2(1.0);
    vec2 cachedAmbientTransport = vec2(1.0);
    int sunTransportCountdown = 0;
    int ambientTransportCountdown = 0;
    float previousDensity = 0.0;
    bool previousHadDensity = false;
    float rainBlend = volumetricCloudRainBlend();

    for (int i = 0; i < viewStepCount; ++i) {
        float sampleT = min(tEnter + (float(i) + rayJitter) * stepLength, tExit);
        vec3 samplePos = rayOrigin + rayDir * sampleT;

        VolumetricCloudSampleContext context = sampleCloudContext(samplePos);
        float macroOccupancy = sampleCloudMacroOccupancyFromContext(context);
        int skipCount = cloudSkipStepCount(macroOccupancy, result.hit < 0.5);
        if (skipCount > 1) {
            i += min(skipCount - 1, viewStepCount - 1 - i);
            continue;
        }

        float density = sampleCloudDensityFromContext(context);
        if (density <= 1e-5) { continue; }

        result.hit = 1.0;
        weightedWorldPos += samplePos * transmittance;
        weightedWorldPosWeight += transmittance;

        float opticalDepth = density * stepLength;
        float stepTransmittance = max(exp(-opticalDepth), exp(-opticalDepth * 0.25) * 0.70);
        vec3 samplePlanetPos = cloudPlanetPos(samplePos);
        vec3 up = cloudSafeNormalize(samplePlanetPos, vec3(0.0, 1.0, 0.0));

        bool refreshSunTransport = sunTransportCountdown <= 0;
        bool refreshAmbientTransport = ambientTransportCountdown <= 0;
        if (previousHadDensity) {
            float densityDelta = abs(density - previousDensity);
            refreshSunTransport = refreshSunTransport || densityDelta > 0.06 || context.coverage < 0.12;
            refreshAmbientTransport = refreshAmbientTransport || densityDelta > 0.08;
        }

        if (refreshSunTransport) {
            if (cloudShadowWeight <= 1e-4) {
                cachedSunTransport = vec2(1.0);
            } else {
                vec3 sunTraceOrigin = samplePos + dominantLightDir * 5.0;
                float sunDistance = volumetricCloudSegmentLength(sunTraceOrigin, dominantLightDir);
                cachedSunTransport =
                    marchCloudLightTransport(sunTraceOrigin, dominantLightDir, sunDistance, lightStepCount, 0.175);
                if (cloudShadowSoftness > 1e-4) {
                    vec3 shadowTangent =
                        cloudSafeNormalize(cross(dominantLightDir, vec3(0.0, 1.0, 0.0)), vec3(1.0, 0.0, 0.0));
                    vec3 shadowBitangent =
                        cloudSafeNormalize(cross(dominantLightDir, shadowTangent), vec3(0.0, 0.0, 1.0));
                    vec3 shadowOffsetDir = cloudSafeNormalize(shadowTangent + shadowBitangent * 0.5, shadowTangent);
                    float shadowSampleRadius = clamp(sunDistance * 0.04, 40.0, 450.0) * cloudShadowSoftness;
                    int offsetStepCount = max(lightStepCount / 2, 1);

                    vec3 positiveSunTraceOrigin = sunTraceOrigin + shadowOffsetDir * shadowSampleRadius;
                    float positiveSunDistance = volumetricCloudSegmentLength(positiveSunTraceOrigin, dominantLightDir);
                    vec2 positiveSunTransport =
                        positiveSunDistance <= 1e-3 ?
                            vec2(1.0) :
                            marchCloudLightTransport(positiveSunTraceOrigin, dominantLightDir, positiveSunDistance,
                                                     offsetStepCount, 0.175);

                    vec3 negativeSunTraceOrigin = sunTraceOrigin - shadowOffsetDir * shadowSampleRadius;
                    float negativeSunDistance = volumetricCloudSegmentLength(negativeSunTraceOrigin, dominantLightDir);
                    vec2 negativeSunTransport =
                        negativeSunDistance <= 1e-3 ?
                            vec2(1.0) :
                            marchCloudLightTransport(negativeSunTraceOrigin, dominantLightDir, negativeSunDistance,
                                                     offsetStepCount, 0.175);

                    vec2 filteredSunTransport =
                        cachedSunTransport * 0.50 + (positiveSunTransport + negativeSunTransport) * 0.25;
                    cachedSunTransport = mix(cachedSunTransport, filteredSunTransport, cloudShadowSoftness);
                }
                cachedSunTransport = mix(vec2(1.0), cachedSunTransport, cloudShadowWeight);
            }
            sunTransportCountdown = lightTransportReuseSteps;
        } else {
            sunTransportCountdown -= 1;
        }

        if (refreshAmbientTransport) {
            vec3 ambientTraceOrigin = samplePos + up * 5.0;
            float ambientDistance = volumetricCloudSegmentLength(ambientTraceOrigin, up);
            cachedAmbientTransport =
                marchCloudLightTransport(ambientTraceOrigin, up, ambientDistance, ambientStepCount, 0.50);
            ambientTransportCountdown = ambientTransportReuseSteps;
        } else {
            ambientTransportCountdown -= 1;
        }

        float sampleRadius = clamp(length(samplePlanetPos), ADV_ATMOSPHERE_RG, ADV_ATMOSPHERE_RT);
        vec3 sunAtmosphereTransmittance =
            sampleCloudAtmosphereTransmittance(sampleRadius, dot(up, dominantLightDir));
        float depthProbability =
            pow(clamp(density * 8.0, 0.0, 1.0), cloudRemapClamped(context.layerHeight01, 0.3, 0.85, 0.5, 2.0)) +
            0.05;
        float verticalProbability = pow(cloudRemapClamped(context.layerHeight01, 0.07, 0.22, 0.1, 1.0), 0.8);
        float powderEffect =
            mix(1.0, powderEffectNew(depthProbability, verticalProbability, viewToLight),
                clamp(ADV_VOLUMETRIC_CLOUD_POWDER_STRENGTH, 0.0, 1.0));

        vec3 ambientIrradiance = sampleCloudAmbientIrradiance(up);
        vec3 topAmbientIrradiance = sampleCloudAmbientIrradianceUp();
        vec3 groundToCloudTransfertIsoScatter =
            volumetricCloudGroundToCloudTransfertIsoScatter(up, transmittance, powderEffect);
        vec3 distantAmbient = mix(belowHorizonBase, topAmbientIrradiance, 0.35);
        vec3 sunlightTerm = sunAtmosphereTransmittance * primaryLightRadiance;
        vec3 sigmaScatter0 = vec3(density);
        vec3 sigmaScatter1 = sigmaScatter0;
        float sigmaE0 = max(density, 1e-5);
        float sigmaE1 = max(sigmaE0 * (0.175 / sunsetScale), 1e-5);

        for (int ms = 1; ms >= 0; --ms) {
            float phaseTerm = ms == 0 ? phases.x : phases.y;
            float lightVisibility = ms == 0 ? cachedSunTransport.x : cachedSunTransport.y;
            vec3 incidentLight = lightVisibility * sunlightTerm * phaseTerm * powderEffect;
            if (ms == 0) {
                incidentLight += cachedAmbientTransport.x * ambientIrradiance * powderEffect;
                incidentLight += groundToCloudTransfertIsoScatter;
                incidentLight += distantAmbient * 0.35 * ADV_VOLUMETRIC_CLOUD_AMBIENT_STRENGTH;
            }

            if (horizonMatch > 1e-4) {
                float incidentLum = volumetricCloudLuminance(max(incidentLight, vec3(0.0)));
                float baseLum = max(volumetricCloudLuminance(max(horizonBase, vec3(0.0))), 1e-4);
                vec3 matchedIncident = max(horizonBase, vec3(0.0)) * (incidentLum / baseLum);
                float matchStrength = horizonMatch * mix(0.9, 0.30, rainBlend);
                incidentLight = mix(incidentLight, matchedIncident, matchStrength);
            }

            vec3 scatterCoefficients = ms == 0 ? sigmaScatter0 : sigmaScatter1;
            float extinctionCoefficients = ms == 0 ? sigmaE0 : sigmaE1;
            vec3 litStep = incidentLight * scatterCoefficients;
            vec3 stepScatter = vec3(transmittance) * (litStep - litStep * stepTransmittance) / extinctionCoefficients;
            scattering += stepScatter;

            if (ms == 0) { transmittance *= stepTransmittance; }
        }

        if (transmittance <= 1e-3) {
            transmittance = 0.0;
            break;
        }

        previousDensity = density;
        previousHadDensity = true;
    }

    result.transmittance = transmittance;
    if (weightedWorldPosWeight <= 1e-4) {
        result.color = skyBackgroundRadiance * transmittance + scattering;
        float missMix = volumetricCloudHorizonMissMix(rayDir);
        if (missMix > 1e-4) {
            vec3 belowBase = volumetricCloudSkyFullBelowHorizonBase(rayDir);
            result.color = mix(result.color, belowBase, missMix);
            result.transmittance = mix(result.transmittance, 1.0, missMix);
        }
        return result;
    }

    vec3 cloudWorldPos = weightedWorldPos / weightedWorldPosWeight;
    float cloudDistance = max(dot(cloudWorldPos - rayOrigin, rayDir), 0.0);
    VolumetricCloudAtmosphereSegmentResult airPerspective = integrateAtmosphereSegment(rayOrigin, rayDir, cloudDistance);
    scattering = scattering * airPerspective.transmittance + airPerspective.scatteredLight * (1.0 - transmittance);

    float fade = volumetricCloudHorizonFade(rayDir, cloudWorldPos, cloudDistance);
    horizonBase = max(skyBackgroundRadiance, horizonBase);
    if (fade > 1e-4) {
        float scatteringLum = volumetricCloudLuminance(max(scattering, vec3(0.0)));
        float horizonLum = max(volumetricCloudLuminance(max(horizonBase, vec3(0.0))), 1e-4);
        vec3 horizonTintedScattering = max(horizonBase, vec3(0.0)) * (scatteringLum / horizonLum);
        scattering = mix(scattering, horizonTintedScattering, fade * mix(0.85, 0.35, rainBlend));
    }
    if (fade > 1e-4) { transmittance = mix(transmittance, 1.0, fade); }
    scattering *= (1.0 - fade);

    result.transmittance = transmittance;
    result.color = horizonBase * transmittance + scattering;

    float missMix = volumetricCloudHorizonMissMix(rayDir);
    if (missMix > 1e-4) {
        vec3 belowBase = volumetricCloudSkyFullBelowHorizonBase(rayDir);
        result.color = mix(result.color, belowBase, missMix);
        result.transmittance = mix(result.transmittance, 1.0, missMix);
    }

    return result;
}

VolumetricCloudResult applyVolumetricCloud(vec3 rayOrigin, vec3 rayDir, vec3 skyBackgroundRadiance) {
    return applyVolumetricCloudBudgeted(rayOrigin, rayDir, skyBackgroundRadiance,
                                        ADV_VOLUMETRIC_CLOUD_VIEW_STEPS,
                                        ADV_VOLUMETRIC_CLOUD_LIGHT_STEPS,
                                        ADV_VOLUMETRIC_CLOUD_AMBIENT_STEPS);
}

#endif
