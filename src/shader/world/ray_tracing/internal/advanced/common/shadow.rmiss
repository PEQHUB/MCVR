#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : require

#include "util/ray.glsl"
#include "util/util.glsl"
#include "common/shared.hpp"

layout(set = 5, binding = 0) uniform sampler2D transLUT;

layout(set = 2, binding = 0) uniform WorldUniform {
    WorldUBO worldUBO;
};

layout(set = 2, binding = 2) uniform SkyUniform {
    SkyUBO skyUBO;
};

layout(location = 1) rayPayloadInEXT ShadowRay shadowRay;

#include "common/celestial.glsl"

vec2 transmittanceUv(float planetRadius, float viewCos) {
    float u = clamp(viewCos * 0.5 + 0.5, 0.0, 1.0);
    float v = clamp((planetRadius - ADV_ATMOSPHERE_RG) / (ADV_ATMOSPHERE_RT - ADV_ATMOSPHERE_RG), 0.0, 1.0);
    return vec2(u, v);
}

vec3 sampleTransmittance(float planetRadius, float viewCos) {
    vec2 uv = transmittanceUv(planetRadius, viewCos);
    return sampleTexture(transLUT, uv, false).rgb;
}

void main() {
    if (shadowRay.pad0 > 0u) {
        shadowRay.radiance += vec3(1.0) * shadowRay.throughput;
        return;
    }

    vec3 sunDirection = celestialSunDirection();
    float celestialFadeThreshold = 0.3;

    vec3 atmosphereCenter = vec3(0.0, -ADV_ATMOSPHERE_RG, 0.0);
    vec3 rayOriginWorld = gl_WorldRayOriginEXT;
    vec3 rayOriginPlanet = rayOriginWorld - atmosphereCenter;

    float planetRadius = length(rayOriginPlanet);
    vec3 planetUp = rayOriginPlanet / max(planetRadius, 1e-6);
    float sunUpDot = clamp(dot(planetUp, sunDirection), -1.0, 1.0);
    planetRadius = clamp(planetRadius, ADV_ATMOSPHERE_RG, ADV_ATMOSPHERE_RT);

    vec3 atmosphereTransmittance = sampleTransmittance(planetRadius, sunUpDot);
    float sunRadianceScale =
        sin(PI / (2.0 * celestialFadeThreshold) * min(max(sunDirection.y, 0.0), celestialFadeThreshold));
    float moonRadianceScale =
        sin(PI / (2.0 * celestialFadeThreshold) * min(max(-sunDirection.y, 0.0), celestialFadeThreshold));
    vec3 radiance = ADV_SUN_RADIANCE * atmosphereTransmittance * sunRadianceScale +
                    ADV_MOON_RADIANCE * moonRadianceScale;

    shadowRay.radiance += radiance * shadowRay.throughput;
}
