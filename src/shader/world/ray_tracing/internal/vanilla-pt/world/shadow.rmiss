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
#include "common/physical_lighting.glsl"

vec2 transmittanceUv(float r, float mu) {
    float u = clamp(mu * 0.5 + 0.5, 0.0, 1.0);
    float v = clamp((r - VPT_ATMOSPHERE_RG) / (VPT_ATMOSPHERE_RT - VPT_ATMOSPHERE_RG), 0.0, 1.0);
    return vec2(u, v);
}

vec3 sampleTransmittance(float r, float mu) {
    vec2 uv = transmittanceUv(r, mu);
    return sampleTexture(transLUT, uv, false).rgb;
}

void main() {
    vec3 toSun = celestialSunDirection();
    vec3 radiance;

    if (toSun.y > 0) {
        vec3 atmosphereCenter = vec3(0.0, -VPT_ATMOSPHERE_RG, 0.0);
        vec3 pWorld = gl_WorldRayOriginEXT;
        vec3 pPlanet = pWorld - atmosphereCenter;

        float r = length(pPlanet);
        vec3 up = pPlanet / max(r, 1e-6);
        float muSun = dot(up, toSun);

        muSun = clamp(muSun, -1.0, 1.0);
        r = clamp(r, VPT_ATMOSPHERE_RG, VPT_ATMOSPHERE_RT);

        vec3 transmittance = sampleTransmittance(r, muSun);

        radiance = vptPhysicalSunRadiance() * transmittance;
    } else {
        radiance = vptPhysicalMoonRadiance();
    }

    float factor = 1.0;
    float threshold = 0.3;
    if (abs(toSun.y) < threshold) { factor = sin(PI / (2 * threshold) * abs(toSun.y)); }
    radiance *= factor;

    shadowRay.radiance += radiance * shadowRay.throughput;
}
