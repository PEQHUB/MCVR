#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : require

#include "../util/ray_payloads.glsl"
#include "common/shared.hpp"

layout(set = 0, binding = 1) uniform sampler2D transLUT;

layout(set = 2, binding = 0) uniform WorldUniform {
    WorldUBO worldUBO;
};

layout(set = 2, binding = 2) uniform SkyUniform {
    SkyUBO skyUBO;
};

layout(location = 1) rayPayloadInEXT ShadowRay shadowRay;

vec2 transmittanceUv(float r, float mu, SkyUBO ubo) {
    float u = clamp(mu * 0.5 + 0.5, 0.0, 1.0);
    float v = clamp((r - ubo.Rg) / (ubo.Rt - ubo.Rg), 0.0, 1.0);
    return vec2(u, v);
}

vec3 sampleTransmittance(float r, float mu) {
    vec2 uv = transmittanceUv(r, mu, skyUBO);
    return texture(transLUT, uv).rgb;
}

void main() {
    vec3 toSun = normalize(skyUBO.sunDirection);
    vec3 radiance;

    if (toSun.y > 0) {
        vec3 C = vec3(0.0, -skyUBO.Rg, 0.0);
        vec3 pWorld = gl_WorldRayOriginEXT;
        vec3 pPlanet = pWorld - C;

        float r = length(pPlanet);
        vec3 up = pPlanet / max(r, 1e-6);
        float muSun = dot(up, toSun);

        muSun = clamp(muSun, -1.0, 1.0);
        r = clamp(r, skyUBO.Rg, skyUBO.Rt);

        vec3 T = sampleTransmittance(r, muSun);

        radiance = skyUBO.sunRadiance * T;
    } else {
        radiance = skyUBO.moonRadiance;
    }

    float factor = 1.0;
    float threshold = 0.3;
    if (abs(toSun.y) < threshold) { factor = sin(PI / (2 * threshold) * abs(toSun.y)); }
    radiance *= factor;

    shadowRay.radiance += radiance * shadowRay.throughput;
}