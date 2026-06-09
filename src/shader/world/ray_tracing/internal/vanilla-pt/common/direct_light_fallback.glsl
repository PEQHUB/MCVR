#ifndef VPT_DIRECT_LIGHT_FALLBACK_GLSL
#define VPT_DIRECT_LIGHT_FALLBACK_GLSL

#include "common/physical_lighting.glsl"

vec2 vptDirectLightTransmittanceUv(float r, float mu) {
    float u = clamp(mu * 0.5 + 0.5, 0.0, 1.0);
    float v = clamp((r - VPT_ATMOSPHERE_RG) / (VPT_ATMOSPHERE_RT - VPT_ATMOSPHERE_RG), 0.0, 1.0);
    return vec2(u, v);
}

vec3 vptDirectLightTransmittance(float r, float mu) {
    return sampleTexture(transLUT, vptDirectLightTransmittanceUv(r, mu), false).rgb;
}

vec3 vptUnoccludedCelestialRadiance(vec3 rayOrigin) {
    vec3 toSun = celestialSunDirection();
    vec3 radiance;

    if (toSun.y > 0.0) {
        vec3 atmosphereCenter = vec3(0.0, -VPT_ATMOSPHERE_RG, 0.0);
        vec3 pPlanet = rayOrigin - atmosphereCenter;

        float r = length(pPlanet);
        vec3 up = pPlanet / max(r, 1e-6);
        float muSun = clamp(dot(up, toSun), -1.0, 1.0);
        r = clamp(r, VPT_ATMOSPHERE_RG, VPT_ATMOSPHERE_RT);

        vec3 transmittance = max(vptDirectLightTransmittance(r, muSun), vec3(0.03));
        radiance = vptPhysicalSunRadiance() * transmittance;
    } else {
        radiance = vptPhysicalMoonRadiance();
    }

    float factor = 1.0;
    float threshold = 0.3;
    if (abs(toSun.y) < threshold) { factor = sin(PI / (2.0 * threshold) * abs(toSun.y)); }
    return radiance * factor;
}

vec3 vptRecoverZeroShadowMissRadiance(vec3 rayOrigin, vec3 tracedRadiance, vec3 tracedThroughput, float hitT) {
    if (hitT != INF_DISTANCE) { return tracedRadiance; }
    if (dot(tracedRadiance, tracedRadiance) > 1e-10) { return tracedRadiance; }
    return vptUnoccludedCelestialRadiance(rayOrigin) * tracedThroughput;
}

#endif
