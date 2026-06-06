#version 460
#extension GL_GOOGLE_include_directive : require

#include "common/shared.hpp"

layout(set = 2, binding = 0) uniform WorldUniform {
    WorldUBO worldUBO;
};

layout(set = 2, binding = 2) uniform SkyUniform {
    SkyUBO skyUBO;
};

#include "common/celestial.glsl"
#include "common/physical_lighting.glsl"

layout(set = 5, binding = 0) uniform sampler2D transLUT;

layout(location = 0) in vec2 texCoord;

layout(location = 0) out vec4 outColor;

bool finiteFloat(float value) {
    return !isnan(value) && !isinf(value);
}

bool finiteVec3(vec3 value) {
    return !any(isnan(value)) && !any(isinf(value));
}

float atmosphereRg() {
    return skyUBO.Rg > 1.0 ? skyUBO.Rg : VPT_ATMOSPHERE_RG;
}

float atmosphereRt() {
    float rg = atmosphereRg();
    return skyUBO.Rt > rg + 1.0 ? skyUBO.Rt : VPT_ATMOSPHERE_RT;
}

float atmosphereHr() {
    return skyUBO.Hr > 1.0 ? skyUBO.Hr : VPT_ATMOSPHERE_HR;
}

float atmosphereHm() {
    return skyUBO.Hm > 1.0 ? skyUBO.Hm : VPT_ATMOSPHERE_HM;
}

float atmosphereMinViewCos() {
    return skyUBO.minViewCos > 0.0 ? skyUBO.minViewCos : VPT_ATMOSPHERE_MIN_VIEW_COS;
}

vec3 atmosphereBetaR() {
    return dot(skyUBO.betaR, skyUBO.betaR) > 0.0 ? skyUBO.betaR : VPT_ATMOSPHERE_BETA_R;
}

vec3 atmosphereBetaM() {
    return dot(skyUBO.betaM, skyUBO.betaM) > 0.0 ? skyUBO.betaM : VPT_ATMOSPHERE_BETA_M;
}

vec3 atmosphereSourceRadiance(bool isSun) {
    return isSun ? vptPhysicalSunRadiance() : vptPhysicalMoonRadiance();
}

float skyLuminance(vec3 value) {
    return dot(max(value, vec3(0.0)), vec3(0.2126, 0.7152, 0.0722));
}

vec3 fallbackSkyRadiance(vec3 rayDir) {
    vec3 zenith = max(skyUBO.baseColor, vec3(0.015, 0.035, 0.08));
    vec3 horizon = max(skyUBO.horizonColor.rgb, zenith);
    float day = smoothstep(-0.10, 0.03, celestialSunDirection().y);
    float up = clamp(rayDir.y, 0.0, 1.0);
    float horizonWeight = pow(1.0 - up, 2.0);
    vec3 daySky = mix(zenith, horizon, horizonWeight) * max(skyUBO.envSky.x, 0.25);
    vec3 nightSky = vptPhysicalNightSkyFloor();
    return max(mix(nightSky, daySky, day), vec3(0.0005));
}

bool intersectSphere(vec3 rayOrigin, vec3 rayDir, float radius, out float tNear, out float tFar) {
    float b = dot(rayOrigin, rayDir);
    float c = dot(rayOrigin, rayOrigin) - radius * radius;
    float height = b * b - c;
    if (height < 0.0) return false;
    height = sqrt(height);
    tNear = -b - height;
    tFar = -b + height;
    return true;
}

vec2 transmittanceUv(float r, float mu) {
    float u = clamp(mu * 0.5 + 0.5, 0.0, 1.0);
    float rg = atmosphereRg();
    float rt = atmosphereRt();
    float v = clamp((r - rg) / max(rt - rg, 1.0), 0.0, 1.0);
    return vec2(u, v);
}

float densityExp(float height, float scaleHeight) {
    return exp(-max(height, 0.0) / max(scaleHeight, 1.0));
}

vec3 sampleTransmittance(float r, float mu) {
    vec2 uv = transmittanceUv(r, mu);
    vec2 invSize = 1.0 / vec2(textureSize(transLUT, 0));
    uv = clamp(uv, 0.5 * invSize, 1.0 - 0.5 * invSize);
    return texture(transLUT, uv).rgb;
}

vec3 integrateSingleScattering(vec3 rayOrigin, vec3 rayDir, bool isSun) {
    float tAtm0, tAtm1;
    float rg = atmosphereRg();
    float rt = atmosphereRt();
    if (!intersectSphere(rayOrigin, rayDir, rt, tAtm0, tAtm1)) return vec3(0.0);
    tAtm0 = max(tAtm0, 0.0);

    float tG0, tG1;
    if (intersectSphere(rayOrigin, rayDir, rg, tG0, tG1)) {
        float tHitG = tG0 > 0.0 ? tG0 : tG1;
        if (tHitG > 0.0) tAtm1 = min(tAtm1, tHitG);
    }

    const int steps = 32;
    float dt = (tAtm1 - tAtm0) / float(steps);

    vec3 scatteredRadiance = vec3(0.0);
    vec3 viewTransmittance = vec3(1.0);

    vec3 lightDir = isSun ? celestialSunDirection() : celestialMoonDirection();
    float cosTheta = dot(lightDir, rayDir);
    float rayleighPhase = 3.0 / (16.0 * PI) * (1.0 + cosTheta * cosTheta);
    float clampedCosTheta = clamp(cosTheta, -1.0, 1.0);
    float mieG = clamp(VPT_ATMOSPHERE_MIE_G, -0.999, 0.999);
    float mieG2 = mieG * mieG;
    float mieDenominatorBase = 1.0 + mieG2 - 2.0 * mieG * clampedCosTheta;
    float mieDenominator = pow(mieDenominatorBase + 1e-6, 1.5);
    float miePhase = (1.0 - mieG2) / (4.0 * PI * mieDenominator);

    for (int i = 0; i < steps; i++) {
        float t = tAtm0 + (float(i) + 0.5) * dt;
        vec3 x = rayOrigin + rayDir * t;

        float r = length(x);
        float height = r - rg;

        float dR = densityExp(height, atmosphereHr());
        float dM = densityExp(height, atmosphereHm());

        vec3 sigmaSR = atmosphereBetaR() * dR;
        vec3 sigmaSM = atmosphereBetaM() * dM;
        vec3 sigmaT = sigmaSR + sigmaSM;

        vec3 up = x / r;
        float muS = dot(up, lightDir);
        vec3 lightTransmittance = sampleTransmittance(r, muS);

        vec3 scattering = sigmaSR * rayleighPhase + sigmaSM * miePhase;
        vec3 scatteredSample =
            viewTransmittance * (lightTransmittance * (scattering * atmosphereSourceRadiance(isSun))) * dt;

        scatteredRadiance += scatteredSample;
        viewTransmittance *= exp(-sigmaT * dt);
    }

    return scatteredRadiance;
}

void main() {
    vec2 cubeUv = texCoord * 2.0 - 1.0;
    vec3 rayDir = normalize(vec3(-cubeUv.x, -cubeUv.y, -1));
    if (FACE == 0) {
        rayDir = normalize(vec3(1, -cubeUv.y, -cubeUv.x));
    } else if (FACE == 1) {
        rayDir = normalize(vec3(-1, -cubeUv.y, cubeUv.x));
    } else if (FACE == 2) {
        rayDir = normalize(vec3(cubeUv.x, 1, cubeUv.y));
    } else if (FACE == 3) {
        rayDir = normalize(vec3(cubeUv.x, -1, -cubeUv.y));
    } else if (FACE == 4) {
        rayDir = normalize(vec3(cubeUv.x, -cubeUv.y, 1));
    }
    float blend = smoothstep(-0.05, 0.02, rayDir.y);
    float minViewCos = atmosphereMinViewCos();
    rayDir.y = max(rayDir.y, minViewCos);
    rayDir = normalize(mix(vec3(rayDir.x, minViewCos, rayDir.z), rayDir, blend));
    float cameraHeight = worldUBO.cameraViewMatInv[3].y;
    vec3 rayOrigin = vec3(0.0, atmosphereRg() + cameraHeight + 70.0, 0.0);

    vec3 scatteredRadiance = integrateSingleScattering(rayOrigin, rayDir, false) + integrateSingleScattering(rayOrigin, rayDir, true);
    if (!finiteVec3(scatteredRadiance) ||
        (skyLuminance(scatteredRadiance) <= 1e-5 && celestialSunDirection().y > -0.08)) {
        scatteredRadiance = fallbackSkyRadiance(rayDir);
    }
    outColor = vec4(scatteredRadiance, 1.0);
}
