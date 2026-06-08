#version 460
#extension GL_GOOGLE_include_directive : require

#include "common/shared.hpp"

layout(set = 2, binding = 2) uniform SkyUniform {
    SkyUBO skyUBO;
};

layout(location = 0) in vec2 texCoord;

layout(location = 0) out vec4 outColor;

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

vec3 atmosphereBetaR() {
    return dot(skyUBO.betaR, skyUBO.betaR) > 0.0 ? skyUBO.betaR : VPT_ATMOSPHERE_BETA_R;
}

vec3 atmosphereBetaM() {
    return dot(skyUBO.betaM, skyUBO.betaM) > 0.0 ? skyUBO.betaM : VPT_ATMOSPHERE_BETA_M;
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

float densityExp(float height, float scaleHeight) {
    return exp(-max(height, 0.0) / max(scaleHeight, 1.0));
}

void main() {
    float mu = texCoord.x * 2.0 - 1.0;
    float rg = atmosphereRg();
    float rt = atmosphereRt();
    float r = mix(rg, rt, texCoord.y);

    vec3 rayOrigin = vec3(0.0, r, 0.0);
    float sinTheta = sqrt(max(1.0 - mu * mu, 0.0));
    vec3 rayDir = normalize(vec3(sinTheta, mu, 0.0));

    float t0, t1;
    if (!intersectSphere(rayOrigin, rayDir, rt, t0, t1)) {
        outColor = vec4(1.0);
        return;
    }
    t0 = max(t0, 0.0);

    const int steps = 128;
    float dt = (t1 - t0) / float(steps);
    vec3 opticalDepth = vec3(0.0);

    for (int i = 0; i < steps; i++) {
        float t = t0 + (float(i) + 0.5) * dt;
        vec3 x = rayOrigin + rayDir * t;
        float rr = length(x);
        float height = rr - rg;

        float dR = densityExp(height, atmosphereHr());
        float dM = densityExp(height, atmosphereHm());

        vec3 sigmaT = atmosphereBetaR() * dR + atmosphereBetaM() * dM;
        opticalDepth += sigmaT * dt;
    }

    outColor = vec4(exp(-opticalDepth), 1.0);
}
