#version 460
#extension GL_GOOGLE_include_directive : require

#include "common/shared.hpp"

layout(set = 1, binding = 1) uniform SkyUniform {
    SkyUBO skyUBO;
};

layout(location = 0) in vec2 texCoord;

layout(location = 0) out vec4 outColor;

bool intersectSphere(vec3 ro, vec3 rd, float R, out float tNear, out float tFar) {
    float b = dot(ro, rd);
    float c = dot(ro, ro) - R * R;
    float h = b * b - c;
    if (h < 0.0) return false;
    h = sqrt(h);
    tNear = -b - h;
    tFar = -b + h;
    return true;
}

float densityExp(float h, float H) {
    return exp(-max(h, 0.0) / H);
}

void main() {
    float mu = texCoord.x * 2.0 - 1.0;               // [-1,1]
    float r = mix(skyUBO.Rg, skyUBO.Rt, texCoord.y); // [Rg,Rt]

    vec3 ro = vec3(0.0, r, 0.0);
    float sinTheta = sqrt(max(1.0 - mu * mu, 0.0));
    vec3 rd = normalize(vec3(sinTheta, mu, 0.0));

    float t0, t1;
    if (!intersectSphere(ro, rd, skyUBO.Rt, t0, t1)) {
        outColor = vec4(1.0);
        return;
    }
    t0 = max(t0, 0.0);

    const int STEPS = 128;
    float dt = (t1 - t0) / float(STEPS);
    vec3 opticalDepth = vec3(0.0);

    for (int i = 0; i < STEPS; i++) {
        float t = t0 + (float(i) + 0.5) * dt;
        vec3 x = ro + rd * t;
        float rr = length(x);
        float h = rr - skyUBO.Rg;

        float dR = densityExp(h, skyUBO.Hr);
        float dM = densityExp(h, skyUBO.Hm);

        vec3 sigmaT = skyUBO.betaR * dR + skyUBO.betaM * dM;
        opticalDepth += sigmaT * dt;
    }

    vec3 T = exp(-opticalDepth);
    outColor = vec4(T, 1.0);
}
