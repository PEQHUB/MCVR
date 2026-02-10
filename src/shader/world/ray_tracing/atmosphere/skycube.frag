#version 460
#extension GL_GOOGLE_include_directive : require

#include "common/shared.hpp"

layout(set = 1, binding = 0) uniform WorldUniform {
    WorldUBO worldUBO;
};

layout(set = 1, binding = 1) uniform SkyUniform {
    SkyUBO skyUBO;
};

layout(set = 0, binding = 0) uniform sampler2D transLUT;

layout(push_constant) uniform Push {
    int face; // 0..5
}
pc;

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

float phaseRayleigh(float cosTheta) {
    return 3.0 / (16.0 * PI) * (1.0 + cosTheta * cosTheta);
}

float phaseMieHG(float cosTheta, float g) {
    cosTheta = clamp(cosTheta, -1.0, 1.0);
    g = clamp(g, -0.999, 0.999);
    float g2 = g * g;
    float d = 1.0 + g2 - 2.0 * g * cosTheta;
    float denom = pow(d + 1e-6, 1.5);
    return (1.0 - g2) / (4.0 * PI * denom);
}

vec2 transmittanceUv(float r, float mu, SkyUBO ubo) {
    float u = clamp(mu * 0.5 + 0.5, 0.0, 1.0);
    float v = clamp((r - ubo.Rg) / (ubo.Rt - ubo.Rg), 0.0, 1.0);
    return vec2(u, v);
}

float densityExp(float h, float H) {
    return exp(-max(h, 0.0) / H);
}

vec3 sampleTransmittance(float r, float mu) {
    vec2 uv = transmittanceUv(r, mu, skyUBO);

    vec2 invSize = 1.0 / vec2(textureSize(transLUT, 0));
    uv = clamp(uv, 0.5 * invSize, 1.0 - 0.5 * invSize);

    return texture(transLUT, uv).rgb;
}

vec3 integrateSingleScattering(vec3 ro, vec3 rd, bool isSun) {
    float tAtm0, tAtm1;
    if (!intersectSphere(ro, rd, skyUBO.Rt, tAtm0, tAtm1)) return vec3(0.0);
    tAtm0 = max(tAtm0, 0.0);

    // stop at ground
    float tG0, tG1;
    if (intersectSphere(ro, rd, skyUBO.Rg, tG0, tG1)) {
        float tHitG = (tG0 > 0.0) ? tG0 : tG1;
        if (tHitG > 0.0) tAtm1 = min(tAtm1, tHitG);
    }

    const int STEPS = 32;
    float dt = (tAtm1 - tAtm0) / float(STEPS);

    vec3 L = vec3(0.0);
    vec3 T_view = vec3(1.0);

    float cosTheta = dot(normalize(isSun ? skyUBO.sunDirection : -skyUBO.sunDirection), rd);
    float pr = phaseRayleigh(cosTheta);
    float pm = phaseMieHG(cosTheta, skyUBO.mieG);

    for (int i = 0; i < STEPS; i++) {
        float t = tAtm0 + (float(i) + 0.5) * dt;
        vec3 x = ro + rd * t;

        float r = length(x);
        float h = r - skyUBO.Rg;

        float dR = densityExp(h, skyUBO.Hr);
        float dM = densityExp(h, skyUBO.Hm);

        vec3 sigmaS_R = skyUBO.betaR * dR;
        vec3 sigmaS_M = skyUBO.betaM * dM;
        vec3 sigmaT = sigmaS_R + sigmaS_M;

        vec3 up = x / r;
        float muS = dot(up, normalize(isSun ? skyUBO.sunDirection : -skyUBO.sunDirection));
        vec3 T_sun = sampleTransmittance(r, muS);

        vec3 S = sigmaS_R * pr + (sigmaS_M * (pm));
        vec3 dL = T_view * (T_sun * (S * (isSun ? skyUBO.sunRadiance : skyUBO.moonRadiance))) * dt;

        L += dL;
        T_view *= exp(-sigmaT * dt);
    }

    return L;
}

vec3 cubemapDir(int face, vec2 uv01) {
    vec2 uv = uv01 * 2.0 - 1.0;
    if (face == 0) return normalize(vec3(1, -uv.y, -uv.x)); // +X
    if (face == 1) return normalize(vec3(-1, -uv.y, uv.x)); // -X
    if (face == 2) return normalize(vec3(uv.x, 1, uv.y));   // +Y
    if (face == 3) return normalize(vec3(uv.x, -1, -uv.y)); // -Y
    if (face == 4) return normalize(vec3(uv.x, -uv.y, 1));  // +Z
    return normalize(vec3(-uv.x, -uv.y, -1));               // -Z
}

void main() {
    vec3 rd = cubemapDir(pc.face, texCoord);
    float blend = smoothstep(-0.05, 0.02, rd.y);
    rd.y = max(rd.y, skyUBO.minViewCos);
    rd = normalize(mix(vec3(rd.x, skyUBO.minViewCos, rd.z), rd, blend));
    float cameraHeight = worldUBO.cameraViewMatInv[3].y;
    vec3 ro = vec3(0.0, skyUBO.Rg + cameraHeight + 70, 0.0); // camera height mapped to radius (+70 for negative y)

    vec3 L = integrateSingleScattering(ro, rd, false) + integrateSingleScattering(ro, rd, true);
    outColor = vec4(L, 1.0);
}
