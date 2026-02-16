#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_nonuniform_qualifier : require

#include "../util/ray_payloads.glsl"
#include "../util/util.glsl"
#include "common/shared.hpp"

layout(set = 0, binding = 0) uniform sampler2D textures[];
layout(set = 0, binding = 1) uniform sampler2D transLUT;
layout(set = 0, binding = 2) uniform samplerCube skyFull;

layout(set = 2, binding = 0) uniform WorldUniform {
    WorldUBO worldUBO;
};

layout(set = 2, binding = 1) uniform LastWorldUniform {
    WorldUBO lastWorldUbo;
};

layout(set = 2, binding = 2) uniform SkyUniform {
    SkyUBO skyUBO;
};

layout(location = 0) rayPayloadInEXT PrimaryRay mainRay;

vec2 transmittanceUv(float r, float mu, SkyUBO ubo) {
    float u = clamp(mu * 0.5 + 0.5, 0.0, 1.0);
    float v = clamp((r - ubo.Rg) / (ubo.Rt - ubo.Rg), 0.0, 1.0);
    return vec2(u, v);
}

vec3 sampleTransmittance(float r, float mu) {
    vec2 uv = transmittanceUv(r, mu, skyUBO);
    return texture(transLUT, uv).rgb;
}

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

void makeBasis(in vec3 n, out vec3 t, out vec3 b) {
    // Frisvad 2012, Building an Orthonormal Basis, Revisited
    float s = (n.z >= 0.0) ? 1.0 : -1.0;
    float a = -1.0 / (s + n.z);
    float k = n.x * n.y * a;
    t = vec3(1.0 + s * n.x * n.x * a, s * k, -s * n.x);
    b = vec3(k, s + n.y * n.y * a, -n.y);
    t = normalize(t);
    b = normalize(b);
}

vec4 evalSunBillboard(vec3 rd) {
    vec3 sunDir = normalize(skyUBO.sunDirection);
    rd = normalize(rd);

    float z = dot(rd, sunDir);
    if (z <= 0.0) return vec4(0.0);

    vec3 right, up;
    makeBasis(sunDir, right, up);

    vec2 p = vec2(dot(rd, right), dot(rd, up));
    vec2 q = p / max(z, 1e-4);

    float tanHalf = tan(0.03 * max(0.05, skyUBO.envCelestial.x)); // tan(x), x: half angle from middle to edge

    vec2 a = abs(q);
    if (a.x > tanHalf || a.y > tanHalf) return vec4(0.0);

    return vec4(1.0);
}

vec4 evalMoonBillboard(vec3 rd) {
    vec3 moonDir = normalize(-skyUBO.sunDirection);
    rd = normalize(rd);

    float z = dot(rd, moonDir);
    if (z <= 0.0) return vec4(0.0);

    vec3 right, up;
    makeBasis(moonDir, right, up);

    vec2 p = vec2(dot(rd, right), dot(rd, up));
    vec2 q = p / max(z, 1e-4);

    float tanHalf = tan(0.05 * max(0.05, skyUBO.envCelestial.y)); // tan(x), x: half angle from middle to edge

    vec2 a = abs(q);
    if (a.x > tanHalf || a.y > tanHalf) return vec4(0.0);

    vec2 uv = q / tanHalf * 0.5 + 0.5;
    uint col = skyUBO.moonPhase % 4;
    uint row = skyUBO.moonPhase / 4 % 2;
    uv = vec2((col + uv.x) / 4, (row + uv.y) / 2);
    return texture(textures[nonuniformEXT(skyUBO.moonTextureID)], uv);
}

void main() {
    vec3 rayDir = normalize(mainRay.direction);

    if (skyUBO.cameraSubmersionType == 0 /*LAVA*/ || skyUBO.cameraSubmersionType == 2 /*POWDER_SNOW*/ ||
        skyUBO.hasBlindnessOrDarkness > 0) {
        mainRay.stop = 1;
        mainRay.hitT = INF_DISTANCE;
    } else {
        switch (skyUBO.skyType) {
            case 0: // NONE
                mainRay.stop = 1;
                mainRay.hitT = INF_DISTANCE;
                return;
            case 2: // END
                mainRay.stop = 1;
                mainRay.hitT = INF_DISTANCE;
                return;
            case 1: // NORMAL
            default: break;
        }

        vec3 rd = normalize(gl_WorldRayDirectionEXT);
        vec3 sunDir = normalize(skyUBO.sunDirection);
        vec3 moonDir = normalize(-skyUBO.sunDirection);

        // Fade sky brightness to black after sunset.
        // sunDir.y is continuous and avoids hard transitions.
        // 1.0 above ~+1 deg, 0.0 below ~-6 deg.
        float daySky = smoothstep(-0.10, 0.02, sunDir.y);

        float progress = clamp(skyUBO.rainGradient * skyUBO.envSky.y, 0.0, 1.0);
        vec3 rainyRadiance = mix(vec3(0.0), vec3(0.1), smoothstep(-0.3, 0.3, sunDir.y));
        vec3 sunnyRadiance = texture(skyFull, rayDir).rgb * skyUBO.envSky.x;
        vec3 skyRadiance = mix(sunnyRadiance, rainyRadiance, progress) * daySky;
        mainRay.radiance += skyRadiance * mainRay.throughput;

        if (worldUBO.skyType == 1) {
            {
                vec4 sunSample = evalSunBillboard(rd);
                if (sunSample.a > 0.0) {
                    vec3 C = vec3(0.0, -skyUBO.Rg, 0.0);
                    vec3 pWorld = gl_WorldRayOriginEXT;
                    vec3 pPlanet = pWorld - C;

                    float tG0, tG1;
                    bool hitGround = intersectSphere(pPlanet, rd, skyUBO.Rg, tG0, tG1);
                    bool blocked = hitGround && (tG1 > 0.0);
                    if (!blocked) {
                        float r = length(pPlanet);
                        vec3 up = pPlanet / max(r, 1e-6);
                        float mu = clamp(dot(up, sunDir), -1.0, 1.0);
                        r = clamp(r, skyUBO.Rg, skyUBO.Rt);
                        vec3 T = sampleTransmittance(r, mu);
                        vec3 sunRadiance = (sunSample.rgb * skyUBO.sunRadiance * skyUBO.envCelestial.z * T * sunSample.a) * daySky;
                        mainRay.radiance += mix(sunRadiance, vec3(0.0), progress) * mainRay.throughput;
                    }
                }
            }

            {
                vec4 moonSample = evalMoonBillboard(rd);
                // Disable fake blue night ambient light for pitch black nights
                vec3 nightCompensite = vec3(0.0);

                if (moonSample.a > 0.0) {
                    vec3 C = vec3(0.0, -skyUBO.Rg, 0.0);
                    vec3 pWorld = gl_WorldRayOriginEXT;
                    vec3 pPlanet = pWorld - C;

                    float tG0, tG1;
                    bool hitGround = intersectSphere(pPlanet, rd, skyUBO.Rg, tG0, tG1);
                    bool blocked = hitGround && (tG1 > 0.0);
                    if (!blocked) {
                        float r = length(pPlanet);
                        vec3 up = pPlanet / max(r, 1e-6);
                        float mu = clamp(dot(up, moonDir), -1.0, 1.0);
                        r = clamp(r, skyUBO.Rg, skyUBO.Rt);
                        vec3 T = sampleTransmittance(r, mu);
                        // Scale down moon radiance by 95% for subtle physical moon
                        vec3 moonRadiance = (moonSample.rgb * skyUBO.moonRadiance * skyUBO.envCelestial.w * T * moonSample.a) * 0.05;
                        mainRay.radiance += mix(moonRadiance, vec3(nightCompensite), progress) * mainRay.throughput;
                    }
                } else {
                    mainRay.radiance += nightCompensite * mainRay.throughput;
                }
            }
        }

        mainRay.stop = 1;
        mainRay.hitT = INF_DISTANCE;
    }
}
