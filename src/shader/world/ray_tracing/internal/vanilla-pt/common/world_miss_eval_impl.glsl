#include "common/shared.hpp"
#include "util/ray.glsl"
#include "util/util.glsl"

layout(set = 0, binding = 0) uniform sampler2D textures[];
layout(set = 5, binding = 0) uniform sampler2D transLUT;
layout(set = 5, binding = 2) uniform samplerCube skyFull;

layout(set = 2, binding = 0) uniform WorldUniform {
    WorldUBO worldUBO;
};

layout(set = 2, binding = 1) uniform LastWorldUniform {
    WorldUBO lastWorldUbo;
};

layout(set = 2, binding = 2) uniform SkyUniform {
    SkyUBO skyUBO;
};

layout(location = 0) rayPayloadInEXT MainRay mainRay;

#include "common/volumetric_cloud.glsl"

bool missIntersectSphere(vec3 rayOrigin, vec3 rayDir, float radius, out float tNear, out float tFar) {
    float b = dot(rayOrigin, rayDir);
    float c = dot(rayOrigin, rayOrigin) - radius * radius;
    float h = b * b - c;
    if (h < 0.0) return false;
    h = sqrt(h);
    tNear = -b - h;
    tFar = -b + h;
    return true;
}

void makeBasis(in vec3 n, out vec3 t, out vec3 b) {
    float s = (n.z >= 0.0) ? 1.0 : -1.0;
    float a = -1.0 / (s + n.z);
    float k = n.x * n.y * a;
    t = vec3(1.0 + s * n.x * n.x * a, s * k, -s * n.x);
    b = vec3(k, s + n.y * n.y * a, -n.y);
    t = normalize(t);
    b = normalize(b);
}

vec4 sampleTextureLod0(sampler2D tex, vec2 uv) {
    ivec2 texSize = textureSize(tex, 0);
    if (texSize.x <= 0 || texSize.y <= 0) return vec4(0.0);
    vec2 halfTexel = 0.5 / vec2(texSize);
    vec2 clampedUv = clamp(uv, halfTexel, vec2(1.0) - halfTexel);
    return sampleTexture(tex, clampedUv, 0.0, false);
}

vec4 sampleAtlasLod0(sampler2D tex, vec2 uv01, uvec2 tileCount, uvec2 tile) {
    ivec2 texSize = textureSize(tex, 0);
    if (texSize.x <= 0 || texSize.y <= 0) return vec4(0.0);

    vec2 invTileCount = 1.0 / vec2(tileCount);
    vec2 tileMin = vec2(tile) * invTileCount;
    vec2 tileMax = tileMin + invTileCount;
    vec2 halfTexel = 0.5 / vec2(texSize);
    vec2 minUv = tileMin + halfTexel;
    vec2 maxUv = tileMax - halfTexel;
    vec2 atlasUv = mix(minUv, maxUv, clamp(uv01, 0.0, 1.0));
    return sampleTexture(tex, atlasUv, 0.0, false);
}

vec4 evalSunBillboard(vec3 rayDir) {
    vec3 sunDir = celestialSunDirection();
    rayDir = normalize(rayDir);
    float z = dot(rayDir, sunDir);
    if (z <= 0.0) return vec4(0.0);

    vec3 right, up;
    makeBasis(sunDir, right, up);
    vec2 p = vec2(dot(rayDir, right), dot(rayDir, up));
    vec2 q = p / max(z, 1e-4);
    float tanHalf = tan(0.03);
    vec2 a = abs(q);
    if (a.x > tanHalf || a.y > tanHalf) return vec4(0.0);
    vec2 uv = q / tanHalf * 0.5 + 0.5;
    return sampleTextureLod0(textures[nonuniformEXT(skyUBO.sunTextureID)], uv);
}

vec4 evalMoonBillboard(vec3 rayDir) {
    vec3 moonDir = celestialMoonDirection();
    rayDir = normalize(rayDir);
    float z = dot(rayDir, moonDir);
    if (z <= 0.0) return vec4(0.0);

    vec3 right, up;
    makeBasis(moonDir, right, up);
    vec2 p = vec2(dot(rayDir, right), dot(rayDir, up));
    vec2 q = p / max(z, 1e-4);
    float tanHalf = tan(0.05);
    vec2 a = abs(q);
    if (a.x > tanHalf || a.y > tanHalf) return vec4(0.0);
    vec2 uv = q / tanHalf * 0.5 + 0.5;
    uvec2 tileCount = uvec2(4u, 2u);
    uvec2 tile = uvec2(skyUBO.moonPhase % tileCount.x, (skyUBO.moonPhase / tileCount.x) % tileCount.y);
    return sampleAtlasLod0(textures[nonuniformEXT(skyUBO.moonTextureID)], uv, tileCount, tile);
}

void main() {
    mainRay.directLightRadiance.x = 1.0;

    if (skyUBO.cameraSubmersionType == 0 || skyUBO.cameraSubmersionType == 2 || skyUBO.hasBlindnessOrDarkness > 0) {
        raySetStop(mainRay, true);
        mainRay.hitT = INF_DISTANCE;
        return;
    }

    switch (worldUBO.skyType) {
        case 0:
        case 2:
            raySetStop(mainRay, true);
            mainRay.hitT = INF_DISTANCE;
            return;
        case 1:
        default: break;
    }

    vec3 rayDir = normalize(gl_WorldRayDirectionEXT);
    vec3 sunDir = celestialSunDirection();
    float progress = skyUBO.rainGradient;
    vec3 rainyRadiance = mix(vec3(0.0), vec3(0.1), smoothstep(-0.3, 0.3, sunDir.y));
    vec3 backgroundRadiance = mix(texture(skyFull, rayDir).rgb, rainyRadiance, progress);

    if (worldUBO.skyType == 1) {
        float cameraHeight = worldUBO.cameraViewMatInv[3].y;
        vec3 pPlanet = vec3(0.0, VPT_ATMOSPHERE_RG + cameraHeight + 70.0, 0.0);
        float r = clamp(length(pPlanet), VPT_ATMOSPHERE_RG, VPT_ATMOSPHERE_RT);
        vec3 up = pPlanet / max(r, 1e-6);
        float mu = clamp(dot(up, rayDir), -1.0, 1.0);
        vec3 transmittance = sampleCloudAtmosphereTransmittance(r, mu);

        vec4 sunSample = evalSunBillboard(rayDir);
        if (sunSample.a > 1e-4) {
            float tG0, tG1;
            bool hitGround = missIntersectSphere(pPlanet, rayDir, VPT_ATMOSPHERE_RG, tG0, tG1);
            bool blocked = hitGround && (tG1 > 1e-3);
            if (!blocked) {
                vec3 sunRadiance = sunSample.rgb * VPT_SUN_RADIANCE * transmittance * sunSample.a;
                backgroundRadiance += mix(sunRadiance, vec3(0.0), progress);
            }
        }

        vec4 moonSample = evalMoonBillboard(rayDir);
        if (moonSample.a > 1e-4) {
            float tG0, tG1;
            bool hitGround = missIntersectSphere(pPlanet, rayDir, VPT_ATMOSPHERE_RG, tG0, tG1);
            bool blocked = hitGround && (tG1 > 1e-3);
            if (!blocked) {
                vec3 moonRadiance = moonSample.rgb * VPT_MOON_RADIANCE * max(transmittance, vec3(0.03));
                backgroundRadiance += mix(moonRadiance, vec3(0.0), progress);
            }
        }
    }

#if VPT_ALLOW_VOLUMETRIC_CLOUD_MISS
    if (VPT_CLOUD_MODE == 2u) {
        VolumetricCloudResult cloudResult =
            rayUseIndirectVolumetricCloud(mainRay) ?
                applyVolumetricCloudBudgeted(gl_WorldRayOriginEXT, rayDir, backgroundRadiance,
                                             VPT_INDIRECT_VOLUMETRIC_CLOUD_VIEW_STEPS,
                                             VPT_INDIRECT_VOLUMETRIC_CLOUD_LIGHT_STEPS,
                                             VPT_INDIRECT_VOLUMETRIC_CLOUD_AMBIENT_STEPS) :
                applyVolumetricCloud(gl_WorldRayOriginEXT, rayDir, backgroundRadiance);
        backgroundRadiance = cloudResult.color;
        vec3 clampedBackground = max(backgroundRadiance, vec3(0.0));
        vec3 cloudOnly = max(cloudResult.color - clampedBackground * cloudResult.transmittance, vec3(0.0));
        float backgroundLum = volumetricCloudLuminance(clampedBackground);
        float cloudLum = volumetricCloudLuminance(cloudOnly);
        float cloudPresence = cloudLum / max(backgroundLum + cloudLum, 1e-4);
        float starVisibility = min(cloudResult.transmittance, 1.0 - cloudPresence * 1.35);
        mainRay.directLightRadiance.x = cloudSaturate(starVisibility);
    }
#endif

    mainRay.radiance += backgroundRadiance * mainRay.throughput;
    raySetStop(mainRay, true);
    mainRay.hitT = INF_DISTANCE;
}
