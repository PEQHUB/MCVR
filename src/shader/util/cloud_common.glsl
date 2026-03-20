#ifndef CLOUD_COMMON_GLSL
#define CLOUD_COMMON_GLSL

// Shared declarations for all cloud compute shaders.
// UBO layouts must match shared.hpp exactly (field order, types, padding).
//
// References:
//   [Schneider15] A. Schneider, "The Real-time Volumetric Cloudscapes of
//                 Horizon: Zero Dawn", SIGGRAPH 2015
//   [Hillaire16]  S. Hillaire, "Physically Based Sky, Atmosphere and Cloud
//                 Rendering in Frostbite", SIGGRAPH 2016
//   [Wrenninge13] M. Wrenninge et al., "Art-directable Multiple Volumetric
//                 Scattering", SIGGRAPH 2013

// WorldUBO: only declare fields up to what cloud shaders need.
// The actual WorldUBO is much larger (material data, dvec4 cameraPos, etc.)
// but uniform blocks can be declared shorter than the actual buffer.
layout(set = 1, binding = 0) uniform WorldUBO {
    mat4 cameraViewMat;
    mat4 cameraEffectedViewMat;
    mat4 cameraProjMat;
    mat4 cameraViewMatInv;
    mat4 cameraEffectedViewMatInv;
    mat4 cameraProjMatInv;
    vec2 cameraJitter;
    float gameTime;
    uint seed;
    mat4 textureMat;
    uint overlayTextureID;
    uint isFirstPerson;
    float fogStart;
    float fogEnd;
    vec4 fogColor;
    uint fogType;
    uint wSkyType;
    uint rayBounces;
    float pad3;
} worldUBO;

// SkyUBO: full layout matching shared.hpp SkyUBO struct.
layout(set = 1, binding = 1) uniform SkyUniform {
    vec3 baseColor;
    uint skyType;

    vec4 horizonColor;

    vec3 sunDirection;
    uint isSunRisingOrSetting;

    vec3 moonDirection;
    float moonDirPad;

    uint isSkyDark;
    uint hasBlindnessOrDarkness;
    uint cameraSubmersionType;
    uint moonPhase;

    float rainGradient;
    float hdrRadianceScale;
    float thunderGradient;
    float wetSurfaceStrength;

    // AtmosphereParams
    float Rg;
    float Rt;
    float Hr;
    float Hm;

    vec3 betaR;
    float mieG;

    vec3 betaM;
    float minViewCos;

    vec3 sunRadiance;
    uint sunTextureID;

    vec3 moonRadiance;
    uint moonTextureID;

    vec4 envCelestial;
    vec4 envWaterTintFog;
    vec4 envSky;

    // Cloud-specific fields
    vec4 envCloud;       // x: base height, y: thickness, z: density scale, w: brightness
    ivec4 cloudTile;     // x: tex index, y/z: center cell, w: reserved
    vec4 cloudWrap;      // x/y: intra-cell offset, z: ticks, w: reserved
    vec4 cloudShape;     // x: puffiness, y: detailScale, z: detailStrength, w: anisotropy
    vec4 cloudLighting;  // x: shadowStr, y: ambientStr, z: sunOcclStr, w: noiseAffectsShadows
} skyUBO;

// Previous frame camera matrices (same layout as WorldUBO, only matrices needed).
layout(set = 1, binding = 2) uniform LastWorldUBO {
    mat4 prevCameraViewMat;
    mat4 prevCameraEffectedViewMat;
    mat4 prevCameraProjMat;
    mat4 prevCameraViewMatInv;
    mat4 prevCameraEffectedViewMatInv;
    mat4 prevCameraProjMatInv;
} lastWorldUBO;

// Push constant layout — must match CloudPushConstant in cloud_module.hpp exactly.
// ALL cloud shaders must include this file for push constants; never re-declare inline.
layout(push_constant) uniform PushConstants {
    uint renderWidth;
    uint renderHeight;
    uint cloudWidth;
    uint cloudHeight;
    float cloudBase;
    float cloudThickness;
    float coverage;
    float cloudType;
    float densityMultiplier;
    float windSpeed;
    float windTime;          // Wrapped at 86400s (24h) to preserve FP32 precision
    uint frameIndex;
    uint marchSteps;
    uint lightSteps;
    float temporalBlend;
    uint shadowMapSize;
    float eyePosX;
    float eyePosY;
    float eyePosZ;
    float detailStrength;
    uint scatterOctaves;
    float powderStrength;    // Beer-powder dark edge intensity [0-2]
    float ambientStrength;   // Density-based ambient occlusion [0-2]
    float pad0;
} pc;

// --- Shared constants ---

// World-space to noise UV scale. The 128^3 noise texture tiles via repeat
// sampler; this controls the world-space period. 1/256 = 256-block period,
// chosen for Minecraft's typical 16-32 chunk render distance to minimize
// visible tiling while keeping adequate detail within the cloud layer.
const float NOISE_SCALE = 1.0 / 256.0;

// Wind displacement applied to noise sampling position.
// X-drift is 2x Z-drift for prevailing-wind asymmetry. [Schneider15 §3.3]
// Tuned for Minecraft scale: ~1 block/sec at windSpeed=1.0.
vec3 windOffset(float windTime) {
    return vec3(windTime * 0.004, 0.0, windTime * 0.002);
}

// --- Shared helpers ---

// Camera world position from push constants (avoids dvec4 precision in UBO)
vec3 getEyePos() {
    return vec3(pc.eyePosX, pc.eyePosY, pc.eyePosZ);
}

// Thunder-boosted cloud thickness. Cumulonimbus towers grow 50% taller during
// thunderstorms, clamped so cloud top doesn't exceed Minecraft ceiling (Y=320).
// MUST be used everywhere that needs effective layer thickness — raymarch,
// temporal reprojection, shadow, composite depth.
float effectiveThickness() {
    float boosted = pc.cloudThickness * mix(1.0, 1.5, skyUBO.thunderGradient);
    return min(boosted, 320.0 - pc.cloudBase);
}

// Sample weather map centered on camera. Coverage area is 1024 blocks.
// uWeatherMap must be declared by the including shader.
#ifdef CLOUD_HAS_WEATHER_SAMPLER
vec4 sampleWeather(sampler2D weatherMap, vec3 worldPos) {
    vec3 eyePos = getEyePos();
    vec2 uv = (worldPos.xz - eyePos.xz) / 1024.0 + 0.5;

    // Fade coverage to zero near weather map edges — prevents solid cloud walls
    // at the 1024-block boundary where CLAMP_TO_EDGE repeats edge texels.
    vec2 edgeDist = min(uv, 1.0 - uv);  // Distance from nearest edge [0, 0.5]
    float edgeFade = smoothstep(0.0, 0.1, min(edgeDist.x, edgeDist.y));

    vec4 weather = texture(weatherMap, uv);
    weather.r *= edgeFade;  // Fade coverage only, preserve cloud type
    return weather;
}
#endif

#endif // CLOUD_COMMON_GLSL
