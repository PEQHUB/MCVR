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
    float ambientStrength;   // Density-based ambient occlusion [0-2]
    float noiseScale;        // Noise texture period in blocks [128-512]
    float cellFrequency;     // Weather noise frequency scale [1-32]
    float atmosphereFadeDist; // Cloud atmospheric fade distance in blocks [200-2000]
    float windAngle;         // Wind direction in radians [0, 2π]
    uint debugMode;          // 0=normal, 1=weather cov, 2=weather type, 3-5=noise R/G/A, 6=hProfile, 7=raw density, 8=final density
} pc;

// --- Shared constants ---

// World-space to noise UV scale. The 128^3 noise texture tiles via repeat
// sampler; this controls the world-space period. Configurable via push constant
// noiseScale (period in blocks). Default 192 for more vertical detail in
// Minecraft's typical 40-block cloud layer.
#define NOISE_SCALE (1.0 / max(pc.noiseScale, 1.0))

// Wind displacement with height-dependent shear [Frostnova, Schneider15 §3.3].
// Higher cloud layers drift faster and shift noise sampling position.
vec3 windOffset(float windTime, float heightFraction) {
    float s = sin(pc.windAngle);
    float c = cos(pc.windAngle);
    vec3 baseWind = vec3(c, 0.0, s);
    vec3 shearWind = baseWind + heightFraction * vec3(0.0, 0.1, 0.0);
    return 0.004 * shearWind * (windTime + heightFraction * 20.0);
}

// Zero-arg overload for contexts without height (weather map, noise gen, debug viz).
vec3 windOffset(float windTime) {
    return windOffset(windTime, 0.0);
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

// Fixed weather map coverage radius in blocks.
// Must match cloud_weather.comp. Independent of atmosphereFadeDist so the
// fade slider doesn't rescale/shift the cloud pattern.
// 4000 blocks = covers full MARCH_CLAMP range from any camera angle.
#define WEATHER_EXTENT 4000.0

// Sample weather map centered on camera.
// Uses fixed extent so cloud placement is independent of fade distance.
// uWeatherMap must be declared by the including shader.
// Snap weather origin to texel grid — prevents sub-texel drift between frames
// that causes cloud edge shimmer through RGBA8 quantization + smoothstep amplification.
vec2 getSnappedEyeXZ() {
    vec3 eyePos = getEyePos();
    float texelSize = WEATHER_EXTENT / 512.0;
    return floor(eyePos.xz / texelSize) * texelSize;
}

#ifdef CLOUD_HAS_WEATHER_SAMPLER
vec4 sampleWeather(sampler2D weatherMap, vec3 worldPos) {
    // Use snapped origin — must match cloud_weather.comp
    vec2 snappedEye = getSnappedEyeXZ();
    vec2 uv = (worldPos.xz - snappedEye) / WEATHER_EXTENT + 0.5;
    // Edge fade is baked into the weather map R channel — not applied here
    return texture(weatherMap, uv);
}
#endif

#endif // CLOUD_COMMON_GLSL
