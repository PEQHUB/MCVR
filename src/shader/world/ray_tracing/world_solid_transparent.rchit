#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_opacity_micromap : require

#include "../util/disney.glsl"
#include "../util/random.glsl"
#include "../util/blue_noise.glsl"
#include "../util/ray_cone.glsl"
#include "../util/ray_payloads.glsl"
#include "../util/util.glsl"
#include "common/shared.hpp"
#include "../util/area_light.glsl"
#include "../util/restir.glsl"
#include "../util/ray_offset.glsl"
#include "../util/colorspace.glsl"
#include "../util/noise.glsl"

layout(set = 0, binding = 0) uniform sampler2D textures[];

#include "../util/sprite_fetch.glsl"
#include "../util/pom.glsl"

#include "../util/clouds.glsl"

layout(set = 1, binding = 0) uniform accelerationStructureEXT topLevelAS;

layout(set = 1, binding = 1) readonly buffer BLASOffsets {
    uint offsets[];
}
blasOffsets;

layout(set = 1, binding = 2) readonly buffer VertexBufferAddr {
    uint64_t addrs[];
}
vertexBufferAddrs;

layout(set = 1, binding = 3) readonly buffer IndexBufferAddr {
    uint64_t addrs[];
}
indexBufferAddrs;

layout(set = 1, binding = 4) readonly buffer LastVertexBufferAddr {
    uint64_t addrs[];
}
lastVertexBufferAddrs;

layout(set = 1, binding = 5) readonly buffer LastIndexBufferAddr {
    uint64_t addrs[];
}
lastIndexBufferAddrs;

layout(set = 1, binding = 6) readonly buffer LastObjToWorldMat {
    mat4 mat[];
}
lastObjToWorldMats;

layout(set = 1, binding = 7) readonly buffer TextureMappingBuffer {
    TextureMapping mapping;
};

#include "../util/material_properties.glsl"

layout(set = 1, binding = 8) readonly buffer AreaLightBuffer {
    AreaLight lights[];
} areaLightBuffer;

layout(set = 1, binding = 9) readonly buffer TileLightBuffer {
    uint data[];
} tileLightBuffer;

layout(set = 1, binding = 10) readonly buffer BiomeColorBuffer {
    uvec4 data[];  // .x=grass, .y=foliage, .z=water (packed 0x00RRGGBB)
} biomeColors;

#include "../util/vertex_fetch.glsl"

const int TILE_SIZE = 16;
const int MAX_LIGHTS_PER_TILE = 512;

layout(set = 2, binding = 0) uniform WorldUniform {
    WorldUBO worldUbo;
};

layout(set = 2, binding = 1) uniform LastWorldUniform {
    WorldUBO lastWorldUbo;
};

layout(set = 2, binding = 2) uniform SkyUniform {
    SkyUBO skyUBO;
};

// VertexBuffer and IndexBuffer declared in vertex_fetch.glsl

layout(push_constant) uniform PushConstant {
    int numRayBounces;
    int flags;
    int areaLightCount;
    float shadowSoftness;
    int risCandidates;
    int temporalMClamp;
    int wClamp;
    float preExposure;
    // POM fields (fields 8-11, 16 bytes)
    float pomHeightScale;       // 0 = disabled, else depth scale (0.01-0.50)
    int   pomSteps;             // linear search steps (8-512)
    int   pomRefinement;        // binary refinement iterations (0-8)
    float pomFadeDistance;      // distance in blocks to fade POM out (8-256)
    float colorExpansion;       // per-block vivid color chroma boost (0.0-2.0, 1.0=neutral)
    uint blueNoiseFrame;                 // alignment padding (offset 52)
    // SHARC BDA fields (offsets 56-119, 64 bytes)
    uint64_t _sharcBDA0;
    uint64_t _sharcBDA1;
    uint64_t _sharcBDA2;
    float _sharcPad0, _sharcPad1, _sharcPad2, _sharcPad3;
    uint _sharcPad4;
    float _sharcPad5;
    uint _sharcPad6;
    float _sharcPad7;
    int _sharcPad8;
    int _sharcPad9;
    // Offline accumulation fields (offset 120)
    int offlineFlags;
    int accumFrameCount;
    float aperture;
    float focalDistance;
    // Material SSBO BDA (offset 128)
    uint64_t materialClassAddr;
} pc;
#define SIMPLIFIED_INDIRECT ((pc.flags & 1) != 0)
#define AREA_LIGHTS_ON ((pc.flags & 2) != 0)
#define NOISE_LOD_ON   ((pc.flags & 64) != 0)
#define RESTIR_ENABLED ((pc.flags & 4) != 0)
#define RESTIR_SIMPLIFIED_BRDF ((pc.flags & 8) != 0)
#define RESTIR_BOUNCE_ENABLED ((pc.flags & 16) != 0)
#define BEER_LAW_SHADOWS   ((pc.flags & 512) != 0)
#define NO_EMISSION_CLAMP  ((pc.flags & 1024) != 0)
#define PHYSICAL_SUN_DISK  ((pc.flags & 2048) != 0)
#define NO_HAND_AMBIENT    ((pc.flags & 4096) != 0)
#define ENTITY_NORMALS_ON  ((pc.flags & 32768) != 0)

layout(set = 3, binding = 3, rgba32f) uniform readonly image2D normalRoughnessImage;
layout(set = 3, binding = 4, rg32f) uniform readonly image2D motionVectorImage;
layout(set = 3, binding = 5, r32f) uniform readonly image2D linearDepthImage;
layout(set = 3, binding = 14, rgba32f) uniform image2D reservoirCurrentImage;
layout(set = 3, binding = 15, rgba32f) uniform image2D reservoirPreviousImage;
layout(set = 3, binding = 18, rgba32f) uniform image2D bounceReservoirImage1;
layout(set = 3, binding = 19, rgba32f) uniform image2D bounceReservoirImage2;
layout(set = 3, binding = 20, rgba32f) uniform image2D bounceReservoirImage3;

// Bounce reservoir load/store helpers (per-bounce-level image selection)
vec4 loadBounceReservoir(uint bounceIdx, ivec2 p) {
    if (bounceIdx == 1u) return imageLoad(bounceReservoirImage1, p);
    if (bounceIdx == 2u) return imageLoad(bounceReservoirImage2, p);
    return imageLoad(bounceReservoirImage3, p);
}
void storeBounceReservoir(uint bounceIdx, ivec2 p, vec4 d) {
    if (bounceIdx == 1u) imageStore(bounceReservoirImage1, p, d);
    else if (bounceIdx == 2u) imageStore(bounceReservoirImage2, p, d);
    else imageStore(bounceReservoirImage3, p, d);
}

layout(location = 0) rayPayloadInEXT PrimaryRay mainRay;
layout(location = 1) rayPayloadEXT ShadowRay shadowRay;
hitAttributeEXT vec2 attribs;

// Sample height value from texture based on height source mode
float sampleHeightRaw(uint texID, vec2 uv, vec2 uvMin, vec2 uvMax, int heightSourceMode, vec3 channelWeights, bool isBlock) {
    vec2 clampedUV = clamp(uv, uvMin, uvMax);
    vec4 s;
    if (isBlock) {
        // Block geometry: sample from sprite texture array using spriteId
        SpriteEntry se = spriteEntries[texID];
        uint layer = spriteAnimLayer(se, worldUbo.animTick);
        s = textureLod(blockAlbedo, vec3(clampedUV, float(layer)), 0);
    } else {
        s = textureLod(textures[nonuniformEXT(texID)], clampedUV, 0);
    }
    if (heightSourceMode == 1) return s.r;       // Red
    if (heightSourceMode == 2) return s.g;       // Green
    if (heightSourceMode == 3) return s.b;       // Blue
    if (heightSourceMode == 4) return s.a;       // Alpha
    if (heightSourceMode == 5) return max(s.r, max(s.g, s.b)); // MaxRGB
    if (heightSourceMode == 6) return min(s.r, min(s.g, s.b)); // MinRGB
    return dot(s.rgb, channelWeights);            // 0 = Luminance (default), 7 = Custom (same)
}

// Process raw height through the full pipeline: normalize -> invert -> remap -> contrast -> offset
float processHeight(float raw, float lumMin, float lumSpan, bool invertH,
                    float remapMin, float remapMax, float contrast, float offset) {
    float h = clamp((raw - lumMin) / lumSpan, 0.0, 1.0);
    if (invertH) h = 1.0 - h;
    h = mix(remapMin, remapMax, h);
    if (contrast != 1.0) h = pow(clamp(h, 0.001, 1.0), contrast);
    return clamp(h + offset, 0.0, 1.0);
}

// GPU-side AutoPBR: derives roughness from albedo luminance via percentile mapping
// and normals via height field derivatives, using precomputed histogram bounds from SSBO pack 8.
void applyAutoPBR(
    inout LabPBRMat mat,
    MaterialClassEntry mc,
    vec3 rawAlbedoLinear,
    uint texID,
    vec2 texUV,
    vec2 uvMin,
    vec2 uvMax,
    float hitDistance,
    bool isBlock
) {
    float lumMin = mc.lumMin;
    float lumMax = mc.lumMax;
    if (lumMin >= lumMax) return;
    float lumSpan = lumMax - lumMin;

    // Safety: skip if no valid AutoPBR data (packed params all zero)
    uint p0 = mc.autoPBRPacked0;
    if (p0 == 0u) return;

    // Unpack roughness params from Pack 8
    float rMin = float(p0 & 0xFFu) / 100.0;
    float rMax = float((p0 >> 8u) & 0xFFu) / 100.0;
    float centerPct = float((p0 >> 16u) & 0xFFu);
    float spreadPct = float((p0 >> 24u) & 0xFFu);

    float normalStrength = mc.normalStrength;
    uint p1 = mc.autoPBRPacked1;
    float heightGamma = max(float(p1 & 0xFFFFu) / 100.0, 0.01);

    bool invertRoughness = (mc.flags & 0x10u) != 0u;
    bool invertNormal    = (mc.flags & 0x20u) != 0u;
    bool invertHeight    = (mc.flags & 0x40u) != 0u;

    // Unpack Pack 4 fields
    uint pp0 = mc.pomPacked0;
    uint pp1 = mc.pomPacked1;
    uint pp2 = mc.pomPacked2;

    int filterMode       = int(pp0 & 0x7u);
    int heightSourceMode = int((pp0 >> 6u) & 0x7u);
    int filterRadiusRaw  = int((pp0 >> 20u) & 0xFu);
    int mipBiasRaw       = int((pp0 >> 24u) & 0xFu);

    float normalClampVal = float(pp1 & 0xFFu) / 100.0;
    float geometricBlendVal = float((pp1 >> 8u) & 0xFFu) / 100.0;
    float heightContrastVal = max(float((pp1 >> 24u) & 0xFFu) / 10.0, 0.01);

    float heightRemapMinVal = float(pp2 & 0xFFu) / 100.0;
    float heightRemapMaxVal = float((pp2 >> 8u) & 0xFFu) / 100.0;
    float heightOffsetVal = (float((pp2 >> 16u) & 0xFFu) - 100.0) / 100.0;
    float normalDistFade = float((pp2 >> 24u) & 0xFFu);

    float filterRadiusVal = 0.5 + float(filterRadiusRaw) * 0.25;

    vec3 channelWeights = vec3(0.2627, 0.6780, 0.0593); // BT.2020 luminance

    // --- Roughness from luminance percentile mapping ---
    float rawLum = sampleHeightRaw(texID, texUV, uvMin, uvMax, heightSourceMode, channelWeights, isBlock);
    if (rawLum < 0.001) return; // Skip transparent/masked pixels

    float normLum = clamp((rawLum - lumMin) / lumSpan, 0.0, 1.0);
    float windowStart = (centerPct - spreadPct * 0.5) / 100.0;
    float windowSize = max(spreadPct / 100.0, 0.01);
    float t = clamp((normLum - windowStart) / windowSize, 0.0, 1.0);
    float roughness = mix(rMax, rMin, t);
    if (invertRoughness) roughness = rMin + rMax - roughness;
    mat.roughness = max(roughness * roughness, 0.01);

    // --- Normal from height field derivatives ---
    // Apply distance fade to normal strength
    float effectiveNormalStr = normalStrength;
    if (normalDistFade > 0.0) {
        effectiveNormalStr *= 1.0 - smoothstep(normalDistFade * 0.5, normalDistFade, hitDistance);
    }

    if (effectiveNormalStr < 0.001) {
        mat.normal = vec3(0.0, 0.0, 1.0); // flat
        mat.height = processHeight(rawLum, lumMin, lumSpan, invertHeight,
                                   heightRemapMinVal, heightRemapMaxVal, heightContrastVal, heightOffsetVal);
        return;
    }

    vec2 texSize;
    if (isBlock) {
        texSize = vec2(textureSize(blockAlbedo, 0).xy);
    } else {
        texSize = vec2(textureSize(textures[nonuniformEXT(texID)], 0));
    }
    vec2 texelStep = filterRadiusVal / texSize;

    // Process center height
    float hCenter = processHeight(rawLum, lumMin, lumSpan, invertHeight,
                                  heightRemapMinVal, heightRemapMaxVal, heightContrastVal, heightOffsetVal);

    float gx, gy;

    if (filterMode == 1) {
        // Central differences: 4 samples, symmetric
        float rawL = sampleHeightRaw(texID, texUV - vec2(texelStep.x, 0), uvMin, uvMax, heightSourceMode, channelWeights, isBlock);
        float rawR = sampleHeightRaw(texID, texUV + vec2(texelStep.x, 0), uvMin, uvMax, heightSourceMode, channelWeights, isBlock);
        float rawD = sampleHeightRaw(texID, texUV - vec2(0, texelStep.y), uvMin, uvMax, heightSourceMode, channelWeights, isBlock);
        float rawU = sampleHeightRaw(texID, texUV + vec2(0, texelStep.y), uvMin, uvMax, heightSourceMode, channelWeights, isBlock);
        float hL = processHeight(rawL, lumMin, lumSpan, invertHeight, heightRemapMinVal, heightRemapMaxVal, heightContrastVal, heightOffsetVal);
        float hR = processHeight(rawR, lumMin, lumSpan, invertHeight, heightRemapMinVal, heightRemapMaxVal, heightContrastVal, heightOffsetVal);
        float hD = processHeight(rawD, lumMin, lumSpan, invertHeight, heightRemapMinVal, heightRemapMaxVal, heightContrastVal, heightOffsetVal);
        float hU = processHeight(rawU, lumMin, lumSpan, invertHeight, heightRemapMinVal, heightRemapMaxVal, heightContrastVal, heightOffsetVal);
        gx = (hR - hL) * 0.5;
        gy = (hU - hD) * 0.5;
    } else if (filterMode == 2) {
        // Sobel 3x3: 8 neighbor samples, best edge detection
        float rawTL = sampleHeightRaw(texID, texUV + vec2(-texelStep.x,  texelStep.y), uvMin, uvMax, heightSourceMode, channelWeights, isBlock);
        float rawT  = sampleHeightRaw(texID, texUV + vec2(0,             texelStep.y), uvMin, uvMax, heightSourceMode, channelWeights, isBlock);
        float rawTR = sampleHeightRaw(texID, texUV + vec2( texelStep.x,  texelStep.y), uvMin, uvMax, heightSourceMode, channelWeights, isBlock);
        float rawL  = sampleHeightRaw(texID, texUV + vec2(-texelStep.x,  0),           uvMin, uvMax, heightSourceMode, channelWeights, isBlock);
        float rawR  = sampleHeightRaw(texID, texUV + vec2( texelStep.x,  0),           uvMin, uvMax, heightSourceMode, channelWeights, isBlock);
        float rawBL = sampleHeightRaw(texID, texUV + vec2(-texelStep.x, -texelStep.y), uvMin, uvMax, heightSourceMode, channelWeights, isBlock);
        float rawB  = sampleHeightRaw(texID, texUV + vec2(0,            -texelStep.y), uvMin, uvMax, heightSourceMode, channelWeights, isBlock);
        float rawBR = sampleHeightRaw(texID, texUV + vec2( texelStep.x, -texelStep.y), uvMin, uvMax, heightSourceMode, channelWeights, isBlock);
        float hTL = processHeight(rawTL, lumMin, lumSpan, invertHeight, heightRemapMinVal, heightRemapMaxVal, heightContrastVal, heightOffsetVal);
        float hT  = processHeight(rawT,  lumMin, lumSpan, invertHeight, heightRemapMinVal, heightRemapMaxVal, heightContrastVal, heightOffsetVal);
        float hTR = processHeight(rawTR, lumMin, lumSpan, invertHeight, heightRemapMinVal, heightRemapMaxVal, heightContrastVal, heightOffsetVal);
        float hL  = processHeight(rawL,  lumMin, lumSpan, invertHeight, heightRemapMinVal, heightRemapMaxVal, heightContrastVal, heightOffsetVal);
        float hR  = processHeight(rawR,  lumMin, lumSpan, invertHeight, heightRemapMinVal, heightRemapMaxVal, heightContrastVal, heightOffsetVal);
        float hBL = processHeight(rawBL, lumMin, lumSpan, invertHeight, heightRemapMinVal, heightRemapMaxVal, heightContrastVal, heightOffsetVal);
        float hB  = processHeight(rawB,  lumMin, lumSpan, invertHeight, heightRemapMinVal, heightRemapMaxVal, heightContrastVal, heightOffsetVal);
        float hBR = processHeight(rawBR, lumMin, lumSpan, invertHeight, heightRemapMinVal, heightRemapMaxVal, heightContrastVal, heightOffsetVal);
        // Sobel X: [-1 0 +1; -2 0 +2; -1 0 +1]
        gx = (-hTL - 2.0*hL - hBL + hTR + 2.0*hR + hBR) / 8.0;
        // Sobel Y: [+1 +2 +1; 0 0 0; -1 -2 -1]
        gy = (hTL + 2.0*hT + hTR - hBL - 2.0*hB - hBR) / 8.0;
    } else if (filterMode == 3) {
        // Bilinear: sample at half-texel offsets for smooth gradients
        vec2 halfStep = texelStep * 0.5;
        float rawL = sampleHeightRaw(texID, texUV - vec2(halfStep.x, 0), uvMin, uvMax, heightSourceMode, channelWeights, isBlock);
        float rawR = sampleHeightRaw(texID, texUV + vec2(halfStep.x, 0), uvMin, uvMax, heightSourceMode, channelWeights, isBlock);
        float rawD = sampleHeightRaw(texID, texUV - vec2(0, halfStep.y), uvMin, uvMax, heightSourceMode, channelWeights, isBlock);
        float rawU = sampleHeightRaw(texID, texUV + vec2(0, halfStep.y), uvMin, uvMax, heightSourceMode, channelWeights, isBlock);
        float hL = processHeight(rawL, lumMin, lumSpan, invertHeight, heightRemapMinVal, heightRemapMaxVal, heightContrastVal, heightOffsetVal);
        float hR = processHeight(rawR, lumMin, lumSpan, invertHeight, heightRemapMinVal, heightRemapMaxVal, heightContrastVal, heightOffsetVal);
        float hD = processHeight(rawD, lumMin, lumSpan, invertHeight, heightRemapMinVal, heightRemapMaxVal, heightContrastVal, heightOffsetVal);
        float hU = processHeight(rawU, lumMin, lumSpan, invertHeight, heightRemapMinVal, heightRemapMaxVal, heightContrastVal, heightOffsetVal);
        gx = hR - hL;
        gy = hU - hD;
    } else if (filterMode == 4) {
        // Bicubic: 4th-order finite difference on each axis
        vec2 s1 = texelStep;
        vec2 s2 = texelStep * 2.0;
        float rawXn2 = sampleHeightRaw(texID, texUV - vec2(s2.x, 0), uvMin, uvMax, heightSourceMode, channelWeights, isBlock);
        float rawXn1 = sampleHeightRaw(texID, texUV - vec2(s1.x, 0), uvMin, uvMax, heightSourceMode, channelWeights, isBlock);
        float rawXp1 = sampleHeightRaw(texID, texUV + vec2(s1.x, 0), uvMin, uvMax, heightSourceMode, channelWeights, isBlock);
        float rawXp2 = sampleHeightRaw(texID, texUV + vec2(s2.x, 0), uvMin, uvMax, heightSourceMode, channelWeights, isBlock);
        float hXn2 = processHeight(rawXn2, lumMin, lumSpan, invertHeight, heightRemapMinVal, heightRemapMaxVal, heightContrastVal, heightOffsetVal);
        float hXn1 = processHeight(rawXn1, lumMin, lumSpan, invertHeight, heightRemapMinVal, heightRemapMaxVal, heightContrastVal, heightOffsetVal);
        float hXp1 = processHeight(rawXp1, lumMin, lumSpan, invertHeight, heightRemapMinVal, heightRemapMaxVal, heightContrastVal, heightOffsetVal);
        float hXp2 = processHeight(rawXp2, lumMin, lumSpan, invertHeight, heightRemapMinVal, heightRemapMaxVal, heightContrastVal, heightOffsetVal);
        // 4th-order central difference: (-f[-2] + 8*f[-1] - 8*f[+1] + f[+2]) / 12
        gx = (-hXn2 + 8.0*hXn1 - 8.0*hXp1 + hXp2) / 12.0;

        float rawYn2 = sampleHeightRaw(texID, texUV - vec2(0, s2.y), uvMin, uvMax, heightSourceMode, channelWeights, isBlock);
        float rawYn1 = sampleHeightRaw(texID, texUV - vec2(0, s1.y), uvMin, uvMax, heightSourceMode, channelWeights, isBlock);
        float rawYp1 = sampleHeightRaw(texID, texUV + vec2(0, s1.y), uvMin, uvMax, heightSourceMode, channelWeights, isBlock);
        float rawYp2 = sampleHeightRaw(texID, texUV + vec2(0, s2.y), uvMin, uvMax, heightSourceMode, channelWeights, isBlock);
        float hYn2 = processHeight(rawYn2, lumMin, lumSpan, invertHeight, heightRemapMinVal, heightRemapMaxVal, heightContrastVal, heightOffsetVal);
        float hYn1 = processHeight(rawYn1, lumMin, lumSpan, invertHeight, heightRemapMinVal, heightRemapMaxVal, heightContrastVal, heightOffsetVal);
        float hYp1 = processHeight(rawYp1, lumMin, lumSpan, invertHeight, heightRemapMinVal, heightRemapMaxVal, heightContrastVal, heightOffsetVal);
        float hYp2 = processHeight(rawYp2, lumMin, lumSpan, invertHeight, heightRemapMinVal, heightRemapMaxVal, heightContrastVal, heightOffsetVal);
        gy = (-hYn2 + 8.0*hYn1 - 8.0*hYp1 + hYp2) / 12.0;
    } else {
        // filterMode == 0: Forward differences (original behavior)
        float rawR = sampleHeightRaw(texID, texUV + vec2(texelStep.x, 0), uvMin, uvMax, heightSourceMode, channelWeights, isBlock);
        float rawU = sampleHeightRaw(texID, texUV + vec2(0, texelStep.y), uvMin, uvMax, heightSourceMode, channelWeights, isBlock);
        float hR = processHeight(rawR, lumMin, lumSpan, invertHeight, heightRemapMinVal, heightRemapMaxVal, heightContrastVal, heightOffsetVal);
        float hU = processHeight(rawU, lumMin, lumSpan, invertHeight, heightRemapMinVal, heightRemapMaxVal, heightContrastVal, heightOffsetVal);
        gx = hR - hCenter;
        gy = hU - hCenter;
    }

    // Normal clamp: limit maximum gradient magnitude to prevent light leak
    if (normalClampVal < 0.99) {
        float gradMag = length(vec2(gx, gy));
        if (gradMag > normalClampVal) {
            float scale = normalClampVal / gradMag;
            gx *= scale;
            gy *= scale;
        }
    }

    gx *= effectiveNormalStr;
    gy *= effectiveNormalStr;

    vec3 localNormal = vec3(-gx, gy, 1.0);
    if (invertNormal) localNormal.xy = -localNormal.xy;
    localNormal = normalize(localNormal);

    // Geometric blend: lerp toward flat normal to prevent light leak at grazing angles
    if (geometricBlendVal > 0.001) {
        localNormal = normalize(mix(localNormal, vec3(0.0, 0.0, 1.0), geometricBlendVal));
    }

    mat.normal = localNormal;
    mat.height = hCenter;
}

vec3 calculateNormal(vec3 p0, vec3 p1, vec3 p2, vec2 uv0, vec2 uv1, vec2 uv2, vec3 matNormal, vec3 viewDir, out vec3 geometricNormal, bool openglNormal) {
    vec3 edge1 = p1 - p0;
    vec3 edge2 = p2 - p0;
    vec3 geoNormalObj = normalize(cross(edge1, edge2));

    mat3 normalMatrix = transpose(mat3(gl_WorldToObject3x4EXT));
    vec3 geometricNormalWorld = normalize(normalMatrix * geoNormalObj);
    geometricNormal = geometricNormalWorld;

    if (any(isnan(matNormal))) { return geometricNormalWorld; }

    // TBN
    vec2 deltaUV1 = uv1 - uv0;
    vec2 deltaUV2 = uv2 - uv0;
    float det = deltaUV1.x * deltaUV2.y - deltaUV2.x * deltaUV1.y;

    vec3 tangentObj;
    if (abs(det) < 1e-6) {
        tangentObj = (abs(geoNormalObj.x) > 0.99) ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    } else {
        float f = 1.0 / det;
        tangentObj.x = f * (deltaUV2.y * edge1.x - deltaUV1.y * edge2.x);
        tangentObj.y = f * (deltaUV2.y * edge1.y - deltaUV1.y * edge2.y);
        tangentObj.z = f * (deltaUV2.y * edge1.z - deltaUV1.y * edge2.z);
    }

    // Gram-Schmidt
    vec3 TObj = normalize(tangentObj - geoNormalObj * dot(geoNormalObj, tangentObj));
    vec3 BObj = cross(geoNormalObj, TObj);

    vec3 T = normalize(normalMatrix * TObj);
    vec3 B = normalize(normalMatrix * BObj);
    vec3 N = geometricNormalWorld;

    // LabPBR uses DirectX (Y-) convention; Blender PBR uses OpenGL (Y+) — no flip needed
    vec3 correctedLocalNormal = matNormal;
    if (!openglNormal) {
        correctedLocalNormal.y = -correctedLocalNormal.y;
    }

    vec3 finalNormal = normalize(T * correctedLocalNormal.x + B * correctedLocalNormal.y + N * correctedLocalNormal.z);

    // unseenable faces
    if (dot(viewDir, finalNormal) < 0.0)
        return geometricNormalWorld;
    else
        return finalNormal;
}

void main() {
    vec3 viewDir = -mainRay.direction;

    uint instanceID = gl_InstanceCustomIndexEXT;
    uint geometryID = gl_GeometryIndexEXT;

    uint blasOffset;
    uint vertexFormat;
    UnpackedVertex v0, v1, v2;
    fetchTriangleVertices(instanceID, geometryID, gl_PrimitiveID, v0, v1, v2, blasOffset, vertexFormat);

    vec3 baryCoords = vec3(1.0 - (attribs.x + attribs.y), attribs.x, attribs.y);
    // Compute world-space hit position from ray parameters instead of vertex buffer positions.
    // This correctly handles per-geometry transforms (mega-BLAS) where vertex buffer positions
    // are in chunk-local space but gl_ObjectToWorldEXT reflects the mega-chunk TLAS transform.
    vec3 worldPos = gl_WorldRayOriginEXT + gl_HitTEXT * gl_WorldRayDirectionEXT;

    // Biome tint: shader-side resolution from per-section SSBO (bits 12-13 of flags)
    // 0=none, 1=grass, 2=foliage, 3=water. Replaces per-vertex colorLayer for biome tints.
    vec3 colorLayer = vec3(1.0);
    uint biomeTintType = (v0.flags >> PBR_FLAG_BIOME_TINT_SHIFT) & 0x3u;
    if (biomeTintType != 0u) {
        uint packed = biomeColors.data[instanceID][biomeTintType - 1u];
        colorLayer = vec3(
            float((packed >> 16u) & 0xFFu) / 255.0,
            float((packed >> 8u) & 0xFFu) / 255.0,
            float(packed & 0xFFu) / 255.0
        );
    } else if ((v0.flags & PBR_FLAG_USE_COLOR_LAYER) != 0u) {
        // Fixed-color tint (birch/spruce leaves, etc.) — per-vertex colorLayer
        colorLayer = baryCoords.x * v0.colorLayer + baryCoords.y * v1.colorLayer + baryCoords.z * v2.colorLayer;
    }

    bool useTexture = (v0.flags & PBR_FLAG_USE_TEXTURE) != 0u;
    float albedoEmission =
        baryCoords.x * v0.albedoEmission + baryCoords.y * v1.albedoEmission + baryCoords.z * v2.albedoEmission;
    uint textureID = v0.textureID;
    vec4 albedoValue;
    vec4 specularValue;
    vec4 normalValue;
    ivec4 flagValue;
    vec2 textureUV;
    vec2 uvMin = vec2(0.0);
    vec2 uvMax = vec2(1.0);
    vec3 rawAlbedoLinear = vec3(1.0);
    bool isBlockGeometry = (v0.flags & PBR_FLAG_BLOCK_GEOMETRY) != 0u;
    int specularTextureID = -1;
    int normalTextureID = -1;
    int flagTextureID = -1;

    if (useTexture) {
        textureUV = baryCoords.x * v0.textureUV + baryCoords.y * v1.textureUV + baryCoords.z * v2.textureUV;

        if (isBlockGeometry) {
            // === BLOCK GEOMETRY: texture array sampling via SpriteRegistry ===
            uint spriteId = textureID; // textureID now holds sequential spriteId for blocks

            // Greedy-merged quads: UVs extend beyond [0,1]. fract() tiles them.
            if ((v0.flags & PBR_FLAG_GREEDY_MERGED) != 0u) {
                textureUV = fract(textureUV);
            }

            uvMin = vec2(0.0);
            uvMax = vec2(1.0);

            float lod = 0;
            albedoValue = fetchBlockAlbedoLod(spriteId, textureUV, worldUbo.animTick, lod);

            rawAlbedoLinear = albedoValue.rgb;
            specularValue = fetchBlockSpecularLod(spriteId, textureUV, lod);
            normalValue = fetchBlockNormalLod(spriteId, textureUV, lod);
            flagValue = ivec4(0); // TODO: flag arrays in Phase 4

            // Grass block side overlay: use SpriteRegistry.overlaySprite
            SpriteEntry se = spriteEntries[spriteId];
            if (se.overlaySprite >= 0 && biomeTintType != 0u) {
                float overlayAlpha = fetchOverlayAlpha(spriteId, textureUV, worldUbo.animTick);
                colorLayer = mix(vec3(1.0), colorLayer, smoothstep(0.1, 0.9, overlayAlpha));
            }
        } else {
            // === ENTITY GEOMETRY: legacy atlas sampling via textures[] ===
            specularTextureID = mapping.entries[textureID].specular;
            normalTextureID = mapping.entries[textureID].normal;
            flagTextureID = mapping.entries[textureID].flag;

            uvMin = min(min(v0.textureUV, v1.textureUV), v2.textureUV);
            uvMax = max(max(v0.textureUV, v1.textureUV), v2.textureUV);

            float lod = 0;
            albedoValue = textureLod(textures[nonuniformEXT(textureID)], textureUV, lod);
            rawAlbedoLinear = albedoValue.rgb;
            if (specularTextureID >= 0) {
                specularValue = textureLod(textures[nonuniformEXT(specularTextureID)], textureUV, lod);
            } else {
                specularValue = vec4(0.0);
            }
            if (normalTextureID >= 0) {
                normalValue = textureLod(textures[nonuniformEXT(normalTextureID)], textureUV, lod);
            } else {
                normalValue = vec4(0.0);
            }
            if (flagTextureID >= 0) {
                vec4 floatFlagValue = textureLod(textures[nonuniformEXT(flagTextureID)], textureUV, ceil(lod));
                flagValue = ivec4(round(floatFlagValue * 255.0));
            } else {
                flagValue = ivec4(0);
            }
        }
    } else {
        albedoValue = vec4(1.0);
        specularValue = vec4(0.0);
        normalValue = vec4(0.0);
        flagValue = ivec4(0);
    }

    vec3 glint = vec3(0.0);
    vec4 overlayColor = vec4(0.0);
    if (!SIMPLIFIED_INDIRECT || mainRay.index <= 1) {
        if (!isBlockGeometry) {
            // Entity glint and overlay (blocks don't use these)
            float useGlint = float((v0.flags & PBR_FLAG_USE_GLINT) != 0u);
            uint glintTexture = v0.glintTexture;
            vec2 glintUV = baryCoords.x * v0.glintUV + baryCoords.y * v1.glintUV + baryCoords.z * v2.glintUV;
            glintUV = (worldUbo.textureMat * vec4(glintUV, 0.0, 1.0)).xy;
            glint = useGlint * texture(textures[nonuniformEXT(glintTexture)], glintUV).rgb;
            glint = glint * glint;

            bool useOverlay = (v0.flags & PBR_FLAG_USE_OVERLAY) != 0u;
            ivec2 overlayUV = ivec2(int(v0.overlayPacked & 0xFFFFu), int(v0.overlayPacked >> 16u));
            overlayColor = texelFetch(textures[nonuniformEXT(worldUbo.overlayTextureID)], overlayUV, 0);
        }
    }

    vec3 tint;
    if ((v0.flags & PBR_FLAG_USE_OVERLAY) != 0u && !isBlockGeometry) {
        tint = mix(overlayColor.rgb, albedoValue.rgb * colorLayer, overlayColor.a) + glint;
    } else {
        tint = albedoValue.rgb * colorLayer + glint;
    }

    tint = CS_BT709_TO_BT2020 * tint;  // BT.709 -> BT.2020 working space

    albedoValue = vec4(tint, albedoValue.a);
    LabPBRMat mat = convertLabPBRMaterial(albedoValue, specularValue, normalValue);

    // No normal texture → AO channel is uninitialized (0), not "full occlusion"
    // For blocks, AO comes from the normal array (defaults to 1.0 for missing normals)
    if (!isBlockGeometry && normalTextureID < 0) mat.ao = 1.0;

    // Save original texture properties before material overrides (for masks + Tex Roughness blend)
    float texSourceRoughness = mat.roughness;
    float texSourceLuminance = dot(mat.albedo, vec3(0.2126, 0.7152, 0.0722));

    // Principled BSDF material override from WorldUBO (5 × vec4 per block)
    // emissiveBlockType packs: bits 0-7 = emissive type, bits 8-15 = material type (ordinal+1, 0=none)
    float matNoiseScale = 0.0;
    float matNoiseStrength = 0.0;
    int matNoiseOctaves = 2;
    int matNoiseType = 0;
    int matNoiseSeed = 0;
    int matNoiseTarget = 1; // bit 0=roughness, bit 1=normal, bit 2=metallic, bit 3=roughness additive
    int matNoiseMaskMode = 0;  // 0=none, 1=luminance, 2=roughness, 3=metallic, 4=normal deviation
    bool matNoiseMaskInvert = false;
    float matNoiseMaskThreshold = 0.5;
    int matNoiseWrap = 0;  // 0=3D, 1=surface, 2=triplanar, 3=XZ, 4=XY, 5=YZ
    float matNoiseRotation = 0.0;  // radians
    float matNoiseAspect = 1.0;    // Y/X ratio
    float matNoiseLacunarity = 2.0;
    float matNoiseContrast = 1.0;
    float matGamutBoost = 1.0;
    uint packedBlockType = v0.emissiveBlockType;
    uint materialType = (packedBlockType >> 8u) & 0xFFu;

    // Material class resolution: vertex materialType → mask texture → SSBO lookup via BDA
    // Priority: vertex materialType (fast, no texture fetch) > mask texture (per-texel) > skip
    uint materialClassIdx = 0u;
    bool hasMaterialClass = false;
    if (materialType > 0u) {
        materialClassIdx = materialType - 1u;
        hasMaterialClass = true;
    } else if (!isBlockGeometry) {
        // Fallback: per-texel mask texture (R8_UNORM, value = MaterialBlock ordinal)
        // Only for entities — block textureID is spriteId, not a GLID for mapping.entries[]
        int maskTexID = mapping.entries[textureID].maskTexture;
        if (maskTexID >= 0) {
            materialClassIdx = uint(texture(textures[nonuniformEXT(maskTexID)], textureUV).r * 255.0 + 0.5);
            hasMaterialClass = (materialClassIdx < 160u);
        }
    }

    if (hasMaterialClass && pc.materialClassAddr != 0) {
        uint idx = materialClassIdx;
        MaterialClassBufferRef matBuf = MaterialClassBufferRef(pc.materialClassAddr);
        MaterialClassEntry mc = matBuf.materialClassMapping.entries[idx];
        vec4 pack0 = vec4(mc.f0, mc.roughness);
        vec4 pack1 = vec4(mc.metallic, mc.transmission, mc.ior, mc.subsurface);
        vec4 pack2 = vec4(mc.anisotropic, mc.sheenWeight, mc.sheenTint, mc.coatWeight);
        vec4 pack3 = vec4(mc.coatRoughness, mc.noiseScale, mc.noiseStrength, 0.0);
        vec4 pack5 = vec4(mc.gamutBoost, mc.noiseMaskThreshold, 0.0, mc.normalStrength);
        vec4 pack6 = vec4(mc.noiseRotation, mc.noiseAspect, mc.noiseLacunarity, mc.noiseContrast);

        {
            // Apply F0 override if set; for dielectrics with zero F0, derive from IOR
            if (dot(pack0.rgb, pack0.rgb) > 0.0001) {
                mat.f0 = pack0.rgb;
                // Auto-detect metals: if F0 is set above dielectric range (> 0.20)
                // and metallic not explicitly set, infer metallic for correct DLSS-RR
                // guide buffers (metals need zero diffuse albedo).
                float overrideMaxF0 = max(pack0.r, max(pack0.g, pack0.b));
                if (overrideMaxF0 > 0.20 && pack1.x < 0.01) {
                    mat.metallic = 1.0;
                    mat.albedo = mat.f0;
                }
            } else if (pack1.x < 0.5) {
                // Dielectric: compute F0 from IOR using Fresnel equation
                float ior = max(pack1.z, 1.0);
                float f0 = ((ior - 1.0) * (ior - 1.0)) / ((ior + 1.0) * (ior + 1.0));
                mat.f0 = vec3(f0);
            }
            // AutoPBR roughness + normal derivation
            if ((mc.flags & 0x8u) != 0u) {
                if (!isBlockGeometry) {
                    // Entity path: GPU-side AutoPBR
                    applyAutoPBR(mat, mc, rawAlbedoLinear, textureID, textureUV, uvMin, uvMax, gl_HitTEXT, false);
                } else if (pack1.y > 0.001) {
                    // Transmissive AutoPBR blocks (slime, glass, ice):
                    // CPU bake was skipped — use material class roughness slider
                    mat.roughness = max(pack0.a * pack0.a, 0.01);
                }
                // Non-transmissive AutoPBR blocks (iron, stone, wood):
                // Roughness from CPU-baked specular (via convertLabPBRMaterial). No override.
            } else {
                // Non-AutoPBR blocks: material class roughness slider
                mat.roughness = max(pack0.a * pack0.a, 0.01);
            }

            // Per-block POM removed — displacement handled by tessellation/DDA
            if (false) {
            }

            mat.metallic = pack1.x;
            if (pack1.y >= 0.0) mat.transmission = pack1.y;
            mat.ior = max(pack1.z, 1.0);
            mat.subSurface = pack1.w;
            mat.anisotropic = pack2.x;
            mat.sheenWeight = pack2.y;
            mat.sheenTint = pack2.z;
            mat.coatWeight = pack2.w;
            mat.coatRoughness = pack3.x;

            // Extract noise parameters from pack3
            matNoiseScale = pack3.y;
            matNoiseStrength = pack3.z;
            // noisePacked: octaves (bits 0-3) | noiseType (bits 4-8) | seed (bits 9-17) | noiseTarget (bits 20-23)
            int noisePacked = int(mc.noisePacked);
            matNoiseOctaves = clamp(noisePacked & 0xF, 1, 8);
            matNoiseType = min((noisePacked >> 4) & 0x1F, 23);  // 5 bits = 0-31, max valid type 23
            matNoiseSeed = (noisePacked >> 9) & 0x1FF;  // 9 bits = 0-511
            matNoiseTarget = (noisePacked >> 20) & 0xF;

            matGamutBoost = pack5.x;
            matNoiseMaskThreshold = pack5.y;
            int maskPacked = int(mc.noiseMaskPacked);
            matNoiseMaskMode = min(maskPacked & 0x7, 4);       // bits 0-2, max valid mode 4
            matNoiseMaskInvert = ((maskPacked >> 3) & 0x1) != 0; // bit 3
            matNoiseWrap = (maskPacked >> 4) & 0x7;    // bits 4-6

            // Normal strength: amplify/attenuate normal map.
            // Skip for AutoPBR entities (GPU applyAutoPBR handles it).
            // Apply for blocks (CPU-baked normals use neutral strength, shader applies runtime value).
            if ((mc.flags & 0x8u) == 0u || isBlockGeometry) {
                // Entity normals master toggle: flatten all entity normals when disabled
                if (!isBlockGeometry && !ENTITY_NORMALS_ON) {
                    mat.normal = vec3(0.0, 0.0, 1.0);
                } else {
                    float matNormalStrength = pack5.w;
                    if (matNormalStrength < 0.01) {
                        mat.normal = vec3(0.0, 0.0, 1.0); // flat — disable normal map
                    } else if (matNormalStrength != 1.0 && length(mat.normal) > 0.01) {
                        mat.normal.xy *= matNormalStrength;
                        mat.normal = normalize(mat.normal);
                    }
                }
            }

            matNoiseRotation = pack6.x;
            matNoiseAspect = pack6.y;
            matNoiseLacunarity = pack6.z;
            matNoiseContrast = pack6.w;

            if (mat.metallic > 0.5) {
                mat.albedo = mat.f0;
            }
        }
    }

    // Texture gamut boost: Oklab chroma scaling on dielectric albedo only
    // Per-material value from UBO; fallback to global push constant for vivid-flagged non-material blocks
    float gamutFactor = (materialType != 0u) ? matGamutBoost
                      : ((v0.emissiveBlockType & 0x10000u) != 0u ? pc.colorExpansion : 1.0);
    // Skip gamut boost for metals and HDR/emissive pixels (Oklab cube root overflows on values > 1.0)
    float maxAlbedo = max(mat.albedo.r, max(mat.albedo.g, mat.albedo.b));
    if (gamutFactor != 1.0 && mat.metallic < 0.5 && maxAlbedo <= 1.0) {
        const mat3 bt2020_to_lms = mat3(
            0.6167557871, 0.2651330639, 0.1001026342,
            0.3601983994, 0.6358393640, 0.2039065193,
            0.0230458134, 0.0990275718, 0.6959908464);
        const mat3 lms_to_lab = mat3(
            0.2104542553,  1.9779984951,  0.0259040371,
            0.7936177850, -2.4285922050,  0.7827717662,
           -0.0040720468,  0.4505937099, -0.8086757660);
        const mat3 lab_to_lms = mat3(
            1.0,           1.0,           1.0,
            0.3963377774, -0.1055613458, -0.0894841775,
            0.2158037573, -0.0638541728, -1.2914855480);
        const mat3 lms_to_bt2020 = mat3(
             2.1399067359, -0.8847358624, -0.0485737581,
            -1.2463895090,  2.1632309822, -0.4545031427,
             0.1064827729, -0.2784951194,  1.5030769008);

        vec3 lms = bt2020_to_lms * max(mat.albedo, vec3(0.0));
        vec3 lms_g = sign(lms) * pow(abs(lms), vec3(1.0 / 3.0));
        vec3 lab = lms_to_lab * lms_g;
        // Self-limiting chroma boost (Special K style): lerp toward target, bounded by sigmoid
        float boostAmount = 1.0 - exp2(-4.0 * gamutFactor * dot(lab.yz, lab.yz));
        lab.yz *= 1.0 + boostAmount * (gamutFactor - 1.0);
        vec3 lms_g2 = lab_to_lms * lab;
        vec3 lms2 = clamp(lms_g2 * lms_g2 * lms_g2, vec3(-10.0), vec3(10.0));
        mat.albedo = max(lms_to_bt2020 * lms2, vec3(0.0));
    }

    // the provided normal is unreliable! (such as grass, etc.)
    // calculate on the fly for now
    vec3 geometricNormal;
    vec3 normal =
        calculateNormal(v0.pos, v1.pos, v2.pos, v0.textureUV, v1.textureUV, v2.textureUV, mat.normal, viewDir, geometricNormal, false);

    // Procedural noise modulation — gated by noiseTarget bits
    // bit 0 = roughness, bit 1 = normal perturbation, bit 2 = metallic, bit 3 = roughness additive only
    // Skip noise when throughput is too low (deep bounces — noise contribution invisible)
    float maxThroughput = max(mainRay.throughput.r, max(mainRay.throughput.g, mainRay.throughput.b));
    if (matNoiseStrength > 0.001 && matNoiseTarget != 0 && maxThroughput > 0.01) {
        // Noise LOD: reduce octaves with distance, skip normal gradient far away
        int effectiveOctaves = matNoiseOctaves;
        float hitDist = gl_HitTEXT;
        bool skipNormalGradient = false;
        if (NOISE_LOD_ON) {
            // Reduce 1 octave per 32 blocks distance (high octaves are invisible at distance)
            effectiveOctaves = max(1, effectiveOctaves - int(hitDist / 32.0));
            // Normal gradient is 6x fbmTyped cost — skip beyond 48 blocks (invisible detail)
            skipNormalGradient = hitDist > 48.0;
        }

        // Absolute world coordinates, wrapped for float precision
        vec3 absWorldPos = worldPos + vec3(worldUbo.cameraPos.xyz);
        absWorldPos = mod(absWorldPos, 256.0);

        // Wrapping mode: determines how 3D position maps to noise coordinates
        vec3 noisePos;
        if (matNoiseWrap == 1) {
            // Surface: project onto face using geometric normal
            vec3 an = abs(geometricNormal);
            vec2 uv;
            if (an.y >= an.x && an.y >= an.z) uv = absWorldPos.xz;      // top/bottom
            else if (an.x >= an.y && an.x >= an.z) uv = absWorldPos.yz;  // east/west
            else uv = absWorldPos.xy;                                      // north/south
            noisePos = vec3(uv * matNoiseScale, 0.0);
        } else if (matNoiseWrap == 2) {
            // Triplanar: blend 3 projections weighted by normal (best quality, 3x cost)
            noisePos = absWorldPos * matNoiseScale; // used for all 3 projections below
        } else if (matNoiseWrap == 3) {
            noisePos = vec3(absWorldPos.xz * matNoiseScale, 0.0); // Planar XZ
        } else if (matNoiseWrap == 4) {
            noisePos = vec3(absWorldPos.xy * matNoiseScale, 0.0); // Planar XY
        } else if (matNoiseWrap == 5) {
            noisePos = vec3(absWorldPos.yz * matNoiseScale, 0.0); // Planar YZ
        } else {
            noisePos = absWorldPos * matNoiseScale; // 3D (default)
        }

        // Apply rotation + aspect ratio to noise coordinates
        if (matNoiseRotation != 0.0 || matNoiseAspect != 1.0) {
            // Rotate XY of noisePos (works for all wrapping modes)
            float cosR = cos(matNoiseRotation), sinR = sin(matNoiseRotation);
            vec2 rotated = vec2(noisePos.x * cosR - noisePos.y * sinR,
                                noisePos.x * sinR + noisePos.y * cosR);
            noisePos.x = rotated.x;
            noisePos.y = rotated.y * matNoiseAspect;
        }

        // Apply seed as spatial offset
        if (matNoiseSeed > 0) {
            noisePos += vec3(float(matNoiseSeed) * 7.13, float(matNoiseSeed) * 11.37, float(matNoiseSeed) * 23.71);
        }

        float n;
        if (matNoiseWrap == 2) {
            // Triplanar: evaluate noise on each plane, blend by normal weight
            vec3 w = abs(geometricNormal);
            w = w / (w.x + w.y + w.z + 0.001); // normalize weights
            vec3 seedOff = (matNoiseSeed > 0) ? vec3(float(matNoiseSeed) * 7.13, float(matNoiseSeed) * 11.37, float(matNoiseSeed) * 23.71) : vec3(0);
            float nXY = fbmTyped(vec3(absWorldPos.xy * matNoiseScale, 0.0) + seedOff, effectiveOctaves, matNoiseType, matNoiseLacunarity);
            float nXZ = fbmTyped(vec3(absWorldPos.xz * matNoiseScale, 0.0) + seedOff, effectiveOctaves, matNoiseType, matNoiseLacunarity);
            float nYZ = fbmTyped(vec3(absWorldPos.yz * matNoiseScale, 0.0) + seedOff, effectiveOctaves, matNoiseType, matNoiseLacunarity);
            n = nXY * w.z + nXZ * w.y + nYZ * w.x;
        } else {
            n = fbmTyped(noisePos, effectiveOctaves, matNoiseType, matNoiseLacunarity);
        }

        // Contrast: gamma-style power curve on noise output
        if (matNoiseContrast != 1.0) {
            n = sign(n) * pow(abs(n), 1.0 / max(matNoiseContrast, 0.01));
        }

        // ── Noise mask: hard binary — above threshold = full noise, below = zero ──
        if (matNoiseMaskMode > 0) {
            float rawMask = 0.0;
            if (matNoiseMaskMode == 1) rawMask = texSourceLuminance;
            else if (matNoiseMaskMode == 2) rawMask = texSourceRoughness;
            else if (matNoiseMaskMode == 3) rawMask = mat.metallic;
            else if (matNoiseMaskMode == 4) rawMask = 1.0 - abs(dot(normal, geometricNormal));

            if (matNoiseMaskInvert) rawMask = 1.0 - rawMask;
            // Hard step: above threshold = 1, below = 0
            float noiseMask = step(matNoiseMaskThreshold, rawMask);
            n *= noiseMask;
        }

        // Roughness modulation (bit 0 or bit 3)
        if ((matNoiseTarget & 1) != 0) {
            // Bidirectional: noise adds and removes roughness
            mat.roughness = clamp(mat.roughness + n * matNoiseStrength, 0.0, 1.0);
        } else if ((matNoiseTarget & 8) != 0) {
            // Additive only: noise only increases roughness (never makes things shinier)
            mat.roughness = clamp(mat.roughness + max(n, 0.0) * matNoiseStrength, 0.0, 1.0);
        }
        // Metallic modulation (bit 2)
        if ((matNoiseTarget & 4) != 0) {
            mat.metallic = clamp(mat.metallic + n * matNoiseStrength, 0.0, 1.0);
        }
        // Normal perturbation from noise gradient (bit 1) — also gated by mask
        if ((matNoiseTarget & 2) != 0 && !skipNormalGradient) {
            float eps = max(0.01 / matNoiseScale, 0.001);
            float normalMask = 1.0;
            if (matNoiseMaskMode > 0) {
                float rawM = 0.0;
                if (matNoiseMaskMode == 1) rawM = texSourceLuminance;
                else if (matNoiseMaskMode == 2) rawM = texSourceRoughness;
                else if (matNoiseMaskMode == 3) rawM = mat.metallic;
                else if (matNoiseMaskMode == 4) rawM = 1.0 - abs(dot(normal, geometricNormal));
                if (matNoiseMaskInvert) rawM = 1.0 - rawM;
                normalMask = step(matNoiseMaskThreshold, rawM);
            }
            vec3 grad = noiseGradientTyped(noisePos, eps, effectiveOctaves, matNoiseType, matNoiseLacunarity) * matNoiseStrength * 0.15 * normalMask;
            float gradLen = length(grad);
            if (gradLen > 1.0) grad /= gradLen;
            // Tangent frame must match the wrapping projection axes
            vec3 T, B;
            if (matNoiseWrap == 1) {
                // Surface mode: tangent axes match the projected UV axes
                vec3 an = abs(geometricNormal);
                if (an.y >= an.x && an.y >= an.z) { T = vec3(1,0,0); B = vec3(0,0,1); }      // top/bottom: UV=XZ
                else if (an.x >= an.y && an.x >= an.z) { T = vec3(0,1,0); B = vec3(0,0,1); }  // east/west: UV=YZ
                else { T = vec3(1,0,0); B = vec3(0,1,0); }                                     // north/south: UV=XY
            } else {
                T = normalize(cross(normal, abs(normal.y) < 0.99 ? vec3(0, 1, 0) : vec3(1, 0, 0)));
                B = cross(normal, T);
            }
            normal = normalize(normal + grad.x * T + grad.y * B);
        }
    }

    // --- Wet Surface Effect (Lagarde "Water Drop" model) ---
    // Rain wets upward-facing porous surfaces: roughness drops, albedo darkens, F0 → water IOR.
    float wetStrength = skyUBO.wetSurfaceStrength;
    if (skyUBO.rainGradient > 0.001 && wetStrength > 0.001
        && mat.metallic < 0.5 && mat.transmission < 0.5) {
        float rainExposure = max(dot(geometricNormal, vec3(0.0, 1.0, 0.0)), 0.0);
        float porosity = smoothstep(0.20, 0.85, mat.roughness);
        float wetness = clamp(skyUBO.rainGradient * rainExposure * porosity * wetStrength, 0.0, 1.0);
        if (wetness > 0.001) {
            float wf = (1.0 - wetness);
            mat.roughness = max(mat.roughness * wf * wf, 0.01);
            float darken = mix(1.0, 0.3, wetness * porosity);
            mat.albedo *= darken;
            albedoValue.rgb *= darken;
            mat.f0 = mix(mat.f0, vec3(0.02), wetness);
        }
    }

    // Write post-override material to mainRay for rgen G-buffer / DLSS-RR guide buffers
    mainRay.roughness = mat.roughness;
    mainRay.metallic = mat.metallic;
    mainRay.f0 = mat.f0;
    mainRay.emission = mat.emission;
    mainRay.subSurface = mat.subSurface;

    // Scene-referred emission normalization: 1.0 = 200 cd/m² (ITU-R BT.2408 paper white).
    // Physical nit values from vertex data are divided by this to keep radiance in a range
    // that fits RGBA16F after pre-exposure multiplication (max ~60 × preExp 32 = 1920).
    const float EMISSION_REFERENCE_NITS = 200.0;

    // Visible area light cube: if this hit point is inside a small light cube, render as solid emissive
    if (AREA_LIGHTS_ON && pc.areaLightCount > 0 && mainRay.index == 0) {
        int checkCount = pc.areaLightCount;

        for (int i = 0; i < checkCount; i++) {
            int idx = i;
            AreaLight al = areaLightBuffer.lights[idx];
            if (al.halfExtent >= 0.3) continue;

            vec3 delta = abs(worldPos - al.position);
            if (all(lessThanEqual(delta, vec3(al.halfExtent)))) {
                // Fade cube brightness at close range to prevent clear/denoised boundary artifacts
                float cubeDist = length(al.position - gl_WorldRayOriginEXT);
                float cubeFade = smoothstep(0.3, 1.5, cubeDist);
                if (cubeFade < 0.01) continue;

                // Visible emissive cube — route through CLEAR path (bypasses NRD denoiser)
                mainRay.radiance = al.color * max(al.intensity, 1.0) * cubeFade * mainRay.throughput;
                mainRay.albedoValue = vec4(al.color, 1.0);
                mainRay.albedoEmission = 0.0;
                mainRay.directLightRadiance = vec3(0.0);
                mainRay.directLightHitT = INF_DISTANCE;
                mainRay.hitT = gl_HitTEXT;
                mainRay.coneWidth += gl_HitTEXT * mainRay.coneSpread;
                mainRay.worldPos = worldPos;
                mainRay.normal = normal;
                mainRay.instanceIndex = instanceID;
                mainRay.geometryIndex = geometryID;
                mainRay.primitiveIndex = gl_PrimitiveID;
                mainRay.baryCoords = baryCoords;
                mainRay.roughness = 1.0;
                // Clear noisy/lobeType, set stop=1 (preserve isHand/insideBoat/emBlockType)
                mainRay.flags = (mainRay.flags & (PR_ISHAND_BIT | PR_INSIDEBOAT_BIT)) | PR_STOP_BIT | (prGetEmBlockType(mainRay) << PR_EMBLOCK_SHIFT);
                return;
            }
        }
    }

    // add glowing radiance (with area light / emissive mutual exclusion)
    // albedoEmission sign convention:
    //   positive = emissive mode (full bounce emission)
    //   negative = area light mode (suppress bounce emission; abs = self-glow value)
    float rawAlbedoEmission = albedoEmission;
    albedoEmission = abs(rawAlbedoEmission);
    bool hasAreaLight = rawAlbedoEmission < -0.0001;

    // Apply per-block emission data from UBO (color override + scalar multiplier)
    // Extract emissive type from lower 8 bits of packed field
    uint emBlockType = packedBlockType & 0xFFu;
    prSetEmBlockType(mainRay, emBlockType);
    vec3 emissionTint = tint; // default: texture albedo (BT.2020)
    bool uniformGlow = false;
    if (emBlockType < 50u && albedoEmission > 0.0) {
        vec4 emData = worldUbo.emissionData[emBlockType];
        uniformGlow = (emData.a < 0.0); // negative multiplier = uniform glow flag
        albedoEmission *= abs(emData.a); // strip sign, apply magnitude
        if (dot(emData.rgb, emData.rgb) > 0.0001) {
            // Flame mask: only bright texture pixels (actual flame) get the spectral color.
            // Dark pixels (wood stick, stone base) keep their texture albedo.
            float texLum = dot(tint, vec3(0.2627, 0.6780, 0.0593)); // BT.2020 luminance
            float flameMask = smoothstep(0.15, 0.5, texLum);
            emissionTint = mix(tint, emData.rgb, flameMask);
        }
    }

    // Per-emissive-block gamut boost (Oklab chroma scaling on emission tint)
    if (emBlockType < 50u) {
        float emGamut = worldUbo.emissiveGamut[emBlockType >> 2u][emBlockType & 3u];
        if (emGamut != 1.0) {
            const mat3 bt2020_to_lms = mat3(
                0.6167557871, 0.2651330639, 0.1001026342,
                0.3601983994, 0.6358393640, 0.2039065193,
                0.0230458134, 0.0990275718, 0.6959908464);
            const mat3 lms_to_lab = mat3(
                0.2104542553,  1.9779984951,  0.0259040371,
                0.7936177850, -2.4285922050,  0.7827717662,
               -0.0040720468,  0.4505937099, -0.8086757660);
            const mat3 lab_to_lms = mat3(
                1.0,           1.0,           1.0,
                0.3963377774, -0.1055613458, -0.0894841775,
                0.2158037573, -0.0638541728, -1.2914855480);
            const mat3 lms_to_bt2020 = mat3(
                 2.1399067359, -0.8847358624, -0.0485737581,
                -1.2463895090,  2.1632309822, -0.4545031427,
                 0.1064827729, -0.2784951194,  1.5030769008);

            vec3 lms = bt2020_to_lms * max(emissionTint, vec3(0.0));
            vec3 lms_g = sign(lms) * pow(abs(lms), vec3(1.0 / 3.0));
            vec3 lab = lms_to_lab * lms_g;
            lab.yz *= emGamut;
            vec3 lms_g2 = lab_to_lms * lab;
            vec3 lms2 = clamp(lms_g2 * lms_g2 * lms_g2, vec3(-10.0), vec3(10.0));
            emissionTint = lms_to_bt2020 * lms2;
            float minC = min(emissionTint.r, min(emissionTint.g, emissionTint.b));
            if (minC < 0.0) {
                float luma = dot(emissionTint, vec3(0.2627, 0.6780, 0.0593));
                float t = luma / max(luma - minC, 1e-6);
                emissionTint = mix(vec3(luma), emissionTint, t);
            }
        }
    }

    // Emission in scene-referred units (1.0 = EMISSION_REFERENCE_NITS cd/m²).
    // Vertex nits (albedoEmission) are authoritative when set; LabPBR is fallback only.
    float combinedEmission = (albedoEmission != 0.0) ? albedoEmission : (mat.emission * EMISSION_REFERENCE_NITS);
    float sceneEmission = combinedEmission / EMISSION_REFERENCE_NITS;

    // Texture-based emission mask: only bright texels (flame head) emit.
    // Dark texels (wood stick, stone base) get zero emission.
    // Uniform glow blocks skip this mask (emit from all texels equally).
    if (!uniformGlow) {
        float maskLum = dot(tint, vec3(0.2627, 0.6780, 0.0593));
        sceneEmission *= max(smoothstep(0.10, 0.40, maskLum), 0.05);
    }

    float factor;
    if (mainRay.index == 0) {
        factor = 1.0; // Primary hit: always show self-glow
    } else if (hasAreaLight) {
        factor = 0.0;
    } else {
        factor = 1.0; // Emissive mode
    }

    // Emission radiance: emissionTint is BT.2020 (texture albedo or spectral flame color).
    vec3 emissionRadiance = factor * emissionTint * sceneEmission * mainRay.throughput;

    // Per-sample contribution clamping to prevent fireflies (scene-referred range)
    // Ground truth: no clamping — accumulation converges naturally
    if (!NO_EMISSION_CLAMP) {
        float emissionLum = luminanceBT2020(emissionRadiance);
        float maxContribution = 1000.0;
        if (emissionLum > maxContribution) {
            emissionRadiance *= maxContribution / emissionLum;
        }
    }

    mainRay.radiance += emissionRadiance;
    mainRay.albedoEmission = albedoEmission; // Store abs value for bloom

    mainRay.hitT = gl_HitTEXT;
    mainRay.coneWidth += gl_HitTEXT * mainRay.coneSpread;

    // Perfect specular surfaces have a Dirac delta BRDF — they redirect light,
    // they don't scatter it. Direct lighting appears through the reflection chain.
    bool isPerfectSpecular = (mat.roughness < 0.001);

    if (!isPerfectSpecular) {
    // shadow ray for direct lighting
    vec3 sunDir = normalize(skyUBO.sunDirection);
    vec3 lightDir = sunDir;
    // Softer sun sampling for hand = wider penumbras (500 vs 3000)
    // Ground truth: physical sun disk half-angle 0.267° → kappa ≈ 46000
    float kappa = PHYSICAL_SUN_DISK ? 46000.0 : ((prGetIsHand(mainRay)) ? 500.0 : 3000.0);
    if (sunDir.y < 0) { lightDir = normalize(skyUBO.moonDirection); }
    vec2 vmfBN = vec2(-1.0); // A/B test: force PCG, disable blue noise
    vec3 sampledLightDir = SampleVMF(mainRay.seed, lightDir, kappa, vmfBN);
    vec3 shadowBiasN = dot(sampledLightDir, geometricNormal) > 0.0 ? geometricNormal : -geometricNormal;
    vec3 shadowRayOrigin = offset_ray(worldPos, shadowBiasN);

    if (worldUbo.skyType == 1) {
        float pdf; // not used
        vec3 lightBRDF = DisneyEval(mat, viewDir, normal, sampledLightDir, pdf, pc.flags);

        shadowRay.radiance = vec3(0.0);
        shadowRay.throughput = vec3(1.0);
        shadowRay.seed = mainRay.seed;
        shadowRay.hitT = INF_DISTANCE;
        shadowRay.insideBoat = prGetInsideBoat(mainRay) ? 1u : 0u;
        shadowRay.bounceIndex = mainRay.index;
        shadowRay.mediumAbsorption = vec3(0.0);
        shadowRay.mediumEntryT = 0.0;

        uint shadowMask = WORLD_MASK;
        if (!prGetIsHand(mainRay)) {
            shadowMask |= PLAYER_MASK | PLAYER_HEAD_MASK;  // world surfaces see player shadows; hand does not (prevents self-shadowing)
        }

        traceRayEXT(topLevelAS, gl_RayFlagsNoneEXT,
                    shadowMask,
                    0,                                     // sbtRecordOffset
                    0,                                     // sbtRecordStride
                    2,                                     // missIndex
                    shadowRayOrigin, 0.0, sampledLightDir, 1000, 1);

        // Add direct lighting contribution
        vec3 lightContribution = shadowRay.radiance;

        // Apply cloud shadowing (procedural volumetric slab).
        // This is evaluated at the shading point so it works for primary and reflected paths.
        float cloudT = cloudTransmittance(worldPos + sampledLightDir * 0.01, sampledLightDir, 0.0, 1000.0, worldUbo, skyUBO, 0);
        float shadowStrength = max(skyUBO.cloudLighting.x, 0.0);
        cloudT = pow(max(cloudT, 1e-6), shadowStrength);
        lightContribution *= cloudT;

        float progress = skyUBO.rainGradient;
        vec3 lightRadiance = lightContribution * mainRay.throughput * lightBRDF;
        vec3 finalLightRadiance = mix(lightRadiance, vec3(0.0), progress);

        // Hand shadow smoothing: apply ambient floor with smooth falloff
        // Prevents harsh black transitions on hand geometry in shadow
        // Ground truth: no fake ambient — accumulation provides correct indirect fill
        if (prGetIsHand(mainRay) && !NO_HAND_AMBIENT) {
            float shadowLum = dot(finalLightRadiance, vec3(0.2627, 0.6780, 0.0593));
            float ambientFloor = 0.08 * dot(mainRay.throughput, vec3(0.2627, 0.6780, 0.0593));
            float blend = smoothstep(0.0, ambientFloor * 2.0, shadowLum);
            vec3 handAmbient = ambientFloor * tint * mainRay.throughput;
            finalLightRadiance = mix(handAmbient, finalLightRadiance, blend);
        }

        mainRay.radiance += finalLightRadiance;

        mainRay.directLightRadiance = finalLightRadiance;
        mainRay.directLightHitT = shadowRay.hitT;
    }

    // Area light illumination
    if (AREA_LIGHTS_ON && pc.areaLightCount > 0 && mainRay.index == 0) {
        int searchCount = pc.areaLightCount;

        if (RESTIR_ENABLED) {
            // ======== Clean ReSTIR DI: RIS → Temporal → Shadow → Shade ========
            ivec2 pixel = ivec2(mainRay.pixelPacked & 0xFFFFu, mainRay.pixelPacked >> 16u);

            // --- Load previous frame reservoir via motion vector reprojection ---
            // MV read is from previous dispatch (stale but consistent per-pixel).
            // No normal/depth validation here — those images have intra-dispatch race
            // conditions (rgen writes after CHS). Temporal merge naturally handles
            // geometry changes via target PDF re-evaluation at current worldPos.
            vec2 mv = imageLoad(motionVectorImage, pixel).xy;
            ivec2 prevPixel = ivec2(round(vec2(pixel) + mv));
            ivec2 imgSize = imageSize(reservoirPreviousImage);
            bool prevValid = (prevPixel.x >= 0 && prevPixel.y >= 0 &&
                              prevPixel.x < imgSize.x && prevPixel.y < imgSize.y);

            vec4 prevPacked = prevValid ? imageLoad(reservoirPreviousImage, prevPixel) : vec4(0.0);
            Reservoir prevRes = unpackReservoir(prevPacked);

            // --- Step 1: RIS candidates — uniform random from tile list ---
            Reservoir currentRes = createReservoir();
            int prevLightIdx = -1;
            uint prevStableId = prevRes.lightStableId;

            // Use full light buffer as candidate pool — temporal reuse (M=20) converges.
            // Per-pixel cost stays at risCandidates (32) evaluations regardless of pool size.
            int effectiveCount = pc.areaLightCount;
            int numCandidates = min(pc.risCandidates, effectiveCount);
            float sourcePdf = 1.0 / float(max(effectiveCount, 1));

            for (int c = 0; c < numCandidates; c++) {
                // Blue noise light selection (first bounce), PCG fallback for indirect
                float lightRand = rand(mainRay.seed); // A/B: PCG only
                int tileSlot = int(lightRand * float(effectiveCount));
                tileSlot = clamp(tileSlot, 0, effectiveCount - 1);
                int idx = tileSlot;
                AreaLight al = areaLightBuffer.lights[idx];


                // Opportunistic stableId match (O(1) — no separate scan)
                if (prevLightIdx < 0 && prevStableId != 0u) {
                    uint sid = floatBitsToUint(al._unused.x);
                    if (sid == prevStableId) prevLightIdx = idx;
                }

                vec3 toLight = al.position - worldPos;
                float dist2 = dot(toLight, toLight);
                float dist = sqrt(dist2);
                vec3 alDir = toLight / max(dist, 0.001);

                // Distance culling (Chebyshev or Euclidean)
                vec3 absToLight = abs(toLight);
                float chebyDist = max(absToLight.x, max(absToLight.y, absToLight.z));
                float cullDist = chebyDist;
                if (cullDist > al.radius) continue;

                float NdotL = dot(normal, alDir);
                if (NdotL < -0.1 || dist < 0.05) continue;

                // Windowed inverse-square attenuation
                float invDist2 = 1.0 / max(dist2, 0.01);
                float f = cullDist / al.radius;
                float window = max(1.0 - f * f, 0.0);
                window *= window;

                float atten;
                if (al.halfExtent >= 0.01) {
                    float faceArea = cubeFaceArea(al.halfExtent);
                    vec3 absDir = abs(alDir);
                    float cosLight = max(absDir.x, max(absDir.y, absDir.z));
                    atten = faceArea * cosLight * invDist2 * window;
                } else {
                    atten = invDist2 * window;
                }
                if (atten < 0.00001) continue;

                // Luminance-only target PDF (canonical ReSTIR — position-dependent, view-independent)
                vec3 unshadowed = al.color * al.intensity * atten;
                float targetPdf = computeTargetPdf(unshadowed);
                if (targetPdf <= 0.0) continue;

                float weight = targetPdf / sourcePdf;
                uint stableId = floatBitsToUint(al._unused.x);
                updateReservoir(currentRes, stableId, idx, weight, targetPdf, unshadowed, alDir, dist, mainRay.seed);
            }

            // Fallback: scan full list for stableId match (uint comparison only)
            if (prevLightIdx < 0 && prevStableId != 0u) {
                int scanCount = effectiveCount;
                for (int i = 0; i < scanCount; i++) {
                    int idx = i;
                    if (floatBitsToUint(areaLightBuffer.lights[idx]._unused.x) == prevStableId) {
                        prevLightIdx = idx;
                        break;
                    }
                }
            }

            // --- Step 2: Temporal reuse — merge previous reservoir ---
            if (prevLightIdx >= 0 && prevRes.M > 0.0) {
                AreaLight prevAL = areaLightBuffer.lights[prevLightIdx];

                vec3 toLight = prevAL.position - worldPos;
                float dist2 = dot(toLight, toLight);
                float dist = sqrt(dist2);

                vec3 absToLight = abs(toLight);
                float chebyDist = max(absToLight.x, max(absToLight.y, absToLight.z));
                float cullDist = chebyDist;

                if (cullDist <= prevAL.radius && dist >= 0.05) {
                    vec3 alDir = toLight / max(dist, 0.001);

                    if (dot(normal, alDir) > -0.1) {
                        float invDist2 = 1.0 / max(dist2, 0.01);
                        float f2 = cullDist / prevAL.radius;
                        float window = max(1.0 - f2 * f2, 0.0);
                        window *= window;

                        float atten;
                        if (prevAL.halfExtent >= 0.01) {
                            float faceArea = cubeFaceArea(prevAL.halfExtent);
                            vec3 absDir = abs(alDir);
                            float cosLight = max(absDir.x, max(absDir.y, absDir.z));
                            atten = faceArea * cosLight * invDist2 * window;
                        } else {
                            atten = invDist2 * window;
                        }

                        vec3 prevUnshadowed = prevAL.color * prevAL.intensity * atten;
                        float prevTargetPdf = computeTargetPdf(prevUnshadowed);

                        if (prevTargetPdf > 0.0) {
                            prevRes.M = min(prevRes.M, float(pc.temporalMClamp));
                            prevRes.unshadowed = prevUnshadowed;
                            prevRes.lightDir = alDir;
                            prevRes.lightDist = dist;
                            prevRes.lightIdx = prevLightIdx;
                            mergeReservoir(currentRes, prevRes, prevTargetPdf, mainRay.seed);
                        }
                    }
                }
            }

            finalizeReservoir(currentRes);
            currentRes.W = min(currentRes.W, float(pc.wClamp));

            // --- Step 3: Shadow ray + shade selected light ---
            vec3 alAccum = vec3(0.0);
            if (currentRes.lightIdx >= 0) {
                AreaLight al = areaLightBuffer.lights[currentRes.lightIdx];

                vec2 cubeBN = vec2(-1.0); // A/B: PCG only
                CubeSample cs = sampleCubeLight(al, worldPos, pc.shadowSoftness, mainRay.seed, cubeBN);
                vec3 toSample = cs.worldPos - worldPos;
                float sDist = length(toSample);
                vec3 sDir = toSample / sDist;
                vec3 shadowOrigin = offset_ray(worldPos, geometricNormal);

                shadowRay.radiance = vec3(0.0);
                shadowRay.throughput = vec3(0.0);
                shadowRay.seed = mainRay.seed;
                shadowRay.hitT = INF_DISTANCE;
                shadowRay.insideBoat = prGetInsideBoat(mainRay) ? 1u : 0u;
                shadowRay.bounceIndex = mainRay.index;

                float safeMargin = al.halfExtent + 0.02;
                float alTMax = sDist - safeMargin;
                if (alTMax > 0.01) {
                    traceRayEXT(topLevelAS,
                        gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT,
                        WORLD_MASK, 0, 0, 3,
                        shadowOrigin, 0.0, sDir, alTMax, 1);
                } else {
                    shadowRay.throughput = vec3(1.0);
                }

                float visibility = shadowRay.throughput.x;

                vec3 brdf;
                if (RESTIR_SIMPLIFIED_BRDF) {
                    float NdotL = max(dot(normal, currentRes.lightDir), 0.0);
                    brdf = NdotL / PI * mat.albedo;
                } else {
                    float pdf;
                    brdf = DisneyEval(mat, viewDir, normal, currentRes.lightDir, pdf, pc.flags);
                }
                alAccum = currentRes.unshadowed * brdf * currentRes.W * visibility;
            }

            mainRay.directLightRadiance += alAccum * mainRay.throughput;
            mainRay.radiance += alAccum * mainRay.throughput;

            // --- Step 4: Store reservoir for next frame (spatial reuse reads this) ---
            imageStore(reservoirCurrentImage, pixel, packReservoir(currentRes));

        } else {
            // ======== Fallback: deterministic top-2 shadowed ========
            int bestIdx[2] = int[](-1, -1);
            float bestContrib[2] = float[](0.0, 0.0);
            vec3 bestUnshadowed[2] = vec3[](vec3(0.0), vec3(0.0));
            vec3 bestDir[2] = vec3[](vec3(0.0), vec3(0.0));
            float bestDist[2] = float[](0.0, 0.0);

            for (int i = 0; i < searchCount; i++) {
                AreaLight al = areaLightBuffer.lights[i];

                vec3 toLight = al.position - worldPos;
                float dist2 = dot(toLight, toLight);
                float dist = sqrt(dist2);
                vec3 alDir = toLight / max(dist, 0.001);

                // Distance culling (Chebyshev or Euclidean)
                vec3 absToLight = abs(toLight);
                float chebyDist = max(absToLight.x, max(absToLight.y, absToLight.z));
                float cullDist = chebyDist;
                if (cullDist > al.radius) continue;

                float NdotL = dot(normal, alDir);
                if (NdotL < -0.1 || dist < 0.05) continue;

                float invDist2 = 1.0 / max(dist2, 0.01);
                float f = cullDist / al.radius;
                float window = max(1.0 - f * f, 0.0);
                window *= window;

                float atten;
                if (al.halfExtent >= 0.01) {
                    float faceArea = cubeFaceArea(al.halfExtent);
                    vec3 absDir = abs(alDir);
                    float cosLight = max(absDir.x, max(absDir.y, absDir.z));
                    atten = faceArea * cosLight * invDist2 * window;
                } else {
                    atten = invDist2 * window;
                }
                if (atten < 0.00001) continue;

                // DisneyEval already includes cosine — no NdotL here
                vec3 unshadowed = al.color * al.intensity * atten;
                float contrib = dot(unshadowed, vec3(0.2627, 0.6780, 0.0593));

                if (contrib > bestContrib[0]) {
                    bestIdx[1] = bestIdx[0]; bestContrib[1] = bestContrib[0];
                    bestUnshadowed[1] = bestUnshadowed[0]; bestDir[1] = bestDir[0]; bestDist[1] = bestDist[0];
                    bestIdx[0] = i; bestContrib[0] = contrib;
                    bestUnshadowed[0] = unshadowed; bestDir[0] = alDir; bestDist[0] = dist;
                } else if (contrib > bestContrib[1]) {
                    bestIdx[1] = i; bestContrib[1] = contrib;
                    bestUnshadowed[1] = unshadowed; bestDir[1] = alDir; bestDist[1] = dist;
                }
            }

            vec3 alAccum = vec3(0.0);
            for (int k = 0; k < 2; k++) {
                if (bestIdx[k] < 0) continue;
                AreaLight al = areaLightBuffer.lights[bestIdx[k]];

                vec2 cubeBN2 = vec2(-1.0); // A/B: PCG only
                CubeSample cs = sampleCubeLight(al, worldPos, pc.shadowSoftness, mainRay.seed, cubeBN2);
                vec3 toSample = cs.worldPos - worldPos;
                float sDist = length(toSample);
                vec3 sDir = toSample / sDist;
                vec3 shadowOrigin = offset_ray(worldPos, geometricNormal);

                shadowRay.radiance = vec3(0.0);
                shadowRay.throughput = vec3(0.0);
                shadowRay.seed = mainRay.seed;
                shadowRay.hitT = INF_DISTANCE;
                shadowRay.insideBoat = prGetInsideBoat(mainRay) ? 1u : 0u;
                shadowRay.bounceIndex = mainRay.index;

                float safeMargin = al.halfExtent + 0.02;
                float alTMax = sDist - safeMargin;
                if (alTMax > 0.01) {
                    traceRayEXT(topLevelAS,
                        gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT,
                        WORLD_MASK, 0, 0, 3,
                        shadowOrigin, 0.0, sDir, alTMax, 1);
                } else {
                    shadowRay.throughput = vec3(1.0);
                }

                float visibility = shadowRay.throughput.x;
                vec3 brdf;
                if (RESTIR_SIMPLIFIED_BRDF) {
                    float NdotL = max(dot(normal, bestDir[k]), 0.0);
                    brdf = NdotL / PI * mat.albedo;
                } else {
                    float pdf;
                    brdf = DisneyEval(mat, viewDir, normal, bestDir[k], pdf, pc.flags);
                }
                alAccum += bestUnshadowed[k] * brdf * visibility;
            }

            mainRay.directLightRadiance += alAccum * mainRay.throughput;
            mainRay.radiance += alAccum * mainRay.throughput;
        }
    } else if (AREA_LIGHTS_ON && pc.areaLightCount > 0 && mainRay.index > 0 && RESTIR_ENABLED && RESTIR_BOUNCE_ENABLED) {
        // ======== Bounce 1-3 ReSTIR DI: RIS → Temporal → Shadow → Shade ========
        // Per-bounce reservoir image, same-pixel temporal reuse (no MV reprojection).
        // Uses global SSBO (camera-sorted) since bounce hits have no screen-space tile list.
        ivec2 pixel = ivec2(mainRay.pixelPacked & 0xFFFFu, mainRay.pixelPacked >> 16u);

        // --- Load previous bounce reservoir for THIS bounce level ---
        Reservoir prevRes = unpackReservoir(loadBounceReservoir(mainRay.index, pixel));

        // --- Step 1: RIS candidates from global SSBO ---
        Reservoir currentRes = createReservoir();
        int prevLightIdx = -1;
        uint prevStableId = prevRes.lightStableId;

        int searchCount = pc.areaLightCount;
        int numCandidates = min(pc.risCandidates, searchCount);
        float sourcePdf = 1.0 / float(max(searchCount, 1));

        for (int c = 0; c < numCandidates; c++) {
            float lightRand2 = rand(mainRay.seed); // A/B: PCG only
            int idx = clamp(int(lightRand2 * float(searchCount)), 0, searchCount - 1);
            AreaLight al = areaLightBuffer.lights[idx];


            // Opportunistic stableId match for temporal reuse
            uint sid = floatBitsToUint(al._unused.x);
            if (prevLightIdx < 0 && prevStableId != 0u && sid == prevStableId)
                prevLightIdx = idx;

            vec3 toLight = al.position - worldPos;
            float dist2 = dot(toLight, toLight);
            float dist = sqrt(dist2);
            vec3 alDir = toLight / max(dist, 0.001);

            vec3 absToLight = abs(toLight);
            float chebyDist = max(absToLight.x, max(absToLight.y, absToLight.z));
            float cullDist = chebyDist;
            if (cullDist > al.radius) continue;
            float NdotL = dot(normal, alDir);
            if (NdotL < -0.1 || dist < 0.05) continue;

            float invDist2 = 1.0 / max(dist2, 0.01);
            float f = cullDist / al.radius;
            float window = max(1.0 - f * f, 0.0);
            window *= window;

            float atten;
            if (al.halfExtent >= 0.01) {
                float faceArea = cubeFaceArea(al.halfExtent);
                vec3 absDir = abs(alDir);
                float cosLight = max(absDir.x, max(absDir.y, absDir.z));
                atten = faceArea * cosLight * invDist2 * window;
            } else {
                atten = invDist2 * window;
            }
            if (atten < 0.00001) continue;

            vec3 unshadowed = al.color * al.intensity * atten;
            float targetPdf = computeTargetPdf(unshadowed);
            if (targetPdf <= 0.0) continue;

            float weight = targetPdf / sourcePdf;
            updateReservoir(currentRes, sid, idx, weight, targetPdf, unshadowed, alDir, dist, mainRay.seed);
        }

        // Fallback: scan full list for stableId match (uint comparison only)
        if (prevLightIdx < 0 && prevStableId != 0u) {
            int scanCount = searchCount;
            for (int i = 0; i < scanCount; i++) {
                if (floatBitsToUint(areaLightBuffer.lights[i]._unused.x) == prevStableId) {
                    prevLightIdx = i;
                    break;
                }
            }
        }

        // --- Step 2: Temporal merge (halved M clamp for bounce stability) ---
        if (prevLightIdx >= 0 && prevRes.M > 0.0) {
            AreaLight prevAL = areaLightBuffer.lights[prevLightIdx];

            vec3 toLight = prevAL.position - worldPos;
            float dist2 = dot(toLight, toLight);
            float dist = sqrt(dist2);

            vec3 absToLight = abs(toLight);
            float chebyDist = max(absToLight.x, max(absToLight.y, absToLight.z));
            float cullDist = chebyDist;

            if (cullDist <= prevAL.radius && dist >= 0.05) {
                vec3 alDir = toLight / max(dist, 0.001);

                if (dot(normal, alDir) > -0.1) {
                    float invDist2 = 1.0 / max(dist2, 0.01);
                    float f2 = cullDist / prevAL.radius;
                    float window = max(1.0 - f2 * f2, 0.0);
                    window *= window;

                    float atten;
                    if (prevAL.halfExtent >= 0.01) {
                        float faceArea = cubeFaceArea(prevAL.halfExtent);
                        vec3 absDir = abs(alDir);
                        float cosLight = max(absDir.x, max(absDir.y, absDir.z));
                        atten = faceArea * cosLight * invDist2 * window;
                    } else {
                        atten = invDist2 * window;
                    }

                    vec3 prevUnshadowed = prevAL.color * prevAL.intensity * atten;
                    float prevTargetPdf = computeTargetPdf(prevUnshadowed);

                    if (prevTargetPdf > 0.0) {
                        prevRes.M = min(prevRes.M, float(pc.temporalMClamp / 2));
                        prevRes.unshadowed = prevUnshadowed;
                        prevRes.lightDir = alDir;
                        prevRes.lightDist = dist;
                        prevRes.lightIdx = prevLightIdx;
                        mergeReservoir(currentRes, prevRes, prevTargetPdf, mainRay.seed);
                    }
                }
            }
        }

        finalizeReservoir(currentRes);
        currentRes.W = min(currentRes.W, float(pc.wClamp));

        // --- Step 3: Shadow ray + shade selected light ---
        vec3 alAccum = vec3(0.0);
        if (currentRes.lightIdx >= 0) {
            AreaLight al = areaLightBuffer.lights[currentRes.lightIdx];

            vec2 cubeBN3 = vec2(-1.0); // A/B: PCG only
            CubeSample cs = sampleCubeLight(al, worldPos, pc.shadowSoftness, mainRay.seed, cubeBN3);
            vec3 toSample = cs.worldPos - worldPos;
            float sDist = length(toSample);
            vec3 sDir = toSample / sDist;
            vec3 shadowOrigin = offset_ray(worldPos, geometricNormal);

            shadowRay.radiance = vec3(0.0);
            shadowRay.throughput = vec3(0.0);
            shadowRay.seed = mainRay.seed;
            shadowRay.hitT = INF_DISTANCE;
            shadowRay.insideBoat = prGetInsideBoat(mainRay) ? 1u : 0u;
            shadowRay.bounceIndex = mainRay.index;

            float safeMargin = al.halfExtent + 0.02;
            float alTMax = sDist - safeMargin;
            if (alTMax > 0.01) {
                traceRayEXT(topLevelAS,
                    gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT,
                    WORLD_MASK, 0, 0, 3,
                    shadowOrigin, 0.0, sDir, alTMax, 1);
            } else {
                shadowRay.throughput = vec3(1.0);
            }

            float visibility = shadowRay.throughput.x;

            // Simplified Lambertian BRDF for indirect bounces (specular invisible behind denoiser)
            float NdotL = max(dot(normal, currentRes.lightDir), 0.0);
            vec3 brdf = NdotL / PI * mat.albedo;
            alAccum = currentRes.unshadowed * brdf * currentRes.W * visibility;
        }

        mainRay.radiance += alAccum * mainRay.throughput;

        // Store reservoir for next frame's temporal reuse
        storeBounceReservoir(mainRay.index, pixel, packReservoir(currentRes));

    } else if (AREA_LIGHTS_ON && pc.areaLightCount > 0 && mainRay.index > 0) {
        // Fallback when ReSTIR disabled: original unshadowed 8-light accumulation
        int bounceSearchCount = pc.areaLightCount;
        vec3 alAccum = vec3(0.0);

        for (int i = 0; i < bounceSearchCount; i++) {
            AreaLight al = areaLightBuffer.lights[i];

            vec3 toLight = al.position - worldPos;
            float dist2 = dot(toLight, toLight);
            float dist = sqrt(dist2);
            vec3 alDir = toLight / max(dist, 0.001);

            vec3 absToLight = abs(toLight);
            float chebyDist = max(absToLight.x, max(absToLight.y, absToLight.z));
            float cullDist = chebyDist;
            if (cullDist > al.radius) continue;

            float NdotL = dot(normal, alDir);
            if (NdotL < -0.1 || dist < 0.05) continue;

            float invDist2 = 1.0 / max(dist2, 0.01);
            float f = cullDist / al.radius;
            float window = max(1.0 - f * f, 0.0);
            window *= window;

            float atten;
            if (al.halfExtent >= 0.01) {
                float faceArea = cubeFaceArea(al.halfExtent);
                vec3 absDir = abs(alDir);
                float cosLight = max(absDir.x, max(absDir.y, absDir.z));
                atten = faceArea * cosLight * invDist2 * window;
            } else {
                atten = invDist2 * window;
            }
            if (atten < 0.00001) continue;

            vec3 unshadowed = al.color * al.intensity * atten;

            vec3 brdf;
            if (RESTIR_SIMPLIFIED_BRDF) {
                float NdotL_b = max(dot(normal, alDir), 0.0);
                brdf = NdotL_b / PI * mat.albedo;
            } else {
                float pdf;
                brdf = DisneyEval(mat, viewDir, normal, alDir, pdf, pc.flags);
            }

            alAccum += unshadowed * brdf;
        }

        mainRay.radiance += alAccum * mainRay.throughput;
    }

    } // !isPerfectSpecular

    mainRay.instanceIndex = instanceID;
    mainRay.geometryIndex = geometryID;
    mainRay.primitiveIndex = gl_PrimitiveID;
    mainRay.baryCoords = baryCoords;
    mainRay.worldPos = worldPos;
    mainRay.normal = normal;
    mainRay.albedoValue = albedoValue;

    // PSR mirror/glass continuation: if roughness is near-zero, trace deterministically
    // instead of stochastic BSDF sampling. The rgen PSR loop will continue tracing.
    if (mat.roughness < 0.001) {
        if (mat.transmission > 0.0) {
            // PSR glass: deterministic Snell refraction for transmissive materials
            bool entering = dot(gl_WorldRayDirectionEXT, geometricNormal) < 0.0;
            float eta = entering ? (1.0 / mat.ior) : mat.ior;
            vec3 faceNormal = entering ? normal : -normal;
            vec3 refractDir = refract(gl_WorldRayDirectionEXT, faceNormal, eta);

            // Fresnel reflectance (Schlick) to determine reflect vs refract
            float cosTheta = max(dot(viewDir, faceNormal), 0.0);
            float r0 = ((1.0 - mat.ior) / (1.0 + mat.ior));
            r0 = r0 * r0;
            float fresnelReflect = r0 + (1.0 - r0) * pow(1.0 - cosTheta, 5.0);
            // Blend reflection amount by (1 - transmission): full transmission → mostly refract
            fresnelReflect *= (1.0 - mat.transmission);

            if (length(refractDir) < 0.001 || fresnelReflect > 0.999) {
                // Total internal reflection fallback
                vec3 reflectDir = reflect(gl_WorldRayDirectionEXT, faceNormal);
                vec3 bounceOffsetN = dot(reflectDir, geometricNormal) > 0.0 ? geometricNormal : -geometricNormal;
                mainRay.origin = offset_ray(worldPos, bounceOffsetN);
                mainRay.direction = reflectDir;
                mainRay.throughput *= mat.f0;
                prSetLobeType(mainRay, 1u); // specular
            } else {
                // Refraction path
                vec3 bounceOffsetN = dot(refractDir, geometricNormal) > 0.0 ? geometricNormal : -geometricNormal;
                mainRay.origin = offset_ray(worldPos, bounceOffsetN);
                mainRay.direction = refractDir;
                mainRay.throughput *= (1.0 - fresnelReflect);
                prSetLobeType(mainRay, 2u); // transmission
            }
            // Clear noisy and stop (PSR continuation)
            mainRay.flags &= ~(PR_NOISY_BIT | PR_STOP_BIT);
            return;
        } else {
            // PSR mirror: opaque specular reflection
            vec3 reflectDir = reflect(gl_WorldRayDirectionEXT, normal);
            vec3 bounceOffsetN = dot(reflectDir, geometricNormal) > 0.0 ? geometricNormal : -geometricNormal;
            mainRay.origin = offset_ray(worldPos, bounceOffsetN);
            mainRay.direction = reflectDir;

            // Fresnel reflectance (Schlick) — preserves mirror tint for metals
            float cosTheta = max(dot(viewDir, normal), 0.0);
            vec3 fresnel = mat.f0 + (1.0 - mat.f0) * pow(1.0 - cosTheta, 5.0);
            mainRay.throughput *= fresnel;

            // Clear noisy and stop, set lobeType=specular
            mainRay.flags &= ~(PR_NOISY_BIT | PR_STOP_BIT);
            prSetLobeType(mainRay, 1u);
            return;
        }
    }

    // sample next direction using Disney BSDF
    // First bounce: Owen-scrambled Sobol for lobe selection (dim 76) + GGX VNDF (dims 77-78)
    vec3 sampleDir;
    float pdf;
    uint lobeType;
    vec3 bsdfXi = vec3(-1.0); // A/B: PCG only
    vec3 bsdf = DisneySample(mat, viewDir, normal, sampleDir, pdf, mainRay.seed, lobeType, pc.flags, bsdfXi);

    prSetLobeType(mainRay, lobeType);
    mainRay.flags |= PR_NOISY_BIT;

    // early exit if sampling failed (check BEFORE updating throughput to avoid amplification)
    if (pdf <= 1e-6) {
        mainRay.flags |= PR_STOP_BIT;
        return;
    }

    // Prevent light leaks: perturbed normals can sample directions below the
    // geometric surface plane. Kill these for non-transmissive lobes (diffuse/specular).
    // Transmission (lobeType 2: glass/water) still needs through-surface rays.
    if (lobeType != 2 && dot(sampleDir, geometricNormal) <= 0.0) {
        mainRay.flags |= PR_STOP_BIT;
        return;
    }

    mainRay.throughput *= bsdf / max(pdf, 1e-4);

    // AO modulation: darken indirect lighting in occluded areas (LabPBR AO + POM AO)
    mainRay.throughput *= mat.ao;

    vec3 bounceOffsetN = dot(sampleDir, geometricNormal) > 0.0 ? geometricNormal : -geometricNormal;
    mainRay.origin = offset_ray(worldPos, bounceOffsetN);

    mainRay.direction = sampleDir;
    mainRay.flags &= ~PR_STOP_BIT;
}
