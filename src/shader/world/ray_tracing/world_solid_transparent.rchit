#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_opacity_micromap : require

#include "../util/disney.glsl"
#include "../util/random.glsl"
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

layout(set = 1, binding = 10) readonly buffer BlenderPBRMappingBuffer {
    BlenderPBRMapping blenderPBR;
};

layout(set = 1, binding = 8) readonly buffer AreaLightBuffer {
    AreaLight lights[];
} areaLightBuffer;

layout(set = 1, binding = 9) readonly buffer TileLightBuffer {
    uint data[];
} tileLightBuffer;

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

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer VertexBuffer {
    PBRTriangle vertices[];
}
vertexBuffer;

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer IndexBuffer {
    uint indices[];
}
indexBuffer;

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
    uint _pad0;                 // alignment padding (offset 52)
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

layout(set = 3, binding = 3, rgba16f) uniform readonly image2D normalRoughnessImage;
layout(set = 3, binding = 4, rg16f) uniform readonly image2D motionVectorImage;
layout(set = 3, binding = 5, r16f) uniform readonly image2D linearDepthImage;
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

    uint blasOffset = blasOffsets.offsets[instanceID];

    IndexBuffer indexBuffer = IndexBuffer(indexBufferAddrs.addrs[blasOffset + geometryID]);
    uint indexBaseID = 3 * gl_PrimitiveID;
    uint i0 = indexBuffer.indices[indexBaseID];
    uint i1 = indexBuffer.indices[indexBaseID + 1];
    uint i2 = indexBuffer.indices[indexBaseID + 2];

    VertexBuffer vertexBuffer = VertexBuffer(vertexBufferAddrs.addrs[blasOffset + geometryID]);
    PBRTriangle v0 = vertexBuffer.vertices[i0];
    PBRTriangle v1 = vertexBuffer.vertices[i1];
    PBRTriangle v2 = vertexBuffer.vertices[i2];

    vec3 baryCoords = vec3(1.0 - (attribs.x + attribs.y), attribs.x, attribs.y);
    vec3 localPos = baryCoords.x * v0.pos + baryCoords.y * v1.pos + baryCoords.z * v2.pos;
    vec3 worldPos = vec4(localPos, 1.0) * gl_ObjectToWorld3x4EXT;

    uint useColorLayer = v0.useColorLayer;
    vec3 colorLayer;
    if (useColorLayer > 0) {
        colorLayer = (baryCoords.x * v0.colorLayer + baryCoords.y * v1.colorLayer + baryCoords.z * v2.colorLayer).rgb;
    } else {
        colorLayer = vec3(1.0);
    }

    uint useTexture = v0.useTexture;
    float albedoEmission =
        baryCoords.x * v0.albedoEmission + baryCoords.y * v1.albedoEmission + baryCoords.z * v2.albedoEmission;
    uint textureID = v0.textureID;
    int specularTextureID = mapping.entries[textureID].specular;
    int normalTextureID = mapping.entries[textureID].normal;
    int flagTextureID = mapping.entries[textureID].flag;
    vec4 albedoValue;
    vec4 specularValue;
    vec4 normalValue;
    ivec4 flagValue;
    vec2 textureUV;
    vec3 rawAlbedoLinear = vec3(1.0); // raw albedo before tinting, for texture roughness derivation
    if (useTexture > 0) {
        textureUV = baryCoords.x * v0.textureUV + baryCoords.y * v1.textureUV + baryCoords.z * v2.textureUV;

        // POM: offset UVs using height field (primary ray only, with normal texture + height data)
        // Only runs when global pomEnabled=true (pomHeightScale > 0 in push constant)
        if (pc.pomHeightScale > 0.0 && normalTextureID >= 0
            && (mapping.entries[textureID].properties & TEX_PROP_HAS_HEIGHT_MAP) != 0
            && mainRay.index == 0) {
            float pomDist = gl_HitTEXT;
            float pomFade = 1.0 - smoothstep(pc.pomFadeDistance * 0.5, pc.pomFadeDistance, pomDist);
            if (pomFade > 0.001) {
                // Tile boundaries from vertex UVs
                vec2 uvMin = min(min(v0.textureUV, v1.textureUV), v2.textureUV);
                vec2 uvMax = max(max(v0.textureUV, v1.textureUV), v2.textureUV);

                // TBN in object space → world space
                vec3 edge1 = v1.pos - v0.pos;
                vec3 edge2 = v2.pos - v0.pos;
                vec3 geoN = normalize(cross(edge1, edge2));
                vec2 duv1 = v1.textureUV - v0.textureUV;
                vec2 duv2 = v2.textureUV - v0.textureUV;
                float pomDet = duv1.x * duv2.y - duv2.x * duv1.y;
                vec3 tangentObj;
                if (abs(pomDet) < 1e-6) {
                    tangentObj = (abs(geoN.x) > 0.99) ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
                } else {
                    float f = 1.0 / pomDet;
                    tangentObj = vec3(
                        f * (duv2.y * edge1.x - duv1.y * edge2.x),
                        f * (duv2.y * edge1.y - duv1.y * edge2.y),
                        f * (duv2.y * edge1.z - duv1.y * edge2.z));
                }
                vec3 TObj = normalize(tangentObj - geoN * dot(geoN, tangentObj));
                vec3 BObj = cross(geoN, TObj) * sign(pomDet);
                mat3 nMat = transpose(mat3(gl_WorldToObject3x4EXT));
                vec3 T = normalize(nMat * TObj);
                vec3 B = normalize(nMat * BObj);
                vec3 N = normalize(nMat * geoN);

                // View direction in tangent space
                vec3 viewDirTS = vec3(dot(viewDir, T), dot(viewDir, B), dot(viewDir, N));

                int effectiveSteps = max(int(float(pc.pomSteps) * pomFade), 4);
                int effectiveRefinement = (pomFade > 0.5) ? pc.pomRefinement : 0;

                int bpHeightTex = blenderPBR.entries[textureID].heightTex;

                textureUV = parallaxOcclusionMapping(
                    textureUV, viewDirTS, normalTextureID,
                    pc.pomHeightScale, effectiveSteps, effectiveRefinement,
                    uvMin, uvMax, bpHeightTex);
            }
        }

        // ray cone
        float coneRadiusWorld = mainRay.coneWidth + gl_HitTEXT * mainRay.coneSpread;
        vec3 dposdu, dposdv;
        computedposduDv(v0.pos, v1.pos, v2.pos, v0.textureUV, v1.textureUV, v2.textureUV, dposdu, dposdv);
        // lod still has issues, temporally disable
        float lod = 0; // lodWithCone(textures[nonuniformEXT(textureID)], textureUV, coneRadiusWorld, dposdu, dposdv);

        albedoValue = textureLod(textures[nonuniformEXT(textureID)], textureUV, lod);
        rawAlbedoLinear = albedoValue.rgb; // save before tinting for texture roughness derivation
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
    } else {
        albedoValue = vec4(1.0);
        specularValue = vec4(0.0);
        normalValue = vec4(0.0);
        flagValue = ivec4(0);
    }

    vec3 glint = vec3(0.0);
    vec4 overlayColor = vec4(0.0);
    if (!SIMPLIFIED_INDIRECT || mainRay.index <= 1) {
        uint useGlint = v0.useGlint;
        uint glintTexture = v0.glintTexture;
        vec2 glintUV = baryCoords.x * v0.glintUV + baryCoords.y * v1.glintUV + baryCoords.z * v2.glintUV;
        glintUV = (worldUbo.textureMat * vec4(glintUV, 0.0, 1.0)).xy;
        glint = useGlint * texture(textures[nonuniformEXT(glintTexture)], glintUV).rgb;
        glint = glint * glint;

        uint useOverlay = v0.useOverlay;
        ivec2 overlayUV = v0.overlayUV;
        overlayColor = texelFetch(textures[nonuniformEXT(worldUbo.overlayTextureID)], overlayUV, 0);
    }

    vec3 tint;
    if (v0.useOverlay > 0) {
        tint = mix(overlayColor.rgb, albedoValue.rgb * colorLayer, overlayColor.a) + glint;
    } else {
        tint = albedoValue.rgb * colorLayer + glint;
    }

    tint = CS_BT709_TO_BT2020 * tint;  // BT.709 -> BT.2020 working space

    albedoValue = vec4(tint, albedoValue.a);
    LabPBRMat mat = convertLabPBRMaterial(albedoValue, specularValue, normalValue);

    // Blender PBR per-channel overlay: sample individual textures if present
    int texProps = mapping.entries[textureID].properties;
    bool hasBlenderNormal = false;
    if ((texProps & TEX_PROP_DIRECT_PBR) != 0 && useTexture > 0) {
        int rTex  = blenderPBR.entries[textureID].roughnessTex;
        int mTex  = blenderPBR.entries[textureID].metallicTex;
        int eTex  = blenderPBR.entries[textureID].emissionTex;
        int nTex  = blenderPBR.entries[textureID].normalBPTex;
        int hTex  = blenderPBR.entries[textureID].heightTex;
        int aeTex = blenderPBR.entries[textureID].aoTex;
        int xTex  = blenderPBR.entries[textureID].extraTex;

        float bpR  = (rTex  >= 0) ? textureLod(textures[nonuniformEXT(rTex)],  textureUV, 0).r : -1.0;
        float bpM  = (mTex  >= 0) ? textureLod(textures[nonuniformEXT(mTex)],  textureUV, 0).r : -1.0;
        float bpE  = (eTex  >= 0) ? textureLod(textures[nonuniformEXT(eTex)],  textureUV, 0).r : -1.0;
        vec2  bpN  = (nTex  >= 0) ? textureLod(textures[nonuniformEXT(nTex)],  textureUV, 0).rg : vec2(-1.0);
        float bpH  = (hTex  >= 0) ? textureLod(textures[nonuniformEXT(hTex)],  textureUV, 0).r : -1.0;
        float bpAO = (aeTex >= 0) ? textureLod(textures[nonuniformEXT(aeTex)], textureUV, 0).r : -1.0;
        vec4  bpX  = (xTex  >= 0) ? textureLod(textures[nonuniformEXT(xTex)],  textureUV, 0)   : vec4(-1.0);

        mat = overlayDirectPBR(albedoValue, mat, bpR, bpM, bpE, bpN, bpH, bpAO, bpX);
        hasBlenderNormal = (nTex >= 0);
    }

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
    if (materialType > 0u && materialType <= 160u) {
        uint idx = materialType - 1u;
        vec4 pack0 = worldUbo.materialData[idx];          // F0.rgb, roughness
        vec4 pack1 = worldUbo.materialData[idx + 160u];   // metallic, transmission, ior, subsurface
        vec4 pack2 = worldUbo.materialData[idx + 320u];   // anisotropic, sheenWeight, sheenTint, coatWeight
        vec4 pack3 = worldUbo.materialData[idx + 480u];   // coatRoughness, noiseScale, noiseStrength, noiseOctaves
        vec4 pack4 = worldUbo.materialData[idx + 640u];   // channelR, channelG, channelB, textureBlend
        vec4 pack5 = worldUbo.materialData[idx + 800u];   // gamutBoost, noiseMaskThreshold, noiseMaskPacked, normalStrength
        vec4 pack6 = worldUbo.materialData[idx + 960u];   // noiseRotation, noiseAspect, noiseLacunarity, noiseContrast

        {
            // Apply F0 override if set; for dielectrics with zero F0, derive from IOR
            if (dot(pack0.rgb, pack0.rgb) > 0.0001) {
                mat.f0 = pack0.rgb;
            } else if (pack1.x < 0.5) {
                // Dielectric: compute F0 from IOR using Fresnel equation
                float ior = max(pack1.z, 1.0);
                float f0 = ((ior - 1.0) * (ior - 1.0)) / ((ior + 1.0) * (ior + 1.0));
                mat.f0 = vec3(f0);
            }
            float matRoughness = pack0.a * pack0.a;  // perceptual → GGX alpha

            // Tex Roughness: blend between slider roughness and AutoPBR/LabPBR per-pixel roughness
            // AutoPBR controls (gamma, variance, edge, min/max) shape texSourceRoughness
            float textureBlend = pack4.w;
            if (textureBlend > 0.001) {
                // Use AutoPBR roughness from specular texture if available,
                // fall back to albedo-channel derivation if no specular texture
                float texRoughness;
                if (specularTextureID >= 0) {
                    texRoughness = texSourceRoughness;  // AutoPBR/resource pack roughness
                } else {
                    float weightSum = pack4.x + pack4.y + pack4.z;
                    float signal = dot(rawAlbedoLinear, pack4.xyz) / max(weightSum, 0.001);
                    texRoughness = (1.0 - signal) * (1.0 - signal);
                }
                mat.roughness = max(mix(matRoughness, texRoughness, textureBlend), 0.01);
            } else if (specularTextureID < 0 || mat.transmission > 0.0) {
                // No specular texture, OR transmissive block → slider roughness
                mat.roughness = max(matRoughness, 0.01);
            }
            // else: opaque with specular texture → keep LabPBR/AutoPBR roughness from texture

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
            // pack3.w packs: octaves (bits 0-3) | noiseType (bits 4-8) | seed (bits 9-17) | noiseTarget (bits 20-23)
            int noisePacked = int(pack3.w);
            matNoiseOctaves = noisePacked & 0xF;
            matNoiseType = (noisePacked >> 4) & 0x1F;  // 5 bits = 0-31
            matNoiseSeed = (noisePacked >> 9) & 0x1FF;  // 9 bits = 0-511
            matNoiseTarget = (noisePacked >> 20) & 0xF;

            matGamutBoost = pack5.x;
            matNoiseMaskThreshold = pack5.y;
            int maskPacked = int(pack5.z);
            matNoiseMaskMode = maskPacked & 0x7;       // bits 0-2
            matNoiseMaskInvert = ((maskPacked >> 3) & 0x1) != 0; // bit 3
            matNoiseWrap = (maskPacked >> 4) & 0x7;    // bits 4-6

            // Normal strength: amplify/attenuate Auto-PBR/LabPBR normal map
            // 0 = no override (default), >0 = scale factor (1.0 = unchanged, 2.0 = 2x stronger)
            float matNormalStrength = pack5.w;
            if (matNormalStrength > 0.01 && matNormalStrength != 1.0 && length(mat.normal) > 0.01) {
                mat.normal.xy *= matNormalStrength;
                mat.normal = normalize(mat.normal);
            }

            matNoiseRotation = pack6.x;
            matNoiseAspect = pack6.y;
            matNoiseLacunarity = pack6.z;
            matNoiseContrast = pack6.w;

            if (mat.metallic > 0.5) {
                if (textureBlend > 0.001) {
                    // Textured metal: preserve texture detail in reflectance
                    // Modulate F0 by texture luminance variation
                    float texLum = dot(mat.albedo, vec3(0.2627, 0.6780, 0.0593));
                    float avgLum = max(texLum, 0.01); // avoid div-by-zero
                    vec3 texDetail = mat.albedo / avgLum; // normalized texture pattern
                    mat.albedo = mat.f0 * mix(vec3(1.0), texDetail, textureBlend);
                } else {
                    mat.albedo = mat.f0;  // flat metal: pure F0 color
                }
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
        vec3 lms2 = lms_g2 * lms_g2 * lms_g2;
        mat.albedo = max(lms_to_bt2020 * lms2, vec3(0.0));
    }

    // the provided normal is unreliable! (such as grass, etc.)
    // calculate on the fly for now
    vec3 geometricNormal;
    vec3 normal =
        calculateNormal(v0.pos, v1.pos, v2.pos, v0.textureUV, v1.textureUV, v2.textureUV, mat.normal, viewDir, geometricNormal, hasBlenderNormal);

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
                mainRay.specularValue = vec4(0.0);
                mainRay.normalValue = vec4(0.5, 0.5, 1.0, 1.0); // flat LabPBR normal (avoids NaN from vec4(0))
                mainRay.flagValue = ivec4(0);
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
                mainRay.noisy = 0;
                mainRay.lobeType = 0;
                mainRay.roughness = 1.0;
                mainRay.stop = 1;
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
    mainRay.emBlockTypeOut = emBlockType;
    vec3 emissionTint = tint; // default: texture albedo (BT.2020)
    if (emBlockType < 50u && albedoEmission > 0.0) {
        vec4 emData = worldUbo.emissionData[emBlockType];
        albedoEmission *= emData.a; // scalar multiplier
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
            vec3 lms2 = lms_g2 * lms_g2 * lms_g2;
            emissionTint = lms_to_bt2020 * lms2;
            float minC = min(emissionTint.r, min(emissionTint.g, emissionTint.b));
            if (minC < 0.0) {
                float luma = dot(emissionTint, vec3(0.2627, 0.6780, 0.0593));
                float t = luma / (luma - minC);
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
    float maskLum = dot(tint, vec3(0.2627, 0.6780, 0.0593));
    sceneEmission *= smoothstep(0.10, 0.40, maskLum);

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
    float kappa = PHYSICAL_SUN_DISK ? 46000.0 : ((mainRay.isHand > 0) ? 500.0 : 3000.0);
    if (sunDir.y < 0) { lightDir = normalize(skyUBO.moonDirection); }
    vec3 sampledLightDir = SampleVMF(mainRay.seed, lightDir, kappa);
    vec3 shadowBiasN = dot(sampledLightDir, geometricNormal) > 0.0 ? geometricNormal : -geometricNormal;
    vec3 shadowRayOrigin = offset_ray(worldPos, shadowBiasN);

    if (worldUbo.skyType == 1) {
        float pdf; // not used
        vec3 lightBRDF = DisneyEval(mat, viewDir, normal, sampledLightDir, pdf, pc.flags);

        shadowRay.radiance = vec3(0.0);
        shadowRay.throughput = vec3(1.0);
        shadowRay.seed = mainRay.seed;
        shadowRay.hitT = INF_DISTANCE;
        shadowRay.insideBoat = mainRay.insideBoat;
        shadowRay.bounceIndex = mainRay.index;
        shadowRay.mediumAbsorption = vec3(0.0);
        shadowRay.mediumEntryT = 0.0;

        uint shadowMask = WORLD_MASK;
        if (mainRay.isHand == 0) {
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
        if (mainRay.isHand > 0 && !NO_HAND_AMBIENT) {
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
                // Uniform random selection from available lights
                int tileSlot = int(rand(mainRay.seed) * float(effectiveCount));
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

                CubeSample cs = sampleCubeLight(al, worldPos, pc.shadowSoftness, mainRay.seed);
                vec3 toSample = cs.worldPos - worldPos;
                float sDist = length(toSample);
                vec3 sDir = toSample / sDist;
                vec3 shadowOrigin = offset_ray(worldPos, geometricNormal);

                shadowRay.radiance = vec3(0.0);
                shadowRay.throughput = vec3(0.0);
                shadowRay.seed = mainRay.seed;
                shadowRay.hitT = INF_DISTANCE;
                shadowRay.insideBoat = mainRay.insideBoat;
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
                    brdf = NdotL / 3.14159265 * mat.albedo;
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

                CubeSample cs = sampleCubeLight(al, worldPos, pc.shadowSoftness, mainRay.seed);
                vec3 toSample = cs.worldPos - worldPos;
                float sDist = length(toSample);
                vec3 sDir = toSample / sDist;
                vec3 shadowOrigin = offset_ray(worldPos, geometricNormal);

                shadowRay.radiance = vec3(0.0);
                shadowRay.throughput = vec3(0.0);
                shadowRay.seed = mainRay.seed;
                shadowRay.hitT = INF_DISTANCE;
                shadowRay.insideBoat = mainRay.insideBoat;
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
                    brdf = NdotL / 3.14159265 * mat.albedo;
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
            int idx = clamp(int(rand(mainRay.seed) * float(searchCount)), 0, searchCount - 1);
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

            CubeSample cs = sampleCubeLight(al, worldPos, pc.shadowSoftness, mainRay.seed);
            vec3 toSample = cs.worldPos - worldPos;
            float sDist = length(toSample);
            vec3 sDir = toSample / sDist;
            vec3 shadowOrigin = offset_ray(worldPos, geometricNormal);

            shadowRay.radiance = vec3(0.0);
            shadowRay.throughput = vec3(0.0);
            shadowRay.seed = mainRay.seed;
            shadowRay.hitT = INF_DISTANCE;
            shadowRay.insideBoat = mainRay.insideBoat;
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
            vec3 brdf = NdotL / 3.14159265 * mat.albedo;
            alAccum = currentRes.unshadowed * brdf * currentRes.W * visibility;
        }

        alAccum *= 1.0;
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
                brdf = NdotL_b / 3.14159265 * mat.albedo;
            } else {
                float pdf;
                brdf = DisneyEval(mat, viewDir, normal, alDir, pdf, pc.flags);
            }

            alAccum += unshadowed * brdf;
        }

        alAccum *= 1.0;
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
    mainRay.specularValue = specularValue;
    mainRay.normalValue = normalValue;
    mainRay.flagValue = flagValue;

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
                mainRay.lobeType = 1; // specular
            } else {
                // Refraction path
                vec3 bounceOffsetN = dot(refractDir, geometricNormal) > 0.0 ? geometricNormal : -geometricNormal;
                mainRay.origin = offset_ray(worldPos, bounceOffsetN);
                mainRay.direction = refractDir;
                mainRay.throughput *= (1.0 - fresnelReflect);
                mainRay.lobeType = 2; // transmission
            }
            mainRay.noisy = 0;
            mainRay.stop = 0;
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

            mainRay.noisy = 0;
            mainRay.lobeType = 1; // specular
            mainRay.stop = 0;
            return;
        }
    }

    // sample next direction using Disney BSDF
    vec3 sampleDir;
    float pdf;
    uint lobeType;
    vec3 bsdf = DisneySample(mat, viewDir, normal, sampleDir, pdf, mainRay.seed, lobeType, pc.flags);

    mainRay.lobeType = lobeType;
    mainRay.noisy = 1;

    // early exit if sampling failed (check BEFORE updating throughput to avoid amplification)
    if (pdf <= 1e-6) {
        mainRay.stop = 1;
        return;
    }

    // Prevent light leaks: perturbed normals can sample directions below the
    // geometric surface plane. Kill these for non-transmissive lobes (diffuse/specular).
    // Transmission (lobeType 2: glass/water) still needs through-surface rays.
    if (lobeType != 2 && dot(sampleDir, geometricNormal) <= 0.0) {
        mainRay.stop = 1;
        return;
    }

    mainRay.throughput *= bsdf / max(pdf, 1e-4);

    vec3 bounceOffsetN = dot(sampleDir, geometricNormal) > 0.0 ? geometricNormal : -geometricNormal;
    mainRay.origin = offset_ray(worldPos, bounceOffsetN);

    mainRay.direction = sampleDir;
    mainRay.stop = 0;
}
