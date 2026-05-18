#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#include "../../common/mapping.hpp"
#include "../../common/shared.hpp"
#include "../util/ray_payloads.glsl"
#include "../util/sprite_fetch.glsl"

layout(set = 0, binding = 0) uniform sampler2D textures[];

layout(set = 1, binding = 0) uniform accelerationStructureEXT topLevelAS;

layout(set = 1, binding = 1) readonly buffer BLASOffsets {
    uint offsets[];
}
blasOffsets;

layout(set = 1, binding = 2) readonly buffer VertexBufferAddr {
    uint64_t addrs[];
}
vertexBufferAddrs;

layout(set = 1, binding = 7) readonly buffer TextureMappingBuffer {
    TextureMapping textureMapping;
};

layout(set = 2, binding = 0) uniform WorldUniform {
    WorldUBO worldUBO;
};

layout(std430, buffer_reference, buffer_reference_align = 16) readonly buffer FaceDataBuffer {
    DisplacedFaceData faces[];
};

layout(std430, buffer_reference, buffer_reference_align = 16) readonly buffer MaterialClassBufferRef {
    MaterialClassMapping materialClassMapping;
};

layout(push_constant) uniform PushConstant {
    int numRayBounces;
    int flags;
    int areaLightCount;
    float shadowSoftness;
    int risCandidates;
    int temporalMClamp;
    int wClamp;
    float preExposure;
    float pomHeightScale;
    int   pomSteps;
    int   pomRefinement;
    float pomFadeDistance;
    float colorExpansion;
    uint blueNoiseFrame;
    uint64_t _sharcBDA0, _sharcBDA1, _sharcBDA2;
    float _sp0, _sp1, _sp2, _sp3;
    uint _sp4;
    float _sp5;
    uint _sp6;
    float _sp7;
    int _sp8, _sp9;
    int offlineFlags;
    int accumFrameCount;
    float aperture;
    float focalDistance;
    uint64_t materialClassAddr;
    uint64_t displacedFaceDataAddr;
    int displacedFaceCount;
    int displacementQuality;
} pc;

// Face normal vectors indexed by faceAxis
const vec3 FACE_NORMALS[6] = vec3[](
    vec3(1,0,0), vec3(-1,0,0), vec3(0,1,0), vec3(0,-1,0), vec3(0,0,1), vec3(0,0,-1));

layout(location = 0) rayPayloadInEXT PrimaryRay mainRay;
layout(location = 1) rayPayloadEXT ShadowRay shadowRay;
hitAttributeEXT vec3 hitAttribs;

const vec3 BT2020_LUM = vec3(0.2627, 0.6780, 0.0593);

float sampleDisplacementHeight(DisplacedFaceData face, vec2 uv, bool textureArray, bool hasLabPBR) {
    vec2 cUV = clamp(uv, face.uvMin, face.uvMax);
    if (hasLabPBR) {
        if (textureArray) return fetchBlockNormalLod(face.textureID, cUV, 0.0).a;
        if (face.normalTexID >= 0) return textureLod(textures[nonuniformEXT(uint(face.normalTexID))], cUV, 0).a;
        return 1.0;
    }

    vec4 s = textureArray
        ? fetchBlockAlbedoLod(face.textureID, cUV, worldUBO.animTick, 0.0)
        : textureLod(textures[nonuniformEXT(face.textureID)], cUV, 0);
    int heightSource = int((face.pomPacked0 >> 6u) & 0x7u);
    float raw;
    if (heightSource == 1) raw = s.r;
    else if (heightSource == 2) raw = s.g;
    else if (heightSource == 3) raw = s.b;
    else if (heightSource == 4) raw = s.a;
    else if (heightSource == 5) raw = max(s.r, max(s.g, s.b));
    else if (heightSource == 6) raw = min(s.r, min(s.g, s.b));
    else raw = dot(s.rgb, BT2020_LUM);

    float heightContrast = float((face.pomPacked1 >> 24u) & 0xFFu) / 10.0;
    float remapMin = float((face.pomPacked2 >> 0u) & 0xFFu) / 100.0;
    float remapMax = float((face.pomPacked2 >> 8u) & 0xFFu) / 100.0;
    float hOffset = (float((face.pomPacked2 >> 16u) & 0xFFu) - 100.0) / 100.0;
    bool invertH = ((face.flags >> 6u) & 1u) != 0u;

    float lumSpan = face.lumMax - face.lumMin;
    float h = (lumSpan > 1e-6) ? clamp((raw - face.lumMin) / lumSpan, 0.0, 1.0) : raw;
    if (invertH) h = 1.0 - h;
    h = mix(remapMin, remapMax, h);
    if (heightContrast != 1.0) h = pow(clamp(h, 0.001, 1.0), heightContrast);
    return clamp(h + hOffset, 0.0, 1.0);
}

void main() {
    uint instanceID = gl_InstanceID;
    uint geometryID = gl_GeometryIndexEXT;
    uint faceIdx = gl_PrimitiveID;

    uint blasOffset = blasOffsets.offsets[instanceID];
    uint64_t faceDataAddr = vertexBufferAddrs.addrs[blasOffset + geometryID];
    if (faceDataAddr == 0) return;

    FaceDataBuffer faceDataBuf = FaceDataBuffer(faceDataAddr);
    DisplacedFaceData face = faceDataBuf.faces[faceIdx];

    // Texture UV from intersection shader hit attributes
    vec2 texUV = hitAttribs.xy;
    vec2 uvMin = face.uvMin;
    vec2 uvMax = face.uvMax;
    texUV = clamp(texUV, uvMin, uvMax);

    bool textureArray = (face.properties & TEX_PROP_TEXTURE_ARRAY) != 0u;
    bool hasLabPBR = (face.properties & TEX_PROP_HAS_HEIGHT_MAP) != 0u &&
                     (textureArray || face.normalTexID >= 0);

    // Sample albedo
    vec4 albedo = textureArray
        ? fetchBlockAlbedoLod(face.textureID, texUV, worldUBO.animTick, 0.0)
        : textureLod(textures[nonuniformEXT(face.textureID)], texUV, 0);
    vec3 albedoLinear = textureArray ? albedo.rgb : pow(albedo.rgb, vec3(2.2));

    // Face normal (flat — no displaced normal derivation yet, will add central differences later)
    vec3 geomNormal = FACE_NORMALS[face.faceAxis];

    // Compute displaced normal via central differences on height field
    vec3 normal = geomNormal;
    {
        vec2 texSize = textureArray
            ? vec2(textureSize(blockAlbedo, 0).xy)
            : vec2(textureSize(textures[nonuniformEXT(face.textureID)], 0));
        vec2 dU = vec2(1.0 / texSize.x, 0.0);
        vec2 dV = vec2(0.0, 1.0 / texSize.y);

        float hC = sampleDisplacementHeight(face, texUV, textureArray, hasLabPBR);
        float hR = sampleDisplacementHeight(face, texUV + dU, textureArray, hasLabPBR);
        float hU = sampleDisplacementHeight(face, texUV + dV, textureArray, hasLabPBR);

        // Edge fade: flatten normals near face edges to match displacement fade.
        // Width formula matches displaced_block.rint exactly.
        vec2 uvSpan = uvMax - uvMin;
        float texelsU = max(texSize.x * uvSpan.x, 1.0);
        float texelsV = max(texSize.y * uvSpan.y, 1.0);
        float edgeWidthU = min(2.0, max(1.0, face.heightScale * texelsU)) / texelsU;
        float edgeWidthV = min(2.0, max(1.0, face.heightScale * texelsV)) / texelsV;
        // Recover parametric u,v from texture UV
        vec2 param = (texUV - uvMin) / max(uvSpan, vec2(1e-6));
        uint fadeMask = face.fadeEdgeMask;
        float edgeFade = 1.0;
        if ((fadeMask & 0x1u) != 0u) edgeFade *= smoothstep(0.0, edgeWidthU, param.x);
        if ((fadeMask & 0x2u) != 0u) edgeFade *= smoothstep(0.0, edgeWidthU, 1.0 - param.x);
        if ((fadeMask & 0x4u) != 0u) edgeFade *= smoothstep(0.0, edgeWidthV, param.y);
        if ((fadeMask & 0x8u) != 0u) edgeFade *= smoothstep(0.0, edgeWidthV, 1.0 - param.y);

        // World-space edge vectors per texel
        vec3 dpdu = face.edgeU / max(texSize.x * uvSpan.x, 1.0);
        vec3 dpdv = face.edgeV / max(texSize.y * uvSpan.y, 1.0);

        float scale = face.heightScale * edgeFade;
        vec3 tangent = dpdu + geomNormal * (hR - hC) * (-scale);
        vec3 bitangent = dpdv + geomNormal * (hU - hC) * (-scale);
        normal = normalize(cross(tangent, bitangent));

        // Ensure normal faces the incoming ray
        if (dot(normal, gl_WorldRayDirectionEXT) > 0.0) normal = -normal;
    }

    // Material class lookup
    float roughness = 0.5;
    float metallic = 0.0;
    vec3 f0 = vec3(0.04);

    uint matType = (face.emissiveBlockType >> 8u) & 0xFFu;
    if (matType > 0u && pc.materialClassAddr != 0) {
        MaterialClassBufferRef matBuf = MaterialClassBufferRef(pc.materialClassAddr);
        MaterialClassEntry mc = matBuf.materialClassMapping.entries[matType - 1u];
        roughness = mc.roughness * mc.roughness; // perceptual → GGX alpha
        metallic = mc.metallic;
        f0 = mc.f0;
    }

    // Apply color layer
    vec3 finalAlbedo = albedoLinear;
    if (face.colorLayer.a > 0.001) {
        finalAlbedo *= face.colorLayer.rgb;
    }

    // World-space hit position
    vec3 worldPos = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;

    // Write primary ray payload (simplified v1 — no area lights, no ReSTIR, no emission)
    mainRay.worldPos = worldPos;
    mainRay.normal = normal;
    mainRay.albedoValue = vec4(finalAlbedo, 1.0);
    mainRay.roughness = roughness;
    mainRay.metallic = metallic;
    mainRay.f0 = f0;
    mainRay.hitT = gl_HitTEXT;
    mainRay.albedoEmission = 0.0;
    // Set cont=1 (continuation), clear stop/noisy, emBlockType=255, preserve isHand/insideBoat
    mainRay.flags = (mainRay.flags & (PR_ISHAND_BIT | PR_INSIDEBOAT_BIT)) | PR_CONT_BIT | (255u << PR_EMBLOCK_SHIFT);

    // Throughput: Lambertian approximation for indirect bounces
    float NdotV = max(dot(normal, -gl_WorldRayDirectionEXT), 0.0);
    mainRay.throughput *= finalAlbedo * NdotV;
}
