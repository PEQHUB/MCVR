#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#include "../../common/mapping.hpp"
#include "../../common/shared.hpp"
#include "../util/ray_payloads.glsl"

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
    uint _pad0;
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

    // Sample albedo
    vec4 albedo = textureLod(textures[nonuniformEXT(face.textureID)], texUV, 0);
    vec3 albedoLinear = pow(albedo.rgb, vec3(2.2)); // sRGB → linear

    // Face normal (flat — no displaced normal derivation yet, will add central differences later)
    vec3 geomNormal = FACE_NORMALS[face.faceAxis];

    // Compute displaced normal via central differences on height field
    // This is a simplified approximation using texel neighbors
    vec3 normal = geomNormal;
    {
        vec2 texSize = vec2(textureSize(textures[nonuniformEXT(face.textureID)], 0));
        vec2 dU = vec2(1.0 / texSize.x, 0.0);
        vec2 dV = vec2(0.0, 1.0 / texSize.y);

        float hC = textureLod(textures[nonuniformEXT(face.textureID)], texUV, 0).r;
        float hR = textureLod(textures[nonuniformEXT(face.textureID)], clamp(texUV + dU, uvMin, uvMax), 0).r;
        float hU = textureLod(textures[nonuniformEXT(face.textureID)], clamp(texUV + dV, uvMin, uvMax), 0).r;

        // World-space edge vectors per texel
        vec3 dpdu = face.edgeU / max(texSize.x * (uvMax.x - uvMin.x), 1.0);
        vec3 dpdv = face.edgeV / max(texSize.y * (uvMax.y - uvMin.y), 1.0);

        float scale = face.heightScale;
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
    mainRay.stop = 0u;
    mainRay.cont = 1u;
    mainRay.noisy = 0u;
    mainRay.albedoEmission = 0.0;
    mainRay.emBlockTypeOut = 255u;

    // Throughput: Lambertian approximation for indirect bounces
    float NdotV = max(dot(normal, -gl_WorldRayDirectionEXT), 0.0);
    mainRay.throughput *= finalAlbedo * NdotV;
}
