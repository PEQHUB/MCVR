#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#include "util/ray_cone.glsl"
#include "util/ray.glsl"
#include "util/color_space.glsl"
#include "common/shared.hpp"

layout(set = 0, binding = 0) uniform sampler2D textures[];

layout(set = 1, binding = 1) readonly buffer BLASOffsets {
    uint offsets[];
}
blasOffsets;

layout(set = 1, binding = 2) readonly buffer IndexBufferAddr {
    uint64_t addrs[];
}
indexBufferAddrs;

layout(set = 1, binding = 3) readonly buffer LastIndexBufferAddr {
    uint64_t addrs[];
}
lastIndexBufferAddrs;

layout(set = 1, binding = 8) readonly buffer LastObjToWorldMat {
    mat4 mat[];
}
lastObjToWorldMats;

layout(set = 1, binding = 7) readonly buffer TextureMappingBuffer {
    TextureMapping mapping;
};

layout(set = 1, binding = 6) readonly buffer LastPositionBufferAddr {
    uint64_t addrs[];
}
lastPositionBufferAddrs;

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer IndexBuffer {
    uint indices[];
}
indexBuffer;

#include "util/vertex.glsl"

layout(location = 0) rayPayloadInEXT MainRay mainRay;
hitAttributeEXT vec2 attribs;

bool loadPreviousScenePos(uint geometryBufferIndex, uint primitiveID, vec3 baryCoords, out vec3 prevScenePos) {
    uint64_t lastPositionAddr = lastPositionBufferAddrs.addrs[geometryBufferIndex];
    uint64_t lastIndexAddr = lastIndexBufferAddrs.addrs[geometryBufferIndex];
    if (lastPositionAddr == 0 || lastIndexAddr == 0) { return false; }

    IndexBuffer lastIndexBuffer = IndexBuffer(lastIndexAddr);
    uint indexBaseID = 3u * primitiveID;
    uint i0 = lastIndexBuffer.indices[indexBaseID];
    uint i1 = lastIndexBuffer.indices[indexBaseID + 1u];
    uint i2 = lastIndexBuffer.indices[indexBaseID + 2u];

    PositionBuffer lastPositionBuffer = PositionBuffer(lastPositionAddr);
    vec3 p0 = lastPositionBuffer.vertices[i0].pos;
    vec3 p1 = lastPositionBuffer.vertices[i1].pos;
    vec3 p2 = lastPositionBuffer.vertices[i2].pos;
    vec3 prevLocalPos = baryCoords.x * p0 + baryCoords.y * p1 + baryCoords.z * p2;

    mat4 lastModelMat = lastObjToWorldMats.mat[gl_InstanceCustomIndexEXT];
    prevScenePos = mat3(lastModelMat) * prevLocalPos + lastModelMat[3].xyz;
    return true;
}

void main() {
    uint instanceID = gl_InstanceCustomIndexEXT;
    uint geometryID = gl_GeometryIndexEXT;
    uint geometryBufferIndex = getGeometryBufferIndex(instanceID, geometryID);

    uint i0, i1, i2;
    PositionVertex p0, p1, p2;
    MaterialVertex m0, m1, m2;
    loadTriangle(geometryBufferIndex, gl_PrimitiveID, i0, i1, i2, p0, p1, p2, m0, m1, m2);

    vec3 bary = vec3(1.0 - (attribs.x + attribs.y), attribs.x, attribs.y);

    vec4 colorLayer = hasColorLayer(m0.packedData) ?
                          (bary.x * m0.colorLayer + bary.y * m1.colorLayer + bary.z * m2.colorLayer) :
                          vec4(1.0);

    vec4 albedo = vec4(1.0);
    float pbrEmission = 0.0;
    if (hasTexture(m0.packedData)) {
        vec2 uv = bary.x * m0.textureUV + bary.y * m1.textureUV + bary.z * m2.textureUV;
        albedo = sampleTexture(textures[nonuniformEXT(m0.textureID)], uv, 0.0, false);

        int specularTextureID = mapping.entries[m0.textureID].specular;
        if (specularTextureID >= 0) {
            vec4 specular = sampleTexture(textures[nonuniformEXT(specularTextureID)], uv, 0.0, false);
            int intEmission = int(round(specular.a * 255.0));
            if (intEmission != 255) { pbrEmission = intEmission / 254.0; }
        }
    }

    vec4 shaded = albedo * colorLayer;
    float alpha = clamp(shaded.a, 0.0, 1.0);
    vec3 shadedRgb = clamp(shaded.rgb, vec3(0.0), vec3(1.0));
    vec3 transmittance = vec3(clamp(albedo.a, 0.0, 1.0));

    float factor = rayBounce(mainRay) == 0u ? VPT_DIRECT_LIGHT_STRENGTH : VPT_INDIRECT_LIGHT_STRENGTH;
    mainRay.radiance += factor * shadedRgb * alpha * pbrEmission * mainRay.throughput;
    mainRay.throughput *= transmittance;

    vec3 worldPos = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
    vec3 geoNormalObj = normalize(cross(p1.pos - p0.pos, p2.pos - p0.pos));
    mat3 normalMatrix = transpose(mat3(gl_WorldToObject3x4EXT));
    vec3 normal = normalize(normalMatrix * geoNormalObj);
    if (dot(normal, -mainRay.direction) < 0.0) { normal = -normal; }
    float defaultF0 = 0.02;
    float sqrtF0 = sqrt(defaultF0);
    float ior = (1.0 + sqrtF0) / max(1.0 - sqrtF0, 1e-6);
    float payloadTransmission = alpha < 0.999999 ? 1.0 : 0.0;

    mainRay.normal = normal;
    rayStoreMaterial(mainRay, vec4(shadedRgb, alpha), vec3(defaultF0), 1.0, 0.0, payloadTransmission, ior, 0.0);
    raySetLobeType(mainRay, 0u);
    raySetNoisy(mainRay, false);
    mainRay.directLightRadiance = vec3(0.0);
    mainRay.hasPrevScenePos = 0u;
    if (rayBounce(mainRay) == 0u) {
        vec3 prevScenePos;
        if (loadPreviousScenePos(geometryBufferIndex, gl_PrimitiveID, bary, prevScenePos)) {
            mainRay.prevScenePos = prevScenePos;
            mainRay.hasPrevScenePos = 1u;
        }
    }

    mainRay.origin = worldPos + mainRay.direction * 0.001;
    mainRay.hitT = gl_HitTEXT;
    mainRay.coneWidth += gl_HitTEXT * mainRay.coneSpread;
    raySetContinue(mainRay, true);
    raySetStop(mainRay, false);
}
