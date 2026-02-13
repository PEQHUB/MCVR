#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#include "../util/disney.glsl"
#include "../util/random.glsl"
#include "../util/ray_payloads.glsl"
#include "../util/util.glsl"
#include "common/shared.hpp"

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

layout(set = 2, binding = 0) uniform WorldUniform {
    WorldUBO worldUbo;
};

layout(set = 2, binding = 1) uniform LastWorldUniform {
    WorldUBO lastWorldUbo;
};

layout(set = 2, binding = 2) uniform SkyUniform {
    SkyUBO skyUBO;
};

layout(set = 3, binding = 1, rgba8) uniform image2D diffuseAlbedoImage;
layout(set = 3, binding = 2, rgba8) uniform image2D specularAlbedoImage;
layout(set = 3, binding = 3, rgba16f) uniform image2D normalRoughnessImage;
layout(set = 3, binding = 4, rg16f) uniform image2D motionVectorImage;
layout(set = 3, binding = 5, r16f) uniform image2D linearDepthImage;

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer VertexBuffer {
    PBRTriangle vertices[];
}
vertexBuffer;

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer IndexBuffer {
    uint indices[];
}
indexBuffer;

layout(location = 0) rayPayloadInEXT PrimaryRay mainRay;
layout(location = 1) rayPayloadEXT ShadowRay shadowRay;
hitAttributeEXT vec2 attribs;

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
    uint coordinate = v0.coordinate;
    vec3 normal = baryCoords.x * v0.norm + baryCoords.y * v1.norm + baryCoords.z * v2.norm;
    if (coordinate == 1) {
        normal = normalize(mat3(worldUbo.cameraViewMatInv) * normal);
    } else {
        normal = normalize(normal);
    }
    LabPBRMat mat;
    mat.albedo = vec3(0);
    mat.f0 = mat.albedo;
    mat.roughness = 1.0;
    mat.metallic = 0.0;
    mat.subSurface = 0.0;
    mat.transmission = 0.0;
    mat.ior = 0.0;
    mat.emission = 0.0;

    uint useColorLayer = v0.useColorLayer;
    vec3 colorLayer;
    if (useColorLayer > 0) {
        colorLayer = (baryCoords.x * v0.colorLayer + baryCoords.y * v1.colorLayer + baryCoords.z * v2.colorLayer).rgb;
    } else {
        colorLayer = vec3(1.0);
    }

    vec3 albedo = vec3(1.0);
    float alpha = 1.0;
    vec3 tint = albedo * colorLayer;

    mainRay.hitT = gl_HitTEXT;

    vec3 rayOrigin = worldPos;

    // shadow ray for direct lighting
    vec3 lightDir = normalize(skyUBO.sunDirection);
    float kappa = 1000;
    if (lightDir.y < 0) { lightDir = -lightDir; }
    vec3 sampledLightDir = lightDir;

    // check if sample is above surface
    if (dot(sampledLightDir, normal) > 0.0) {
    } else {
        normal = -normal;
    }
    float pdf; // not used
    vec3 lightBRDF = DisneyEval(mat, viewDir, normal, sampledLightDir, pdf);
    lightBRDF.r = max(lightBRDF.r, 0.1);
    lightBRDF.g = max(lightBRDF.g, 0.1);
    lightBRDF.b = max(lightBRDF.b, 0.1);

    shadowRay.radiance = vec3(0.0);
    shadowRay.throughput = vec3(1.0);
    shadowRay.seed = mainRay.seed;

    traceRayEXT(topLevelAS, gl_RayFlagsNoneEXT,
                WORLD_MASK, // masks
                0,          // sbtRecordOffset
                0,          // sbtRecordStride
                2,          // missIndex
                rayOrigin, 0.001, sampledLightDir, 1000, 1);

    vec3 lightContribution = shadowRay.radiance;

    float progress = clamp(skyUBO.rainGradient * skyUBO.envSky.y, 0.0, 1.0);
    vec3 lightRadiance = lightContribution * mainRay.throughput * lightBRDF;
    vec3 rainyRadiance = mix(vec3(0.04, 0.05, 0.1) * 0.8, vec3(0.1), smoothstep(-0.3, 0.3, lightDir.y));
    rainyRadiance *= skyUBO.envSky.x;
    mainRay.radiance += mix(lightRadiance, rainyRadiance, progress);

    mainRay.hitT = gl_HitTEXT;

    mainRay.instanceIndex = instanceID;
    mainRay.geometryIndex = geometryID;
    mainRay.primitiveIndex = gl_PrimitiveID;
    mainRay.baryCoords = baryCoords;
    mainRay.worldPos = worldPos;
    mainRay.normal = vec3(0);
    mainRay.albedoValue = vec4(0);
    mainRay.specularValue = vec4(0);
    mainRay.normalValue = vec4(0);
    mainRay.flagValue = ivec4(0);
    mainRay.noisy = 0;
    mainRay.stop = 1;
}
