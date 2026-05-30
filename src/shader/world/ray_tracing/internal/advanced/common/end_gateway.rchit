#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#include "util/disney.glsl"
#include "util/random.glsl"
#include "util/ray_cone.glsl"
#include "util/ray.glsl"
#include "util/util.glsl"
#include "common/shared.hpp"

layout(set = 0, binding = 0) uniform sampler2D textures[];

layout(set = 1, binding = 0) uniform accelerationStructureEXT topLevelAS;

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

layout(set = 2, binding = 0) uniform WorldUniform {
    WorldUBO worldUBO;
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

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer IndexBuffer {
    uint indices[];
}
indexBuffer;

#include "util/vertex.glsl"

layout(location = 0) rayPayloadInEXT MainRay mainRay;
hitAttributeEXT vec2 attribs;

#include "common/end_portal.glsl"

void main() {
    vec3 viewDir = -mainRay.direction;

    uint instanceID = gl_InstanceCustomIndexEXT;
    uint geometryID = gl_GeometryIndexEXT;

    uint geometryBufferIndex = getGeometryBufferIndex(instanceID, geometryID);

    uint i0;
    uint i1;
    uint i2;
    loadTriangleIndices(geometryBufferIndex, gl_PrimitiveID, i0, i1, i2);

    PositionVertex p0;
    PositionVertex p1;
    PositionVertex p2;
    loadTrianglePositions(geometryBufferIndex, i0, i1, i2, p0, p1, p2);

    vec3 baryCoords = vec3(1.0 - (attribs.x + attribs.y), attribs.x, attribs.y);
    vec3 localPos = baryCoords.x * p0.pos + baryCoords.y * p1.pos + baryCoords.z * p2.pos;
    vec3 worldPos = vec4(localPos, 1.0) * gl_ObjectToWorld3x4EXT;

    vec4 texProj0 =
        projectPosition(worldUBO.cameraProjMat * worldUBO.cameraEffectedViewMat * vec4(worldPos, 1.0));
    vec3 color = computeEndPortalColor(texProj0, 15, worldUBO.endSkyTextureID, worldUBO.endPortalTextureID, worldUBO.gameTime);

    // add glowing radiance
    mainRay.radiance += 4 * color;
    mainRay.hitT = gl_HitTEXT;
    mainRay.normal = vec3(0.0);
    rayClearMaterial(mainRay);
    raySetNoisy(mainRay, false);
    mainRay.hasPrevScenePos = 0u;
    raySetStop(mainRay, true);
}
