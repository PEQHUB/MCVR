#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#include "util/disney.glsl"
#include "util/random.glsl"
#include "util/ray_cone.glsl"
#include "util/ray.glsl"
#include "util/util.glsl"
#include "common/shared.hpp"
#include "common/chunk_lookup.glsl"

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

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer IndexBuffer {
    uint indices[];
}
indexBuffer;

#include "util/vertex.glsl"

layout(location = 0) rayPayloadInEXT MainRay mainRay;
hitAttributeEXT vec2 attribs;

void main() {
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
    mainRay.origin = worldPos + mainRay.direction * 0.001;

    raySetInsideBoat(mainRay, true);
    raySetContinue(mainRay, true);
    mainRay.hitT = gl_HitTEXT;
    mainRay.coneWidth += gl_HitTEXT * mainRay.coneSpread;
}
