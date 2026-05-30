#ifndef VERTEX_GLSL
#define VERTEX_GLSL

#include "common/shared.hpp"
#include "common/constants.glsl"

const uint USE_COLOR_LAYER_BIT = 1u << 0u;
const uint USE_TEXTURE_BIT = 1u << 1u;
const uint USE_OVERLAY_BIT = 1u << 2u;
const uint USE_GLINT_BIT = 1u << 3u;
const uint USE_NORM_BIT = 1u << 4u;
const uint USE_LIGHT_BIT = 1u << 5u;
const uint ALPHA_MODE_SHIFT = 8u;
const uint COORDINATE_SHIFT = 12u;
const uint NO_HEIGHT_SURFACE_BIT = 1u << 16u;

#ifndef CONST_ONLY
layout(set = 1, binding = 4) readonly buffer PositionBufferAddr {
    uint64_t addrs[];
}
positionBufferAddrs;

layout(set = 1, binding = 5) readonly buffer MaterialBufferAddr {
    uint64_t addrs[];
}
materialBufferAddrs;

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer PositionBuffer {
    PositionVertex vertices[];
}
positionBuffer;

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer MaterialBuffer {
    MaterialVertex vertices[];
}
materialBuffer;

uint getGeometryBufferIndex(uint instanceID, uint geometryID) {
    return blasOffsets.offsets[instanceID] + geometryID;
}

bool hasColorLayer(uint packedData) {
    return (packedData & USE_COLOR_LAYER_BIT) != 0u;
}

bool hasTexture(uint packedData) {
    return (packedData & USE_TEXTURE_BIT) != 0u;
}

bool hasOverlay(uint packedData) {
    return (packedData & USE_OVERLAY_BIT) != 0u;
}

bool hasGlint(uint packedData) {
    return (packedData & USE_GLINT_BIT) != 0u;
}

bool hasNorm(uint packedData) {
    return (packedData & USE_NORM_BIT) != 0u;
}

bool hasLight(uint packedData) {
    return (packedData & USE_LIGHT_BIT) != 0u;
}

bool hasNoHeightSurface(uint packedData) {
    return (packedData & NO_HEIGHT_SURFACE_BIT) != 0u;
}

uint getAlphaMode(uint packedData) {
    return (packedData >> ALPHA_MODE_SHIFT) & 0xFu;
}

uint getCoordinate(uint packedData) {
    return (packedData >> COORDINATE_SHIFT) & 0xFu;
}

void loadTriangleIndices(uint geometryBufferIndex, uint primitiveID, out uint i0, out uint i1, out uint i2) {
    IndexBuffer indexBuffer = IndexBuffer(indexBufferAddrs.addrs[geometryBufferIndex]);
    uint indexBaseID = 3u * primitiveID;
    i0 = indexBuffer.indices[indexBaseID];
    i1 = indexBuffer.indices[indexBaseID + 1u];
    i2 = indexBuffer.indices[indexBaseID + 2u];
}

void loadTrianglePositions(uint geometryBufferIndex,
                           uint i0,
                           uint i1,
                           uint i2,
                           out PositionVertex p0,
                           out PositionVertex p1,
                           out PositionVertex p2) {
    PositionBuffer positionBufferRef = PositionBuffer(positionBufferAddrs.addrs[geometryBufferIndex]);
    p0 = positionBufferRef.vertices[i0];
    p1 = positionBufferRef.vertices[i1];
    p2 = positionBufferRef.vertices[i2];
}

void loadTriangleMaterial(uint geometryBufferIndex,
                          uint i0,
                          uint i1,
                          uint i2,
                          out MaterialVertex m0,
                          out MaterialVertex m1,
                          out MaterialVertex m2) {
    MaterialBuffer materialBufferRef = MaterialBuffer(materialBufferAddrs.addrs[geometryBufferIndex]);
    m0 = materialBufferRef.vertices[i0];
    m1 = materialBufferRef.vertices[i1];
    m2 = materialBufferRef.vertices[i2];
}

void loadTriangle(uint geometryBufferIndex,
                  uint primitiveID,
                  out uint i0,
                  out uint i1,
                  out uint i2,
                  out PositionVertex p0,
                  out PositionVertex p1,
                  out PositionVertex p2,
                  out MaterialVertex m0,
                  out MaterialVertex m1,
                  out MaterialVertex m2) {
    loadTriangleIndices(geometryBufferIndex, primitiveID, i0, i1, i2);
    loadTrianglePositions(geometryBufferIndex, i0, i1, i2, p0, p1, p2);
    loadTriangleMaterial(geometryBufferIndex, i0, i1, i2, m0, m1, m2);
}
#endif

#endif
