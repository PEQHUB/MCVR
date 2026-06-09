#ifndef RADIANCE_VERTEX_GLSL
#define RADIANCE_VERTEX_GLSL

#include "common/shared.hpp"
#include "common/constants.glsl"

struct PositionVertex {
    vec3 pos;
};

struct MaterialVertex {
    vec3 norm;
    uint textureID;
    vec4 colorLayer;
    vec3 postBase;
    vec2 textureUV;
    ivec2 overlayUV;
    vec2 glintUV;
    uint glintTexture;
    float albedoEmission;
    ivec2 lightUV;
    uint packedData;
    uint emissiveBlockType;
};

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer PositionBuffer {
    PBRTriangle vertices[];
};



layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer MaterialBuffer {
    PBRTriangle vertices[];
};



layout(set = 1, binding = 2) readonly buffer VertexBufferAddr {
    uint64_t addrs[];
}
vertexBufferAddrs;

layout(set = 1, binding = 3) readonly buffer IndexBufferAddr {
    uint64_t addrs[];
}
indexBufferAddrs;

uint getGeometryBufferIndex(uint instanceID, uint geometryID) {
    return blasOffsets.offsets[instanceID] + geometryID;
}

uint getGeometryFormat(uint instanceID) {
    return 0u;
}

bool hasColorLayer(uint packedData) {
    return (packedData & PBR_FLAG_USE_COLOR_LAYER) != 0u;
}

bool hasTexture(uint packedData) {
    return (packedData & PBR_FLAG_USE_TEXTURE) != 0u;
}

bool hasOverlay(uint packedData) {
    return (packedData & PBR_FLAG_USE_OVERLAY) != 0u;
}

bool hasGlint(uint packedData) {
    return (packedData & PBR_FLAG_USE_GLINT) != 0u;
}

bool hasNorm(uint packedData) {
    return (packedData & PBR_FLAG_USE_NORM) != 0u;
}

bool hasLight(uint packedData) {
    return (packedData & PBR_FLAG_USE_LIGHT) != 0u;
}

bool hasNoHeightSurface(uint packedData) {
    return false;
}

uint getAlphaMode(uint packedData) {
    return (packedData & PBR_FLAG_ALPHA_MODE_MASK) >> PBR_FLAG_ALPHA_MODE_SHIFT;
}

uint getTextMode(uint packedData) {
    return (packedData & PBR_FLAG_TEXT_MODE_MASK) >> PBR_FLAG_TEXT_MODE_SHIFT;
}

uint getCoordinate(uint packedData) {
    return (packedData & PBR_FLAG_COORD_MASK) >> PBR_FLAG_COORD_SHIFT;
}

PositionVertex toPositionVertex(PBRTriangle v) {
    PositionVertex outV;
    outV.pos = v.pos;
    return outV;
}



MaterialVertex toMaterialVertex(PBRTriangle v) {
    MaterialVertex outV;
    outV.norm = v.norm;
    outV.textureID = v.textureID;
    outV.colorLayer = v.colorLayer;
    outV.postBase = v.postBase;
    outV.textureUV = v.textureUV;
    outV.overlayUV = ivec2(int(v.overlayPacked & 0xFFFFu), int(v.overlayPacked >> 16u));
    outV.glintUV = v.glintUV;
    outV.glintTexture = v.glintTexture;
    outV.albedoEmission = v.albedoEmission;
    outV.lightUV = ivec2(int(v.lightPacked & 0xFFFFu), int(v.lightPacked >> 16u));
    outV.packedData = v.flags;
    outV.emissiveBlockType = v.emissiveBlockType;
    return outV;
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
    PositionBuffer buf = PositionBuffer(vertexBufferAddrs.addrs[geometryBufferIndex]);
    p0 = toPositionVertex(buf.vertices[i0]);
    p1 = toPositionVertex(buf.vertices[i1]);
    p2 = toPositionVertex(buf.vertices[i2]);
}
void loadTriangleMaterial(uint geometryBufferIndex,
                          uint i0,
                          uint i1,
                          uint i2,
                          out MaterialVertex m0,
                          out MaterialVertex m1,
                          out MaterialVertex m2) {
    MaterialBuffer buf = MaterialBuffer(vertexBufferAddrs.addrs[geometryBufferIndex]);
    m0 = toMaterialVertex(buf.vertices[i0]);
    m1 = toMaterialVertex(buf.vertices[i1]);
    m2 = toMaterialVertex(buf.vertices[i2]);
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
