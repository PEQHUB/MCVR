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

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer CompactPositionBuffer {
    PBRTriangleCompact vertices[];
};

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer LosslessPositionBuffer {
    PBRTriangleLossless vertices[];
};

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer MaterialBuffer {
    PBRTriangle vertices[];
};

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer CompactMaterialBuffer {
    PBRTriangleCompact vertices[];
};

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer LosslessMaterialBuffer {
    PBRTriangleLossless vertices[];
};

layout(set = 1, binding = 4) readonly buffer PositionBufferAddr {
    uint64_t addrs[];
}
positionBufferAddrs;

layout(set = 1, binding = 5) readonly buffer MaterialBufferAddr {
    uint64_t addrs[];
}
materialBufferAddrs;

uint getGeometryBufferIndex(uint instanceID, uint geometryID) {
    return (blasOffsets.offsets[instanceID] & 0x3FFFFFFFu) + geometryID;
}

uint getGeometryFormat(uint instanceID) {
    return blasOffsets.offsets[instanceID] >> 30u;
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
    return 0u;
}

uint getCoordinate(uint packedData) {
    return (packedData & PBR_FLAG_COORD_MASK) >> PBR_FLAG_COORD_SHIFT;
}

PositionVertex toPositionVertex(PBRTriangle v) {
    PositionVertex outV;
    outV.pos = v.pos;
    return outV;
}

PositionVertex toPositionVertex(PBRTriangleCompact v) {
    PositionVertex outV;
    outV.pos = v.pos;
    return outV;
}

PositionVertex toPositionVertex(PBRTriangleLossless v) {
    PositionVertex outV;
    outV.pos = v.pos;
    return outV;
}

MaterialVertex toMaterialVertex(PBRTriangle v) {
    MaterialVertex outV;
    outV.norm = v.norm;
    outV.textureID = v.textureID;
    outV.colorLayer = v.colorLayer;
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

MaterialVertex toMaterialVertex(PBRTriangleCompact v) {
    MaterialVertex outV;
    outV.norm = vec3(0.0, 1.0, 0.0);
    outV.textureID = v.packed0 >> 16u;
    outV.colorLayer = vec4(
        float(v.colorPacked & 0xFFu) / 255.0,
        float((v.colorPacked >> 8u) & 0xFFu) / 255.0,
        float((v.colorPacked >> 16u) & 0xFFu) / 255.0,
        float((v.colorPacked >> 24u) & 0xFFu) / 255.0);
    outV.textureUV = v.textureUV;
    outV.overlayUV = ivec2(0);
    outV.glintUV = vec2(0.0);
    outV.glintTexture = 0u;
    outV.albedoEmission = unpackHalf2x16(v.packed1).x;
    outV.lightUV = ivec2(0);
    outV.packedData = (v.packed0 & 0x77FFu) | (((v.packed0 >> 11u) & 1u) << 11u);
    outV.emissiveBlockType = (v.packed1 >> 16u) | (((v.packed0 >> 11u) & 1u) << 16u);
    return outV;
}

MaterialVertex toMaterialVertex(PBRTriangleLossless v) {
    MaterialVertex outV;
    outV.norm = vec3(0.0, 1.0, 0.0);
    outV.textureID = v.textureID_glint & 0xFFFFu;
    outV.colorLayer = v.colorLayer;
    outV.textureUV = v.textureUV;
    outV.overlayUV = ivec2(int(v.overlayPacked & 0xFFFFu), int(v.overlayPacked >> 16u));
    outV.glintUV = v.glintUV;
    outV.glintTexture = v.textureID_glint >> 16u;
    outV.albedoEmission = v.albedoEmission;
    outV.lightUV = ivec2(0);
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
    uint format = 0u;
    if (format == 1u) {
        CompactPositionBuffer buf = CompactPositionBuffer(positionBufferAddrs.addrs[geometryBufferIndex]);
        p0 = toPositionVertex(buf.vertices[i0]);
        p1 = toPositionVertex(buf.vertices[i1]);
        p2 = toPositionVertex(buf.vertices[i2]);
    } else if (format == 2u) {
        LosslessPositionBuffer buf = LosslessPositionBuffer(positionBufferAddrs.addrs[geometryBufferIndex]);
        p0 = toPositionVertex(buf.vertices[i0]);
        p1 = toPositionVertex(buf.vertices[i1]);
        p2 = toPositionVertex(buf.vertices[i2]);
    } else {
        PositionBuffer buf = PositionBuffer(positionBufferAddrs.addrs[geometryBufferIndex]);
        p0 = toPositionVertex(buf.vertices[i0]);
        p1 = toPositionVertex(buf.vertices[i1]);
        p2 = toPositionVertex(buf.vertices[i2]);
    }
}

void loadTriangleMaterial(uint geometryBufferIndex,
                          uint i0,
                          uint i1,
                          uint i2,
                          out MaterialVertex m0,
                          out MaterialVertex m1,
                          out MaterialVertex m2) {
    uint rawOffset = blasOffsets.offsets[gl_InstanceCustomIndexEXT];
    uint format = rawOffset >> 30u;
    if (format == 1u) {
        CompactMaterialBuffer buf = CompactMaterialBuffer(positionBufferAddrs.addrs[geometryBufferIndex]);
        m0 = toMaterialVertex(buf.vertices[i0]);
        m1 = toMaterialVertex(buf.vertices[i1]);
        m2 = toMaterialVertex(buf.vertices[i2]);
    } else if (format == 2u) {
        LosslessMaterialBuffer buf = LosslessMaterialBuffer(positionBufferAddrs.addrs[geometryBufferIndex]);
        m0 = toMaterialVertex(buf.vertices[i0]);
        m1 = toMaterialVertex(buf.vertices[i1]);
        m2 = toMaterialVertex(buf.vertices[i2]);
    } else {
        MaterialBuffer buf = MaterialBuffer(materialBufferAddrs.addrs[geometryBufferIndex]);
        m0 = toMaterialVertex(buf.vertices[i0]);
        m1 = toMaterialVertex(buf.vertices[i1]);
        m2 = toMaterialVertex(buf.vertices[i2]);
    }
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
