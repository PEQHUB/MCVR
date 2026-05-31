// vertex_fetch.glsl — Triple-path vertex fetch for three formats:
//   0 = full PBRTriangle (96 bytes, entities)
//   1 = compact PBRTriangleCompact (32 bytes, far chunks — lossy)
//   2 = lossless PBRTriangleLossless (64 bytes, near chunks — bit-identical)
//
// Format is encoded in upper 2 bits of blasOffsets.offsets[instanceID].
// Shader extracts: format = offset >> 30, blasOffset = offset & 0x3FFFFFFF.

// Buffer references for all vertex formats
layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer VertexBuffer {
    PBRTriangle vertices[];
};

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer IndexBuffer {
    uint indices[];
};

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer CompactVertexBuffer {
    PBRTriangleCompact vertices[];
};

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer LosslessVertexBuffer {
    PBRTriangleLossless vertices[];
};

// Unpacked vertex data — uniform interface for all three formats
struct UnpackedVertex {
    vec3 pos;
    uint flags;
    uint textureID;
    vec2 textureUV;
    vec3 colorLayer;
    float colorLayerAlpha;
    float albedoEmission;
    uint emissiveBlockType;
    vec2 glintUV;
    uint glintTexture;
    uint overlayPacked;
    uint lightPacked;
};

UnpackedVertex unpackCompactVertex(PBRTriangleCompact c) {
    UnpackedVertex v;
    v.pos = c.pos;
    v.flags = c.packed0 & 0x77FFu;          // bits 0-10 + 12-14: flags + BIOME_TINT + BLOCK_GEOMETRY (skip bit 11 = VIVID)
    v.textureID = c.packed0 >> 16u;          // bits 16-31: textureID
    v.textureUV = c.textureUV;
    v.colorLayer = vec3(
        float(c.colorPacked & 0xFFu) / 255.0,
        float((c.colorPacked >> 8u) & 0xFFu) / 255.0,
        float((c.colorPacked >> 16u) & 0xFFu) / 255.0
    );
    v.colorLayerAlpha = float((c.colorPacked >> 24u) & 0xFFu) / 255.0;
    v.albedoEmission = unpackHalf2x16(c.packed1).x;  // lower 16 bits = fp16 emission
    // Restore emissiveBlockType: bits 0-15 from packed1 upper, bit 16 (vivid)
    // from packed0 bit 11, and thin plant bit 31 from compact carrier bit 15.
    v.emissiveBlockType = (c.packed1 >> 16u) |
                           (((c.packed0 >> 11u) & 1u) << 16u) |
                           (((c.packed0 >> 15u) & 1u) << 31u);
    v.glintUV = vec2(0.0);
    v.glintTexture = 0u;
    v.overlayPacked = 0u;
    v.lightPacked = 0u;
    return v;
}

UnpackedVertex unpackLosslessVertex(PBRTriangleLossless l) {
    UnpackedVertex v;
    v.pos = l.pos;
    v.flags = l.flags;
    v.textureID = l.textureID_glint & 0xFFFFu;
    v.glintTexture = l.textureID_glint >> 16u;
    v.textureUV = l.textureUV;
    v.colorLayer = l.colorLayer.rgb;
    v.colorLayerAlpha = l.colorLayer.a;
    v.albedoEmission = l.albedoEmission;
    v.emissiveBlockType = l.emissiveBlockType;
    v.glintUV = l.glintUV;
    v.overlayPacked = l.overlayPacked;
    v.lightPacked = 0u;
    return v;
}

UnpackedVertex unpackFullVertex(PBRTriangle t) {
    UnpackedVertex v;
    v.pos = t.pos;
    v.flags = t.flags;
    v.textureID = t.textureID;
    v.textureUV = t.textureUV;
    v.colorLayer = t.colorLayer.rgb;
    v.colorLayerAlpha = t.colorLayer.a;
    v.albedoEmission = t.albedoEmission;
    v.emissiveBlockType = t.emissiveBlockType;
    v.glintUV = t.glintUV;
    v.glintTexture = t.glintTexture;
    v.overlayPacked = t.overlayPacked;
    v.lightPacked = t.lightPacked;
    return v;
}

// Extract raw blasOffset (lower 30 bits) from format-encoded offset.
uint decodeBlasOffset(uint rawOffset) {
    return rawOffset & 0x3FFFFFFFu;
}

// Position-only fetch for motion vectors: reads only .pos from 3 vertices,
// branching on format for correct stride. All formats have pos at offset 0.
void fetchTrianglePositions(uint format, uint64_t vertexAddr, uint64_t indexAddr, uint primitiveID,
                            vec3 baryCoords, out vec3 interpPos) {
    IndexBuffer ib = IndexBuffer(indexAddr);
    uint base = 3u * primitiveID;
    uint i0 = ib.indices[base];
    uint i1 = ib.indices[base + 1u];
    uint i2 = ib.indices[base + 2u];

    vec3 p0, p1, p2;
    if (format == 1u) {
        CompactVertexBuffer cvb = CompactVertexBuffer(vertexAddr);
        p0 = cvb.vertices[i0].pos; p1 = cvb.vertices[i1].pos; p2 = cvb.vertices[i2].pos;
    } else if (format == 2u) {
        LosslessVertexBuffer lvb = LosslessVertexBuffer(vertexAddr);
        p0 = lvb.vertices[i0].pos; p1 = lvb.vertices[i1].pos; p2 = lvb.vertices[i2].pos;
    } else {
        VertexBuffer vb = VertexBuffer(vertexAddr);
        p0 = vb.vertices[i0].pos; p1 = vb.vertices[i1].pos; p2 = vb.vertices[i2].pos;
    }
    interpPos = baryCoords.x * p0 + baryCoords.y * p1 + baryCoords.z * p2;
}

// Format-branched vertex fetch: extracts format from blasOffset upper 2 bits,
// loads 3 triangle vertices via BDA, and unpacks to UnpackedVertex.
// Returns the raw blasOffset (lower 30 bits) and vertex format via out parameters.
void fetchTriangleVertices(uint instanceID, uint geometryID, uint primitiveID,
                           out UnpackedVertex v0, out UnpackedVertex v1, out UnpackedVertex v2,
                           out uint outBlasOffset, out uint outFormat) {
    uint rawOffset = blasOffsets.offsets[instanceID];
    uint format = rawOffset >> 30u;
    uint blasOffset = rawOffset & 0x3FFFFFFFu;
    outBlasOffset = blasOffset;
    outFormat = format;

    IndexBuffer indexBuffer = IndexBuffer(indexBufferAddrs.addrs[blasOffset + geometryID]);
    uint indexBaseID = 3u * primitiveID;
    uint i0 = indexBuffer.indices[indexBaseID];
    uint i1 = indexBuffer.indices[indexBaseID + 1u];
    uint i2 = indexBuffer.indices[indexBaseID + 2u];

    if (format == 1u) {
        // Compact 32-byte vertices (far chunks)
        CompactVertexBuffer cvb = CompactVertexBuffer(vertexBufferAddrs.addrs[blasOffset + geometryID]);
        v0 = unpackCompactVertex(cvb.vertices[i0]);
        v1 = unpackCompactVertex(cvb.vertices[i1]);
        v2 = unpackCompactVertex(cvb.vertices[i2]);
    } else if (format == 2u) {
        // Lossless 64-byte vertices (near chunks)
        LosslessVertexBuffer lvb = LosslessVertexBuffer(vertexBufferAddrs.addrs[blasOffset + geometryID]);
        v0 = unpackLosslessVertex(lvb.vertices[i0]);
        v1 = unpackLosslessVertex(lvb.vertices[i1]);
        v2 = unpackLosslessVertex(lvb.vertices[i2]);
    } else {
        // Full 96-byte vertices (entities, format 0)
        VertexBuffer vb = VertexBuffer(vertexBufferAddrs.addrs[blasOffset + geometryID]);
        v0 = unpackFullVertex(vb.vertices[i0]);
        v1 = unpackFullVertex(vb.vertices[i1]);
        v2 = unpackFullVertex(vb.vertices[i2]);
    }
}
