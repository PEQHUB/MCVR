// vertex_fetch.glsl — Triple-path vertex fetch for three formats:
//   0 = full PBRTriangle (96 bytes, entities)
//   1 = compact PBRTriangleCompact (32 bytes, far chunks — lossy)
//   2 = lossless PBRTriangleLossless (64 bytes, near chunks — bit-identical)
//
// Format is encoded in upper 2 bits of blasOffsets.offsets[instanceID].
// Shader extracts: format = offset >> 30, blasOffset = offset & 0x3FFFFFFF.

// Buffer references for each format
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
    v.flags = c.packed0 & 0x7FFu;           // bits 0-10: original flags (USE_GLINT/OVERLAY cleared at pack time)
    v.textureID = c.packed0 >> 16u;          // bits 16-31: textureID
    v.textureUV = c.textureUV;
    v.colorLayer = vec3(
        float(c.colorPacked & 0xFFu) / 255.0,
        float((c.colorPacked >> 8u) & 0xFFu) / 255.0,
        float((c.colorPacked >> 16u) & 0xFFu) / 255.0
    );
    v.colorLayerAlpha = float((c.colorPacked >> 24u) & 0xFFu) / 255.0;
    v.albedoEmission = unpackHalf2x16(c.packed1).x;  // lower 16 bits = fp16 emission
    // Restore emissiveBlockType: bits 0-15 from packed1 upper, bit 16 (vivid) from flags bit 11
    v.emissiveBlockType = (c.packed1 >> 16u) | (((c.packed0 >> 11u) & 1u) << 16u);
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
