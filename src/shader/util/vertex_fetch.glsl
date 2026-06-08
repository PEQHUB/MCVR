// vertex_fetch.glsl - full PBRTriangle fetch only.

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer VertexBuffer {
    PBRTriangle vertices[];
};

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer IndexBuffer {
    uint indices[];
};

struct UnpackedVertex {
    vec3 pos;
    uint flags;
    uint textureID;
    vec2 textureUV;
    vec3 colorLayer;
    float colorLayerAlpha;
    float albedoEmission;
    vec3 postBase;
    uint emissiveBlockType;
    vec2 glintUV;
    uint glintTexture;
    uint overlayPacked;
    uint lightPacked;
};

UnpackedVertex unpackFullVertex(PBRTriangle t) {
    UnpackedVertex v;
    v.pos = t.pos;
    v.flags = t.flags;
    v.textureID = t.textureID;
    v.textureUV = t.textureUV;
    v.colorLayer = t.colorLayer.rgb;
    v.colorLayerAlpha = t.colorLayer.a;
    v.albedoEmission = t.albedoEmission;
    v.postBase = t.postBase;
    v.emissiveBlockType = t.emissiveBlockType;
    v.glintUV = t.glintUV;
    v.glintTexture = t.glintTexture;
    v.overlayPacked = t.overlayPacked;
    v.lightPacked = t.lightPacked;
    return v;
}

uint decodeBlasOffset(uint rawOffset) {
    return rawOffset;
}

void fetchTrianglePositions(uint format, uint64_t vertexAddr, uint64_t indexAddr, uint primitiveID,
                            vec3 baryCoords, out vec3 interpPos) {
    IndexBuffer ib = IndexBuffer(indexAddr);
    uint base = 3u * primitiveID;
    uint i0 = ib.indices[base];
    uint i1 = ib.indices[base + 1u];
    uint i2 = ib.indices[base + 2u];

    VertexBuffer vb = VertexBuffer(vertexAddr);
    vec3 p0 = vb.vertices[i0].pos;
    vec3 p1 = vb.vertices[i1].pos;
    vec3 p2 = vb.vertices[i2].pos;
    interpPos = baryCoords.x * p0 + baryCoords.y * p1 + baryCoords.z * p2;
}

void fetchTriangleVertices(uint instanceID, uint geometryID, uint primitiveID,
                           out UnpackedVertex v0, out UnpackedVertex v1, out UnpackedVertex v2,
                           out uint outBlasOffset, out uint outFormat) {
    uint blasOffset = blasOffsets.offsets[instanceID];
    outBlasOffset = blasOffset;
    outFormat = 0u;

    IndexBuffer indexBuffer = IndexBuffer(indexBufferAddrs.addrs[blasOffset + geometryID]);
    uint indexBaseID = 3u * primitiveID;
    uint i0 = indexBuffer.indices[indexBaseID];
    uint i1 = indexBuffer.indices[indexBaseID + 1u];
    uint i2 = indexBuffer.indices[indexBaseID + 2u];

    VertexBuffer vb = VertexBuffer(vertexBufferAddrs.addrs[blasOffset + geometryID]);
    v0 = unpackFullVertex(vb.vertices[i0]);
    v1 = unpackFullVertex(vb.vertices[i1]);
    v2 = unpackFullVertex(vb.vertices[i2]);
}
