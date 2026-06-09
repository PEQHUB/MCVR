#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_nonuniform_qualifier : require

// V2 PR39-prep MVP closest-hit shader.
//
// Uses gl_InstanceCustomIndexEXT to look up this chunk's vertex/index BDAs from
// the per-instance SSBOs that TlasService maintains. Then fetches the 3 vertices
// for this triangle, derives a face normal, and shades with the SkyUBO sun direction.
//
// Vertex format: V1's PBRTriangle (96 bytes) — but we only read pos (vec3 at offset 0)
// and colorLayer (vec4 at offset 32) for the MVP.

// Per-vertex payload — must match common/shared.hpp PBRTriangle layout enough for
// the fields we read. Buffer reference pointers go through scalar block layout
// so the 96-byte struct stride is exact.
struct V2Vertex {
    vec3  pos;          // 0..11
    uint  flags;        // 12..15
    vec3  norm;         // 16..27
    float albedoEmission; // 28..31
    vec4  colorLayer;   // 32..47
    vec3  postBase;     // 48..59
    uint  emissiveBlockType; // 60..63
    vec2  textureUV;    // 64..71
    vec2  glintUV;      // 72..79
    uint  textureID;    // 80..83
    uint  glintTexture; // 84..87
    uint  overlayPacked;// 88..91
    uint  lightPacked;  // 92..95
};

layout(buffer_reference, scalar) readonly buffer VertexBuffer {
    V2Vertex v[];
};

layout(buffer_reference, scalar) readonly buffer IndexBuffer {
    uint i[];
};

// Per-instance BDA SSBOs from TlasService
layout(set = 1, binding = 2, std430) readonly buffer VertexBufferAddrs {
    uint64_t addrs[];
} vertexBufferAddrs;

layout(set = 1, binding = 3, std430) readonly buffer IndexBufferAddrs {
    uint64_t addrs[];
} indexBufferAddrs;

layout(set = 2, binding = 2, std140) uniform SkyUBO {
    vec3 baseColor;
    uint skyType;
    vec4 horizonColor;
    vec3 sunDirection;
    uint isSunRisingOrSetting;
    vec3 moonDirection;
    float moonDirPad;
} sky;

layout(location = 0) rayPayloadInEXT vec4 hitPayload;
hitAttributeEXT vec2 attribs;

void main() {
    uint instanceIdx = gl_InstanceCustomIndexEXT;
    uint primIdx     = gl_PrimitiveID;

    VertexBuffer verts  = VertexBuffer(vertexBufferAddrs.addrs[instanceIdx]);
    IndexBuffer  idxs   = IndexBuffer(indexBufferAddrs.addrs[instanceIdx]);

    uint i0 = idxs.i[primIdx * 3 + 0];
    uint i1 = idxs.i[primIdx * 3 + 1];
    uint i2 = idxs.i[primIdx * 3 + 2];

    V2Vertex v0 = verts.v[i0];
    V2Vertex v1 = verts.v[i1];
    V2Vertex v2 = verts.v[i2];

    // Face normal (cross of two edges). The greedy mesher emits ccw triangles in
    // world space, so this should be outward-facing.
    vec3 e1 = v1.pos - v0.pos;
    vec3 e2 = v2.pos - v0.pos;
    vec3 faceN = normalize(cross(e1, e2));

    // Barycentric interpolation of vertex color
    vec3 bary = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);
    vec4 albedoA = v0.colorLayer * bary.x + v1.colorLayer * bary.y + v2.colorLayer * bary.z;
    vec3 albedo  = albedoA.rgb;

    // Lambertian shading from the SkyUBO sun direction.
    // Add a small ambient term so back faces aren't pitch black.
    vec3 L = normalize(sky.sunDirection);
    float NdotL = max(dot(faceN, L), 0.0);
    vec3 ambient = vec3(0.15);
    vec3 sunCol  = vec3(1.0, 0.95, 0.85);
    vec3 shaded  = albedo * (ambient + sunCol * NdotL);

    hitPayload = vec4(shaded, 1.0);
}
