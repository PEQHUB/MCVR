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
// Tier 1a addition: samples blockAlbedo sampler2DArray using textureUV +
// textureID (sprite-registry index) from the vertex. SpriteRegistry SSBO maps
// spriteId → array layer. Animation is skipped in MVP (uses baseLayer directly).
// Falls back to vertex color when the registry stub indicates "not yet finalized"
// (frameCount == 0), so pre-finalize frames render solid colors instead of garbage.

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

// SpriteRegistry SSBO — matches V1 SpriteEntry (32 bytes, common/shared.hpp).
// TextureService writes this once during finalize. Before finalize, the
// SceneResourceService stub is bound; its entries have frameCount == 0,
// which we use as a sentinel to skip texture sampling.
struct SpriteEntry {
    uint baseLayer;
    uint frameCount;
    uint tickRate;
    uint flags;
    int  specularLayer;
    int  normalLayer;
    int  overlaySprite;
    int  maskLayer;
};
layout(std430, set = 1, binding = 13) readonly buffer SpriteRegistryBuffer {
    SpriteEntry spriteEntries[];
};

// Biome color SSBO — uvec4[] per-TLAS-instance (.x=grass, .y=foliage, .z=water).
// Packed 0x00RRGGBB. Populated per-chunk by raytracing_adapter each frame.
// Matches v2_full/world_solid_transparent.rchit layout at set 1 binding 10.
layout(set = 1, binding = 10) readonly buffer BiomeColorBuffer {
    uvec4 data[];
} biomeColors;

// Flag bit constants (must match MCVR/src/common/shared.hpp PBRTriangle::PBR_FLAG_*).
const uint V2_FLAG_USE_COLOR_LAYER  = 1u << 1;
const uint V2_FLAG_BIOME_TINT_SHIFT = 12u;

// Block sprite albedo array. PARTIALLY_BOUND in the descriptor layout, so it
// is safe to declare even when the adapter has not written it yet (early frames
// before TextureService finalize). The frameCount sentinel above keeps us from
// actually sampling it in that window.
layout(set = 0, binding = 3) uniform sampler2DArray blockAlbedo;

struct HitPayload {
    vec4 colorAndHit;     // rgb + hit flag
    vec4 posAndT;         // world pos + distance
    vec4 normalAndRough;  // world normal + roughness
};
layout(location = 0) rayPayloadInEXT HitPayload hitPayload;
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

    // Barycentric interpolation of vertex color (base per-vertex tint).
    vec3 bary = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);
    vec4 albedoA = v0.colorLayer * bary.x + v1.colorLayer * bary.y + v2.colorLayer * bary.z;
    vec3 tint    = albedoA.rgb;

    // Biome tint resolution (grass/foliage/water): C++ mesher leaves colorLayer = (1,1,1,1)
    // when quad.tintIndex >= 0 and encodes type in flag bits 12-13. Shader resolves the
    // per-instance biome color here — otherwise grass/leaves/water render pure white.
    // Mirrors v2_full/world_solid_transparent.rchit:474-488.
    uint biomeTintType = (v0.flags >> V2_FLAG_BIOME_TINT_SHIFT) & 0x3u;
    if (biomeTintType != 0u) {
        uint packed = biomeColors.data[instanceIdx][biomeTintType - 1u];
        tint = vec3(
            float((packed >> 16u) & 0xFFu) / 255.0,
            float((packed >> 8u)  & 0xFFu) / 255.0,
            float( packed         & 0xFFu) / 255.0
        );
    }
    // Fixed-color tint (PBR_FLAG_USE_COLOR_LAYER) already lives in v*.colorLayer — tint above.

    // Texture sampling. spriteId comes from v0.textureID — the greedy mesher
    // emits the same id for all 3 verts of a triangle, so v0 is sufficient.
    // Skip animation (just baseLayer) — animTick isn't yet routed into V2's UBO.
    vec2 uv = v0.textureUV * bary.x + v1.textureUV * bary.y + v2.textureUV * bary.z;
    uint spriteId = v0.textureID;
    SpriteEntry se = spriteEntries[spriteId];
    vec3 albedo;
    if (se.frameCount > 0u) {
        vec4 albedoTex = texture(blockAlbedo, vec3(uv, float(se.baseLayer)));
        albedo = albedoTex.rgb * tint;
    } else {
        albedo = tint;
    }

    // Lambertian shading from the SkyUBO sun direction.
    // Add a small ambient term so back faces aren't pitch black.
    vec3 L = normalize(sky.sunDirection);
    float NdotL = max(dot(faceN, L), 0.0);
    vec3 ambient = vec3(0.15);
    vec3 sunCol  = vec3(1.0, 0.95, 0.85);
    vec3 shaded  = albedo * (ambient + sunCol * NdotL);

    // World-space hit position
    vec3 worldPos = gl_WorldRayOriginEXT + gl_HitTEXT * gl_WorldRayDirectionEXT;

    hitPayload.colorAndHit    = vec4(shaded, 1.0);
    hitPayload.posAndT        = vec4(worldPos, gl_HitTEXT);
    hitPayload.normalAndRough = vec4(faceN, 1.0);
}
