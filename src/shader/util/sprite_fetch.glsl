// sprite_fetch.glsl — Texture array sprite sampling via SpriteRegistry SSBO.
//
// Provides helpers for block geometry to sample albedo, specular, and normal
// from sampler2DArray textures using per-sprite metadata.
//
// Usage: #include this after vertex_fetch.glsl in any ray tracing shader.
// Block geometry (format 1/2) uses these helpers.
// Entity geometry (format 0) continues using textures[] bindless array.

#ifndef SPRITE_FETCH_GLSL
#define SPRITE_FETCH_GLSL

// --- Sprite Registry SSBO ---
// SpriteEntry struct defined in common/shared.hpp (dual C++/GLSL header)

layout(std430, set = 1, binding = 13) readonly buffer SpriteRegistryBuffer {
    SpriteEntry spriteEntries[];
};

// --- Texture Arrays ---

layout(set = 0, binding = 3) uniform sampler2DArray blockAlbedo;
layout(set = 0, binding = 4) uniform sampler2DArray blockSpecular;
layout(set = 0, binding = 5) uniform sampler2DArray blockNormal;

// --- Helpers ---

SpriteEntry safeSpriteEntry(uint spriteId) {
    return spriteEntries[min(spriteId, SPRITE_MAX_ENTRIES - 1u)];
}

// Compute the current animation layer for a sprite.
uint spriteAnimLayer(SpriteEntry se, uint animTick) {
    return se.baseLayer + ((animTick / se.tickRate) % se.frameCount);
}

// Sample block albedo texture from the sprite array.
vec4 fetchBlockAlbedoTex(uint spriteId, vec2 uv, uint animTick) {
    SpriteEntry se = safeSpriteEntry(spriteId);
    uint layer = spriteAnimLayer(se, animTick);
    return texture(blockAlbedo, vec3(uv, float(layer)));
}

// Sample block albedo with explicit LOD.
vec4 fetchBlockAlbedoLod(uint spriteId, vec2 uv, uint animTick, float lod) {
    SpriteEntry se = safeSpriteEntry(spriteId);
    uint layer = spriteAnimLayer(se, animTick);
    return textureLod(blockAlbedo, vec3(uv, float(layer)), lod);
}

// Sample block specular from the specular array (static, not animated).
vec4 fetchBlockSpecularTex(uint spriteId, vec2 uv) {
    SpriteEntry se = safeSpriteEntry(spriteId);
    if (se.specularLayer < 0) return vec4(0.0);
    return texture(blockSpecular, vec3(uv, float(se.specularLayer)));
}

vec4 fetchBlockSpecularLod(uint spriteId, vec2 uv, float lod) {
    SpriteEntry se = safeSpriteEntry(spriteId);
    if (se.specularLayer < 0) return vec4(0.0);
    return textureLod(blockSpecular, vec3(uv, float(se.specularLayer)), lod);
}

// Sample block normal from the normal array (static, not animated).
vec4 fetchBlockNormalTex(uint spriteId, vec2 uv) {
    SpriteEntry se = safeSpriteEntry(spriteId);
    if (se.normalLayer < 0) return vec4(0.5, 0.5, 1.0, 1.0); // flat normal
    return texture(blockNormal, vec3(uv, float(se.normalLayer)));
}

vec4 fetchBlockNormalLod(uint spriteId, vec2 uv, float lod) {
    SpriteEntry se = safeSpriteEntry(spriteId);
    if (se.normalLayer < 0) return vec4(0.5, 0.5, 1.0, 1.0);
    return textureLod(blockNormal, vec3(uv, float(se.normalLayer)), lod);
}

// Fetch overlay alpha for grass block sides.
// Returns overlay alpha (0 = no overlay, >0 = tinted region).
float fetchOverlayAlpha(uint spriteId, vec2 uv, uint animTick) {
    SpriteEntry se = safeSpriteEntry(spriteId);
    if (se.overlaySprite < 0) return 0.0;

    SpriteEntry overlaySe = safeSpriteEntry(uint(se.overlaySprite));
    uint layer = spriteAnimLayer(overlaySe, animTick);
    return texture(blockAlbedo, vec3(uv, float(layer))).a;
}

#endif // SPRITE_FETCH_GLSL
