// sprite_fetch.glsl — Texture array sprite sampling via SpriteRegistry SSBO.
//
// Provides helpers for block geometry to sample albedo, specular, normal, and flags
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
layout(set = 0, binding = 6) uniform sampler2DArray blockFlag;

#ifndef TEXTURE_RULES_GLSL
layout(std430, set = 1, binding = 11) readonly buffer TextureRuleRegistryBuffer {
    TextureRuleEntry textureRuleEntries[];
};

TextureRuleEntry safeTextureRuleEntry(uint spriteId) {
    return textureRuleEntries[min(spriteId, SPRITE_MAX_ENTRIES - 1u)];
}
#endif

// --- Helpers ---

SpriteEntry safeSpriteEntry(uint spriteId) {
    return spriteEntries[min(spriteId, SPRITE_MAX_ENTRIES - 1u)];
}

// Compute the current animation layer for a sprite.
uint spriteAnimLayer(SpriteEntry se, uint animTick) {
    uint tickRate = max(se.tickRate, 1u);
    uint frameCount = max(se.frameCount, 1u);
    return se.baseLayer + ((animTick / tickRate) % frameCount);
}

bool spriteLayerInRange(uint layer, ivec3 textureDims) {
    return textureDims.x > 0 && textureDims.y > 0 &&
           textureDims.z > 0 && layer < uint(textureDims.z);
}

bool spriteLayerInRange(int layer, ivec3 textureDims) {
    return layer >= 0 && textureDims.x > 0 && textureDims.y > 0 &&
           textureDims.z > 0 && layer < textureDims.z;
}

bool spriteAlbedoLayerInRange(SpriteEntry se, uint animTick, out uint layer) {
    layer = spriteAnimLayer(se, animTick);
    return spriteLayerInRange(layer, textureSize(blockAlbedo, 0));
}

bool spriteSpecularLayerInRange(SpriteEntry se) {
    if (se.specularLayer < 0) return false;
    return spriteLayerInRange(se.specularLayer, textureSize(blockSpecular, 0));
}

bool spriteNormalLayerInRange(SpriteEntry se) {
    if (se.normalLayer < 0) return false;
    return spriteLayerInRange(se.normalLayer, textureSize(blockNormal, 0));
}

bool spriteFlagLayerInRange(uint spriteId) {
    return spriteLayerInRange(spriteId, textureSize(blockFlag, 0));
}

vec2 materialRuleUv(uint spriteId, vec2 uv) {
    TextureRuleEntry rule = safeTextureRuleEntry(spriteId);
    if ((rule.flags & TEXTURE_RULE_ENABLED) == 0u ||
        (rule.flags & TEXTURE_RULE_UV_TRANSFORM) == 0u) {
        return uv;
    }
    vec2 scale = clamp(vec2(rule.uvScaleU, rule.uvScaleV), vec2(0.01), vec2(16.0));
    return fract(uv * scale + vec2(rule.uvOffsetU, rule.uvOffsetV));
}

float materialRuleLod(uint spriteId, float lod) {
    TextureRuleEntry rule = safeTextureRuleEntry(spriteId);
    if ((rule.flags & TEXTURE_RULE_ENABLED) == 0u) {
        return lod;
    }
    float adjusted = lod;
    if ((rule.flags & TEXTURE_RULE_MIP_BIAS) != 0u) {
        adjusted += clamp(rule.mipBias, -8.0, 8.0);
    }
    if ((rule.flags & TEXTURE_RULE_FILTER_RADIUS) != 0u) {
        adjusted += log2(1.0 + clamp(rule.filterRadius, 0.0, 16.0) * 0.25);
    }
    return adjusted;
}

// Sample block albedo texture from the sprite array.
vec4 fetchBlockAlbedoTex(uint spriteId, vec2 uv, uint animTick) {
    SpriteEntry se = safeSpriteEntry(spriteId);
    uint layer;
    if (!spriteAlbedoLayerInRange(se, animTick, layer)) return vec4(1.0);
    return texture(blockAlbedo, vec3(materialRuleUv(spriteId, uv), float(layer)));
}

// Sample block albedo with explicit LOD.
vec4 fetchBlockAlbedoLod(uint spriteId, vec2 uv, uint animTick, float lod) {
    SpriteEntry se = safeSpriteEntry(spriteId);
    uint layer;
    if (!spriteAlbedoLayerInRange(se, animTick, layer)) return vec4(1.0);
    return textureLod(blockAlbedo, vec3(materialRuleUv(spriteId, uv), float(layer)), materialRuleLod(spriteId, lod));
}

// Sample block specular from the specular array (static, not animated).
vec4 fetchBlockSpecularTex(uint spriteId, vec2 uv) {
    SpriteEntry se = safeSpriteEntry(spriteId);
    if (!spriteSpecularLayerInRange(se)) return vec4(0.0);
    return texture(blockSpecular, vec3(materialRuleUv(spriteId, uv), float(se.specularLayer)));
}

vec4 fetchBlockSpecularLod(uint spriteId, vec2 uv, float lod) {
    SpriteEntry se = safeSpriteEntry(spriteId);
    if (!spriteSpecularLayerInRange(se)) return vec4(0.0);
    return textureLod(blockSpecular, vec3(materialRuleUv(spriteId, uv), float(se.specularLayer)), materialRuleLod(spriteId, lod));
}

// Sample block normal from the normal array (static, not animated).
vec4 fetchBlockNormalTex(uint spriteId, vec2 uv) {
    SpriteEntry se = safeSpriteEntry(spriteId);
    if (!spriteNormalLayerInRange(se)) return vec4(0.5, 0.5, 1.0, 1.0); // flat normal
    return texture(blockNormal, vec3(materialRuleUv(spriteId, uv), float(se.normalLayer)));
}

vec4 fetchBlockNormalLod(uint spriteId, vec2 uv, float lod) {
    SpriteEntry se = safeSpriteEntry(spriteId);
    if (!spriteNormalLayerInRange(se)) return vec4(0.5, 0.5, 1.0, 1.0);
    return textureLod(blockNormal, vec3(materialRuleUv(spriteId, uv), float(se.normalLayer)), materialRuleLod(spriteId, lod));
}

ivec4 fetchBlockFlagLod(uint spriteId, vec2 uv, float lod) {
    if (!spriteFlagLayerInRange(spriteId)) return ivec4(0);
    vec4 flagValue = textureLod(blockFlag, vec3(materialRuleUv(spriteId, uv), float(spriteId)), materialRuleLod(spriteId, lod));
    return ivec4(round(flagValue * 255.0));
}

struct BlockOverlayMaterial {
    bool present;
    bool hasSpecular;
    bool hasNormal;
    bool hasFlags;
    bool emissiveOverlay;
    float alpha;
    uint spriteId;
    vec4 albedo;
    vec4 specular;
    vec4 normal;
    ivec4 flags;
};

BlockOverlayMaterial fetchBlockOverlayMaterialLod(uint spriteId, vec2 uv, uint animTick, float lod) {
    BlockOverlayMaterial overlay;
    overlay.present = false;
    overlay.hasSpecular = false;
    overlay.hasNormal = false;
    overlay.hasFlags = false;
    overlay.emissiveOverlay = false;
    overlay.alpha = 0.0;
    overlay.spriteId = spriteId;
    overlay.albedo = vec4(0.0);
    overlay.specular = vec4(0.0);
    overlay.normal = vec4(0.5, 0.5, 1.0, 1.0);
    overlay.flags = ivec4(0);

    SpriteEntry se = safeSpriteEntry(spriteId);
    if (se.overlaySprite < 0) return overlay;

    uint overlaySpriteId = uint(se.overlaySprite);
    SpriteEntry overlaySe = safeSpriteEntry(overlaySpriteId);
    uint layer;
    if (!spriteAlbedoLayerInRange(overlaySe, animTick, layer)) return overlay;

    overlay.present = true;
    overlay.spriteId = overlaySpriteId;
    overlay.albedo = fetchBlockAlbedoLod(overlaySpriteId, uv, animTick, lod);
    overlay.alpha = clamp(overlay.albedo.a, 0.0, 1.0);
    overlay.emissiveOverlay = (overlaySe.flags & SPRITE_FLAG_EMISSIVE_OVERLAY) != 0u;

    overlay.hasSpecular = spriteSpecularLayerInRange(overlaySe);
    if (overlay.hasSpecular) {
        overlay.specular = fetchBlockSpecularLod(overlaySpriteId, uv, lod);
    }

    overlay.hasNormal = spriteNormalLayerInRange(overlaySe);
    if (overlay.hasNormal) {
        overlay.normal = fetchBlockNormalLod(overlaySpriteId, uv, lod);
    }

    overlay.hasFlags = spriteFlagLayerInRange(overlaySpriteId);
    if (overlay.hasFlags) {
        overlay.flags = fetchBlockFlagLod(overlaySpriteId, uv, lod);
    }

    return overlay;
}

bool applyBlockOverlayMaterialLod(inout vec4 albedoValue,
                                  inout vec4 specularValue,
                                  inout vec4 normalValue,
                                  inout ivec4 flagValue,
                                  uint spriteId,
                                  vec2 uv,
                                  uint animTick,
                                  float lod,
                                  vec3 overlayTint,
                                  out uint materialRuleSpriteId) {
    materialRuleSpriteId = spriteId;
    BlockOverlayMaterial overlay = fetchBlockOverlayMaterialLod(spriteId, uv, animTick, lod);
    if (!overlay.present || overlay.alpha <= 0.0) return false;

    float a = overlay.alpha;
    albedoValue.rgb = mix(albedoValue.rgb, overlay.albedo.rgb * overlayTint, a);

    if (overlay.hasSpecular) {
        specularValue = mix(specularValue, overlay.specular, a);
    }
    if (overlay.hasNormal) {
        normalValue = mix(normalValue, overlay.normal, a);
    }
    if (overlay.hasFlags && a >= 0.5) {
        flagValue = overlay.flags;
        materialRuleSpriteId = overlay.spriteId;
    }
    if (overlay.emissiveOverlay) {
        specularValue.a = max(specularValue.a, min(254.0 / 255.0, a * (254.0 / 255.0)));
        materialRuleSpriteId = overlay.spriteId;
    }

    return true;
}

#endif // SPRITE_FETCH_GLSL
