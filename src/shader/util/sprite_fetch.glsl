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

#include "material_fetch.glsl"

// --- Texture Arrays ---

layout(set = 0, binding = 3) uniform sampler2DArray blockAlbedo[MATERIAL_TEXTURE_PAGE_MAX];
layout(set = 0, binding = 4) uniform sampler2DArray blockSpecular[MATERIAL_TEXTURE_PAGE_MAX];
layout(set = 0, binding = 5) uniform sampler2DArray blockNormal[MATERIAL_TEXTURE_PAGE_MAX];
layout(set = 0, binding = 6) uniform sampler2DArray blockFlag[MATERIAL_TEXTURE_PAGE_MAX];

#ifndef TEXTURE_RULES_GLSL
layout(std430, set = 1, binding = 11) readonly buffer TextureRuleRegistryBuffer {
    TextureRuleEntry textureRuleEntries[];
};

TextureRuleEntry safeTextureRuleEntry(uint materialId) {
    uint spriteId = materialRuleSpriteId(materialId);
    return textureRuleEntries[min(spriteId, SPRITE_MAX_ENTRIES - 1u)];
}
#endif

// --- Helpers ---

SpriteEntry safeSpriteEntry(uint spriteId) {
    return spriteEntries[min(spriteId, SPRITE_MAX_ENTRIES - 1u)];
}

SpriteEntry safeMaterialSpriteEntry(uint materialId) {
    return safeSpriteEntry(materialBaseSpriteId(materialId));
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

uint materialTexturePage(uint page) {
    return min(materialPackedPageIndex(page), MATERIAL_TEXTURE_PAGE_MAX - 1u);
}

// V4: Compute texture descriptor index from namespace, tier, page
uint textureDescriptorIndexV4(uint namespaceId, uint tier, uint page) {
    if (namespaceId == 1u) {
        return min(1u + tier * 8u + page, MATERIAL_TEXTURE_PAGE_MAX - 1u);
    }
    if (namespaceId == 2u) {
        return min(64u + max(page, 8u) - 8u, MATERIAL_TEXTURE_PAGE_MAX - 1u);
    }
    return min(page, MATERIAL_TEXTURE_PAGE_MAX - 1u);
}

bool materialHasV4PageAddress(MaterialEntry material) {
    return (material.flags & MATERIAL_FLAG_V4_PAGE_ADDRESS) != 0u;
}

uint materialDescriptorPage(MaterialEntry material, uint packedPage) {
    if (!materialHasV4PageAddress(material)) return materialTexturePage(packedPage);
    return textureDescriptorIndexV4(materialPackedPageNamespace(packedPage),
        materialPackedPageTier(packedPage), materialPackedPageIndex(packedPage));
}

ivec3 materialAlbedoTextureSize(MaterialEntry material) {
    return textureSize(blockAlbedo[nonuniformEXT(materialTexturePage(material.albedoPage))], 0);
}

ivec3 materialAlbedoTextureSizeV4(MaterialEntry material) {
    return textureSize(blockAlbedo[nonuniformEXT(textureDescriptorIndexV4(
        materialAlbedoNamespace(material), materialAlbedoTier(material),
        materialPackedPageIndex(material.albedoPage)))], 0);
}

ivec3 materialSpecularTextureSize(MaterialEntry material) {
    return textureSize(blockSpecular[nonuniformEXT(materialTexturePage(material.specularPage))], 0);
}

ivec3 materialNormalTextureSize(MaterialEntry material) {
    return textureSize(blockNormal[nonuniformEXT(materialTexturePage(material.normalPage))], 0);
}

ivec3 materialFlagTextureSize(MaterialEntry material) {
    return textureSize(blockFlag[nonuniformEXT(materialTexturePage(material.flagPage))], 0);
}

bool materialAlbedoLayerInRange(uint materialId, SpriteEntry se, uint animTick, out uint layer) {
    MaterialEntry material = safeMaterialEntry(materialId);
    bool useSpriteArray = !materialHasV4PageAddress(material) &&
        (materialUsesSpriteArray(material) || material.albedoLayer < 0);
    if (useSpriteArray) {
        layer = spriteAnimLayer(se, animTick);
    } else if (materialHasV4PageAddress(material)) {
        uint tickRate = max(se.tickRate, 1u);
        uint frameCount = max(se.frameCount, 1u);
        layer = uint(material.albedoLayer) + ((animTick / tickRate) % frameCount);
    } else {
        layer = uint(material.albedoLayer);
    }
    ivec3 dims = textureSize(blockAlbedo[nonuniformEXT(materialDescriptorPage(material, material.albedoPage))], 0);
    return spriteLayerInRange(layer, dims);
}

int materialSpecularLayer(uint materialId, SpriteEntry se) {
    MaterialEntry material = safeMaterialEntry(materialId);
    return material.specularLayer >= 0 ? material.specularLayer : se.specularLayer;
}

bool materialSpecularLayerInRange(uint materialId, SpriteEntry se, out int layer) {
    MaterialEntry material = safeMaterialEntry(materialId);
    layer = materialSpecularLayer(materialId, se);
    ivec3 dims = textureSize(blockSpecular[nonuniformEXT(materialDescriptorPage(material, material.specularPage))], 0);
    return layer >= 0 && spriteLayerInRange(layer, dims);
}

int materialNormalLayer(uint materialId, SpriteEntry se) {
    MaterialEntry material = safeMaterialEntry(materialId);
    return material.normalLayer >= 0 ? material.normalLayer : se.normalLayer;
}

bool materialNormalLayerInRange(uint materialId, SpriteEntry se, out int layer) {
    MaterialEntry material = safeMaterialEntry(materialId);
    layer = materialNormalLayer(materialId, se);
    ivec3 dims = textureSize(blockNormal[nonuniformEXT(materialDescriptorPage(material, material.normalPage))], 0);
    return layer >= 0 && spriteLayerInRange(layer, dims);
}

int materialFlagLayer(uint materialId) {
    MaterialEntry material = safeMaterialEntry(materialId);
    if (material.flagLayer >= 0) return material.flagLayer;
    return int(materialBaseSpriteId(materialId));
}

bool materialFlagLayerInRange(uint materialId, out int layer) {
    MaterialEntry material = safeMaterialEntry(materialId);
    layer = materialFlagLayer(materialId);
    ivec3 dims = textureSize(blockFlag[nonuniformEXT(materialDescriptorPage(material, material.flagPage))], 0);
    return layer >= 0 && spriteLayerInRange(layer, dims);
}

vec2 materialRegistryUv(uint materialId, vec2 uv) {
    MaterialEntry material = safeMaterialEntry(materialId);
    vec2 scale = clamp(vec2(material.uvScaleU, material.uvScaleV), vec2(0.01), vec2(16.0));
    return fract(uv * scale + vec2(material.uvOffsetU, material.uvOffsetV));
}

vec2 materialRuleUv(uint materialId, vec2 uv) {
    TextureRuleEntry rule = safeTextureRuleEntry(materialId);
    vec2 materialUv = materialRegistryUv(materialId, uv);
    if ((rule.flags & TEXTURE_RULE_ENABLED) == 0u ||
        (rule.flags & TEXTURE_RULE_UV_TRANSFORM) == 0u) {
        return materialUv;
    }
    vec2 scale = clamp(vec2(rule.uvScaleU, rule.uvScaleV), vec2(0.01), vec2(16.0));
    return fract(materialUv * scale + vec2(rule.uvOffsetU, rule.uvOffsetV));
}

float materialRuleLod(uint materialId, float lod) {
    TextureRuleEntry rule = safeTextureRuleEntry(materialId);
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
vec4 fetchBlockAlbedoTex(uint materialId, vec2 uv, uint animTick) {
    MaterialEntry material = safeMaterialEntry(materialId);
    SpriteEntry se = safeMaterialSpriteEntry(materialId);
    uint layer;
    if (!materialAlbedoLayerInRange(materialId, se, animTick, layer)) return vec4(1.0);
    return texture(blockAlbedo[nonuniformEXT(materialDescriptorPage(material, material.albedoPage))],
        vec3(materialRuleUv(materialId, uv), float(layer)));
}

// Sample block albedo with explicit LOD.
vec4 fetchBlockAlbedoLod(uint materialId, vec2 uv, uint animTick, float lod) {
    MaterialEntry material = safeMaterialEntry(materialId);
    SpriteEntry se = safeMaterialSpriteEntry(materialId);
    uint layer;
    if (!materialAlbedoLayerInRange(materialId, se, animTick, layer)) return vec4(1.0);
    return textureLod(blockAlbedo[nonuniformEXT(materialDescriptorPage(material, material.albedoPage))],
        vec3(materialRuleUv(materialId, uv), float(layer)), materialRuleLod(materialId, lod));
}

// Sample block specular from the specular array (static, not animated).
vec4 fetchBlockSpecularTex(uint materialId, vec2 uv) {
    MaterialEntry material = safeMaterialEntry(materialId);
    SpriteEntry se = safeMaterialSpriteEntry(materialId);
    int layer;
    if (!materialSpecularLayerInRange(materialId, se, layer)) return vec4(0.0);
    return texture(blockSpecular[nonuniformEXT(materialDescriptorPage(material, material.specularPage))],
        vec3(materialRuleUv(materialId, uv), float(layer)));
}

vec4 fetchBlockSpecularLod(uint materialId, vec2 uv, float lod) {
    MaterialEntry material = safeMaterialEntry(materialId);
    SpriteEntry se = safeMaterialSpriteEntry(materialId);
    int layer;
    if (!materialSpecularLayerInRange(materialId, se, layer)) return vec4(0.0);
    return textureLod(blockSpecular[nonuniformEXT(materialDescriptorPage(material, material.specularPage))],
        vec3(materialRuleUv(materialId, uv), float(layer)), materialRuleLod(materialId, lod));
}

// Sample block normal from the normal array (static, not animated).
vec4 fetchBlockNormalTex(uint materialId, vec2 uv) {
    MaterialEntry material = safeMaterialEntry(materialId);
    SpriteEntry se = safeMaterialSpriteEntry(materialId);
    int layer;
    if (!materialNormalLayerInRange(materialId, se, layer)) return vec4(0.5, 0.5, 1.0, 1.0); // flat normal
    return texture(blockNormal[nonuniformEXT(materialDescriptorPage(material, material.normalPage))],
        vec3(materialRuleUv(materialId, uv), float(layer)));
}

vec4 fetchBlockNormalLod(uint materialId, vec2 uv, float lod) {
    MaterialEntry material = safeMaterialEntry(materialId);
    SpriteEntry se = safeMaterialSpriteEntry(materialId);
    int layer;
    if (!materialNormalLayerInRange(materialId, se, layer)) return vec4(0.5, 0.5, 1.0, 1.0);
    return textureLod(blockNormal[nonuniformEXT(materialDescriptorPage(material, material.normalPage))],
        vec3(materialRuleUv(materialId, uv), float(layer)), materialRuleLod(materialId, lod));
}

ivec4 fetchBlockFlagLod(uint materialId, vec2 uv, float lod) {
    MaterialEntry material = safeMaterialEntry(materialId);
    int layer;
    if (!materialFlagLayerInRange(materialId, layer)) return ivec4(0);
    vec4 flagValue = textureLod(blockFlag[nonuniformEXT(materialDescriptorPage(material, material.flagPage))],
        vec3(materialRuleUv(materialId, uv), float(layer)), materialRuleLod(materialId, lod));
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

BlockOverlayMaterial fetchBlockOverlayMaterialLod(uint materialId, vec2 uv, uint animTick, float lod) {
    BlockOverlayMaterial overlay;
    overlay.present = false;
    overlay.hasSpecular = false;
    overlay.hasNormal = false;
    overlay.hasFlags = false;
    overlay.emissiveOverlay = false;
    overlay.alpha = 0.0;
    overlay.spriteId = materialId;
    overlay.albedo = vec4(0.0);
    overlay.specular = vec4(0.0);
    overlay.normal = vec4(0.5, 0.5, 1.0, 1.0);
    overlay.flags = ivec4(0);

    MaterialEntry material = safeMaterialEntry(materialId);
    SpriteEntry se = safeMaterialSpriteEntry(materialId);
    int overlayMaterial = material.overlayMaterialId >= 0 ? material.overlayMaterialId : se.overlaySprite;
    if (overlayMaterial < 0) return overlay;

    uint overlayMaterialId = uint(overlayMaterial);
    SpriteEntry overlaySe = safeMaterialSpriteEntry(overlayMaterialId);
    uint layer;
    if (!materialAlbedoLayerInRange(overlayMaterialId, overlaySe, animTick, layer)) return overlay;

    overlay.present = true;
    overlay.spriteId = overlayMaterialId;
    overlay.albedo = fetchBlockAlbedoLod(overlayMaterialId, uv, animTick, lod);
    overlay.alpha = clamp(overlay.albedo.a, 0.0, 1.0);
    overlay.emissiveOverlay = (overlaySe.flags & SPRITE_FLAG_EMISSIVE_OVERLAY) != 0u;

    int specularLayer;
    overlay.hasSpecular = materialSpecularLayerInRange(overlayMaterialId, overlaySe, specularLayer);
    if (overlay.hasSpecular) {
        overlay.specular = fetchBlockSpecularLod(overlayMaterialId, uv, lod);
    }

    int normalLayer;
    overlay.hasNormal = materialNormalLayerInRange(overlayMaterialId, overlaySe, normalLayer);
    if (overlay.hasNormal) {
        overlay.normal = fetchBlockNormalLod(overlayMaterialId, uv, lod);
    }

    int flagLayer;
    overlay.hasFlags = materialFlagLayerInRange(overlayMaterialId, flagLayer);
    if (overlay.hasFlags) {
        overlay.flags = fetchBlockFlagLod(overlayMaterialId, uv, lod);
    }

    return overlay;
}

bool applyBlockOverlayMaterialLod(inout vec4 albedoValue,
                                  inout vec4 specularValue,
                                  inout vec4 normalValue,
                                  inout ivec4 flagValue,
                                  uint materialId,
                                  vec2 uv,
                                  uint animTick,
                                  float lod,
                                  vec3 overlayTint,
                                  out uint materialRuleSpriteId) {
    materialRuleSpriteId = materialId;
    BlockOverlayMaterial overlay = fetchBlockOverlayMaterialLod(materialId, uv, animTick, lod);
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
