#ifndef TEXTURE_RULES_GLSL
#define TEXTURE_RULES_GLSL

#define TEXTURE_RULE_MODE_VOLUME_MASK 0x3u
#define TEXTURE_RULE_MODE_THICKNESS_MASK (0x3u << 2u)
#define TEXTURE_RULE_MODE_COAT_MASK (0x3u << 4u)

#include "material_fetch.glsl"

layout(std430, set = 1, binding = 11) readonly buffer TextureRuleRegistryBuffer {
    TextureRuleEntry textureRuleEntries[];
};

TextureRuleEntry safeTextureRuleEntry(uint materialId) {
    uint spriteId = materialRuleSpriteId(materialEffectiveId(materialId));
    return textureRuleEntries[min(spriteId, SPRITE_MAX_ENTRIES - 1u)];
}

bool textureRuleHasTransmission(uint materialId) {
    TextureRuleEntry rule = safeTextureRuleEntry(materialId);
    return (rule.flags & TEXTURE_RULE_ENABLED) != 0u &&
           (rule.flags & TEXTURE_RULE_TRANSMISSION) != 0u &&
           rule.transmission > 0.0;
}

void applyTextureRule(uint materialId, inout LabPBRMat mat) {
    TextureRuleEntry rule = safeTextureRuleEntry(materialId);
    if ((rule.flags & TEXTURE_RULE_ENABLED) == 0u) {
        return;
    }
    if ((rule.flags & TEXTURE_RULE_TRANSMISSION) != 0u) {
        mat.transmission = clamp(rule.transmission, 0.0, 1.0);
    }
    if ((rule.flags & TEXTURE_RULE_ABSORPTION) != 0u) {
        mat.absorption = clamp(vec3(rule.absorptionR, rule.absorptionG, rule.absorptionB), vec3(0.0), vec3(1.0));
        mat.absorptionDistance = max(rule.absorptionDistance, 0.01);
    }
    if ((rule.flags & TEXTURE_RULE_THICKNESS) != 0u) {
        mat.thickness = clamp(rule.thicknessAmount, 0.0, 1.0);
    }
    if ((rule.flags & TEXTURE_RULE_VOLUME_MODE) != 0u) {
        uint modeMask = TEXTURE_RULE_MODE_VOLUME_MASK | TEXTURE_RULE_MODE_THICKNESS_MASK | TEXTURE_RULE_MODE_COAT_MASK;
        mat.materialModeFlags = (mat.materialModeFlags & ~modeMask) | (rule.modeFlags & modeMask);
    }
    if ((rule.flags & TEXTURE_RULE_DIFFUSE_MODEL) != 0u) {
        mat.materialModeFlags =
            (mat.materialModeFlags & ~TEXTURE_RULE_MODE_DIFFUSE_MASK) |
            (rule.modeFlags & TEXTURE_RULE_MODE_DIFFUSE_MASK);
    }
    if ((rule.flags & TEXTURE_RULE_REFRACTION_ROUGHNESS) != 0u) {
        mat.refractionRoughness = clamp(rule.refractionRoughness, 0.0, 1.0);
    }
    if ((rule.flags & TEXTURE_RULE_METALLIC) != 0u) {
        mat.metallic = clamp(rule.metallic, 0.0, 1.0);
    }
    if ((rule.flags & TEXTURE_RULE_ROUGHNESS) != 0u) {
        mat.roughness = clamp(rule.roughness, 0.02, 1.0);
    }
    if ((rule.flags & TEXTURE_RULE_IOR) != 0u) {
        mat.ior = clamp(rule.ior, 1.0, 3.0);
    }
    if ((rule.flags & TEXTURE_RULE_F0) != 0u) {
        mat.f0 = vec3(clamp(rule.f0, 0.0, 0.99));
    }
    if ((rule.flags & TEXTURE_RULE_CONDUCTOR_F0_RGB) != 0u) {
        vec3 conductorF0 = clamp(vec3(rule.conductorF0R, rule.conductorF0G, rule.conductorF0B), vec3(0.0), vec3(0.99));
        mat.f0 = conductorF0;
        mat.albedo = conductorF0;
        mat.metallic = 1.0;
    }
    if ((rule.flags & TEXTURE_RULE_ANISOTROPIC) != 0u) {
        mat.anisotropic = clamp(rule.anisotropic, 0.0, 1.0);
    }
    if ((rule.flags & TEXTURE_RULE_ANISOTROPIC_ROTATION) != 0u) {
        mat.anisotropicRotation = fract(rule.anisotropicRotation);
    }
    if ((rule.flags & TEXTURE_RULE_SHEEN) != 0u) {
        mat.sheenWeight = clamp(rule.sheenWeight, 0.0, 1.0);
        mat.sheenTint = clamp(rule.sheenTint, 0.0, 1.0);
    }
    if ((rule.flags & TEXTURE_RULE_SHEEN_ROUGHNESS) != 0u) {
        mat.sheenRoughness = clamp(rule.sheenRoughness, 0.0, 1.0);
    }
    if ((rule.flags & TEXTURE_RULE_SUBSURFACE_EXT) != 0u) {
        mat.subSurfaceRadius = clamp(rule.subSurfaceRadius, 0.0, 1.0);
        mat.subSurfaceThickness = clamp(rule.subSurfaceThickness, 0.0, 1.0);
        mat.subSurfaceTint = clamp(vec3(rule.subSurfaceTintR, rule.subSurfaceTintG, rule.subSurfaceTintB), vec3(0.0), vec3(1.0));
    }
    if ((rule.flags & TEXTURE_RULE_COAT) != 0u) {
        mat.coatWeight = clamp(rule.coatWeight, 0.0, 1.0);
        mat.coatRoughness = clamp(rule.coatRoughness, 0.0, 1.0);
    }
    if ((rule.flags & TEXTURE_RULE_COAT_EXT) != 0u) {
        mat.coatIor = clamp(rule.coatIor, 1.0, 3.0);
        mat.coatTint = clamp(vec3(rule.coatTintR, rule.coatTintG, rule.coatTintB), vec3(0.0), vec3(1.0));
        mat.coatMask = clamp(rule.coatMask, 0.0, 1.0);
    }
    if ((rule.flags & TEXTURE_RULE_EMISSION) != 0u) {
        mat.emission = clamp(rule.emission, 0.0, 1.0);
    }
    if ((rule.flags & TEXTURE_RULE_EMISSION_TINT_NITS) != 0u) {
        mat.emissionTint = clamp(vec3(rule.emissionTintR, rule.emissionTintG, rule.emissionTintB), vec3(0.0), vec3(1.0));
        mat.emissionNits = max(rule.emissionNits, 0.0);
    }
    if ((rule.flags & TEXTURE_RULE_UV_TRANSFORM) != 0u) {
        mat.uvScale = clamp(vec2(rule.uvScaleU, rule.uvScaleV), vec2(0.01), vec2(16.0));
        mat.uvOffset = vec2(rule.uvOffsetU, rule.uvOffsetV);
    }
    if ((rule.flags & TEXTURE_RULE_FILTER_RADIUS) != 0u) {
        mat.filterRadius = clamp(rule.filterRadius, 0.0, 16.0);
    }
    if ((rule.flags & TEXTURE_RULE_MIP_BIAS) != 0u) {
        mat.mipBias = clamp(rule.mipBias, -8.0, 8.0);
    }
    if ((rule.flags & TEXTURE_RULE_DISPLACEMENT_SCALE) != 0u) {
        mat.displacementScale = clamp(rule.displacementScale, 0.0, 4.0);
    }
}

#endif
