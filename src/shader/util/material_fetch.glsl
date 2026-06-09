#ifndef MATERIAL_FETCH_GLSL
#define MATERIAL_FETCH_GLSL

layout(std430, set = 1, binding = 14) readonly buffer MaterialRegistryBuffer {
    MaterialEntry materialEntries[];
};

MaterialEntry safeMaterialEntry(uint materialId) {
    return materialEntries[min(materialId, MATERIAL_MAX_ENTRIES - 1u)];
}

uint materialBaseSpriteId(uint materialId) {
    MaterialEntry material = safeMaterialEntry(materialId);
    if ((material.flags & MATERIAL_FLAG_VALID) == 0u) return 0u;
    return uint(max(material.baseSpriteId, 0));
}

uint materialFallbackId(uint materialId) {
    MaterialEntry material = safeMaterialEntry(materialId);
    return uint(max(material.fallbackMaterialId, 0));
}

uint materialRuleSpriteId(uint materialId) {
    MaterialEntry material = safeMaterialEntry(materialId);
    if ((material.flags & MATERIAL_FLAG_VALID) == 0u) return 0u;
    if (material.baseSpriteId >= 0) return uint(material.baseSpriteId);
    return materialBaseSpriteId(materialFallbackId(materialId));
}

bool materialUsesSpriteArray(MaterialEntry material) {
    return (material.flags & (MATERIAL_FLAG_VANILLA_SPRITE | MATERIAL_FLAG_FALLBACK)) != 0u;
}

// V4: Extract namespace from material entry
uint materialAlbedoNamespace(MaterialEntry m) { return m.albedoNamespace; }
uint materialAlbedoTier(MaterialEntry m) { return m.albedoTier; }
uint materialSpecularNamespace(MaterialEntry m) { return m.specularNamespace; }
uint materialSpecularTier(MaterialEntry m) { return m.specularTier; }
uint materialNormalNamespace(MaterialEntry m) { return m.normalNamespace; }
uint materialNormalTier(MaterialEntry m) { return m.normalTier; }
uint materialFlagNamespace(MaterialEntry m) { return m.flagNamespace; }
uint materialFlagTier(MaterialEntry m) { return m.flagTier; }

// V4: Residency state
bool materialIsGpuResident(MaterialEntry m) { return m.residencyState == 3u; }
bool materialIsPending(MaterialEntry m) { return m.residencyState == 1u || m.residencyState == 2u; }

#endif // MATERIAL_FETCH_GLSL
