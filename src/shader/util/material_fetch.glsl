#ifndef MATERIAL_FETCH_GLSL
#define MATERIAL_FETCH_GLSL

layout(std430, set = 1, binding = 14) readonly buffer MaterialRegistryBuffer {
    MaterialEntry materialEntries[];
};

MaterialEntry safeMaterialEntry(uint materialId) {
    return materialEntries[min(materialId, MATERIAL_MAX_ENTRIES - 1u)];
}

bool materialCanSample(MaterialEntry material) {
    return (material.flags & MATERIAL_FLAG_VALID) != 0u &&
           (material.flags & MATERIAL_FLAG_GPU_RESIDENT) != 0u &&
           (material.flags & MATERIAL_FLAG_PENDING_RESIDENCY) == 0u;
}

uint materialEffectiveId(uint materialId) {
    MaterialEntry material = safeMaterialEntry(materialId);
    if (materialCanSample(material)) return materialId;

    uint fallbackId = uint(max(material.fallbackMaterialId, 0));
    if (fallbackId != materialId && materialCanSample(safeMaterialEntry(fallbackId))) {
        return fallbackId;
    }
    return 0u;
}

MaterialEntry materialSamplingEntry(uint materialId) {
    return safeMaterialEntry(materialEffectiveId(materialId));
}

uint materialBaseSpriteId(uint materialId) {
    MaterialEntry material = materialSamplingEntry(materialId);
    if ((material.flags & MATERIAL_FLAG_VALID) == 0u) return 0u;
    return uint(max(material.baseSpriteId, 0));
}

uint materialFallbackId(uint materialId) {
    MaterialEntry material = safeMaterialEntry(materialId);
    return uint(max(material.fallbackMaterialId, 0));
}

uint materialRuleSpriteId(uint materialId) {
    MaterialEntry material = materialSamplingEntry(materialId);
    if ((material.flags & MATERIAL_FLAG_VALID) == 0u) return 0u;
    if (material.baseSpriteId >= 0) return uint(material.baseSpriteId);
    return materialBaseSpriteId(materialFallbackId(materialId));
}

bool materialUsesSpriteArray(MaterialEntry material) {
    if ((material.flags & MATERIAL_FLAG_V4_PAGE_ADDRESS) != 0u) return false;
    return (material.flags & (MATERIAL_FLAG_VANILLA_SPRITE | MATERIAL_FLAG_FALLBACK)) != 0u;
}

uint materialPackedPageNamespace(uint packedPage) {
    uint ns = packedPage & MATERIAL_PAGE_NAMESPACE_MASK;
    if (ns == MATERIAL_PAGE_NAMESPACE_VANILLA_TIER) return 1u;
    if (ns == MATERIAL_PAGE_NAMESPACE_CTM) return 2u;
    if (ns == 0x30000000u) return 3u;
    return 0u;
}

uint materialPackedPageIndex(uint packedPage) {
    return packedPage & MATERIAL_PAGE_INDEX_MASK;
}

uint materialPackedPageTier(uint packedPage) {
    return (packedPage & MATERIAL_PAGE_TIER_MASK) >> 24u;
}

// V4 semantics over the preserved 80-byte ABI: namespace and tier are encoded
// into each packed page handle rather than separate struct fields.
uint materialAlbedoNamespace(MaterialEntry m) { return materialPackedPageNamespace(m.albedoPage); }
uint materialAlbedoTier(MaterialEntry m) { return materialPackedPageTier(m.albedoPage); }
uint materialSpecularNamespace(MaterialEntry m) { return materialPackedPageNamespace(m.specularPage); }
uint materialSpecularTier(MaterialEntry m) { return materialPackedPageTier(m.specularPage); }
uint materialNormalNamespace(MaterialEntry m) { return materialPackedPageNamespace(m.normalPage); }
uint materialNormalTier(MaterialEntry m) { return materialPackedPageTier(m.normalPage); }
uint materialFlagNamespace(MaterialEntry m) { return materialPackedPageNamespace(m.flagPage); }
uint materialFlagTier(MaterialEntry m) { return materialPackedPageTier(m.flagPage); }

// V4: Residency state
bool materialIsGpuResident(MaterialEntry m) { return (m.flags & MATERIAL_FLAG_GPU_RESIDENT) != 0u; }
bool materialIsPending(MaterialEntry m) { return (m.flags & MATERIAL_FLAG_PENDING_RESIDENCY) != 0u; }

#endif // MATERIAL_FETCH_GLSL
