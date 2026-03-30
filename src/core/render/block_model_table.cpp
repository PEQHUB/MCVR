#include "block_model_table.hpp"
#include "core/render/renderer.hpp"
#include <cstring>
#include <iostream>

void BlockModelTable::load(const BlockModelEntry* entries, uint32_t entryCount,
                           const BlockModelQuad* quads, uint32_t quadCount) {
    // Find max state ID for sparse array sizing
    maxStateId_ = 0;
    for (uint32_t i = 0; i < entryCount; i++) {
        if (entries[i].globalStateId > maxStateId_)
            maxStateId_ = entries[i].globalStateId;
    }

    // Allocate sparse array indexed by globalStateId
    entries_.clear();
    entries_.resize(maxStateId_ + 1);
    std::memset(entries_.data(), 0, entries_.size() * sizeof(BlockModelEntry));

    // Populate entries at their globalStateId positions
    uint32_t modelCount = 0;
    for (uint32_t i = 0; i < entryCount; i++) {
        uint32_t id = entries[i].globalStateId;
        entries_[id] = entries[i];
        if (entries[i].renderType != 0) modelCount++;
    }

    // Extract water data from pure water entry (renderType==2, fluidType==1)
    waterSpriteStill_ = 0;
    waterSpriteFlow_  = 0;
    waterMaterialOrdinal_ = 255;
    for (uint32_t i = 0; i < entryCount; i++) {
        if (entries[i].renderType == 2 && entries[i].fluidType == 1) {
            waterSpriteStill_ = entries[i].fluidSpriteStill();
            waterSpriteFlow_  = entries[i].fluidSpriteFlow();
            waterMaterialOrdinal_ = entries[i].materialOrdinal;
            break;
        }
    }

    // Copy quad array (UVs are in atlas space, need normalizeQuadUVs() later)
    quads_.assign(quads, quads + quadCount);
    uvsNormalized_ = false;

    std::cout << "[BlockModelTable] Loaded " << entryCount << " block states ("
              << modelCount << " with models), " << quadCount << " quads, "
              << "max stateId=" << maxStateId_ << std::endl;
}

void BlockModelTable::loadBiomeTints(const BiomeTintEntry* tints, uint32_t count) {
    maxBiomeId_ = 0;
    for (uint32_t i = 0; i < count; i++) {
        if (tints[i].biomeRawId > maxBiomeId_)
            maxBiomeId_ = tints[i].biomeRawId;
    }

    biomeTints_.clear();
    biomeTints_.resize((maxBiomeId_ + 1) * 3, glm::u8vec3(255, 255, 255));

    for (uint32_t i = 0; i < count; i++) {
        uint32_t idx = tints[i].biomeRawId * 3 + tints[i].tintType;
        if (idx < biomeTints_.size()) {
            biomeTints_[idx] = glm::u8vec3(tints[i].r, tints[i].g, tints[i].b);
        }
    }

    std::cout << "[BlockModelTable] Loaded " << count << " biome tint entries, "
              << "max biomeId=" << maxBiomeId_ << std::endl;
}

void BlockModelTable::normalizeQuadUVs() {
    if (uvsNormalized_) {
        std::cout << "[BlockModelTable] UVs already normalized, skipping" << std::endl;
        return;
    }

    auto& texSys = Renderer::textureSystem;
    if (texSys.spriteCount() == 0) {
        std::cout << "[BlockModelTable] No texture system data, skipping UV normalization" << std::endl;
        return;
    }

    uint32_t normalized = 0;
    for (auto& quad : quads_) {
        auto& bounds = texSys.getSpriteBounds(quad.spriteId);
        float sizeU = bounds.maxU - bounds.minU;
        float sizeV = bounds.maxV - bounds.minV;
        if (sizeU < 1e-6f) sizeU = 1.0f;
        if (sizeV < 1e-6f) sizeV = 1.0f;

        for (int v = 0; v < 4; v++) {
            quad.uvs[v][0] = (quad.uvs[v][0] - bounds.minU) / sizeU;
            quad.uvs[v][1] = (quad.uvs[v][1] - bounds.minV) / sizeV;
        }
        normalized++;
    }

    uvsNormalized_ = true;
    std::cout << "[BlockModelTable] Pre-normalized UVs for " << normalized << " quads" << std::endl;
}

const BlockModelEntry* BlockModelTable::getEntry(uint32_t globalStateId) const {
    if (globalStateId > maxStateId_) return nullptr;
    const auto& entry = entries_[globalStateId];
    if (entry.renderType == 0) return nullptr;
    return &entry;
}

const BlockModelQuad* BlockModelTable::getFaceQuads(const BlockModelEntry& entry,
                                                     uint8_t direction,
                                                     uint8_t& outCount) const {
    if (direction > 6) { outCount = 0; return nullptr; }
    outCount = entry.faceQuadCounts[direction];
    if (outCount == 0) return nullptr;

    // Compute offset into quad array for this direction
    uint32_t offset = entry.quadOffset;
    for (uint8_t d = 0; d < direction; d++) {
        offset += entry.faceQuadCounts[d] * sizeof(BlockModelQuad);
    }

    uint32_t quadIndex = offset / sizeof(BlockModelQuad);
    if (quadIndex + outCount > quads_.size()) { outCount = 0; return nullptr; }
    return &quads_[quadIndex];
}

glm::u8vec3 BlockModelTable::getBiomeTint(uint16_t biomeId, uint8_t tintType) const {
    if (tintType > 2) return glm::u8vec3(255, 255, 255);
    uint32_t idx = biomeId * 3 + tintType;
    if (idx >= biomeTints_.size()) return glm::u8vec3(255, 255, 255);
    return biomeTints_[idx];
}

