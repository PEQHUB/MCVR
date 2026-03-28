#include "block_model_table.hpp"
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

    // Copy quad array
    quads_.assign(quads, quads + quadCount);

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

void BlockModelTable::loadSpriteBounds(const SpriteUVBounds* bounds, uint32_t count) {
    maxSpriteId_ = 0;
    for (uint32_t i = 0; i < count; i++) {
        if (bounds[i].spriteId > maxSpriteId_)
            maxSpriteId_ = bounds[i].spriteId;
    }

    // Default bounds: full [0,1] range
    SpriteUVBounds defaultBounds{};
    defaultBounds.minU = 0; defaultBounds.maxU = 1;
    defaultBounds.minV = 0; defaultBounds.maxV = 1;

    spriteBounds_.clear();
    spriteBounds_.resize(maxSpriteId_ + 1, defaultBounds);

    for (uint32_t i = 0; i < count; i++) {
        spriteBounds_[bounds[i].spriteId] = bounds[i];
    }

    std::cout << "[BlockModelTable] Loaded " << count << " sprite bounds, "
              << "max spriteId=" << maxSpriteId_ << std::endl;
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

const SpriteUVBounds* BlockModelTable::getSpriteBounds(uint16_t spriteId) const {
    if (spriteId > maxSpriteId_) return nullptr;
    return &spriteBounds_[spriteId];
}
