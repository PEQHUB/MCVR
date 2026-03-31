#include "core/render/anvil/chunk_decoder.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>

namespace anvil {

const DecodedSection* DecodedChunk::getSection(int32_t sectionY) const {
    for (auto& s : sections) {
        if (s.sectionY == sectionY) return &s;
    }
    return nullptr;
}

DecodedChunk ChunkDecoder::decode(const uint8_t* nbtData, size_t nbtSize,
                                  int32_t chunkX, int32_t chunkZ,
                                  const BlockStateRegistry& registry) {
    DecodedChunk chunk;
    chunk.chunkX = chunkX;
    chunk.chunkZ = chunkZ;

    NbtTag root;
    try {
        root = NbtReader::parse(nbtData, nbtSize);
    } catch (const std::exception& e) {
        std::cerr << "[ChunkDecoder] NBT parse error for (" << chunkX << "," << chunkZ
                  << "): " << e.what() << std::endl;
        return chunk;
    }

    // sections list
    auto* sectionsList = root.getList("sections");
    if (!sectionsList) return chunk;

    for (auto& sectionTag : sectionsList->listItems) {
        if (sectionTag.type != TagType::Compound) continue;

        DecodedSection section;
        section.sectionY = sectionTag.getByte("Y", 0);
        std::memset(section.blockStates, 0, sizeof(section.blockStates));
        std::memset(section.biomes, 0, sizeof(section.biomes));

        // ---- Block states ----
        auto* blockStatesTag = sectionTag.getCompound("block_states");
        if (blockStatesTag) {
            auto* palette = blockStatesTag->getList("palette");
            if (palette && !palette->listItems.empty()) {
                // Build state ID array from palette
                std::vector<uint32_t> paletteIds;
                paletteIds.reserve(palette->listItems.size());

                for (auto& entry : palette->listItems) {
                    if (entry.type != TagType::Compound) {
                        paletteIds.push_back(0); // air
                        continue;
                    }

                    const std::string& name = entry.getString("Name");
                    auto* propsTag = entry.getCompound("Properties");

                    if (propsTag && !propsTag->compounds.empty()) {
                        // Build properties vector for resolve()
                        std::vector<std::pair<std::string, std::string>> props;
                        props.reserve(propsTag->compounds.size());
                        for (auto& [key, val] : propsTag->compounds) {
                            if (val.type == TagType::String) {
                                props.emplace_back(key, val.stringVal);
                            }
                        }
                        paletteIds.push_back(registry.resolve(name, props));
                    } else {
                        paletteIds.push_back(registry.resolve(name));
                    }
                }

                if (paletteIds.size() == 1) {
                    // Single-entry palette: entire section is this block
                    uint32_t id = paletteIds[0];
                    for (int i = 0; i < 4096; i++) section.blockStates[i] = id;
                    section.empty = (id == 0); // 0 = air
                } else {
                    // Decode packed long array
                    auto& data = blockStatesTag->getLongArray("data");
                    if (!data.empty()) {
                        int32_t bitsPerEntry = std::max(4, static_cast<int32_t>(
                            std::ceil(std::log2(static_cast<double>(paletteIds.size())))));
                        uint32_t indices[4096];
                        decodePacked(data, bitsPerEntry, 4096, indices);

                        bool allAir = true;
                        for (int i = 0; i < 4096; i++) {
                            uint32_t idx = indices[i];
                            uint32_t stateId = (idx < paletteIds.size()) ? paletteIds[idx] : 0;
                            section.blockStates[i] = stateId;
                            if (stateId != 0) allAir = false;
                        }
                        section.empty = allAir;
                    }
                }
            }
        }

        // ---- Biomes ----
        auto* biomesTag = sectionTag.getCompound("biomes");
        if (biomesTag) {
            auto* biomePalette = biomesTag->getList("palette");
            if (biomePalette && !biomePalette->listItems.empty()) {
                // Biome palette is string names (e.g., "minecraft:plains")
                // We store biome raw IDs. For now, use palette index as raw ID
                // (the exact biome ID mapping is less critical — affects only tinting).
                std::vector<uint16_t> biomePaletteIds;
                for (size_t i = 0; i < biomePalette->listItems.size(); i++) {
                    // Use palette index as biome raw ID for now.
                    // A proper biome registry bridge (like BlockStateRegistry) would be needed
                    // for exact biome color matching.
                    biomePaletteIds.push_back(static_cast<uint16_t>(i));
                }

                if (biomePaletteIds.size() == 1) {
                    for (int i = 0; i < 64; i++) section.biomes[i] = biomePaletteIds[0];
                } else {
                    auto& biomeData = biomesTag->getLongArray("data");
                    if (!biomeData.empty()) {
                        int32_t bitsPerEntry = std::max(1, static_cast<int32_t>(
                            std::ceil(std::log2(static_cast<double>(biomePaletteIds.size())))));
                        uint32_t biomeIndices[64];
                        decodePacked(biomeData, bitsPerEntry, 64, biomeIndices);
                        for (int i = 0; i < 64; i++) {
                            uint32_t idx = biomeIndices[i];
                            section.biomes[i] = (idx < biomePaletteIds.size()) ? biomePaletteIds[idx] : 0;
                        }
                    }
                }
            }
        }

        chunk.sections.push_back(std::move(section));
    }

    // Sort sections by Y for predictable access
    std::sort(chunk.sections.begin(), chunk.sections.end(),
              [](const DecodedSection& a, const DecodedSection& b) {
                  return a.sectionY < b.sectionY;
              });

    return chunk;
}

void ChunkDecoder::decodePacked(const std::vector<int64_t>& packed,
                                int32_t bitsPerEntry, int32_t count,
                                uint32_t* output) {
    // Minecraft's packed format (1.16+):
    // Indices are packed into int64s, but an index NEVER crosses a 64-bit boundary.
    // Each int64 holds floor(64 / bitsPerEntry) indices, remaining high bits are unused.
    int32_t indicesPerLong = 64 / bitsPerEntry;
    uint64_t mask = (1ULL << bitsPerEntry) - 1;

    int32_t outputIdx = 0;
    for (size_t longIdx = 0; longIdx < packed.size() && outputIdx < count; longIdx++) {
        uint64_t val = static_cast<uint64_t>(packed[longIdx]);
        for (int32_t j = 0; j < indicesPerLong && outputIdx < count; j++) {
            output[outputIdx++] = static_cast<uint32_t>(val & mask);
            val >>= bitsPerEntry;
        }
    }

    // Zero-fill any remaining entries (shouldn't happen with correct data)
    while (outputIdx < count) output[outputIdx++] = 0;
}

} // namespace anvil
