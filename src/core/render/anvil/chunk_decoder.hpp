#pragma once

#include "core/render/anvil/nbt.hpp"
#include "core/render/block_state_registry.hpp"
#include <cstdint>
#include <vector>

namespace anvil {

/// Decoded section data, ready to feed into BlockMesher::SectionInput.
struct DecodedSection {
    int32_t sectionY;                 // Section Y coordinate (-4 to 19 for overworld)
    uint32_t blockStates[4096];       // Global state IDs (y*256 + z*16 + x)
    uint16_t biomes[64];              // Biome raw IDs (4×4×4 grid)
    bool empty = true;                // True if section is all air (skip meshing)
};

/// Decoded chunk column — all sections for a single (chunkX, chunkZ).
struct DecodedChunk {
    int32_t chunkX, chunkZ;
    std::vector<DecodedSection> sections; // Sorted by sectionY ascending

    /// Find section by Y coordinate. Returns nullptr if not present.
    const DecodedSection* getSection(int32_t sectionY) const;
};

/// Decodes raw chunk NBT data into block state arrays.
/// Uses BlockStateRegistry to resolve palette strings → globalStateIds.
class ChunkDecoder {
  public:
    /// Decode a chunk from raw NBT bytes (already decompressed).
    /// Returns empty DecodedChunk (no sections) on failure.
    static DecodedChunk decode(const uint8_t* nbtData, size_t nbtSize,
                               int32_t chunkX, int32_t chunkZ,
                               const BlockStateRegistry& registry);

  private:
    /// Decode the packed long array (block_states.data or biomes.data).
    /// Minecraft's packed format: bitsPerEntry indices packed into int64s,
    /// indices do NOT cross 64-bit boundaries.
    static void decodePacked(const std::vector<int64_t>& packed,
                             int32_t bitsPerEntry, int32_t count,
                             uint32_t* output);
};

} // namespace anvil
