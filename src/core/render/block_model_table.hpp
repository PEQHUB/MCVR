#pragma once

#include <cstdint>
#include <vector>
#include <glm/glm.hpp>

/// Pre-baked block model data, serialized from Java at resource load.
/// Immutable after load — safe for concurrent reads from any thread.
///
/// Layout matches Java's BlockModelBridge serialization format exactly.
/// All structs are tightly packed for direct memcpy from JNI.

#pragma pack(push, 1)

/// Per-face quad from a block model (88 bytes).
struct BlockModelQuad {
    float positions[4][3]; // 48B: 4 vertices × xyz in [0,1] block-local space
    float uvs[4][2];       // 32B: 4 vertices × uv in atlas space
    int8_t  tintIndex;     // 1B:  -1 = no tint, 0-2 = tint type
    uint8_t direction;     // 1B:  0-5 = face direction (DOWN,UP,N,S,W,E), 6 = unculled
    uint8_t renderLayer;   // 1B:  0 = solid, 1 = cutout, 2 = cutout_mipped, 3 = translucent
    uint8_t shade;         // 1B:  ambient occlusion shade flag
    uint16_t spriteId;     // 2B:  texture atlas sprite index
    uint16_t padding;      // 2B:  alignment
};
static_assert(sizeof(BlockModelQuad) == 88, "BlockModelQuad must be 88 bytes");

/// Per-block-state entry (32 bytes).
struct BlockModelEntry {
    uint32_t globalStateId;          // 4B: Minecraft registry raw ID
    uint8_t  renderType;             // 1B: 0=invisible, 1=model, 2=fluid
    uint8_t  isFullOpaqueCube;       // 1B: 1 if all 6 faces are solid (for neighbor face culling)
    uint8_t  emissiveOrdinal;        // 1B: EmissiveBlock.ordinal() or 255
    uint8_t  materialOrdinal;        // 1B: MaterialBlock ordinal or 255
    uint16_t blockTypeId;            // 2B: BlockTypeIdRegistry packed ID + thin plant semantic
    uint8_t  isVivid;               // 1B: VividColorBlock flag
    uint8_t  fluidType;             // 1B: 0=none, 1=water, 2=lava, 3=flowing_water, 4=flowing_lava
    uint32_t quadOffset;             // 4B: byte offset into quad array (NOT index — for direct pointer math)
    uint8_t  faceQuadCounts[7];      // 7B: quad count per direction (DOWN,UP,N,S,W,E,UNCULLED)
    uint8_t  totalQuadCount;         // 1B: sum of faceQuadCounts
    float    emissionNits;           // 4B: emission intensity in nits (0 = not emissive)
    uint8_t  tintColorType;          // 1B: 0=grass, 1=foliage, 2=water, 3=fixed, 255=none
    uint8_t  fixedTintR;             // 1B: fixed tint red   (when tintColorType=3)
    uint8_t  fixedTintG;             // 1B: fixed tint green (when tintColorType=3)
    uint8_t  fixedTintB;             // 1B: fixed tint blue  (when tintColorType=3)

    // Fluid-specific accessors (valid only when fluidType != 0).
    // These read repurposed fields: quadOffset stores packed sprite IDs, totalQuadCount stores level.
    uint8_t  fluidLevel()       const { return totalQuadCount; }
    uint16_t fluidSpriteStill() const { return static_cast<uint16_t>(quadOffset & 0xFFFF); }
    uint16_t fluidSpriteFlow()  const { return static_cast<uint16_t>((quadOffset >> 16) & 0xFFFF); }
};
static_assert(sizeof(BlockModelEntry) == 32, "BlockModelEntry must be 32 bytes");

/// Biome tint entry (8 bytes).
struct BiomeTintEntry {
    uint16_t biomeRawId;   // 2B: biome registry raw ID
    uint8_t  tintType;     // 1B: 0=grass, 1=foliage, 2=water
    uint8_t  r, g, b;      // 3B: RGB color
    uint16_t padding;       // 2B: alignment
};
static_assert(sizeof(BiomeTintEntry) == 8, "BiomeTintEntry must be 8 bytes");

#pragma pack(pop)

/// Immutable block model lookup table.
/// Loaded once from Java, queried by block state ID during C++ meshing.
class BlockModelTable {
  public:
    /// Load from serialized Java data (memcpy-safe, packed structs).
    void load(const BlockModelEntry* entries, uint32_t entryCount,
              const BlockModelQuad* quads, uint32_t quadCount);

    /// Load biome tint colors.
    void loadBiomeTints(const BiomeTintEntry* tints, uint32_t count);

    /// Pre-normalize quad UVs from atlas space to [0,1] within sprite bounds.
    /// Must be called after load() and after TextureSystem is finalized.
    void normalizeQuadUVs();

    /// Query model data for a global block state ID. Returns nullptr if invisible/air.
    const BlockModelEntry* getEntry(uint32_t globalStateId) const;

    /// Get face quads for a specific direction (0-5) or unculled (6).
    /// Returns pointer to first quad and count.
    const BlockModelQuad* getFaceQuads(const BlockModelEntry& entry, uint8_t direction,
                                       uint8_t& outCount) const;

    /// Get biome tint color. Returns white (255,255,255) if not found.
    glm::u8vec3 getBiomeTint(uint16_t biomeId, uint8_t tintType) const;

    /// Is the table loaded and ready?
    bool isLoaded() const { return !entries_.empty(); }
    uint64_t generation() const { return generation_; }

    /// Total block states loaded.
    uint32_t stateCount() const { return static_cast<uint32_t>(entries_.size()); }

    /// Max global state ID (for bounds checking).
    uint32_t maxStateId() const { return maxStateId_; }

    /// Water data (extracted from pure water block entry during load).
    /// Used by mesher for waterlogged blocks whose fields store model data, not fluid data.
    uint16_t waterSpriteStill() const { return waterSpriteStill_; }
    uint16_t waterSpriteFlow()  const { return waterSpriteFlow_; }
    uint8_t  waterMaterialOrdinal() const { return waterMaterialOrdinal_; }

  private:
    std::vector<BlockModelEntry> entries_;     // Indexed by globalStateId (sparse — gaps are zeroed)
    std::vector<BlockModelQuad> quads_;         // Flat quad array, entries index into this
    uint32_t maxStateId_ = 0;

    // Biome tint lookup: biomeId * 3 + tintType → RGB
    std::vector<glm::u8vec3> biomeTints_;       // Indexed by (biomeId * 3 + tintType)
    uint32_t maxBiomeId_ = 0;

    // Guard: UVs are only normalized once per load
    bool uvsNormalized_ = false;
    uint64_t generation_ = 0;

    // Water data (from pure water FluidBlock, for waterlogged block meshing)
    uint16_t waterSpriteStill_ = 0;
    uint16_t waterSpriteFlow_  = 0;
    uint8_t  waterMaterialOrdinal_ = 255;
};
