#pragma once

#include "core/render/block_model_table.hpp"
#include "core/vulkan/vertex.hpp"
#include "common/shared.hpp"

#include <cstdint>
#include <vector>
#include <glm/glm.hpp>

struct ChunkLightEntry;

/// Stateless block mesher: generates PBRTriangle vertices from block state arrays.
/// Thread-safe — all inputs are const, no shared mutable state.
/// Output is identical to what Java's SectionBuilderMixins + PBRVertexConsumer produce.
class BlockMesher {
  public:
    /// Input: a 16x16x16 section's block state data + context.
    struct SectionInput {
        /// Block states for the 16x16x16 section (palette-decoded global state IDs).
        /// Index: y * 256 + z * 16 + x (Minecraft's Y-Z-X order).
        uint32_t blockStates[4096];

        /// Biome IDs for the 4x4x4 biome grid within this section.
        /// Index: (y/4) * 16 + (z/4) * 4 + (x/4).
        uint16_t biomes[64];

        /// Neighbor block states for face culling at section boundaries.
        /// [direction][index]: 256 block states per face.
        /// DOWN/UP: 16x16 (z*16+x), NORTH/SOUTH: 16x16 (y*16+x), WEST/EAST: 16x16 (y*16+z)
        uint32_t neighborStates[6][256];

        /// Optional one-block halo around the section, indexed as
        /// (y + 1) * 18 * 18 + (z + 1) * 18 + (x + 1), where x/y/z are in [-1, 16].
        /// This gives fluid corner-height sampling the same diagonal neighborhood vanilla sees.
        uint32_t haloStates[18 * 18 * 18]{};
        bool hasHaloStates = false;

        /// Section origin in world coordinates.
        int originX, originY, originZ;

        /// GL texture ID for the block atlas (used as textureID in PBRTriangle).
        uint32_t blockAtlasTextureId;
    };

    /// Output: per-geometry-type vertex/index arrays, ready for greedy meshing + BLAS.
    struct SectionOutput {
        std::vector<vk::VertexFormat::PBRTriangle> solidVertices;
        std::vector<uint32_t> solidIndices;
        std::vector<vk::VertexFormat::PBRTriangle> cutoutVertices;
        std::vector<uint32_t> cutoutIndices;
        std::vector<vk::VertexFormat::PBRTriangle> translucentVertices;
        std::vector<uint32_t> translucentIndices;
        std::vector<ChunkLightEntry> lights;
    };

    /// Mesh a section using the block model table.
    /// Produces output identical to Java's meshing pipeline.
    static SectionOutput mesh(const SectionInput& input, const BlockModelTable& table);

  private:
    /// Get the neighbor block state for face culling.
    /// Returns the block state of the neighbor in the given direction from local pos (x,y,z).
    /// Handles section boundaries using neighborStates.
    static uint32_t getNeighborState(const SectionInput& input, int x, int y, int z, int direction);

    /// Emit 4 PBRTriangle vertices + 6 indices for one quad.
    static void emitQuad(std::vector<vk::VertexFormat::PBRTriangle>& vertices,
                         std::vector<uint32_t>& indices,
                         const BlockModelQuad& quad,
                         const BlockModelEntry& entry,
                         const BlockModelTable& table,
                         int blockX, int blockY, int blockZ,
                         uint32_t textureId,
                         const glm::vec4* overlayBounds = nullptr);

    /// Get block state at arbitrary position within or adjacent to the section.
    /// Handles section boundaries using haloStates when present, otherwise neighborStates.
    static uint32_t getBlockAt(const SectionInput& input, int x, int y, int z);

    /// Get fluid height at a position (0.0 if not fluid, up to 1.0 for submerged).
    static float getFluidHeightAt(const SectionInput& input, const BlockModelTable& table,
                                   int x, int y, int z, uint8_t fluidType);

    /// Generate fluid quads (top, bottom, sides) for a single fluid block.
    static void emitFluidQuads(const SectionInput& input, const BlockModelTable& table,
                                const BlockModelEntry& entry,
                                int x, int y, int z, SectionOutput& output);
};
