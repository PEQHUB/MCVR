#include "block_mesher.hpp"
#include "core/render/chunks.hpp"

#include <cstring>
#include <cmath>

using PBRTriangle = vk::VertexFormat::PBRTriangle;

// Direction normals: DOWN, UP, NORTH, SOUTH, WEST, EAST
static const glm::vec3 DIRECTION_NORMALS[6] = {
    { 0, -1,  0}, // DOWN
    { 0,  1,  0}, // UP
    { 0,  0, -1}, // NORTH
    { 0,  0,  1}, // SOUTH
    {-1,  0,  0}, // WEST
    { 1,  0,  0}, // EAST
};

uint32_t BlockMesher::getNeighborState(const SectionInput& input, int x, int y, int z, int direction) {
    // Compute neighbor position
    int nx = x, ny = y, nz = z;
    switch (direction) {
        case 0: ny--; break; // DOWN
        case 1: ny++; break; // UP
        case 2: nz--; break; // NORTH
        case 3: nz++; break; // SOUTH
        case 4: nx--; break; // WEST
        case 5: nx++; break; // EAST
    }

    // Inside section bounds: direct lookup
    if (nx >= 0 && nx < 16 && ny >= 0 && ny < 16 && nz >= 0 && nz < 16) {
        return input.blockStates[ny * 256 + nz * 16 + nx];
    }

    // At section boundary: use neighbor data
    // Each face stores 256 entries in the same indexing as the face plane
    switch (direction) {
        case 0: // DOWN: looking at bottom face of section below, index = z*16+x
            return input.neighborStates[0][nz * 16 + nx];
        case 1: // UP: looking at top face of section above, index = z*16+x
            return input.neighborStates[1][nz * 16 + nx];
        case 2: // NORTH: looking at south face of section to north, index = y*16+x
            return input.neighborStates[2][ny * 16 + nx];
        case 3: // SOUTH: looking at north face of section to south, index = y*16+x
            return input.neighborStates[3][ny * 16 + nx];
        case 4: // WEST: looking at east face of section to west, index = y*16+z
            return input.neighborStates[4][ny * 16 + nz];
        case 5: // EAST: looking at west face of section to east, index = y*16+z
            return input.neighborStates[5][ny * 16 + nz];
    }
    return 0; // air
}

void BlockMesher::emitQuad(std::vector<PBRTriangle>& vertices,
                            std::vector<uint32_t>& indices,
                            const BlockModelQuad& quad,
                            const BlockModelEntry& entry,
                            const BlockModelTable& table,
                            int blockX, int blockY, int blockZ,
                            uint32_t textureId,
                            const glm::vec4* overlayBounds) {
    float lx = static_cast<float>(blockX);
    float ly = static_cast<float>(blockY);
    float lz = static_cast<float>(blockZ);

    glm::vec3 normal = (quad.direction < 6) ? DIRECTION_NORMALS[quad.direction] : glm::vec3(0, 1, 0);

    uint32_t flags = vk::VertexFormat::PBR_FLAG_USE_NORM
                   | vk::VertexFormat::PBR_FLAG_USE_TEXTURE
                   | vk::VertexFormat::PBR_FLAG_BLOCK_GEOMETRY;

    // Biome tint: encode type in flag bits 12-13 for shader-side resolution.
    // Overlay is now handled via SpriteRegistry.overlaySprite — no colorLayer hack needed.
    glm::vec4 color(1.0f);
    if (quad.tintIndex >= 0 && entry.tintColorType <= 2) {
        // Biome-dependent tint (grass, foliage, water) — shader resolves from SSBO
        flags |= (static_cast<uint32_t>(entry.tintColorType + 1) << vk::VertexFormat::PBR_FLAG_BIOME_TINT_SHIFT);
    } else if (quad.tintIndex >= 0 && entry.tintColorType == 3) {
        // Fixed color (birch/spruce leaves, lily pad)
        color = glm::vec4(entry.fixedTintR / 255.0f, entry.fixedTintG / 255.0f,
                          entry.fixedTintB / 255.0f, 1.0f);
        flags |= vk::VertexFormat::PBR_FLAG_USE_COLOR_LAYER;
    }

    uint32_t materialVal = (entry.materialOrdinal != 255) ? (entry.materialOrdinal + 1) : 0;
    uint32_t emissiveBlockType = (entry.emissiveOrdinal & 0xFF)
                               | ((materialVal & 0xFF) << 8)
                               | (entry.isVivid ? 0x10000u : 0u)
                               | ((entry.blockTypeId & 0x7FFF) << 17);

    // Look up sprite UV bounds for atlas→[0,1] conversion.
    // spriteId from Java is now a sequential layer index (assigned during atlas stitching).
    const SpriteUVBounds* bounds = table.getSpriteBounds(quad.spriteId);
    float spriteMinU = bounds ? bounds->minU : 0.0f;
    float spriteMinV = bounds ? bounds->minV : 0.0f;
    float spriteSizeU = bounds ? (bounds->maxU - bounds->minU) : 1.0f;
    float spriteSizeV = bounds ? (bounds->maxV - bounds->minV) : 1.0f;
    // Avoid division by zero for degenerate sprites
    if (spriteSizeU < 1e-6f) spriteSizeU = 1.0f;
    if (spriteSizeV < 1e-6f) spriteSizeV = 1.0f;

    uint32_t baseIdx = static_cast<uint32_t>(vertices.size());

    for (int v = 0; v < 4; v++) {
        PBRTriangle vert;
        std::memset(&vert, 0, sizeof(PBRTriangle));

        vert.pos = glm::vec3(
            lx + quad.positions[v][0],
            ly + quad.positions[v][1],
            lz + quad.positions[v][2]
        );

        vert.flags = flags;
        vert.norm = normal;
        vert.albedoEmission = entry.emissionNits;
        vert.colorLayer = color;
        vert.postBase = vert.pos;
        vert.emissiveBlockType = emissiveBlockType;

        // Convert atlas UVs to [0,1] normalized within sprite
        float atlasU = quad.uvs[v][0];
        float atlasV = quad.uvs[v][1];
        vert.textureUV = glm::vec2(
            (atlasU - spriteMinU) / spriteSizeU,
            (atlasV - spriteMinV) / spriteSizeV
        );

        vert.glintUV = glm::vec2(0.0f);
        vert.textureID = quad.spriteId; // sequential layer index (not atlas GLID)
        vert.glintTexture = 0;
        vert.overlayPacked = 0;
        vert.lightPacked = 0;

        vertices.push_back(vert);
    }

    indices.push_back(baseIdx + 0);
    indices.push_back(baseIdx + 1);
    indices.push_back(baseIdx + 2);
    indices.push_back(baseIdx + 2);
    indices.push_back(baseIdx + 3);
    indices.push_back(baseIdx + 0);
}

BlockMesher::SectionOutput BlockMesher::mesh(const SectionInput& input,
                                              const BlockModelTable& table) {
    SectionOutput output;

    // Reserve typical capacity (most sections have 1000-3000 quads)
    output.solidVertices.reserve(4096);
    output.solidIndices.reserve(6144);

    // Iterate all 4096 blocks in Y-Z-X order (matching Minecraft)
    for (int y = 0; y < 16; y++) {
        for (int z = 0; z < 16; z++) {
            for (int x = 0; x < 16; x++) {
                uint32_t stateId = input.blockStates[y * 256 + z * 16 + x];
                if (stateId == 0) continue; // air (state 0 is always air)

                const BlockModelEntry* entry = table.getEntry(stateId);
                if (!entry) continue; // invisible or unknown

                // Collect light source if emissive
                if (entry->emissiveOrdinal != 255 && entry->emissionNits > 0) {
                    ChunkLightEntry light;
                    light.worldX = static_cast<float>(input.originX + x) + 0.5f;
                    light.worldY = static_cast<float>(input.originY + y) + 0.5f;
                    light.worldZ = static_cast<float>(input.originZ + z) + 0.5f;
                    light.lightTypeId = entry->emissiveOrdinal; // TODO: map to LightSourceRegistry typeId
                    output.lights.push_back(light);
                }

                // Select output buffers based on render layer
                // We use the first quad's renderLayer to determine the geometry type
                // (all quads in a block model typically share the same layer)

                // Process directional faces (face-culled)
                for (int dir = 0; dir < 6; dir++) {
                    // Face culling: skip if neighbor is a full opaque cube
                    uint32_t neighborState = getNeighborState(input, x, y, z, dir);
                    if (neighborState != 0) {
                        const BlockModelEntry* neighborEntry = table.getEntry(neighborState);
                        if (neighborEntry && neighborEntry->isFullOpaqueCube) {
                            continue; // face hidden by opaque neighbor
                        }
                    }

                    uint8_t quadCount = 0;
                    const BlockModelQuad* quads = table.getFaceQuads(*entry, static_cast<uint8_t>(dir), quadCount);
                    if (!quads || quadCount == 0) continue;

                    // RT grass side fix: for grass-type blocks on side faces with
                    // coplanar base+overlay quads, extract overlay sprite bounds and
                    // emit only the base quad with an overlay alpha mask flag.
                    // Grass block sides: overlay quads are still skipped in geometry.
                    // The shader now resolves overlay via SpriteRegistry.overlaySprite.
                    bool isGrassSide = (entry->tintColorType == 0 && dir >= 2 && dir <= 5 && quadCount >= 2);

                    for (uint8_t q = 0; q < quadCount; q++) {
                        const auto& quad = quads[q];

                        // Skip overlay quads on grass sides (shader handles via SpriteRegistry)
                        if (isGrassSide && quad.tintIndex >= 0) continue;

                        auto& verts = (quad.renderLayer >= 3) ? output.translucentVertices
                                    : (quad.renderLayer >= 1) ? output.cutoutVertices
                                    : output.solidVertices;
                        auto& idxs  = (quad.renderLayer >= 3) ? output.translucentIndices
                                    : (quad.renderLayer >= 1) ? output.cutoutIndices
                                    : output.solidIndices;

                        emitQuad(verts, idxs, quad, *entry, table,
                                 x, y, z, input.blockAtlasTextureId, nullptr);
                    }
                }

                // Process unculled quads (direction = 6, always rendered)
                {
                    uint8_t quadCount = 0;
                    const BlockModelQuad* quads = table.getFaceQuads(*entry, 6, quadCount);
                    if (quads && quadCount > 0) {
                        for (uint8_t q = 0; q < quadCount; q++) {
                            const auto& quad = quads[q];
                            auto& verts = (quad.renderLayer >= 3) ? output.translucentVertices
                                        : (quad.renderLayer >= 1) ? output.cutoutVertices
                                        : output.solidVertices;
                            auto& idxs  = (quad.renderLayer >= 3) ? output.translucentIndices
                                        : (quad.renderLayer >= 1) ? output.cutoutIndices
                                        : output.solidIndices;

                            emitQuad(verts, idxs, quad, *entry, table,
                                     x, y, z, input.blockAtlasTextureId);
                        }
                    }
                }
            }
        }
    }

    return output;
}
