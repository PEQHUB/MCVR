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

    // UVs are pre-normalized to [0,1] within sprite bounds during model table load.
    // No per-vertex UV conversion needed here.

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

        vert.textureUV = glm::vec2(quad.uvs[v][0], quad.uvs[v][1]);

        vert.glintUV = glm::vec2(0.0f);
        vert.textureID = quad.spriteId; // deterministic sorted index
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

// ---- Fluid meshing ----

uint32_t BlockMesher::getBlockAt(const SectionInput& input, int x, int y, int z) {
    if (x >= 0 && x < 16 && y >= 0 && y < 16 && z >= 0 && z < 16)
        return input.blockStates[y * 256 + z * 16 + x];
    // Single-axis out of bounds: use neighbor face data
    if (y < 0)  return input.neighborStates[0][z * 16 + x];    // DOWN
    if (y > 15) return input.neighborStates[1][z * 16 + x];    // UP
    if (z < 0)  return input.neighborStates[2][y * 16 + x];    // NORTH
    if (z > 15) return input.neighborStates[3][y * 16 + x];    // SOUTH
    if (x < 0)  return input.neighborStates[4][y * 16 + z];    // WEST
    if (x > 15) return input.neighborStates[5][y * 16 + z];    // EAST
    return 0; // diagonal out-of-bounds: treat as air
}

float BlockMesher::getFluidHeightAt(const SectionInput& input, const BlockModelTable& table,
                                     int x, int y, int z, uint8_t fluidType) {
    uint32_t stateId = getBlockAt(input, x, y, z);
    if (stateId == 0) return 0.0f;
    const BlockModelEntry* entry = table.getEntry(stateId);
    if (!entry || entry->fluidType == 0) return 0.0f;
    // Must be same fluid type (water=1, lava=2)
    uint8_t ft = entry->fluidType;
    uint8_t baseType = (ft <= 2) ? ft : ((ft == 3) ? 1 : 2); // normalize flowing→base
    uint8_t queryBase = (fluidType <= 2) ? fluidType : ((fluidType == 3) ? 1 : 2);
    if (baseType != queryBase) return 0.0f;

    // Check if fluid above → submerged = full height
    uint32_t aboveState = getBlockAt(input, x, y + 1, z);
    if (aboveState != 0) {
        const BlockModelEntry* aboveEntry = table.getEntry(aboveState);
        if (aboveEntry && aboveEntry->fluidType != 0) {
            uint8_t aboveBase = (aboveEntry->fluidType <= 2) ? aboveEntry->fluidType
                              : ((aboveEntry->fluidType == 3) ? 1 : 2);
            if (aboveBase == queryBase) return 1.0f;
        }
    }

    // Waterlogged blocks (renderType!=2) are always source-level water;
    // their totalQuadCount field holds model quad count, not fluid level.
    if (entry->renderType != 2) return 8.0f / 9.0f;

    uint8_t level = entry->fluidLevel();
    if (level >= 8) return 8.0f / 9.0f; // source block
    if (level == 0) return 1.0f;         // falling
    return (8.0f - static_cast<float>(level)) / 9.0f; // flowing 1-7
}

static float averageCornerHeight(float h0, float h1, float h2, float h3) {
    // Average non-zero heights (Minecraft's corner averaging algorithm)
    float sum = 0; int count = 0;
    if (h0 > 0) { sum += h0; count++; }
    if (h1 > 0) { sum += h1; count++; }
    if (h2 > 0) { sum += h2; count++; }
    if (h3 > 0) { sum += h3; count++; }
    return count > 0 ? sum / count : 0.0f;
}

static void emitFluidFace(std::vector<PBRTriangle>& vertices, std::vector<uint32_t>& indices,
                           const glm::vec3 pos[4], glm::vec3 normal,
                           uint16_t spriteId, uint8_t fluidMaterialOrdinal,
                           uint8_t tintType) {
    uint32_t flags = vk::VertexFormat::PBR_FLAG_USE_NORM
                   | vk::VertexFormat::PBR_FLAG_USE_TEXTURE
                   | vk::VertexFormat::PBR_FLAG_BLOCK_GEOMETRY;

    // Biome tint (use override tintType — allows waterlogged blocks to force TINT_WATER)
    if (tintType <= 2) {
        flags |= (static_cast<uint32_t>(tintType + 1) << vk::VertexFormat::PBR_FLAG_BIOME_TINT_SHIFT);
    }

    // Always use the fluid's own material (e.g. WATER_MAT ordinal 103), never the host block's.
    // This ensures waterlogged block water surfaces get proper water rendering (IOR, roughness).
    uint32_t materialVal = (fluidMaterialOrdinal != 255) ? (fluidMaterialOrdinal + 1) : 0;
    uint32_t emissiveBlockType = 0xFF  // emissiveOrdinal=255 (no emission)
                               | ((materialVal & 0xFF) << 8);

    // Simple UV mapping: 4 corners of the sprite [0,1]
    static const glm::vec2 uvs[4] = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};

    uint32_t baseIdx = static_cast<uint32_t>(vertices.size());
    for (int v = 0; v < 4; v++) {
        PBRTriangle vert;
        std::memset(&vert, 0, sizeof(PBRTriangle));
        vert.pos = pos[v];
        vert.flags = flags;
        vert.norm = normal;
        vert.albedoEmission = 0.0f;
        vert.colorLayer = glm::vec4(1.0f);
        vert.postBase = vert.pos;
        vert.emissiveBlockType = emissiveBlockType;
        vert.textureUV = uvs[v];
        vert.textureID = spriteId;
        vertices.push_back(vert);
    }
    // Double-sided: emit both windings so fluid faces are visible from both directions
    // Front face (outward-facing, e.g. top face visible from above)
    indices.push_back(baseIdx + 2);
    indices.push_back(baseIdx + 1);
    indices.push_back(baseIdx + 0);
    indices.push_back(baseIdx + 0);
    indices.push_back(baseIdx + 3);
    indices.push_back(baseIdx + 2);
    // Back face (inward-facing, e.g. top face visible from underwater)
    indices.push_back(baseIdx + 0);
    indices.push_back(baseIdx + 1);
    indices.push_back(baseIdx + 2);
    indices.push_back(baseIdx + 2);
    indices.push_back(baseIdx + 3);
    indices.push_back(baseIdx + 0);
}

void BlockMesher::emitFluidQuads(const SectionInput& input, const BlockModelTable& table,
                                  const BlockModelEntry& entry,
                                  int x, int y, int z, SectionOutput& output) {
    float bx = static_cast<float>(x);
    float by = static_cast<float>(y);
    float bz = static_cast<float>(z);

    uint8_t fluidType = entry.fluidType;
    bool isWater = (fluidType == 1 || fluidType == 3);
    auto& verts = isWater ? output.translucentVertices : output.solidVertices;
    auto& idxs  = isWater ? output.translucentIndices  : output.solidIndices;

    // Waterlogged blocks (renderType==1) store model data in quadOffset, not sprite IDs.
    // Use table-stored water data instead. Pure fluids (renderType==2) use entry data.
    bool waterlogged = (entry.renderType != 2);
    uint16_t stillSprite = waterlogged ? table.waterSpriteStill() : entry.fluidSpriteStill();
    uint16_t flowSprite  = waterlogged ? table.waterSpriteFlow()  : entry.fluidSpriteFlow();
    uint8_t fluidTint = waterlogged ? 2 : entry.tintColorType;
    // Material ordinal: always use water's material (WATER_MAT = ordinal 103) for correct IOR/roughness.
    // Waterlogged blocks get it from the table; pure water entries already have it.
    uint8_t fluidMaterial = waterlogged ? table.waterMaterialOrdinal() : entry.materialOrdinal;

    // Compute corner heights for top face
    float hCenter = getFluidHeightAt(input, table, x, y, z, fluidType);
    float hN  = getFluidHeightAt(input, table, x, y, z - 1, fluidType);
    float hS  = getFluidHeightAt(input, table, x, y, z + 1, fluidType);
    float hE  = getFluidHeightAt(input, table, x + 1, y, z, fluidType);
    float hW  = getFluidHeightAt(input, table, x - 1, y, z, fluidType);
    float hNE = getFluidHeightAt(input, table, x + 1, y, z - 1, fluidType);
    float hNW = getFluidHeightAt(input, table, x - 1, y, z - 1, fluidType);
    float hSE = getFluidHeightAt(input, table, x + 1, y, z + 1, fluidType);
    float hSW = getFluidHeightAt(input, table, x - 1, y, z + 1, fluidType);

    // Corner heights: average of 4 blocks touching each corner
    float cNW = averageCornerHeight(hCenter, hN, hW, hNW);
    float cNE = averageCornerHeight(hCenter, hN, hE, hNE);
    float cSW = averageCornerHeight(hCenter, hS, hW, hSW);
    float cSE = averageCornerHeight(hCenter, hS, hE, hSE);

    // Helper: check if neighbor is same fluid
    auto isSameFluid = [&](int nx, int ny, int nz) -> bool {
        uint32_t ns = getBlockAt(input, nx, ny, nz);
        if (ns == 0) return false;
        const BlockModelEntry* ne = table.getEntry(ns);
        if (!ne || ne->fluidType == 0) return false;
        uint8_t nBase = (ne->fluidType <= 2) ? ne->fluidType : ((ne->fluidType == 3) ? 1 : 2);
        uint8_t myBase = isWater ? 1 : 2;
        return nBase == myBase;
    };

    auto isOpaqueAt = [&](int nx, int ny, int nz) -> bool {
        uint32_t ns = getBlockAt(input, nx, ny, nz);
        if (ns == 0) return false;
        const BlockModelEntry* ne = table.getEntry(ns);
        return ne && ne->isFullOpaqueCube;
    };

    // TOP FACE: render if block above is not same fluid
    if (!isSameFluid(x, y + 1, z)) {
        glm::vec3 pos[4] = {
            {bx,     by + cNW, bz},       // NW
            {bx + 1, by + cNE, bz},       // NE
            {bx + 1, by + cSE, bz + 1},   // SE
            {bx,     by + cSW, bz + 1},   // SW
        };
        // Normal from height gradient
        glm::vec3 normal = glm::normalize(glm::vec3(cNW - cNE + cSW - cSE, 2.0f, cNW + cNE - cSW - cSE));
        emitFluidFace(verts, idxs, pos, normal, stillSprite, fluidMaterial, fluidTint);
    }

    // BOTTOM FACE: render if block below is not same fluid and not opaque
    if (!isSameFluid(x, y - 1, z) && !isOpaqueAt(x, y - 1, z)) {
        glm::vec3 pos[4] = {
            {bx,     by + 0.001f, bz + 1},
            {bx + 1, by + 0.001f, bz + 1},
            {bx + 1, by + 0.001f, bz},
            {bx,     by + 0.001f, bz},
        };
        emitFluidFace(verts, idxs, pos, {0, -1, 0}, stillSprite, fluidMaterial, fluidTint);
    }

    // SIDE FACES: render if neighbor is not same fluid and not full opaque
    // NORTH (z-1)
    if (!isSameFluid(x, y, z - 1) && !isOpaqueAt(x, y, z - 1)) {
        float hL = cNW, hR = cNE;
        glm::vec3 pos[4] = {
            {bx + 1, by + hR, bz},
            {bx,     by + hL, bz},
            {bx,     by,      bz},
            {bx + 1, by,      bz},
        };
        emitFluidFace(verts, idxs, pos, {0, 0, -1}, flowSprite, fluidMaterial, fluidTint);
    }

    // SOUTH (z+1)
    if (!isSameFluid(x, y, z + 1) && !isOpaqueAt(x, y, z + 1)) {
        float hL = cSE, hR = cSW;
        glm::vec3 pos[4] = {
            {bx,     by + hR, bz + 1},
            {bx + 1, by + hL, bz + 1},
            {bx + 1, by,      bz + 1},
            {bx,     by,      bz + 1},
        };
        emitFluidFace(verts, idxs, pos, {0, 0, 1}, flowSprite, fluidMaterial, fluidTint);
    }

    // WEST (x-1)
    if (!isSameFluid(x - 1, y, z) && !isOpaqueAt(x - 1, y, z)) {
        float hL = cSW, hR = cNW;
        glm::vec3 pos[4] = {
            {bx, by + hR, bz},
            {bx, by + hL, bz + 1},
            {bx, by,      bz + 1},
            {bx, by,      bz},
        };
        emitFluidFace(verts, idxs, pos, {-1, 0, 0}, flowSprite, fluidMaterial, fluidTint);
    }

    // EAST (x+1)
    if (!isSameFluid(x + 1, y, z) && !isOpaqueAt(x + 1, y, z)) {
        float hL = cNE, hR = cSE;
        glm::vec3 pos[4] = {
            {bx + 1, by + hR, bz + 1},
            {bx + 1, by + hL, bz},
            {bx + 1, by,      bz},
            {bx + 1, by,      bz + 1},
        };
        emitFluidFace(verts, idxs, pos, {1, 0, 0}, flowSprite, fluidMaterial, fluidTint);
    }
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

                // Fluid blocks: generate dynamic fluid quads
                if (entry->fluidType != 0) {
                    emitFluidQuads(input, table, *entry, x, y, z, output);
                    // Pure fluids (renderType==2): done, no model quads.
                    // Waterlogged (renderType==1): fall through to also emit model quads.
                    if (entry->renderType == 2) continue;
                }

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
