#include "block_mesher.hpp"
#include "core/render/chunks.hpp"

#include <cstring>
#include <cmath>
#include <algorithm>

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

    constexpr uint8_t TINT_FLAG_THIN_CUTOUT_PLANT = 0x80u;
    uint8_t tintType = entry.tintColorType & 0x7Fu;

    uint32_t flags = vk::VertexFormat::PBR_FLAG_USE_NORM
                   | vk::VertexFormat::PBR_FLAG_USE_TEXTURE
                   | vk::VertexFormat::PBR_FLAG_BLOCK_GEOMETRY;
    if (entry.tintColorType != 255 && (entry.tintColorType & TINT_FLAG_THIN_CUTOUT_PLANT) != 0) {
        flags |= vk::VertexFormat::PBR_FLAG_THIN_CUTOUT_CARD;
    }

    // Biome tint: encode type in flag bits 12-13 for shader-side resolution.
    // Overlay is now handled via SpriteRegistry.overlaySprite — no colorLayer hack needed.
    glm::vec4 color(1.0f);
    if (quad.tintIndex >= 0 && tintType <= 2) {
        // Biome-dependent tint (grass, foliage, water) — shader resolves from SSBO
        flags |= (static_cast<uint32_t>(tintType + 1) << vk::VertexFormat::PBR_FLAG_BIOME_TINT_SHIFT);
    } else if (quad.tintIndex >= 0 && tintType == 3) {
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

    if (input.hasHaloStates &&
        x >= -1 && x <= 16 &&
        y >= -1 && y <= 16 &&
        z >= -1 && z <= 16) {
        return input.haloStates[(y + 1) * 18 * 18 + (z + 1) * 18 + (x + 1)];
    }

    // Neighbor faces only cover single-axis section boundaries. Diagonal samples
    // happen during fluid corner-height averaging; treat those as air instead of
    // indexing the wrong row of a neighbor face.
    const int outAxes = (x < 0 || x > 15 ? 1 : 0)
                      + (y < 0 || y > 15 ? 1 : 0)
                      + (z < 0 || z > 15 ? 1 : 0);
    if (outAxes != 1) return 0;

    // Single-axis out of bounds: use neighbor face data
    if (y < 0)  return input.neighborStates[0][z * 16 + x];    // DOWN
    if (y > 15) return input.neighborStates[1][z * 16 + x];    // UP
    if (z < 0)  return input.neighborStates[2][y * 16 + x];    // NORTH
    if (z > 15) return input.neighborStates[3][y * 16 + x];    // SOUTH
    if (x < 0)  return input.neighborStates[4][y * 16 + z];    // WEST
    if (x > 15) return input.neighborStates[5][y * 16 + z];    // EAST
    return 0;
}

float BlockMesher::getFluidHeightAt(const SectionInput& input, const BlockModelTable& table,
                                     int x, int y, int z, uint8_t fluidType) {
    uint32_t stateId = getBlockAt(input, x, y, z);
    if (stateId == 0) return 0.0f;
    const BlockModelEntry* entry = table.getEntry(stateId);
    if (!entry) return 0.0f;
    if (entry->fluidType == 0) return entry->isFullOpaqueCube ? -1.0f : 0.0f;
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

    // FluidState.getLevel(): 8=source, 7=highest flowing, 1=lowest flowing, 0=empty/falling
    uint8_t level = entry->fluidLevel();
    if (level == 0) return 8.0f / 9.0f; // falling — same visual height as source
    return static_cast<float>(level) / 9.0f; // source=8/9, flowing 1-7 proportional
}

static void addVanillaFluidHeight(float& weightedSum, float& weight, float height) {
    if (height >= 0.8f) {
        weightedSum += height * 10.0f;
        weight += 10.0f;
    } else if (height >= 0.0f) {
        weightedSum += height;
        weight += 1.0f;
    }
}

static float calculateVanillaCornerHeight(float originHeight,
                                          float northSouthHeight,
                                          float eastWestHeight,
                                          float diagonalHeight) {
    if (eastWestHeight >= 1.0f || northSouthHeight >= 1.0f) {
        return 1.0f;
    }

    float weightedSum = 0.0f;
    float weight = 0.0f;
    if (eastWestHeight > 0.0f || northSouthHeight > 0.0f) {
        if (diagonalHeight >= 1.0f) {
            return 1.0f;
        }
        addVanillaFluidHeight(weightedSum, weight, diagonalHeight);
    }

    addVanillaFluidHeight(weightedSum, weight, originHeight);
    addVanillaFluidHeight(weightedSum, weight, eastWestHeight);
    addVanillaFluidHeight(weightedSum, weight, northSouthHeight);
    return weight > 0.0f ? weightedSum / weight : 0.0f;
}

static void emitFluidFace(std::vector<PBRTriangle>& vertices, std::vector<uint32_t>& indices,
                           const glm::vec3 pos[4], glm::vec3 normal,
                           uint16_t spriteId, uint8_t fluidMaterialOrdinal,
                           uint8_t tintType, const BlockModelEntry& entry,
                           const glm::vec2* customUvs = nullptr) {
    uint32_t flags = vk::VertexFormat::PBR_FLAG_USE_NORM
                   | vk::VertexFormat::PBR_FLAG_USE_TEXTURE
                   | vk::VertexFormat::PBR_FLAG_BLOCK_GEOMETRY
                   | vk::VertexFormat::PBR_FLAG_FLUID_GEOMETRY;

    // Biome tint (use override tintType — allows waterlogged blocks to force TINT_WATER)
    if (tintType <= 2) {
        flags |= (static_cast<uint32_t>(tintType + 1) << vk::VertexFormat::PBR_FLAG_BIOME_TINT_SHIFT);
    }

    // Pack emission + material identically to the solid block path
    uint32_t materialVal = (fluidMaterialOrdinal != 255) ? (fluidMaterialOrdinal + 1) : 0;
    uint32_t emissiveBlockType = (entry.emissiveOrdinal & 0xFF)
                               | ((materialVal & 0xFF) << 8)
                               | (entry.isVivid ? 0x10000u : 0u)
                               | ((entry.blockTypeId & 0x7FFF) << 17);

    // Simple UV mapping: 4 corners of the sprite [0,1]
    static const glm::vec2 uvs[4] = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
    const glm::vec2* faceUvs = customUvs ? customUvs : uvs;

    uint32_t baseIdx = static_cast<uint32_t>(vertices.size());
    for (int v = 0; v < 4; v++) {
        PBRTriangle vert;
        std::memset(&vert, 0, sizeof(PBRTriangle));
        vert.pos = pos[v];
        vert.flags = flags;
        vert.norm = normal;
        vert.albedoEmission = entry.emissionNits;
        vert.colorLayer = glm::vec4(1.0f);
        vert.postBase = vert.pos;
        vert.emissiveBlockType = emissiveBlockType;
        vert.textureUV = faceUvs[v];
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

    float cNW, cNE, cSW, cSE;
    if (hCenter >= 1.0f) {
        cNW = cNE = cSW = cSE = 1.0f;
    } else {
        cNW = calculateVanillaCornerHeight(hCenter, hN, hW, hNW);
        cNE = calculateVanillaCornerHeight(hCenter, hN, hE, hNE);
        cSW = calculateVanillaCornerHeight(hCenter, hS, hW, hSW);
        cSE = calculateVanillaCornerHeight(hCenter, hS, hE, hSE);
    }

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

    auto isSideCoveredAt = [&](int nx, int ny, int nz, int direction, float height) -> bool {
        uint32_t ns = getBlockAt(input, nx, ny, nz);
        if (ns == 0) return false;
        const BlockModelEntry* ne = table.getEntry(ns);
        if (!ne || !ne->isFullOpaqueCube) return false;
        // Mirrors vanilla FluidRenderer.isSideCovered for full-cube culling faces:
        // horizontal/down sides are covered by a full cube, but UP only covers a full-height fluid.
        return direction != 1 || height >= 0.999f;
    };

    bool renderBottom = !isSameFluid(x, y - 1, z) && !isSideCoveredAt(x, y - 1, z, 0, 8.0f / 9.0f);
    float bottomYOffset = renderBottom ? 0.001f : 0.0f;

    float sideNW = cNW;
    float sideNE = cNE;
    float sideSW = cSW;
    float sideSE = cSE;

    float minSurfaceHeight = std::min(std::min(cNW, cSW), std::min(cSE, cNE));
    bool renderTop = !isSameFluid(x, y + 1, z) && !isSideCoveredAt(x, y + 1, z, 1, minSurfaceHeight);

    // TOP FACE: render if block above is not same fluid and vanilla culling would not cover it.
    if (renderTop) {
        sideNW = std::max(0.0f, cNW - 0.001f);
        sideNE = std::max(0.0f, cNE - 0.001f);
        sideSE = std::max(0.0f, cSE - 0.001f);
        sideSW = std::max(0.0f, cSW - 0.001f);
        glm::vec3 pos[4] = {
            {bx,     by + sideNW, bz},       // NW
            {bx + 1, by + sideNE, bz},       // NE
            {bx + 1, by + sideSE, bz + 1},   // SE
            {bx,     by + sideSW, bz + 1},   // SW
        };
        // Normal from height gradient
        glm::vec3 normal = glm::normalize(glm::vec3(sideNW - sideNE + sideSW - sideSE, 1.0f, sideNW + sideNE - sideSW - sideSE));
        bool flowingSurface = (fluidType == 3 || fluidType == 4)
            || std::abs(sideNW - sideNE) > 0.0005f
            || std::abs(sideNW - sideSW) > 0.0005f
            || std::abs(sideSE - sideNE) > 0.0005f
            || std::abs(sideSE - sideSW) > 0.0005f;
        emitFluidFace(verts, idxs, pos, normal,
                      flowingSurface ? flowSprite : stillSprite,
                      fluidMaterial, fluidTint, entry);
    }

    // BOTTOM FACE: render if vanilla would expose the lower fluid side.
    if (renderBottom) {
        glm::vec3 pos[4] = {
            {bx,     by + 0.001f, bz + 1},
            {bx + 1, by + 0.001f, bz + 1},
            {bx + 1, by + 0.001f, bz},
            {bx,     by + 0.001f, bz},
        };
        emitFluidFace(verts, idxs, pos, {0, -1, 0}, stillSprite, fluidMaterial, fluidTint, entry);
    }

    auto sideV = [](float height) {
        return std::clamp((1.0f - height) * 0.5f, 0.0f, 0.5f);
    };

    // SIDE FACES: follow the upstream vanilla mixin's side offsets and corner pairing.
    // NORTH (z-1)
    if (!isSameFluid(x, y, z - 1) && !isSideCoveredAt(x, y, z - 1, 2, std::max(sideNW, sideNE))) {
        float yStart = sideNW;
        float yEnd = sideNE;
        glm::vec3 pos[4] = {
            {bx,     by + yStart,        bz + 0.001f},
            {bx + 1, by + yEnd,          bz + 0.001f},
            {bx + 1, by + bottomYOffset, bz + 0.001f},
            {bx,     by + bottomYOffset, bz + 0.001f},
        };
        glm::vec2 uvs[4] = {
            {0.0f, sideV(yStart)},
            {0.5f, sideV(yEnd)},
            {0.5f, 0.5f},
            {0.0f, 0.5f},
        };
        emitFluidFace(verts, idxs, pos, {0, 0, -1}, flowSprite, fluidMaterial, fluidTint, entry, uvs);
    }

    // SOUTH (z+1)
    if (!isSameFluid(x, y, z + 1) && !isSideCoveredAt(x, y, z + 1, 3, std::max(sideSE, sideSW))) {
        float yStart = sideSE;
        float yEnd = sideSW;
        glm::vec3 pos[4] = {
            {bx + 1, by + yStart,        bz + 1 - 0.001f},
            {bx,     by + yEnd,          bz + 1 - 0.001f},
            {bx,     by + bottomYOffset, bz + 1 - 0.001f},
            {bx + 1, by + bottomYOffset, bz + 1 - 0.001f},
        };
        glm::vec2 uvs[4] = {
            {0.0f, sideV(yStart)},
            {0.5f, sideV(yEnd)},
            {0.5f, 0.5f},
            {0.0f, 0.5f},
        };
        emitFluidFace(verts, idxs, pos, {0, 0, 1}, flowSprite, fluidMaterial, fluidTint, entry, uvs);
    }

    // WEST (x-1)
    if (!isSameFluid(x - 1, y, z) && !isSideCoveredAt(x - 1, y, z, 4, std::max(sideSW, sideNW))) {
        float yStart = sideSW;
        float yEnd = sideNW;
        glm::vec3 pos[4] = {
            {bx + 0.001f, by + yStart,        bz + 1},
            {bx + 0.001f, by + yEnd,          bz},
            {bx + 0.001f, by + bottomYOffset, bz},
            {bx + 0.001f, by + bottomYOffset, bz + 1},
        };
        glm::vec2 uvs[4] = {
            {0.0f, sideV(yStart)},
            {0.5f, sideV(yEnd)},
            {0.5f, 0.5f},
            {0.0f, 0.5f},
        };
        emitFluidFace(verts, idxs, pos, {-1, 0, 0}, flowSprite, fluidMaterial, fluidTint, entry, uvs);
    }

    // EAST (x+1)
    if (!isSameFluid(x + 1, y, z) && !isSideCoveredAt(x + 1, y, z, 5, std::max(sideNE, sideSE))) {
        float yStart = sideNE;
        float yEnd = sideSE;
        glm::vec3 pos[4] = {
            {bx + 1 - 0.001f, by + yStart,        bz},
            {bx + 1 - 0.001f, by + yEnd,          bz + 1},
            {bx + 1 - 0.001f, by + bottomYOffset, bz + 1},
            {bx + 1 - 0.001f, by + bottomYOffset, bz},
        };
        glm::vec2 uvs[4] = {
            {0.0f, sideV(yStart)},
            {0.5f, sideV(yEnd)},
            {0.5f, 0.5f},
            {0.0f, 0.5f},
        };
        emitFluidFace(verts, idxs, pos, {1, 0, 0}, flowSprite, fluidMaterial, fluidTint, entry, uvs);
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

                // Collect light source if emissive (before fluid continue — lava needs area lights)
                if (entry->emissiveOrdinal != 255 && entry->emissionNits > 0) {
                    ChunkLightEntry light;
                    light.worldX = static_cast<float>(input.originX + x) + 0.5f;
                    light.worldY = static_cast<float>(input.originY + y) + 0.5f;
                    light.worldZ = static_cast<float>(input.originZ + z) + 0.5f;
                    light.lightTypeId = entry->emissiveOrdinal; // TODO: map to LightSourceRegistry typeId
                    output.lights.push_back(light);
                }

                // Fluid blocks: generate dynamic fluid quads
                if (entry->fluidType != 0) {
                    emitFluidQuads(input, table, *entry, x, y, z, output);
                    // Pure fluids (renderType==2): done, no model quads.
                    // Waterlogged (renderType==1): fall through to also emit model quads.
                    if (entry->renderType == 2) continue;
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
                    uint8_t tintType = entry->tintColorType & 0x7Fu;
                    bool isGrassSide = (tintType == 0 && dir >= 2 && dir <= 5 && quadCount >= 2);

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
