#include "core/render/greedy_mesher.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>

#include <glm/glm.hpp>

namespace {

// ---------------------------------------------------------------------------
// Material key — two quads can merge only if ALL fields match.
// colorPacked is intentionally EXCLUDED: we use per-vertex color interpolation
// instead, which preserves biome blending gradients across merged quads.
// ---------------------------------------------------------------------------
struct MaterialKey {
    uint32_t textureID;
    uint32_t emissiveBlockType;
    uint32_t flags;
    int32_t  albedoEmissionBits; // bit_cast of float
    uint8_t  uvOrientation;      // UV rotation relative to face (prevents merging rotated textures)

    bool operator==(const MaterialKey&) const = default;
};

// ---------------------------------------------------------------------------
// Face direction enumeration
// ---------------------------------------------------------------------------
enum FaceDir { POS_X = 0, NEG_X, POS_Y, NEG_Y, POS_Z, NEG_Z, FACE_COUNT };

static constexpr float AXIS_THRESHOLD = 0.9f;

static int classifyFaceDirection(const glm::vec3& normal) {
    float ax = std::abs(normal.x), ay = std::abs(normal.y), az = std::abs(normal.z);
    float maxComp = std::max({ax, ay, az});
    float len = glm::length(normal);
    if (len < 1e-6f || maxComp / len < AXIS_THRESHOLD) return -1;

    if (ax >= ay && ax >= az) return normal.x > 0 ? POS_X : NEG_X;
    if (ay >= ax && ay >= az) return normal.y > 0 ? POS_Y : NEG_Y;
    return normal.z > 0 ? POS_Z : NEG_Z;
}

// ---------------------------------------------------------------------------
// Grid axis mapping per face direction
// ---------------------------------------------------------------------------
static void getGridAxes(int faceDir, int& uAxis, int& vAxis, int& normalAxis) {
    switch (faceDir) {
        case POS_X: case NEG_X: uAxis = 1; vAxis = 2; normalAxis = 0; break;
        case POS_Y: case NEG_Y: uAxis = 0; vAxis = 2; normalAxis = 1; break;
        case POS_Z: case NEG_Z: uAxis = 0; vAxis = 1; normalAxis = 2; break;
        default: uAxis = 0; vAxis = 1; normalAxis = 2; break;
    }
}

static int safeLocalCoord(float worldCoord) {
    int i = static_cast<int>(std::floor(worldCoord));
    return ((i % 16) + 16) % 16;
}

// Mask for material-relevant flag bits: USE_TEXTURE, USE_COLOR_LAYER, COORD bits.
// Excludes per-instance flags (USE_NORM, USE_OVERLAY, USE_GLINT, USE_LIGHT, GREEDY_MERGED)
// that don't affect material identity.
static constexpr uint32_t MATERIAL_FLAGS_MASK =
    vk::VertexFormat::PBR_FLAG_USE_TEXTURE |
    vk::VertexFormat::PBR_FLAG_USE_COLOR_LAYER |
    vk::VertexFormat::PBR_FLAG_COORD_MASK;

// Compute UV orientation: encodes how texture UV axes map to face tangent axes.
// Blocks with different UV rotations (Minecraft randomizes for visual variety) get
// different orientations and won't merge, preventing texture shift on rebuild.
static uint8_t computeUVOrientation(const glm::vec2* uvs, const glm::vec3* positions,
                                     int minIdx, int uAxis, int vAxis) {
    int next = (minIdx + 1) % 4;
    int prev = (minIdx + 3) % 4;

    // Determine which quad edge aligns with grid U
    glm::vec3 edgeNext = positions[next] - positions[minIdx];
    glm::vec3 edgePrev = positions[prev] - positions[minIdx];
    int uNextIdx = std::abs(edgeNext[uAxis]) > std::abs(edgePrev[uAxis]) ? next : prev;
    int vNextIdx = (uNextIdx == next) ? prev : next;

    glm::vec2 uvDeltaU = uvs[uNextIdx] - uvs[minIdx];
    glm::vec2 uvDeltaV = uvs[vNextIdx] - uvs[minIdx];

    // 3 bits: bit 2 = rotated (grid U maps to texture V), bit 1 = V sign, bit 0 = U sign
    uint8_t orient = 0;
    if (std::abs(uvDeltaU.x) > std::abs(uvDeltaU.y)) {
        // Grid U → texture U (normal)
        if (uvDeltaU.x < 0) orient |= 1;
        if (uvDeltaV.y < 0) orient |= 2;
    } else {
        // Grid U → texture V (90° rotated)
        orient |= 4;
        if (uvDeltaU.y < 0) orient |= 1;
        if (uvDeltaV.x < 0) orient |= 2;
    }
    return orient;
}

static MaterialKey makeMaterialKey(const vk::VertexFormat::PBRTriangle& v, uint8_t uvOrient) {
    MaterialKey k;
    k.textureID = v.textureID;
    k.emissiveBlockType = v.emissiveBlockType;
    k.flags = v.flags & MATERIAL_FLAGS_MASK;
    int32_t bits;
    std::memcpy(&bits, &v.albedoEmission, sizeof(int32_t));
    k.albedoEmissionBits = bits;
    k.uvOrientation = uvOrient;
    return k;
}

// ---------------------------------------------------------------------------
// Parsed quad: 4 vertex indices + grid placement
// ---------------------------------------------------------------------------
struct ParsedQuad {
    uint32_t vertIdx[4]; // indices into original vertex array (quad order: v0,v1,v2,v3)
    MaterialKey key;
    int faceDir;
    int layer, localU, localV;
};

// ---------------------------------------------------------------------------
// Find the vertex of a quad closest to a specific corner in grid space.
// maxU/maxV select which corner: (false,false)=min, (true,false)=+U, etc.
// ---------------------------------------------------------------------------
static glm::vec4 getCornerVertexColor(
    const std::vector<vk::VertexFormat::PBRTriangle>& verts,
    const ParsedQuad& quad,
    int uAxis, int vAxis,
    bool maxU, bool maxV)
{
    float bestVal = -1e30f;
    int bestIdx = 0;
    for (int i = 0; i < 4; i++) {
        const auto& pos = verts[quad.vertIdx[i]].pos;
        float score = pos[uAxis] * (maxU ? 1.0f : -1.0f)
                    + pos[vAxis] * (maxV ? 1.0f : -1.0f);
        if (score > bestVal) { bestVal = score; bestIdx = i; }
    }
    return verts[quad.vertIdx[bestIdx]].colorLayer;
}

// Face normals for winding verification
static constexpr glm::vec3 FACE_NORMALS[6] = {
    {1,0,0}, {-1,0,0}, {0,1,0}, {0,-1,0}, {0,0,1}, {0,0,-1}
};

// Max merge size for biome-tinted blocks (USE_COLOR_LAYER).
// Limits bilinear approximation stretch to within biome blending radius.
static constexpr int MAX_TINTED_MERGE = 8;

} // anonymous namespace

// ===========================================================================
// Main merge entry point
// ===========================================================================
GreedyMesher::Result GreedyMesher::merge(
    const std::vector<vk::VertexFormat::PBRTriangle>& inVerts,
    const std::vector<uint32_t>& inIndices)
{
    Result result;
    result.origTriCount = static_cast<uint32_t>(inIndices.size()) / 3;

    uint32_t numQuads = static_cast<uint32_t>(inIndices.size()) / 6;
    if (numQuads == 0) {
        result.vertices = inVerts;
        result.indices = inIndices;
        result.mergedTriCount = result.origTriCount;
        return result;
    }

    // ----- Step 1: Parse quads -----
    std::vector<ParsedQuad> allQuads(numQuads);
    std::vector<bool> inGrid(numQuads, false);
    std::vector<bool> parsed(numQuads, false); // true if vertIdx was set

    // Per-face per-layer quad index lists
    std::vector<std::vector<uint32_t>> faceLayerQuadIdx[FACE_COUNT];
    for (int d = 0; d < FACE_COUNT; d++) faceLayerQuadIdx[d].resize(16);

    for (uint32_t q = 0; q < numQuads; q++) {
        uint32_t base = q * 6;
        uint32_t a0 = inIndices[base], a1 = inIndices[base+1], a2 = inIndices[base+2];
        uint32_t b0 = inIndices[base+3], b1 = inIndices[base+4], b2 = inIndices[base+5];

        auto& pq = allQuads[q];
        pq.faceDir = -1; // default: passthrough

        // Detect triangulation pattern and extract 4 unique vertices
        if (a2 == b0 && a0 == b2) {
            // Pattern: (i0,i1,i2, i2,i3,i0)
            pq.vertIdx[0] = a0; pq.vertIdx[1] = a1; pq.vertIdx[2] = a2; pq.vertIdx[3] = b1;
        } else if (a0 == b0 && a2 == b1) {
            // Pattern: (i0,i1,i2, i0,i2,i3)
            pq.vertIdx[0] = a0; pq.vertIdx[1] = a1; pq.vertIdx[2] = a2; pq.vertIdx[3] = b2;
        } else {
            continue; // unknown triangulation — passthrough (parsed stays false)
        }
        parsed[q] = true;

        // Classify face direction
        glm::vec3 edge1 = inVerts[pq.vertIdx[1]].pos - inVerts[pq.vertIdx[0]].pos;
        glm::vec3 edge2 = inVerts[pq.vertIdx[2]].pos - inVerts[pq.vertIdx[0]].pos;
        glm::vec3 normal = glm::cross(edge1, edge2);

        int faceDir = classifyFaceDirection(normal);
        if (faceDir < 0) continue; // non-axis-aligned — passthrough

        // Only merge full 1x1 block faces. Stairs/slabs/fences have partial faces
        // that would collide in the grid. Check that both quad edges are ~1.0 length.
        glm::vec3 edge03 = inVerts[pq.vertIdx[3]].pos - inVerts[pq.vertIdx[0]].pos;
        float len1 = glm::length(edge1);
        float len3 = glm::length(edge03);
        if (std::abs(len1 - 1.0f) > 0.05f || std::abs(len3 - 1.0f) > 0.05f)
            continue; // partial face — passthrough

        // Block position from face center
        glm::vec3 faceCenter = (inVerts[pq.vertIdx[0]].pos + inVerts[pq.vertIdx[1]].pos +
                                inVerts[pq.vertIdx[2]].pos + inVerts[pq.vertIdx[3]].pos) * 0.25f;
        glm::vec3 normDir = glm::normalize(normal);
        glm::vec3 blockPosF = faceCenter - normDir * 0.5f;

        int uAxis, vAxis, normalAxis;
        getGridAxes(faceDir, uAxis, vAxis, normalAxis);

        // Compute UV orientation (detects Minecraft's per-block UV rotation)
        glm::vec3 positions[4] = {inVerts[pq.vertIdx[0]].pos, inVerts[pq.vertIdx[1]].pos,
                                   inVerts[pq.vertIdx[2]].pos, inVerts[pq.vertIdx[3]].pos};
        glm::vec2 uvs[4] = {inVerts[pq.vertIdx[0]].textureUV, inVerts[pq.vertIdx[1]].textureUV,
                              inVerts[pq.vertIdx[2]].textureUV, inVerts[pq.vertIdx[3]].textureUV};
        // Find min-corner vertex for orientation computation
        int minIdx = 0;
        float minVal = positions[0][uAxis] + positions[0][vAxis];
        for (int i = 1; i < 4; i++) {
            float val = positions[i][uAxis] + positions[i][vAxis];
            if (val < minVal) { minVal = val; minIdx = i; }
        }
        uint8_t uvOrient = computeUVOrientation(uvs, positions, minIdx, uAxis, vAxis);

        pq.key = makeMaterialKey(inVerts[pq.vertIdx[0]], uvOrient);
        pq.faceDir = faceDir;
        pq.localU = safeLocalCoord(blockPosF[uAxis]);
        pq.localV = safeLocalCoord(blockPosF[vAxis]);
        pq.layer = safeLocalCoord(blockPosF[normalAxis]);

        faceLayerQuadIdx[faceDir][pq.layer].push_back(q);
    }

    // ----- Step 2: Build grids and place quads -----
    // Grid: [u][v] -> quad index (-1 = empty)
    int grid[16][16];

    result.vertices.reserve(inVerts.size());
    result.indices.reserve(inIndices.size());

    for (int d = 0; d < FACE_COUNT; d++) {
        int uAxis, vAxis, normalAxis;
        getGridAxes(d, uAxis, vAxis, normalAxis);

        for (int layer = 0; layer < 16; layer++) {
            auto& quadIndices = faceLayerQuadIdx[d][layer];
            if (quadIndices.empty()) continue;

            // Clear grid
            std::memset(grid, -1, sizeof(grid));

            // Place quads (first-come wins; duplicates become passthrough)
            for (uint32_t qi : quadIndices) {
                auto& pq = allQuads[qi];
                int u = pq.localU, v = pq.localV;
                if (grid[u][v] >= 0) continue; // duplicate position
                grid[u][v] = static_cast<int>(qi);
                inGrid[qi] = true;
            }

            // ----- Step 3: Greedy rectangle extraction -----
            bool visited[16][16] = {};

            for (int v = 0; v < 16; v++) {
                for (int u = 0; u < 16; u++) {
                    if (grid[u][v] < 0 || visited[u][v]) continue;

                    const auto& anchorQuad = allQuads[grid[u][v]];
                    const MaterialKey& key = anchorQuad.key;
                    bool isTinted = (key.flags & vk::VertexFormat::PBR_FLAG_USE_COLOR_LAYER) != 0;
                    int maxMerge = isTinted ? MAX_TINTED_MERGE : 16;

                    // Expand width
                    int w = 1;
                    while (u + w < 16 && w < maxMerge &&
                           grid[u + w][v] >= 0 && !visited[u + w][v] &&
                           allQuads[grid[u + w][v]].key == key) {
                        w++;
                    }

                    // Expand height
                    int h = 1;
                    while (v + h < 16 && h < maxMerge) {
                        bool rowOK = true;
                        for (int du = 0; du < w; du++) {
                            if (grid[u + du][v + h] < 0 || visited[u + du][v + h] ||
                                !(allQuads[grid[u + du][v + h]].key == key)) {
                                rowOK = false;
                                break;
                            }
                        }
                        if (!rowOK) break;
                        h++;
                    }

                    // Mark visited
                    for (int dv = 0; dv < h; dv++)
                        for (int du = 0; du < w; du++)
                            visited[u + du][v + dv] = true;

                    // ----- Step 4: Generate merged quad vertices -----
                    const auto& av0 = inVerts[anchorQuad.vertIdx[0]];
                    const auto& av1 = inVerts[anchorQuad.vertIdx[1]];
                    const auto& av2 = inVerts[anchorQuad.vertIdx[2]];
                    const auto& av3 = inVerts[anchorQuad.vertIdx[3]];

                    // 1x1: emit original vertices unchanged
                    if (w == 1 && h == 1) {
                        uint32_t bv = static_cast<uint32_t>(result.vertices.size());
                        result.vertices.push_back(av0);
                        result.vertices.push_back(av1);
                        result.vertices.push_back(av2);
                        result.vertices.push_back(av3);
                        result.indices.insert(result.indices.end(),
                            {bv, bv+1, bv+2, bv+2, bv+3, bv});
                        continue;
                    }

                    // Find min-corner vertex (smallest u + v in grid space)
                    glm::vec3 positions[4] = {av0.pos, av1.pos, av2.pos, av3.pos};
                    glm::vec2 uvs[4] = {av0.textureUV, av1.textureUV, av2.textureUV, av3.textureUV};

                    int minIdx = 0;
                    float minVal = positions[0][uAxis] + positions[0][vAxis];
                    for (int i = 1; i < 4; i++) {
                        float val = positions[i][uAxis] + positions[i][vAxis];
                        if (val < minVal) { minVal = val; minIdx = i; }
                    }

                    // Edges from min-corner to adjacent vertices in quad order
                    int next = (minIdx + 1) % 4;
                    int prev = (minIdx + 3) % 4;
                    glm::vec3 edgeNext = positions[next] - positions[minIdx];
                    glm::vec3 edgePrev = positions[prev] - positions[minIdx];

                    // Which edge aligns with grid U axis?
                    int uNextIdx, vNextIdx;
                    if (std::abs(edgeNext[uAxis]) > std::abs(edgePrev[uAxis])) {
                        uNextIdx = next; vNextIdx = prev;
                    } else {
                        uNextIdx = prev; vNextIdx = next;
                    }

                    // Exact integer axis stepping (avoids FP accumulation from edge multiplication).
                    // uStep/vStep are +1 or -1 along the world axis, computed from edge sign.
                    glm::vec3 uStep(0.0f), vStep(0.0f);
                    uStep[uAxis] = (positions[uNextIdx][uAxis] > positions[minIdx][uAxis]) ? 1.0f : -1.0f;
                    vStep[vAxis] = (positions[vNextIdx][vAxis] > positions[minIdx][vAxis]) ? 1.0f : -1.0f;

                    // Snap min-corner position to exact integer coordinates on face tangent axes.
                    // The normal-axis coordinate stays as-is (face offset like y=64.0 for +Y face).
                    glm::vec3 snappedMin = positions[minIdx];
                    snappedMin[uAxis] = std::round(snappedMin[uAxis]);
                    snappedMin[vAxis] = std::round(snappedMin[vAxis]);

                    // UV deltas from min-corner vertex
                    glm::vec2 minUV = uvs[minIdx];
                    glm::vec2 uvDeltaU = uvs[uNextIdx] - minUV;
                    glm::vec2 uvDeltaV = uvs[vNextIdx] - minUV;

                    // Corner colors from the quads at rectangle corners (Option B)
                    glm::vec4 colorMinMin = getCornerVertexColor(inVerts, anchorQuad, uAxis, vAxis, false, false);
                    glm::vec4 colorMaxMin = colorMinMin, colorMinMax = colorMinMin, colorMaxMax = colorMinMin;

                    // Quad at (u+w-1, v) — +U corner
                    if (grid[u + w - 1][v] >= 0) {
                        colorMaxMin = getCornerVertexColor(inVerts, allQuads[grid[u + w - 1][v]],
                                                           uAxis, vAxis, true, false);
                    }
                    // Quad at (u, v+h-1) — +V corner
                    if (grid[u][v + h - 1] >= 0) {
                        colorMinMax = getCornerVertexColor(inVerts, allQuads[grid[u][v + h - 1]],
                                                           uAxis, vAxis, false, true);
                    }
                    // Quad at (u+w-1, v+h-1) — diagonal corner
                    if (grid[u + w - 1][v + h - 1] >= 0) {
                        colorMaxMax = getCornerVertexColor(inVerts, allQuads[grid[u + w - 1][v + h - 1]],
                                                           uAxis, vAxis, true, true);
                    }

                    // Face normal for merged vertices
                    glm::vec3 faceNormal = FACE_NORMALS[d];

                    // Sprite bounds from vertex UVs (one tile)
                    glm::vec2 spriteMin = glm::min(glm::min(uvs[0], uvs[1]), glm::min(uvs[2], uvs[3]));
                    glm::vec2 spriteMax = glm::max(glm::max(uvs[0], uvs[1]), glm::max(uvs[2], uvs[3]));
                    glm::vec2 spriteSize = spriteMax - spriteMin;

                    // Build 4 corner vertices using exact integer axis stepping
                    auto makeVertex = [&](int du, int dv, const glm::vec4& color) {
                        vk::VertexFormat::PBRTriangle vert = av0; // template
                        vert.pos = snappedMin
                                 + uStep * static_cast<float>(du)
                                 + vStep * static_cast<float>(dv);
                        vert.norm = faceNormal;
                        vert.flags |= vk::VertexFormat::PBR_FLAG_GREEDY_MERGED;
                        vert.flags &= ~(vk::VertexFormat::PBR_FLAG_USE_GLINT | vk::VertexFormat::PBR_FLAG_USE_OVERLAY);
                        vert.postBase = vert.pos; // static chunk — zero motion vector
                        vert.textureUV = minUV
                                       + uvDeltaU * static_cast<float>(du)
                                       + uvDeltaV * static_cast<float>(dv);
                        vert.colorLayer = color;
                        // Repurpose unused fields to carry sprite bounds per-vertex:
                        // glintUV → spriteMin, overlayPacked → spriteSizeU, lightPacked → spriteSizeV
                        vert.glintUV = spriteMin;
                        std::memcpy(&vert.overlayPacked, &spriteSize.x, sizeof(float));
                        std::memcpy(&vert.lightPacked, &spriteSize.y, sizeof(float));
                        return vert;
                    };

                    uint32_t bv = static_cast<uint32_t>(result.vertices.size());
                    result.vertices.push_back(makeVertex(0, 0, colorMinMin));
                    result.vertices.push_back(makeVertex(w, 0, colorMaxMin));
                    result.vertices.push_back(makeVertex(w, h, colorMaxMax));
                    result.vertices.push_back(makeVertex(0, h, colorMinMax));

                    // Winding from exact axis steps (no FP error to check)
                    glm::vec3 mergedNormal = glm::cross(uStep, vStep);
                    if (glm::dot(mergedNormal, faceNormal) > 0) {
                        result.indices.insert(result.indices.end(),
                            {bv, bv+1, bv+2, bv+2, bv+3, bv});
                    } else {
                        // Flip winding
                        result.indices.insert(result.indices.end(),
                            {bv, bv+3, bv+2, bv+2, bv+1, bv});
                    }
                }
            }
        }
    }

    // ----- Step 5: Passthrough non-merged quads -----
    for (uint32_t q = 0; q < numQuads; q++) {
        if (inGrid[q]) continue;

        uint32_t base = q * 6;
        if (!parsed[q]) {
            // Unknown triangulation — emit original 6 indices with original vertices
            for (int j = 0; j < 6; j++) {
                uint32_t origIdx = inIndices[base + j];
                uint32_t newIdx = static_cast<uint32_t>(result.vertices.size());
                result.vertices.push_back(inVerts[origIdx]);
                result.indices.push_back(newIdx);
            }
        } else {
            // Parsed quad (non-axis-aligned or partial face) — emit with correct vertices
            auto& pq = allQuads[q];
            uint32_t bv = static_cast<uint32_t>(result.vertices.size());
            result.vertices.push_back(inVerts[pq.vertIdx[0]]);
            result.vertices.push_back(inVerts[pq.vertIdx[1]]);
            result.vertices.push_back(inVerts[pq.vertIdx[2]]);
            result.vertices.push_back(inVerts[pq.vertIdx[3]]);
            result.indices.insert(result.indices.end(),
                {bv, bv+1, bv+2, bv+2, bv+3, bv});
        }
    }

    // Passthrough any trailing indices that don't form a complete quad
    uint32_t trailingStart = numQuads * 6;
    if (trailingStart < inIndices.size()) {
        // Build a vertex remap for trailing triangles
        for (uint32_t i = trailingStart; i < inIndices.size(); i++) {
            uint32_t origIdx = inIndices[i];
            uint32_t newIdx = static_cast<uint32_t>(result.vertices.size());
            result.vertices.push_back(inVerts[origIdx]);
            result.indices.push_back(newIdx);
        }
    }

    result.mergedTriCount = static_cast<uint32_t>(result.indices.size()) / 3;

    // Log significant merges
    if (result.origTriCount > 100 && result.mergedTriCount < result.origTriCount) {
        int pct = 100 - static_cast<int>(result.mergedTriCount * 100u / result.origTriCount);
        std::cout << "[GreedyMesher] " << result.origTriCount << " -> " << result.mergedTriCount
                  << " tris (" << pct << "% reduction)" << std::endl;
    }

    return result;
}
