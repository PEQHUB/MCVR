#include "tessellator.hpp"

#include <algorithm>
#include <cmath>
#include <glm/glm.hpp>

// Face normal vectors indexed by faceAxis (0=+X, 1=-X, 2=+Y, 3=-Y, 4=+Z, 5=-Z)
static const glm::vec3 FACE_NORMALS[6] = {
    {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};

// CPU smoothstep matching GLSL smoothstep(edge0, edge1, x)
static inline float smoothstep(float edge0, float edge1, float x) {
    float t = std::clamp((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

// Bilinear interpolation of a float
static inline float bilerp(float v00, float v10, float v01, float v11, float u, float v) {
    return (1 - u) * (1 - v) * v00 + u * (1 - v) * v10 + (1 - u) * v * v01 + u * v * v11;
}

static inline glm::vec2 bilerp(glm::vec2 a, glm::vec2 b, glm::vec2 c, glm::vec2 d, float u, float v) {
    return (1 - u) * (1 - v) * a + u * (1 - v) * b + (1 - u) * v * c + u * v * d;
}

static inline glm::vec3 bilerp(glm::vec3 a, glm::vec3 b, glm::vec3 c, glm::vec3 d, float u, float v) {
    return (1 - u) * (1 - v) * a + u * (1 - v) * b + (1 - u) * v * c + u * v * d;
}

static inline glm::vec4 bilerp(glm::vec4 a, glm::vec4 b, glm::vec4 c, glm::vec4 d, float u, float v) {
    return (1 - u) * (1 - v) * a + u * (1 - v) * b + (1 - u) * v * c + u * v * d;
}

static inline glm::ivec2 bilerpIvec2(glm::ivec2 a, glm::ivec2 b, glm::ivec2 c, glm::ivec2 d, float u, float v) {
    glm::vec2 result = (1 - u) * (1 - v) * glm::vec2(a) + u * (1 - v) * glm::vec2(b) +
                        (1 - u) * v * glm::vec2(c) + u * v * glm::vec2(d);
    return glm::ivec2(glm::round(result));
}

static inline glm::ivec2 unpackIvec2(uint32_t packed) {
    return glm::ivec2(int16_t(packed & 0xFFFF), int16_t((packed >> 16) & 0xFFFF));
}

static inline uint32_t packIvec2(glm::ivec2 v) {
    return (uint32_t(v.x) & 0xFFFF) | ((uint32_t(v.y) & 0xFFFF) << 16);
}

static inline uint32_t bilerpPacked(uint32_t a, uint32_t b, uint32_t c, uint32_t d, float u, float v) {
    return packIvec2(bilerpIvec2(unpackIvec2(a), unpackIvec2(b), unpackIvec2(c), unpackIvec2(d), u, v));
}

// Map quad vertices v0,v1,v2,v3 to a consistent bilinear parameterization.
// Quad index pattern: [v0,v1,v2, v2,v3,v0] — two triangles.
// We need to find corner mapping: (0,0), (1,0), (1,1), (0,1) in UV space.
// Since faces are axis-aligned, we find the corner with min textureUV and map from there.
using QuadCorners = Tessellator::QuadCorners;

static QuadCorners mapCorners(const Tessellator::Input &input) {
    // Find the vertex with min UV (bottom-left in atlas) as c00
    const vk::VertexFormat::PBRTriangle *verts[4] = {input.v0, input.v1, input.v2, input.v3};
    int minIdx = 0;
    for (int i = 1; i < 4; i++) {
        if (verts[i]->textureUV.x < verts[minIdx]->textureUV.x - 0.0001f ||
            (std::abs(verts[i]->textureUV.x - verts[minIdx]->textureUV.x) < 0.0001f &&
             verts[i]->textureUV.y < verts[minIdx]->textureUV.y)) {
            minIdx = i;
        }
    }

    const auto *corner = verts[minIdx];
    glm::vec2 cornerUV = corner->textureUV;

    // Find edges from corner: one along U (dUV.x > 0), one along V (dUV.y > 0)
    const vk::VertexFormat::PBRTriangle *uNeighbor = nullptr;
    const vk::VertexFormat::PBRTriangle *vNeighbor = nullptr;
    const vk::VertexFormat::PBRTriangle *diagonal = nullptr;

    for (int i = 0; i < 4; i++) {
        if (i == minIdx) continue;
        glm::vec2 dUV = verts[i]->textureUV - cornerUV;
        if (!uNeighbor && std::abs(dUV.x) > 0.0001f && std::abs(dUV.y) < 0.0001f) {
            uNeighbor = verts[i];
        } else if (!vNeighbor && std::abs(dUV.y) > 0.0001f && std::abs(dUV.x) < 0.0001f) {
            vNeighbor = verts[i];
        } else {
            diagonal = verts[i];
        }
    }

    // Fallback: if UV axes aren't perfectly aligned, assign remaining verts
    if (!uNeighbor || !vNeighbor) {
        for (int i = 0; i < 4; i++) {
            if (verts[i] == corner || verts[i] == uNeighbor || verts[i] == vNeighbor) continue;
            if (!uNeighbor)
                uNeighbor = verts[i];
            else if (!vNeighbor)
                vNeighbor = verts[i];
            else
                diagonal = verts[i];
        }
    }
    if (!diagonal) {
        // Find the remaining vertex
        for (int i = 0; i < 4; i++) {
            if (verts[i] != corner && verts[i] != uNeighbor && verts[i] != vNeighbor) {
                diagonal = verts[i];
                break;
            }
        }
    }

    return {corner, uNeighbor, vNeighbor, diagonal};
}

// Helper: interpolate all PBRTriangle fields at parametric (u,v) from quad corners
static void interpVertex(vk::VertexFormat::PBRTriangle &vert, const QuadCorners &q, float u, float v) {
    vert.pos = bilerp(q.c00->pos, q.c10->pos, q.c01->pos, q.c11->pos, u, v);
    vert.flags = q.c00->flags; // all booleans + coordinate: corner-copied
    vert.norm = q.c00->norm;
    vert.albedoEmission = q.c00->albedoEmission;
    vert.colorLayer = bilerp(q.c00->colorLayer, q.c10->colorLayer, q.c01->colorLayer, q.c11->colorLayer, u, v);
    vert.postBase = bilerp(q.c00->postBase, q.c10->postBase, q.c01->postBase, q.c11->postBase, u, v);
    vert.emissiveBlockType = q.c00->emissiveBlockType;
    vert.textureUV = bilerp(q.c00->textureUV, q.c10->textureUV, q.c01->textureUV, q.c11->textureUV, u, v);
    vert.glintUV = bilerp(q.c00->glintUV, q.c10->glintUV, q.c01->glintUV, q.c11->glintUV, u, v);
    vert.textureID = q.c00->textureID;
    vert.glintTexture = q.c00->glintTexture;
    vert.overlayPacked = bilerpPacked(q.c00->overlayPacked, q.c10->overlayPacked, q.c01->overlayPacked, q.c11->overlayPacked, u, v);
    vert.lightPacked = bilerpPacked(q.c00->lightPacked, q.c10->lightPacked, q.c01->lightPacked, q.c11->lightPacked, u, v);
}

Tessellator::Output Tessellator::tessellate(const Input &input) {
    Output out;
    uint32_t N = std::clamp(input.tessLevel, 2u, 32u);

    QuadCorners q = mapCorners(input);
    if (!q.c00 || !q.c10 || !q.c01 || !q.c11) {
        return out;
    }

    glm::vec3 faceNormal = (input.faceAxis < 6) ? FACE_NORMALS[input.faceAxis] : glm::vec3(0, 1, 0);
    int filterMode = input.material ? static_cast<int>(input.material->pomPacked0 & 0x7) : 1;
    bool useNearest = (filterMode == 0);

    // Winding: check if grid UV cross-product matches face normal
    glm::vec3 gridU = q.c10->pos - q.c00->pos;
    glm::vec3 gridV = q.c01->pos - q.c00->pos;
    bool flipWinding = glm::dot(glm::cross(gridU, gridV), faceNormal) < 0.0f;

    if (useNearest) {
        // === DISCONNECTED QUADS + SIDE WALLS ===
        // Each texel = flat quad at its height. Vertical walls fill gaps between
        // adjacent texels at different heights. Full watertight geometry.
        float invN = 1.0f / static_cast<float>(N);

        // Pass 1: compute displacement for each texel
        std::vector<float> disp(N * N, 0.0f);
        for (uint32_t cy = 0; cy < N; cy++) {
            for (uint32_t cx = 0; cx < N; cx++) {
                float centerU = (cx + 0.5f) * invN;
                float centerV = (cy + 0.5f) * invN;
                glm::vec2 texUV = bilerp(q.c00->textureUV, q.c10->textureUV,
                                         q.c01->textureUV, q.c11->textureUV, centerU, centerV);
                float height = HeightSampler::sample(
                    input.labPbrHeight, input.autoPbrHeight,
                    input.normalRGBA, input.albedoRGBA, input.material,
                    input.hasLabPBRHeight, input.isAutoPBR,
                    texUV.x, texUV.y,
                    input.uvMinX, input.uvMinY, input.uvMaxX, input.uvMaxY);
                // Per-edge fade: smoothstep matching rint/rchit shaders.
                // Only fade edges marked in fadeEdgeMask (perpendicular edges).
                // Coplanar edges (adjacent tessellated block) keep full displacement.
                float fN = static_cast<float>(N);
                float edgeWidth = std::min(2.0f, std::max(1.0f, input.heightScale * fN)) / fN; // parametric [0,1], capped at 2 texels
                float edgeFade = 1.0f;
                if (input.fadeEdgeMask & 0x1) edgeFade *= smoothstep(0.0f, edgeWidth, centerU);
                if (input.fadeEdgeMask & 0x2) edgeFade *= smoothstep(0.0f, edgeWidth, 1.0f - centerU);
                if (input.fadeEdgeMask & 0x4) edgeFade *= smoothstep(0.0f, edgeWidth, centerV);
                if (input.fadeEdgeMask & 0x8) edgeFade *= smoothstep(0.0f, edgeWidth, 1.0f - centerV);
                disp[cy * N + cx] = (1.0f - height) * input.heightScale * edgeFade;
            }
        }

        // Export edge displacement values for seam stitching with neighbors
        out.edgeDispU0.resize(N); out.edgeDispU1.resize(N);
        out.edgeDispV0.resize(N); out.edgeDispV1.resize(N);
        for (uint32_t k = 0; k < N; k++) {
            out.edgeDispU0[k] = disp[k * N];           // column 0, rows 0..N-1
            out.edgeDispU1[k] = disp[k * N + N - 1];   // column N-1, rows 0..N-1
            out.edgeDispV0[k] = disp[k];               // row 0, columns 0..N-1
            out.edgeDispV1[k] = disp[(N - 1) * N + k]; // row N-1, columns 0..N-1
        }

        // Count side walls needed (each wall is double-sided = 2 quads)
        uint32_t wallCount = 0;
        // Internal adjacencies
        for (uint32_t cy = 0; cy < N; cy++)
            for (uint32_t cx = 0; cx < N - 1; cx++)
                if (std::abs(disp[cy * N + cx] - disp[cy * N + cx + 1]) > 1e-5f) wallCount += 2;
        for (uint32_t cy = 0; cy < N - 1; cy++)
            for (uint32_t cx = 0; cx < N; cx++)
                if (std::abs(disp[cy * N + cx] - disp[(cy + 1) * N + cx]) > 1e-5f) wallCount += 2;
        // Border walls only on faded edges (perpendicular boundaries)
        // Skip border walls on coplanar edges (fadeEdgeMask bit clear = no fade = no wall)
        if (input.fadeEdgeMask & 0x4) // V=0 edge (top row)
            for (uint32_t cx = 0; cx < N; cx++)
                if (disp[cx] > 1e-5f) wallCount += 2;
        if (input.fadeEdgeMask & 0x8) // V=1 edge (bottom row)
            for (uint32_t cx = 0; cx < N; cx++)
                if (disp[(N - 1) * N + cx] > 1e-5f) wallCount += 2;
        if (input.fadeEdgeMask & 0x1) // U=0 edge (left col)
            for (uint32_t cy = 0; cy < N; cy++)
                if (disp[cy * N] > 1e-5f) wallCount += 2;
        if (input.fadeEdgeMask & 0x2) // U=1 edge (right col)
            for (uint32_t cy = 0; cy < N; cy++)
                if (disp[cy * N + N - 1] > 1e-5f) wallCount += 2;

        uint32_t faceQuads = N * N;
        uint32_t totalQuads = faceQuads + wallCount;
        out.vertices.resize(totalQuads * 4);
        out.indices.reserve(totalQuads * 6);

        // Pass 2: generate face quads
        for (uint32_t cy = 0; cy < N; cy++) {
            for (uint32_t cx = 0; cx < N; cx++) {
                uint32_t base = (cy * N + cx) * 4;
                float u0 = cx * invN, u1 = (cx + 1) * invN;
                float v0 = cy * invN, v1 = (cy + 1) * invN;

                interpVertex(out.vertices[base + 0], q, u0, v0);
                interpVertex(out.vertices[base + 1], q, u1, v0);
                interpVertex(out.vertices[base + 2], q, u1, v1);
                interpVertex(out.vertices[base + 3], q, u0, v1);

                glm::vec3 offset = -faceNormal * disp[cy * N + cx];
                for (int vi = 0; vi < 4; vi++) {
                    out.vertices[base + vi].pos += offset;
                    out.vertices[base + vi].postBase += offset;
                }

                if (flipWinding) {
                    out.indices.push_back(base + 0);
                    out.indices.push_back(base + 2);
                    out.indices.push_back(base + 1);
                    out.indices.push_back(base + 0);
                    out.indices.push_back(base + 3);
                    out.indices.push_back(base + 2);
                } else {
                    out.indices.push_back(base + 0);
                    out.indices.push_back(base + 1);
                    out.indices.push_back(base + 2);
                    out.indices.push_back(base + 0);
                    out.indices.push_back(base + 2);
                    out.indices.push_back(base + 3);
                }
            }
        }

        // Pass 3: generate vertical side walls between adjacent texels
        uint32_t wallIdx = faceQuads;
        auto addWall = [&](float eu0, float ev0, float eu1, float ev1, float dA, float dB) {
            // Wall connects the shared edge at two different displacement depths.
            // Double-sided: emit both windings to avoid back-face culling.
            float dNear = std::min(dA, dB);
            float dFar  = std::max(dA, dB);

            for (int side = 0; side < 2; side++) {
                uint32_t base = wallIdx * 4;

                interpVertex(out.vertices[base + 0], q, eu0, ev0);
                interpVertex(out.vertices[base + 1], q, eu1, ev1);
                interpVertex(out.vertices[base + 2], q, eu1, ev1);
                interpVertex(out.vertices[base + 3], q, eu0, ev0);

                out.vertices[base + 0].pos += -faceNormal * dNear;
                out.vertices[base + 1].pos += -faceNormal * dNear;
                out.vertices[base + 2].pos += -faceNormal * dFar;
                out.vertices[base + 3].pos += -faceNormal * dFar;
                out.vertices[base + 0].postBase += -faceNormal * dNear;
                out.vertices[base + 1].postBase += -faceNormal * dNear;
                out.vertices[base + 2].postBase += -faceNormal * dFar;
                out.vertices[base + 3].postBase += -faceNormal * dFar;

                // Wall normal: perpendicular to face normal and edge direction
                glm::vec3 edgeDir = out.vertices[base + 1].pos - out.vertices[base + 0].pos;
                glm::vec3 wallNorm = glm::normalize(glm::cross(edgeDir, faceNormal));
                if (side == 1) wallNorm = -wallNorm;
                for (int vi = 0; vi < 4; vi++)
                    out.vertices[base + vi].norm = wallNorm;

                // Opposite winding per side
                if (side == 0) {
                    out.indices.push_back(base + 0);
                    out.indices.push_back(base + 1);
                    out.indices.push_back(base + 2);
                    out.indices.push_back(base + 0);
                    out.indices.push_back(base + 2);
                    out.indices.push_back(base + 3);
                } else {
                    out.indices.push_back(base + 0);
                    out.indices.push_back(base + 2);
                    out.indices.push_back(base + 1);
                    out.indices.push_back(base + 0);
                    out.indices.push_back(base + 3);
                    out.indices.push_back(base + 2);
                }
                wallIdx++;
            }
        };

        // Horizontal adjacencies (cx, cx+1 at same cy)
        for (uint32_t cy = 0; cy < N; cy++) {
            for (uint32_t cx = 0; cx < N - 1; cx++) {
                float dL = disp[cy * N + cx], dR = disp[cy * N + cx + 1];
                if (std::abs(dL - dR) > 1e-5f) {
                    float eu = (cx + 1) * invN;
                    addWall(eu, cy * invN, eu, (cy + 1) * invN, dL, dR);
                }
            }
        }
        // Vertical adjacencies (cy, cy+1 at same cx)
        for (uint32_t cy = 0; cy < N - 1; cy++) {
            for (uint32_t cx = 0; cx < N; cx++) {
                float dT = disp[cy * N + cx], dB = disp[(cy + 1) * N + cx];
                if (std::abs(dT - dB) > 1e-5f) {
                    float ev = (cy + 1) * invN;
                    addWall(cx * invN, ev, (cx + 1) * invN, ev, dT, dB);
                }
            }
        }
        // Border walls only on faded edges (perpendicular boundaries).
        // Skip on coplanar edges — adjacent block continues seamlessly.
        if (input.fadeEdgeMask & 0x4) // V=0 edge
            for (uint32_t cx = 0; cx < N; cx++) {
                float d = disp[cx];
                if (d > 1e-5f) addWall(cx * invN, 0, (cx + 1) * invN, 0, 0.0f, d);
            }
        if (input.fadeEdgeMask & 0x8) // V=1 edge
            for (uint32_t cx = 0; cx < N; cx++) {
                float d = disp[(N - 1) * N + cx];
                if (d > 1e-5f) addWall(cx * invN, 1.0f, (cx + 1) * invN, 1.0f, 0.0f, d);
            }
        if (input.fadeEdgeMask & 0x1) // U=0 edge
            for (uint32_t cy = 0; cy < N; cy++) {
                float d = disp[cy * N];
                if (d > 1e-5f) addWall(0, cy * invN, 0, (cy + 1) * invN, 0.0f, d);
            }
        if (input.fadeEdgeMask & 0x2) // U=1 edge
            for (uint32_t cy = 0; cy < N; cy++) {
                float d = disp[cy * N + N - 1];
                if (d > 1e-5f) addWall(1.0f, cy * invN, 1.0f, (cy + 1) * invN, 0.0f, d);
            }
    } else {
        // === CONNECTED GRID: bilinear interpolation for smooth displacement ===
        uint32_t gridW = N + 1;
        out.vertices.resize(gridW * gridW);
        out.indices.reserve(N * N * 6);

        for (uint32_t row = 0; row < gridW; row++) {
            for (uint32_t col = 0; col < gridW; col++) {
                float u = static_cast<float>(col) / static_cast<float>(N);
                float v = static_cast<float>(row) / static_cast<float>(N);
                auto &vert = out.vertices[row * gridW + col];

                interpVertex(vert, q, u, v);

                float height = HeightSampler::sample(
                    input.labPbrHeight, input.autoPbrHeight,
                    input.normalRGBA, input.albedoRGBA, input.material,
                    input.hasLabPBRHeight, input.isAutoPBR,
                    vert.textureUV.x, vert.textureUV.y,
                    input.uvMinX, input.uvMinY, input.uvMaxX, input.uvMaxY);

                float fN = static_cast<float>(N);
                float edgeWidth = std::min(2.0f, std::max(1.0f, input.heightScale * fN)) / fN; // parametric [0,1], capped at 2 texels
                float edgeFade = 1.0f;
                if (input.fadeEdgeMask & 0x1) edgeFade *= smoothstep(0.0f, edgeWidth, u);
                if (input.fadeEdgeMask & 0x2) edgeFade *= smoothstep(0.0f, edgeWidth, 1.0f - u);
                if (input.fadeEdgeMask & 0x4) edgeFade *= smoothstep(0.0f, edgeWidth, v);
                if (input.fadeEdgeMask & 0x8) edgeFade *= smoothstep(0.0f, edgeWidth, 1.0f - v);

                float displacement = (1.0f - height) * input.heightScale * edgeFade;
                glm::vec3 offset = -faceNormal * displacement;
                vert.pos += offset;
                vert.postBase += offset;
            }
        }

        for (uint32_t row = 0; row < N; row++) {
            for (uint32_t col = 0; col < N; col++) {
                uint32_t i00 = row * gridW + col;
                uint32_t i10 = row * gridW + col + 1;
                uint32_t i01 = (row + 1) * gridW + col;
                uint32_t i11 = (row + 1) * gridW + col + 1;
                if (flipWinding) {
                    out.indices.push_back(i00);
                    out.indices.push_back(i11);
                    out.indices.push_back(i10);
                    out.indices.push_back(i00);
                    out.indices.push_back(i01);
                    out.indices.push_back(i11);
                } else {
                    out.indices.push_back(i00);
                    out.indices.push_back(i10);
                    out.indices.push_back(i11);
                    out.indices.push_back(i00);
                    out.indices.push_back(i11);
                    out.indices.push_back(i01);
                }
            }
        }
    }

    return out;
}

Tessellator::QuadCorners Tessellator::mapCornersPublic(const Input &input) {
    return mapCorners(input);
}

void Tessellator::interpVertexPublic(vk::VertexFormat::PBRTriangle &vert, const QuadCorners &q, float u, float v) {
    interpVertex(vert, q, u, v);
}

uint32_t Tessellator::computeTessLevel(float distanceToCamera, uint32_t textureResolution,
                                       uint32_t maxTessLevel,
                                       float nearDist, float midDist, float farDist) {
    // Cap by texture resolution (no point exceeding texel density)
    uint32_t cap = std::min(textureResolution, maxTessLevel);
    cap = std::max(cap, 2u);

    if (distanceToCamera < nearDist) return cap;                    // Near: full
    if (distanceToCamera < midDist) return std::max(cap / 2, 2u);  // Mid: half
    if (distanceToCamera < farDist) return std::max(cap / 4, 2u);  // Far: quarter
    return 2;                                                        // Very far: minimal
}
