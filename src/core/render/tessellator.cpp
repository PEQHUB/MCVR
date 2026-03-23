#include "tessellator.hpp"

#include <algorithm>
#include <cmath>
#include <glm/glm.hpp>

// Face normal vectors indexed by faceAxis (0=+X, 1=-X, 2=+Y, 3=-Y, 4=+Z, 5=-Z)
static const glm::vec3 FACE_NORMALS[6] = {
    {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};

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

// Map quad vertices v0,v1,v2,v3 to a consistent bilinear parameterization.
// Quad index pattern: [v0,v1,v2, v2,v3,v0] — two triangles.
// We need to find corner mapping: (0,0), (1,0), (1,1), (0,1) in UV space.
// Since faces are axis-aligned, we find the corner with min textureUV and map from there.
struct QuadCorners {
    const vk::VertexFormat::PBRTriangle *c00; // parametric (0,0)
    const vk::VertexFormat::PBRTriangle *c10; // parametric (1,0)
    const vk::VertexFormat::PBRTriangle *c01; // parametric (0,1)
    const vk::VertexFormat::PBRTriangle *c11; // parametric (1,1)
};

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

Tessellator::Output Tessellator::tessellate(const Input &input) {
    Output out;
    uint32_t N = std::clamp(input.tessLevel, 2u, 32u);
    uint32_t gridW = N + 1;
    uint32_t vertCount = gridW * gridW;
    uint32_t triCount = N * N * 2;

    out.vertices.resize(vertCount);
    out.indices.reserve(triCount * 3);

    QuadCorners q = mapCorners(input);

    // Safety: skip tessellation if mapCorners couldn't find 4 distinct corners
    if (!q.c00 || !q.c10 || !q.c01 || !q.c11) {
        return out; // Return empty — caller keeps original quad
    }

    glm::vec3 faceNormal = (input.faceAxis < 6) ? FACE_NORMALS[input.faceAxis] : glm::vec3(0, 1, 0);

    // Generate vertex grid
    for (uint32_t row = 0; row < gridW; row++) {
        for (uint32_t col = 0; col < gridW; col++) {
            float u = static_cast<float>(col) / static_cast<float>(N);
            float v = static_cast<float>(row) / static_cast<float>(N);
            uint32_t idx = row * gridW + col;

            auto &vert = out.vertices[idx];

            // Bilinearly interpolate all PBRTriangle fields from 4 corners
            vert.pos = bilerp(q.c00->pos, q.c10->pos, q.c01->pos, q.c11->pos, u, v);
            vert.norm = q.c00->norm; // Face normal is uniform across the quad
            vert.useNorm = q.c00->useNorm;
            vert.useColorLayer = q.c00->useColorLayer;
            vert.colorLayer = bilerp(q.c00->colorLayer, q.c10->colorLayer,
                                     q.c01->colorLayer, q.c11->colorLayer, u, v);
            vert.useTexture = q.c00->useTexture;
            vert.useOverlay = q.c00->useOverlay;
            vert.textureUV = bilerp(q.c00->textureUV, q.c10->textureUV,
                                    q.c01->textureUV, q.c11->textureUV, u, v);
            vert.overlayUV = bilerpIvec2(q.c00->overlayUV, q.c10->overlayUV,
                                         q.c01->overlayUV, q.c11->overlayUV, u, v);
            vert.useGlint = q.c00->useGlint;
            vert.textureID = q.c00->textureID;
            vert.glintUV = bilerp(q.c00->glintUV, q.c10->glintUV,
                                  q.c01->glintUV, q.c11->glintUV, u, v);
            vert.glintTexture = q.c00->glintTexture;
            vert.useLight = q.c00->useLight;
            vert.lightUV = bilerpIvec2(q.c00->lightUV, q.c10->lightUV,
                                       q.c01->lightUV, q.c11->lightUV, u, v);
            vert.coordinate = q.c00->coordinate;
            vert.albedoEmission = q.c00->albedoEmission;
            vert.postBase = bilerp(q.c00->postBase, q.c10->postBase,
                                   q.c01->postBase, q.c11->postBase, u, v);
            vert.emissiveBlockType = q.c00->emissiveBlockType;

            // Sample height and displace along negative face normal (inward from surface)
            float height = HeightSampler::sample(
                input.normalRGBA, input.albedoRGBA, input.material,
                input.hasLabPBRHeight, input.isAutoPBR,
                vert.textureUV.x, vert.textureUV.y,
                input.uvMinX, input.uvMinY, input.uvMaxX, input.uvMaxY);

            // Displacement: height=1 means surface (no displacement), height=0 means max depth
            // Edge fade: displacement must be zero at face boundaries to maintain watertight
            // cube edges. Without this, adjacent perpendicular faces tear apart at their seams.
            float edgeFade = std::min({u, 1.0f - u, v, 1.0f - v}) * static_cast<float>(N);
            edgeFade = std::clamp(edgeFade, 0.0f, 1.0f); // 0 at edge, 1 one texel inward

            float displacement = (1.0f - height) * input.heightScale * edgeFade;
            glm::vec3 offset = -faceNormal * displacement;
            vert.pos += offset;
            vert.postBase += offset; // Keep motion vectors consistent
        }
    }

    // Generate triangle indices (CCW winding)
    for (uint32_t row = 0; row < N; row++) {
        for (uint32_t col = 0; col < N; col++) {
            uint32_t i00 = row * gridW + col;
            uint32_t i10 = row * gridW + col + 1;
            uint32_t i01 = (row + 1) * gridW + col;
            uint32_t i11 = (row + 1) * gridW + col + 1;
            // Triangle 1: bottom-left
            out.indices.push_back(i00);
            out.indices.push_back(i10);
            out.indices.push_back(i11);
            // Triangle 2: top-right
            out.indices.push_back(i00);
            out.indices.push_back(i11);
            out.indices.push_back(i01);
        }
    }

    return out;
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
