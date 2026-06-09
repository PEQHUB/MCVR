#pragma once

#include "common/shared.hpp"
#include "core/render/height_sampler.hpp"
#include "core/render/textures.hpp"

#include <cstdint>
#include <vector>

// Tessellates axis-aligned Minecraft block face quads into NxN triangle grids
// with height-displaced vertices. Output feeds directly into WORLD_SOLID BLAS.
class Tessellator {
  public:
    struct Input {
        // Original quad vertices (4 corners, same order as chunk builder)
        // Index pattern: [v0, v1, v2, v2, v3, v0] — so unique verts are v0,v1,v2,v3
        const vk::VertexFormat::PBRTriangle *v0;
        const vk::VertexFormat::PBRTriangle *v1;
        const vk::VertexFormat::PBRTriangle *v2;
        const vk::VertexFormat::PBRTriangle *v3;

        // Face normal direction (0=+X, 1=-X, 2=+Y, 3=-Y, 4=+Z, 5=-Z)
        uint32_t faceAxis;

        // Height data
        HeightLayerView labPbrHeight;  // Texture-array LabPBR height plane (normal alpha)
        HeightLayerView autoPbrHeight; // Texture-array AutoPBR luminance plane
        const Textures::TextureRGBAData *normalRGBA;  // For LabPBR height (normal alpha)
        const Textures::TextureRGBAData *albedoRGBA;   // For AutoPBR height (albedo luminance)
        const vk::Data::MaterialClassEntry *material;  // Material class params (may be null)
        bool hasLabPBRHeight;
        bool isAutoPBR;

        // Tile UV bounds in atlas
        float uvMinX, uvMinY, uvMaxX, uvMaxY;

        // Tessellation params
        uint32_t tessLevel;  // NxN grid (2-32)
        float heightScale;   // Displacement depth in world units (pomDepth)

        // Per-edge fade mask: bit set = fade that edge (perpendicular/exposed).
        // Bit 0 = U=0 edge, Bit 1 = U=1 edge, Bit 2 = V=0 edge, Bit 3 = V=1 edge.
        // Coplanar edges (adjacent tessellated block) should NOT be faded.
        uint32_t fadeEdgeMask = 0xF; // default: fade all edges
    };

    struct Output {
        std::vector<vk::VertexFormat::PBRTriangle> vertices;
        std::vector<uint32_t> indices;
        // Edge displacement values for seam stitching with neighbors.
        // Each has N entries (one per texel along that edge).
        std::vector<float> edgeDispU0, edgeDispU1, edgeDispV0, edgeDispV1;
    };

    // UV-ordered quad corners for bilinear interpolation
    struct QuadCorners {
        const vk::VertexFormat::PBRTriangle *c00; // parametric (0,0)
        const vk::VertexFormat::PBRTriangle *c10; // parametric (1,0)
        const vk::VertexFormat::PBRTriangle *c01; // parametric (0,1)
        const vk::VertexFormat::PBRTriangle *c11; // parametric (1,1)
    };

    // Tessellate a quad into an NxN triangle grid with height displacement.
    // Returns (tessLevel+1)^2 vertices and 2*tessLevel^2 triangles.
    static Output tessellate(const Input &input);

    // Map raw quad vertices to UV-ordered bilinear corners.
    static QuadCorners mapCornersPublic(const Input &input);

    // Interpolate all PBRTriangle fields at parametric (u,v) from UV-ordered corners.
    static void interpVertexPublic(vk::VertexFormat::PBRTriangle &vert, const QuadCorners &q, float u, float v);

    // Determine tessellation level based on distance and texture resolution.
    // Returns a level capped by texture resolution and max setting, reduced by distance LOD.
    static uint32_t computeTessLevel(float distanceToCamera, uint32_t textureResolution,
                                     uint32_t maxTessLevel,
                                     float nearDist = 32.0f, float midDist = 96.0f, float farDist = 192.0f);
};
