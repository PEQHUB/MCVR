#pragma once

#include "common/shared.hpp"
#include <vector>

// CPU-side greedy meshing for WORLD_SOLID geometry.
// Merges adjacent coplanar block faces with identical materials into larger quads.
// Reduces triangle count by 50-70%, directly cutting RT trace and BLAS build costs.
class GreedyMesher {
public:
    struct Result {
        std::vector<vk::VertexFormat::PBRTriangle> vertices;
        std::vector<uint32_t> indices;
        uint32_t origTriCount;
        uint32_t mergedTriCount;
    };

    // Merge coplanar axis-aligned quads in a WORLD_SOLID chunk section.
    // Non-axis-aligned geometry (stairs, slabs, fences) passes through unchanged.
    static Result merge(const std::vector<vk::VertexFormat::PBRTriangle>& vertices,
                        const std::vector<uint32_t>& indices);
};
