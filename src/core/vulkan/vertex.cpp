#include "core/vulkan/vertex.hpp"

#include "common/shared.hpp"

template <>
vk::VertexLayoutInfo &vk::Vertex::vertexLayoutInfo<vk::VertexFormat::Triangle>() {
    static std::vector<VertexAttribute> attributes = {
        {VK_FORMAT_R32G32B32_SFLOAT, offsetof(vk::VertexFormat::Triangle, pos)},
        {VK_FORMAT_R32G32B32_SFLOAT, offsetof(vk::VertexFormat::Triangle, color)},
    };

    static vk::VertexLayoutInfo vertexLayoutInfo = initVertexLayout<vk::VertexFormat::Triangle>(attributes);
    return vertexLayoutInfo;
}

template <>
vk::VertexLayoutInfo &vk::Vertex::vertexLayoutInfo<vk::VertexFormat::TexturedTriangle>() {
    static std::vector<VertexAttribute> attributes = {
        {VK_FORMAT_R32G32B32_SFLOAT, offsetof(vk::VertexFormat::TexturedTriangle, pos)},
        {VK_FORMAT_R32G32_SFLOAT, offsetof(vk::VertexFormat::TexturedTriangle, uv)},
    };
    static vk::VertexLayoutInfo vertexLayoutInfo = initVertexLayout<vk::VertexFormat::TexturedTriangle>(attributes);
    return vertexLayoutInfo;
}

template <>
vk::VertexLayoutInfo &vk::Vertex::vertexLayoutInfo<vk::VertexFormat::ArrayTexturedTriangle>() {
    static std::vector<VertexAttribute> attributes = {
        {VK_FORMAT_R32G32B32_SFLOAT, offsetof(vk::VertexFormat::ArrayTexturedTriangle, pos)},
        {VK_FORMAT_R32G32_SFLOAT, offsetof(vk::VertexFormat::ArrayTexturedTriangle, uv)},
        {VK_FORMAT_R32_SFLOAT, offsetof(vk::VertexFormat::ArrayTexturedTriangle, textureLayer)},
    };
    static vk::VertexLayoutInfo vertexLayoutInfo = initVertexLayout<vk::VertexFormat::ArrayTexturedTriangle>(attributes);
    return vertexLayoutInfo;
}