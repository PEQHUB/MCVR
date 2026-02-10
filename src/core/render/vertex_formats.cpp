#include "common/shared.hpp"

#include "common/singleton.hpp"
#include "core/all_extern.hpp"
#include "core/vulkan/all_core_vulkan.hpp"

template <>
vk::VertexLayoutInfo &vk::Vertex::vertexLayoutInfo<vk::VertexFormat::PositionTexColor>() {
    static std::vector<VertexAttribute> attributes = {
        {VK_FORMAT_R32G32B32_SFLOAT, offsetof(VertexFormat::PositionTexColor, position)},
        {VK_FORMAT_R32G32_SFLOAT, offsetof(VertexFormat::PositionTexColor, uv)},
        {VK_FORMAT_R8G8B8A8_UNORM, offsetof(VertexFormat::PositionTexColor, color)},
    };
    static vk::VertexLayoutInfo vertexLayoutInfo = initVertexLayout<vk::VertexFormat::PositionTexColor>(attributes);
    return vertexLayoutInfo;
}

template <>
vk::VertexLayoutInfo &vk::Vertex::vertexLayoutInfo<vk::VertexFormat::PositionColor>() {
    static std::vector<VertexAttribute> attributes = {
        {VK_FORMAT_R32G32B32_SFLOAT, offsetof(VertexFormat::PositionColor, position)},
        {VK_FORMAT_R8G8B8A8_UNORM, offsetof(VertexFormat::PositionColor, color)},
    };
    static vk::VertexLayoutInfo vertexLayoutInfo = initVertexLayout<vk::VertexFormat::PositionColor>(attributes);
    return vertexLayoutInfo;
}

template <>
vk::VertexLayoutInfo &vk::Vertex::vertexLayoutInfo<vk::VertexFormat::PositionTex>() {
    static std::vector<VertexAttribute> attributes = {
        {VK_FORMAT_R32G32B32_SFLOAT, offsetof(VertexFormat::PositionTex, position)},
        {VK_FORMAT_R32G32_SFLOAT, offsetof(VertexFormat::PositionTex, uv)},
    };
    static vk::VertexLayoutInfo vertexLayoutInfo = initVertexLayout<vk::VertexFormat::PositionTex>(attributes);
    return vertexLayoutInfo;
}

template <>
vk::VertexLayoutInfo &vk::Vertex::vertexLayoutInfo<vk::VertexFormat::PositionColorTexLight>() {
    static std::vector<VertexAttribute> attributes = {
        {VK_FORMAT_R32G32B32_SFLOAT, offsetof(VertexFormat::PositionColorTexLight, position)},
        {VK_FORMAT_R8G8B8A8_UNORM, offsetof(VertexFormat::PositionColorTexLight, color)},
        {VK_FORMAT_R32G32_SFLOAT, offsetof(VertexFormat::PositionColorTexLight, uv0)},
        {VK_FORMAT_R16G16_SINT, offsetof(VertexFormat::PositionColorTexLight, uv2)},
    };
    static vk::VertexLayoutInfo vertexLayoutInfo =
        initVertexLayout<vk::VertexFormat::PositionColorTexLight>(attributes);
    return vertexLayoutInfo;
}

template <>
vk::VertexLayoutInfo &vk::Vertex::vertexLayoutInfo<vk::VertexFormat::PositionColorTexOverlayLightNormal>() {
    static std::vector<VertexAttribute> attributes = {
        {VK_FORMAT_R32G32B32_SFLOAT, offsetof(VertexFormat::PositionColorTexOverlayLightNormal, position)},
        {VK_FORMAT_R8G8B8A8_UNORM, offsetof(VertexFormat::PositionColorTexOverlayLightNormal, color)},
        {VK_FORMAT_R32G32_SFLOAT, offsetof(VertexFormat::PositionColorTexOverlayLightNormal, uv0)},
        {VK_FORMAT_R16G16_SINT, offsetof(VertexFormat::PositionColorTexOverlayLightNormal, uv1)},
        {VK_FORMAT_R16G16_SINT, offsetof(VertexFormat::PositionColorTexOverlayLightNormal, uv2)},
        {VK_FORMAT_R8G8B8A8_SNORM, offsetof(VertexFormat::PositionColorTexOverlayLightNormal, normal)},
    };
    static vk::VertexLayoutInfo vertexLayoutInfo =
        initVertexLayout<vk::VertexFormat::PositionColorTexOverlayLightNormal>(attributes);
    return vertexLayoutInfo;
}

template <>
vk::VertexLayoutInfo &vk::Vertex::vertexLayoutInfo<vk::VertexFormat::PositionOnly>() {
    static std::vector<VertexAttribute> attributes = {
        {VK_FORMAT_R32G32B32_SFLOAT, offsetof(VertexFormat::PositionOnly, position)},
    };
    static vk::VertexLayoutInfo vertexLayoutInfo = initVertexLayout<vk::VertexFormat::PositionOnly>(attributes);
    return vertexLayoutInfo;
}

template <>
vk::VertexLayoutInfo &vk::Vertex::vertexLayoutInfo<vk::VertexFormat::PBRTriangle>() {
    static std::vector<VertexAttribute> attributes = {
        {VK_FORMAT_R32G32B32_SFLOAT, offsetof(VertexFormat::PBRTriangle, pos)},
        {VK_FORMAT_R32_UINT, offsetof(VertexFormat::PBRTriangle, useNorm)},
        {VK_FORMAT_R32G32B32_SFLOAT, offsetof(VertexFormat::PBRTriangle, norm)},
        {VK_FORMAT_R32_UINT, offsetof(VertexFormat::PBRTriangle, useColorLayer)},
        {VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(VertexFormat::PBRTriangle, colorLayer)},
        {VK_FORMAT_R32_UINT, offsetof(VertexFormat::PBRTriangle, useTexture)},
        {VK_FORMAT_R32_UINT, offsetof(VertexFormat::PBRTriangle, useOverlay)},
        {VK_FORMAT_R32G32_SFLOAT, offsetof(VertexFormat::PBRTriangle, textureUV)},
        {VK_FORMAT_R32G32_SINT, offsetof(VertexFormat::PBRTriangle, overlayUV)},
        {VK_FORMAT_R32_UINT, offsetof(VertexFormat::PBRTriangle, useGlint)},
        {VK_FORMAT_R32_UINT, offsetof(VertexFormat::PBRTriangle, textureID)},
        {VK_FORMAT_R32G32_SFLOAT, offsetof(VertexFormat::PBRTriangle, glintUV)},
        {VK_FORMAT_R32_UINT, offsetof(VertexFormat::PBRTriangle, glintTexture)},
        {VK_FORMAT_R32_UINT, offsetof(VertexFormat::PBRTriangle, useLight)},
        {VK_FORMAT_R32G32_SINT, offsetof(VertexFormat::PBRTriangle, lightUV)},
        {VK_FORMAT_R32_UINT, offsetof(VertexFormat::PBRTriangle, coordinate)},
        {VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(VertexFormat::PBRTriangle, postBase)},
    };
    static vk::VertexLayoutInfo vertexLayoutInfo = initVertexLayout<vk::VertexFormat::PBRTriangle>(attributes);
    return vertexLayoutInfo;
}
