#pragma once

#include "core/all_extern.hpp"

#include <vector>

namespace vk {
struct VertexLayoutInfo {
    VkVertexInputBindingDescription bindingDescription;
    std::vector<VkVertexInputAttributeDescription> attributeDescriptions;
};

struct VertexAttribute {
    VkFormat format;
    uint32_t offset;
};

struct Vertex {
    template <typename T>
    static VertexLayoutInfo initVertexLayout(std::vector<VertexAttribute> &attributes);

    template <typename T>
    static void buildVertexLayoutInfo(VertexLayoutInfo &vertexLayoutInfo, std::vector<VertexAttribute> &attributes);

    template <typename T>
    static VertexLayoutInfo &vertexLayoutInfo();
};

template <typename T>
vk::VertexLayoutInfo vk::Vertex::initVertexLayout(std::vector<VertexAttribute> &attributes) {
    vk::VertexLayoutInfo vertexLayoutInfo{};
    buildVertexLayoutInfo<T>(vertexLayoutInfo, attributes);
    return vertexLayoutInfo;
}

template <typename T>
void Vertex::buildVertexLayoutInfo(VertexLayoutInfo &vertexLayoutInfo, std::vector<VertexAttribute> &attributes) {
    vertexLayoutInfo.bindingDescription.binding = 0;
    vertexLayoutInfo.bindingDescription.stride = sizeof(T);
    vertexLayoutInfo.bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    for (uint32_t i = 0; i < attributes.size(); i++) {
        vertexLayoutInfo.attributeDescriptions.push_back({
            .location = i,
            .binding = 0,
            .format = attributes[i].format,
            .offset = attributes[i].offset,
        });
    }
}
}; // namespace vk