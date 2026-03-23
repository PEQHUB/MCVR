#include "core/render/entities.hpp"

#include "core/render/buffers.hpp"
#include "core/render/render_framework.hpp"
#include "core/render/renderer.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <unordered_map>

using Vertex = glm::vec3;
using Triangle = std::array<Vertex, 3>;
using VertexIdentifier = std::array<uint32_t, 2>;
using TriangleIdentifier = std::array<VertexIdentifier, 3>;

struct TriangleHash {
    static inline void hash_combine(std::size_t &seed, std::size_t h) noexcept {
        seed ^= h + 0x9e3779b97f4a7c15ull + (seed << 6) + (seed >> 2);
    }

    std::size_t operator()(Triangle const &k) const noexcept {
        std::size_t seed = 0;
        for (auto const &v : k) { hash_combine(seed, std::hash<glm::vec3>{}(v)); }
        return seed;
    }
};


EntityBuildData::EntityBuildData(int hashCode,
                                 double x,
                                 double y,
                                 double z,
                                 int rtFlag,
                                 int prebuiltBLAS,
                                 World::Coordinates coordinate,
                                 uint32_t geometryCount,
                                 std::vector<World::GeometryTypes> &&geometryTypes,
                                 std::vector<std::vector<vk::VertexFormat::PBRTriangle>> &&vertices,
                                 std::vector<std::vector<uint32_t>> &&indices)
    : hashCode(hashCode),
      x(x),
      y(y),
      z(z),
      rtFlag(rtFlag),
      prebuiltBLAS(prebuiltBLAS),
      coordinate(coordinate),
      geometryCount(geometryCount),
      geometryTypes(std::move(geometryTypes)),
      vertices(std::move(vertices)),
      indices(std::move(indices)),
      vertexBufferAddresses(),
      indexBufferAddresses() {}

void EntityBuildDataBatch::addData(std::shared_ptr<EntityBuildData> data) {
    datas.push_back(data);
}

void EntityBuildDataBatch::build() {
    auto framework = Renderer::instance().framework();
    auto vma = framework->vma();
    auto device = framework->device();
    auto physicalDevice = framework->physicalDevice();

    std::vector<uint32_t> instanceOffsets;
    std::vector<uint32_t> geometryVertexOffsets; // number of vertices to skip
    std::vector<uint32_t> geometryIndexOffsets;  // number of indices to skip
    uint32_t totalGeometryCount = 0;
    uint32_t totalVertexCount = 0;
    uint32_t totalIndexCount = 0;

    for (auto data : datas) {
        instanceOffsets.push_back(totalGeometryCount);
        for (int i = 0; i < data->geometryCount; i++) {
            geometryVertexOffsets.push_back(totalVertexCount);
            geometryIndexOffsets.push_back(totalIndexCount);

            totalVertexCount += data->vertices[i].size();
            totalIndexCount += data->indices[i].size();
        }

        totalGeometryCount += data->geometryCount;
    }

    vertexBuffer = vk::DeviceLocalBuffer::create(
        vma, device, totalVertexCount * sizeof(vk::VertexFormat::PBRTriangle),
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    indexBuffer = vk::DeviceLocalBuffer::create(
        vma, device, totalIndexCount * sizeof(uint32_t),
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    vk::VertexFormat::PBRTriangle *vertexPtr = static_cast<vk::VertexFormat::PBRTriangle *>(vertexBuffer->mappedPtr());
    uint32_t *indexPtr = static_cast<uint32_t *>(indexBuffer->mappedPtr());
    for (auto data : datas) {
        for (int i = 0; i < data->geometryCount; i++) {
            std::memcpy(vertexPtr, data->vertices[i].data(),
                        data->vertices[i].size() * sizeof(vk::VertexFormat::PBRTriangle));
            vertexPtr += data->vertices[i].size();

            std::memcpy(indexPtr, data->indices[i].data(), data->indices[i].size() * sizeof(uint32_t));
            indexPtr += data->indices[i].size();
        }

        totalGeometryCount += data->geometryCount;
    }
    vertexBuffer->flushStagingBuffer();
    indexBuffer->flushStagingBuffer();

    blasBatchBuilder = vk::BLASBatchBuilder::create();
    std::vector<uint32_t> nonPrebuildInstances;
    for (int instanceIndex = 0; auto data : datas) {
        auto instanceOffset = instanceOffsets[instanceIndex];
        std::shared_ptr<vk::BLASBuilder> blasBuilder = nullptr;
        std::shared_ptr<vk::BLASBuilder::BLASGeometryBuilder> blasGeometryBuilder = nullptr;
        if (data->prebuiltBLAS < 0) {
            nonPrebuildInstances.push_back(instanceIndex);
            blasBuilder = blasBatchBuilder->defineBLASBuilder();
            blasGeometryBuilder = blasBuilder->beginGeometries();
        }
        for (int i = 0; i < data->geometryCount; i++) {
            VkDeviceAddress vertexBufferAddress =
                vertexBuffer->bufferAddress() +
                geometryVertexOffsets[instanceOffset + i] * sizeof(vk::VertexFormat::PBRTriangle);
            VkDeviceAddress indexBufferAddress =
                indexBuffer->bufferAddress() + geometryIndexOffsets[instanceOffset + i] * sizeof(uint32_t);
            data->vertexBufferAddresses.push_back(vertexBufferAddress);
            data->indexBufferAddresses.push_back(indexBufferAddress);
            if (data->prebuiltBLAS < 0) {
                blasGeometryBuilder->defineTriangleGeomrtry<vk::VertexFormat::PBRTriangle>(
                    vertexBufferAddress, data->vertices[i].size(), indexBufferAddress, data->indices[i].size(),
                    data->geometryTypes[i] == World::WORLD_SOLID);
            }
        }
        if (data->prebuiltBLAS < 0) {
            blasGeometryBuilder->endGeometries();
            blasBuilder->defineBuildProperty(VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR)
                ->querySizeInfo(device);
        }

        instanceIndex++;
    }

    auto blass = blasBatchBuilder->allocateBuffers(physicalDevice, device, vma)->build(device);
    for (int i = 0; i < nonPrebuildInstances.size(); i++) { datas[nonPrebuildInstances[i]]->blas = blass[i]; }
}

void EntityPostBuildDataBatch::addData(std::shared_ptr<EntityBuildData> data) {
    datas.push_back(data);
}

Entity::Entity(std::shared_ptr<EntityBuildData> chunkBuildData) {
    hashCode = chunkBuildData->hashCode;
    x = chunkBuildData->x;
    y = chunkBuildData->y;
    z = chunkBuildData->z;
    rtFlag = chunkBuildData->rtFlag;
    prebuiltBLAS = chunkBuildData->prebuiltBLAS;
    coordinate = chunkBuildData->coordinate;

    blas = chunkBuildData->blas;
    vertexBufferAddresses =
        std::make_shared<std::vector<VkDeviceAddress>>(std::move(chunkBuildData->vertexBufferAddresses));
    indexBufferAddresses =
        std::make_shared<std::vector<VkDeviceAddress>>(std::move(chunkBuildData->indexBufferAddresses));

    geometryCount = chunkBuildData->geometryCount;
    geometryTypes = std::make_shared<std::vector<World::GeometryTypes>>(std::move(chunkBuildData->geometryTypes));
    vertices =
        std::make_shared<std::vector<std::vector<vk::VertexFormat::PBRTriangle>>>(std::move(chunkBuildData->vertices));
    indices = std::make_shared<std::vector<std::vector<uint32_t>>>(std::move(chunkBuildData->indices));
}

EntityBatch::EntityBatch(std::shared_ptr<EntityBuildDataBatch> entityBuildDataBatch) {
    for (auto data : entityBuildDataBatch->datas) {
        auto entity = Entity::create(data);
        entity->vertexBuffer = entityBuildDataBatch->vertexBuffer;
        entity->indexBuffer = entityBuildDataBatch->indexBuffer;
        entities.push_back(entity);
    }

    vertexBuffer = entityBuildDataBatch->vertexBuffer;
    indexBuffer = entityBuildDataBatch->indexBuffer;
}

EntityPost::EntityPost(std::shared_ptr<EntityBuildData> chunkBuildData) {
    x = chunkBuildData->x;
    y = chunkBuildData->y;
    z = chunkBuildData->z;

    geometryCount = chunkBuildData->geometryCount;
    vertices = std::move(chunkBuildData->vertices);
    indices = std::move(chunkBuildData->indices);

    auto framework = Renderer::instance().framework();
    auto vma = framework->vma();
    auto device = framework->device();
    auto physicalDevice = framework->physicalDevice();

    for (int i = 0; i < geometryCount; i++) {
        auto vertexBuffer = vk::DeviceLocalBuffer::create(
            vma, device, vertices[i].size() * sizeof(vk::VertexFormat::PBRTriangle), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
        auto indexBuffer = vk::DeviceLocalBuffer::create(vma, device, indices[i].size() * sizeof(uint32_t),
                                                         VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

        vertexBuffer->uploadToStagingBuffer(vertices[i].data());
        indexBuffer->uploadToStagingBuffer(indices[i].data());

        vertexBuffers.push_back(vertexBuffer);
        indexBuffers.push_back(indexBuffer);
    }
}

EntityPostBatch::EntityPostBatch(std::shared_ptr<EntityPostBuildDataBatch> entityPostBuildDataBatch) {
    for (auto data : entityPostBuildDataBatch->datas) { entities.push_back(EntityPost::create(data)); }
}

Entities::Entities(std::shared_ptr<Framework> framework) {}

void Entities::resetFrame() {
    auto framework = Renderer::instance().framework();
    framework->safeAcquireCurrentContext();
    auto &gc = framework->gc();

    gc.collect(entityBuildDataBatch_);
    entityBuildDataBatch_ = EntityBuildDataBatch::create();

    gc.collect(entityPostBuildDataBatch_);
    entityPostBuildDataBatch_ = EntityPostBuildDataBatch::create();

    gc.collect(entityBatch_);
    entityBatch_ = nullptr;

    gc.collect(entityPostBatch_);
    entityPostBatch_ = nullptr;

    gc.collect(blasBatchBuilder_);
    blasBatchBuilder_ = nullptr;
}

void Entities::queueBuild(EntitiesBuildTask task) {
    Renderer::instance().framework()->safeAcquireCurrentContext();
    auto framework = Renderer::instance().framework();
    auto vma = framework->vma();
    auto device = framework->device();
    auto physicalDevice = framework->physicalDevice();

    std::set<int> textureIDs;

    uint32_t geometryAccu = 0;
    for (int e = 0; e < task.entityCount; e++) {
        uint32_t geometryIndex = geometryAccu;
        uint32_t geometryCountIncludeGlint = task.entityGeometryCounts[e];
        geometryAccu += geometryCountIncludeGlint;

        uint32_t allVertexCount = 0, allIndexCount = 0;
        std::vector<World::GeometryTypes> geometryTypes;
        std::vector<std::vector<vk::VertexFormat::PBRTriangle>> vertices;
        std::vector<std::vector<uint32_t>> indices;
        int hashCode = task.entityHashCodes[e];
        double x = task.entityXs[e];
        double y = task.entityYs[e];
        double z = task.entityZs[e];
        int rtFlag = task.entityRTFlags[e];
        int prebuiltBLAS = task.entityPrebuiltBLASs[e];
        World::Coordinates coordinate = task.coordinate;
        bool post = task.entityPosts[e];

        uint32_t geometryCountWithoutGlint = 0;
        for (int i = 0; i < task.entityGeometryCounts[e]; i++) {
            World::GeometryTypes geometryType =
                static_cast<World::GeometryTypes>(task.geometryTypes[geometryIndex + i]);
            int geometryTexture = task.geometryTextures[geometryIndex + i];
            geometryTypes.push_back(geometryType);

            auto &geometryVertices = vertices.emplace_back();
            auto &geometryIndices = indices.emplace_back();

            if (task.vertexFormats[geometryIndex + i] == World::PBR_TRIANGLE) {
                geometryVertices.resize(task.vertexCounts[geometryIndex + i]);
                std::memcpy(geometryVertices.data(), task.vertices[geometryIndex + i],
                            task.vertexCounts[geometryIndex + i] * sizeof(vk::VertexFormat::PBRTriangle));
            } else {
                for (int j = 0; j < task.vertexCounts[geometryIndex + i]; j++) {
                    vk::VertexFormat::PBRTriangle vertex{};

                    switch (task.vertexFormats[geometryIndex + i]) {
                        case World::POSITION_COLOR_TEXTURE_LIGHT_NORMAL: {
                            vk::VertexFormat::PositionColorTexLightNormal *vertices =
                                static_cast<vk::VertexFormat::PositionColorTexLightNormal *>(
                                    task.vertices[geometryIndex + i]);

                            vertex.pos = vertices[j].position;

                            vertex.useColorLayer = 1;
                            vertex.colorLayer = glm::vec4{
                                vertices[j].color & 0xFF,
                                (vertices[j].color >> 8) & 0xFF,
                                (vertices[j].color >> 16) & 0xFF,
                                (vertices[j].color >> 24) & 0xFF,
                            };
                            vertex.colorLayer /= 255.0;

                            vertex.useTexture = 1;
                            vertex.textureUV = vertices[j].uv0;

                            vertex.useLight = 1;
                            vertex.lightUV = glm::ivec2{
                                vertices[j].uv2 & 0xFFFF,
                                (vertices[j].uv2 >> 16) & 0xFFFF,
                            };

                            vertex.useNorm = 1;
                            vertex.norm = glm::vec3{
                                (int8_t)(vertices[j].normal & 0xFF),
                                (int8_t)((vertices[j].normal >> 8) & 0xFF),
                                (int8_t)((vertices[j].normal >> 16) & 0xFF),
                            };

                            break;
                        }
                        case World::POSITION_COLOR_TEXTURE_OVERLAY_LIGHT_NORMAL: {
                            vk::VertexFormat::PositionColorTexOverlayLightNormal *vertices =
                                static_cast<vk::VertexFormat::PositionColorTexOverlayLightNormal *>(
                                    task.vertices[geometryIndex + i]);

                            vertex.pos = vertices[j].position;

                            vertex.useColorLayer = 1;
                            vertex.colorLayer = glm::vec4{
                                vertices[j].color & 0xFF,
                                (vertices[j].color >> 8) & 0xFF,
                                (vertices[j].color >> 16) & 0xFF,
                                (vertices[j].color >> 24) & 0xFF,
                            };
                            vertex.colorLayer /= 255.0;

                            vertex.useTexture = 1;
                            vertex.textureUV = vertices[j].uv0;

                            vertex.useOverlay = 1;
                            vertex.overlayUV = glm::ivec2{vertices[j].uv1 & 0xFFFF, (vertices[j].uv1 >> 16) & 0xFFFF};

                            vertex.useLight = 1;
                            vertex.lightUV = glm::vec2{
                                vertices[j].uv2 & 0xFFFF,
                                (vertices[j].uv2 >> 16) & 0xFFFF,
                            };

                            vertex.useNorm = 1;
                            vertex.norm = glm::vec3{
                                (int8_t)(vertices[j].normal & 0xFF),
                                (int8_t)((vertices[j].normal >> 8) & 0xFF),
                                (int8_t)((vertices[j].normal >> 16) & 0xFF),
                            };

                            break;
                        }
                        case World::POSITION_TEXTURE_COLOR_LIGHT: {
                            vk::VertexFormat::PositionTexColorLight *vertices =
                                static_cast<vk::VertexFormat::PositionTexColorLight *>(
                                    task.vertices[geometryIndex + i]);

                            vertex.pos = vertices[j].position;

                            vertex.useTexture = 1;
                            vertex.textureUV = vertices[j].uv0;

                            vertex.useColorLayer = 1;
                            vertex.colorLayer = glm::vec4{
                                vertices[j].color & 0xFF,
                                (vertices[j].color >> 8) & 0xFF,
                                (vertices[j].color >> 16) & 0xFF,
                                (vertices[j].color >> 24) & 0xFF,
                            };
                            vertex.colorLayer /= 255.0;

                            vertex.useLight = 1;
                            vertex.lightUV = glm::vec2{
                                vertices[j].uv2 & 0xFFFF,
                                (vertices[j].uv2 >> 16) & 0xFFFF,
                            };

                            break;
                        }
                        case World::POSITION: {
                            vk::VertexFormat::PositionOnly *vertices =
                                static_cast<vk::VertexFormat::PositionOnly *>(task.vertices[geometryIndex + i]);

                            vertex.pos = vertices[j].position;

                            break;
                        }
                        case World::POSITION_COLOR: {
                            vk::VertexFormat::PositionColor *vertices =
                                static_cast<vk::VertexFormat::PositionColor *>(task.vertices[geometryIndex + i]);

                            vertex.pos = vertices[j].position;

                            vertex.useColorLayer = 1;
                            vertex.colorLayer = glm::vec4{
                                vertices[j].color & 0xFF,
                                (vertices[j].color >> 8) & 0xFF,
                                (vertices[j].color >> 16) & 0xFF,
                                (vertices[j].color >> 24) & 0xFF,
                            };
                            vertex.colorLayer /= 255.0;

                            break;
                        }
                        case World::LINES: {
                            vk::VertexFormat::PositionColorNormal *vertices =
                                static_cast<vk::VertexFormat::PositionColorNormal *>(task.vertices[geometryIndex + i]);

                            vertex.pos = vertices[j].position;

                            vertex.useColorLayer = 1;
                            vertex.colorLayer = glm::vec4{
                                vertices[j].color & 0xFF,
                                (vertices[j].color >> 8) & 0xFF,
                                (vertices[j].color >> 16) & 0xFF,
                                (vertices[j].color >> 24) & 0xFF,
                            };
                            vertex.colorLayer /= 255.0;

                            vertex.useNorm = 1;
                            vertex.norm = glm::vec3{
                                (int8_t)(vertices[j].normal & 0xFF),
                                (int8_t)((vertices[j].normal >> 8) & 0xFF),
                                (int8_t)((vertices[j].normal >> 16) & 0xFF),
                            };

                            break;
                        }
                        case World::POSITION_COLOR_LIGHT: {
                            vk::VertexFormat::PositionColorLight *vertices =
                                static_cast<vk::VertexFormat::PositionColorLight *>(task.vertices[geometryIndex + i]);

                            vertex.pos = vertices[j].position;

                            vertex.useColorLayer = 1;
                            vertex.colorLayer = glm::vec4{
                                vertices[j].color & 0xFF,
                                (vertices[j].color >> 8) & 0xFF,
                                (vertices[j].color >> 16) & 0xFF,
                                (vertices[j].color >> 24) & 0xFF,
                            };
                            vertex.colorLayer /= 255.0;

                            vertex.useLight = 1;
                            vertex.lightUV = glm::vec2{
                                vertices[j].uv2 & 0xFFFF,
                                (vertices[j].uv2 >> 16) & 0xFFFF,
                            };

                            break;
                        }
                        case World::POSITION_TEXTURE: {
                            vk::VertexFormat::PositionTex *vertices =
                                static_cast<vk::VertexFormat::PositionTex *>(task.vertices[geometryIndex + i]);

                            vertex.pos = vertices[j].position;

                            vertex.useTexture = 1;
                            vertex.textureUV = vertices[j].uv;

                            break;
                        }
                        case World::POSITION_TEXTURE_COLOR: {
                            vk::VertexFormat::PositionTexColor *vertices =
                                static_cast<vk::VertexFormat::PositionTexColor *>(task.vertices[geometryIndex + i]);

                            vertex.pos = vertices[j].position;

                            vertex.useTexture = 1;
                            vertex.textureUV = vertices[j].uv;

                            vertex.useColorLayer = 1;
                            vertex.colorLayer = glm::vec4{
                                vertices[j].color & 0xFF,
                                (vertices[j].color >> 8) & 0xFF,
                                (vertices[j].color >> 16) & 0xFF,
                                (vertices[j].color >> 24) & 0xFF,
                            };
                            vertex.colorLayer /= 255.0;

                            break;
                        }
                        case World::POSITION_COLOR_TEXTURE_LIGHT: {
                            vk::VertexFormat::PositionColorTexLight *vertices =
                                static_cast<vk::VertexFormat::PositionColorTexLight *>(
                                    task.vertices[geometryIndex + i]);

                            vertex.pos = vertices[j].position;

                            vertex.useColorLayer = 1;
                            vertex.colorLayer = glm::vec4{
                                vertices[j].color & 0xFF,
                                (vertices[j].color >> 8) & 0xFF,
                                (vertices[j].color >> 16) & 0xFF,
                                (vertices[j].color >> 24) & 0xFF,
                            };
                            vertex.colorLayer /= 255.0;

                            vertex.useTexture = 1;
                            vertex.textureUV = vertices[j].uv0;

                            vertex.useLight = 1;
                            vertex.lightUV = glm::vec2{
                                vertices[j].uv2 & 0xFFFF,
                                (vertices[j].uv2 >> 16) & 0xFFFF,
                            };

                            break;
                        }
                        case World::POSITION_TEXTURE_LIGHT_COLOR: {
                            vk::VertexFormat::PositionTexLightColor *vertices =
                                static_cast<vk::VertexFormat::PositionTexLightColor *>(
                                    task.vertices[geometryIndex + i]);

                            vertex.pos = vertices[j].position;

                            vertex.useTexture = 1;
                            vertex.textureUV = vertices[j].uv0;

                            vertex.useLight = 1;
                            vertex.lightUV = glm::vec2{
                                vertices[j].uv2 & 0xFFFF,
                                (vertices[j].uv2 >> 16) & 0xFFFF,
                            };

                            vertex.useColorLayer = 1;
                            vertex.colorLayer = glm::vec4{
                                vertices[j].color & 0xFF,
                                (vertices[j].color >> 8) & 0xFF,
                                (vertices[j].color >> 16) & 0xFF,
                                (vertices[j].color >> 24) & 0xFF,
                            };
                            vertex.colorLayer /= 255.0;

                            break;
                        }
                        case World::POSITION_TEXTURE_COLOR_NORMAL: {
                            vk::VertexFormat::PositionTexColorNormal *vertices =
                                static_cast<vk::VertexFormat::PositionTexColorNormal *>(
                                    task.vertices[geometryIndex + i]);

                            vertex.pos = vertices[j].position;

                            vertex.useTexture = 1;
                            vertex.textureUV = vertices[j].uv0;

                            vertex.useColorLayer = 1;
                            vertex.colorLayer = glm::vec4{
                                vertices[j].color & 0xFF,
                                (vertices[j].color >> 8) & 0xFF,
                                (vertices[j].color >> 16) & 0xFF,
                                (vertices[j].color >> 24) & 0xFF,
                            };
                            vertex.colorLayer /= 255.0;

                            vertex.useNorm = 1;
                            vertex.norm = glm::vec3{
                                (int8_t)(vertices[j].normal & 0xFF),
                                (int8_t)((vertices[j].normal >> 8) & 0xFF),
                                (int8_t)((vertices[j].normal >> 16) & 0xFF),
                            };

                            break;
                        }
                    }

                    vertex.textureID = geometryTexture;

                    geometryVertices.push_back(vertex);
                }
            }

            auto orthonormalBasis = [](const glm::dvec3 &a_unit,
                                       const glm::dvec3 &ref) -> std::pair<glm::dvec3, glm::dvec3> {
                const double EPS = 1e-6f;

                glm::dvec3 w = ref;
                if (glm::length(w) < EPS || std::fabs(glm::dot(glm::normalize(w), a_unit)) > 0.99f) {
                    w = (std::fabs(a_unit.z) < 0.99f) ? glm::dvec3(0, 0, 1) : glm::dvec3(0, 1, 0);
                }

                glm::dvec3 u = glm::normalize(glm::cross(a_unit, w));
                if (glm::length(u) < EPS) {
                    w = glm::dvec3(1, 0, 0);
                    u = glm::normalize(glm::cross(a_unit, w));
                }
                glm::dvec3 v = glm::cross(a_unit, u);
                return {u, v};
            };

            auto cubeCornersFromFaceCenters = [orthonormalBasis](const glm::dvec3 &v1, const glm::dvec3 &v2,
                                                                 double d) -> std::pair<bool, std::array<glm::dvec3, 8>> {
                if (!(d > 0.0f)) throw std::invalid_argument("edge length d must be > 0");

                glm::dvec3 axis = v2 - v1;
                double L = glm::length(axis);
                if (!(L > 0.0f)) { return {false, {}}; }

                glm::dvec3 a = glm::normalize(axis);
                auto [u, v] = orthonormalBasis(a, glm::dvec3{0, 1, 0});

                double h = 0.5f * d;

                // v1 面（底）
                glm::dvec3 b00 = v1 - a * 0.5 * d - u * h - v * h;
                glm::dvec3 b10 = v1 - a * 0.5 * d + u * h - v * h;
                glm::dvec3 b11 = v1 - a * 0.5 * d + u * h + v * h;
                glm::dvec3 b01 = v1 - a * 0.5 * d - u * h + v * h;

                // v2 面（顶）
                glm::dvec3 t00 = v2 + a * 0.5 * d - u * h - v * h;
                glm::dvec3 t10 = v2 + a * 0.5 * d + u * h - v * h;
                glm::dvec3 t11 = v2 + a * 0.5 * d + u * h + v * h;
                glm::dvec3 t01 = v2 + a * 0.5 * d - u * h + v * h;

                return {true, {b00, b10, b11, b01, t00, t10, t11, t01}};
            };

            switch (static_cast<World::DrawMode>(task.indexFormats[geometryIndex + i])) {
                case World::DrawMode::QUADS: {
                    for (int j = 0; j < task.vertexCounts[geometryIndex + i]; j += 4) {
                        geometryIndices.push_back(j + 0);
                        geometryIndices.push_back(j + 1);
                        geometryIndices.push_back(j + 2);
                        geometryIndices.push_back(j + 2);
                        geometryIndices.push_back(j + 3);
                        geometryIndices.push_back(j + 0);

                        if (task.normalOffset) {
                            if (geometryVertices[j + 0].useNorm)
                                geometryVertices[j + 0].pos += 0.00001f * glm::normalize(geometryVertices[j + 0].norm);
                            if (geometryVertices[j + 1].useNorm)
                                geometryVertices[j + 1].pos += 0.00001f * glm::normalize(geometryVertices[j + 1].norm);
                            if (geometryVertices[j + 2].useNorm)
                                geometryVertices[j + 2].pos += 0.00001f * glm::normalize(geometryVertices[j + 2].norm);
                            if (geometryVertices[j + 3].useNorm)
                                geometryVertices[j + 3].pos += 0.00001f * glm::normalize(geometryVertices[j + 3].norm);
                        }

                        geometryVertices[j + 0].coordinate = coordinate;
                        geometryVertices[j + 1].coordinate = coordinate;
                        geometryVertices[j + 2].coordinate = coordinate;
                        geometryVertices[j + 3].coordinate = coordinate;

                        if (post) {
                            geometryVertices[j + 0].postBase = {x, y, z};
                            geometryVertices[j + 1].postBase = {x, y, z};
                            geometryVertices[j + 2].postBase = {x, y, z};
                            geometryVertices[j + 3].postBase = {x, y, z};
                        }

                        if (geometryVertices[j + 3].useTexture) {
                            textureIDs.insert(geometryVertices[j + 0].textureID);
                            textureIDs.insert(geometryVertices[j + 1].textureID);
                            textureIDs.insert(geometryVertices[j + 2].textureID);
                            textureIDs.insert(geometryVertices[j + 3].textureID);
                        }
                    }

                    break;
                }
                case World::DrawMode::TRIANGLE_STRIP: {
                    std::vector<vk::VertexFormat::PBRTriangle> fixedVertices;
                    for (int j = 2; j < task.vertexCounts[geometryIndex + i]; j += 2) {
                        auto v0 = geometryVertices[j - 2];
                        auto v1 = geometryVertices[j - 1];
                        auto v2 = geometryVertices[j - 1];
                        auto v3 = geometryVertices[j - 2];

                        v3.pos = geometryVertices[j].pos;
                        v2.pos = geometryVertices[j + 1].pos;
                        fixedVertices.push_back(v0);
                        fixedVertices.push_back(v1);
                        fixedVertices.push_back(v2);
                        fixedVertices.push_back(v3);
                    }
                    geometryVertices = fixedVertices;
                    for (int j = 0; j < geometryVertices.size(); j += 4) {
                        geometryIndices.push_back(j + 0);
                        geometryIndices.push_back(j + 1);
                        geometryIndices.push_back(j + 2);
                        geometryIndices.push_back(j + 2);
                        geometryIndices.push_back(j + 3);
                        geometryIndices.push_back(j + 0);

                        geometryVertices[j + 0].coordinate = coordinate;
                        geometryVertices[j + 1].coordinate = coordinate;
                        geometryVertices[j + 2].coordinate = coordinate;
                        geometryVertices[j + 3].coordinate = coordinate;

                        if (post) {
                            geometryVertices[j + 0].postBase = {x, y, z};
                            geometryVertices[j + 1].postBase = {x, y, z};
                            geometryVertices[j + 2].postBase = {x, y, z};
                            geometryVertices[j + 3].postBase = {x, y, z};
                        }
                    }
                    break;
                }
                case World::DrawMode::LINE_STRIP: {
                    std::vector<vk::VertexFormat::PBRTriangle> fixedVertices;
                    for (int j = 1; j < task.vertexCounts[geometryIndex + i]; j++) {
                        fixedVertices.push_back(geometryVertices[j - 1]);
                        fixedVertices.push_back(geometryVertices[j]);
                    }
                    geometryVertices = fixedVertices;

                    fixedVertices.clear();

                    int accu = 0;
                    for (int j = 1; j < geometryVertices.size(); j += 2) {
                        auto [success, cubePoints] = cubeCornersFromFaceCenters(
                            geometryVertices[j - 1].pos, geometryVertices[j].pos, task.lineWidth);

                        if (!success) { continue; }

                        for (int k = 0; k < 8; k++) {
                            fixedVertices.push_back({
                                .pos = cubePoints[k],
                                .useColorLayer = 1,
                                .colorLayer =
                                    k < 4 ? geometryVertices[j - 1].colorLayer : geometryVertices[j].colorLayer,
                            });
                        }

                        std::vector<uint32_t> indices_ = {{0, 3, 2, 0, 2, 1,
                                                           // top (+a)
                                                           4, 5, 6, 4, 6, 7,
                                                           // -v side
                                                           0, 1, 5, 0, 5, 4,
                                                           // +u side
                                                           1, 2, 6, 1, 6, 5,
                                                           // +v side
                                                           2, 3, 7, 2, 7, 6,
                                                           // -u side
                                                           3, 0, 4, 3, 4, 7}};
                        for (int k = 0; k < 36; k++) { geometryIndices.push_back(accu + indices_[k]); }
                        accu += 8;
                    }

                    geometryVertices = fixedVertices;

                    for (int j = 0; j < geometryVertices.size(); j++) {
                        if (task.normalOffset) {
                            if (geometryVertices[j + 0].useNorm)
                                geometryVertices[j + 0].pos += 0.00001f * glm::normalize(geometryVertices[j + 0].norm);
                        }

                        geometryVertices[j + 0].coordinate = coordinate;

                        if (post) { geometryVertices[j + 0].postBase = {x, y, z}; }
                    }

                    break;
                }
                case World::DrawMode::LINES: {
                    std::vector<vk::VertexFormat::PBRTriangle> fixedVertices;
                    for (int j = 0; j + 3 < task.vertexCounts[geometryIndex + i]; j += 4) {
                        fixedVertices.push_back(geometryVertices[j]);
                        fixedVertices.push_back(geometryVertices[j + 1]);
                        fixedVertices.push_back(geometryVertices[j + 2]);
                        fixedVertices.push_back(geometryVertices[j + 3]);
                        fixedVertices.push_back(geometryVertices[j + 2]);
                        fixedVertices.push_back(geometryVertices[j + 1]);
                    }
                    geometryVertices = fixedVertices;

                    fixedVertices.clear();

                    int accu = 0;
                    for (int j = 1; j < geometryVertices.size(); j += 2) {
                        auto [success, cubePoints] = cubeCornersFromFaceCenters(
                            geometryVertices[j - 1].pos, geometryVertices[j].pos, task.lineWidth);

                        if (!success) { continue; }

                        for (int k = 0; k < 8; k++) {
                            fixedVertices.push_back({
                                .pos = cubePoints[k],
                                .useColorLayer = 1,
                                .colorLayer =
                                    k < 4 ? geometryVertices[j - 1].colorLayer : geometryVertices[j].colorLayer,
                            });
                        }

                        std::vector<uint32_t> indices_ = {{0, 3, 2, 0, 2, 1,
                                                           // top (+a)
                                                           4, 5, 6, 4, 6, 7,
                                                           // -v side
                                                           0, 1, 5, 0, 5, 4,
                                                           // +u side
                                                           1, 2, 6, 1, 6, 5,
                                                           // +v side
                                                           2, 3, 7, 2, 7, 6,
                                                           // -u side
                                                           3, 0, 4, 3, 4, 7}};
                        for (int k = 0; k < 36; k++) { geometryIndices.push_back(accu + indices_[k]); }
                        accu += 8;
                    }

                    geometryVertices = fixedVertices;

                    for (int j = 0; j < geometryVertices.size(); j++) {
                        if (task.normalOffset) {
                            if (geometryVertices[j + 0].useNorm)
                                geometryVertices[j + 0].pos += 0.00001f * glm::normalize(geometryVertices[j + 0].norm);
                        }

                        geometryVertices[j + 0].coordinate = coordinate;

                        if (post) { geometryVertices[j + 0].postBase = {x, y, z}; }
                    }

                    break;
                }
                default: {
                    throw std::runtime_error("Shouldn't be touched");
                }
            }

            if (geometryVertices.empty() || geometryIndices.empty()) {
                vertices.pop_back();
                indices.pop_back();
                geometryTypes.pop_back();
            } else {
                allVertexCount += geometryVertices.size();
                allIndexCount += geometryIndices.size();
                geometryCountWithoutGlint++;
            }
        }

        if (geometryCountWithoutGlint == 0) { continue; }

        std::shared_ptr<EntityBuildData> chunkBuildData =
            EntityBuildData::create(hashCode, x, y, z, rtFlag, prebuiltBLAS, coordinate, geometryCountWithoutGlint,
                                    std::move(geometryTypes), std::move(vertices), std::move(indices));

        if (post) {
            entityPostBuildDataBatch_->addData(chunkBuildData);
        } else {
            entityBuildDataBatch_->addData(chunkBuildData);
        }

        // std::cout << "used texture ids: ";
        // for (auto id : textureIDs) { std::cout << id << " "; }
        // std::cout << std::endl;
    }
}

void Entities::build() {
    Renderer::instance().framework()->safeAcquireCurrentContext();
    auto framework = Renderer::instance().framework();
    auto vma = framework->vma();
    auto device = framework->device();
    auto physicalDevice = framework->physicalDevice();

    entityBuildDataBatch_->build();

    Renderer::instance().buffers()->queueImportantWorldUpload(entityBuildDataBatch_->vertexBuffer,
                                                              entityBuildDataBatch_->indexBuffer);
    blasBatchBuilder_ = entityBuildDataBatch_->blasBatchBuilder;

    entityBatch_ = EntityBatch::create(entityBuildDataBatch_);
    entityPostBatch_ = EntityPostBatch::create(entityPostBuildDataBatch_);

    for (auto entity : entityPostBatch_->entities) {
        for (int i = 0; i < entity->geometryCount; i++) {
            Renderer::instance().buffers()->queueImportantWorldUpload(entity->vertexBuffers[i],
                                                                      entity->indexBuffers[i]);
        }
    }
}

std::shared_ptr<EntityBatch> Entities::entityBatch() {
    Renderer::instance().framework()->safeAcquireCurrentContext();

    if (entityBatch_)
        return entityBatch_;
    else
        return nullptr;
}

std::shared_ptr<EntityPostBatch> Entities::entityPostBatch() {
    Renderer::instance().framework()->safeAcquireCurrentContext();

    if (entityPostBatch_)
        return entityPostBatch_;
    else
        return nullptr;
}

std::shared_ptr<vk::BLASBatchBuilder> Entities::blasBatchBuilder() {
    return blasBatchBuilder_;
}