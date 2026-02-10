#pragma once

#include "common/shared.hpp"
#include "common/singleton.hpp"
#include "core/all_extern.hpp"
#include "core/vulkan/all_core_vulkan.hpp"

#include "core/render/world.hpp"

#include <chrono>
#include <condition_variable>
#include <deque>
#include <list>
#include <map>
#include <mutex>
#include <queue>
#include <set>
#include <unordered_map>
#include <vector>

class Framework;

struct EntitiesBuildTask {
    float lineWidth;
    World::Coordinates coordinate;
    bool normalOffset;
    int entityCount;
    int *entityHashCodes;
    double *entityXs;
    double *entityYs;
    double *entityZs;
    int *entityRTFlags;
    int *entityPrebuiltBLASs;
    int *entityPosts;
    int *entityGeometryCounts;
    int *geometryTypes;
    int *geometryTextures;
    int *vertexFormats;
    int *indexFormats;
    int *vertexCounts;
    void **vertices;
};

struct EntityBuildData : public SharedObject<EntityBuildData> {
    int hashCode;
    double x, y, z;
    int rtFlag;
    int prebuiltBLAS;
    World::Coordinates coordinate;
    uint32_t geometryCount;
    std::vector<World::GeometryTypes> geometryTypes;
    std::vector<std::vector<vk::VertexFormat::PBRTriangle>> vertices;
    std::vector<std::vector<uint32_t>> indices;
    std::vector<VkDeviceAddress> vertexBufferAddresses;
    std::vector<VkDeviceAddress> indexBufferAddresses;
    std::shared_ptr<vk::BLAS> blas;

    EntityBuildData(int hashCode,
                    double x,
                    double y,
                    double z,
                    int rtFlag,
                    int prebuiltBLAS,
                    World::Coordinates coordinate,
                    uint32_t geometryCount,
                    std::vector<World::GeometryTypes> &&geometryTypes,
                    std::vector<std::vector<vk::VertexFormat::PBRTriangle>> &&vertices,
                    std::vector<std::vector<uint32_t>> &&indices);
};

struct EntityBuildDataBatch : public SharedObject<EntityBuildDataBatch> {
    std::vector<std::shared_ptr<EntityBuildData>> datas;

    std::shared_ptr<vk::DeviceLocalBuffer> vertexBuffer;
    std::shared_ptr<vk::DeviceLocalBuffer> indexBuffer;
    std::shared_ptr<vk::BLASBatchBuilder> blasBatchBuilder;

    void addData(std::shared_ptr<EntityBuildData> data);
    void build();
};

struct EntityPostBuildDataBatch : public SharedObject<EntityPostBuildDataBatch> {
    std::vector<std::shared_ptr<EntityBuildData>> datas;

    void addData(std::shared_ptr<EntityBuildData> data);
};

struct Entity;
struct EntityBatch;
struct EntityPost;
struct EntityPostBatch;

struct Entity : public SharedObject<Entity> {
    int hashCode;
    double x, y, z;
    int rtFlag;
    int prebuiltBLAS;
    World::Coordinates coordinate;

    std::shared_ptr<vk::BLAS> blas;
    std::shared_ptr<std::vector<VkDeviceAddress>> vertexBufferAddresses;
    std::shared_ptr<std::vector<VkDeviceAddress>> indexBufferAddresses;
    std::shared_ptr<vk::DeviceLocalBuffer> vertexBuffer;
    std::shared_ptr<vk::DeviceLocalBuffer> indexBuffer;

    uint32_t geometryCount;
    std::shared_ptr<std::vector<World::GeometryTypes>> geometryTypes;
    std::shared_ptr<std::vector<std::vector<vk::VertexFormat::PBRTriangle>>> vertices;
    std::shared_ptr<std::vector<std::vector<uint32_t>>> indices;

    Entity(std::shared_ptr<EntityBuildData> entityBuildData);
};

struct EntityBatch : public SharedObject<EntityBatch> {
    std::vector<std::shared_ptr<Entity>> entities;

    std::shared_ptr<vk::DeviceLocalBuffer> vertexBuffer;
    std::shared_ptr<vk::DeviceLocalBuffer> indexBuffer;

    EntityBatch(std::shared_ptr<EntityBuildDataBatch> entityBuildDataBatch);
};

struct EntityPost : public SharedObject<EntityPost> {
    double x, y, z;

    uint32_t geometryCount;
    std::vector<std::vector<vk::VertexFormat::PBRTriangle>> vertices;
    std::vector<std::vector<uint32_t>> indices;

    std::vector<std::shared_ptr<vk::DeviceLocalBuffer>> vertexBuffers;
    std::vector<std::shared_ptr<vk::DeviceLocalBuffer>> indexBuffers;

    EntityPost(std::shared_ptr<EntityBuildData> entityBuildData);
};

struct EntityPostBatch : public SharedObject<EntityPostBatch> {
    std::vector<std::shared_ptr<EntityPost>> entities;

    EntityPostBatch(std::shared_ptr<EntityPostBuildDataBatch> entityPostBuildDataBatch);
};

class Entities : public SharedObject<Entities> {
    friend World;

  public:
    Entities(std::shared_ptr<Framework> framework);

    void resetFrame();
    void queueBuild(EntitiesBuildTask task);
    void build();
    std::shared_ptr<EntityBatch> entityBatch();
    std::shared_ptr<EntityPostBatch> entityPostBatch();
    std::shared_ptr<vk::BLASBatchBuilder> blasBatchBuilder();

  private:
    std::shared_ptr<EntityBatch> entityBatch_;
    std::shared_ptr<EntityPostBatch> entityPostBatch_;
    std::shared_ptr<EntityBuildDataBatch> entityBuildDataBatch_;
    std::shared_ptr<EntityPostBuildDataBatch> entityPostBuildDataBatch_;

    std::shared_ptr<vk::BLASBatchBuilder> blasBatchBuilder_;
};