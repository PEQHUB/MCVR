#pragma once

#include "core/all_extern.hpp"

#include "core/vulkan/buffer.hpp"
#include "core/vulkan/vertex.hpp"

#include <vector>

namespace vk {
class DeviceLocalBuffer;
class HostVisibleBuffer;
class CommandBuffer;
class PhysicalDevice;
class Device;
class VMA;

class BLAS : public SharedObject<BLAS> {
  public:
    BLAS(std::shared_ptr<Device> device,
         VkAccelerationStructureKHR blas,
         std::shared_ptr<DeviceLocalBuffer> blasBuffer);
    ~BLAS();

    std::shared_ptr<DeviceLocalBuffer> blasBuffer();
    VkAccelerationStructureKHR &blas();
    VkDeviceAddress &blasDeviceAddress();

  private:
    std::shared_ptr<Device> device_;
    std::shared_ptr<DeviceLocalBuffer> blasBuffer_;

    VkAccelerationStructureKHR blas_;
    VkDeviceAddress blasDeviceAddress_;
};

class TLAS : public SharedObject<TLAS> {
  public:
    TLAS(std::shared_ptr<Device> device,
         VkAccelerationStructureKHR tlas,
         std::shared_ptr<DeviceLocalBuffer> tlasBuffer);
    ~TLAS();

    std::shared_ptr<DeviceLocalBuffer> tlasBuffer();
    VkAccelerationStructureKHR &tlas();
    VkDeviceAddress &tlasDeviceAddress();

  private:
    std::shared_ptr<Device> device_;
    std::shared_ptr<DeviceLocalBuffer> tlasBuffer_;

    VkAccelerationStructureKHR tlas_;
    VkDeviceAddress tlasDeviceAddress_;
};

class BLASBatchBuilder;

class BLASBuilder : public SharedObject<BLASBuilder> {
    friend BLASBatchBuilder;

  public:
    struct BLASGeometryBuilder {
        BLASBuilder &parent;

        std::vector<VkAccelerationStructureGeometryKHR> geometries;
        std::vector<uint32_t> primitiveCounts;

        BLASGeometryBuilder(BLASBuilder &parent);

        template <typename T>
        BLASGeometryBuilder &defineTriangleGeomrtry(std::shared_ptr<DeviceLocalBuffer> vertexBuffer,
                                                    uint32_t numVertices,
                                                    std::shared_ptr<DeviceLocalBuffer> indexBuffer,
                                                    uint32_t numIndices,
                                                    bool isOpaque);
        template <typename T>
        BLASGeometryBuilder &defineTriangleGeomrtry(VkDeviceAddress vertexBufferAddress,
                                                    uint32_t numVertices,
                                                    VkDeviceAddress indexBufferAddress,
                                                    uint32_t numIndices,
                                                    bool isOpaque);

        BLASGeometryBuilder &definePlaceholderGeometry();

        std::shared_ptr<BLASBuilder> endGeometries();
    };

  public:
    BLASBuilder();

    std::shared_ptr<BLASGeometryBuilder> beginGeometries();
    std::shared_ptr<BLASBuilder> defineBuildProperty(VkBuildAccelerationStructureFlagsKHR flags);
    std::shared_ptr<BLASBuilder> defineUpdateProperty(VkBuildAccelerationStructureFlagsKHR flags,
                                                      VkAccelerationStructureKHR srcBLAS);
    std::shared_ptr<BLASBuilder> querySizeInfo(std::shared_ptr<Device> device);
    std::shared_ptr<BLASBuilder> allocateBuffers(std::shared_ptr<PhysicalDevice> physicalDevice,
                                                 std::shared_ptr<Device> device,
                                                 std::shared_ptr<VMA> vma);
    std::shared_ptr<BLAS> buildAndSubmit(std::shared_ptr<Device> device, std::shared_ptr<CommandBuffer> commandBuffer);
    std::shared_ptr<BLAS> build(std::shared_ptr<Device> device);
    std::shared_ptr<BLAS>
    buildExternal(std::shared_ptr<Device> device, std::shared_ptr<DeviceLocalBuffer> buffer, VkDeviceSize offset);
    void submit(std::shared_ptr<CommandBuffer> commandBuffer);
    void submitExternal(std::shared_ptr<CommandBuffer> commandBuffer, VkDeviceAddress scratchBufferAddress);
    static void batchSubmit(std::vector<std::shared_ptr<BLASBuilder>> &builders,
                            std::shared_ptr<CommandBuffer> commandBuffer);
    static void batchSubmitExternal(std::vector<std::shared_ptr<BLASBuilder>> &builders,
                                    std::vector<VkDeviceAddress> scratchBufferAddress,
                                    std::shared_ptr<CommandBuffer> commandBuffer);

  private:
    BLASGeometryBuilder geometryBuilder_;

    VkBuildAccelerationStructureModeKHR mode_;
    VkBuildAccelerationStructureFlagsKHR flags_;

    VkAccelerationStructureBuildSizesInfoKHR sizeInfo_{};

    std::shared_ptr<DeviceLocalBuffer> blasBuffer_;
    std::shared_ptr<DeviceLocalBuffer> scratchBuffer_;

    VkAccelerationStructureKHR srcBLAS_ = VK_NULL_HANDLE;
    VkAccelerationStructureKHR dstBLAS_ = VK_NULL_HANDLE;
};

class BLASBatchBuilder : public SharedObject<BLASBatchBuilder> {
  public:
    std::shared_ptr<BLASBuilder> defineBLASBuilder();
    std::shared_ptr<BLASBatchBuilder> allocateBuffers(std::shared_ptr<PhysicalDevice> physicalDevice,
                                                      std::shared_ptr<Device> device,
                                                      std::shared_ptr<VMA> vma);
    std::vector<std::shared_ptr<BLAS>> build(std::shared_ptr<Device> device);
    void submit(std::shared_ptr<CommandBuffer> commandBuffer);

  private:
    std::vector<std::shared_ptr<BLASBuilder>> builders_;

    std::shared_ptr<DeviceLocalBuffer> blasBuffer_;
    std::shared_ptr<DeviceLocalBuffer> scratchBuffer_;

    std::vector<VkDeviceSize> blasOffsets_;
    std::vector<VkDeviceSize> scratchOffsets_;
    std::vector<VkDeviceSize> scratchAddresses_;
    VkDeviceSize totalBlasSize_ = 0;
    VkDeviceSize totalScratchSize_ = 0;
};

class TLASBuilder : public SharedObject<TLASBuilder> {
  public:
    struct TLASInstanceBuilder {
        TLASBuilder &parent;

        // std::vector<VkTransformMatrixKHR> transforms;
        // std::vector<uint32_t> customIndices;
        // std::vector<uint32_t> masks;
        // std::vector<uint32_t> offsets;
        // std::vector<VkGeometryInstanceFlagsKHR> flags;
        // std::vector<std::shared_ptr<BLAS>> blass;
        std::vector<std::tuple<VkTransformMatrixKHR,
                               uint32_t,
                               uint32_t,
                               uint32_t,
                               VkGeometryInstanceFlagsKHR,
                               std::shared_ptr<BLAS>>>
            instances;

        std::shared_ptr<HostVisibleBuffer> instanceBuffer;
        std::vector<VkAccelerationStructureGeometryKHR> geometries;

        TLASInstanceBuilder(TLASBuilder &parent);

        TLASInstanceBuilder &defineInstance(VkTransformMatrixKHR transform,
                                            uint32_t customIndex,
                                            uint32_t mask,
                                            uint32_t offset,
                                            VkGeometryInstanceFlagsKHR flag,
                                            std::shared_ptr<BLAS> blas);
        std::shared_ptr<TLASBuilder> endInstanceBuilder(std::shared_ptr<Device> device, std::shared_ptr<VMA> vma);
    };

  public:
    TLASBuilder();

    TLASInstanceBuilder &beginInstanceBuilder();
    std::shared_ptr<TLASBuilder> defineBuildProperty(VkBuildAccelerationStructureFlagsKHR flags);
    std::shared_ptr<TLASBuilder> defineUpdateProperty(VkBuildAccelerationStructureFlagsKHR flags,
                                                      VkAccelerationStructureKHR srcTLAS);
    std::shared_ptr<TLASBuilder> querySizeInfo(std::shared_ptr<Device> device);
    std::shared_ptr<TLASBuilder> allocateBuffers(std::shared_ptr<PhysicalDevice> physicalDevice,
                                                 std::shared_ptr<Device> device,
                                                 std::shared_ptr<VMA> vma);
    std::shared_ptr<TLAS> buildAndSubmit(std::shared_ptr<Device> device, std::shared_ptr<CommandBuffer> commandBuffer);

  private:
    TLASInstanceBuilder tlasInstanceBuilder_;

    VkBuildAccelerationStructureModeKHR mode_;
    VkBuildAccelerationStructureFlagsKHR flags_;

    VkAccelerationStructureBuildSizesInfoKHR sizeInfo_{};

    std::shared_ptr<DeviceLocalBuffer> tlasBuffer_;
    std::shared_ptr<DeviceLocalBuffer> scratchBuffer_;

    VkAccelerationStructureKHR srcTLAS_ = VK_NULL_HANDLE;
    VkAccelerationStructureKHR dstTLAS_ = VK_NULL_HANDLE;
};
}; // namespace vk

template <typename T>
vk::BLASBuilder::BLASGeometryBuilder &
vk::BLASBuilder::BLASGeometryBuilder::defineTriangleGeomrtry(std::shared_ptr<DeviceLocalBuffer> vertexBuffer,
                                                             uint32_t numVertices,
                                                             std::shared_ptr<DeviceLocalBuffer> indexBuffer,
                                                             uint32_t numIndices,
                                                             bool isOpaque) {
    return defineTriangleGeomrtry<T>(vertexBuffer->bufferAddress(), numVertices, indexBuffer->bufferAddress(),
                                     numIndices, isOpaque);
}

template <typename T>
vk::BLASBuilder::BLASGeometryBuilder &
vk::BLASBuilder::BLASGeometryBuilder::defineTriangleGeomrtry(VkDeviceAddress vertexBufferAddress,
                                                             uint32_t numVertices,
                                                             VkDeviceAddress indexBufferAddress,
                                                             uint32_t numIndices,
                                                             bool isOpaque) {
    VkAccelerationStructureGeometryKHR geom{};
    geom.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geom.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    geom.flags = isOpaque ? VK_GEOMETRY_OPAQUE_BIT_KHR : VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR;

    // 三角形数据设置
    auto &triangles = geom.geometry.triangles;
    triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT; // 顶点pos格式，强制在第一位
    triangles.vertexData.deviceAddress = vertexBufferAddress;
    triangles.vertexStride = sizeof(T);
    triangles.maxVertex = numVertices - 1;
    triangles.indexType = VK_INDEX_TYPE_UINT32;
    triangles.indexData.deviceAddress = indexBufferAddress;

    geometries.push_back(geom);
    primitiveCounts.push_back(numIndices / 3);

    return *this;
}