#pragma once

// Ownership: EngineServices (1:1).
// Thread: Main thread only.
// Dependencies: DeviceService, BlasService, optional ResourceGC for deferred-delete.
//
// Builds the world TLAS from all chunk BLAS instances.
// Full rebuild each frame initially. UPDATE mode can be added later.
//
// Each TLAS instance carries a translation transform derived from the section
// origin (originX/Y/Z in BlasData) so the BLAS triangles can stay in the
// section-local space the mesher produces.

#include "scene_types.hpp"
#include "platform/vulkan/vk2_buffer.hpp"
#include "platform/vulkan/vk2_result.hpp"

#include <cstdint>
#include <vulkan/vulkan.h>

namespace engine {

class BlasService;
class ResourceGC;

namespace vk2 { class DeviceService; }

class TlasService {
public:
    TlasService() = default;
    ~TlasService();

    TlasService(const TlasService&) = delete;
    TlasService& operator=(const TlasService&) = delete;

    // gc is optional. When provided, old TLAS destruction on rebuild is deferred
    // into the ResourceGC ring so the GPU finishes any in-flight work first.
    vk2::Result<void> init(vk2::DeviceService& device, ResourceGC* gc = nullptr);
    void shutdown();

    // Rebuild TLAS from all current BLAS instances.
    // Records build commands into cmd. Returns true if TLAS was built.
    bool rebuild(VkCommandBuffer cmd, const BlasService& blasService);

    VkAccelerationStructureKHR handle() const { return tlas_; }
    VkDeviceAddress address() const { return tlasAddress_; }
    uint32_t instanceCount() const { return instanceCount_; }
    bool isValid() const { return tlas_ != VK_NULL_HANDLE && instanceCount_ > 0; }
    bool isInitialized() const { return initialized_; }

    // Per-instance vertex/index BDA SSBOs (built alongside TLAS).
    // Indexed by gl_InstanceCustomIndexEXT in shaders.
    VkBuffer vertexBdaBuffer() const { return vertexBdaBuffer_.handle(); }
    VkBuffer indexBdaBuffer() const { return indexBdaBuffer_.handle(); }
    VkDeviceSize bdaArraySize() const { return instanceCount_ * sizeof(uint64_t); }

private:
    vk2::DeviceService* device_ = nullptr;
    ResourceGC* gc_ = nullptr;

    VkAccelerationStructureKHR tlas_ = VK_NULL_HANDLE;
    vk2::Buffer tlasBuffer_;
    vk2::Buffer instanceBuffer_;   // VkAccelerationStructureInstanceKHR array
    vk2::Buffer scratchBuffer_;

    // Auxiliary SSBOs for the RT shader's vertex fetch path
    vk2::Buffer vertexBdaBuffer_;     // host-visible uint64[]
    vk2::Buffer indexBdaBuffer_;      // host-visible uint64[]
    VkDeviceSize bdaCapacity_ = 0;    // current allocated capacity in instances

    VkDeviceAddress tlasAddress_ = 0;
    uint32_t instanceCount_ = 0;
    bool initialized_ = false;

    // Destroy (or defer destruction of) the old TLAS handle + buffer on rebuild.
    void retireTlas(VkAccelerationStructureKHR handle, vk2::Buffer buffer);
};

} // namespace engine
