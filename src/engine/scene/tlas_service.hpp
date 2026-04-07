#pragma once

// Ownership: EngineServices (1:1).
// Thread: Main thread only.
// Dependencies: DeviceService, BlasService.
//
// Builds the world TLAS from all chunk BLAS instances.
// Full rebuild each frame initially. UPDATE mode can be added later.

#include "scene_types.hpp"
#include "platform/vulkan/vk2_buffer.hpp"
#include "platform/vulkan/vk2_result.hpp"

#include <cstdint>
#include <vulkan/vulkan.h>

namespace engine {

class BlasService;

namespace vk2 { class DeviceService; }

class TlasService {
public:
    TlasService() = default;
    ~TlasService();

    TlasService(const TlasService&) = delete;
    TlasService& operator=(const TlasService&) = delete;

    vk2::Result<void> init(vk2::DeviceService& device);
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
    // Per-instance chunk origin (xyz + pad) SSBO. Used by shaders to offset
    // ray hits back into world space (chunk origin is baked into vertex pos).
    VkBuffer chunkOriginBuffer() const { return chunkOriginBuffer_.handle(); }
    VkDeviceSize bdaArraySize() const { return instanceCount_ * sizeof(uint64_t); }
    VkDeviceSize chunkOriginArraySize() const { return instanceCount_ * sizeof(float) * 4; }

private:
    vk2::DeviceService* device_ = nullptr;

    VkAccelerationStructureKHR tlas_ = VK_NULL_HANDLE;
    vk2::Buffer tlasBuffer_;
    vk2::Buffer instanceBuffer_;   // VkAccelerationStructureInstanceKHR array
    vk2::Buffer scratchBuffer_;

    // Auxiliary SSBOs for the RT shader's vertex fetch path
    vk2::Buffer vertexBdaBuffer_;     // host-visible uint64[]
    vk2::Buffer indexBdaBuffer_;      // host-visible uint64[]
    vk2::Buffer chunkOriginBuffer_;   // host-visible vec4[] (xyz + pad)
    VkDeviceSize bdaCapacity_ = 0;    // current allocated capacity in instances
    VkDeviceSize originCapacity_ = 0;

    VkDeviceAddress tlasAddress_ = 0;
    uint32_t instanceCount_ = 0;
    bool initialized_ = false;

    void destroyTlas();
};

} // namespace engine
