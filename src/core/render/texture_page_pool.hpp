#pragma once

#include "core/render/gpu_upload_service.hpp"
#include <array>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

/// Tiered texture page pool for texture_loader_v4.
///
/// Manages texture array pages organized by namespace and tier.
/// Pages are allocated once and updated via subrange uploads.
/// No new page image set per small CTM batch.
///
/// Namespaces: Fallback, Vanilla, CTM, Dynamic
/// Tiers: 16, 32, 64, 128, 256, 512, 1024
class TexturePagePool {
public:
    static constexpr uint32_t kMaxTiers = 7;
    static constexpr uint32_t kMaxPagesPerNamespace = 64;

    enum Namespace : uint32_t {
        Fallback = 0,
        Vanilla  = 1,
        Ctm      = 2,
        Dynamic  = 3
    };

    struct PageHandle {
        uint32_t namespaceId = 0;
        uint32_t tier = 0;
        uint32_t page = 0;
        uint32_t layer = 0;
        uint32_t mipCount = 1;
    };

    struct Allocation {
        PageHandle first;
        uint32_t layerCount = 0;
        bool valid = false;
    };

    bool initialize(std::shared_ptr<vk::Device> device, std::shared_ptr<vk::VMA> vma, GpuUploadService* uploads);
    void resetGeneration(uint64_t generation);
    void cancelGeneration(uint64_t generation);

    /// Allocate layers in a namespace/tier. Returns an allocation handle.
    Allocation allocate(uint64_t generation, Namespace ns, uint32_t tier, uint32_t layerCount, bool visible);

    /// Allocate an exact page/layer range in a namespace/tier.
    /// Java provides exact placement; native must allocate that exact range or reject.
    Allocation allocateExact(uint64_t generation, Namespace ns, uint32_t tier,
                            uint32_t page, uint32_t startLayer, uint32_t layerCount,
                            uint32_t layerCapacity, bool visible);

    /// Upload pixel data to an allocated range.
    bool upload(uint64_t generation, const Allocation& allocation,
                const uint8_t* rgba, uint64_t bytes, VkFormat format,
                bool visible, GpuUploadService::Priority priority);

    /// Upload one or more material planes to an allocated range.
    bool upload(uint64_t generation, const Allocation& allocation,
                const uint8_t* albedo, const uint8_t* specular, const uint8_t* normal,
                const uint8_t* flag, uint64_t bytesPerLayer, uint32_t channelMask,
                VkFormat format, bool visible);

    /// Check if a specific page/layer is ready (uploaded + mips generated).
    bool isReady(uint64_t generation, Namespace ns, uint32_t tier, uint32_t page, uint32_t layer) const;

    /// Mark an allocation as ready after upload + mipgen complete.
    void markReady(uint64_t generation, const Allocation& allocation);

    /// Status JSON for DebugBridge.
    std::string statusJson() const;

    /// CTM capacity metrics.
    uint32_t ctmResidentCapacity() const;
    uint32_t ctmPresentMaterials() const;
    bool ctmPagesExhausted() const;
    uint32_t ctmUnaddressableMaterials() const;
    uint32_t unreadyAllocatedPageCount(uint64_t generation) const;

    /// Count allocated layers that are not both uploaded and mip-ready.
    /// If visibleOnly, count only layers where layerVisible is true.
    uint32_t unreadyAllocatedLayerCount(uint64_t generation, bool visibleOnly) const;

    /// Count pages that have at least one allocated layer with layerMipsReady == false.
    /// If visibleOnly, restrict the check to visible layers.
    uint32_t pendingMipPageCount(uint64_t generation, bool visibleOnly) const;

private:
    struct Page {
        uint64_t generation = 0;
        uint32_t namespaceId = 0;
        uint32_t tier = 0;
        uint32_t page = 0;
        uint32_t layerCapacity = 0;
        uint32_t layersUsed = 0;
        uint32_t readyLayers = 0;
        uint32_t mipCount = 1;
        bool allocated = false;
        bool mipsReady = false;
        VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;
        std::shared_ptr<vk::DeviceLocalImage> albedoImage;
        std::shared_ptr<vk::DeviceLocalImage> specularImage;
        std::shared_ptr<vk::DeviceLocalImage> normalImage;
        std::shared_ptr<vk::DeviceLocalImage> flagImage;
        std::shared_ptr<vk::Sampler> sampler;
        uint32_t albedoArrayId = UINT32_MAX;
        uint32_t specularArrayId = UINT32_MAX;
        uint32_t normalArrayId = UINT32_MAX;
        uint32_t flagArrayId = UINT32_MAX;
        // Per-layer tracking
        std::vector<bool> layerAllocated;  // [layerCapacity] true after allocate()
        std::vector<bool> layerVisible;    // [layerCapacity] true if visible at allocate time
        std::vector<bool> layerUploaded;   // [layerCapacity] true after GPU copy complete
        std::vector<bool> layerMipsReady;  // [layerCapacity] true after mipgen complete
    };

    Page& pageForAllocationLocked(uint64_t generation, Namespace ns, uint32_t tier, uint32_t neededLayers);
    Page& pageForExactAllocationLocked(uint64_t generation, Namespace ns, uint32_t tier,
                                        uint32_t page, uint32_t layerCapacity);
    Page* findPageLocked(uint64_t generation, uint32_t namespaceId, uint32_t tier, uint32_t page);
    void markCopyComplete(uint64_t generation, uint32_t namespaceId, uint32_t tier,
                          uint32_t page, uint32_t startLayer, uint32_t layerCount);
    uint32_t ctmResidentCapacityLocked() const;
    uint32_t ctmPresentMaterialsLocked() const;
    bool ctmPagesExhaustedLocked() const;
    uint32_t ctmUnaddressableMaterialsLocked() const;
    uint32_t unreadyAllocatedPageCountLocked(uint64_t generation) const;
    uint32_t unreadyAllocatedLayerCountLocked(uint64_t generation, bool visibleOnly) const;
    uint32_t pendingMipPageCountLocked(uint64_t generation, bool visibleOnly) const;
    uint32_t tierSize(uint32_t tier) const;
    uint32_t pageLayerCapacity(uint32_t tier) const;

    /// Static version of pageLayerCapacity for use outside an instance (e.g. JNI queries).
    static uint32_t pageLayerCapacityStatic(uint32_t tier);

    std::shared_ptr<vk::Device> device_;
    std::shared_ptr<vk::VMA> vma_;
    GpuUploadService* uploads_ = nullptr;

    std::vector<Page> pages_;
    mutable std::mutex mutex_;

    uint64_t activeGeneration_ = 0;
    uint64_t pageImageAllocations_ = 0;
    uint64_t pageSubrangeUploads_ = 0;
};
