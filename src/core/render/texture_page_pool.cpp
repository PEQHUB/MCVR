#include "core/render/texture_page_pool.hpp"
#include "core/render/renderer.hpp"
#include "core/render/render_framework.hpp"
#include <algorithm>
#include <iostream>
#include <sstream>

static const uint32_t TIER_SIZES[] = {16, 32, 64, 128, 256, 512, 1024};
static const uint32_t DEFAULT_LAYERS_PER_PAGE = 256;
static constexpr uint32_t CHANNEL_ALBEDO   = 1u << 0;
static constexpr uint32_t CHANNEL_SPECULAR = 1u << 1;
static constexpr uint32_t CHANNEL_NORMAL   = 1u << 2;
static constexpr uint32_t CHANNEL_FLAG     = 1u << 3;

uint32_t TexturePagePool::tierSize(uint32_t tier) const {
    return tier < kMaxTiers ? TIER_SIZES[tier] : 0;
}

uint32_t TexturePagePool::pageLayerCapacity(uint32_t tier) const {
    return pageLayerCapacityStatic(tier);
}

uint32_t TexturePagePool::pageLayerCapacityStatic(uint32_t tier) {
    // Larger tiers get fewer layers per page to stay within memory budgets
    if (tier >= kMaxTiers) return 0;
    uint32_t size = TIER_SIZES[tier];
    if (size <= 16) return 2048;
    if (size <= 32) return 1024;
    if (size <= 64) return 512;
    if (size <= 128) return 256;
    if (size <= 256) return 128;
    if (size <= 512) return 64;
    return 32; // 1024
}

bool TexturePagePool::initialize(std::shared_ptr<vk::Device> device,
    std::shared_ptr<vk::VMA> vma, GpuUploadService* uploads) {
    std::lock_guard<std::mutex> lock(mutex_);
    device_ = std::move(device);
    vma_ = std::move(vma);
    uploads_ = uploads;
    pages_.clear();
    activeGeneration_ = 0;
    pageImageAllocations_ = 0;
    pageSubrangeUploads_ = 0;
    return device_ && vma_ && uploads_;
}

void TexturePagePool::resetGeneration(uint64_t generation) {
    std::lock_guard<std::mutex> lock(mutex_);
    activeGeneration_ = generation;
    // Clear all pages so every newly allocated image can use VK_IMAGE_LAYOUT_UNDEFINED.
    // Do not reuse page images across generations unless layout tracking is implemented.
    pages_.clear();
}

void TexturePagePool::cancelGeneration(uint64_t generation) {
    std::lock_guard<std::mutex> lock(mutex_);
    // Remove all pages belonging to the cancelled generation
    pages_.erase(
        std::remove_if(pages_.begin(), pages_.end(),
            [generation](const Page& p) { return p.generation == generation; }),
        pages_.end());
}

TexturePagePool::Allocation TexturePagePool::allocate(
    uint64_t generation, Namespace ns, uint32_t tier, uint32_t layerCount, bool visible) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (generation == 0) return {PageHandle{}, 0, false};
    if (generation != activeGeneration_) return {PageHandle{}, 0, false};
    if (static_cast<uint32_t>(ns) > 3) return {PageHandle{}, 0, false};
    if (tier >= kMaxTiers) return {PageHandle{}, 0, false};
    if (layerCount == 0) return {PageHandle{}, 0, false};
    if (!device_ || !vma_ || !uploads_) return {PageHandle{}, 0, false};

    Page& page = pageForAllocationLocked(generation, ns, tier, layerCount);
    if (!page.allocated) {
        return {PageHandle{}, 0, false};
    }
    if (!page.albedoImage || page.albedoImage->vkImage() == VK_NULL_HANDLE
        || !page.specularImage || page.specularImage->vkImage() == VK_NULL_HANDLE
        || !page.normalImage || page.normalImage->vkImage() == VK_NULL_HANDLE
        || !page.flagImage || page.flagImage->vkImage() == VK_NULL_HANDLE) {
        return {PageHandle{}, 0, false};
    }
    uint32_t startLayer = page.layersUsed;
    if (startLayer + layerCount > page.layerCapacity) {
        return {PageHandle{}, 0, false};
    }
    // Mark per-layer state
    for (uint32_t l = startLayer; l < startLayer + layerCount; ++l) {
        if (l < page.layerAllocated.size()) page.layerAllocated[l] = true;
        if (l < page.layerVisible.size()) page.layerVisible[l] = visible;
        if (l < page.layerUploaded.size()) page.layerUploaded[l] = false;
        if (l < page.layerMipsReady.size()) page.layerMipsReady[l] = false;
    }
    page.layersUsed += layerCount;
    PageHandle handle;
    handle.namespaceId = static_cast<uint32_t>(ns);
    handle.tier = tier;
    handle.page = page.page;
    handle.layer = startLayer;
    handle.mipCount = 1; // Will be updated after upload
    return {handle, layerCount, true};
}

TexturePagePool::Allocation TexturePagePool::allocateExact(
    uint64_t generation, Namespace ns, uint32_t tier,
    uint32_t page, uint32_t startLayer, uint32_t layerCount,
    uint32_t layerCapacity, bool visible) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (generation == 0) return {PageHandle{}, 0, false};
    if (generation != activeGeneration_) return {PageHandle{}, 0, false};
    if (static_cast<uint32_t>(ns) > 3) return {PageHandle{}, 0, false};
    if (tier >= kMaxTiers) return {PageHandle{}, 0, false};
    if (layerCount == 0) return {PageHandle{}, 0, false};
    if (layerCapacity == 0) return {PageHandle{}, 0, false};
    if (startLayer >= layerCapacity || layerCount > layerCapacity - startLayer) {
        return {PageHandle{}, 0, false};
    }
    if (!device_ || !vma_ || !uploads_) return {PageHandle{}, 0, false};

    Page& p = pageForExactAllocationLocked(generation, ns, tier, page, layerCapacity);
    if (!p.allocated) {
        return {PageHandle{}, 0, false};
    }
    if (!p.albedoImage || p.albedoImage->vkImage() == VK_NULL_HANDLE
        || !p.specularImage || p.specularImage->vkImage() == VK_NULL_HANDLE
        || !p.normalImage || p.normalImage->vkImage() == VK_NULL_HANDLE
        || !p.flagImage || p.flagImage->vkImage() == VK_NULL_HANDLE) {
        return {PageHandle{}, 0, false};
    }
    if (startLayer >= p.layerCapacity || layerCount > p.layerCapacity - startLayer) {
        return {PageHandle{}, 0, false};
    }

    for (uint32_t l = startLayer; l < startLayer + layerCount; ++l) {
        if (l >= p.layerAllocated.size() || p.layerAllocated[l]) {
            std::cout << "[TexturePagePool] allocateExact REJECTED: overlap"
                      << " ns=" << static_cast<uint32_t>(ns)
                      << " tier=" << tier
                      << " page=" << page
                      << " layer=" << l << std::endl;
            return {PageHandle{}, 0, false};
        }
    }

    // Mark per-layer state only after the full range is known to be free.
    for (uint32_t l = startLayer; l < startLayer + layerCount; ++l) {
        if (l < p.layerAllocated.size()) p.layerAllocated[l] = true;
        if (l < p.layerVisible.size()) p.layerVisible[l] = visible;
        if (l < p.layerUploaded.size()) p.layerUploaded[l] = false;
        if (l < p.layerMipsReady.size()) p.layerMipsReady[l] = false;
    }
    p.layersUsed = std::max(p.layersUsed, startLayer + layerCount);
    PageHandle handle;
    handle.namespaceId = static_cast<uint32_t>(ns);
    handle.tier = tier;
    handle.page = p.page;
    handle.layer = startLayer;
    handle.mipCount = 1;
    return {handle, layerCount, true};
}

bool TexturePagePool::upload(uint64_t generation, const Allocation& allocation,
    const uint8_t* rgba, uint64_t bytes, VkFormat format,
    bool visible, GpuUploadService::Priority priority) {
    // Capture what we need under the lock, then release before calling GpuUploadService.
    std::shared_ptr<vk::DeviceLocalImage> image;
    uint32_t size = 0;
    uint32_t namespaceId = 0;
    uint32_t tier = 0;
    uint32_t pageId = 0;
    uint32_t startLayer = 0;
    uint32_t layerCount = 0;
    bool layersValid = true;

    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!allocation.valid || !rgba || bytes == 0) return false;
        if (generation != activeGeneration_) return false;
        if (!uploads_) return false;
        if (format != VK_FORMAT_R8G8B8A8_UNORM) return false;

        Page* page = findPageLocked(generation, allocation.first.namespaceId, allocation.first.tier, allocation.first.page);
        if (!page || !page->allocated || !page->albedoImage) return false;
        if (page->albedoImage->vkImage() == VK_NULL_HANDLE) return false;

        if (allocation.first.layer >= page->layerCapacity ||
            allocation.layerCount == 0 ||
            allocation.first.layer + allocation.layerCount > page->layerCapacity) {
            return false;
        }

        // Validate all layers in the range are allocated
        for (uint32_t l = allocation.first.layer; l < allocation.first.layer + allocation.layerCount; ++l) {
            if (l >= page->layerAllocated.size() || !page->layerAllocated[l]) {
                layersValid = false;
                break;
            }
        }
        if (!layersValid) return false;

        size = tierSize(allocation.first.tier);
        if (size == 0) return false;

        // Overflow-safe byte count validation
        const uint64_t bytesPerLayer = uint64_t(size) * uint64_t(size) * 4ull;
        if (bytesPerLayer == 0 || bytesPerLayer > UINT32_MAX) return false;
        if (allocation.layerCount > UINT64_MAX / bytesPerLayer) return false;
        const uint64_t requiredBytes = bytesPerLayer * allocation.layerCount;
        if (bytes != requiredBytes) return false;

        // Mark layers as not-yet-uploaded
        for (uint32_t l = allocation.first.layer; l < allocation.first.layer + allocation.layerCount; ++l) {
            if (l < page->layerUploaded.size()) page->layerUploaded[l] = false;
            if (l < page->layerMipsReady.size()) page->layerMipsReady[l] = false;
        }

        // Capture for use outside the lock
        image = page->albedoImage;
        namespaceId = allocation.first.namespaceId;
        tier = allocation.first.tier;
        pageId = allocation.first.page;
        startLayer = allocation.first.layer;
        layerCount = allocation.layerCount;
    } // mutex_ released

    // Build upload request outside the page-pool lock to avoid lock inversion.
    // The upload service completion path can call back into markCopyComplete/markReady.
    GpuUploadService::TextureUpload upload{};
    upload.generation = generation;
    upload.namespaceId = namespaceId;
    upload.tier = tier;
    upload.page = pageId;
    upload.layer = startLayer;
    upload.layerCount = layerCount;
    upload.width = size;
    upload.height = size;
    upload.bytesPerLayer = static_cast<uint32_t>(uint64_t(size) * uint64_t(size) * 4ull);
    upload.format = format;
    upload.dstImage = image->vkImage();
    upload.mipLevels = 1;
    upload.data = rgba;
    upload.bytes = uint64_t(size) * uint64_t(size) * 4ull * layerCount;
    upload.visible = visible;
    upload.priority = priority;
    upload.onComplete = [this, generation, namespaceId, tier, pageId, startLayer, layerCount, image](uint64_t) {
        (void)image; // Keep image alive until upload completes
        markCopyComplete(generation, namespaceId, tier, pageId, startLayer, layerCount);
    };

    if (!uploads_->enqueueTextureUpload(upload)) {
        return false;
    }
    pageSubrangeUploads_++;
    return true;
}

bool TexturePagePool::upload(uint64_t generation, const Allocation& allocation,
    const uint8_t* albedo, const uint8_t* specular, const uint8_t* normal,
    const uint8_t* flag, uint64_t bytesPerLayer, uint32_t channelMask,
    VkFormat format, bool visible) {
    // Capture what we need under the lock, then release before calling GpuUploadService.
    std::shared_ptr<vk::DeviceLocalImage> albedoImage;
    std::shared_ptr<vk::DeviceLocalImage> specularImage;
    std::shared_ptr<vk::DeviceLocalImage> normalImage;
    std::shared_ptr<vk::DeviceLocalImage> flagImage;
    uint32_t size = 0;
    uint32_t namespaceId = 0;
    uint32_t tier = 0;
    uint32_t pageId = 0;
    uint32_t startLayer = 0;
    uint32_t layerCount = 0;
    uint32_t mipCount = 1;
    bool layersValid = true;

    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!allocation.valid || !albedo || bytesPerLayer == 0) return false;
        if (generation != activeGeneration_) return false;
        if (!uploads_) return false;
        if (format != VK_FORMAT_R8G8B8A8_UNORM) return false;

        Page* page = findPageLocked(generation, allocation.first.namespaceId, allocation.first.tier, allocation.first.page);
        if (!page || !page->allocated || !page->albedoImage) return false;
        if (page->albedoImage->vkImage() == VK_NULL_HANDLE) return false;

        if (allocation.first.layer >= page->layerCapacity ||
            allocation.layerCount == 0 ||
            allocation.first.layer + allocation.layerCount > page->layerCapacity) {
            return false;
        }

        // Validate all layers in the range are allocated
        for (uint32_t l = allocation.first.layer; l < allocation.first.layer + allocation.layerCount; ++l) {
            if (l >= page->layerAllocated.size() || !page->layerAllocated[l]) {
                layersValid = false;
                break;
            }
        }
        if (!layersValid) return false;

        size = tierSize(allocation.first.tier);
        if (size == 0) return false;

        // Overflow-safe byte count validation
        const uint64_t requiredBytesPerLayer = uint64_t(size) * uint64_t(size) * 4ull;
        if (requiredBytesPerLayer == 0 || requiredBytesPerLayer > UINT32_MAX) return false;
        if (bytesPerLayer != requiredBytesPerLayer) return false;
        if (allocation.layerCount > UINT64_MAX / bytesPerLayer) return false;

        // Prevalidate channelMask: every set bit must have a non-null image
        if ((channelMask & CHANNEL_SPECULAR) && (!page->specularImage || page->specularImage->vkImage() == VK_NULL_HANDLE)) {
            return false;
        }
        if ((channelMask & CHANNEL_NORMAL) && (!page->normalImage || page->normalImage->vkImage() == VK_NULL_HANDLE)) {
            return false;
        }
        if ((channelMask & CHANNEL_FLAG) && (!page->flagImage || page->flagImage->vkImage() == VK_NULL_HANDLE)) {
            return false;
        }

        // Mark layers as not-yet-uploaded
        for (uint32_t l = allocation.first.layer; l < allocation.first.layer + allocation.layerCount; ++l) {
            if (l < page->layerUploaded.size()) page->layerUploaded[l] = false;
            if (l < page->layerMipsReady.size()) page->layerMipsReady[l] = false;
        }

        // Capture for use outside the lock
        albedoImage = page->albedoImage;
        specularImage = page->specularImage;
        normalImage = page->normalImage;
        flagImage = page->flagImage;
        namespaceId = allocation.first.namespaceId;
        tier = allocation.first.tier;
        pageId = allocation.first.page;
        startLayer = allocation.first.layer;
        layerCount = allocation.layerCount;
        mipCount = page->mipCount;
    } // mutex_ released

    const uint64_t requiredBytes = bytesPerLayer * layerCount;

    struct Plane {
        const uint8_t* data;
        std::shared_ptr<vk::DeviceLocalImage> image;
        GpuUploadService::Priority priority;
    };

    std::vector<Plane> planes;
    planes.push_back({albedo, albedoImage,
        visible ? GpuUploadService::Priority::FirstFrameAlbedo : GpuUploadService::Priority::BackgroundCtm});

    if ((channelMask & CHANNEL_SPECULAR) && specular && specularImage) {
        planes.push_back({specular, specularImage,
            visible ? GpuUploadService::Priority::FirstFrameAux : GpuUploadService::Priority::BackgroundCtm});
    }
    if ((channelMask & CHANNEL_NORMAL) && normal && normalImage) {
        planes.push_back({normal, normalImage,
            visible ? GpuUploadService::Priority::FirstFrameAux : GpuUploadService::Priority::BackgroundCtm});
    }
    if ((channelMask & CHANNEL_FLAG) && flag && flagImage) {
        planes.push_back({flag, flagImage,
            visible ? GpuUploadService::Priority::FirstFrameAux : GpuUploadService::Priority::BackgroundCtm});
    }

    auto remaining = std::make_shared<std::atomic<uint32_t>>(static_cast<uint32_t>(planes.size()));

    for (const auto& plane : planes) {
        if (!plane.data || !plane.image) return false;
        if (plane.image->vkImage() == VK_NULL_HANDLE) return false;

        GpuUploadService::TextureUpload upload{};
        upload.generation = generation;
        upload.namespaceId = namespaceId;
        upload.tier = tier;
        upload.page = pageId;
        upload.layer = startLayer;
        upload.layerCount = layerCount;
        upload.width = size;
        upload.height = size;
        upload.bytesPerLayer = static_cast<uint32_t>(bytesPerLayer);
        upload.format = format;
        upload.dstImage = plane.image->vkImage();
        upload.mipLevels = mipCount;
        upload.data = plane.data;
        upload.bytes = requiredBytes;
        upload.visible = visible;
        upload.priority = plane.priority;
        upload.onComplete = [this, generation, namespaceId, tier, pageId,
                            startLayer, layerCount, remaining, image = plane.image](uint64_t) {
            (void)image; // Keep image alive until upload completes
            if (remaining->fetch_sub(1, std::memory_order_acq_rel) == 1) {
                markCopyComplete(generation, namespaceId, tier, pageId, startLayer, layerCount);
            }
        };

        // Call enqueueTextureUpload outside the page-pool lock
        if (!uploads_->enqueueTextureUpload(upload)) {
            return false;
        }
        pageSubrangeUploads_++;
    }

    return true;
}






bool TexturePagePool::isReady(uint64_t generation, Namespace ns, uint32_t tier,
                               uint32_t page, uint32_t layer) const {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& p : pages_) {
        if (p.generation == generation
            && p.namespaceId == static_cast<uint32_t>(ns)
            && p.tier == tier
            && p.page == page) {
            if (layer >= p.layerCapacity) return false;
            if (layer < p.layerUploaded.size() && layer < p.layerMipsReady.size()) {
                return p.layerUploaded[layer] && p.layerMipsReady[layer];
            }
            return false;
        }
    }
    return false;
}

void TexturePagePool::markReady(uint64_t generation, const Allocation& allocation) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (generation != activeGeneration_) return;
    if (!allocation.valid) return;

    for (auto& p : pages_) {
        if (p.generation == generation &&
            p.namespaceId == allocation.first.namespaceId &&
            p.tier == allocation.first.tier &&
            p.page == allocation.first.page) {
            // Mark individual layers as uploaded and mip-ready
            for (uint32_t l = allocation.first.layer;
                 l < allocation.first.layer + allocation.layerCount && l < p.layerCapacity; l++) {
                if (l < p.layerUploaded.size()) p.layerUploaded[l] = true;
                // Mip generation is not yet split out; v4 currently clamps to mip 0 for uploaded pages.
                // True mip generation is not implemented yet.
                if (l < p.layerMipsReady.size()) p.layerMipsReady[l] = true;
            }
            // Recompute readyLayers from allocated layers only;
            // do not count unallocated gaps as ready.
            uint32_t ready = 0;
            for (uint32_t l = 0; l < p.layersUsed; ++l) {
                if (l < p.layerAllocated.size() && p.layerAllocated[l] &&
                    l < p.layerUploaded.size() && p.layerUploaded[l] &&
                    l < p.layerMipsReady.size() && p.layerMipsReady[l]) {
                    ++ready;
                }
            }
            p.readyLayers = ready;
            p.mipsReady = (ready == p.layersUsed);
            break;
        }
    }
}

uint32_t TexturePagePool::ctmResidentCapacityLocked() const {
    uint32_t capacity = 0;
    for (const auto& p : pages_) {
        if (p.namespaceId == Ctm && p.allocated) {
            capacity += p.layerCapacity;
        }
    }
    return capacity;
}

uint32_t TexturePagePool::ctmPresentMaterialsLocked() const {
    uint32_t count = 0;
    for (const auto& p : pages_) {
        if (p.namespaceId == Ctm && p.allocated) {
            count += p.layersUsed;
        }
    }
    return count;
}

bool TexturePagePool::ctmPagesExhaustedLocked() const {
    uint32_t ctmPages = 0;
    for (const auto& p : pages_) {
        if (p.namespaceId == Ctm && p.allocated) {
            ctmPages++;
            if (p.layersUsed < p.layerCapacity) return false;
        }
    }
    return ctmPages >= kMaxPagesPerNamespace;
}

uint32_t TexturePagePool::ctmUnaddressableMaterialsLocked() const {
    return ctmPagesExhaustedLocked() ? 1u : 0u;
}

uint32_t TexturePagePool::unreadyAllocatedPageCountLocked(uint64_t generation) const {
    uint32_t count = 0;
    for (const auto& p : pages_) {
        if (!p.allocated || p.generation != generation || p.layersUsed == 0) {
            continue;
        }
        for (uint32_t l = 0; l < p.layersUsed && l < p.layerUploaded.size()
             && l < p.layerMipsReady.size(); ++l) {
            if (!p.layerUploaded[l] || !p.layerMipsReady[l]) {
                count++;
                break;
            }
        }
    }
    return count;
}

uint32_t TexturePagePool::ctmResidentCapacity() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return ctmResidentCapacityLocked();
}

uint32_t TexturePagePool::ctmPresentMaterials() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return ctmPresentMaterialsLocked();
}

bool TexturePagePool::ctmPagesExhausted() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return ctmPagesExhaustedLocked();
}

uint32_t TexturePagePool::ctmUnaddressableMaterials() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return ctmUnaddressableMaterialsLocked();
}

uint32_t TexturePagePool::unreadyAllocatedPageCount(uint64_t generation) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return unreadyAllocatedPageCountLocked(generation);
}

uint32_t TexturePagePool::unreadyAllocatedLayerCount(uint64_t generation, bool visibleOnly) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return unreadyAllocatedLayerCountLocked(generation, visibleOnly);
}

uint32_t TexturePagePool::pendingMipPageCount(uint64_t generation, bool visibleOnly) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return pendingMipPageCountLocked(generation, visibleOnly);
}

uint32_t TexturePagePool::unreadyAllocatedLayerCountLocked(uint64_t generation, bool visibleOnly) const {
    uint32_t count = 0;
    for (const auto& p : pages_) {
        if (p.generation != generation || !p.allocated) continue;
        for (uint32_t l = 0; l < p.layersUsed; ++l) {
            if (l >= p.layerAllocated.size() || !p.layerAllocated[l]) continue;
            if (visibleOnly && (l >= p.layerVisible.size() || !p.layerVisible[l])) continue;
            if (l >= p.layerUploaded.size() || !p.layerUploaded[l] ||
                l >= p.layerMipsReady.size() || !p.layerMipsReady[l]) {
                ++count;
            }
        }
    }
    return count;
}

uint32_t TexturePagePool::pendingMipPageCountLocked(uint64_t generation, bool visibleOnly) const {
    uint32_t count = 0;
    for (const auto& p : pages_) {
        if (p.generation != generation || !p.allocated) continue;
        bool hasPendingMip = false;
        for (uint32_t l = 0; l < p.layersUsed; ++l) {
            if (l >= p.layerAllocated.size() || !p.layerAllocated[l]) continue;
            if (visibleOnly && (l >= p.layerVisible.size() || !p.layerVisible[l])) continue;
            if (l >= p.layerMipsReady.size() || !p.layerMipsReady[l]) {
                hasPendingMip = true;
                break;
            }
        }
        if (hasPendingMip) ++count;
    }
    return count;
}

std::string TexturePagePool::statusJson() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::ostringstream out;
    out << "{"
        << "\"schema\":\"radser_texture_page_pool_status_v4\","
        << "\"generation\":" << activeGeneration_ << ","
        << "\"totalPages\":" << pages_.size() << ","
        << "\"planesPerPage\":4,"
        << "\"pageUploadMode\":\"async_four_plane_texture_subresource\","
        << "\"pageImageAllocations\":" << pageImageAllocations_ << ","
        << "\"pageSubrangeUploads\":" << pageSubrangeUploads_ << ","
        << "\"ctmResidentCapacity\":" << ctmResidentCapacityLocked() << ","
        << "\"ctmPresentMaterials\":" << ctmPresentMaterialsLocked() << ","
        << "\"ctmPagesExhausted\":" << (ctmPagesExhaustedLocked() ? "true" : "false") << ","
        << "\"ctmUnaddressableMaterials\":" << ctmUnaddressableMaterialsLocked() << ","
        << "\"nativeUnreadyAllocatedPageCount\":" << unreadyAllocatedPageCountLocked(activeGeneration_) << ","
        << "\"nativeUnreadyAllocatedLayerCount\":" << unreadyAllocatedLayerCountLocked(activeGeneration_, false) << ","
        << "\"nativeUnreadyAllocatedVisibleLayerCount\":" << unreadyAllocatedLayerCountLocked(activeGeneration_, true) << ","
        << "\"nativePendingMipPageCount\":" << pendingMipPageCountLocked(activeGeneration_, false) << ","
        << "\"nativePendingMipVisiblePageCount\":" << pendingMipPageCountLocked(activeGeneration_, true) << ",";

    // Per-namespace summary
    out << "\"namespaces\":{";
    const char* nsNames[] = {"fallback", "vanilla", "ctm", "dynamic"};
    for (int ns = 0; ns < 4; ns++) {
        if (ns > 0) out << ",";
        out << "\"" << nsNames[ns] << "\":{";
        uint32_t totalPages = 0, totalLayers = 0, readyLayers = 0;
        for (const auto& p : pages_) {
            if (p.namespaceId == static_cast<uint32_t>(ns)) {
                totalPages++;
                totalLayers += p.layersUsed;
                readyLayers += p.readyLayers;
            }
        }
        out << "\"pages\":" << totalPages
            << ",\"layers\":" << totalLayers
            << ",\"readyLayers\":" << readyLayers
            << "}";
    }
    out << "}}";
    return out.str();
}

TexturePagePool::Page& TexturePagePool::pageForAllocationLocked(
    uint64_t generation, Namespace ns, uint32_t tier, uint32_t neededLayers) {
    // Validate dependencies
    if (!device_ || !vma_ || !uploads_) {
        static Page invalid;
        return invalid;
    }
    if (tier >= kMaxTiers) {
        static Page invalid;
        return invalid;
    }
    const uint32_t size = tierSize(tier);
    if (size == 0) {
        static Page invalid;
        return invalid;
    }
    const uint32_t capacity = pageLayerCapacity(tier);
    if (capacity == 0 || neededLayers == 0 || neededLayers > capacity) {
        static Page invalid;
        return invalid;
    }

    // Find an existing page with room — only if generation, namespace, tier match,
    // allocated, and albedoImage is valid with a non-null Vulkan image handle.
    for (auto& page : pages_) {
        if (page.generation == generation &&
            page.namespaceId == static_cast<uint32_t>(ns) &&
            page.tier == tier &&
            page.allocated &&
            page.albedoImage &&
            page.albedoImage->vkImage() != VK_NULL_HANDLE &&
            page.layersUsed + neededLayers <= page.layerCapacity) {
            return page;
        }
    }

    // Allocate a new page
    uint32_t newPageIndex = 0;
    for (const auto& p : pages_) {
        if (p.namespaceId == static_cast<uint32_t>(ns) && p.tier == tier) {
            newPageIndex = std::max(newPageIndex, p.page + 1);
        }
    }
    if (newPageIndex >= kMaxPagesPerNamespace) {
        static Page invalid;
        return invalid;
    }

    Page newPage;
    newPage.generation = generation;
    newPage.namespaceId = static_cast<uint32_t>(ns);
    newPage.tier = tier;
    newPage.page = newPageIndex;
    newPage.layerCapacity = capacity;
    newPage.layersUsed = 0;
    newPage.readyLayers = 0;
    newPage.mipCount = 1;
    newPage.allocated = true;
    newPage.mipsReady = false;
    newPage.format = VK_FORMAT_R8G8B8A8_UNORM;

    // Initialize per-layer state arrays
    newPage.layerAllocated.assign(capacity, false);
    newPage.layerVisible.assign(capacity, false);
    newPage.layerUploaded.assign(capacity, false);
    newPage.layerMipsReady.assign(capacity, false);

    VkImageUsageFlags usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    newPage.albedoImage = vk::DeviceLocalImage::create(
        device_, vma_, false, newPage.mipCount, size, size, newPage.layerCapacity,
        newPage.format, usage, 0, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
    // Allocate all four plane images for v4 four-plane uploads
    newPage.specularImage = vk::DeviceLocalImage::create(
        device_, vma_, false, newPage.mipCount, size, size, newPage.layerCapacity,
        newPage.format, usage, 0, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
    newPage.normalImage = vk::DeviceLocalImage::create(
        device_, vma_, false, newPage.mipCount, size, size, newPage.layerCapacity,
        newPage.format, usage, 0, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
    newPage.flagImage = vk::DeviceLocalImage::create(
        device_, vma_, false, newPage.mipCount, size, size, newPage.layerCapacity,
        newPage.format, usage, 0, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);

    // Increment pageImageAllocations_ only after ALL plane images exist with valid VkImage handles
    if (!newPage.albedoImage || newPage.albedoImage->vkImage() == VK_NULL_HANDLE
        || !newPage.specularImage || newPage.specularImage->vkImage() == VK_NULL_HANDLE
        || !newPage.normalImage || newPage.normalImage->vkImage() == VK_NULL_HANDLE
        || !newPage.flagImage || newPage.flagImage->vkImage() == VK_NULL_HANDLE) {
        static Page invalid;
        return invalid;
    }

    newPage.sampler = vk::Sampler::create(
        device_, VK_FILTER_NEAREST, VK_SAMPLER_MIPMAP_MODE_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);

    pages_.push_back(newPage);
    pageImageAllocations_++;
    return pages_.back();
}

TexturePagePool::Page& TexturePagePool::pageForExactAllocationLocked(
    uint64_t generation, Namespace ns, uint32_t tier,
    uint32_t page, uint32_t layerCapacity) {
    // Validate dependencies
    if (!device_ || !vma_ || !uploads_) {
        static Page invalid;
        return invalid;
    }
    if (tier >= kMaxTiers) {
        static Page invalid;
        return invalid;
    }
    const uint32_t size = tierSize(tier);
    if (size == 0) {
        static Page invalid;
        return invalid;
    }
    if (layerCapacity == 0) {
        static Page invalid;
        return invalid;
    }

    // Find existing page with exact match
    for (auto& p : pages_) {
        if (p.generation == generation &&
            p.namespaceId == static_cast<uint32_t>(ns) &&
            p.tier == tier &&
            p.page == page) {
            if (p.allocated && p.albedoImage && p.albedoImage->vkImage() != VK_NULL_HANDLE) {
                return p;
            }
            static Page invalid;
            return invalid;
        }
    }

    // Page does not exist — allocate it at the exact page index
    if (page >= kMaxPagesPerNamespace) {
        static Page invalid;
        return invalid;
    }

    Page newPage;
    newPage.generation = generation;
    newPage.namespaceId = static_cast<uint32_t>(ns);
    newPage.tier = tier;
    newPage.page = page;
    newPage.layerCapacity = layerCapacity;
    newPage.layersUsed = 0;
    newPage.readyLayers = 0;
    newPage.mipCount = 1;
    newPage.allocated = true;
    newPage.mipsReady = false;
    newPage.format = VK_FORMAT_R8G8B8A8_UNORM;

    // Initialize per-layer state arrays
    newPage.layerAllocated.assign(layerCapacity, false);
    newPage.layerVisible.assign(layerCapacity, false);
    newPage.layerUploaded.assign(layerCapacity, false);
    newPage.layerMipsReady.assign(layerCapacity, false);

    VkImageUsageFlags usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    newPage.albedoImage = vk::DeviceLocalImage::create(
        device_, vma_, false, newPage.mipCount, size, size, newPage.layerCapacity,
        newPage.format, usage, 0, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
    // Allocate all four plane images for v4 four-plane uploads
    newPage.specularImage = vk::DeviceLocalImage::create(
        device_, vma_, false, newPage.mipCount, size, size, newPage.layerCapacity,
        newPage.format, usage, 0, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
    newPage.normalImage = vk::DeviceLocalImage::create(
        device_, vma_, false, newPage.mipCount, size, size, newPage.layerCapacity,
        newPage.format, usage, 0, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
    newPage.flagImage = vk::DeviceLocalImage::create(
        device_, vma_, false, newPage.mipCount, size, size, newPage.layerCapacity,
        newPage.format, usage, 0, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);

    if (!newPage.albedoImage || newPage.albedoImage->vkImage() == VK_NULL_HANDLE
        || !newPage.specularImage || newPage.specularImage->vkImage() == VK_NULL_HANDLE
        || !newPage.normalImage || newPage.normalImage->vkImage() == VK_NULL_HANDLE
        || !newPage.flagImage || newPage.flagImage->vkImage() == VK_NULL_HANDLE) {
        static Page invalid;
        return invalid;
    }

    newPage.sampler = vk::Sampler::create(
        device_, VK_FILTER_NEAREST, VK_SAMPLER_MIPMAP_MODE_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);

    pages_.push_back(newPage);
    pageImageAllocations_++;
    return pages_.back();
}

TexturePagePool::Page* TexturePagePool::findPageLocked(uint64_t generation,
                                                       uint32_t namespaceId,
                                                       uint32_t tier,
                                                       uint32_t page) {
    for (auto& p : pages_) {
        if (p.generation == generation
            && p.namespaceId == namespaceId
            && p.tier == tier
            && p.page == page) {
            return &p;
        }
    }
    return nullptr;
}

void TexturePagePool::markCopyComplete(uint64_t generation, uint32_t namespaceId, uint32_t tier, uint32_t page, uint32_t startLayer, uint32_t layerCount) {
    std::lock_guard<std::mutex> lock(mutex_);
    Page* p = findPageLocked(generation, namespaceId, tier, page);
    if (!p || startLayer >= p->layerCapacity) return;
    if (generation != activeGeneration_) return;
    uint32_t endLayer = std::min(p->layerCapacity, startLayer + layerCount);
    for (uint32_t l = startLayer; l < endLayer; ++l) {
        if (l < p->layerUploaded.size()) p->layerUploaded[l] = true;
        // Mip generation is not yet split out; v4 currently clamps to mip 0 for uploaded pages.
        // True mip generation is not implemented yet. Under mip0Clamp mode, mark mips ready
        // immediately since there is only one mip level. When real mipgen lands, remove this.
        if (l < p->layerMipsReady.size()) p->layerMipsReady[l] = true;
    }
    // Recompute readyLayers from allocated layers only
    uint32_t ready = 0;
    for (uint32_t l = 0; l < p->layersUsed; ++l) {
        if (l < p->layerAllocated.size() && p->layerAllocated[l] &&
            l < p->layerUploaded.size() && p->layerUploaded[l] &&
            l < p->layerMipsReady.size() && p->layerMipsReady[l]) {
            ++ready;
        }
    }
    p->readyLayers = ready;
    p->mipsReady = (ready == p->layersUsed);
}
