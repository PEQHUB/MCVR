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
                                  std::shared_ptr<vk::VMA> vma,
                                  GpuUploadService* uploads) {
    std::lock_guard<std::mutex> lock(mutex_);
    device_ = device;
    vma_ = vma;
    uploads_ = uploads;
    pages_.clear();
    return true;
}

void TexturePagePool::resetGeneration(uint64_t generation) {
    std::lock_guard<std::mutex> lock(mutex_);
    activeGeneration_ = generation;
    // Mark all pages as belonging to the new generation
    for (auto& page : pages_) {
        page.generation = generation;
        page.layersUsed = 0;
        page.readyLayers = 0;
        page.mipsReady = false;
        std::fill(page.layerUploaded.begin(), page.layerUploaded.end(), false);
        std::fill(page.layerMipsReady.begin(), page.layerMipsReady.end(), false);
    }
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
    if (generation == 0 || tier >= kMaxTiers || layerCount == 0) {
        return {PageHandle{}, 0, false};
    }

    Page& page = pageForAllocationLocked(generation, ns, tier, layerCount);
    if (!page.allocated) {
        return {PageHandle{}, 0, false};
    }

    uint32_t startLayer = page.layersUsed;
    if (startLayer + layerCount > page.layerCapacity) {
        return {PageHandle{}, 0, false};
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

bool TexturePagePool::upload(uint64_t generation, const Allocation& allocation,
                              const uint8_t* rgba, uint64_t bytes, VkFormat format,
                              bool visible, GpuUploadService::Priority priority) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!allocation.valid || !rgba || bytes == 0) return false;
    if (generation != activeGeneration_) return false;
    if (!uploads_) return false;

    Page* page = findPageLocked(generation, allocation.first.namespaceId,
        allocation.first.tier, allocation.first.page);
    if (!page || !page->allocated || !page->albedoImage) return false;
    if (allocation.first.layer >= page->layerCapacity
        || allocation.layerCount == 0
        || allocation.first.layer + allocation.layerCount > page->layerCapacity) {
        return false;
    }

    const uint32_t size = tierSize(allocation.first.tier);
    const uint64_t bytesPerLayer = static_cast<uint64_t>(size) * size * 4;
    const uint64_t requiredBytes = bytesPerLayer * allocation.layerCount;
    if (size == 0 || bytes < requiredBytes) return false;

    for (uint32_t l = allocation.first.layer;
         l < allocation.first.layer + allocation.layerCount; ++l) {
        if (l < page->layerUploaded.size()) page->layerUploaded[l] = false;
        if (l < page->layerMipsReady.size()) page->layerMipsReady[l] = false;
    }

    GpuUploadService::TextureUpload upload{};
    upload.generation = generation;
    upload.namespaceId = allocation.first.namespaceId;
    upload.tier = allocation.first.tier;
    upload.page = allocation.first.page;
    upload.layer = allocation.first.layer;
    upload.layerCount = allocation.layerCount;
    upload.width = size;
    upload.height = size;
    upload.bytesPerLayer = static_cast<uint32_t>(bytesPerLayer);
    upload.format = format;
    upload.dstImage = page->albedoImage->vkImage();
    upload.mipLevels = page->mipCount;
    upload.data = rgba;
    upload.bytes = requiredBytes;
    upload.visible = visible;
    upload.priority = priority;
    upload.onComplete = [this, generation,
                         namespaceId = allocation.first.namespaceId,
                         tier = allocation.first.tier,
                         pageId = allocation.first.page,
                         startLayer = allocation.first.layer,
                         layerCount = allocation.layerCount](uint64_t) {
        markCopyComplete(generation, namespaceId, tier, pageId, startLayer, layerCount);
    };

    if (!uploads_->enqueueTextureUpload(upload)) {
        return false;
    }
    pageSubrangeUploads_++;
    return true;
}

bool TexturePagePool::upload(uint64_t generation, const Allocation& allocation,
                              const uint8_t* albedo, const uint8_t* specular,
                              const uint8_t* normal, const uint8_t* flag,
                              uint64_t bytesPerLayer, uint32_t channelMask,
                              VkFormat format, bool visible) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!allocation.valid || !albedo || bytesPerLayer == 0) return false;
    if (generation != activeGeneration_) return false;
    if (!uploads_) return false;

    Page* page = findPageLocked(generation, allocation.first.namespaceId,
        allocation.first.tier, allocation.first.page);
    if (!page || !page->allocated || !page->albedoImage) return false;
    if (allocation.first.layer >= page->layerCapacity
        || allocation.layerCount == 0
        || allocation.first.layer + allocation.layerCount > page->layerCapacity) {
        return false;
    }

    const uint32_t size = tierSize(allocation.first.tier);
    const uint64_t requiredBytesPerLayer = static_cast<uint64_t>(size) * size * 4;
    if (size == 0 || bytesPerLayer < requiredBytesPerLayer) return false;
    const uint64_t requiredBytes = requiredBytesPerLayer * allocation.layerCount;

    for (uint32_t l = allocation.first.layer;
         l < allocation.first.layer + allocation.layerCount; ++l) {
        if (l < page->layerUploaded.size()) page->layerUploaded[l] = false;
        if (l < page->layerMipsReady.size()) page->layerMipsReady[l] = false;
    }

    struct Plane {
        const uint8_t* data;
        std::shared_ptr<vk::DeviceLocalImage> image;
        GpuUploadService::Priority priority;
    };
    std::vector<Plane> planes;
    planes.push_back({albedo, page->albedoImage, visible
        ? GpuUploadService::Priority::FirstFrameAlbedo
        : GpuUploadService::Priority::BackgroundCtm});
    if ((channelMask & CHANNEL_SPECULAR) && specular && page->specularImage) {
        planes.push_back({specular, page->specularImage, visible
            ? GpuUploadService::Priority::FirstFrameAux
            : GpuUploadService::Priority::BackgroundCtm});
    }
    if ((channelMask & CHANNEL_NORMAL) && normal && page->normalImage) {
        planes.push_back({normal, page->normalImage, visible
            ? GpuUploadService::Priority::FirstFrameAux
            : GpuUploadService::Priority::BackgroundCtm});
    }
    if ((channelMask & CHANNEL_FLAG) && flag && page->flagImage) {
        planes.push_back({flag, page->flagImage, visible
            ? GpuUploadService::Priority::FirstFrameAux
            : GpuUploadService::Priority::BackgroundCtm});
    }

    auto remaining = std::make_shared<std::atomic<uint32_t>>(static_cast<uint32_t>(planes.size()));
    for (const auto& plane : planes) {
        if (!plane.data || !plane.image) return false;
        GpuUploadService::TextureUpload upload{};
        upload.generation = generation;
        upload.namespaceId = allocation.first.namespaceId;
        upload.tier = allocation.first.tier;
        upload.page = allocation.first.page;
        upload.layer = allocation.first.layer;
        upload.layerCount = allocation.layerCount;
        upload.width = size;
        upload.height = size;
        upload.bytesPerLayer = static_cast<uint32_t>(requiredBytesPerLayer);
        upload.format = format;
        upload.dstImage = plane.image->vkImage();
        upload.mipLevels = page->mipCount;
        upload.data = plane.data;
        upload.bytes = requiredBytes;
        upload.visible = visible;
        upload.priority = plane.priority;
        upload.onComplete = [this, generation,
                             namespaceId = allocation.first.namespaceId,
                             tier = allocation.first.tier,
                             pageId = allocation.first.page,
                             startLayer = allocation.first.layer,
                             layerCount = allocation.layerCount,
                             remaining](uint64_t) {
            if (remaining->fetch_sub(1, std::memory_order_acq_rel) == 1) {
                markCopyComplete(generation, namespaceId, tier, pageId, startLayer, layerCount);
            }
        };
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
    for (auto& p : pages_) {
        if (p.generation == generation
            && p.namespaceId == allocation.first.namespaceId
            && p.tier == allocation.first.tier
            && p.page == allocation.first.page) {
            // Mark individual layers as uploaded and mip-ready
            for (uint32_t l = allocation.first.layer;
                 l < allocation.first.layer + allocation.layerCount && l < p.layerCapacity; l++) {
                if (l < p.layerUploaded.size()) p.layerUploaded[l] = true;
                if (l < p.layerMipsReady.size()) p.layerMipsReady[l] = true;
            }
            // Update readyLayers to the highest contiguous ready layer
            uint32_t maxReady = 0;
            for (uint32_t l = 0; l < p.layerCapacity && l < p.layerUploaded.size(); l++) {
                if (p.layerUploaded[l] && p.layerMipsReady[l]) {
                    maxReady = l + 1;
                } else {
                    break;
                }
            }
            p.readyLayers = std::max(p.readyLayers, maxReady);
            if (p.readyLayers >= p.layersUsed) {
                p.mipsReady = true;
            }
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
        << "\"nativeUnreadyAllocatedPageCount\":" << unreadyAllocatedPageCountLocked(activeGeneration_) << ",";

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
    // Find an existing page with room
    for (auto& page : pages_) {
        if (page.generation == generation
            && page.namespaceId == static_cast<uint32_t>(ns)
            && page.tier == tier
            && page.allocated
            && page.layersUsed + neededLayers <= page.layerCapacity) {
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
    newPage.layerCapacity = pageLayerCapacity(tier);
    newPage.layersUsed = 0;
    newPage.readyLayers = 0;
    newPage.mipCount = 1;
    newPage.allocated = true;
    newPage.mipsReady = false;
    newPage.format = VK_FORMAT_R8G8B8A8_UNORM;
    newPage.layerUploaded.resize(newPage.layerCapacity, false);
    newPage.layerMipsReady.resize(newPage.layerCapacity, false);

    uint32_t size = tierSize(tier);
    VkImageUsageFlags usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    newPage.albedoImage = vk::DeviceLocalImage::create(
        device_, vma_, false, newPage.mipCount, size, size, newPage.layerCapacity,
        newPage.format, usage, 0, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
    newPage.specularImage = vk::DeviceLocalImage::create(
        device_, vma_, false, newPage.mipCount, size, size, newPage.layerCapacity,
        newPage.format, usage, 0, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
    newPage.normalImage = vk::DeviceLocalImage::create(
        device_, vma_, false, newPage.mipCount, size, size, newPage.layerCapacity,
        newPage.format, usage, 0, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
    newPage.flagImage = vk::DeviceLocalImage::create(
        device_, vma_, false, newPage.mipCount, size, size, newPage.layerCapacity,
        newPage.format, usage, 0, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
    if (!newPage.albedoImage || !newPage.specularImage || !newPage.normalImage || !newPage.flagImage) {
        static Page invalid;
        return invalid;
    }
    newPage.sampler = vk::Sampler::create(
        device_, VK_FILTER_NEAREST, VK_SAMPLER_MIPMAP_MODE_NEAREST,
        VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);

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

void TexturePagePool::markCopyComplete(uint64_t generation, uint32_t namespaceId,
                                       uint32_t tier, uint32_t page,
                                       uint32_t startLayer, uint32_t layerCount) {
    std::lock_guard<std::mutex> lock(mutex_);
    Page* p = findPageLocked(generation, namespaceId, tier, page);
    if (!p || startLayer >= p->layerCapacity) return;
    uint32_t endLayer = std::min(p->layerCapacity, startLayer + layerCount);
    for (uint32_t l = startLayer; l < endLayer; ++l) {
        if (l < p->layerUploaded.size()) p->layerUploaded[l] = true;
        // Mip generation is not yet split out; v4 currently clamps to mip 0 for uploaded pages.
        if (l < p->layerMipsReady.size()) p->layerMipsReady[l] = true;
    }
    uint32_t ready = 0;
    for (uint32_t l = 0; l < p->layersUsed && l < p->layerUploaded.size()
         && l < p->layerMipsReady.size(); ++l) {
        if (p->layerUploaded[l] && p->layerMipsReady[l]) {
            ready++;
        }
    }
    p->readyLayers = ready;
    p->mipsReady = p->readyLayers >= p->layersUsed;
}
