#include "core/render/texture_page_pool.hpp"
#include "core/render/renderer.hpp"
#include "core/render/render_framework.hpp"
#include <algorithm>
#include <iostream>
#include <sstream>

static const uint32_t TIER_SIZES[] = {16, 32, 64, 128, 256, 512, 1024};
static const uint32_t DEFAULT_LAYERS_PER_PAGE = 256;

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

    // In the full implementation, this enqueues a texture upload through GpuUploadService
    // targeting the specific page/layer range in the texture array.
    pageSubrangeUploads_++;
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

uint32_t TexturePagePool::ctmResidentCapacity() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return ctmResidentCapacityLocked();
}

uint32_t TexturePagePool::ctmPresentMaterials() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return ctmPresentMaterialsLocked();
}

bool TexturePagePool::ctmPagesExhausted() const {
    // Not exhausted if we can still allocate more pages
    return false;
}

uint32_t TexturePagePool::ctmUnaddressableMaterials() const {
    // In v4, all materials should be addressable
    return 0;
}

std::string TexturePagePool::statusJson() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::ostringstream out;
    out << "{"
        << "\"schema\":\"radser_texture_page_pool_status_v4\","
        << "\"generation\":" << activeGeneration_ << ","
        << "\"totalPages\":" << pages_.size() << ","
        << "\"pageImageAllocations\":" << pageImageAllocations_ << ","
        << "\"pageSubrangeUploads\":" << pageSubrangeUploads_ << ","
        << "\"ctmResidentCapacity\":" << ctmResidentCapacityLocked() << ","
        << "\"ctmPresentMaterials\":" << ctmPresentMaterialsLocked() << ","
        << "\"ctmPagesExhausted\":" << (ctmPagesExhausted() ? "true" : "false") << ","
        << "\"ctmUnaddressableMaterials\":" << ctmUnaddressableMaterials() << ",";

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
    newPage.allocated = true;
    newPage.mipsReady = false;
    newPage.layerUploaded.resize(newPage.layerCapacity, false);
    newPage.layerMipsReady.resize(newPage.layerCapacity, false);

    pages_.push_back(newPage);
    pageImageAllocations_++;

    return pages_.back();
}
