#include "core/render/texture_loader_v4.hpp"
#include "core/render/renderer.hpp"
#include "core/render/render_framework.hpp"
#include "core/build/build_info.hpp"
#include <iostream>
#include <sstream>

bool TextureLoaderV4::initialize(std::shared_ptr<vk::Device> device,
                                  std::shared_ptr<vk::VMA> vma) {
    if (initialized_.load(std::memory_order_acquire)) return true;
    if (!device || !vma) return false;

    if (!uploadService_.initialize(device, vma, 256 * 1024 * 1024)) {
        std::cerr << "[TextureLoaderV4] Failed to initialize GPU upload service" << std::endl;
        return false;
    }

    if (!pagePool_.initialize(device, vma, &uploadService_)) {
        std::cerr << "[TextureLoaderV4] Failed to initialize texture page pool" << std::endl;
        return false;
    }

    initialized_.store(true, std::memory_order_release);
    std::cout << "[TextureLoaderV4] Initialized: abi=" << build_info::kTextureLoaderAbiVersion
              << " cacheSchema=" << build_info::kCacheSchemaVersion << std::endl;
    return true;
}

void TextureLoaderV4::shutdown() {
    uploadService_.shutdown();
    initialized_.store(false, std::memory_order_release);
}

bool TextureLoaderV4::beginGeneration(uint64_t generation) {
    if (!initialized_.load(std::memory_order_acquire)) return false;
    if (generation == 0) return false;

    std::lock_guard<std::mutex> lock(mutex_);
    activeGeneration_.store(generation, std::memory_order_release);
    generationCommitted_.store(false, std::memory_order_release);
    pagePool_.resetGeneration(generation);

    std::cout << "[TextureLoaderV4] Begin generation " << generation << std::endl;
    return true;
}

bool TextureLoaderV4::enqueueUpload(const UploadRequest& request) {
    if (!initialized_.load(std::memory_order_acquire)) {
        std::cout << "[TextureLoaderV4] enqueueUpload REJECTED: not initialized" << std::endl;
        return false;
    }
    if (request.generation != activeGeneration_.load(std::memory_order_acquire)) {
        std::cout << "[TextureLoaderV4] enqueueUpload REJECTED: gen mismatch gen=" << request.generation << " active=" << activeGeneration_.load() << std::endl;
        return false;
    }
    if (request.generation == 0) return false;
    if (request.albedoData == nullptr || request.bytesPerLayer == 0) return false;
    if (request.tier >= TexturePagePool::kMaxTiers) return false;
    if (request.layerCount == 0) return false;
    if (request.layerCapacity == 0) return false;
    if (request.startLayer >= request.layerCapacity
        || request.layerCount > request.layerCapacity - request.startLayer) {
        return false;
    }
    // Reject chunks that exceed native page capacity — Java should have chunked them
    if (request.layerCount > TexturePagePool::pageLayerCapacityStatic(request.tier)) {
        std::cout << "[TextureLoaderV4] enqueueUpload REJECTED: layerCount=" << request.layerCount
                  << " exceeds native page capacity=" << TexturePagePool::pageLayerCapacityStatic(request.tier)
                  << " for tier=" << request.tier << std::endl;
        return false;
    }

    const uint32_t expectedSize = request.tier < TexturePagePool::kMaxTiers
        ? (16u << request.tier)
        : 0u;
    if (expectedSize == 0 || request.width != expectedSize || request.height != expectedSize) {
        std::cout << "[TextureLoaderV4] enqueueUpload REJECTED: size mismatch tier=" << request.tier << " expected=" << expectedSize << " w=" << request.width << " h=" << request.height << std::endl;
        return false;
    }
    const uint64_t expectedBytesPerLayer = static_cast<uint64_t>(expectedSize) * expectedSize * 4u;
    if (request.bytesPerLayer != expectedBytesPerLayer) return false;

    // Allocate page pool layers. Java provides absolute page and startLayer values.
    //
    // CONTRACT: Java's mixin loop sends exactly ONE page per tier. Within that page,
    // startLayer is the absolute tier-local layer index (0..totalTierSprites-1).
    // The tier-local index is split across native pages by dividing by nativeCapacity.
    //
    // Java sends: page = VANILLA_TIER_FIRST_PAGE + tierIndex (1 per tier)
    //            startLayer = absolute tier-local layer index
    //            layerCapacity = total sprites in this tier
    //
    // Native computes: nativePage = startLayer / nativeCapacity
    //                  nativeStartLayer = startLayer % nativeCapacity
    //                  nativeCapacity = per-native-page limit (e.g., 256 for T128)
    //
    // IMPORTANT: If Java ever sends multiple pages per tier, this normalization
    // must incorporate the Java page index: absoluteLayer = javaPageIndex * javaPageCapacity + startLayer.
    const uint32_t nativeCapacity = TexturePagePool::pageLayerCapacityStatic(request.tier);
    if (nativeCapacity == 0) {
        std::cout << "[TextureLoaderV4] enqueueUpload REJECTED: zero native capacity for tier=" << request.tier << std::endl;
        return false;
    }

    uint32_t normalizedPage = request.page;
    uint32_t normalizedStartLayer = request.startLayer;
    uint32_t normalizedCapacity = request.layerCapacity;

    if (request.page != UINT32_MAX && request.startLayer != UINT32_MAX) {
        // CONTRACT ASSERTION: request.page must equal VANILLA_TIER_FIRST_PAGE + tier,
        // confirming one Java page per tier. Currently VANILLA_TIER_FIRST_PAGE = 1, so
        // for tier 0 the page should be 1, tier 1 -> page 2, etc.
        // Bad page hints are rejected before placement so stale callers fail closed.
        const uint32_t expectedPage = 1 + request.tier;
        if (request.page != expectedPage) {
            javaPageContractRejects_.fetch_add(1, std::memory_order_relaxed);
            std::cout << "[TextureLoaderV4] enqueueUpload REJECTED: javaPage="
                      << request.page << " expected=" << expectedPage
                      << " tier=" << request.tier << std::endl;
            return false;
        }

        // Compute tier-local native page index from Java's absolute startLayer
        normalizedPage = request.startLayer / nativeCapacity;
        normalizedStartLayer = request.startLayer % nativeCapacity;
        normalizedCapacity = nativeCapacity;

        // Cross-page boundary guard: reject uploads that would span two native pages.
        // Java must chunk at nativeCapacity boundaries; this is the safety net.
        if (request.layerCount > nativeCapacity - normalizedStartLayer) {
            std::cout << "[TextureLoaderV4] enqueueUpload REJECTED: cross-page boundary"
                      << " startLayer=" << request.startLayer << " layerCount=" << request.layerCount
                      << " nativePage=" << normalizedPage << " nativeStartLayer=" << normalizedStartLayer
                      << " nativeCapacity=" << nativeCapacity << std::endl;
            return false;
        }
    }

    std::cout << "[TextureLoaderV4] enqueueUpload gen=" << request.generation << " ns=" << request.namespaceId
              << " tier=" << request.tier
              << " javaPage=" << request.page << " javaStartLayer=" << request.startLayer << " javaCapacity=" << request.layerCapacity
              << " -> nativePage=" << normalizedPage << " nativeStartLayer=" << normalizedStartLayer << " nativeCapacity=" << normalizedCapacity
              << " layers=" << request.layerCount
              << " channelMask=0x" << std::hex << request.channelMask << std::dec << std::endl;

    auto ns = static_cast<TexturePagePool::Namespace>(request.namespaceId);
    TexturePagePool::Allocation alloc;
    if (request.page != UINT32_MAX && request.startLayer != UINT32_MAX) {
        // Java provided exact placement — use normalized tier-local values
        alloc = pagePool_.allocateExact(request.generation, ns, request.tier,
                                         normalizedPage, normalizedStartLayer,
                                         request.layerCount, normalizedCapacity,
                                         request.visible);
        if (!alloc.valid) {
            std::cout << "[TextureLoaderV4] enqueueUpload REJECTED: pagePool_.allocateExact() returned invalid"
                      << " page=" << normalizedPage << " startLayer=" << normalizedStartLayer
                      << " layerCount=" << request.layerCount << " capacity=" << normalizedCapacity << std::endl;
            return false;
        }
    } else {
        // Dynamic allocation — native chooses placement
        alloc = pagePool_.allocate(request.generation, ns, request.tier,
                                    request.layerCount, request.visible);
        if (!alloc.valid) {
            std::cout << "[TextureLoaderV4] enqueueUpload REJECTED: pagePool_.allocate() returned invalid" << std::endl;
            return false;
        }
    }

    return pagePool_.upload(request.generation, alloc,
        request.albedoData,
        request.specularData,
        request.normalData,
        request.flagData,
        request.bytesPerLayer,
        request.channelMask,
        request.format,
        request.visible);
}

bool TextureLoaderV4::commitGeneration(uint64_t generation) {
    if (!initialized_.load(std::memory_order_acquire)) return false;
    if (generation != activeGeneration_.load(std::memory_order_acquire)) return false;

    std::lock_guard<std::mutex> lock(mutex_);
    generationCommitted_.store(true, std::memory_order_release);

    std::cout << "[TextureLoaderV4] Committed generation " << generation << std::endl;
    return true;
}

bool TextureLoaderV4::cancelGeneration(uint64_t generation, int reasonCode) {
    if (!initialized_.load(std::memory_order_acquire)) return false;

    uploadService_.cancelGeneration(generation);
    pagePool_.cancelGeneration(generation);

    std::cout << "[TextureLoaderV4] Cancelled generation " << generation
              << " reason=" << reasonCode << std::endl;
    return true;
}

void TextureLoaderV4::pump(uint64_t frameBudgetBytes) {
    if (!initialized_.load(std::memory_order_acquire)) return;
    uploadService_.pump(frameBudgetBytes);
    uploadService_.pollCompletions();
}

void TextureLoaderV4::pollCompletions() {
    if (!initialized_.load(std::memory_order_acquire)) return;
    uploadService_.pollCompletions();
}

bool TextureLoaderV4::generationIdle(uint64_t generation, bool visibleOnly) const {
    if (!initialized_.load(std::memory_order_acquire)) return true;
    return uploadService_.generationIdle(generation, visibleOnly);
}

std::string TextureLoaderV4::statusJson() const {
    std::ostringstream out;
    out << "{"
        << "\"schema\":\"radser_texture_loader_v4_status\","
        << "\"activeUploadMode\":\"texture_loader_v4\","
        << "\"abiVersion\":" << build_info::kTextureLoaderAbiVersion << ","
        << "\"cacheSchemaVersion\":" << build_info::kCacheSchemaVersion << ","
        << "\"generation\":" << activeGeneration_.load(std::memory_order_acquire) << ","
        << "\"committed\":" << (generationCommitted_.load(std::memory_order_acquire) ? "true" : "false") << ","
        << "\"javaPageContractWarnings\":0,"
        << "\"javaPageContractRejects\":" << javaPageContractRejects_.load(std::memory_order_relaxed) << ","
        << "\"vanillaBlockAtlasBypass\":true,"
        << "\"fixedCompatibilityUploadBytes\":0,"
        << "\"legacyFixedBlockUploadCalls\":0,"
        << "\"v4ActualVkCopyCommands\":" << uploadService_.status().actualVkCopyCommands << ","
        << "\"tieredArrays\":true,"
        << "\"auxPlaneUploadsAccepted\":true,"
        << "\"fourPlanePageUploads\":true,"
        << "\"shaderVisibleMaterialTableUpload\":false,"
        << "\"spriteRegistrySparseUpdates\":false,"
        << "\"ctmLoadGraphWorkItems\":false,"
        << "\"ctmTieredPages\":false,"
        << "\"mipGeneration\":false,"
        << "\"mip0Clamp\":true,"
        << "\"diskCacheEnabled\":true,"
        << "\"timeoutReadinessAllowed\":false,"
        << "\"uploadService\":" << uploadService_.statusJson() << ","
        << "\"pagePool\":" << pagePool_.statusJson()
        << "}";
    return out.str();
}

std::string TextureLoaderV4::tierStatusJson() const {
    return pagePool_.statusJson();
}

std::string TextureLoaderV4::uploadQueueStatusJson() const {
    return uploadService_.statusJson();
}

std::string TextureLoaderV4::pagePoolStatusJson() const {
    return pagePool_.statusJson();
}

std::string TextureLoaderV4::firstFrameReadinessJson(uint64_t generation) const {
    std::ostringstream out;
    bool idle = generationIdle(generation, true);
    auto uploadStatus = uploadService_.status();
    auto unreadyPages = pagePool_.unreadyAllocatedPageCount(generation);
    out << "{"
        << "\"schema\":\"radser_first_frame_native_readiness_v4\","
        << "\"generation\":" << generation << ","
        << "\"pendingVisibleUploadBytes\":" << uploadStatus.pendingVisibleUploadBytes << ","
        << "\"nativePendingMipPageCount\":" << pagePool_.pendingMipPageCount(generation, true) << ","
        << "\"nativeUnreadyAllocatedPageCount\":" << unreadyPages << ","
        << "\"nativeUnreadyAllocatedLayerCount\":" << pagePool_.unreadyAllocatedLayerCount(generation, true) << ","
        << "\"pendingVisibleMaterialTableUpdates\":0,"
        << "\"generationIdle\":" << (idle ? "true" : "false")
        << "}";
    return out.str();
}
