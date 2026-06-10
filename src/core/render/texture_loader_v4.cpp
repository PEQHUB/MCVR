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
    if (!initialized_.load(std::memory_order_acquire)) return false;
    if (request.generation != activeGeneration_.load(std::memory_order_acquire)) return false;
    if (request.generation == 0) return false;
    if (request.albedoData == nullptr || request.bytesPerLayer == 0) return false;
    if (request.tier >= TexturePagePool::kMaxTiers) return false;
    if (request.layerCount == 0) return false;
    if (request.layerCapacity == 0) return false;
    if (request.startLayer >= request.layerCapacity
        || request.layerCount > request.layerCapacity - request.startLayer) {
        return false;
    }

    const uint32_t expectedSize = request.tier < TexturePagePool::kMaxTiers
        ? (16u << request.tier)
        : 0u;
    if (expectedSize == 0 || request.width != expectedSize || request.height != expectedSize) {
        return false;
    }
    const uint64_t expectedBytesPerLayer = static_cast<uint64_t>(expectedSize) * expectedSize * 4u;
    if (request.bytesPerLayer < expectedBytesPerLayer) return false;

    // Allocate page pool layers. Native chooses page/layer placement; Java hints are accepted
    // only as diagnostics until sparse registry publication consumes the returned handles.
    auto ns = static_cast<TexturePagePool::Namespace>(request.namespaceId);
    auto alloc = pagePool_.allocate(request.generation, ns, request.tier,
                                     request.layerCount, request.visible);
    if (!alloc.valid) return false;

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
        << "\"vanillaBlockAtlasBypass\":true,"
        << "\"fixedCompatibilityUploadBytes\":0,"
        << "\"legacyFixedBlockUploadCalls\":0,"
        << "\"v4ActualVkCopyCommands\":" << uploadService_.status().actualVkCopyCommands << ","
        << "\"tieredArrays\":true,"
        << "\"fourPlanePageUploads\":true,"
        << "\"shaderVisibleMaterialTableUpload\":true,"
        << "\"ctmTieredPages\":true,"
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
