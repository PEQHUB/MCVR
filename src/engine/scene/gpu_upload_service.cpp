#include "gpu_upload_service.hpp"
#include "chunk_registry.hpp"
#include "platform/vulkan/vk2_device.hpp"
#include "frame/resource_gc.hpp"
#include "diagnostics/log.hpp"
#include "config/engine_config.hpp"

#include <volk.h>
#include <cstring>
#include <utility>

namespace engine {

vk2::Result<void> GpuUploadService::init(vk2::DeviceService& device, uint32_t framesInFlight,
                                         ResourceGC* gc, VkDeviceSize stagingSize) {
    device_ = &device;
    gc_ = gc;
    stagingSize_ = stagingSize;
    framesInFlight_ = std::max(framesInFlight, 1u);
    slotSize_ = stagingSize_ / framesInFlight_;

    // Create staging buffer (host-visible, persistently mapped)
    vk2::Buffer::Desc bd{};
    bd.size = stagingSize;
    bd.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bd.vmaFlags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
                | VMA_ALLOCATION_CREATE_MAPPED_BIT;

    auto r = vk2::Buffer::create(device.device(), device.vma(), bd);
    if (!r) return vk2::makeError(r.error().vkResult, "Staging buffer: " + r.error().message);
    stagingBuffer_ = std::move(r.value());

    initialized_ = true;
    log::info("gpu-upload", "Initialized with " + std::to_string(stagingSize / (1024 * 1024)) + " MB staging"
              + (gc_ ? " (deferred-delete: on)" : " (deferred-delete: OFF)"));
    return {};
}

void GpuUploadService::shutdown() {
    // Force synchronous destruction on shutdown — the frame loop is done and the GC
    // is about to be torn down, so any deferred lambdas would be stranded.
    chunks_.clear();
    stagingBuffer_ = {};
    initialized_ = false;
    gc_ = nullptr;
}

UploadResult GpuUploadService::uploadDirtyChunks(VkCommandBuffer cmd,
                                                   const std::vector<ChunkGeometry>& dirtyChunks,
                                                   uint32_t frameIndex) {
    UploadResult result;
    if (!initialized_ || dirtyChunks.empty()) return result;

    resetStaging(frameIndex);
    // BDA is always required: VK_KHR_ray_tracing_pipeline mandates bufferDeviceAddress.
    // The old conditional was wrong — if useBDA were ever false the vertex/index buffers
    // would be created without SHADER_DEVICE_ADDRESS_BIT, deviceAddress() would return 0,
    // and the TLAS would embed a null BDA for every instance, causing the RT shader to
    // dereference 0x0 and produce a GPU IP_FAULT / DEVICE_LOST.
    constexpr bool useBDA = true;

    for (size_t i = 0; i < dirtyChunks.size(); ++i) {
        const auto& geo = dirtyChunks[i];
        if (geo.empty()) {
            // Empty chunk — retire existing GPU data via GC (if present)
            auto it = chunks_.find(geo.id);
            if (it != chunks_.end()) {
                retireChunk(std::move(it->second));
                chunks_.erase(it);
            }
            continue;
        }

        VkDeviceSize vertSize = geo.vertexData.size();
        VkDeviceSize idxSize = geo.indexData.size() * sizeof(uint32_t);

        // Allocate staging space
        VkDeviceSize vertStaging = allocStaging(vertSize);
        VkDeviceSize idxStaging = allocStaging(idxSize);
        if (vertStaging == UINT64_MAX || idxStaging == UINT64_MAX) {
            // Staging ring is full — record ALL remaining ids (including this one)
            // as skipped so the caller can re-queue them. This prevents silent data
            // loss when the burst exceeds the per-frame slot capacity.
            result.skipped.reserve(dirtyChunks.size() - i);
            for (size_t j = i; j < dirtyChunks.size(); ++j) {
                if (!dirtyChunks[j].empty()) {
                    result.skipped.push_back(dirtyChunks[j].id);
                }
            }
            log::warn("gpu-upload", "Staging buffer full at chunk " + std::to_string(i)
                     + "/" + std::to_string(dirtyChunks.size())
                     + " — " + std::to_string(result.skipped.size()) + " chunks re-queued for next frame");
            break;
        }

        // Copy vertex data to staging
        auto* staging = static_cast<uint8_t*>(stagingBuffer_.mappedPtr());
        std::memcpy(staging + vertStaging, geo.vertexData.data(), vertSize);
        std::memcpy(staging + idxStaging, geo.indexData.data(), idxSize);

        // Create device-local vertex buffer
        VkBufferUsageFlags vertUsage = VK_BUFFER_USAGE_TRANSFER_DST_BIT
            | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
            | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
        if (useBDA) vertUsage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

        vk2::Buffer::Desc vbd{};
        vbd.size = vertSize;
        vbd.usage = vertUsage;
        auto vertR = vk2::Buffer::create(device_->device(), device_->vma(), vbd);
        if (!vertR) {
            log::warn("gpu-upload", "Vertex buffer alloc failed for chunk — re-queueing for retry");
            result.skipped.push_back(geo.id);
            continue;
        }

        // Create device-local index buffer
        VkBufferUsageFlags idxUsage = VK_BUFFER_USAGE_TRANSFER_DST_BIT
            | VK_BUFFER_USAGE_INDEX_BUFFER_BIT
            | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
            | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
        if (useBDA) idxUsage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

        vk2::Buffer::Desc ibd{};
        ibd.size = idxSize;
        ibd.usage = idxUsage;
        auto idxR = vk2::Buffer::create(device_->device(), device_->vma(), ibd);
        if (!idxR) {
            log::warn("gpu-upload", "Index buffer alloc failed for chunk — re-queueing for retry");
            result.skipped.push_back(geo.id);
            continue;
        }

        // Record copy commands
        VkBufferCopy vertCopy{vertStaging, 0, vertSize};
        vkCmdCopyBuffer(cmd, stagingBuffer_.handle(), vertR.value().handle(), 1, &vertCopy);

        VkBufferCopy idxCopy{idxStaging, 0, idxSize};
        vkCmdCopyBuffer(cmd, stagingBuffer_.handle(), idxR.value().handle(), 1, &idxCopy);

        // Retire the old entry before overwriting so in-flight GPU reads finish first.
        auto existing = chunks_.find(geo.id);
        if (existing != chunks_.end()) {
            retireChunk(std::move(existing->second));
            chunks_.erase(existing);
        }

        // Store GPU data
        GpuChunkData data;
        data.triangleCount = geo.triangleCount;
        data.indexCount = static_cast<uint32_t>(geo.indexData.size());
        data.revision = geo.revision;
        data.originX = geo.originX;
        data.originY = geo.originY;
        data.originZ = geo.originZ;
        data.vertexAddress = vertR.value().deviceAddress();
        data.indexAddress = idxR.value().deviceAddress();
        data.vertexBuffer = std::move(vertR.value());
        data.indexBuffer = std::move(idxR.value());

        chunks_[geo.id] = std::move(data);

        // Structured diagnostic event: upload_complete (gated on DiagFlags::UPLOAD)
        if ((diagFlags_ & DiagFlags::UPLOAD) != 0) {
            auto toHex = [](uint64_t v) -> std::string {
                char buf[20]; std::snprintf(buf, sizeof(buf), "0x%llx", static_cast<unsigned long long>(v));
                return buf;
            };
            const auto& stored = chunks_[geo.id];
            log::event("gpu-upload", "upload_complete", {
                {"chunkX",      std::to_string(geo.id.x)},
                {"chunkY",      std::to_string(geo.id.y)},
                {"chunkZ",      std::to_string(geo.id.z)},
                {"vertexBda",   toHex(stored.vertexAddress)},
                {"indexBda",    toHex(stored.indexAddress)},
                {"vertexBytes", std::to_string(vertSize)},
                {"indexBytes",  std::to_string(idxSize)},
                {"tris",        std::to_string(geo.triangleCount)},
                {"stagingSlot", std::to_string(frameIndex % framesInFlight_)}
            });
        }

        // BDA trace: per-chunk address validation (gated on DiagFlags::BDA)
        if ((diagFlags_ & DiagFlags::BDA) != 0) {
            auto toHex = [](uint64_t v) -> std::string {
                char buf[20]; std::snprintf(buf, sizeof(buf), "0x%llx", static_cast<unsigned long long>(v));
                return buf;
            };
            const auto& stored = chunks_[geo.id];
            log::trace("gpu-upload", "[BDA] upload chunk ("
                + std::to_string(geo.id.x) + "," + std::to_string(geo.id.y) + "," + std::to_string(geo.id.z)
                + ") vertexBda=" + toHex(stored.vertexAddress)
                + " indexBda=" + toHex(stored.indexAddress)
                + (stored.vertexAddress == 0 || stored.indexAddress == 0 ? " *** NULL BDA ***" : ""));
        }

        ++result.uploaded;
    }

    // Flush the staging buffer once after all chunks are written.
    // This ensures CPU writes are visible to the GPU transfer engine on non-coherent heaps
    // (vmaFlushAllocation is a no-op on HOST_COHERENT memory, so always safe to call).
    if (result.uploaded > 0) {
        stagingBuffer_.flush();
    }

    return result;
}

void GpuUploadService::removeChunk(ChunkId id) {
    auto it = chunks_.find(id);
    if (it == chunks_.end()) return;
    retireChunk(std::move(it->second));
    chunks_.erase(it);
}

void GpuUploadService::clearAll() {
    for (auto& [id, data] : chunks_) {
        retireChunk(std::move(data));
    }
    chunks_.clear();
}

const GpuChunkData* GpuUploadService::getChunk(ChunkId id) const {
    auto it = chunks_.find(id);
    return it != chunks_.end() ? &it->second : nullptr;
}

VkDeviceSize GpuUploadService::allocStaging(VkDeviceSize size, VkDeviceSize alignment) {
    // Align up
    VkDeviceSize aligned = (stagingOffset_ + alignment - 1) & ~(alignment - 1);
    if (aligned + size > slotBase_ + slotSize_) return UINT64_MAX;
    stagingOffset_ = aligned + size;
    return aligned;
}

void GpuUploadService::resetStaging(uint32_t frameIndex) {
    slotBase_ = (frameIndex % framesInFlight_) * slotSize_;
    stagingOffset_ = slotBase_;
}

void GpuUploadService::retireChunk(GpuChunkData data) {
    if (gc_) {
        // Wrap in shared_ptr so the lambda is copyable (std::function requirement).
        // The buffers are destroyed on GC tick() after the ring cycles past this slot,
        // guaranteeing in-flight GPU work has completed.
        auto vb = std::make_shared<vk2::Buffer>(std::move(data.vertexBuffer));
        auto ib = std::make_shared<vk2::Buffer>(std::move(data.indexBuffer));
        gc_->defer([vb, ib]() {
            // shared_ptr release — vk2::Buffer dtors run if this was the last ref
        });
    }
    // If no GC wired, the buffers get destroyed by `data`'s destructor when it goes
    // out of scope at the end of this function (legacy synchronous fallback).
}

} // namespace engine
