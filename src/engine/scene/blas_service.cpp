#include "blas_service.hpp"
#include "gpu_upload_service.hpp"
#include "platform/vulkan/vk2_device.hpp"
#include "frame/resource_gc.hpp"
#include "diagnostics/log.hpp"
#include "config/engine_config.hpp"

#include <utility>
#include <volk.h>

namespace engine {

BlasService::~BlasService() { shutdown(); }

vk2::Result<void> BlasService::init(vk2::DeviceService& device, ResourceGC* gc,
                                     VkDeviceSize scratchSize, uint32_t framesInFlight) {
    device_ = &device;
    gc_ = gc;
    scratchSize_ = scratchSize;
    (void)framesInFlight; // always 2; kept in signature for documentation clarity

    if (!device.caps().accelerationStructure) {
        log::warn("blas", "Acceleration structures not supported — BLAS service disabled");
        return {};
    }

    // Create double-buffered scratch buffers (one per frame-in-flight slot).
    // This prevents Frame N-1's GPU BLAS build from aliasing Frame N's scratch:
    // Frame N uses slot (N & 1), and the fence wait guarantees Frame N-2
    // (same slot, previous use) has completed before we reuse its scratch region.
    vk2::Buffer::Desc bd{};
    bd.size = scratchSize;
    bd.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    bd.minAlignment = 256; // AS scratch alignment requirement

    for (uint32_t s = 0; s < 2; ++s) {
        auto r = vk2::Buffer::create(device.device(), device.vma(), bd);
        if (!r) return vk2::makeError(r.error().vkResult,
            "BLAS scratch buffer[" + std::to_string(s) + "]: " + r.error().message);
        scratchBuffers_[s] = std::move(r.value());
        scratchAddresses_[s] = scratchBuffers_[s].deviceAddress();
    }

    initialized_ = true;
    log::info("blas", "Initialized with " + std::to_string(scratchSize / (1024 * 1024)) + " MB scratch"
              + (gc_ ? " (deferred-delete: on)" : " (deferred-delete: OFF)"));
    return {};
}

void BlasService::shutdown() {
    // On shutdown, force synchronous destruction — the frame loop is done, no GPU work in flight.
    // Bypass the GC path even if it's wired, since the GC itself is about to be torn down.
    for (auto& [id, data] : blas_) {
        if (data.handle != VK_NULL_HANDLE && device_) {
            vkDestroyAccelerationStructureKHR(device_->device(), data.handle, nullptr);
            data.handle = VK_NULL_HANDLE;
        }
        // vk2::Buffer destructor runs when `data` goes out of scope below.
    }
    blas_.clear();
    scratchBuffers_[0] = {};
    scratchBuffers_[1] = {};
    initialized_ = false;
    gc_ = nullptr;
}

BlasBuildResult BlasService::buildDirtyChunks(VkCommandBuffer cmd,
                                                const std::vector<ChunkId>& dirtyIds,
                                                const std::unordered_map<ChunkId, GpuChunkData>& gpuChunks,
                                                uint32_t frameIndex) {
    BlasBuildResult result;
    if (!initialized_ || !device_->caps().accelerationStructure) return result;

    // Select the double-buffered scratch slot for this frame.
    // fence[frameIndex & 1] was waited before this call, guaranteeing Frame N-2's
    // GPU BLAS builds (same slot) are complete. Frame N-1 uses the other slot.
    const uint32_t slot = frameIndex & 1;
    const VkDeviceAddress scratchAddress = scratchAddresses_[slot];

    VkDeviceSize scratchOffset = 0;

    for (size_t i = 0; i < dirtyIds.size(); ++i) {
        const auto& chunkId = dirtyIds[i];
        auto it = gpuChunks.find(chunkId);
        if (it == gpuChunks.end() || it->second.triangleCount == 0) continue;

        const auto& gpu = it->second;

        // Geometry description
        VkAccelerationStructureGeometryKHR geometry{};
        geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
        geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;

        // OMM geometry flag handling:
        //   When ommEnabled_ is true we OMIT VK_GEOMETRY_OPAQUE_BIT_KHR so that
        //   any-hit shaders fire per-ray for alpha testing. This is the correct V2
        //   fallback for opacity micromaps: alpha texture data is stored GPU-side
        //   after CmdTextureFinalize and is not accessible CPU-side at BLAS build
        //   time, so we cannot call OMMBaker or attach VkAccelerationStructureTrianglesOpacityMicromapEXT
        //   here. Dropping the OPAQUE flag costs some RT throughput but keeps
        //   alpha-tested geometry (leaves, fences, glass panes) visually correct.
        //   This mirrors V1's behaviour for chunks still awaiting OMM bake.
        //
        //   When ommEnabled_ is false (default), mark all geometry opaque — any-hit
        //   shaders are skipped and RT performance is maximised.
        if (ommEnabled_) {
            geometry.flags = 0;  // No OPAQUE flag — any-hit fires for alpha testing
            log::debug("blas", "OMM fallback active: geometry built without OPAQUE flag (any-hit enabled)");
        } else {
            geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
        }

        geometry.geometry.triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
        geometry.geometry.triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
        geometry.geometry.triangles.vertexData.deviceAddress = gpu.vertexAddress;
        geometry.geometry.triangles.vertexStride = 96; // PBRTriangle stride
        geometry.geometry.triangles.maxVertex = static_cast<uint32_t>(gpu.vertexBuffer.size() / 96) - 1;
        geometry.geometry.triangles.indexType = VK_INDEX_TYPE_UINT32;
        geometry.geometry.triangles.indexData.deviceAddress = gpu.indexAddress;

        // Build info
        VkAccelerationStructureBuildGeometryInfoKHR buildInfo{};
        buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
        buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
        buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        buildInfo.geometryCount = 1;
        buildInfo.pGeometries = &geometry;

        // Query size requirements
        VkAccelerationStructureBuildSizesInfoKHR sizeInfo{};
        sizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
        uint32_t primCount = gpu.triangleCount;
        vkGetAccelerationStructureBuildSizesKHR(device_->device(),
            VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &buildInfo, &primCount, &sizeInfo);

        // Check scratch fits
        VkDeviceSize alignedScratch = (scratchOffset + 255) & ~255ULL;
        if (alignedScratch + sizeInfo.buildScratchSize > scratchSize_) {
            // Scratch exhausted — re-queue this id and all remaining ids so they
            // get rebuilt next frame. Without this, uploaded chunks that skipped
            // BLAS construction remain silently invisible to the RT pipeline
            // (present in the registry + gpu upload map, but no AS instance).
            result.skipped.reserve(dirtyIds.size() - i);
            for (size_t j = i; j < dirtyIds.size(); ++j) {
                auto it2 = gpuChunks.find(dirtyIds[j]);
                if (it2 != gpuChunks.end() && it2->second.triangleCount > 0) {
                    result.skipped.push_back(dirtyIds[j]);

                    // CRITICAL: if this chunk was re-uploaded (new device-local vertex buffer),
                    // the existing BlasData still holds the OLD vertexAddress/indexAddress.
                    // The old device-local buffer is now deferred to GC and will be freed in
                    // ~32 frames. The TLAS reads BlasData.vertexAddress to populate
                    // vertexBdaBuffers_[slot] — so update it to the new address NOW, before the
                    // TLAS builds this frame. The BLAS AS itself is re-queued for rebuild, so the
                    // existing BLAS still describes the OLD geometry (fine for ray traversal
                    // direction/hit — minor visual artifact) but the RT shader's vertex fetch will
                    // use the correct (live) device-local buffer address.
                    auto blasIt = blas_.find(dirtyIds[j]);
                    if (blasIt != blas_.end()) {
                        blasIt->second.vertexAddress = it2->second.vertexAddress;
                        blasIt->second.indexAddress  = it2->second.indexAddress;
                    }
                }
            }
            log::warn("blas", "Scratch buffer full at chunk " + std::to_string(i)
                     + "/" + std::to_string(dirtyIds.size())
                     + " — " + std::to_string(result.skipped.size()) + " chunks re-queued for next frame");
            break;
        }

        // Create AS buffer
        vk2::Buffer::Desc asBd{};
        asBd.size = sizeInfo.accelerationStructureSize;
        asBd.usage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR
                   | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        auto asR = vk2::Buffer::create(device_->device(), device_->vma(), asBd);
        if (!asR) {
            log::warn("blas", "AS buffer alloc failed — re-queueing for retry");
            // CRITICAL: if this chunk was re-uploaded, the existing BlasData still holds
            // the OLD vertexAddress/indexAddress which points to a GC-deferred (soon freed)
            // device-local buffer. Update the BDAs to the live GPU buffer NOW so the TLAS
            // built this frame does not embed a dangling BDA → RT shader null dereference.
            {
                auto blasIt = blas_.find(chunkId);
                if (blasIt != blas_.end()) {
                    blasIt->second.vertexAddress = gpu.vertexAddress;
                    blasIt->second.indexAddress  = gpu.indexAddress;
                }
            }
            result.skipped.push_back(chunkId);
            continue;
        }

        // Create AS
        VkAccelerationStructureCreateInfoKHR asCI{};
        asCI.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
        asCI.buffer = asR.value().handle();
        asCI.size = sizeInfo.accelerationStructureSize;
        asCI.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;

        VkAccelerationStructureKHR as = VK_NULL_HANDLE;
        if (vkCreateAccelerationStructureKHR(device_->device(), &asCI, nullptr, &as) != VK_SUCCESS) {
            log::warn("blas", "vkCreateAccelerationStructureKHR failed — re-queueing for retry");
            // CRITICAL: same dangling-BDA fix as the AS buffer alloc failure path above.
            {
                auto blasIt = blas_.find(chunkId);
                if (blasIt != blas_.end()) {
                    blasIt->second.vertexAddress = gpu.vertexAddress;
                    blasIt->second.indexAddress  = gpu.indexAddress;
                }
            }
            result.skipped.push_back(chunkId);
            continue;
        }

        // Get device address
        VkAccelerationStructureDeviceAddressInfoKHR addrInfo{};
        addrInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
        addrInfo.accelerationStructure = as;
        VkDeviceAddress asAddr = vkGetAccelerationStructureDeviceAddressKHR(device_->device(), &addrInfo);

        // Guard: driver returned 0 for the AS device address.
        // vkGetAccelerationStructureDeviceAddressKHR has no error return — a zero result
        // indicates the driver could not produce a valid address (feature not enabled, or
        // internal driver failure). Storing this into BlasData and then emitting it into
        // the TLAS would cause the RT shader to traverse an invalid AS, producing a GPU
        // page fault. Skip this chunk and re-queue; the next frame will retry.
        if (asAddr == 0) {
            log::warn("blas", "vkGetAccelerationStructureDeviceAddressKHR returned 0 — "
                              "skipping chunk BLAS to prevent GPU fault (re-queued for retry)");
            vkDestroyAccelerationStructureKHR(device_->device(), as, nullptr);
            result.skipped.push_back(chunkId);
            continue;
        }

        // Build
        buildInfo.dstAccelerationStructure = as;
        buildInfo.scratchData.deviceAddress = scratchAddress + alignedScratch;

        VkAccelerationStructureBuildRangeInfoKHR rangeInfo{};
        rangeInfo.primitiveCount = primCount;
        const VkAccelerationStructureBuildRangeInfoKHR* pRange = &rangeInfo;

        nvInsertCheckpoint(cmd, (diagFlags_ & DiagFlags::CRASH) != 0,
                           device_->caps().nvCheckpoints, "BLAS_BUILD_START");
        vkCmdBuildAccelerationStructuresKHR(cmd, 1, &buildInfo, &pRange);

        scratchOffset = alignedScratch + sizeInfo.buildScratchSize;

        // Retire old BLAS (deferred if GC is wired). We move-out the old data so the
        // in-flight destructor captures by value and the map slot is immediately free
        // for the new entry.
        auto existing = blas_.find(chunkId);
        if (existing != blas_.end()) {
            retireBlas(std::move(existing->second));
            blas_.erase(existing);
        }

        // Store
        BlasData data;
        data.handle = as;
        data.buffer = std::move(asR.value());
        data.address = asAddr;
        data.vertexAddress = gpu.vertexAddress;
        data.indexAddress = gpu.indexAddress;
        data.originX = gpu.originX;
        data.originY = gpu.originY;
        data.originZ = gpu.originZ;
        data.revision = gpu.revision;
        blas_[chunkId] = std::move(data);

        // Structured diagnostic event: blas_built (gated on DiagFlags::BLAS)
        if ((diagFlags_ & DiagFlags::BLAS) != 0) {
            auto toHex = [](uint64_t v) -> std::string {
                char buf[20]; std::snprintf(buf, sizeof(buf), "0x%llx", static_cast<unsigned long long>(v));
                return buf;
            };
            const auto& stored = blas_[chunkId];
            log::event("blas", "blas_built", {
                {"chunkX",        std::to_string(chunkId.x)},
                {"chunkY",        std::to_string(chunkId.y)},
                {"chunkZ",        std::to_string(chunkId.z)},
                {"asAddr",        toHex(stored.address)},
                {"vertexBda",     toHex(stored.vertexAddress)},
                {"indexBda",      toHex(stored.indexAddress)},
                {"scratchOffset", std::to_string(alignedScratch)},
                {"tris",          std::to_string(gpu.triangleCount)}
            });
        }

        // BDA trace: per-BLAS address validation (gated on DiagFlags::BDA)
        if ((diagFlags_ & DiagFlags::BDA) != 0) {
            auto toHex = [](uint64_t v) -> std::string {
                char buf[20]; std::snprintf(buf, sizeof(buf), "0x%llx", static_cast<unsigned long long>(v));
                return buf;
            };
            const auto& stored = blas_[chunkId];
            log::trace("blas", "[BDA] built chunk ("
                + std::to_string(chunkId.x) + "," + std::to_string(chunkId.y) + "," + std::to_string(chunkId.z)
                + ") asAddr=" + toHex(stored.address)
                + " vertexBda=" + toHex(stored.vertexAddress)
                + " indexBda=" + toHex(stored.indexAddress)
                + (stored.address == 0 ? " *** NULL AS_ADDR ***" : ""));
        }

        ++result.built;
    }

    // Memory barrier after all builds
    if (result.built > 0) {
        VkMemoryBarrier2 barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        barrier.srcStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
        barrier.srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR
                             | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
        barrier.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;

        VkDependencyInfo dep{};
        dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.memoryBarrierCount = 1;
        dep.pMemoryBarriers = &barrier;
        vkCmdPipelineBarrier2(cmd, &dep);
    }

    return result;
}

void BlasService::removeChunk(ChunkId id) {
    auto it = blas_.find(id);
    if (it != blas_.end()) {
        retireBlas(std::move(it->second));
        blas_.erase(it);
    }
}

void BlasService::clearAll() {
    for (auto& [id, data] : blas_) {
        retireBlas(std::move(data));
    }
    blas_.clear();
}

const BlasData* BlasService::getBlas(ChunkId id) const {
    auto it = blas_.find(id);
    return it != blas_.end() ? &it->second : nullptr;
}

void BlasService::retireBlas(BlasData data) {
    if (data.handle == VK_NULL_HANDLE) return;

    if (gc_) {
        // Defer destruction: wrap the AS handle and buffer in a shared state so
        // the capture is copyable (std::function requires CopyConstructible).
        // The state is destroyed when the GC ring cycles past the lambda's slot,
        // guaranteeing the GPU is no longer reading the resources.
        VkDevice dev = device_->device();
        VkAccelerationStructureKHR as = data.handle;
        auto buf = std::make_shared<vk2::Buffer>(std::move(data.buffer));
        gc_->defer([dev, as, buf]() {
            if (as != VK_NULL_HANDLE) {
                vkDestroyAccelerationStructureKHR(dev, as, nullptr);
            }
            // buf's shared_ptr releases here; if last reference, vk2::Buffer dtor runs
        });
    } else {
        // Synchronous fallback (used at shutdown)
        if (device_) {
            vkDestroyAccelerationStructureKHR(device_->device(), data.handle, nullptr);
        }
        // data.buffer destructor runs when `data` goes out of scope
    }
}

} // namespace engine
