#include "tlas_service.hpp"
#include "blas_service.hpp"
#include "entity_blas_service.hpp"
#include "platform/vulkan/vk2_device.hpp"
#include "frame/resource_gc.hpp"
#include "diagnostics/log.hpp"
#include "config/engine_config.hpp"

#include <volk.h>
#include <algorithm>
#include <cstring>
#include <utility>
#include <vector>

namespace engine {

TlasService::~TlasService() { shutdown(); }

vk2::Result<void> TlasService::init(vk2::DeviceService& device, ResourceGC* gc) {
    device_ = &device;
    gc_ = gc;

    if (!device.caps().accelerationStructure) {
        log::warn("tlas", "Acceleration structures not supported — TLAS service disabled");
        return {};
    }

    initialized_ = true;
    log::info("tlas", std::string("TLAS service initialized")
              + (gc_ ? " (deferred-delete: on)" : " (deferred-delete: OFF)"));
    return {};
}

void TlasService::shutdown() {
    // On shutdown, force synchronous destruction — frame loop is done, GC is about to die.
    for (int i = 0; i < 2; ++i) {
        if (tlas_[i] != VK_NULL_HANDLE && device_) {
            vkDestroyAccelerationStructureKHR(device_->device(), tlas_[i], nullptr);
        }
        tlas_[i] = VK_NULL_HANDLE;
        tlasAddress_[i] = 0;
        tlasBuffer_[i] = {};
    }
    instanceBuffers_[0] = {}; instanceBuffers_[1] = {};
    scratchBuffer_ = {};
    vertexBdaBuffers_[0] = {}; vertexBdaBuffers_[1] = {};
    indexBdaBuffers_[0] = {};  indexBdaBuffers_[1] = {};
    blasOffsetsBuffers_[0] = {}; blasOffsetsBuffers_[1] = {};
    bdaInstanceCounts_[0] = bdaInstanceCounts_[1] = 0;
    bdaCapacity_[0] = bdaCapacity_[1] = 0;
    lastVertexBdaBuffers_[0] = {}; lastVertexBdaBuffers_[1] = {};
    lastIndexBdaBuffers_[0] = {};  lastIndexBdaBuffers_[1] = {};
    lastTransformsBuffers_[0] = {}; lastTransformsBuffers_[1] = {};
    lastBdaCapacities_[0] = lastBdaCapacities_[1] = 0;
    lastInstanceCounts_[0] = lastInstanceCounts_[1] = 0;
    instanceCount_ = 0;
    initialized_ = false;
    gc_ = nullptr;
}

bool TlasService::rebuild(VkCommandBuffer cmd, const BlasService& blasService,
                          uint32_t frameIndex,
                          const EntityBlasService* entityService,
                          const CameraData* camera) {
    const uint32_t slot = frameIndex & 1;
    if (!initialized_ || !device_->caps().accelerationStructure) return false;
    if (blasService.blasCount() == 0 && (!entityService || entityService->entityCount() == 0)) return false;

    // Collect instances + parallel BDA arrays.
    // instanceCustomIndex = position in vertexBdas/indexBdas, so the closest-hit
    // shader can use gl_InstanceCustomIndexEXT to find this chunk's vertex/index buffers.
    //
    // The section origin is baked into the 3x4 row-major transform as a pure translation:
    //   | 1 0 0 ox |
    //   | 0 1 0 oy |
    //   | 0 0 1 oz |
    // so the RT hardware transforms the BLAS's section-local vertices into world space
    // on every ray-triangle intersection, and gl_WorldRayDirectionEXT / gl_WorldRayOriginEXT
    // in shaders are properly matched to the vertex data.
    std::vector<VkAccelerationStructureInstanceKHR> instances;
    std::vector<uint64_t> vertexBdas;
    std::vector<uint64_t> indexBdas;
    uint32_t totalCount = blasService.blasCount();
    if (entityService) totalCount += entityService->entityCount();
    instances.reserve(totalCount);
    vertexBdas.reserve(totalCount);
    indexBdas.reserve(totalCount);

    uint32_t nextCustomIdx = 0;

    // --- Entity instances (inserted BEFORE chunks for lower custom indices) ---
    if (entityService && entityService->entityCount() > 0) {
        for (const auto& ent : entityService->entities()) {
            VkAccelerationStructureInstanceKHR inst{};

            // Compute transform based on coordSystem
            if (ent.coordSystem == 0) {
                // WORLD: identity rotation + entity position (camera-relative)
                float ex = ent.posX;
                float ey = ent.posY;
                float ez = ent.posZ;
                if (camera && camera->valid) {
                    ex -= static_cast<float>(camera->posX);
                    ey -= static_cast<float>(camera->posY);
                    ez -= static_cast<float>(camera->posZ);
                }
                inst.transform.matrix[0][0] = 1.0f;
                inst.transform.matrix[0][3] = ex;
                inst.transform.matrix[1][1] = 1.0f;
                inst.transform.matrix[1][3] = ey;
                inst.transform.matrix[2][2] = 1.0f;
                inst.transform.matrix[2][3] = ez;
            } else if (ent.coordSystem == 1) {
                // CAMERA: inverse view matrix (already in camera space)
                if (camera && camera->valid) {
                    // View matrix is column-major 4x4. The 3x3 rotation part transposed
                    // gives the inverse rotation. Translation is already handled by
                    // the camera-relative origin convention.
                    const float* v = camera->view;
                    // Row 0 of inverse = column 0 of view
                    inst.transform.matrix[0][0] = v[0]; inst.transform.matrix[0][1] = v[4]; inst.transform.matrix[0][2] = v[8];
                    inst.transform.matrix[1][0] = v[1]; inst.transform.matrix[1][1] = v[5]; inst.transform.matrix[1][2] = v[9];
                    inst.transform.matrix[2][0] = v[2]; inst.transform.matrix[2][1] = v[6]; inst.transform.matrix[2][2] = v[10];
                    // No translation — camera-space entities are at camera origin
                    inst.transform.matrix[0][3] = 0.0f;
                    inst.transform.matrix[1][3] = 0.0f;
                    inst.transform.matrix[2][3] = 0.0f;
                } else {
                    inst.transform.matrix[0][0] = 1.0f;
                    inst.transform.matrix[1][1] = 1.0f;
                    inst.transform.matrix[2][2] = 1.0f;
                }
            } else {
                // CAMERA_SHIFT: identity rotation + view translation
                // Particles rendered at camera origin with camera-relative positions
                inst.transform.matrix[0][0] = 1.0f;
                inst.transform.matrix[1][1] = 1.0f;
                inst.transform.matrix[2][2] = 1.0f;
                // No additional translation — positions already camera-relative in vertex data
            }

            // Safety: skip any entity BLAS with a null AS address or null BDA.
            // A zero address means the BLAS build failed or BDA was not queried;
            // emitting it into the TLAS would cause the RT shader to dereference 0x0
            // via gl_InstanceCustomIndexEXT → null GPU pointer → IP_FAULT / DEVICE_LOST.
            if (ent.address == 0 || ent.vertexAddress == 0 || ent.indexAddress == 0) {
                log::warn("tlas", "Entity BLAS has null address/BDA — skipping to prevent GPU fault");
                continue;
            }

            inst.instanceCustomIndex = nextCustomIdx;
            inst.mask = 0xFF;
            // SBT record offset 0 is correct: the shaders use sbtRecordOffset=0 for shadow rays
            // (lands on hit group 0 = shadow_rchit) and sbtRecordOffset=1 for primary/bounce rays
            // (lands on hit group 1 = world_solid_transparent_rchit). Setting instanceOffset=1
            // shifts both by 1, sending shadow rays to the wrong hit group and primary rays
            // out-of-range, producing a persistent instruction-fetch DEVICE_FAULT.
            inst.instanceShaderBindingTableRecordOffset = 0;
            inst.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
            inst.accelerationStructureReference = ent.address;
            instances.push_back(inst);

            vertexBdas.push_back(ent.vertexAddress);
            indexBdas.push_back(ent.indexAddress);

            ++nextCustomIdx;
        }
    }

    blasService.forEachBlas([&](const ChunkId& id, const BlasData& blas) {
        // Safety: skip any chunk BLAS with a null AS address or null vertex/index BDA.
        // This can occur when buffers were created without SHADER_DEVICE_ADDRESS_BIT
        // or when a BLAS build failed silently. Emitting a zero BDA into the TLAS
        // causes the RT shader to dereference 0x0 via gl_InstanceCustomIndexEXT,
        // producing a GPU IP_FAULT (type=6) and DEVICE_LOST within ~2 seconds.
        if (blas.address == 0 || blas.vertexAddress == 0 || blas.indexAddress == 0) {
            log::warn("tlas", "Chunk BLAS has null address/BDA — skipping instance to prevent GPU fault");
            return;
        }

        VkAccelerationStructureInstanceKHR inst{};
        // Translation transform: identity rotation, origin translation in column [3].
        inst.transform.matrix[0][0] = 1.0f;
        inst.transform.matrix[0][3] = blas.originX;
        inst.transform.matrix[1][1] = 1.0f;
        inst.transform.matrix[1][3] = blas.originY;
        inst.transform.matrix[2][2] = 1.0f;
        inst.transform.matrix[2][3] = blas.originZ;
        inst.instanceCustomIndex = nextCustomIdx;
        inst.mask = 0xFF;
        // SBT record offset 0: shadow rays use sbtRecordOffset=0 → hit group 0 (shadow_rchit),
        // primary/bounce rays use sbtRecordOffset=1 → hit group 1 (world_solid_transparent_rchit).
        inst.instanceShaderBindingTableRecordOffset = 0;
        inst.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
        inst.accelerationStructureReference = blas.address;
        instances.push_back(inst);

        vertexBdas.push_back(blas.vertexAddress);
        indexBdas.push_back(blas.indexAddress);

        ++nextCustomIdx;
    });

    if (instances.empty()) return false;

    // ------------------------------------------------------------------
    // Snapshot previous-slot BDA data into THIS slot's "last" buffers.
    //
    // Layout: lastVertexBdaBuffers_[slot] holds the snapshot of prevSlot's data
    // for use as "previous frame" in motion vector calculations.
    //
    // Safety: rebuild() runs after fence[slot] wait, which guarantees Frame N-2
    // (the last time this slot was submitted) has completed. Writing into
    // lastXxxBuffers_[slot] is therefore safe — no in-flight GPU work references it.
    // (Frame N-1 used slot^1 and wrote into lastXxxBuffers_[slot^1], which we
    //  are NOT touching here.)
    // ------------------------------------------------------------------
    {
        const uint32_t prevSlot = slot ^ 1;  // the slot written last frame
        uint32_t prevCount = bdaInstanceCounts_[prevSlot];
        if (prevCount > 0 && vertexBdaBuffers_[prevSlot].handle()) {
            VkDeviceSize bdaBytes = prevCount * sizeof(uint64_t);

            // (Re)allocate this slot's last-frame buffers if capacity is insufficient.
            // Immediate destruction is safe: slot's fence wait guarantees Frame N-2 is done.
            // No GC deferral needed because we're the only writer for this slot.
            if (lastBdaCapacities_[slot] < prevCount) {
                VkDeviceSize newCap = std::max<VkDeviceSize>(prevCount * 2, 64);
                vk2::Buffer::Desc bd{};
                bd.size = newCap * sizeof(uint64_t);
                bd.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
                bd.vmaFlags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
                            | VMA_ALLOCATION_CREATE_MAPPED_BIT;

                auto rv = vk2::Buffer::create(device_->device(), device_->vma(), bd);
                if (rv) lastVertexBdaBuffers_[slot] = std::move(rv.value());

                auto ri = vk2::Buffer::create(device_->device(), device_->vma(), bd);
                if (ri) lastIndexBdaBuffers_[slot] = std::move(ri.value());

                // Transforms: float[12] per instance = 48 bytes each
                vk2::Buffer::Desc td{};
                td.size = newCap * sizeof(float) * 12;
                td.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
                td.vmaFlags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
                            | VMA_ALLOCATION_CREATE_MAPPED_BIT;
                auto rt = vk2::Buffer::create(device_->device(), device_->vma(), td);
                if (rt) lastTransformsBuffers_[slot] = std::move(rt.value());

                lastBdaCapacities_[slot] = newCap;
            }

            // Copy prevSlot BDA arrays -> this slot's last buffers.
            // Safe: fence[slot] waited above guarantees Frame N-2 is done reading these.
            if (lastVertexBdaBuffers_[slot].mappedPtr() && vertexBdaBuffers_[prevSlot].mappedPtr()) {
                std::memcpy(lastVertexBdaBuffers_[slot].mappedPtr(), vertexBdaBuffers_[prevSlot].mappedPtr(), bdaBytes);
                lastVertexBdaBuffers_[slot].flush();
            }
            if (lastIndexBdaBuffers_[slot].mappedPtr() && indexBdaBuffers_[prevSlot].mappedPtr()) {
                std::memcpy(lastIndexBdaBuffers_[slot].mappedPtr(), indexBdaBuffers_[prevSlot].mappedPtr(), bdaBytes);
                lastIndexBdaBuffers_[slot].flush();
            }

            // Extract per-instance transforms from the instance buffer.
            // VkAccelerationStructureInstanceKHR has a VkTransformMatrixKHR at offset 0
            // which is a 3x4 row-major float matrix (12 floats = 48 bytes).
            // Use prevSlot's instance buffer (fence[slot] waited, so Frame N-2 is done;
            // prevSlot's last submission was Frame N-1 but we only READ from it here, no write).
            // Actually we need to snapshot the prevSlot's instance buffer to get the transforms.
            // instanceBuffers_[prevSlot] was written last frame and GPU N-1 built TLAS from it;
            // that GPU work is in-flight but only READS the buffer — our CPU read here is safe
            // because CPU-side writes to instanceBuffers_[prevSlot] for Frame N-1 happened before
            // the submission and are complete; we are just reading the mapped data now.
            if (lastTransformsBuffers_[slot].mappedPtr() && instanceBuffers_[prevSlot].mappedPtr()) {
                auto* srcInst = static_cast<const VkAccelerationStructureInstanceKHR*>(instanceBuffers_[prevSlot].mappedPtr());
                auto* dstXform = static_cast<float*>(lastTransformsBuffers_[slot].mappedPtr());
                for (uint32_t i = 0; i < prevCount; ++i) {
                    std::memcpy(dstXform + i * 12, &srcInst[i].transform, sizeof(float) * 12);
                }
                lastTransformsBuffers_[slot].flush();
            }
        }
        lastInstanceCounts_[slot] = prevCount;
    }

    instanceCount_ = static_cast<uint32_t>(instances.size());
    VkDeviceSize instanceSize = instanceCount_ * sizeof(VkAccelerationStructureInstanceKHR);

    // Create/resize instance buffer (host-visible, double-buffered per slot).
    // Double-buffering prevents the CPU write for Frame N from racing with the GPU
    // read of Frame N-1's TLAS build which uses instanceBuffers_[prevSlot].
    // When this slot's buffer must grow, retire the old one via GC — the fence wait
    // guarantees Frame N-2 (last use of this slot) is done, but GC deferral is still
    // used for safety in case the size query path re-uses it in the same frame.
    if (!instanceBuffers_[slot].handle() || instanceBuffers_[slot].size() < instanceSize) {
        if (instanceBuffers_[slot].handle()) retireBuffer(std::move(instanceBuffers_[slot]));
        vk2::Buffer::Desc bd{};
        bd.size = instanceSize;
        bd.usage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR
                 | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        bd.vmaFlags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
                    | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        auto r = vk2::Buffer::create(device_->device(), device_->vma(), bd);
        if (!r) { log::error("tlas", "Instance buffer[" + std::to_string(slot) + "]: " + r.error().message); return false; }
        instanceBuffers_[slot] = std::move(r.value());
    }

    // Upload instance data into this slot's buffer
    std::memcpy(instanceBuffers_[slot].mappedPtr(), instances.data(), instanceSize);
    instanceBuffers_[slot].flush();

    // Allocate and upload BDA arrays for THIS frame's slot.
    // The other slot is still being read by last frame's in-flight GPU commands.
    // framesInFlight = 2 and the fence wait before rebuild() guarantees slot is free.
    VkDeviceSize bdaSize = instanceCount_ * sizeof(uint64_t);
    if (!vertexBdaBuffers_[slot].handle() || bdaCapacity_[slot] < instanceCount_) {
        vk2::Buffer::Desc bd{};
        VkDeviceSize newCapacity = std::max<VkDeviceSize>(instanceCount_ * 2, 64);
        bd.size = newCapacity * sizeof(uint64_t);
        bd.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        bd.vmaFlags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
                    | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        auto rv = vk2::Buffer::create(device_->device(), device_->vma(), bd);
        if (!rv) { log::error("tlas", "Vertex BDA buffer[" + std::to_string(slot) + "]: " + rv.error().message); return false; }
        vertexBdaBuffers_[slot] = std::move(rv.value());

        auto ri = vk2::Buffer::create(device_->device(), device_->vma(), bd);
        if (!ri) { log::error("tlas", "Index BDA buffer[" + std::to_string(slot) + "]: " + ri.error().message); return false; }
        indexBdaBuffers_[slot] = std::move(ri.value());

        // BLAS offsets SSBO (same capacity)
        vk2::Buffer::Desc od{};
        od.size = newCapacity * sizeof(uint32_t);
        od.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        od.vmaFlags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
                    | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        auto ro = vk2::Buffer::create(device_->device(), device_->vma(), od);
        if (!ro) { log::error("tlas", "BLAS offsets buffer[" + std::to_string(slot) + "]: " + ro.error().message); return false; }
        blasOffsetsBuffers_[slot] = std::move(ro.value());

        bdaCapacity_[slot] = newCapacity;
    }
    // Write BDAs and instance count for this slot
    bdaInstanceCounts_[slot] = instanceCount_;
    std::memcpy(vertexBdaBuffers_[slot].mappedPtr(), vertexBdas.data(), bdaSize);
    vertexBdaBuffers_[slot].flush();
    std::memcpy(indexBdaBuffers_[slot].mappedPtr(), indexBdas.data(), bdaSize);
    indexBdaBuffers_[slot].flush();

    // Upload BLAS offsets: offsets[i] = i (format 0, 96-byte PBRTriangle stride).
    {
        auto* dst = static_cast<uint32_t*>(blasOffsetsBuffers_[slot].mappedPtr());
        for (uint32_t i = 0; i < instanceCount_; ++i) {
            dst[i] = i;
        }
        blasOffsetsBuffers_[slot].flush();
    }

    // Build geometry info
    VkAccelerationStructureGeometryKHR geometry{};
    geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    geometry.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    geometry.geometry.instances.arrayOfPointers = VK_FALSE;
    geometry.geometry.instances.data.deviceAddress = instanceBuffers_[slot].deviceAddress();

    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{};
    buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    buildInfo.geometryCount = 1;
    buildInfo.pGeometries = &geometry;

    // Query size
    VkAccelerationStructureBuildSizesInfoKHR sizeInfo{};
    sizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    vkGetAccelerationStructureBuildSizesKHR(device_->device(),
        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &buildInfo, &instanceCount_, &sizeInfo);

    // Recreate TLAS buffer for this slot if needed (retire old one via GC).
    // By writing into tlas_[slot] instead of a single tlas_, Frame N-1's GPU
    // vkCmdTraceRaysKHR continues reading tlas_[slot^1] uninterrupted while we
    // rebuild tlas_[slot] (the WAR hazard that caused type=6 DEVICE_FAULTs).
    if (!tlasBuffer_[slot].handle() || tlasBuffer_[slot].size() < sizeInfo.accelerationStructureSize) {
        // Retire the old TLAS handle + backing buffer for this slot (deferred via GC)
        retireTlas(tlas_[slot], std::move(tlasBuffer_[slot]));
        tlas_[slot] = VK_NULL_HANDLE;
        tlasAddress_[slot] = 0;

        vk2::Buffer::Desc bd{};
        bd.size = sizeInfo.accelerationStructureSize;
        bd.usage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR
                 | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        auto r = vk2::Buffer::create(device_->device(), device_->vma(), bd);
        if (!r) { log::error("tlas", "TLAS buffer[" + std::to_string(slot) + "]: " + r.error().message); return false; }
        tlasBuffer_[slot] = std::move(r.value());

        VkAccelerationStructureCreateInfoKHR asCI{};
        asCI.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
        asCI.buffer = tlasBuffer_[slot].handle();
        asCI.size = sizeInfo.accelerationStructureSize;
        asCI.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;

        if (vkCreateAccelerationStructureKHR(device_->device(), &asCI, nullptr, &tlas_[slot]) != VK_SUCCESS) {
            log::error("tlas", "vkCreateAccelerationStructureKHR[" + std::to_string(slot) + "] failed");
            return false;
        }

        VkAccelerationStructureDeviceAddressInfoKHR addrInfo{};
        addrInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
        addrInfo.accelerationStructure = tlas_[slot];
        tlasAddress_[slot] = vkGetAccelerationStructureDeviceAddressKHR(device_->device(), &addrInfo);
    }

    // Recreate scratch buffer if needed.
    // Defer the old one via GC — its device address is embedded in the previous
    // frame's buildInfo.scratchData and the GPU may still be reading it.
    if (!scratchBuffer_.handle() || scratchBuffer_.size() < sizeInfo.buildScratchSize) {
        if (scratchBuffer_.handle()) retireBuffer(std::move(scratchBuffer_));
        vk2::Buffer::Desc bd{};
        bd.size = sizeInfo.buildScratchSize;
        bd.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        bd.minAlignment = 256;
        auto r = vk2::Buffer::create(device_->device(), device_->vma(), bd);
        if (!r) { log::error("tlas", "Scratch buffer: " + r.error().message); return false; }
        scratchBuffer_ = std::move(r.value());
    }

    // Build — write into this frame's slot only
    buildInfo.dstAccelerationStructure = tlas_[slot];
    buildInfo.scratchData.deviceAddress = scratchBuffer_.deviceAddress();

    VkAccelerationStructureBuildRangeInfoKHR rangeInfo{};
    rangeInfo.primitiveCount = instanceCount_;
    const VkAccelerationStructureBuildRangeInfoKHR* pRange = &rangeInfo;

    // Structured diagnostic event: tlas_prebuild (gated on DiagFlags::TLAS)
    if ((diagFlags_ & DiagFlags::TLAS) != 0) {
        auto toHex = [](uint64_t v) -> std::string {
            char buf[20]; std::snprintf(buf, sizeof(buf), "0x%llx", static_cast<unsigned long long>(v));
            return buf;
        };
        uint32_t zeroBdaCount = 0;
        for (auto bda : vertexBdas) if (bda == 0) ++zeroBdaCount;
        log::event("tlas", "tlas_prebuild", {
            {"frame",         std::to_string(frameIndex)},
            {"instanceCount", std::to_string(instanceCount_)},
            {"firstBda",      vertexBdas.empty() ? "none" : toHex(vertexBdas.front())},
            {"lastBda",       vertexBdas.empty() ? "none" : toHex(vertexBdas.back())},
            {"zeroBdaCount",  std::to_string(zeroBdaCount)},
            {"slot",          std::to_string(slot)}
        });
    }

    // BDA trace: dump all instance vertex/index BDAs (gated on DiagFlags::BDA)
    if ((diagFlags_ & DiagFlags::BDA) != 0) {
        auto toHex = [](uint64_t v) -> std::string {
            char buf[20]; std::snprintf(buf, sizeof(buf), "0x%llx", static_cast<unsigned long long>(v));
            return buf;
        };
        for (size_t i = 0; i < vertexBdas.size(); ++i) {
            bool nullVertex = (vertexBdas[i] == 0);
            bool nullIndex  = (i < indexBdas.size() && indexBdas[i] == 0);
            if (nullVertex || nullIndex) {
                // Only log anomalous entries at trace level to keep volume manageable
                log::trace("tlas", "[BDA] instance[" + std::to_string(i) + "]"
                    + " vertexBda=" + toHex(vertexBdas[i])
                    + " indexBda=" + (i < indexBdas.size() ? toHex(indexBdas[i]) : "n/a")
                    + " *** NULL BDA — should have been caught by null guard ***");
            }
        }
        log::trace("tlas", "[BDA] tlas frame=" + std::to_string(frameIndex)
            + " instances=" + std::to_string(instanceCount_)
            + " vertex_bdas=" + std::to_string(vertexBdas.size())
            + " index_bdas=" + std::to_string(indexBdas.size()));
    }

    nvInsertCheckpoint(cmd, (diagFlags_ & DiagFlags::CRASH) != 0,
                       device_->caps().nvCheckpoints, "TLAS_BUILD_START");
    vkCmdBuildAccelerationStructuresKHR(cmd, 1, &buildInfo, &pRange);

    // Barrier: TLAS build → RT shader read
    // dstAccessMask includes SHADER_STORAGE_READ_BIT because RT shaders also read the
    // BDA/vertex/index SSBOs (vertexBdaBuffers_, indexBdaBuffers_, blasOffsetsBuffers_)
    // that are bound at descriptor set 1. ACCELERATION_STRUCTURE_READ alone only covers
    // the AS structure itself; without SHADER_STORAGE_READ the shader may see stale SSBO
    // content from a previous frame, leading to missed hits and a black output image.
    VkMemoryBarrier2 barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    barrier.srcStageMask  = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
    barrier.srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    barrier.dstStageMask  = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
    barrier.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR
                          | VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
    VkDependencyInfo dep{};
    dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.memoryBarrierCount = 1;
    dep.pMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(cmd, &dep);

    return true;
}

void TlasService::retireTlas(VkAccelerationStructureKHR handle, vk2::Buffer buffer) {
    if (handle == VK_NULL_HANDLE) return;

    if (gc_) {
        VkDevice dev = device_->device();
        auto buf = std::make_shared<vk2::Buffer>(std::move(buffer));
        gc_->defer([dev, handle, buf]() {
            vkDestroyAccelerationStructureKHR(dev, handle, nullptr);
            // buf's shared_ptr releases here; vk2::Buffer dtor runs if last ref
        });
    } else {
        if (device_) {
            vkDestroyAccelerationStructureKHR(device_->device(), handle, nullptr);
        }
        // buffer destructor runs as arg goes out of scope
    }
}

void TlasService::retireBuffer(vk2::Buffer buffer) {
    if (!buffer.handle()) return;
    if (gc_) {
        auto buf = std::make_shared<vk2::Buffer>(std::move(buffer));
        gc_->defer([buf]() {
            // buf's shared_ptr releases here; vk2::Buffer dtor destroys VkBuffer + VmaAllocation
        });
    }
    // If no GC wired, the buffer is destroyed synchronously as `buffer` goes out of scope.
    // This is safe only if called outside an active submission window (e.g., shutdown).
}

} // namespace engine
