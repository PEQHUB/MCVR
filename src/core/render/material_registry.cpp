#include "core/render/material_registry.hpp"
#include "core/render/renderer.hpp"
#include "core/render/render_framework.hpp"

#include <algorithm>
#include <iostream>
#include <limits>
#include <sstream>
#include <utility>

namespace {
vk::Data::MaterialEntry makeDefaultMaterialEntry(bool fallback) {
    vk::Data::MaterialEntry defaultEntry{};
    defaultEntry.materialId = 0;
    defaultEntry.baseSpriteId = 0;
    defaultEntry.fallbackMaterialId = 0;
    defaultEntry.flags = vk::Data::MATERIAL_FLAG_VALID |
                         vk::Data::MATERIAL_FLAG_VANILLA_SPRITE |
                         vk::Data::MATERIAL_FLAG_GPU_RESIDENT;
    if (fallback) {
        defaultEntry.flags |= vk::Data::MATERIAL_FLAG_FALLBACK;
    }
    defaultEntry.albedoPage = 0;
    defaultEntry.albedoLayer = 0;
    defaultEntry.specularPage = 0;
    defaultEntry.specularLayer = -1;
    defaultEntry.normalPage = 0;
    defaultEntry.normalLayer = -1;
    defaultEntry.flagPage = 0;
    defaultEntry.flagLayer = 0;
    defaultEntry.overlayMaterialId = -1;
    defaultEntry.displacementPolicy = 0;
    defaultEntry.displacementScale = 0.0f;
    defaultEntry.heightRangePacked = -1;
    defaultEntry.uvScaleU = 1.0f;
    defaultEntry.uvScaleV = 1.0f;
    defaultEntry.uvOffsetU = 0.0f;
    defaultEntry.uvOffsetV = 0.0f;
    return defaultEntry;
}
}

void MaterialRegistry::pollCompletedUploadsLocked() const {
    for (auto it = pendingUploads_.begin(); it != pendingUploads_.end();) {
        if (!it->device || (!it->fence && !it->timeline)) {
            pendingUploadBytes_ -= std::min<uint64_t>(pendingUploadBytes_, it->bytes);
            it = pendingUploads_.erase(it);
            continue;
        }
        bool complete = false;
        if (it->timeline && it->timelineValue > 0) {
            uint64_t currentValue = it->timeline->getValue();
            complete = currentValue >= it->timelineValue;
            if (complete) {
                lastCompletedTimelineValue_ = std::max(lastCompletedTimelineValue_, it->timelineValue);
            }
        } else if (it->fence) {
            complete = vkGetFenceStatus(it->device->vkDevice(), it->fence->vkFence()) == VK_SUCCESS;
        }
        if (complete) {
            if (it->deviceLocalStagingOwner) {
                it->deviceLocalStagingOwner->releaseStagingBuffer();
            }
            pendingUploadBytes_ -= std::min<uint64_t>(pendingUploadBytes_, it->bytes);
            asyncCompletions_++;
            it = pendingUploads_.erase(it);
        } else {
            ++it;
        }
    }
}

bool MaterialRegistry::uploadMaterials(const vk::Data::MaterialEntry* entries, uint32_t count,
                                       std::shared_ptr<vk::VMA> vma,
                                       std::shared_ptr<vk::Device> device) {
    if (!entries || count == 0 || !vma || !device) return false;
    std::lock_guard<std::mutex> operationLock(operationMutex_);

    auto renderer = Renderer::try_instance();
    if (!renderer || !renderer->framework()) return false;
    auto framework = renderer->framework();

    vk::Data::MaterialEntry defaultEntry = makeDefaultMaterialEntry(false);

    std::vector<vk::Data::MaterialEntry> uploadEntries(vk::Data::MATERIAL_MAX_ENTRIES, defaultEntry);
    const size_t copyCount = std::min<size_t>(count, uploadEntries.size());
    std::copy(entries, entries + copyCount, uploadEntries.begin());

    VkDeviceSize dataSize = uploadEntries.size() * sizeof(vk::Data::MaterialEntry);
    auto nextSsbo = vk::DeviceLocalBuffer::create(
        vma, device, true, dataSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    nextSsbo->uploadToStagingBuffer(uploadEntries.data(), static_cast<size_t>(dataSize), 0);

    auto fence = vk::Fence::create(device);
    auto cmd = vk::CommandBuffer::create(device, framework->mainCommandPool());
    cmd->begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
    nextSsbo->uploadToBuffer(cmd);
    cmd->end();
    auto timeline = device->materialUploadSemaphore();
    uint64_t timelineValue = 0;
    if (timeline) {
        VkTimelineSemaphoreSubmitInfo timelineInfo{};
        timelineInfo.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
        timelineInfo.signalSemaphoreValueCount = 1;
        timelineInfo.pSignalSemaphoreValues = &timelineValue;

        VkSemaphore signalSemaphore = timeline->vkSemaphore();
        VkCommandBuffer commandBuffer = cmd->vkCommandBuffer();
        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.pNext = &timelineInfo;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = &signalSemaphore;

        {
            std::lock_guard<std::mutex> lock(mutex_);
            pollCompletedUploadsLocked();
            timelineValue = ++materialUploadTimelineValue_;
        }
        timelineInfo.pSignalSemaphoreValues = &timelineValue;

        std::lock_guard<std::mutex> qLock(device->queueMutex());
        VkResult submitResult = vkQueueSubmit(device->mainVkQueue(), 1, &submitInfo, VK_NULL_HANDLE);
        if (submitResult != VK_SUCCESS) {
            std::lock_guard<std::mutex> lock(mutex_);
            --materialUploadTimelineValue_;
            return false;
        }
    } else {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            pollCompletedUploadsLocked();
        }
        cmd->submitMainQueueIndividual(device, fence);
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (ssbo_) {
            framework->gc().collect(ssbo_);
        }
        ssbo_ = std::move(nextSsbo);
        entries_.assign(entries, entries + copyCount);
        materialCount_ = static_cast<uint32_t>(copyCount);
        fullUploads_++;
        usingFallback_ = false;
        revision_++;
        asyncSubmissions_++;
        if (timeline) {
            timelineSubmissions_++;
        } else {
            fenceSubmissions_++;
        }
        pendingUploadBytes_ += static_cast<uint64_t>(dataSize);
        pendingUploads_.push_back(PendingUpload{
            .device = device,
            .commandBuffer = cmd,
            .fence = timeline ? nullptr : fence,
            .timeline = timeline,
            .deviceLocalStagingOwner = ssbo_,
            .targetBuffer = ssbo_,
            .bytes = static_cast<uint64_t>(dataSize),
            .entries = static_cast<uint32_t>(copyCount),
            .timelineValue = timelineValue,
            .sparse = false,
        });
    }

    std::cout << "[MaterialRegistry] Queued async full upload for " << copyCount
              << " materials (" << dataSize << " bytes padded) to GPU SSBO" << std::endl;
    return true;
}

bool MaterialRegistry::ensureFallbackMaterials(std::shared_ptr<vk::VMA> vma,
                                               std::shared_ptr<vk::Device> device) {
    if (!vma || !device) return false;
    std::lock_guard<std::mutex> operationLock(operationMutex_);

    auto renderer = Renderer::try_instance();
    if (!renderer || !renderer->framework()) return false;
    auto framework = renderer->framework();

    std::vector<vk::Data::MaterialEntry> uploadEntries;
    std::shared_ptr<vk::DeviceLocalBuffer> nextSsbo;
    std::shared_ptr<vk::CommandBuffer> cmd;
    std::shared_ptr<vk::Fence> fence;
    std::shared_ptr<vk::TimelineSemaphore> timeline;
    VkTimelineSemaphoreSubmitInfo timelineInfo{};
    VkSubmitInfo submitInfo{};
    VkSemaphore signalSemaphore = VK_NULL_HANDLE;
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    uint64_t timelineValue = 0;
    const VkDeviceSize dataSize = static_cast<VkDeviceSize>(vk::Data::MATERIAL_MAX_ENTRIES)
        * sizeof(vk::Data::MaterialEntry);

    {
        std::lock_guard<std::mutex> lock(mutex_);
        pollCompletedUploadsLocked();
        if (ssbo_ && ssbo_->isValid()) return true;

        uploadEntries.assign(vk::Data::MATERIAL_MAX_ENTRIES, makeDefaultMaterialEntry(true));
        nextSsbo = vk::DeviceLocalBuffer::create(
            vma, device, true, dataSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        if (!nextSsbo || !nextSsbo->isValid()) return false;
        nextSsbo->uploadToStagingBuffer(uploadEntries.data(), static_cast<size_t>(dataSize), 0);

        fence = vk::Fence::create(device);
        cmd = vk::CommandBuffer::create(device, framework->mainCommandPool());
        cmd->begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
        nextSsbo->uploadToBuffer(cmd);
        cmd->end();

        timeline = device->materialUploadSemaphore();
        if (timeline) {
            timelineValue = ++materialUploadTimelineValue_;
            timelineInfo.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
            timelineInfo.signalSemaphoreValueCount = 1;
            timelineInfo.pSignalSemaphoreValues = &timelineValue;

            signalSemaphore = timeline->vkSemaphore();
            commandBuffer = cmd->vkCommandBuffer();
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitInfo.pNext = &timelineInfo;
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &commandBuffer;
            submitInfo.signalSemaphoreCount = 1;
            submitInfo.pSignalSemaphores = &signalSemaphore;
        }
    }

    if (timeline) {
        std::lock_guard<std::mutex> qLock(device->queueMutex());
        VkResult submitResult = vkQueueSubmit(device->mainVkQueue(), 1, &submitInfo, VK_NULL_HANDLE);
        if (submitResult != VK_SUCCESS) {
            std::lock_guard<std::mutex> lock(mutex_);
            --materialUploadTimelineValue_;
            return false;
        }
    } else {
        cmd->submitMainQueueIndividual(device, fence);
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (ssbo_) {
            framework->gc().collect(ssbo_);
        }
        ssbo_ = std::move(nextSsbo);
        entries_.clear();
        materialCount_ = 0;
        fallbackUploads_++;
        usingFallback_ = true;
        revision_++;
        asyncSubmissions_++;
        if (timeline) {
            timelineSubmissions_++;
        } else {
            fenceSubmissions_++;
        }
        pendingUploadBytes_ += static_cast<uint64_t>(dataSize);
        pendingUploads_.push_back(PendingUpload{
            .device = device,
            .commandBuffer = cmd,
            .fence = timeline ? nullptr : fence,
            .timeline = timeline,
            .deviceLocalStagingOwner = ssbo_,
            .targetBuffer = ssbo_,
            .bytes = static_cast<uint64_t>(dataSize),
            .entries = vk::Data::MATERIAL_MAX_ENTRIES,
            .timelineValue = timelineValue,
            .sparse = false,
        });
    }

    std::cout << "[MaterialRegistry] Queued V4 fallback material table upload ("
              << dataSize << " bytes padded)" << std::endl;
    return true;
}

bool MaterialRegistry::updateMaterialsSparse(const vk::Data::MaterialEntry* entries, uint32_t count,
                                             uint64_t generation,
                                             std::shared_ptr<vk::VMA> vma,
                                             std::shared_ptr<vk::Device> device) {
    if (!entries || count == 0 || !vma || !device) return false;
    std::lock_guard<std::mutex> operationLock(operationMutex_);

    auto renderer = Renderer::try_instance();
    if (!renderer || !renderer->framework()) return false;
    auto& uploadService = renderer->textureLoaderV4().uploadService();

    std::vector<vk::Data::MaterialEntry> sparseEntries;
    std::vector<VkBufferCopy> copyRegions;
    std::vector<vk::Data::MaterialEntry> nextEntries;
    std::shared_ptr<vk::DeviceLocalBuffer> targetSsbo;
    uint32_t updated = 0;
    uint32_t minMaterialId = std::numeric_limits<uint32_t>::max();
    uint32_t maxMaterialId = 0;

    {
        std::lock_guard<std::mutex> lock(mutex_);
        pollCompletedUploadsLocked();

        if (!ssbo_ || !ssbo_->isValid()) {
            std::cerr << "[MaterialRegistry] Sparse update rejected: material SSBO is not initialized" << std::endl;
            return false;
        }
        targetSsbo = ssbo_;
        nextEntries = entries_;

        sparseEntries.reserve(count);
        for (uint32_t i = 0; i < count; i++) {
            const auto& entry = entries[i];
            if (entry.materialId >= vk::Data::MATERIAL_MAX_ENTRIES) {
                rejectedSparseEntries_++;
                continue;
            }
            auto existing = std::find_if(sparseEntries.begin(), sparseEntries.end(),
                [&](const vk::Data::MaterialEntry& candidate) {
                    return candidate.materialId == entry.materialId;
                });
            if (existing != sparseEntries.end()) {
                *existing = entry;
            } else {
                sparseEntries.push_back(entry);
            }
        }

        if (sparseEntries.empty()) return true;

        std::sort(sparseEntries.begin(), sparseEntries.end(),
            [](const vk::Data::MaterialEntry& a, const vk::Data::MaterialEntry& b) {
                return a.materialId < b.materialId;
            });

        copyRegions.reserve(sparseEntries.size());
        for (const auto& entry : sparseEntries) {
            if (nextEntries.size() <= entry.materialId) {
                nextEntries.resize(static_cast<size_t>(entry.materialId) + 1);
            }
            nextEntries[entry.materialId] = entry;
            VkBufferCopy region{};
            region.srcOffset = static_cast<VkDeviceSize>(updated) * sizeof(vk::Data::MaterialEntry);
            region.dstOffset = static_cast<VkDeviceSize>(entry.materialId) * sizeof(vk::Data::MaterialEntry);
            region.size = sizeof(vk::Data::MaterialEntry);
            copyRegions.push_back(region);
            updated++;
            minMaterialId = std::min(minMaterialId, entry.materialId);
            maxMaterialId = std::max(maxMaterialId, entry.materialId);
        }

        if (updated == 0) return true;
    }

    const uint64_t dataSize = static_cast<uint64_t>(sparseEntries.size())
        * sizeof(vk::Data::MaterialEntry);
    GpuUploadService::BufferUpload upload{};
    upload.generation = generation;
    upload.dst = targetSsbo->vkBuffer();
    upload.data = reinterpret_cast<const uint8_t*>(sparseEntries.data());
    upload.bytes = dataSize;
    upload.regions = copyRegions.data();
    upload.regionCount = static_cast<uint32_t>(copyRegions.size());
    upload.visible = true;
    upload.priority = GpuUploadService::Priority::FirstFrameAux;
    upload.dstOwner = targetSsbo;
    if (!uploadService.enqueueBufferUpload(upload)) {
        std::cerr << "[MaterialRegistry] Sparse update rejected: upload service enqueue failed"
                  << " entries=" << updated
                  << " bytes=" << dataSize << std::endl;
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        entries_ = std::move(nextEntries);
        materialCount_ = std::max(materialCount_, static_cast<uint32_t>(entries_.size()));
        sparseUpdates_++;
        usingFallback_ = false;
        revision_++;
        asyncSubmissions_++;
        lastSparseEntryCount_ = updated;
        lastSparseMinMaterialId_ = minMaterialId == std::numeric_limits<uint32_t>::max() ? 0 : minMaterialId;
        lastSparseMaxMaterialId_ = maxMaterialId;
    }

    std::cout << "[MaterialRegistry] Queued upload-service sparse update for " << updated
              << " material entries minId="
              << (minMaterialId == std::numeric_limits<uint32_t>::max() ? 0 : minMaterialId)
              << " maxId=" << maxMaterialId << std::endl;
    return true;
}

bool MaterialRegistry::hasBuffer() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return ssbo_ && ssbo_->isValid();
}

bool MaterialRegistry::usingFallback() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return usingFallback_;
}

uint64_t MaterialRegistry::revision() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return revision_;
}

std::string MaterialRegistry::statusJson() const {
    std::lock_guard<std::mutex> lock(mutex_);
    pollCompletedUploadsLocked();
    std::ostringstream out;
    uint64_t pendingEntries = 0;
    uint32_t pendingFullUploads = 0;
    uint32_t pendingSparseUploads = 0;
    for (const auto& upload : pendingUploads_) {
        pendingEntries += upload.entries;
        if (upload.sparse) {
            pendingSparseUploads++;
        } else {
            pendingFullUploads++;
        }
    }
    out << "{"
        << "\"schema\":\"radser_material_table_status_v1\","
        << "\"materialCount\":" << materialCount_ << ","
        << "\"hasBuffer\":" << ((ssbo_ && ssbo_->isValid()) ? "true" : "false") << ","
        << "\"usingFallback\":" << (usingFallback_ ? "true" : "false") << ","
        << "\"fallbackUploads\":" << fallbackUploads_ << ","
        << "\"revision\":" << revision_ << ","
        << "\"fullUploads\":" << fullUploads_ << ","
        << "\"sparseUpdates\":" << sparseUpdates_ << ","
        << "\"blockingFenceUploads\":false,"
        << "\"asyncMainQueueUploads\":true,"
        << "\"timelineUploadCompletion\":" << (timelineSubmissions_ > 0 ? "true" : "false") << ","
        << "\"timelineSubmissions\":" << timelineSubmissions_ << ","
        << "\"fenceSubmissions\":" << fenceSubmissions_ << ","
        << "\"lastSubmittedTimelineValue\":" << materialUploadTimelineValue_ << ","
        << "\"lastCompletedTimelineValue\":" << lastCompletedTimelineValue_ << ","
        << "\"asyncSubmissions\":" << asyncSubmissions_ << ","
        << "\"asyncCompletions\":" << asyncCompletions_ << ","
        << "\"pendingAsyncUploads\":" << pendingUploads_.size() << ","
        << "\"pendingAsyncFullUploads\":" << pendingFullUploads << ","
        << "\"pendingAsyncSparseUploads\":" << pendingSparseUploads << ","
        << "\"pendingAsyncMaterialEntries\":" << pendingEntries << ","
        << "\"pendingAsyncUploadBytes\":" << pendingUploadBytes_ << ","
        << "\"asyncTransferQueueUpload\":false,"
        << "\"fencePollingUploadCompletion\":true,"
        << "\"retainsInFlightTargetsUntilFence\":true,"
        << "\"lastSparseEntryCount\":" << lastSparseEntryCount_ << ","
        << "\"lastSparseMinMaterialId\":" << lastSparseMinMaterialId_ << ","
        << "\"lastSparseMaxMaterialId\":" << lastSparseMaxMaterialId_ << ","
        << "\"rejectedSparseEntries\":" << rejectedSparseEntries_
        << "}";
    return out.str();
}

void MaterialRegistry::reset() {
    std::lock_guard<std::mutex> operationLock(operationMutex_);
    std::lock_guard<std::mutex> lock(mutex_);
    entries_.clear();
    materialCount_ = 0;
    fullUploads_ = 0;
    sparseUpdates_ = 0;
    fallbackUploads_ = 0;
    usingFallback_ = false;
    revision_++;
    asyncSubmissions_ = 0;
    asyncCompletions_ = 0;
    timelineSubmissions_ = 0;
    fenceSubmissions_ = 0;
    lastSparseEntryCount_ = 0;
    lastSparseMinMaterialId_ = 0;
    lastSparseMaxMaterialId_ = 0;
    rejectedSparseEntries_ = 0;
    pollCompletedUploadsLocked();
    ssbo_.reset();
}

void MaterialRegistry::retire(GarbageCollector& gc) {
    std::lock_guard<std::mutex> operationLock(operationMutex_);
    std::lock_guard<std::mutex> lock(mutex_);
    pollCompletedUploadsLocked();
    gc.collect(ssbo_);
    entries_.clear();
    materialCount_ = 0;
    fullUploads_ = 0;
    sparseUpdates_ = 0;
    fallbackUploads_ = 0;
    usingFallback_ = false;
    revision_++;
    asyncSubmissions_ = 0;
    asyncCompletions_ = 0;
    timelineSubmissions_ = 0;
    fenceSubmissions_ = 0;
    lastSparseEntryCount_ = 0;
    lastSparseMinMaterialId_ = 0;
    lastSparseMaxMaterialId_ = 0;
    rejectedSparseEntries_ = 0;
    ssbo_.reset();
}
