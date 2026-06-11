#include "core/render/material_registry.hpp"
#include "core/render/renderer.hpp"
#include "core/render/render_framework.hpp"

#include <algorithm>
#include <iostream>
#include <limits>
#include <sstream>
#include <utility>

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

    vk::Data::MaterialEntry defaultEntry{};
    defaultEntry.materialId = 0;
    defaultEntry.baseSpriteId = 0;
    defaultEntry.fallbackMaterialId = 0;
    defaultEntry.flags = vk::Data::MATERIAL_FLAG_VALID |
                         vk::Data::MATERIAL_FLAG_VANILLA_SPRITE |
                         vk::Data::MATERIAL_FLAG_GPU_RESIDENT;
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

bool MaterialRegistry::updateMaterialsSparse(const vk::Data::MaterialEntry* entries, uint32_t count,
                                             std::shared_ptr<vk::VMA> vma,
                                             std::shared_ptr<vk::Device> device) {
    if (!entries || count == 0 || !vma || !device) return false;
    std::lock_guard<std::mutex> operationLock(operationMutex_);

    auto renderer = Renderer::try_instance();
    if (!renderer || !renderer->framework()) return false;
    auto framework = renderer->framework();

    std::vector<vk::Data::MaterialEntry> sparseEntries;
    std::vector<VkBufferCopy> copyRegions;
    std::vector<vk::Data::MaterialEntry> nextEntries;
    std::shared_ptr<vk::DeviceLocalBuffer> targetSsbo;
    std::shared_ptr<vk::HostVisibleBuffer> staging;
    std::shared_ptr<vk::CommandBuffer> cmd;
    std::shared_ptr<vk::Fence> fence;
    std::shared_ptr<vk::TimelineSemaphore> timeline;
    VkTimelineSemaphoreSubmitInfo timelineInfo{};
    VkSubmitInfo submitInfo{};
    VkSemaphore signalSemaphore = VK_NULL_HANDLE;
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    uint64_t timelineValue = 0;
    uint64_t pendingBytes = 0;
    bool createdLazySsbo = false;
    uint32_t updated = 0;
    uint32_t minMaterialId = std::numeric_limits<uint32_t>::max();
    uint32_t maxMaterialId = 0;

    {
        std::lock_guard<std::mutex> lock(mutex_);
        pollCompletedUploadsLocked();

        nextEntries = entries_;
        if (!ssbo_ || !ssbo_->isValid()) {
            std::cout << "[MaterialRegistry] Lazy SSBO creation for sparse V4 path" << std::endl;
            vk::Data::MaterialEntry defaultEntry{};
            defaultEntry.materialId = 0;
            defaultEntry.baseSpriteId = 0;
            defaultEntry.fallbackMaterialId = 0;
            defaultEntry.flags = vk::Data::MATERIAL_FLAG_VALID |
                                 vk::Data::MATERIAL_FLAG_VANILLA_SPRITE |
                                 vk::Data::MATERIAL_FLAG_GPU_RESIDENT;
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

            nextEntries.assign(vk::Data::MATERIAL_MAX_ENTRIES, defaultEntry);
            const VkDeviceSize dataSize = nextEntries.size() * sizeof(vk::Data::MaterialEntry);
            targetSsbo = vk::DeviceLocalBuffer::create(
                vma, device, true, dataSize,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
            if (!targetSsbo || !targetSsbo->isValid()) {
                std::cerr << "[MaterialRegistry] Lazy SSBO creation failed" << std::endl;
                return false;
            }
            targetSsbo->uploadToStagingBuffer(nextEntries.data(), static_cast<size_t>(dataSize), 0);
            pendingBytes += static_cast<uint64_t>(dataSize);
            createdLazySsbo = true;
        } else {
            targetSsbo = ssbo_;
        }

        sparseEntries.reserve(count);
        copyRegions.reserve(count);
        for (uint32_t i = 0; i < count; i++) {
            const auto& entry = entries[i];
            if (entry.materialId >= vk::Data::MATERIAL_MAX_ENTRIES) {
                rejectedSparseEntries_++;
                continue;
            }
            if (nextEntries.size() <= entry.materialId) {
                nextEntries.resize(static_cast<size_t>(entry.materialId) + 1);
            }
            nextEntries[entry.materialId] = entry;
            sparseEntries.push_back(entry);
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

        const VkDeviceSize dataSize = static_cast<VkDeviceSize>(sparseEntries.size())
            * sizeof(vk::Data::MaterialEntry);
        staging = vk::HostVisibleBuffer::create(
            vma, device, static_cast<size_t>(dataSize), VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
        staging->uploadToBuffer(sparseEntries.data(), static_cast<size_t>(dataSize), 0);
        pendingBytes += static_cast<uint64_t>(dataSize);

        fence = vk::Fence::create(device);
        cmd = vk::CommandBuffer::create(device, framework->mainCommandPool());
        cmd->begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
        if (createdLazySsbo) {
            targetSsbo->uploadToBuffer(cmd);
        }
        vkCmdCopyBuffer(cmd->vkCommandBuffer(), staging->vkBuffer(), targetSsbo->vkBuffer(),
                        static_cast<uint32_t>(copyRegions.size()), copyRegions.data());
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
        if (createdLazySsbo) {
            if (ssbo_ && ssbo_ != targetSsbo) {
                framework->gc().collect(ssbo_);
            }
            ssbo_ = targetSsbo;
        }
        entries_ = std::move(nextEntries);
        materialCount_ = std::max(materialCount_, static_cast<uint32_t>(entries_.size()));
        sparseUpdates_++;
        asyncSubmissions_++;
        if (timeline) {
            timelineSubmissions_++;
        } else {
            fenceSubmissions_++;
        }
        pendingUploadBytes_ += pendingBytes;
        pendingUploads_.push_back(PendingUpload{
            .device = device,
            .commandBuffer = cmd,
            .fence = timeline ? nullptr : fence,
            .timeline = timeline,
            .deviceLocalStagingOwner = createdLazySsbo ? targetSsbo : nullptr,
            .targetBuffer = targetSsbo,
            .hostStaging = staging,
            .bytes = pendingBytes,
            .entries = updated,
            .timelineValue = timelineValue,
            .sparse = true,
        });
        lastSparseEntryCount_ = updated;
        lastSparseMinMaterialId_ = minMaterialId == std::numeric_limits<uint32_t>::max() ? 0 : minMaterialId;
        lastSparseMaxMaterialId_ = maxMaterialId;
    }

    std::cout << "[MaterialRegistry] Queued async sparse update for " << updated
              << " material entries minId="
              << (minMaterialId == std::numeric_limits<uint32_t>::max() ? 0 : minMaterialId)
              << " maxId=" << maxMaterialId << std::endl;
    return true;
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
