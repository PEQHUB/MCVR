#include "core/render/material_registry.hpp"
#include "core/render/renderer.hpp"
#include "core/render/render_framework.hpp"

#include <algorithm>
#include <iostream>
#include <limits>
#include <sstream>
#include <utility>

bool MaterialRegistry::uploadMaterials(const vk::Data::MaterialEntry* entries, uint32_t count,
                                       std::shared_ptr<vk::VMA> vma,
                                       std::shared_ptr<vk::Device> device) {
    if (!entries || count == 0 || !vma || !device) return false;
    std::lock_guard<std::mutex> lock(mutex_);

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
    defaultEntry.displacementPolicy = vk::Data::MATERIAL_DISPLACEMENT_DISABLED;
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
    cmd->submitMainQueueIndividual(device, fence);
    vkWaitForFences(device->vkDevice(), 1, &fence->vkFence(), VK_TRUE, UINT64_MAX);

    if (ssbo_) {
        framework->gc().collect(ssbo_);
    }
    ssbo_ = std::move(nextSsbo);
    entries_.assign(entries, entries + copyCount);
    materialCount_ = static_cast<uint32_t>(copyCount);
    fullUploads_++;

    std::cout << "[MaterialRegistry] Uploaded " << materialCount_
              << " materials (" << dataSize << " bytes padded) to GPU SSBO" << std::endl;
    return true;
}

bool MaterialRegistry::updateMaterialsSparse(const vk::Data::MaterialEntry* entries, uint32_t count,
                                             std::shared_ptr<vk::VMA> vma,
                                             std::shared_ptr<vk::Device> device) {
    if (!entries || count == 0 || !vma || !device) return false;
    std::lock_guard<std::mutex> lock(mutex_);
    if (!ssbo_ || !ssbo_->isValid() || entries_.empty()) return false;

    auto renderer = Renderer::try_instance();
    if (!renderer || !renderer->framework()) return false;
    auto framework = renderer->framework();

    uint32_t updated = 0;
    uint32_t minMaterialId = std::numeric_limits<uint32_t>::max();
    uint32_t maxMaterialId = 0;
    for (uint32_t i = 0; i < count; i++) {
        const auto& entry = entries[i];
        if (entry.materialId >= vk::Data::MATERIAL_MAX_ENTRIES) {
            rejectedSparseEntries_++;
            continue;
        }
        if (entries_.size() <= entry.materialId) {
            entries_.resize(static_cast<size_t>(entry.materialId) + 1);
        }
        entries_[entry.materialId] = entry;
        size_t offset = static_cast<size_t>(entry.materialId) * sizeof(vk::Data::MaterialEntry);
        ssbo_->uploadToStagingBuffer(
            const_cast<vk::Data::MaterialEntry*>(&entry),
            sizeof(vk::Data::MaterialEntry),
            offset);
        updated++;
        minMaterialId = std::min(minMaterialId, entry.materialId);
        maxMaterialId = std::max(maxMaterialId, entry.materialId);
    }
    if (updated == 0) return true;

    auto fence = vk::Fence::create(device);
    auto cmd = vk::CommandBuffer::create(device, framework->mainCommandPool());
    cmd->begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
    for (uint32_t i = 0; i < count; i++) {
        const auto& entry = entries[i];
        if (entry.materialId >= vk::Data::MATERIAL_MAX_ENTRIES) {
            continue;
        }
        size_t offset = static_cast<size_t>(entry.materialId) * sizeof(vk::Data::MaterialEntry);
        ssbo_->uploadToBuffer(cmd, sizeof(vk::Data::MaterialEntry), offset, offset);
    }
    cmd->end();
    cmd->submitMainQueueIndividual(device, fence);
    vkWaitForFences(device->vkDevice(), 1, &fence->vkFence(), VK_TRUE, UINT64_MAX);

    materialCount_ = std::max(materialCount_, static_cast<uint32_t>(entries_.size()));
    sparseUpdates_++;
    lastSparseEntryCount_ = updated;
    lastSparseMinMaterialId_ = minMaterialId == std::numeric_limits<uint32_t>::max() ? 0 : minMaterialId;
    lastSparseMaxMaterialId_ = maxMaterialId;
    std::cout << "[MaterialRegistry] Sparse updated " << updated
              << " material entries minId=" << lastSparseMinMaterialId_
              << " maxId=" << lastSparseMaxMaterialId_ << std::endl;
    return true;
}

std::string MaterialRegistry::statusJson() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::ostringstream out;
    out << "{"
        << "\"schema\":\"radser_material_table_status_v1\","
        << "\"materialCount\":" << materialCount_ << ","
        << "\"fullUploads\":" << fullUploads_ << ","
        << "\"sparseUpdates\":" << sparseUpdates_ << ","
        << "\"lastSparseEntryCount\":" << lastSparseEntryCount_ << ","
        << "\"lastSparseMinMaterialId\":" << lastSparseMinMaterialId_ << ","
        << "\"lastSparseMaxMaterialId\":" << lastSparseMaxMaterialId_ << ","
        << "\"rejectedSparseEntries\":" << rejectedSparseEntries_
        << "}";
    return out.str();
}

void MaterialRegistry::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    entries_.clear();
    materialCount_ = 0;
    fullUploads_ = 0;
    sparseUpdates_ = 0;
    lastSparseEntryCount_ = 0;
    lastSparseMinMaterialId_ = 0;
    lastSparseMaxMaterialId_ = 0;
    rejectedSparseEntries_ = 0;
    ssbo_.reset();
}

void MaterialRegistry::retire(GarbageCollector& gc) {
    std::lock_guard<std::mutex> lock(mutex_);
    gc.collect(ssbo_);
    entries_.clear();
    materialCount_ = 0;
    fullUploads_ = 0;
    sparseUpdates_ = 0;
    lastSparseEntryCount_ = 0;
    lastSparseMinMaterialId_ = 0;
    lastSparseMaxMaterialId_ = 0;
    rejectedSparseEntries_ = 0;
    ssbo_.reset();
}
