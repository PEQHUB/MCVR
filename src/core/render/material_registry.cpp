#include "core/render/material_registry.hpp"
#include "core/render/renderer.hpp"
#include "core/render/render_framework.hpp"

#include <algorithm>
#include <iostream>
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

    std::cout << "[MaterialRegistry] Uploaded " << materialCount_
              << " materials (" << dataSize << " bytes padded) to GPU SSBO" << std::endl;
    return true;
}

void MaterialRegistry::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    entries_.clear();
    materialCount_ = 0;
    ssbo_.reset();
}

void MaterialRegistry::retire(GarbageCollector& gc) {
    std::lock_guard<std::mutex> lock(mutex_);
    gc.collect(ssbo_);
    entries_.clear();
    materialCount_ = 0;
    ssbo_.reset();
}
