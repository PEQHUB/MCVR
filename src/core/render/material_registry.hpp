#pragma once

#include "common/shared.hpp"
#include "core/vulkan/all_core_vulkan.hpp"

#include <cstdint>
#include <mutex>
#include <vector>

class GarbageCollector;

class MaterialRegistry {
  public:
    MaterialRegistry() = default;

    bool uploadMaterials(const vk::Data::MaterialEntry* entries, uint32_t count,
                         std::shared_ptr<vk::VMA> vma,
                         std::shared_ptr<vk::Device> device);
    bool updateMaterialsSparse(const vk::Data::MaterialEntry* entries, uint32_t count,
                               std::shared_ptr<vk::VMA> vma,
                               std::shared_ptr<vk::Device> device);

    std::shared_ptr<vk::DeviceLocalBuffer> getBuffer() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return ssbo_;
    }

    uint32_t count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return materialCount_;
    }

    void reset();
    void retire(GarbageCollector& gc);

  private:
    std::vector<vk::Data::MaterialEntry> entries_;
    uint32_t materialCount_ = 0;
    std::shared_ptr<vk::DeviceLocalBuffer> ssbo_;
    mutable std::mutex mutex_;
};
