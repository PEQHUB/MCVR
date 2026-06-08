#pragma once

#include "common/shared.hpp"
#include "core/vulkan/all_core_vulkan.hpp"

#include <cstdint>
#include <mutex>
#include <vector>

class GarbageCollector;

class TextureRuleRegistry {
  public:
    TextureRuleRegistry() = default;

    bool uploadRules(const vk::Data::TextureRuleEntry* entries, uint32_t count,
                     std::shared_ptr<vk::VMA> vma,
                     std::shared_ptr<vk::Device> device);

    std::shared_ptr<vk::DeviceLocalBuffer> getBuffer() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return ssbo_;
    }

    void reset();
    void retire(GarbageCollector& gc);

  private:
    std::vector<vk::Data::TextureRuleEntry> entries_;
    std::shared_ptr<vk::DeviceLocalBuffer> ssbo_;
    mutable std::mutex mutex_;
};
