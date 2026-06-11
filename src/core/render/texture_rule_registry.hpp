#pragma once

#include "common/shared.hpp"
#include "core/vulkan/all_core_vulkan.hpp"

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

class GarbageCollector;

class TextureRuleRegistry {
  public:
    TextureRuleRegistry() = default;

    bool uploadRules(const vk::Data::TextureRuleEntry* entries, uint32_t count,
                     std::shared_ptr<vk::VMA> vma,
                     std::shared_ptr<vk::Device> device);
    bool ensureDefaultRules(std::shared_ptr<vk::VMA> vma,
                            std::shared_ptr<vk::Device> device);

    bool hasBuffer() const;
    bool usingDefaultRules() const;
    uint64_t revision() const;
    std::string statusJson() const;

    std::shared_ptr<vk::DeviceLocalBuffer> getBuffer() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return ssbo_;
    }

    void reset();
    void retire(GarbageCollector& gc);

  private:
    std::vector<vk::Data::TextureRuleEntry> entries_;
    std::shared_ptr<vk::DeviceLocalBuffer> ssbo_;
    uint64_t revision_ = 0;
    bool usingDefaultRules_ = false;
    mutable std::mutex mutex_;
};
