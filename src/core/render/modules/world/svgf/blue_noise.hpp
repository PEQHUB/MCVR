#pragma once

#include "common/shared.hpp"
#include "common/singleton.hpp"
#include "core/all_extern.hpp"
#include "core/vulkan/all_core_vulkan.hpp"

#include <memory>

class BlueNoise : public SharedObject<BlueNoise> {
  public:
    BlueNoise(std::shared_ptr<vk::Device> device, std::shared_ptr<vk::VMA> vma);

    std::shared_ptr<vk::HostVisibleBuffer> sobolBuffer();
    std::shared_ptr<vk::HostVisibleBuffer> scramblingBuffer();

    // Data sizes
    static constexpr size_t SOBOL_SIZE = 256 * 256;           // 256 samples x 256 dimensions
    static constexpr size_t SCRAMBLING_SIZE = 128 * 128 * 8;  // 128x128 tile x 8 dimensions

  private:
    std::shared_ptr<vk::HostVisibleBuffer> m_sobolBuffer;
    std::shared_ptr<vk::HostVisibleBuffer> m_scramblingBuffer;
};
