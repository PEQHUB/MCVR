#include "blue_noise.hpp"

#include <iostream>

// Include the FFX blue noise data
// The file defines global arrays:
//   - sobol_256spp_256d[256*256]
//   - scramblingTile[128*128*8]
#include "../../../extern/FidelityFX-SDK/sdk/src/components/sssr/samplerBlueNoiseErrorDistribution_128x128_OptimizedFor_2d2d2d2d_1spp.cpp"

std::ostream &blueNoiseCout() {
    return std::cout << "[BlueNoise] ";
}

BlueNoise::BlueNoise(std::shared_ptr<vk::Device> device, std::shared_ptr<vk::VMA> vma) {
    blueNoiseCout() << "Initializing blue noise buffers..." << std::endl;

    // Use HostVisibleBuffer (BAR) for constant data — no staging→device copy needed.
    // The GPU reads directly from host-visible memory. This avoids the DEVICE_LOST
    // crash caused by RT shaders reading from uninitialized device-local buffers
    // when the staging→device copy hasn't been submitted yet.
    constexpr VkBufferUsageFlags usage =
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

    // Sobol buffer (256*256 uint32 = 256KB)
    m_sobolBuffer = vk::HostVisibleBuffer::create(vma, device,
        SOBOL_SIZE * sizeof(uint32_t), usage);
    {
        std::vector<uint32_t> sobolData(SOBOL_SIZE);
        for (size_t i = 0; i < SOBOL_SIZE; i++) {
            sobolData[i] = static_cast<uint32_t>(sobol_256spp_256d[i]);
        }
        m_sobolBuffer->uploadToBuffer(sobolData.data());
        m_sobolBuffer->flush();
    }

    // Scrambling tile buffer (128*128*8 uint32 = 512KB)
    m_scramblingBuffer = vk::HostVisibleBuffer::create(vma, device,
        SCRAMBLING_SIZE * sizeof(uint32_t), usage);
    {
        std::vector<uint32_t> scramblingData(SCRAMBLING_SIZE);
        for (size_t i = 0; i < SCRAMBLING_SIZE; i++) {
            scramblingData[i] = static_cast<uint32_t>(scramblingTile[i]);
        }
        m_scramblingBuffer->uploadToBuffer(scramblingData.data());
        m_scramblingBuffer->flush();
    }

    blueNoiseCout() << "Blue noise buffers created (host-visible): Sobol(" << SOBOL_SIZE * sizeof(uint32_t) / 1024
                    << "KB), Scrambling(" << SCRAMBLING_SIZE * sizeof(uint32_t) / 1024 << "KB)" << std::endl;
}

std::shared_ptr<vk::HostVisibleBuffer> BlueNoise::sobolBuffer() {
    return m_sobolBuffer;
}

std::shared_ptr<vk::HostVisibleBuffer> BlueNoise::scramblingBuffer() {
    return m_scramblingBuffer;
}
