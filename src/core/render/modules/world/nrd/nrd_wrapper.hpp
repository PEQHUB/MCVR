/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */


/*
 * MODIFICATIONS and INTEGRATION:
 *
 * Copyright (c) 2026 Radiance
 *
 * This file has been modified from its original version to be integrated into
 * Radiance Mod.
 *
 * Modifications include:
 * - Integration with Radiance Mod's vulkan rendering system.
 * - Refactoring of APIs, introducing RAII and supporting shared_ptr.
 *
 * These modifications are licensed under the GNU General Public License
 * as published by the Free Software Foundation; either version 3 of the License,
 * or (at your option) any later version.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <NRD.h>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <volk.h>

// Engine-level Vulkan abstractions (optional but used in current codebase)
#include "core/render/buffers.hpp"
#include "core/render/renderer.hpp"

class NrdWrapper {
  public:
    NrdWrapper();
    ~NrdWrapper();

    bool init(std::shared_ptr<vk::Device> device,
              std::shared_ptr<vk::VMA> vma,
              std::shared_ptr<vk::PhysicalDevice> physicalDevice,
              uint32_t width,
              uint32_t height,
              uint32_t contextCount);

    void updateSettings(const nrd::CommonSettings &commonSettings, const nrd::ReblurSettings &reblurSettings);

    void denoise(VkCommandBuffer cmd,
                 uint32_t frameIndex,
                 const std::map<nrd::ResourceType, std::shared_ptr<vk::DeviceLocalImage>> &userTextures);

  private:
    struct NRDPipeline {
        VkPipeline pipeline = VK_NULL_HANDLE;
        VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
        VkDescriptorSetLayout resourceSetLayout = VK_NULL_HANDLE;
        uint32_t numBindings = 0;
    };

    void createPipelines();
    bool createSamplers();
    std::shared_ptr<vk::DeviceLocalImage>
    createInternalTexture(const nrd::TextureDesc &tDesc, uint32_t width, uint32_t height);
    void dispatch(VkCommandBuffer cmd, const nrd::DispatchDesc &dispatchDesc, uint32_t frameIndex);

    nrd::Instance *m_nrdInstance = nullptr;
    nrd::Identifier m_denoiserIdentifier = (nrd::Identifier)nrd::Denoiser::REBLUR_DIFFUSE_SPECULAR;

    std::shared_ptr<vk::Device> m_device;
    std::shared_ptr<vk::PhysicalDevice> m_physicalDevice;
    std::shared_ptr<vk::VMA> m_vma;

    uint32_t m_width = 0;
    uint32_t m_height = 0;
    uint32_t m_contextCount = 0;

    std::vector<std::shared_ptr<vk::DeviceLocalImage>> m_permanentTextures;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> m_transientTextures;
    std::shared_ptr<vk::DeviceLocalBuffer> m_constantBuffer;

    std::vector<VkSampler> m_samplers;

    std::vector<NRDPipeline> m_pipelines;

    std::map<nrd::ResourceType, std::shared_ptr<vk::DeviceLocalImage>> m_userTexturePool;

    uint32_t m_resourcesSpaceIndex = 0;
    uint32_t m_samplersSpaceIndex = 1;
    uint32_t m_constantBufferSpaceIndex = 1;
    bool m_useSeparateSets = true;
    bool m_usePushDescriptors = false;

    VkDescriptorSetLayout m_samplerConstSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool m_samplerConstDescriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet m_samplerConstDescriptorSet = VK_NULL_HANDLE;

    VkDescriptorPool m_resourceDescriptorPool = VK_NULL_HANDLE;
    std::vector<std::vector<VkDescriptorSet>> m_resourceDescriptorSets;
};
