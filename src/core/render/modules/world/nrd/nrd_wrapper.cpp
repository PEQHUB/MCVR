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

#include "nrd_wrapper.hpp"
#include "core/vulkan/command.hpp"
#include <algorithm>
#include <cstring>
#include <iostream>

#define VK_CHECK(result)                                                                                               \
    do {                                                                                                               \
        VkResult res = (result);                                                                                       \
        if (res != VK_SUCCESS) {                                                                                       \
            std::cerr << "Vulkan error at " << __FILE__ << ":" << __LINE__ << ": " << (int)res << std::endl;           \
        }                                                                                                              \
    } while (0)

static const VkFormat g_NRDFormatToVkFormat[] = {
    // R8
    VK_FORMAT_R8_UNORM,
    VK_FORMAT_R8_SNORM,
    VK_FORMAT_R8_UINT,
    VK_FORMAT_R8_SINT,
    // RG8
    VK_FORMAT_R8G8_UNORM,
    VK_FORMAT_R8G8_SNORM,
    VK_FORMAT_R8G8_UINT,
    VK_FORMAT_R8G8_SINT,
    // RGBA8
    VK_FORMAT_R8G8B8A8_UNORM,
    VK_FORMAT_R8G8B8A8_SNORM,
    VK_FORMAT_R8G8B8A8_UINT,
    VK_FORMAT_R8G8B8A8_SINT,
    VK_FORMAT_R8G8B8A8_SRGB,
    // R16
    VK_FORMAT_R16_UNORM,
    VK_FORMAT_R16_SNORM,
    VK_FORMAT_R16_UINT,
    VK_FORMAT_R16_SINT,
    VK_FORMAT_R16_SFLOAT,
    // RG16
    VK_FORMAT_R16G16_UNORM,
    VK_FORMAT_R16G16_SNORM,
    VK_FORMAT_R16G16_UINT,
    VK_FORMAT_R16G16_SINT,
    VK_FORMAT_R16G16_SFLOAT,
    // RGBA16
    VK_FORMAT_R16G16B16A16_UNORM,
    VK_FORMAT_R16G16B16A16_SNORM,
    VK_FORMAT_R16G16B16A16_UINT,
    VK_FORMAT_R16G16B16A16_SINT,
    VK_FORMAT_R16G16B16A16_SFLOAT,
    // R32
    VK_FORMAT_R32_UINT,
    VK_FORMAT_R32_SINT,
    VK_FORMAT_R32_SFLOAT,
    // RG32
    VK_FORMAT_R32G32_UINT,
    VK_FORMAT_R32G32_SINT,
    VK_FORMAT_R32G32_SFLOAT,
    // RGB32
    VK_FORMAT_R32G32B32_UINT,
    VK_FORMAT_R32G32B32_SINT,
    VK_FORMAT_R32G32B32_SFLOAT,
    // RGBA32
    VK_FORMAT_R32G32B32A32_UINT,
    VK_FORMAT_R32G32B32A32_SINT,
    VK_FORMAT_R32G32B32A32_SFLOAT,
    // Special
    VK_FORMAT_A2B10G10R10_UNORM_PACK32,
    VK_FORMAT_A2B10G10R10_UINT_PACK32,
    VK_FORMAT_B10G11R11_UFLOAT_PACK32,
    VK_FORMAT_E5B9G9R9_UFLOAT_PACK32,
};

static VkFormat NrdToVkFormat(nrd::Format format) {
    size_t index = size_t(format);
    if (index >= sizeof(g_NRDFormatToVkFormat) / sizeof(g_NRDFormatToVkFormat[0])) { return VK_FORMAT_UNDEFINED; }
    return g_NRDFormatToVkFormat[index];
}

NrdWrapper::NrdWrapper() = default;

NrdWrapper::~NrdWrapper() {
    if (!m_device) return;
    VkDevice device = m_device->vkDevice();

    for (auto &pipeline : m_pipelines) {
        if (pipeline.pipeline != VK_NULL_HANDLE) vkDestroyPipeline(device, pipeline.pipeline, nullptr);
        if (pipeline.pipelineLayout != VK_NULL_HANDLE)
            vkDestroyPipelineLayout(device, pipeline.pipelineLayout, nullptr);
        if (pipeline.resourceSetLayout != VK_NULL_HANDLE)
            vkDestroyDescriptorSetLayout(device, pipeline.resourceSetLayout, nullptr);
    }

    for (auto sampler : m_samplers) {
        if (sampler != VK_NULL_HANDLE) vkDestroySampler(device, sampler, nullptr);
    }

    if (m_samplerConstSetLayout != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(device, m_samplerConstSetLayout, nullptr);
    if (m_samplerConstDescriptorPool != VK_NULL_HANDLE)
        vkDestroyDescriptorPool(device, m_samplerConstDescriptorPool, nullptr);
    if (m_resourceDescriptorPool != VK_NULL_HANDLE) vkDestroyDescriptorPool(device, m_resourceDescriptorPool, nullptr);

    if (m_nrdInstance) nrd::DestroyInstance(*m_nrdInstance);
}

bool NrdWrapper::init(std::shared_ptr<vk::Device> device,
                      std::shared_ptr<vk::VMA> vma,
                      std::shared_ptr<vk::PhysicalDevice> physicalDevice,
                      uint32_t width,
                      uint32_t height,
                      uint32_t contextCount) {
    m_device = device;
    m_vma = vma;
    m_physicalDevice = physicalDevice;
    m_width = width;
    m_height = height;
    m_contextCount = contextCount;

    m_denoiserIdentifier = (nrd::Identifier)nrd::Denoiser::REBLUR_DIFFUSE_SPECULAR;

    std::vector<nrd::DenoiserDesc> denoisers{
        {nrd::Identifier(nrd::Denoiser::REBLUR_DIFFUSE_SPECULAR), nrd::Denoiser::REBLUR_DIFFUSE_SPECULAR},
    };

    nrd::InstanceCreationDesc instanceDesc{{}, denoisers.data(), uint32_t(denoisers.size())};
    nrd::Instance *tempInstance = nullptr;
    nrd::Result result = nrd::CreateInstance(instanceDesc, tempInstance);
    m_nrdInstance = tempInstance;
    if (result != nrd::Result::SUCCESS) {
        std::cerr << "Failed to create NRD instance: " << (int)result << std::endl;
        return false;
    }

    const nrd::InstanceDesc *iDesc = nrd::GetInstanceDesc(*m_nrdInstance);
    const nrd::LibraryDesc *lDesc = nrd::GetLibraryDesc();

    m_resourcesSpaceIndex = iDesc->resourcesSpaceIndex;
    m_samplersSpaceIndex = iDesc->constantBufferAndSamplersSpaceIndex;
    m_constantBufferSpaceIndex = iDesc->constantBufferAndSamplersSpaceIndex;

    m_usePushDescriptors = false;
    uint32_t extCount = 0;
    vkEnumerateDeviceExtensionProperties(m_physicalDevice->vkPhysicalDevice(), nullptr, &extCount, nullptr);
    std::vector<VkExtensionProperties> exts(extCount);
    vkEnumerateDeviceExtensionProperties(m_physicalDevice->vkPhysicalDevice(), nullptr, &extCount, exts.data());

    bool pushDescriptorSupported = false;
    for (const auto &ext : exts) {
        if (strcmp(ext.extensionName, VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME) == 0) {
            pushDescriptorSupported = true;
            break;
        }
    }

    if (pushDescriptorSupported) {
        m_usePushDescriptors = true;
#ifdef DEBUG
        std::cout << "[NrdWrapper] Push Descriptors: ENABLED" << std::endl;
#endif
    } else {
        m_usePushDescriptors = false;
        std::cerr << "[NrdWrapper] Push Descriptors: DISABLED (Not supported)" << std::endl;
    }

    m_useSeparateSets = (iDesc->resourcesSpaceIndex != iDesc->constantBufferAndSamplersSpaceIndex);

    for (uint32_t i = 0; i < iDesc->permanentPoolSize; ++i) {
        m_permanentTextures.push_back(createInternalTexture(iDesc->permanentPool[i], width, height));
    }

    for (uint32_t i = 0; i < iDesc->transientPoolSize; ++i) {
        m_transientTextures.push_back(createInternalTexture(iDesc->transientPool[i], width, height));
    }

    if (!m_permanentTextures.empty() || !m_transientTextures.empty()) {
        auto cmdPool = std::make_shared<vk::CommandPool>(m_physicalDevice, m_device);
        auto cmd = std::make_shared<vk::CommandBuffer>(m_device, cmdPool);
        cmd->begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

        std::vector<vk::CommandBuffer::ImageMemoryBarrier> barriers;
        auto addBarrier = [&](const std::shared_ptr<vk::DeviceLocalImage> &img) {
            if (!img) return;
            barriers.push_back({
                .srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                .srcAccessMask = 0,
                .dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                .dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = img,
                .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, img->layer()},
            });
        };

        for (auto &img : m_transientTextures) addBarrier(img);
        for (auto &img : m_permanentTextures) addBarrier(img);

        if (!barriers.empty()) cmd->barriersBufferImage({}, barriers);

        VkClearColorValue clearValue{};
        auto clearImage = [&](const std::shared_ptr<vk::DeviceLocalImage> &img) {
            if (!img) return;
            VkImageSubresourceRange range{};
            range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            range.baseMipLevel = 0;
            range.levelCount = 1;
            range.baseArrayLayer = 0;
            range.layerCount = img->layer();
            vkCmdClearColorImage(cmd->vkCommandBuffer(), img->vkImage(), VK_IMAGE_LAYOUT_GENERAL, &clearValue, 1,
                                 &range);
            img->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;
        };

        for (auto &img : m_transientTextures) clearImage(img);
        for (auto &img : m_permanentTextures) clearImage(img);

        cmd->end();
        cmd->submitMainQueueIndividual(m_device);
        vkQueueWaitIdle(m_device->mainVkQueue());
    }

    m_constantBuffer =
        vk::DeviceLocalBuffer::create(m_vma, m_device, iDesc->constantBufferMaxDataSize,
                                      VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);

    if (!createSamplers()) return false;
    createPipelines();

    return true;
}

void NrdWrapper::updateSettings(const nrd::CommonSettings &commonSettings, const nrd::ReblurSettings &reblurSettings) {
    if (!m_nrdInstance) return;
    nrd::SetCommonSettings(*m_nrdInstance, commonSettings);
    nrd::SetDenoiserSettings(*m_nrdInstance, m_denoiserIdentifier, &reblurSettings);
}

void NrdWrapper::denoise(VkCommandBuffer cmd,
                         uint32_t frameIndex,
                         const std::map<nrd::ResourceType, std::shared_ptr<vk::DeviceLocalImage>> &userTextures) {
    if (!m_nrdInstance) return;

    m_userTexturePool = userTextures;

    const nrd::DispatchDesc *dispatchDescs = nullptr;
    uint32_t dispatchDescsNum = 0;
    nrd::GetComputeDispatches(*m_nrdInstance, &m_denoiserIdentifier, 1, dispatchDescs, dispatchDescsNum);

    for (uint32_t i = 0; i < dispatchDescsNum; ++i) { dispatch(cmd, dispatchDescs[i], frameIndex); }
}

std::shared_ptr<vk::DeviceLocalImage>
NrdWrapper::createInternalTexture(const nrd::TextureDesc &tDesc, uint32_t width, uint32_t height) {
    uint32_t texWidth = (width + tDesc.downsampleFactor - 1) / tDesc.downsampleFactor;
    uint32_t texHeight = (height + tDesc.downsampleFactor - 1) / tDesc.downsampleFactor;
    VkFormat vkFormat = NrdToVkFormat(tDesc.format);
    return vk::DeviceLocalImage::create(m_device, m_vma, false, texWidth, texHeight, 1, vkFormat,
                                        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT |
                                            VK_IMAGE_USAGE_TRANSFER_DST_BIT);
}

bool NrdWrapper::createSamplers() {
    VkDevice device = m_device->vkDevice();
    const nrd::InstanceDesc *iDesc = nrd::GetInstanceDesc(*m_nrdInstance);
    const nrd::LibraryDesc *lDesc = nrd::GetLibraryDesc();

    m_samplers.resize(iDesc->samplersNum);
    for (uint32_t i = 0; i < iDesc->samplersNum; ++i) {
        VkSamplerCreateInfo samplerInfo = {VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        samplerInfo.magFilter = (i == (uint32_t)nrd::Sampler::NEAREST_CLAMP) ? VK_FILTER_NEAREST : VK_FILTER_LINEAR;
        samplerInfo.minFilter = samplerInfo.magFilter;
        samplerInfo.addressModeU = samplerInfo.addressModeV = samplerInfo.addressModeW =
            VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        VK_CHECK(vkCreateSampler(device, &samplerInfo, nullptr, &m_samplers[i]));
    }

    if (!m_useSeparateSets) return true;

    std::vector<VkDescriptorSetLayoutBinding> samplerBindings;
    for (uint32_t i = 0; i < iDesc->samplersNum; ++i) {
        samplerBindings.push_back({lDesc->spirvBindingOffsets.samplerOffset + iDesc->samplersBaseRegisterIndex + i,
                                   VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, &m_samplers[i]});
    }
    samplerBindings.push_back({lDesc->spirvBindingOffsets.constantBufferOffset + iDesc->constantBufferRegisterIndex,
                               VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr});

    VkDescriptorSetLayoutCreateInfo layoutInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layoutInfo.bindingCount = (uint32_t)samplerBindings.size();
    layoutInfo.pBindings = samplerBindings.data();
    VK_CHECK(vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &m_samplerConstSetLayout));

    VkDescriptorPoolSize poolSizes[] = {{VK_DESCRIPTOR_TYPE_SAMPLER, iDesc->samplersNum},
                                        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1}};
    VkDescriptorPoolCreateInfo poolInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, nullptr, 0, 1, 2, poolSizes};
    VK_CHECK(vkCreateDescriptorPool(device, &poolInfo, nullptr, &m_samplerConstDescriptorPool));

    VkDescriptorSetAllocateInfo allocInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr,
                                             m_samplerConstDescriptorPool, 1, &m_samplerConstSetLayout};
    VK_CHECK(vkAllocateDescriptorSets(device, &allocInfo, &m_samplerConstDescriptorSet));

    VkDescriptorBufferInfo bufferInfo = {m_constantBuffer->vkBuffer(), 0, VK_WHOLE_SIZE};
    VkWriteDescriptorSet cbWrite = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                    nullptr,
                                    m_samplerConstDescriptorSet,
                                    lDesc->spirvBindingOffsets.constantBufferOffset +
                                        iDesc->constantBufferRegisterIndex,
                                    0,
                                    1,
                                    VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                    nullptr,
                                    &bufferInfo,
                                    nullptr};
    vkUpdateDescriptorSets(device, 1, &cbWrite, 0, nullptr);

    return true;
}

void NrdWrapper::createPipelines() {
    VkDevice device = m_device->vkDevice();
    const nrd::InstanceDesc *iDesc = nrd::GetInstanceDesc(*m_nrdInstance);
    const nrd::LibraryDesc *lDesc = nrd::GetLibraryDesc();

    m_pipelines.resize(iDesc->pipelinesNum);

    if (!m_usePushDescriptors) {
        uint32_t setCount = iDesc->pipelinesNum * std::max(1u, m_contextCount);
        VkDescriptorPoolSize poolSizes[] = {
            {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, setCount * iDesc->descriptorPoolDesc.perSetTexturesMaxNum},
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, setCount * iDesc->descriptorPoolDesc.perSetStorageTexturesMaxNum}};
        VkDescriptorPoolCreateInfo poolInfo = {
            VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, nullptr, 0, setCount, 2, poolSizes};
        VK_CHECK(vkCreateDescriptorPool(device, &poolInfo, nullptr, &m_resourceDescriptorPool));
        m_resourceDescriptorSets.resize(iDesc->pipelinesNum);
    }

    for (uint32_t i = 0; i < iDesc->pipelinesNum; ++i) {
        const nrd::PipelineDesc &pDesc = iDesc->pipelines[i];
        NRDPipeline &nrdPipeline = m_pipelines[i];
        std::vector<VkDescriptorSetLayoutBinding> allBindings;

        for (uint32_t b = 0; b < iDesc->descriptorPoolDesc.perSetTexturesMaxNum; ++b) {
            allBindings.push_back({lDesc->spirvBindingOffsets.textureOffset + iDesc->resourcesBaseRegisterIndex + b,
                                   VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr});
        }
        for (uint32_t b = 0; b < iDesc->descriptorPoolDesc.perSetStorageTexturesMaxNum; ++b) {
            allBindings.push_back(
                {lDesc->spirvBindingOffsets.storageTextureAndBufferOffset + iDesc->resourcesBaseRegisterIndex + b,
                 VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr});
        }

        if (!m_useSeparateSets) {
            for (uint32_t s = 0; s < iDesc->samplersNum; ++s) {
                allBindings.push_back({lDesc->spirvBindingOffsets.samplerOffset + iDesc->samplersBaseRegisterIndex + s,
                                       VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, &m_samplers[s]});
            }
            allBindings.push_back({lDesc->spirvBindingOffsets.constantBufferOffset + iDesc->constantBufferRegisterIndex,
                                   VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr});
        }

        VkDescriptorSetLayoutCreateInfo layoutInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        layoutInfo.flags = m_usePushDescriptors ? VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR : 0;
        layoutInfo.bindingCount = (uint32_t)allBindings.size();
        layoutInfo.pBindings = allBindings.data();
        VK_CHECK(vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &nrdPipeline.resourceSetLayout));
        nrdPipeline.numBindings = (uint32_t)allBindings.size();

        std::vector<VkDescriptorSetLayout> setLayouts;
        if (m_useSeparateSets) {
            uint32_t maxIdx = std::max(m_resourcesSpaceIndex, m_samplersSpaceIndex);
            setLayouts.assign(maxIdx + 1, VK_NULL_HANDLE);
            setLayouts[m_resourcesSpaceIndex] = nrdPipeline.resourceSetLayout;
            setLayouts[m_samplersSpaceIndex] = m_samplerConstSetLayout;
        } else {
            setLayouts = {nrdPipeline.resourceSetLayout};
        }

        VkPipelineLayoutCreateInfo plInfo = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                             nullptr,
                                             0,
                                             (uint32_t)setLayouts.size(),
                                             setLayouts.data(),
                                             0,
                                             nullptr};
        VK_CHECK(vkCreatePipelineLayout(device, &plInfo, nullptr, &nrdPipeline.pipelineLayout));

        if (!m_usePushDescriptors) {
            m_resourceDescriptorSets[i].resize(std::max(1u, m_contextCount));
            std::vector<VkDescriptorSetLayout> layouts(m_resourceDescriptorSets[i].size(),
                                                       nrdPipeline.resourceSetLayout);
            VkDescriptorSetAllocateInfo resAllocInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr,
                                                        m_resourceDescriptorPool, (uint32_t)layouts.size(),
                                                        layouts.data()};
            VK_CHECK(vkAllocateDescriptorSets(device, &resAllocInfo, m_resourceDescriptorSets[i].data()));
        }

        VkShaderModuleCreateInfo smInfo = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0,
                                           pDesc.computeShaderSPIRV.size,
                                           (const uint32_t *)pDesc.computeShaderSPIRV.bytecode};
        VkShaderModule shaderModule;
        VK_CHECK(vkCreateShaderModule(device, &smInfo, nullptr, &shaderModule));

        VkComputePipelineCreateInfo cpInfo = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                                              nullptr,
                                              0,
                                              {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                                               VK_SHADER_STAGE_COMPUTE_BIT, shaderModule, "main", nullptr},
                                              nrdPipeline.pipelineLayout};
        VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpInfo, nullptr, &nrdPipeline.pipeline));
#ifdef DEBUG
        std::cout << "[NRD] pipeline created " << nrdPipeline.pipeline << " idx=" << i
                  << " shader=" << pDesc.shaderIdentifier << std::endl;
#endif
        vkDestroyShaderModule(device, shaderModule, nullptr);
    }
}

void NrdWrapper::dispatch(VkCommandBuffer cmd, const nrd::DispatchDesc &dispatchDesc, uint32_t frameIndex) {
    const nrd::InstanceDesc *iDesc = nrd::GetInstanceDesc(*m_nrdInstance);
    const nrd::LibraryDesc *lDesc = nrd::GetLibraryDesc();
    const nrd::PipelineDesc &pDesc = iDesc->pipelines[dispatchDesc.pipelineIndex];
    NRDPipeline &nrdPipeline = m_pipelines[dispatchDesc.pipelineIndex];

    if (dispatchDesc.name && std::strstr(dispatchDesc.name, "Clear") != nullptr) {
        std::vector<VkImageMemoryBarrier> barriers;
        VkClearColorValue clearValue{};

        auto addBarrier = [&](const std::shared_ptr<vk::DeviceLocalImage> &img) {
            if (!img) return;
            VkAccessFlags srcAccess = 0;
            VkImageLayout oldLayout = img->imageLayout();
            if (oldLayout == VK_IMAGE_LAYOUT_GENERAL) {
                srcAccess = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            } else if (oldLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
                srcAccess = VK_ACCESS_SHADER_READ_BIT;
            }
            barriers.push_back({VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                                nullptr,
                                srcAccess,
                                VK_ACCESS_TRANSFER_WRITE_BIT,
                                oldLayout,
                                VK_IMAGE_LAYOUT_GENERAL,
                                VK_QUEUE_FAMILY_IGNORED,
                                VK_QUEUE_FAMILY_IGNORED,
                                img->vkImage(),
                                {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}});
        };

        uint32_t resIdx = 0;
        for (uint32_t r = 0; r < pDesc.resourceRangesNum; ++r) {
            const nrd::ResourceRangeDesc &range = pDesc.resourceRanges[r];
            for (uint32_t b = 0; b < range.descriptorsNum; ++b) {
                const nrd::ResourceDesc &res = dispatchDesc.resources[resIdx++];
                std::shared_ptr<vk::DeviceLocalImage> image =
                    (res.type == nrd::ResourceType::PERMANENT_POOL) ? m_permanentTextures[res.indexInPool] :
                    (res.type == nrd::ResourceType::TRANSIENT_POOL) ? m_transientTextures[res.indexInPool] :
                                                                      m_userTexturePool[res.type];
                addBarrier(image);
            }
        }

        if (!barriers.empty()) {
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,
                                 nullptr, 0, nullptr, (uint32_t)barriers.size(), barriers.data());
        }

        resIdx = 0;
        for (uint32_t r = 0; r < pDesc.resourceRangesNum; ++r) {
            const nrd::ResourceRangeDesc &range = pDesc.resourceRanges[r];
            for (uint32_t b = 0; b < range.descriptorsNum; ++b) {
                const nrd::ResourceDesc &res = dispatchDesc.resources[resIdx++];
                std::shared_ptr<vk::DeviceLocalImage> image =
                    (res.type == nrd::ResourceType::PERMANENT_POOL) ? m_permanentTextures[res.indexInPool] :
                    (res.type == nrd::ResourceType::TRANSIENT_POOL) ? m_transientTextures[res.indexInPool] :
                                                                      m_userTexturePool[res.type];
                if (!image) continue;
                VkImageSubresourceRange rangeVk{};
                rangeVk.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                rangeVk.baseMipLevel = 0;
                rangeVk.levelCount = 1;
                rangeVk.baseArrayLayer = 0;
                rangeVk.layerCount = 1;
                vkCmdClearColorImage(cmd, image->vkImage(), VK_IMAGE_LAYOUT_GENERAL, &clearValue, 1, &rangeVk);
                image->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;
            }
        }
        return;
    }

    if (dispatchDesc.constantBufferDataSize > 0) {
        m_constantBuffer->uploadToStagingBuffer((void *)dispatchDesc.constantBufferData,
                                                dispatchDesc.constantBufferDataSize, 0);
        m_constantBuffer->uploadToBuffer(cmd, dispatchDesc.constantBufferDataSize, 0, 0);
        VkBufferMemoryBarrier barrier = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                                         nullptr,
                                         VK_ACCESS_TRANSFER_WRITE_BIT,
                                         VK_ACCESS_UNIFORM_READ_BIT,
                                         VK_QUEUE_FAMILY_IGNORED,
                                         VK_QUEUE_FAMILY_IGNORED,
                                         m_constantBuffer->vkBuffer(),
                                         0,
                                         dispatchDesc.constantBufferDataSize};
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr,
                             1, &barrier, 0, nullptr);
    }

    std::vector<VkImageMemoryBarrier> barriers;
    auto chooseSrcAccess = [&](VkImageLayout oldLayout) -> VkAccessFlags {
        switch (oldLayout) {
            case VK_IMAGE_LAYOUT_UNDEFINED: return 0;
            case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL: return VK_ACCESS_SHADER_READ_BIT;
            case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL: return VK_ACCESS_TRANSFER_WRITE_BIT;
            case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL: return VK_ACCESS_TRANSFER_READ_BIT;
            case VK_IMAGE_LAYOUT_GENERAL:
            default: return VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        }
    };
    auto pushBarrier = [&](const std::shared_ptr<vk::DeviceLocalImage> &img, VkImageLayout oldLayout) {
        if (!img) return;
        VkAccessFlags srcAccess = chooseSrcAccess(oldLayout);
        barriers.push_back({VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                            nullptr,
                            srcAccess,
                            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                            oldLayout,
                            VK_IMAGE_LAYOUT_GENERAL,
                            VK_QUEUE_FAMILY_IGNORED,
                            VK_QUEUE_FAMILY_IGNORED,
                            img->vkImage(),
                            {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}});
    };
    auto addPoolBarriers = [&](const std::vector<std::shared_ptr<vk::DeviceLocalImage>> &pool, bool discard = false) {
        for (auto &img : pool) {
            if (!img) continue;
            VkImageLayout oldLayout = discard ? VK_IMAGE_LAYOUT_UNDEFINED : img->imageLayout();
            pushBarrier(img, oldLayout);
        }
    };
    addPoolBarriers(m_permanentTextures, false);
    addPoolBarriers(m_transientTextures, true);
    for (auto &pair : m_userTexturePool) {
        if (!pair.second) continue;
        VkImageLayout oldLayout = pair.second->imageLayout();
        pushBarrier(pair.second, oldLayout);
    }

    if (!barriers.empty()) {
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0,
                             nullptr, 0, nullptr, (uint32_t)barriers.size(), barriers.data());
    }

    for (auto &img : m_permanentTextures)
        if (img) img->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;
    for (auto &img : m_transientTextures)
        if (img) img->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;
    for (auto &pair : m_userTexturePool) {
        if (pair.second) pair.second->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;
    }

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, nrdPipeline.pipeline);

    std::vector<VkWriteDescriptorSet> writes;
    std::vector<VkDescriptorImageInfo> imgInfos;
    std::vector<VkDescriptorBufferInfo> bufInfos;

    imgInfos.reserve(nrdPipeline.numBindings);
    writes.reserve(nrdPipeline.numBindings + 1);

    if (m_useSeparateSets) {
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, nrdPipeline.pipelineLayout, m_samplersSpaceIndex,
                                1, &m_samplerConstDescriptorSet, 0, nullptr);
    } else if (dispatchDesc.constantBufferDataSize > 0) {
        bufInfos.push_back({m_constantBuffer->vkBuffer(), 0, dispatchDesc.constantBufferDataSize});
        writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, VK_NULL_HANDLE,
                          lDesc->spirvBindingOffsets.constantBufferOffset + iDesc->constantBufferRegisterIndex, 0, 1,
                          VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &bufInfos.back(), nullptr});
    }

    uint32_t resIdx = 0;
    for (uint32_t r = 0; r < pDesc.resourceRangesNum; ++r) {
        const nrd::ResourceRangeDesc &range = pDesc.resourceRanges[r];
        for (uint32_t b = 0; b < range.descriptorsNum; ++b) {
            const nrd::ResourceDesc &res = dispatchDesc.resources[resIdx++];
            std::shared_ptr<vk::DeviceLocalImage> image =
                (res.type == nrd::ResourceType::PERMANENT_POOL) ? m_permanentTextures[res.indexInPool] :
                (res.type == nrd::ResourceType::TRANSIENT_POOL) ? m_transientTextures[res.indexInPool] :
                                                                  m_userTexturePool[res.type];
            if (!image) continue;
            imgInfos.push_back({VK_NULL_HANDLE, image->vkImageView(0), VK_IMAGE_LAYOUT_GENERAL});
            writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, VK_NULL_HANDLE,
                              ((range.descriptorType == nrd::DescriptorType::TEXTURE) ?
                                   lDesc->spirvBindingOffsets.textureOffset :
                                   lDesc->spirvBindingOffsets.storageTextureAndBufferOffset) +
                                  iDesc->resourcesBaseRegisterIndex + b,
                              0, 1,
                              (range.descriptorType == nrd::DescriptorType::TEXTURE) ?
                                  VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE :
                                  VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                              &imgInfos.back(), nullptr, nullptr});
        }
    }

    if (m_usePushDescriptors) {
        // vkCmdPushDescriptorSetKHR is loaded by volk
        // std::cout << "[NrdWrapper] Dispatching pipeline " << dispatchDesc.pipelineIndex << " with Push Descriptors"
        // << std::endl;
        uint32_t setIndex = m_useSeparateSets ? m_resourcesSpaceIndex : 0;
        vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                  m_pipelines[dispatchDesc.pipelineIndex].pipelineLayout, setIndex,
                                  (uint32_t)writes.size(), writes.data());
    } else {
        // std::cout << "[NrdWrapper] Dispatching pipeline " << dispatchDesc.pipelineIndex << " with Descriptor Sets" <<
        // std::endl;
        if (dispatchDesc.pipelineIndex >= m_resourceDescriptorSets.size() ||
            frameIndex >= m_resourceDescriptorSets[dispatchDesc.pipelineIndex].size()) {
            std::cerr << "[NrdWrapper] Error: Invalid pipeline/frame index for descriptor sets" << std::endl;
            return;
        }
        VkDescriptorSet resSet = m_resourceDescriptorSets[dispatchDesc.pipelineIndex][frameIndex];
        if (resSet == VK_NULL_HANDLE) {
            std::cerr << "[NrdWrapper] Error: Descriptor set is NULL for pipeline " << dispatchDesc.pipelineIndex
                      << " frame " << frameIndex << std::endl;
            return;
        }
        for (auto &w : writes) w.dstSet = resSet;
        vkUpdateDescriptorSets(m_device->vkDevice(), (uint32_t)writes.size(), writes.data(), 0, nullptr);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, nrdPipeline.pipelineLayout, m_resourcesSpaceIndex,
                                1, &resSet, 0, nullptr);
    }

    vkCmdDispatch(cmd, dispatchDesc.gridWidth, dispatchDesc.gridHeight, 1);
}
