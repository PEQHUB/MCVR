#include "texture_service.hpp"
#include "platform/vulkan/vk2_device.hpp"
#include "frame/resource_gc.hpp"
#include "diagnostics/log.hpp"

#include <volk.h>
#include <cstring>
#include <algorithm>

namespace engine {

TextureService::~TextureService() {
    shutdown();
}

vk2::Result<void> TextureService::init(vk2::DeviceService& device, ResourceGC* gc) {
    if (initialized_) return {};
    device_ = &device;
    gc_ = gc;

    VkDevice dev = device.device();

    // Create nearest sampler (for albedo — point sample with mip nearest)
    {
        VkSamplerCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        ci.magFilter = VK_FILTER_NEAREST;
        ci.minFilter = VK_FILTER_NEAREST;
        ci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        ci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        ci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        ci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        ci.maxLod = 0.0f;
        VkResult r = vkCreateSampler(dev, &ci, nullptr, &nearestSampler_);
        if (r != VK_SUCCESS) {
            return vk2::Error{r, "TextureService: failed to create nearest sampler"};
        }
    }

    // Create linear sampler (for specular/normal)
    {
        VkSamplerCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        ci.magFilter = VK_FILTER_LINEAR;
        ci.minFilter = VK_FILTER_LINEAR;
        ci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        ci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        ci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        ci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        ci.maxLod = 0.0f;
        VkResult r = vkCreateSampler(dev, &ci, nullptr, &linearSampler_);
        if (r != VK_SUCCESS) {
            destroySamplers();
            return vk2::Error{r, "TextureService: failed to create linear sampler"};
        }
    }

    initialized_ = true;
    log::info("texture", "TextureService initialized");
    return {};
}

void TextureService::shutdown() {
    if (!initialized_) return;

    albedoArray_ = {};
    specularArray_ = {};
    normalArray_ = {};
    spriteRegistrySSBO_ = {};
    stagingBuffer_ = {};

    destroySamplers();

    spriteTable_.clear();
    albedoPixels_.clear();
    specularPixels_.clear();
    normalPixels_.clear();
    animData_.clear();
    totalAlbedoLayers_ = 0;

    finalized_ = false;
    pendingFinalize_ = false;
    initialized_ = false;
    log::info("texture", "TextureService shut down");
}

void TextureService::destroySamplers() {
    if (!device_) return;
    VkDevice dev = device_->device();
    if (nearestSampler_ != VK_NULL_HANDLE) {
        vkDestroySampler(dev, nearestSampler_, nullptr);
        nearestSampler_ = VK_NULL_HANDLE;
    }
    if (linearSampler_ != VK_NULL_HANDLE) {
        vkDestroySampler(dev, linearSampler_, nullptr);
        linearSampler_ = VK_NULL_HANDLE;
    }
}

VkDeviceSize TextureService::spriteRegistrySize() const {
    return static_cast<VkDeviceSize>(spriteCount_) * sizeof(SpriteEntry);
}

void TextureService::receiveSpriteTable(const uint8_t* metadata, uint32_t count,
                                        uint32_t atlasW, uint32_t atlasH, uint32_t spriteSize) {
    spriteCount_ = count;
    atlasWidth_ = atlasW;
    atlasHeight_ = atlasH;
    spriteSize_ = spriteSize;

    spriteTable_.resize(count);
    if (count > 0 && metadata) {
        std::memcpy(spriteTable_.data(), metadata, count * sizeof(SpriteMetadata));
    }

    pendingFinalize_ = true;
    log::info("texture", "Received sprite table: " + std::to_string(count)
              + " sprites, atlas " + std::to_string(atlasW) + "x" + std::to_string(atlasH)
              + ", spriteSize=" + std::to_string(spriteSize));
}

void TextureService::receiveSpritePixels(const uint8_t* data, size_t size) {
    if (data && size > 0) {
        albedoPixels_.assign(data, data + size);
    }
    log::info("texture", "Received sprite pixels: " + std::to_string(size) + " bytes");
}

void TextureService::receiveAuxPixels(const uint8_t* specular, size_t specSize,
                                      const uint8_t* normal, size_t normSize) {
    if (specular && specSize > 0) {
        specularPixels_.assign(specular, specular + specSize);
    }
    if (normal && normSize > 0) {
        normalPixels_.assign(normal, normal + normSize);
    }
    log::info("texture", "Received aux pixels: spec=" + std::to_string(specSize)
              + " norm=" + std::to_string(normSize));
}

void TextureService::receiveAnimationFrames(const uint8_t* data, size_t size) {
    if (!data || size == 0) {
        log::info("texture", "Animation frames: empty (no animated sprites)");
        return;
    }

    // Parse format: repeated [spriteId(u16), frameIndex(u16), pixels(w*h*4)] entries.
    size_t layerBytes = static_cast<size_t>(spriteSize_) * spriteSize_ * 4;
    size_t offset = 0;
    uint32_t framesParsed = 0;

    while (offset + 4 <= size) {
        uint16_t spriteId, frameIndex;
        std::memcpy(&spriteId, data + offset, 2);
        std::memcpy(&frameIndex, data + offset + 2, 2);
        offset += 4;

        if (offset + layerBytes > size) {
            log::warn("texture", "Animation frame data truncated at sprite "
                      + std::to_string(spriteId) + " frame " + std::to_string(frameIndex));
            break;
        }

        // Look up frameCount from sprite table to pre-allocate
        uint16_t frameCount = 1;
        if (spriteId < spriteTable_.size()) {
            frameCount = std::max<uint16_t>(spriteTable_[spriteId].frameCount, 1);
        }

        auto& af = animData_[spriteId];
        if (af.frames.empty()) {
            af.frames.resize(frameCount);
        }

        if (frameIndex < af.frames.size()) {
            af.frames[frameIndex].assign(data + offset, data + offset + layerBytes);
        }

        offset += layerBytes;
        ++framesParsed;
    }

    log::info("texture", "Animation frames parsed: " + std::to_string(framesParsed)
              + " frames for " + std::to_string(animData_.size()) + " animated sprites");
}

bool TextureService::finalize(VkCommandBuffer cmd) {
    if (!initialized_ || !pendingFinalize_ || spriteCount_ == 0) return false;

    VkDevice dev = device_->device();
    VmaAllocator vma = device_->vma();

    uint32_t layerW = spriteSize_;
    uint32_t layerH = spriteSize_;
    size_t layerBytes = static_cast<size_t>(layerW) * layerH * 4;

    // Validate frame-0 pixel data
    size_t frame0Size = layerBytes * spriteCount_;
    if (albedoPixels_.size() < frame0Size) {
        log::warn("texture", "Albedo pixels too small: got " + std::to_string(albedoPixels_.size())
                  + " expected " + std::to_string(frame0Size) + " — padding with zeros");
        albedoPixels_.resize(frame0Size, 0);
    }

    // --- 1. Compute layer counts ---
    // Albedo: animated sprites get N consecutive layers; static sprites get 1.
    // Specular/Normal: 1 layer per sprite (static), same count as spriteCount_.
    // Build per-sprite base layer mapping.
    std::vector<uint32_t> albedoBaseLayer(spriteCount_);
    uint32_t totalAlbedoLayers = 0;
    uint32_t animatedCount = 0;
    for (uint32_t i = 0; i < spriteCount_; ++i) {
        albedoBaseLayer[i] = totalAlbedoLayers;
        uint32_t frames = std::max<uint32_t>(spriteTable_[i].frameCount, 1);
        totalAlbedoLayers += frames;
        if (frames > 1) ++animatedCount;
    }
    totalAlbedoLayers_ = totalAlbedoLayers;

    // Ensure at least 1 layer for degenerate cases
    if (totalAlbedoLayers == 0) totalAlbedoLayers = 1;

    // --- 2. Prepare specular/normal data ---
    bool hasSpecular = !specularPixels_.empty();
    bool hasNormal = !normalPixels_.empty();
    if (hasSpecular && specularPixels_.size() < frame0Size) {
        specularPixels_.resize(frame0Size, 0);
    }
    if (hasNormal && normalPixels_.size() < frame0Size) {
        normalPixels_.resize(frame0Size, 0);
    }
    if (!hasSpecular) {
        specularPixels_.resize(frame0Size, 0);
    }
    if (!hasNormal) {
        normalPixels_.resize(frame0Size, 0);
        for (size_t i = 0; i < normalPixels_.size(); i += 4) {
            normalPixels_[i + 0] = 128;
            normalPixels_[i + 1] = 128;
            normalPixels_[i + 2] = 255;
            normalPixels_[i + 3] = 255;
        }
    }

    // --- 3. Build albedo staging with animation frames ---
    // Layout: totalAlbedoLayers consecutive layers, then spriteCount_ specular, then spriteCount_ normal.
    size_t albedoTotalBytes = static_cast<size_t>(totalAlbedoLayers) * layerBytes;
    size_t specNormBytes = frame0Size; // spriteCount_ layers each
    size_t totalStagingSize = albedoTotalBytes + specNormBytes * 2;

    // Assemble the albedo staging data: frame-0 for all sprites, plus extra frames for animated ones.
    std::vector<uint8_t> albedoStaging(albedoTotalBytes, 0);
    for (uint32_t i = 0; i < spriteCount_; ++i) {
        uint32_t baseLayer = albedoBaseLayer[i];
        uint32_t frames = std::max<uint32_t>(spriteTable_[i].frameCount, 1);
        size_t srcOffset = static_cast<size_t>(i) * layerBytes;

        // Layer 0: always the frame-0 albedo pixels from receiveSpritePixels.
        size_t dstOffset = static_cast<size_t>(baseLayer) * layerBytes;
        if (srcOffset + layerBytes <= albedoPixels_.size()) {
            std::memcpy(albedoStaging.data() + dstOffset, albedoPixels_.data() + srcOffset, layerBytes);
        }

        // Layers 1..N-1: animation frames from animData_.
        if (frames > 1) {
            auto it = animData_.find(static_cast<uint16_t>(i));
            for (uint32_t f = 1; f < frames; ++f) {
                size_t frameDstOffset = static_cast<size_t>(baseLayer + f) * layerBytes;
                if (it != animData_.end() && f < it->second.frames.size()
                    && !it->second.frames[f].empty()) {
                    std::memcpy(albedoStaging.data() + frameDstOffset,
                                it->second.frames[f].data(),
                                std::min(it->second.frames[f].size(), layerBytes));
                } else {
                    // Missing frame — duplicate frame 0
                    std::memcpy(albedoStaging.data() + frameDstOffset,
                                albedoStaging.data() + dstOffset, layerBytes);
                }
            }
            // Overwrite layer 0 with animData frame 0 if present (it may differ from the
            // frame-0 pixels sent via receiveSpritePixels in some edge cases).
            if (it != animData_.end() && !it->second.frames.empty()
                && !it->second.frames[0].empty()) {
                std::memcpy(albedoStaging.data() + dstOffset,
                            it->second.frames[0].data(),
                            std::min(it->second.frames[0].size(), layerBytes));
            }
        }
    }

    // --- 4. Create GPU images ---
    auto createArray = [&](VkFormat format, uint32_t layers) -> vk2::Result<vk2::Image> {
        vk2::Image::Desc desc;
        desc.width = layerW;
        desc.height = layerH;
        desc.layers = layers;
        desc.mipLevels = 1;
        desc.format = format;
        desc.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        desc.imageType = VK_IMAGE_TYPE_2D;
        return vk2::Image::create(dev, vma, desc);
    };

    auto albedoR = createArray(VK_FORMAT_R8G8B8A8_SRGB, totalAlbedoLayers);
    if (!albedoR) {
        log::error("texture", "Failed to create albedo array: " + albedoR.error().message);
        return false;
    }
    albedoArray_ = std::move(albedoR.value());

    auto specR = createArray(VK_FORMAT_R8G8B8A8_UNORM, spriteCount_);
    if (!specR) {
        log::error("texture", "Failed to create specular array: " + specR.error().message);
        return false;
    }
    specularArray_ = std::move(specR.value());

    auto normR = createArray(VK_FORMAT_R8G8B8A8_UNORM, spriteCount_);
    if (!normR) {
        log::error("texture", "Failed to create normal array: " + normR.error().message);
        return false;
    }
    normalArray_ = std::move(normR.value());

    // --- 5. Create staging buffer and copy pixel data ---
    vk2::Buffer::Desc stagingDesc;
    stagingDesc.size = totalStagingSize;
    stagingDesc.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    stagingDesc.vmaFlags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                           VMA_ALLOCATION_CREATE_MAPPED_BIT;
    stagingDesc.memoryUsage = VMA_MEMORY_USAGE_AUTO;

    auto stagingR = vk2::Buffer::create(dev, vma, stagingDesc);
    if (!stagingR) {
        log::error("texture", "Failed to create staging buffer: " + stagingR.error().message);
        return false;
    }
    stagingBuffer_ = std::move(stagingR.value());

    uint8_t* mapped = static_cast<uint8_t*>(stagingBuffer_.mappedPtr());
    std::memcpy(mapped, albedoStaging.data(), albedoTotalBytes);
    std::memcpy(mapped + albedoTotalBytes, specularPixels_.data(), specNormBytes);
    std::memcpy(mapped + albedoTotalBytes + specNormBytes, normalPixels_.data(), specNormBytes);

    // --- 6. Transition images: UNDEFINED -> TRANSFER_DST_OPTIMAL ---
    VkImageMemoryBarrier2 preCopyBarriers[3]{};
    VkImage images[3] = {albedoArray_.handle(), specularArray_.handle(), normalArray_.handle()};
    uint32_t layerCounts[3] = {totalAlbedoLayers, spriteCount_, spriteCount_};
    for (int i = 0; i < 3; ++i) {
        preCopyBarriers[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        preCopyBarriers[i].srcStageMask = VK_PIPELINE_STAGE_2_NONE;
        preCopyBarriers[i].srcAccessMask = VK_ACCESS_2_NONE;
        preCopyBarriers[i].dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        preCopyBarriers[i].dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        preCopyBarriers[i].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        preCopyBarriers[i].newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        preCopyBarriers[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        preCopyBarriers[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        preCopyBarriers[i].image = images[i];
        preCopyBarriers[i].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, layerCounts[i]};
    }

    VkDependencyInfo preDep{};
    preDep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    preDep.imageMemoryBarrierCount = 3;
    preDep.pImageMemoryBarriers = preCopyBarriers;
    vkCmdPipelineBarrier2(cmd, &preDep);

    // --- 7. Record buffer-to-image copies ---
    auto recordCopies = [&](VkImage image, VkDeviceSize bufferOffset, uint32_t layers) {
        std::vector<VkBufferImageCopy> regions(layers);
        for (uint32_t layer = 0; layer < layers; ++layer) {
            regions[layer] = {};
            regions[layer].bufferOffset = bufferOffset + static_cast<VkDeviceSize>(layer) * layerBytes;
            regions[layer].bufferRowLength = 0;
            regions[layer].bufferImageHeight = 0;
            regions[layer].imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            regions[layer].imageSubresource.mipLevel = 0;
            regions[layer].imageSubresource.baseArrayLayer = layer;
            regions[layer].imageSubresource.layerCount = 1;
            regions[layer].imageOffset = {0, 0, 0};
            regions[layer].imageExtent = {layerW, layerH, 1};
        }
        vkCmdCopyBufferToImage(cmd, stagingBuffer_.handle(), image,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                               static_cast<uint32_t>(regions.size()), regions.data());
    };

    recordCopies(albedoArray_.handle(), 0, totalAlbedoLayers);
    recordCopies(specularArray_.handle(), albedoTotalBytes, spriteCount_);
    recordCopies(normalArray_.handle(), albedoTotalBytes + specNormBytes, spriteCount_);

    // --- 8. Transition images: TRANSFER_DST -> SHADER_READ_ONLY_OPTIMAL ---
    VkImageMemoryBarrier2 postCopyBarriers[3]{};
    for (int i = 0; i < 3; ++i) {
        postCopyBarriers[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        postCopyBarriers[i].srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        postCopyBarriers[i].srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        postCopyBarriers[i].dstStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
        postCopyBarriers[i].dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
        postCopyBarriers[i].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        postCopyBarriers[i].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        postCopyBarriers[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        postCopyBarriers[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        postCopyBarriers[i].image = images[i];
        postCopyBarriers[i].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, layerCounts[i]};
    }

    VkDependencyInfo postDep{};
    postDep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    postDep.imageMemoryBarrierCount = 3;
    postDep.pImageMemoryBarriers = postCopyBarriers;
    vkCmdPipelineBarrier2(cmd, &postDep);

    // --- 9. Build SpriteRegistry SSBO (matches V1 shared.hpp SpriteEntry) ---
    {
        std::vector<SpriteEntry> entries(spriteCount_);

        for (uint32_t i = 0; i < spriteCount_; ++i) {
            const auto& meta = spriteTable_[i];
            uint32_t frames = std::max<uint32_t>(meta.frameCount, 1);
            uint32_t tickRate = std::max<uint32_t>(meta.tickRate, 1);

            entries[i].baseLayer     = albedoBaseLayer[i];
            entries[i].frameCount    = frames;
            entries[i].tickRate      = tickRate;
            entries[i].flags         = static_cast<uint32_t>(meta.flags);
            entries[i].specularLayer = static_cast<int32_t>(i);   // 1:1 specular layers
            entries[i].normalLayer   = static_cast<int32_t>(i);   // 1:1 normal layers
            entries[i].overlaySprite = static_cast<int32_t>(meta.overlayOf);
            entries[i].maskLayer     = -1;  // no material class mask in V2 yet
        }

        vk2::Buffer::Desc ssboDesc;
        ssboDesc.size = entries.size() * sizeof(SpriteEntry);
        ssboDesc.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        ssboDesc.memoryUsage = VMA_MEMORY_USAGE_AUTO;
        ssboDesc.vmaFlags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                            VMA_ALLOCATION_CREATE_MAPPED_BIT;

        auto ssboR = vk2::Buffer::create(dev, vma, ssboDesc);
        if (!ssboR) {
            log::error("texture", "Failed to create sprite registry SSBO: " + ssboR.error().message);
            return false;
        }
        spriteRegistrySSBO_ = std::move(ssboR.value());
        std::memcpy(spriteRegistrySSBO_.mappedPtr(), entries.data(), entries.size() * sizeof(SpriteEntry));
        spriteRegistrySSBO_.flush(); // Ensure GPU-visible on non-coherent heaps

        // DIAG: dump first 8 entries so we can verify per-sprite baseLayer differs
        for (uint32_t i = 0; i < std::min<uint32_t>(8u, spriteCount_); ++i) {
            log::warn("app",
                "DIAG reg[" + std::to_string(i) + "] baseLayer=" + std::to_string(entries[i].baseLayer)
                + " frames=" + std::to_string(entries[i].frameCount)
                + " atlasXY=" + std::to_string(spriteTable_[i].x) + "," + std::to_string(spriteTable_[i].y)
                + " WH=" + std::to_string(spriteTable_[i].w) + "x" + std::to_string(spriteTable_[i].h));
        }
    }

    // --- 10. Defer staging buffer destruction ---
    if (gc_) {
        auto retired = std::make_shared<vk2::Buffer>(std::move(stagingBuffer_));
        gc_->defer([retired]() { /* shared_ptr destructor releases Buffer */ });
    }

    // --- 10b. Snapshot per-sprite atlas bounds before spriteTable_ is freed.
    // BlockModelTable::normalizeQuadUVsWithBounds() needs these to convert
    // atlas-space model UVs into sprite-local [0,1] UVs (called by
    // engine_app::processScene right after this finalize() returns).
    {
        atlasBounds_.assign(spriteCount_, glm::vec4(0.0f, 1.0f, 0.0f, 1.0f));
        const float aw = atlasWidth_  > 0 ? float(atlasWidth_)  : 1.0f;
        const float ah = atlasHeight_ > 0 ? float(atlasHeight_) : 1.0f;
        for (uint32_t i = 0; i < spriteCount_; ++i) {
            const auto& m = spriteTable_[i];
            atlasBounds_[i] = glm::vec4(
                float(m.x) / aw,
                float(m.x + m.w) / aw,
                float(m.y) / ah,
                float(m.y + m.h) / ah);
        }
    }

    // --- 11. Free CPU-side pixel data ---
    albedoPixels_.clear();
    albedoPixels_.shrink_to_fit();
    specularPixels_.clear();
    specularPixels_.shrink_to_fit();
    normalPixels_.clear();
    normalPixels_.shrink_to_fit();
    spriteTable_.clear();
    spriteTable_.shrink_to_fit();
    animData_.clear();

    finalized_ = true;
    pendingFinalize_ = false;

    log::info("texture", "TextureService finalized: " + std::to_string(spriteCount_)
              + " sprites (" + std::to_string(animatedCount) + " animated), "
              + std::to_string(totalAlbedoLayers) + " albedo layers, "
              + std::to_string(layerW) + "x" + std::to_string(layerH) + " px/layer");
    return true;
}
} // namespace engine
