#include "core/render/texture_arrays.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>

namespace {
constexpr VkPipelineStageFlags2 kTextureReadStages =
    VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
    VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;

size_t subresourceIndex(const TextureArrayManager::ArrayInfo& info, uint32_t layer, uint32_t mip) {
    return static_cast<size_t>(layer) * info.mipLevels + mip;
}

VkImageLayout getTrackedLayout(const TextureArrayManager::ArrayInfo& info, uint32_t layer, uint32_t mip) {
    if (layer >= info.layerCount || mip >= info.mipLevels) return VK_IMAGE_LAYOUT_UNDEFINED;
    size_t index = subresourceIndex(info, layer, mip);
    if (index >= info.subresourceLayouts.size()) return VK_IMAGE_LAYOUT_UNDEFINED;
    return info.subresourceLayouts[index];
}

void setTrackedLayout(TextureArrayManager::ArrayInfo& info, uint32_t layer, uint32_t mip, VkImageLayout layout) {
    if (layer >= info.layerCount || mip >= info.mipLevels) return;
    size_t index = subresourceIndex(info, layer, mip);
    if (index >= info.subresourceLayouts.size()) return;
    info.subresourceLayouts[index] = layout;
}

VkPipelineStageFlags2 sourceStageFor(VkImageLayout layout) {
    switch (layout) {
        case VK_IMAGE_LAYOUT_UNDEFINED:
            return VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
        case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
            return VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
            return VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
            return kTextureReadStages;
        default:
            return VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    }
}

VkAccessFlags2 sourceAccessFor(VkImageLayout layout) {
    switch (layout) {
        case VK_IMAGE_LAYOUT_UNDEFINED:
            return 0;
        case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
            return VK_ACCESS_2_TRANSFER_WRITE_BIT;
        case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
            return VK_ACCESS_2_TRANSFER_READ_BIT;
        case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
            return VK_ACCESS_2_SHADER_READ_BIT;
        default:
            return VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
    }
}

VkPipelineStageFlags2 destinationStageFor(VkImageLayout layout) {
    switch (layout) {
        case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
            return VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
            return VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
            return kTextureReadStages;
        default:
            return VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    }
}

VkAccessFlags2 destinationAccessFor(VkImageLayout layout) {
    switch (layout) {
        case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
            return VK_ACCESS_2_TRANSFER_WRITE_BIT;
        case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
            return VK_ACCESS_2_TRANSFER_READ_BIT;
        case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
            return VK_ACCESS_2_SHADER_READ_BIT;
        default:
            return VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
    }
}

void transitionSubresource(TextureArrayManager::ArrayInfo& info,
                           std::shared_ptr<vk::CommandBuffer> cmdBuffer,
                           uint32_t layer,
                           uint32_t mip,
                           VkImageLayout newLayout) {
    VkImageLayout oldLayout = getTrackedLayout(info, layer, mip);
    if (oldLayout == newLayout) return;

    VkImageSubresourceRange range{};
    range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    range.baseMipLevel = mip;
    range.levelCount = 1;
    range.baseArrayLayer = layer;
    range.layerCount = 1;

    cmdBuffer->barriersBufferImage({}, {{
        .srcStageMask = sourceStageFor(oldLayout),
        .srcAccessMask = sourceAccessFor(oldLayout),
        .dstStageMask = destinationStageFor(newLayout),
        .dstAccessMask = destinationAccessFor(newLayout),
        .oldLayout = oldLayout,
        .newLayout = newLayout,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = info.image,
        .subresourceRange = range,
    }});

    setTrackedLayout(info, layer, mip, newLayout);
}

void generateMipChainForLayer(TextureArrayManager::ArrayInfo& info,
                              std::shared_ptr<vk::CommandBuffer> cmdBuffer,
                              uint32_t layer) {
    int32_t mipWidth = static_cast<int32_t>(info.spriteSize);
    int32_t mipHeight = static_cast<int32_t>(info.spriteSize);

    for (uint32_t mip = 1; mip < info.mipLevels; mip++) {
        transitionSubresource(info, cmdBuffer, layer, mip - 1,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        transitionSubresource(info, cmdBuffer, layer, mip,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

        int32_t nextWidth = mipWidth > 1 ? mipWidth / 2 : 1;
        int32_t nextHeight = mipHeight > 1 ? mipHeight / 2 : 1;

        VkImageBlit blit{};
        blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.srcSubresource.mipLevel = mip - 1;
        blit.srcSubresource.baseArrayLayer = layer;
        blit.srcSubresource.layerCount = 1;
        blit.srcOffsets[0] = {0, 0, 0};
        blit.srcOffsets[1] = {mipWidth, mipHeight, 1};
        blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.dstSubresource.mipLevel = mip;
        blit.dstSubresource.baseArrayLayer = layer;
        blit.dstSubresource.layerCount = 1;
        blit.dstOffsets[0] = {0, 0, 0};
        blit.dstOffsets[1] = {nextWidth, nextHeight, 1};

        vkCmdBlitImage(cmdBuffer->vkCommandBuffer(),
            info.image->vkImage(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            info.image->vkImage(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1, &blit, VK_FILTER_LINEAR);

        transitionSubresource(info, cmdBuffer, layer, mip - 1,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        transitionSubresource(info, cmdBuffer, layer, mip,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

        mipWidth = nextWidth;
        mipHeight = nextHeight;
    }
}
}

uint32_t TextureArrayManager::computeMipLevels(uint32_t spriteSize) {
    return static_cast<uint32_t>(std::floor(std::log2(spriteSize))) + 1;
}

uint32_t TextureArrayManager::createArray(std::shared_ptr<vk::VMA> vma,
                                           std::shared_ptr<vk::Device> device,
                                           uint32_t spriteSize,
                                           uint32_t layerCount,
                                           VkFormat format,
                                           bool generateMips) {
    std::lock_guard<std::mutex> lock(mutex_);

    uint32_t mipLevels = generateMips ? computeMipLevels(spriteSize) : 1;
    uint32_t id = nextArrayId_++;

    VkImageUsageFlags usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    if (generateMips && !vk::formatIsBlockCompressed(format)) {
        usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT; // needed for vkCmdBlitImage mip gen
    }

    auto image = vk::DeviceLocalImage::create(
        device, vma, false, mipLevels,
        spriteSize, spriteSize, layerCount,
        format, usage, 0, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);

    auto sampler = vk::Sampler::create(
        device, VK_FILTER_NEAREST, VK_SAMPLER_MIPMAP_MODE_NEAREST,
        VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);

	ArrayInfo info;
	info.image = image;
	info.sampler = sampler;
	info.spriteSize = spriteSize;
	info.layerCount = layerCount;
	info.mipLevels = mipLevels;
	info.format = format;
	info.subresourceLayouts.assign(
		static_cast<size_t>(layerCount) * mipLevels,
		VK_IMAGE_LAYOUT_UNDEFINED);
	arrays_[id] = std::move(info);
    std::cout << "[TextureArrayManager] Created array " << id
              << ": " << spriteSize << "x" << spriteSize
              << " x " << layerCount << " layers"
              << ", " << mipLevels << " mips"
              << ", format=" << format << std::endl;

    return id;
}

void TextureArrayManager::stageLayerPixels(uint32_t arrayId,
                                            uint32_t layer,
                                            uint32_t mipLevel,
                                            const uint8_t* pixels,
                                            size_t pixelSize) {
    std::lock_guard<std::mutex> lock(mutex_);

    StagedUpload upload;
    upload.arrayId = arrayId;
    upload.layer = layer;
    upload.mipLevel = mipLevel;
    upload.pixels.assign(pixels, pixels + pixelSize);
    stagedUploads_.push_back(std::move(upload));
}

TextureArrayManager::DirtyLayers TextureArrayManager::flushUploads(
	std::shared_ptr<vk::VMA> vma,
	std::shared_ptr<vk::Device> device,
	std::shared_ptr<vk::CommandBuffer> cmdBuffer,
	uint32_t albedoId,
	uint32_t specId,
	uint32_t normId,
	size_t maxBytes) {
	DirtyLayers dirty;

	// Phase 1: Snapshot staged uploads under mutex (~1us for 51 entries).
	// After the move, the Java thread can immediately stage new uploads
	// to the now-empty stagedUploads_ -- no contention.
	std::vector<StagedUpload> snapshot;
	{
		std::lock_guard<std::mutex> lock(mutex_);
		if (stagedUploads_.empty()) return dirty;
		snapshot = std::move(stagedUploads_);
		// stagedUploads_ is now empty -- Java thread can stage new uploads immediately
	}

	currentFrameStagingBuffers_.clear();

	// Phase 2: Process snapshot WITHOUT mutex (1-3ms).
	// arrays_ is read-only during rendering (only written during finalize after vkDeviceWaitIdle).
	// subresourceLayouts_ is single-writer (render thread only).
	// currentFrameStagingBuffers_ is single-thread (render thread only).

	for (auto& upload : snapshot) {
		auto it = arrays_.find(upload.arrayId);
		if (it == arrays_.end()) {
			std::cerr << "[TextureArrayManager] Unknown array ID " << upload.arrayId << std::endl;
			continue;
		}

		auto& info = it->second;
		auto& image = info.image;

		if (upload.layer >= info.layerCount) {
			std::cerr << "[TextureArrayManager] Layer " << upload.layer
				<< " out of range (max " << info.layerCount << ") for array "
				<< upload.arrayId << ", skipping" << std::endl;
			continue;
		}

		uint32_t mipSize = info.spriteSize >> upload.mipLevel;
		if (mipSize == 0) mipSize = 1;

		size_t bytesPerPixel = vk::formatToByte(info.format);
		size_t expectedSize = mipSize * mipSize * bytesPerPixel;
		if (upload.pixels.size() < expectedSize) {
			std::cerr << "[TextureArrayManager] Pixel data too small for array "
				<< upload.arrayId << " layer " << upload.layer
				<< " mip " << upload.mipLevel << std::endl;
			continue;
		}

		// Create staging buffer for this layer
		auto staging = vk::HostVisibleBuffer::create(
			vma, device, upload.pixels.size(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
		staging->uploadToBuffer(upload.pixels.data());
		staging->flush();
		currentFrameStagingBuffers_.push_back(staging);

		transitionSubresource(info, cmdBuffer, upload.layer, upload.mipLevel,
			VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

		// Copy staging buffer to image layer
		VkBufferImageCopy region{};
		region.bufferOffset = 0;
		region.bufferRowLength = 0;
		region.bufferImageHeight = 0;
		region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		region.imageSubresource.mipLevel = upload.mipLevel;
		region.imageSubresource.baseArrayLayer = upload.layer;
		region.imageSubresource.layerCount = 1;
		region.imageOffset = {0, 0, 0};
		region.imageExtent = {mipSize, mipSize, 1};

		vkCmdCopyBufferToImage(cmdBuffer->vkCommandBuffer(),
			staging->vkBuffer(), image->vkImage(),
			VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			1, &region);

		transitionSubresource(info, cmdBuffer, upload.layer, upload.mipLevel,
			VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

		// Track dirty layers for mipgen
		if (upload.arrayId == albedoId) dirty.albedo.push_back(upload.layer);
		else if (upload.arrayId == specId) dirty.specular.push_back(upload.layer);
		else if (upload.arrayId == normId) dirty.normal.push_back(upload.layer);
	}

	return dirty;
}

void TextureArrayManager::generateMipmaps(uint32_t arrayId,
                                           std::shared_ptr<vk::CommandBuffer> cmdBuffer) {
    auto it = arrays_.find(arrayId);
    if (it == arrays_.end()) return;

    auto& info = it->second;

    // BC7 mips must be pre-computed on CPU — skip GPU mip gen
    if (vk::formatIsBlockCompressed(info.format)) return;
    if (info.mipLevels <= 1) return;

    for (uint32_t layer = 0; layer < info.layerCount; layer++) {
        generateMipChainForLayer(info, cmdBuffer, layer);
    }
    info.image->imageLayout() = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    std::cout << "[TextureArrayManager] Generated mipmaps for array " << arrayId
              << " (" << info.layerCount << " layers, " << info.mipLevels << " mips)" << std::endl;
}

void TextureArrayManager::generateMipmapsForLayer(uint32_t arrayId, uint32_t layer,
                                                    std::shared_ptr<vk::CommandBuffer> cmdBuffer) {
    auto it = arrays_.find(arrayId);
    if (it == arrays_.end()) return;

    auto& info = it->second;
    if (vk::formatIsBlockCompressed(info.format)) return;
    if (info.mipLevels <= 1) return;
    if (layer >= info.layerCount) return;

    generateMipChainForLayer(info, cmdBuffer, layer);
    info.image->imageLayout() = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
}

void TextureArrayManager::generateMipmapsForLayers(uint32_t arrayId,
                                                   const std::vector<uint32_t>& layers,
                                                   std::shared_ptr<vk::CommandBuffer> cmdBuffer) {
    if (layers.empty()) return;

    auto it = arrays_.find(arrayId);
    if (it == arrays_.end()) return;

    auto& info = it->second;
    if (vk::formatIsBlockCompressed(info.format)) return;
    if (info.mipLevels <= 1) return;

    std::vector<uint32_t> uniqueLayers = layers;
    std::sort(uniqueLayers.begin(), uniqueLayers.end());
    uniqueLayers.erase(std::unique(uniqueLayers.begin(), uniqueLayers.end()), uniqueLayers.end());

    for (uint32_t layer : uniqueLayers) {
        if (layer < info.layerCount) {
            generateMipChainForLayer(info, cmdBuffer, layer);
        }
    }
    info.image->imageLayout() = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
}

const TextureArrayManager::ArrayInfo* TextureArrayManager::getArray(uint32_t arrayId) const {
	auto it = arrays_.find(arrayId);
    if (it == arrays_.end()) return nullptr;
    return &it->second;
}

bool TextureArrayManager::hasPendingUploads() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return !stagedUploads_.empty();
}

std::vector<uint32_t> TextureArrayManager::collectDirtyLayers(uint32_t arrayId) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<uint32_t> layers;
    for (const auto& upload : stagedUploads_) {
        if (upload.arrayId == arrayId) layers.push_back(upload.layer);
    }
    return layers;
}

std::vector<std::shared_ptr<vk::HostVisibleBuffer>> TextureArrayManager::takeStagingBuffers() {
    return std::move(currentFrameStagingBuffers_);
}

void TextureArrayManager::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    arrays_.clear();
    stagedUploads_.clear();
    currentFrameStagingBuffers_.clear();
    nextArrayId_ = 0;
}
