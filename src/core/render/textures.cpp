#include "core/render/textures.hpp"

#include "core/render/modules/ui_module.hpp"
#include "core/render/pipeline.hpp"
#include "core/render/render_framework.hpp"
#include "core/render/renderer.hpp"

std::ostream &texturesCout() {
    return std::cout << "[Textures] ";
}

std::ostream &texturesCerr() {
    return std::cerr << "[Textures] ";
}

Textures::Textures(std::shared_ptr<Framework> framework) {}

void Textures::reset() {
    textures_.clear();
    nextID = 0;
}

void Textures::resetFrame() {
    auto framework = Renderer::instance().framework();

    framework->gc().collect(uploadQueue_);
    uploadQueue_ = std::make_shared<std::map<uint32_t, std::vector<VkBufferImageCopy>>>();

    for (auto &entry : caches_) {
        auto &cache = entry.second;
        cache->reset();
    }
}

uint32_t Textures::allocateTexture() {
    std::unique_lock<std::recursive_mutex> lck(mutex_);

    textures_.emplace(std::make_pair(nextID, nullptr));
    samplers.emplace(std::make_pair(nextID, nullptr));
    return nextID++;
}

void Textures::initializeTexture(uint32_t id, uint32_t maxLevel, uint32_t width, uint32_t height, VkFormat format) {
    auto device = Renderer::instance().framework()->device();
    auto vma = Renderer::instance().framework()->vma();

    std::unique_lock<std::recursive_mutex> lck(mutex_);

    auto textureIter = textures_.find(id);
    if (textureIter == textures_.end()) {
        texturesCerr() << "The given texture id: " << id << " is not allocated for texture" << std::endl;
        exit(EXIT_FAILURE);
    }

    auto framework = Renderer::instance().framework();
    framework->gc().collect(textures_[id]);
    textures_[id] = vk::DeviceLocalImage::create(device, vma, false, maxLevel, width, height, 1, format,
                                                 VK_IMAGE_USAGE_SAMPLED_BIT, 0, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);

    auto samplerIter = samplers.find(id);
    if (samplerIter == samplers.end()) {
        texturesCerr() << "The given texture id: " << id << " is not allocated for sampler" << std::endl;
        exit(EXIT_FAILURE);
    }
    framework->gc().collect(samplers[id]);
    samplers[id] =
        vk::Sampler::create(device, VK_FILTER_NEAREST, VK_SAMPLER_MIPMAP_MODE_NEAREST, VK_SAMPLER_ADDRESS_MODE_REPEAT);

    Renderer::instance().framework()->pipeline()->bindTexture(samplers[id], textures_[id], id);
}

void Textures::setSamplingMode(uint32_t id, VkFilter samplingMode, VkSamplerMipmapMode mipmapMode) {
    auto device = Renderer::instance().framework()->device();

    std::unique_lock<std::recursive_mutex> lck(mutex_);

    auto samplerIter = samplers.find(id);
    if (samplerIter == samplers.end()) {
        texturesCerr() << "The given texture id: " << id << " is not allocated for sampler" << std::endl;
        exit(EXIT_FAILURE);
    }
    if (samplers[id]->vkSamplingMode() != samplingMode) {
        VkSamplerAddressMode addressMode = samplers[id]->vkAddressMode();

        auto framework = Renderer::instance().framework();
        framework->gc().collect(samplers[id]);
        samplers[id] = vk::Sampler::create(device, samplingMode, mipmapMode, addressMode);
    }

    Renderer::instance().framework()->pipeline()->bindTexture(samplers[id], textures_[id], id);
}

void Textures::setAddressMode(uint32_t id, VkSamplerAddressMode addressMode) {
    auto device = Renderer::instance().framework()->device();

    std::unique_lock<std::recursive_mutex> lck(mutex_);

    auto samplerIter = samplers.find(id);
    if (samplerIter == samplers.end()) {
        texturesCerr() << "The given texture id: " << id << " is not allocated for sampler" << std::endl;
        exit(EXIT_FAILURE);
    }
    if (samplers[id]->vkAddressMode() != addressMode) {
        VkFilter samplingMode = samplers[id]->vkSamplingMode();
        VkSamplerMipmapMode mipmapMode = samplers[id]->vkMipmapMode();

        auto framework = Renderer::instance().framework();
        framework->gc().collect(samplers[id]);
        samplers[id] = vk::Sampler::create(device, samplingMode, mipmapMode, addressMode);
    }

    Renderer::instance().framework()->pipeline()->bindTexture(samplers[id], textures_[id], id);
}

void Textures::queueUpload(uint8_t *srcPointer,
                           uint32_t srcSizeInBytes,
                           uint32_t srcRowPixels,
                           uint32_t dstId,
                           int srcOffsetX,
                           int srcOffsetY,
                           int dstOffsetX,
                           int dstOffsetY,
                           uint32_t width,
                           uint32_t height,
                           uint32_t level) {
    std::unique_lock<std::recursive_mutex> lck(mutex_);

    auto framework = Renderer::instance().framework();

    auto device = Renderer::instance().framework()->device();
    auto vma = Renderer::instance().framework()->vma();
    auto dstTextureIter = textures_.find(dstId);
    if (dstTextureIter == textures_.end()) {
        texturesCerr() << "The dstID " << dstId << " is not registered yet!" << std::endl;
        exit(EXIT_FAILURE);
    }
    auto dstTexture = (*dstTextureIter).second;

    auto cacheIter = caches_.find(dstId);
    if (cacheIter == caches_.end()) {
        cacheIter = caches_
                        .emplace(std::make_pair(
                            dstId, ImageBufferCache::create(vma, device, framework->swapchain()->imageCount())))
                        .first;
    }

    auto cache = cacheIter->second;
    size_t offset = cache->append(srcPointer, srcSizeInBytes);

    auto format = dstTexture->vkFormat();
    uint32_t bytePerPixel = vk::formatToByte(format);

    VkBufferImageCopy region = {};
    region.bufferRowLength = srcRowPixels;
    region.bufferOffset = offset + srcOffsetY * srcRowPixels * bytePerPixel + srcOffsetX * bytePerPixel;
    region.imageSubresource = vk::wholeColorSubresourceLayers;
    region.imageSubresource.mipLevel = level;
    region.imageExtent = {width, height, 1};
    region.imageOffset = {dstOffsetX, dstOffsetY, 0};

    auto dstTextureUploadQueueIter = uploadQueue_->find(dstId);
    if (dstTextureUploadQueueIter == uploadQueue_->end()) {
        dstTextureUploadQueueIter = uploadQueue_->emplace(dstId, std::vector<VkBufferImageCopy>{}).first;
    }
    dstTextureUploadQueueIter->second.emplace_back(region);
}

void Textures::performQueuedUpload() {
    std::unique_lock<std::recursive_mutex> lck(mutex_);

    std::shared_ptr<vk::CommandBuffer> cmdBuffer =
        Renderer::instance().framework()->safeAcquireCurrentContext()->uploadCommandBuffer;

    auto physicalDevice = Renderer::instance().framework()->physicalDevice();
    auto mainQueueIndex = physicalDevice->mainQueueIndex();

    std::vector<vk::CommandBuffer::ImageMemoryBarrier> uploadPreImageBarriers, uploadPostImageBarriers;

    for (auto &entry : *uploadQueue_) {
        auto &textureId = entry.first;
        auto textureIter = textures_.find(textureId);
        if (textureIter == textures_.end()) {
            texturesCerr() << "The textureId " << textureId << " is not registered yet!" << std::endl;
            exit(EXIT_FAILURE);
        }
        auto texture = textureIter->second;
        uploadPreImageBarriers.push_back({
            .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
            .oldLayout = texture->imageLayout(),
            .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .srcQueueFamilyIndex = mainQueueIndex,
            .dstQueueFamilyIndex = mainQueueIndex,
            .image = texture,
            .subresourceRange = vk::wholeColorSubresourceRange,
        });
        texture->imageLayout() = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;

        uploadPostImageBarriers.push_back({
            .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                            VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            .srcQueueFamilyIndex = mainQueueIndex,
            .dstQueueFamilyIndex = mainQueueIndex,
            .image = texture,
            .subresourceRange = vk::wholeColorSubresourceRange,
        });
        texture->imageLayout() = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }

    cmdBuffer->barriersBufferImage({}, uploadPreImageBarriers);

    for (auto &entry : *uploadQueue_) {
        auto &textureId = entry.first;
        auto &regions = entry.second;

        auto textureIter = textures_.find(textureId);
        if (textureIter == textures_.end()) {
            texturesCerr() << "The textureId " << textureId << " is not registered yet!" << std::endl;
            exit(EXIT_FAILURE);
        }
        auto texture = textureIter->second;

        auto cacheIter = caches_.find(textureId);
        if (cacheIter == caches_.end()) { continue; }
        auto cache = cacheIter->second;

        vkCmdCopyBufferToImage(cmdBuffer->vkCommandBuffer(), cache->vkBuffer(), texture->vkImage(),
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, regions.size(), regions.data());

        cmdBuffer->barriersBufferImage(
            {}, {{
                    .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                    .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                    .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                                    VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                    .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                    .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    .srcQueueFamilyIndex = mainQueueIndex,
                    .dstQueueFamilyIndex = mainQueueIndex,
                    .image = texture,
                    .subresourceRange = vk::wholeColorSubresourceRange,
                }});
    }

    cmdBuffer->barriersBufferImage({}, uploadPostImageBarriers);
}

void Textures::bindAllTextures() {
    auto device = Renderer::instance().framework()->device();

    std::unique_lock<std::recursive_mutex> lck(mutex_);

    for (const auto &[id, texture] : textures_) {
        if (texture == nullptr) {
            continue; // only allocated, but not initialized yet
        }

        Renderer::instance().framework()->pipeline()->bindTexture(samplers[id], texture, id);
    }
}

ImageBufferCache::ImageBufferCache(std::shared_ptr<vk::VMA> vma, std::shared_ptr<vk::Device> device, uint32_t frameNum)
    : vma_(vma), device_(device) {
    capacities_.resize(frameNum);
    bases_.resize(frameNum);
    caches_.resize(frameNum);

    for (int i = 0; i < frameNum; i++) {
        caches_[i] = vk::HostVisibleBuffer::create(vma_, device_, BASE_SIZE, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
        capacities_[i] = BASE_SIZE;
        bases_[i] = 0;
    }
}

ImageBufferCache::~ImageBufferCache() {
#ifdef DEBUG
// std::cout << "ImageBufferCache deconstructed" << std::endl;
#endif
}

size_t ImageBufferCache::append(void *src, size_t size) {
    if (bases_[current_] + size >= capacities_[current_]) {
        size_t newCapacity = capacities_[current_] * 2;

        while (newCapacity < bases_[current_] + size) { newCapacity *= 2; }

        auto newCache = vk::HostVisibleBuffer::create(vma_, device_, newCapacity, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);

        std::memcpy(newCache->mappedPtr(), caches_[current_]->mappedPtr(), bases_[current_]);

        auto framework = Renderer::instance().framework();
        framework->gc().collect(caches_[current_]);
        caches_[current_] = newCache;
        capacities_[current_] = newCapacity;
    }

    size_t ret = bases_[current_];
    std::memcpy(static_cast<uint8_t *>(caches_[current_]->mappedPtr()) + bases_[current_], src, size);
    bases_[current_] = (bases_[current_] + size + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
    return ret;
}

void ImageBufferCache::flush() {
    caches_[current_]->flush();
}

VkBuffer &ImageBufferCache::vkBuffer() {
    return caches_[current_]->vkBuffer();
}

void ImageBufferCache::reset() {
    current_ = (current_ + 1) % caches_.size();
    bases_[current_] = 0;
}