#pragma once

#include "common/singleton.hpp"
#include "core/all_extern.hpp"
#include "core/vulkan/all_core_vulkan.hpp"

#include <functional>
#include <map>
#include <mutex>

class Framework;

class ImageBufferCache;

class Textures : public SharedObject<Textures> {
  public:
    // 0 = FULLY_OPAQUE, 1 = FULLY_TRANSPARENT, 2 = MIXED
    enum class AlphaClass : int32_t { FULLY_OPAQUE = 0, FULLY_TRANSPARENT = 1, MIXED = 2 };

    struct TextureAlphaData {
        std::vector<uint8_t> alpha; // single-channel, 1 byte per texel
        uint32_t width = 0;
        uint32_t height = 0;
        bool animated = false; // true if re-uploaded (animation frame change)
    };

    Textures(std::shared_ptr<Framework> framework);

    void reset();
    void resetFrame();
    bool hasPendingUploads() const;
    uint32_t allocateTexture();
    void initializeTexture(uint32_t id, uint32_t maxLevel, uint32_t width, uint32_t height, VkFormat format);
    void setSamplingMode(uint32_t id, VkFilter samplingMode, VkSamplerMipmapMode mipmapMode);
    void setAddressMode(uint32_t id, VkSamplerAddressMode addressMode);
    void queueUpload(uint8_t *srcPointer,
                     uint32_t srcSizeInBytes,
                     uint32_t srcRowPixels,
                     uint32_t dstId,
                     int srcOffsetX,
                     int srcOffsetY,
                     int dstOffsetX,
                     int dstOffsetY,
                     uint32_t width,
                     uint32_t height,
                     uint32_t level);
    void performQueuedUpload();
    void bindAllTextures();
    void destroyTexture(uint32_t id);

    void setTextureAlphaClass(uint32_t id, AlphaClass alphaClass);
    AlphaClass getTextureAlphaClass(uint32_t id) const;

    const TextureAlphaData *getTextureAlphaData(uint32_t id) const;

  private:
    std::map<uint32_t, std::shared_ptr<vk::DeviceLocalImage>> textures_;
    std::map<uint32_t, std::shared_ptr<vk::Sampler>> samplers;
    uint32_t nextID = 0;
    std::vector<uint32_t> freeList_;
    std::recursive_mutex mutex_;

    std::map<uint32_t, std::shared_ptr<ImageBufferCache>> caches_;
    std::shared_ptr<std::map<uint32_t, std::vector<VkBufferImageCopy>>> uploadQueue_;

    std::map<uint32_t, AlphaClass> textureAlphaClass_;
    std::map<uint32_t, TextureAlphaData> textureAlphaData_;
};

class ImageBufferCache : public SharedObject<ImageBufferCache> {
  public:
    constexpr static size_t BASE_SIZE = 16 * 1024; // 1KB
    constexpr static size_t ALIGNMENT = 4;

    ImageBufferCache(std::shared_ptr<vk::VMA> vma, std::shared_ptr<vk::Device> device, uint32_t frameNum);
    ~ImageBufferCache();

    size_t append(void *src, size_t size);
    void flush();
    void reset();

    VkBuffer &vkBuffer();

  private:
    std::shared_ptr<vk::VMA> vma_;
    std::shared_ptr<vk::Device> device_;

    uint32_t current_ = 0;
    std::vector<size_t> capacities_;
    std::vector<size_t> bases_;
    std::vector<std::shared_ptr<vk::HostVisibleBuffer>> caches_;
};