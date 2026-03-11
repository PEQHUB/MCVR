#include "core/render/modules/world/ray_tracing/ray_tracing_module.hpp"

#include "core/render/buffers.hpp"
#include "core/render/modules/world/ray_tracing/submodules/atmosphere.hpp"
#include "core/render/modules/world/ray_tracing/submodules/world_prepare.hpp"
#include "core/render/pipeline.hpp"
#include "core/render/render_framework.hpp"
#include "core/render/renderer.hpp"

RayTracingModule::RayTracingModule() {}

void RayTracingModule::init(std::shared_ptr<Framework> framework, std::shared_ptr<WorldPipeline> worldPipeline) {
    WorldModule::init(framework, worldPipeline);

    uint32_t size = framework->swapchain()->imageCount();

    hdrNoisyOutputImages_.resize(size);
    diffuseAlbedoImages_.resize(size);
    specularAlbedoImages_.resize(size);
    normalRoughnessImages_.resize(size);
    motionVectorImages_.resize(size);
    linearDepthImages_.resize(size);
    specularHitDepthImages_.resize(size);
    firstHitDepthImages_.resize(size);
    firstHitDiffuseDirectLightImages_.resize(size);
    firstHitDiffuseIndirectLightImages_.resize(size);
    firstHitSpecularImages_.resize(size);
    firstHitClearImages_.resize(size);
    firstHitBaseEmissionImages_.resize(size);
    directLightDepthImages_.resize(size);
    diffuseRayDirHitDistImages_.resize(size);
    specularRayDirHitDistImages_.resize(size);

    atmosphere_ = Atmosphere::create(framework, shared_from_this());
    worldPrepare_ = WorldPrepare::create(framework, shared_from_this());
}

bool RayTracingModule::setOrCreateInputImages(std::vector<std::shared_ptr<vk::DeviceLocalImage>> &images,
                                              std::vector<VkFormat> &formats,
                                              uint32_t frameIndex) {
    return true;
}

bool RayTracingModule::setOrCreateOutputImages(std::vector<std::shared_ptr<vk::DeviceLocalImage>> &images,
                                               std::vector<VkFormat> &formats,
                                               uint32_t frameIndex) {
    uint32_t width, height;
    bool set = false;
    for (auto &image : images) {
        if (image != nullptr) {
            if (!set) {
                width = image->width();
                height = image->height();
                set = true;
            } else {
                if (image->width() != width || image->height() != height) { return false; }
            }
        }
    }

    if (!set) { return false; }

    auto framework = framework_.lock();
    for (int i = 0; i < images.size(); i++) {
        if (images[i] == nullptr) {
            images[i] = vk::DeviceLocalImage::create(
                framework->device(), framework->vma(), false, width, height, 1, formats[i],
                VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
        }
    }

    hdrNoisyOutputImages_[frameIndex] = images[0];
    diffuseAlbedoImages_[frameIndex] = images[1];
    specularAlbedoImages_[frameIndex] = images[2];
    normalRoughnessImages_[frameIndex] = images[3];
    motionVectorImages_[frameIndex] = images[4];
    linearDepthImages_[frameIndex] = images[5];
    specularHitDepthImages_[frameIndex] = images[6];
    firstHitDepthImages_[frameIndex] = images[7];
    firstHitDiffuseDirectLightImages_[frameIndex] = images[8];
    firstHitDiffuseIndirectLightImages_[frameIndex] = images[9];
    firstHitSpecularImages_[frameIndex] = images[10];
    firstHitClearImages_[frameIndex] = images[11];
    firstHitBaseEmissionImages_[frameIndex] = images[12];
    directLightDepthImages_[frameIndex] = images[13];
    diffuseRayDirHitDistImages_[frameIndex] = images[14];
    specularRayDirHitDistImages_[frameIndex] = images[15];

    // Create reservoir images for ReSTIR DI (only once, shared across frames)
    if (!reservoirImages_[0]) {
        for (int r = 0; r < 2; r++) {
            reservoirImages_[r] = vk::DeviceLocalImage::create(
                framework->device(), framework->vma(), false, width, height, 1, VK_FORMAT_R32G32B32A32_SFLOAT,
                VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
        }
    }

    // Create bounce ReSTIR DI reservoir images (per-bounce temporal reuse, bounces 1-3)
    if (!bounceReservoirImages_[0]) {
        for (int b = 0; b < 3; b++) {
            bounceReservoirImages_[b] = vk::DeviceLocalImage::create(
                framework->device(), framework->vma(), false, width, height, 1, VK_FORMAT_R32G32B32A32_SFLOAT,
                VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
        }
    }

    return true;
}

void RayTracingModule::setAttributes(int attributeCount, std::vector<std::string> &attributeKVs) {
    auto parseBool = [](const std::string &value) {
        return value == "1" || value == "true" || value == "True" || value == "TRUE";
    };

    for (int i = 0; i < attributeCount; i++) {
        const std::string &key = attributeKVs[2 * i];
        const std::string &value = attributeKVs[2 * i + 1];

        if (key == "render_pipeline.module.dlss.attribute.num_ray_bounces") {
            numRayBounces_ = std::stoi(value);
        } else if (key == "render_pipeline.module.ray_tracing.attribute.use_jitter") {
            useJitter_ = parseBool(value);
            Renderer::instance().buffers()->setUseJitter(useJitter_);
        }
    }
}

void RayTracingModule::build() {
    atmosphere_->build();
    worldPrepare_->build();

    auto framework = framework_.lock();
    auto worldPipeline = worldPipeline_.lock();
    uint32_t size = framework->swapchain()->imageCount();

    contexts_.resize(size);

    initDescriptorTables();
    initImages();
    initPipeline();
    initSBT();
    initSpatialPipeline();
    initClusterPipeline();

    for (int i = 0; i < size; i++) {
        contexts_[i] =
            RayTracingModuleContext::create(framework->contexts()[i], worldPipeline->contexts()[i], shared_from_this());

        // set rayTracingModuleContext of sub-modules, order is important
        atmosphere_->contexts_[i]->rayTracingModuleContext =
            std::static_pointer_cast<RayTracingModuleContext>(contexts_[i]);
        worldPrepare_->contexts_[i]->rayTracingModuleContext =
            std::static_pointer_cast<RayTracingModuleContext>(contexts_[i]);
    }
}

std::vector<std::shared_ptr<WorldModuleContext>> &RayTracingModule::contexts() {
    return contexts_;
}

void RayTracingModule::bindTexture(std::shared_ptr<vk::Sampler> sampler,
                                   std::shared_ptr<vk::DeviceLocalImage> image,
                                   int index) {
    auto framework = framework_.lock();

    uint32_t size = framework->swapchain()->imageCount();
    for (int i = 0; i < size; i++) {
        if (rayTracingDescriptorTables_[i] != nullptr)
            rayTracingDescriptorTables_[i]->bindSamplerImage(sampler, image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                                             0, 0, index);
    }
}

void RayTracingModule::preClose() {
    auto framework = framework_.lock();
    if (framework) {
        VkDevice dev = framework->device()->vkDevice();
        if (spatialPipeline_ != VK_NULL_HANDLE) vkDestroyPipeline(dev, spatialPipeline_, nullptr);
        if (spatialPipelineLayout_ != VK_NULL_HANDLE) vkDestroyPipelineLayout(dev, spatialPipelineLayout_, nullptr);
        if (spatialDescSetLayout_ != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(dev, spatialDescSetLayout_, nullptr);
        if (spatialDescPool_ != VK_NULL_HANDLE) vkDestroyDescriptorPool(dev, spatialDescPool_, nullptr);
        if (clusterPipeline_ != VK_NULL_HANDLE) vkDestroyPipeline(dev, clusterPipeline_, nullptr);
        if (clusterPipelineLayout_ != VK_NULL_HANDLE) vkDestroyPipelineLayout(dev, clusterPipelineLayout_, nullptr);
        if (clusterDescSetLayout_ != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(dev, clusterDescSetLayout_, nullptr);
        if (clusterDescPool_ != VK_NULL_HANDLE) vkDestroyDescriptorPool(dev, clusterDescPool_, nullptr);
    }
}

void RayTracingModule::initDescriptorTables() {
    auto framework = framework_.lock();

    uint32_t size = framework->swapchain()->imageCount();
    rayTracingDescriptorTables_.resize(size);

    for (int i = 0; i < size; i++) {
        rayTracingDescriptorTables_[i] =
            vk::DescriptorTableBuilder{}
                .beginDescriptorLayoutSet() // set 0
                .beginDescriptorLayoutSetBinding()
                .defineDescriptorLayoutSetBinding({
                    .binding = 0,
                    .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    .descriptorCount = 4096, // a very big number
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                                  VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 1, // world atmosphere LUT
                    .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                                  VK_SHADER_STAGE_FRAGMENT_BIT,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 2, // world atmosphere cube map
                    .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                                  VK_SHADER_STAGE_FRAGMENT_BIT,
                })
                .endDescriptorLayoutSetBinding()
                .endDescriptorLayoutSet()
                .beginDescriptorLayoutSet() // set 1
                .beginDescriptorLayoutSetBinding()
                .defineDescriptorLayoutSetBinding({
                    .binding = 0, // binding 0: TLAS(s)
                    .descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 1, // binding 1: blasOffsets
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 2, // binding 2: vertex buffer addrs
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 3, // binding 3: index buffer addrs
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 4, // binding 4: last vertex buffer addrs
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 5, // binding 5: last index buffer addrs
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 6, // binding 6: last obj to world mat
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 7, // binding 7: texture mapping
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                                  VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 8, // binding 8: area light SSBO
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 9, // binding 9: tile light buffer (light clustering)
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,
                })
                .endDescriptorLayoutSetBinding()
                .endDescriptorLayoutSet()
                .beginDescriptorLayoutSet() // set 2
                .beginDescriptorLayoutSetBinding()
                .defineDescriptorLayoutSetBinding({
                    .binding = 0, // binding 0: current world ubo
                    .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                                  VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 1, // binding 1: last world ubo
                    .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 2, // binding 2: sky ubo
                    .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                                  VK_SHADER_STAGE_ANY_HIT_BIT_KHR | VK_SHADER_STAGE_VERTEX_BIT |
                                  VK_SHADER_STAGE_FRAGMENT_BIT,
                })
                .endDescriptorLayoutSetBinding()
                .endDescriptorLayoutSet()
                .beginDescriptorLayoutSet() // set 3
                .beginDescriptorLayoutSetBinding()
                .defineDescriptorLayoutSetBinding({
                    .binding = 0, // binding 0: hdrNoisyImage
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 1, // binding 1: diffuseAlbedoImage
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 2, // binding 2: specularAlbedoImage
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 3, // binding 3: normalRoughnessImage
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 4, // binding 4: motionVectorImage
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 5, // binding 5: linearDepthImage
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 6, // binding 6: specularHitDepth
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 7, // binding 7: firstHitDepthImage
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                                  VK_SHADER_STAGE_FRAGMENT_BIT,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 8, // binding 8: firstHitDiffuseDirectLightImage
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                                  VK_SHADER_STAGE_FRAGMENT_BIT,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 9, // binding 9: firstHitDiffuseIndirectLightImage
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                                  VK_SHADER_STAGE_FRAGMENT_BIT,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 10, // binding 10: firstHitSpecularImage;
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                                  VK_SHADER_STAGE_FRAGMENT_BIT,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 11, // binding 11: firstHitClearImage;
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                                  VK_SHADER_STAGE_FRAGMENT_BIT,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 12, // binding 12: firstHitBaseEmissionImage;
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                                  VK_SHADER_STAGE_FRAGMENT_BIT,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 13, // binding 13: directLightDepthImage;
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                                  VK_SHADER_STAGE_FRAGMENT_BIT,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 14, // binding 14: reservoirCurrentImage (ReSTIR DI write)
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 15, // binding 15: reservoirPreviousImage (ReSTIR DI read)
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 16, // binding 16: diffuseRayDirHitDistImage (DLSS-RR guide)
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 17, // binding 17: specularRayDirHitDistImage (DLSS-RR guide)
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 18, // binding 18: bounceReservoirImage1 (bounce 1 ReSTIR DI)
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 19, // binding 19: bounceReservoirImage2 (bounce 2 ReSTIR DI)
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 20, // binding 20: bounceReservoirImage3 (bounce 3 ReSTIR DI)
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,
                })
                .endDescriptorLayoutSetBinding()
                .endDescriptorLayoutSet()
                .definePushConstant({
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                                  VK_SHADER_STAGE_INTERSECTION_BIT_KHR,
                    .offset = 0,
                    .size = sizeof(RayTracingPushConstant),
                })
                .build(framework->device());
    }
}

void RayTracingModule::initImages() {
    auto framework = framework_.lock();

    uint32_t size = framework->swapchain()->imageCount();

    for (int i = 0; i < size; i++) {
        rayTracingDescriptorTables_[i]->bindSamplerImageForShader(atmosphere_->atmLUTImageSampler_,
                                                                  atmosphere_->atmLUTImage_, 0, 1);
        rayTracingDescriptorTables_[i]->bindSamplerImageForShader(atmosphere_->atmCubeMapImageSamplers_[i],
                                                                  atmosphere_->atmCubeMapImages_[i], 0, 2, 7);

        rayTracingDescriptorTables_[i]->bindImage(hdrNoisyOutputImages_[i], VK_IMAGE_LAYOUT_GENERAL, 3, 0);
        rayTracingDescriptorTables_[i]->bindImage(diffuseAlbedoImages_[i], VK_IMAGE_LAYOUT_GENERAL, 3, 1);
        rayTracingDescriptorTables_[i]->bindImage(specularAlbedoImages_[i], VK_IMAGE_LAYOUT_GENERAL, 3, 2);
        rayTracingDescriptorTables_[i]->bindImage(normalRoughnessImages_[i], VK_IMAGE_LAYOUT_GENERAL, 3, 3);
        rayTracingDescriptorTables_[i]->bindImage(motionVectorImages_[i], VK_IMAGE_LAYOUT_GENERAL, 3, 4);
        rayTracingDescriptorTables_[i]->bindImage(linearDepthImages_[i], VK_IMAGE_LAYOUT_GENERAL, 3, 5);
        rayTracingDescriptorTables_[i]->bindImage(specularHitDepthImages_[i], VK_IMAGE_LAYOUT_GENERAL, 3, 6);
        rayTracingDescriptorTables_[i]->bindImage(firstHitDepthImages_[i], VK_IMAGE_LAYOUT_GENERAL, 3, 7);
        rayTracingDescriptorTables_[i]->bindImage(firstHitDiffuseDirectLightImages_[i], VK_IMAGE_LAYOUT_GENERAL, 3, 8);
        rayTracingDescriptorTables_[i]->bindImage(firstHitDiffuseIndirectLightImages_[i], VK_IMAGE_LAYOUT_GENERAL, 3,
                                                  9);
        rayTracingDescriptorTables_[i]->bindImage(firstHitSpecularImages_[i], VK_IMAGE_LAYOUT_GENERAL, 3, 10);
        rayTracingDescriptorTables_[i]->bindImage(firstHitClearImages_[i], VK_IMAGE_LAYOUT_GENERAL, 3, 11);
        rayTracingDescriptorTables_[i]->bindImage(firstHitBaseEmissionImages_[i], VK_IMAGE_LAYOUT_GENERAL, 3, 12);
        rayTracingDescriptorTables_[i]->bindImage(directLightDepthImages_[i], VK_IMAGE_LAYOUT_GENERAL, 3, 13);
        rayTracingDescriptorTables_[i]->bindImage(diffuseRayDirHitDistImages_[i], VK_IMAGE_LAYOUT_GENERAL, 3, 16);
        rayTracingDescriptorTables_[i]->bindImage(specularRayDirHitDistImages_[i], VK_IMAGE_LAYOUT_GENERAL, 3, 17);

        // ReSTIR DI reservoir images (initial binding, rebound each frame in render)
        if (reservoirImages_[0]) {
            rayTracingDescriptorTables_[i]->bindImage(reservoirImages_[0], VK_IMAGE_LAYOUT_GENERAL, 3, 14);
            rayTracingDescriptorTables_[i]->bindImage(reservoirImages_[1], VK_IMAGE_LAYOUT_GENERAL, 3, 15);
        }
    }
}

void RayTracingModule::initPipeline() {
    auto framework = framework_.lock();
    auto device = framework->device();

    std::filesystem::path shaderPath = Renderer::folderPath / "shaders";
    worldRayGenShader_ = vk::Shader::create(device, (shaderPath / "world/ray_tracing/world_rgen.spv").string());
    worldRayMissShader_ = vk::Shader::create(device, (shaderPath / "world/ray_tracing/world_rmiss.spv").string());
    handRayMissShader_ = vk::Shader::create(device, (shaderPath / "world/ray_tracing/hand_rmiss.spv").string());
    shadowRayMissShader_ = vk::Shader::create(device, (shaderPath / "world/ray_tracing/shadow_rmiss.spv").string());
    pointLightShadowMissShader_ =
        vk::Shader::create(device, (shaderPath / "world/ray_tracing/point_light_shadow_rmiss.spv").string());
    shadowRayClosestHitShader_ =
        vk::Shader::create(device, (shaderPath / "world/ray_tracing/shadow_rchit.spv").string());
    worldSolidTransparentClosestHitShader_ =
        vk::Shader::create(device, (shaderPath / "world/ray_tracing/world_solid_transparent_rchit.spv").string());
    worldNoReflectClosestHitShader_ =
        vk::Shader::create(device, (shaderPath / "world/ray_tracing/world_no_reflect_rchit.spv").string());
    worldCloudClosestHitShader_ =
        vk::Shader::create(device, (shaderPath / "world/ray_tracing/world_cloud_rchit.spv").string());
    shadowAnyHitShader_ = vk::Shader::create(device, (shaderPath / "world/ray_tracing/shadow_rahit.spv").string());
    worldTransparentAnyHitShader_ =
        vk::Shader::create(device, (shaderPath / "world/ray_tracing/world_transparent_rahit.spv").string());
    worldNoReflectAnyHitShader_ =
        vk::Shader::create(device, (shaderPath / "world/ray_tracing/world_no_reflect_rahit.spv").string());
    worldCloudAnyHitShader_ =
        vk::Shader::create(device, (shaderPath / "world/ray_tracing/world_cloud_rahit.spv").string());

    boatWaterMaskClosestHitShader_ =
        vk::Shader::create(device, (shaderPath / "world/ray_tracing/boat_water_mask_rchit.spv").string());
    boatWaterMaskAnyHitShader_ =
        vk::Shader::create(device, (shaderPath / "world/ray_tracing/boat_water_mask_rahit.spv").string());

    endPortalClosestHitShader_ =
        vk::Shader::create(device, (shaderPath / "world/ray_tracing/end_portal_rchit.spv").string());
    endPortalAnyHitShader_ =
        vk::Shader::create(device, (shaderPath / "world/ray_tracing/end_portal_rahit.spv").string());

    endGatewayClosestHitShader_ =
        vk::Shader::create(device, (shaderPath / "world/ray_tracing/end_gateway_rchit.spv").string());
    endGatewayAnyHitShader_ =
        vk::Shader::create(device, (shaderPath / "world/ray_tracing/end_gateway_rahit.spv").string());

    rayTracingPipeline_ =
        vk::RayTracingPipelineBuilder{}
            .beginShaderStage()
            .defineShaderStage(worldRayGenShader_, VK_SHADER_STAGE_RAYGEN_BIT_KHR)                          // 0
            .defineShaderStage(worldRayMissShader_, VK_SHADER_STAGE_MISS_BIT_KHR)                           // 1
            .defineShaderStage(handRayMissShader_, VK_SHADER_STAGE_MISS_BIT_KHR)                            // 2
            .defineShaderStage(shadowRayMissShader_, VK_SHADER_STAGE_MISS_BIT_KHR)                          // 3
            .defineShaderStage(worldSolidTransparentClosestHitShader_, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR) // 4
            .defineShaderStage(worldNoReflectClosestHitShader_, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR)        // 5
            .defineShaderStage(worldCloudClosestHitShader_, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR)            // 6
            .defineShaderStage(worldTransparentAnyHitShader_, VK_SHADER_STAGE_ANY_HIT_BIT_KHR)              // 7
            .defineShaderStage(worldNoReflectAnyHitShader_, VK_SHADER_STAGE_ANY_HIT_BIT_KHR)                // 8
            .defineShaderStage(worldCloudAnyHitShader_, VK_SHADER_STAGE_ANY_HIT_BIT_KHR)                    // 9
            .defineShaderStage(shadowRayClosestHitShader_, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR)             // 10
            .defineShaderStage(shadowAnyHitShader_, VK_SHADER_STAGE_ANY_HIT_BIT_KHR)                        // 11
            .defineShaderStage(boatWaterMaskClosestHitShader_, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR)         // 12
            .defineShaderStage(boatWaterMaskAnyHitShader_, VK_SHADER_STAGE_ANY_HIT_BIT_KHR)                 // 13
            .defineShaderStage(endPortalClosestHitShader_, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR)             // 14
            .defineShaderStage(endPortalAnyHitShader_, VK_SHADER_STAGE_ANY_HIT_BIT_KHR)                     // 15
            .defineShaderStage(endGatewayClosestHitShader_, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR)            // 16
            .defineShaderStage(endGatewayAnyHitShader_, VK_SHADER_STAGE_ANY_HIT_BIT_KHR)                    // 17
            .defineShaderStage(pointLightShadowMissShader_, VK_SHADER_STAGE_MISS_BIT_KHR)                // 18
            .endShaderStage()
            .beginShaderGroup()
            .defineShaderGroup(VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR, 0, VK_SHADER_UNUSED_KHR,
                               VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR)
            .defineShaderGroup(VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR, 1, VK_SHADER_UNUSED_KHR,
                               VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR)
            .defineShaderGroup(VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR, 2, VK_SHADER_UNUSED_KHR,
                               VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR)
            .defineShaderGroup(VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR, 3, VK_SHADER_UNUSED_KHR,
                               VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR)
            .defineShaderGroup(VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR, 18, VK_SHADER_UNUSED_KHR,
                               VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR) // point light shadow miss
            .defineShaderGroup(VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR, VK_SHADER_UNUSED_KHR, 10, 11,
                               VK_SHADER_UNUSED_KHR) // shadow
            .defineShaderGroup(VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR, VK_SHADER_UNUSED_KHR, 4,
                               VK_SHADER_UNUSED_KHR,
                               VK_SHADER_UNUSED_KHR) // world solid
            .defineShaderGroup(VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR, VK_SHADER_UNUSED_KHR, 4, 7,
                               VK_SHADER_UNUSED_KHR) // world transparent
            .defineShaderGroup(VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR, VK_SHADER_UNUSED_KHR, 5, 8,
                               VK_SHADER_UNUSED_KHR) // world no reflect
            .defineShaderGroup(VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR, VK_SHADER_UNUSED_KHR, 6, 9,
                               VK_SHADER_UNUSED_KHR) // world cloud
            .defineShaderGroup(VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR, VK_SHADER_UNUSED_KHR, 12, 13,
                               VK_SHADER_UNUSED_KHR) // boat water mask
            .defineShaderGroup(VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR, VK_SHADER_UNUSED_KHR, 14, 15,
                               VK_SHADER_UNUSED_KHR) // end portal
            .defineShaderGroup(VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR, VK_SHADER_UNUSED_KHR, 16, 17,
                               VK_SHADER_UNUSED_KHR) // end gateway
            .endShaderGroup()
            .definePipelineLayout(rayTracingDescriptorTables_[0])
            .build(device);
}

void RayTracingModule::initSBT() {
    auto framework = framework_.lock();

    sbts_.resize(framework->swapchain()->imageCount());
    for (int i = 0; i < framework->swapchain()->imageCount(); i++) {
        sbts_[i] = vk::SBT::create(framework->physicalDevice(), framework->device(), framework->vma(),
                                   rayTracingPipeline_, 4, 8);
    }
}

void RayTracingModule::initSpatialPipeline() {
    auto framework = framework_.lock();
    auto device = framework->device();
    VkDevice dev = device->vkDevice();
    uint32_t size = framework->swapchain()->imageCount();

    // Load compute shader
    std::filesystem::path shaderPath = Renderer::folderPath / "shaders";
    spatialShader_ = vk::Shader::create(device, (shaderPath / "world/ray_tracing/restir_spatial_comp.spv").string());
    if (!spatialShader_) {
        std::cerr << "[ReSTIR] Failed to load restir_spatial_comp.spv" << std::endl;
        return;
    }

    // Descriptor set layout: 4 storage images
    std::vector<VkDescriptorSetLayoutBinding> bindings = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };
    VkDescriptorSetLayoutCreateInfo layoutInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layoutInfo.bindingCount = (uint32_t)bindings.size();
    layoutInfo.pBindings = bindings.data();
    vkCreateDescriptorSetLayout(dev, &layoutInfo, nullptr, &spatialDescSetLayout_);

    // Pipeline layout with push constant (6 int32s = 24 bytes)
    VkPushConstantRange pushRange = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 24};
    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &spatialDescSetLayout_;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushRange;
    vkCreatePipelineLayout(dev, &pipelineLayoutInfo, nullptr, &spatialPipelineLayout_);

    // Compute pipeline
    VkComputePipelineCreateInfo pipelineInfo = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipelineInfo.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineInfo.stage.module = spatialShader_->vkShaderModule();
    pipelineInfo.stage.pName = "main";
    pipelineInfo.layout = spatialPipelineLayout_;
    vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &spatialPipeline_);

    // Descriptor pool
    VkDescriptorPoolSize poolSizes[] = {
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 4 * size},
    };
    VkDescriptorPoolCreateInfo poolInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfo.maxSets = size;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = poolSizes;
    vkCreateDescriptorPool(dev, &poolInfo, nullptr, &spatialDescPool_);

    // Allocate descriptor sets
    std::vector<VkDescriptorSetLayout> layouts(size, spatialDescSetLayout_);
    VkDescriptorSetAllocateInfo allocInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocInfo.descriptorPool = spatialDescPool_;
    allocInfo.descriptorSetCount = size;
    allocInfo.pSetLayouts = layouts.data();
    spatialDescSets_.resize(size);
    vkAllocateDescriptorSets(dev, &allocInfo, spatialDescSets_.data());
}

void RayTracingModule::initClusterPipeline() {
    auto framework = framework_.lock();
    auto device = framework->device();
    VkDevice dev = device->vkDevice();
    uint32_t size = framework->swapchain()->imageCount();

    // Load compute shader
    std::filesystem::path shaderPath = Renderer::folderPath / "shaders";
    clusterShader_ = vk::Shader::create(device, (shaderPath / "world/ray_tracing/light_clustering_comp.spv").string());
    if (!clusterShader_) {
        std::cerr << "[Clustering] Failed to load light_clustering_comp.spv" << std::endl;
        return;
    }

    // Descriptor set layout: 2 storage buffers (area lights + tile buffer)
    std::vector<VkDescriptorSetLayoutBinding> bindings = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };
    VkDescriptorSetLayoutCreateInfo layoutInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layoutInfo.bindingCount = (uint32_t)bindings.size();
    layoutInfo.pBindings = bindings.data();
    vkCreateDescriptorSetLayout(dev, &layoutInfo, nullptr, &clusterDescSetLayout_);

    // Pipeline layout with push constant (width, height, lightCount, maxPerTile, mat4 vpCameraRel)
    VkPushConstantRange pushRange = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 80}; // 16 bytes ints + 64 bytes mat4
    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &clusterDescSetLayout_;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushRange;
    vkCreatePipelineLayout(dev, &pipelineLayoutInfo, nullptr, &clusterPipelineLayout_);

    // Compute pipeline
    VkComputePipelineCreateInfo pipelineInfo = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipelineInfo.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineInfo.stage.module = clusterShader_->vkShaderModule();
    pipelineInfo.stage.pName = "main";
    pipelineInfo.layout = clusterPipelineLayout_;
    vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &clusterPipeline_);

    // Descriptor pool
    VkDescriptorPoolSize poolSize = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 * size};
    VkDescriptorPoolCreateInfo poolInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfo.maxSets = size;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    vkCreateDescriptorPool(dev, &poolInfo, nullptr, &clusterDescPool_);

    // Allocate descriptor sets
    std::vector<VkDescriptorSetLayout> clusterLayouts(size, clusterDescSetLayout_);
    VkDescriptorSetAllocateInfo clusterAllocInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    clusterAllocInfo.descriptorPool = clusterDescPool_;
    clusterAllocInfo.descriptorSetCount = size;
    clusterAllocInfo.pSetLayouts = clusterLayouts.data();
    clusterDescSets_.resize(size);
    vkAllocateDescriptorSets(dev, &clusterAllocInfo, clusterDescSets_.data());

    // Create tile light buffer (will be resized per-frame if needed)
    // Initial size based on common resolution
    int tilesX = (1920 + TILE_SIZE - 1) / TILE_SIZE;
    int tilesY = (1080 + TILE_SIZE - 1) / TILE_SIZE;
    int totalTiles = tilesX * tilesY;
    size_t bufferSize = totalTiles * (1 + MAX_LIGHTS_PER_TILE) * sizeof(uint32_t);
    tileLightBuffer_ = vk::DeviceLocalBuffer::create(
        framework->vma(), device, bufferSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
}

RayTracingModuleContext::RayTracingModuleContext(std::shared_ptr<FrameworkContext> frameworkContext,
                                                 std::shared_ptr<WorldPipelineContext> worldPipelineContext,
                                                 std::shared_ptr<RayTracingModule> rayTracingModule)
    : WorldModuleContext(frameworkContext, worldPipelineContext),
      rayTracingModule(rayTracingModule),
      rayTracingDescriptorTable(rayTracingModule->rayTracingDescriptorTables_[frameworkContext->frameIndex]),
      sbt(rayTracingModule->sbts_[frameworkContext->frameIndex]),
      hdrNoisyOutputImage(rayTracingModule->hdrNoisyOutputImages_[frameworkContext->frameIndex]),
      diffuseAlbedoImage(rayTracingModule->diffuseAlbedoImages_[frameworkContext->frameIndex]),
      specularAlbedoImage(rayTracingModule->specularAlbedoImages_[frameworkContext->frameIndex]),
      normalRoughnessImage(rayTracingModule->normalRoughnessImages_[frameworkContext->frameIndex]),
      motionVectorImage(rayTracingModule->motionVectorImages_[frameworkContext->frameIndex]),
      linearDepthImage(rayTracingModule->linearDepthImages_[frameworkContext->frameIndex]),
      specularHitDepthImage(rayTracingModule->specularHitDepthImages_[frameworkContext->frameIndex]),
      firstHitDepthImage(rayTracingModule->firstHitDepthImages_[frameworkContext->frameIndex]),
      firstHitDiffuseDirectLightImage(
          rayTracingModule->firstHitDiffuseDirectLightImages_[frameworkContext->frameIndex]),
      firstHitDiffuseIndirectLightImage(
          rayTracingModule->firstHitDiffuseIndirectLightImages_[frameworkContext->frameIndex]),
      firstHitSpecularImage(rayTracingModule->firstHitSpecularImages_[frameworkContext->frameIndex]),
      firstHitClearImage(rayTracingModule->firstHitClearImages_[frameworkContext->frameIndex]),
      firstHitBaseEmissionImage(rayTracingModule->firstHitBaseEmissionImages_[frameworkContext->frameIndex]),
      directLightDepthImage(rayTracingModule->directLightDepthImages_[frameworkContext->frameIndex]),
      diffuseRayDirHitDistImage(rayTracingModule->diffuseRayDirHitDistImages_[frameworkContext->frameIndex]),
      specularRayDirHitDistImage(rayTracingModule->specularRayDirHitDistImages_[frameworkContext->frameIndex]),
      atmosphereContext(rayTracingModule->atmosphere_->contexts_[frameworkContext->frameIndex]),
      worldPrepareContext(rayTracingModule->worldPrepare_->contexts_[frameworkContext->frameIndex]) {}

void RayTracingModuleContext::render() {
    atmosphereContext->render();
    worldPrepareContext->render();

    if (worldPrepareContext->tlas == nullptr) {
        std::cout << "tlas is nullptr" << std::endl;
        return;
    }

    auto context = frameworkContext.lock();
    auto framework = context->framework.lock();
    auto worldCommandBuffer = context->worldCommandBuffer;
    auto mainQueueIndex = framework->physicalDevice()->mainQueueIndex();

    auto module = rayTracingModule.lock();

    rayTracingDescriptorTable->bindAS(worldPrepareContext->tlas, 1, 0);
    rayTracingDescriptorTable->bindBuffer(worldPrepareContext->blasOffsetsBuffer, 1, 1);
    rayTracingDescriptorTable->bindBuffer(worldPrepareContext->vertexBufferAddr, 1, 2);
    rayTracingDescriptorTable->bindBuffer(worldPrepareContext->indexBufferAddr, 1, 3);
    rayTracingDescriptorTable->bindBuffer(worldPrepareContext->lastVertexBufferAddr, 1, 4);
    rayTracingDescriptorTable->bindBuffer(worldPrepareContext->lastIndexBufferAddr, 1, 5);
    rayTracingDescriptorTable->bindBuffer(worldPrepareContext->lastObjToWorldMat, 1, 6);

    auto buffers = Renderer::instance().buffers();
    auto worldBuffer = buffers->worldUniformBuffer();

    rayTracingDescriptorTable->bindBuffer(buffers->textureMappingBuffer(), 1, 7);
    rayTracingDescriptorTable->bindBuffer(worldPrepareContext->areaLightBuffer, 1, 8);
    if (module->tileLightBuffer_) {
        rayTracingDescriptorTable->bindBuffer(module->tileLightBuffer_, 1, 9);
    }
    rayTracingDescriptorTable->bindBuffer(worldBuffer, 2, 0);
    rayTracingDescriptorTable->bindBuffer(buffers->lastWorldUniformBuffer(), 2, 1);
    rayTracingDescriptorTable->bindBuffer(buffers->skyUniformBuffer(), 2, 2);

    // ReSTIR DI: fixed-role reservoir binding
    // Binding 14 = reservoir[0] (CHS writes temporal output)
    // Binding 15 = reservoir[1] when spatial enabled (CHS reads spatial output)
    //            = reservoir[0] when spatial disabled (CHS self-reads temporal, safe per-pixel)
    if (module->reservoirImages_[0]) {
        bool spatialEnabled = Renderer::options.restirEnabled && Renderer::options.restirSpatialEnabled && module->spatialPipeline_ != VK_NULL_HANDLE;
        rayTracingDescriptorTable->bindImage(module->reservoirImages_[0], VK_IMAGE_LAYOUT_GENERAL, 3, 14);
        rayTracingDescriptorTable->bindImage(
            module->reservoirImages_[spatialEnabled ? 1 : 0], VK_IMAGE_LAYOUT_GENERAL, 3, 15);
    }

    // Bounce ReSTIR DI: per-bounce reservoir images (bindings 18-20)
    for (int b = 0; b < 3; b++) {
        if (module->bounceReservoirImages_[b]) {
            rayTracingDescriptorTable->bindImage(
                module->bounceReservoirImages_[b], VK_IMAGE_LAYOUT_GENERAL, 3, 18 + b);
        }
    }

    RayTracingPushConstant pushConstant{};
    pushConstant.numRayBounces = static_cast<int>(Renderer::options.rayBounces);
    pushConstant.flags = (Renderer::options.simplifiedIndirect ? 1 : 0)
                       | (Renderer::options.areaLightsEnabled ? 2 : 0)
                       | (Renderer::options.restirEnabled ? 4 : 0)
                       | (Renderer::options.restirSimplifiedBRDF ? 8 : 0)
                       | (Renderer::options.restirBounceEnabled ? 16 : 0);
    pushConstant.areaLightCount = worldPrepareContext->areaLightCount;
    pushConstant.shadowSoftness = Renderer::options.shadowSoftness;
    pushConstant.risCandidates = Renderer::options.restirCandidates;
    pushConstant.temporalMClamp = Renderer::options.restirTemporalMClamp;
    pushConstant.wClamp = Renderer::options.restirWClamp;
    // Only apply pre-exposure when DLSS-RR is active (it undoes it via InExposureScale).
    // When DLSS-RR is off, tone mapper would see double exposure (preExposure × autoExposure).
    pushConstant.preExposure = (Renderer::options.denoiserMode == 1) ? Renderer::preExposure : 1.0f;

    vkCmdPushConstants(worldCommandBuffer->vkCommandBuffer(), rayTracingDescriptorTable->vkPipelineLayout(),
                       VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                           VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                           VK_SHADER_STAGE_INTERSECTION_BIT_KHR,
                       0, sizeof(RayTracingPushConstant), &pushConstant);

    auto chooseSrc = [](VkImageLayout oldLayout, VkPipelineStageFlags2 fallbackStage, VkAccessFlags2 fallbackAccess,
                        VkPipelineStageFlags2 &outStage, VkAccessFlags2 &outAccess) {
        if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED) {
            outStage = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
            outAccess = 0;
        } else {
            outStage = fallbackStage;
            outAccess = fallbackAccess;
        }
    };

    std::vector<vk::CommandBuffer::ImageMemoryBarrier> barriers;
    auto addBarrier = [&](const std::shared_ptr<vk::DeviceLocalImage> &img, VkImageLayout newLayout) {
        if (!img) return;
        VkPipelineStageFlags2 srcStage = 0;
        VkAccessFlags2 srcAccess = 0;
        chooseSrc(img->imageLayout(), VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                  VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT, srcStage, srcAccess);
        barriers.push_back({
            .srcStageMask = srcStage,
            .srcAccessMask = srcAccess,
            .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
            .oldLayout = img->imageLayout(),
            .newLayout = newLayout,
            .srcQueueFamilyIndex = mainQueueIndex,
            .dstQueueFamilyIndex = mainQueueIndex,
            .image = img,
            .subresourceRange = vk::wholeColorSubresourceRange,
        });
        img->imageLayout() = newLayout;
    };

    addBarrier(hdrNoisyOutputImage, VK_IMAGE_LAYOUT_GENERAL);
    addBarrier(diffuseAlbedoImage, VK_IMAGE_LAYOUT_GENERAL);
    addBarrier(specularAlbedoImage, VK_IMAGE_LAYOUT_GENERAL);
    addBarrier(normalRoughnessImage, VK_IMAGE_LAYOUT_GENERAL);
    addBarrier(motionVectorImage, VK_IMAGE_LAYOUT_GENERAL);
    addBarrier(linearDepthImage, VK_IMAGE_LAYOUT_GENERAL);
    addBarrier(specularHitDepthImage, VK_IMAGE_LAYOUT_GENERAL);
    addBarrier(firstHitDepthImage, VK_IMAGE_LAYOUT_GENERAL);
    addBarrier(firstHitDiffuseDirectLightImage, VK_IMAGE_LAYOUT_GENERAL);
    addBarrier(firstHitDiffuseIndirectLightImage, VK_IMAGE_LAYOUT_GENERAL);
    addBarrier(firstHitSpecularImage, VK_IMAGE_LAYOUT_GENERAL);
    addBarrier(firstHitClearImage, VK_IMAGE_LAYOUT_GENERAL);
    addBarrier(firstHitBaseEmissionImage, VK_IMAGE_LAYOUT_GENERAL);
    addBarrier(directLightDepthImage, VK_IMAGE_LAYOUT_GENERAL);
    addBarrier(diffuseRayDirHitDistImage, VK_IMAGE_LAYOUT_GENERAL);
    addBarrier(specularRayDirHitDistImage, VK_IMAGE_LAYOUT_GENERAL);
    addBarrier(module->reservoirImages_[0], VK_IMAGE_LAYOUT_GENERAL);
    addBarrier(module->reservoirImages_[1], VK_IMAGE_LAYOUT_GENERAL);
    for (int b = 0; b < 3; b++) {
        addBarrier(module->bounceReservoirImages_[b], VK_IMAGE_LAYOUT_GENERAL);
    }
    addBarrier(atmosphereContext->atmCubeMapImage, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    if (!barriers.empty()) { worldCommandBuffer->barriersBufferImage({}, barriers); }

    // Light clustering compute pass — DISABLED: contribution-sorted global list replaces tile clustering.
    // Tile buffer stays allocated (descriptor layout unchanged); CHS reads tileCount=0 → global fallback.
    if (false && Renderer::options.restirEnabled && Renderer::options.areaLightsEnabled
        && module->clusterPipeline_ != VK_NULL_HANDLE && module->tileLightBuffer_
        && worldPrepareContext->areaLightCount > 0) {
        VkCommandBuffer cmd = worldCommandBuffer->vkCommandBuffer();
        uint32_t frameIdx = context->frameIndex;

        int w = hdrNoisyOutputImage->width();
        int h = hdrNoisyOutputImage->height();
        int tilesX = (w + RayTracingModule::TILE_SIZE - 1) / RayTracingModule::TILE_SIZE;
        int tilesY = (h + RayTracingModule::TILE_SIZE - 1) / RayTracingModule::TILE_SIZE;
        int totalTiles = tilesX * tilesY;

        // Resize tile buffer if needed
        size_t requiredSize = totalTiles * (1 + RayTracingModule::MAX_LIGHTS_PER_TILE) * sizeof(uint32_t);
        if (module->tileLightBuffer_->size() < requiredSize) {
            module->tileLightBuffer_ = vk::DeviceLocalBuffer::create(
                framework->vma(), framework->device(), requiredSize,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
            rayTracingDescriptorTable->bindBuffer(module->tileLightBuffer_, 1, 9);
        }

        // Update cluster descriptor set bindings
        VkDescriptorBufferInfo lightBufInfo = {worldPrepareContext->areaLightBuffer->vkBuffer(), 0, VK_WHOLE_SIZE};
        VkDescriptorBufferInfo tileBufInfo = {module->tileLightBuffer_->vkBuffer(), 0, VK_WHOLE_SIZE};
        VkWriteDescriptorSet writes[2] = {};
        writes[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        writes[0].dstSet = module->clusterDescSets_[frameIdx];
        writes[0].dstBinding = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[0].pBufferInfo = &lightBufInfo;
        writes[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        writes[1].dstSet = module->clusterDescSets_[frameIdx];
        writes[1].dstBinding = 1;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[1].pBufferInfo = &tileBufInfo;
        vkUpdateDescriptorSets(framework->device()->vkDevice(), 2, writes, 0, nullptr);

        // Compute view-proj matrix for camera-relative positions (rotation only, no translation)
        auto worldUBO = static_cast<vk::Data::WorldUBO *>(buffers->worldUniformBuffer()->mappedPtr());
        glm::mat4 viewRot = glm::mat4(glm::mat3(worldUBO->cameraViewMat));
        glm::mat4 vpCameraRel = worldUBO->cameraProjMat * viewRot;

        struct {
            int32_t width, height, lightCount, maxPerTile;
            glm::mat4 vpCameraRel;
        } clusterPC = {w, h, worldPrepareContext->areaLightCount, RayTracingModule::MAX_LIGHTS_PER_TILE, vpCameraRel};

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, module->clusterPipeline_);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, module->clusterPipelineLayout_,
            0, 1, &module->clusterDescSets_[frameIdx], 0, nullptr);
        vkCmdPushConstants(cmd, module->clusterPipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT,
            0, sizeof(clusterPC), &clusterPC);
        vkCmdDispatch(cmd, totalTiles, 1, 1);

        // Barrier: compute writes → RT reads
        VkMemoryBarrier clusterBarrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        clusterBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        clusterBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
            0, 1, &clusterBarrier, 0, nullptr, 0, nullptr);
    }

    worldCommandBuffer->bindDescriptorTable(rayTracingDescriptorTable, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR)
        ->bindRTPipeline(module->rayTracingPipeline_)
        ->raytracing(sbt, hdrNoisyOutputImage->width(), hdrNoisyOutputImage->height(), 1);

    // Spatial reuse compute pass (when ReSTIR and spatial reuse are both enabled)
    if (Renderer::options.restirEnabled && Renderer::options.restirSpatialEnabled && module->spatialPipeline_ != VK_NULL_HANDLE) {
        VkCommandBuffer cmd = worldCommandBuffer->vkCommandBuffer();

        // Barrier: RT shader writes → compute shader reads
        VkMemoryBarrier spatialBarrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        spatialBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        spatialBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 1, &spatialBarrier, 0, nullptr, 0, nullptr);

        // Update spatial descriptor set with current frame's images
        uint32_t frameIdx = context->frameIndex;
        VkDescriptorSet spatialSet = module->spatialDescSets_[frameIdx];

        auto addSpatialImg = [&](uint32_t binding, const std::shared_ptr<vk::DeviceLocalImage>& img,
                                 std::vector<VkWriteDescriptorSet>& writes,
                                 std::vector<std::unique_ptr<VkDescriptorImageInfo>>& infos) {
            auto info = std::make_unique<VkDescriptorImageInfo>();
            info->imageView = img->vkImageView(0);
            info->imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            info->sampler = VK_NULL_HANDLE;
            writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, spatialSet, binding, 0, 1,
                             VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, info.get(), nullptr, nullptr});
            infos.push_back(std::move(info));
        };

        std::vector<VkWriteDescriptorSet> spatialWrites;
        std::vector<std::unique_ptr<VkDescriptorImageInfo>> spatialInfos;
        addSpatialImg(0, module->reservoirImages_[0], spatialWrites, spatialInfos);
        addSpatialImg(1, module->reservoirImages_[1], spatialWrites, spatialInfos);
        addSpatialImg(2, normalRoughnessImage, spatialWrites, spatialInfos);
        addSpatialImg(3, linearDepthImage, spatialWrites, spatialInfos);

        vkUpdateDescriptorSets(framework->device()->vkDevice(),
            (uint32_t)spatialWrites.size(), spatialWrites.data(), 0, nullptr);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, module->spatialPipeline_);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, module->spatialPipelineLayout_,
            0, 1, &spatialSet, 0, nullptr);

        struct { int32_t width, height, spatialTaps, spatialRadius, temporalMClamp, wClamp; } spatialPC = {
            (int32_t)hdrNoisyOutputImage->width(),
            (int32_t)hdrNoisyOutputImage->height(),
            Renderer::options.restirSpatialTaps,
            Renderer::options.restirSpatialRadius,
            Renderer::options.restirTemporalMClamp,
            Renderer::options.restirWClamp
        };
        vkCmdPushConstants(cmd, module->spatialPipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT,
            0, sizeof(spatialPC), &spatialPC);

        vkCmdDispatch(cmd,
            (hdrNoisyOutputImage->width() + 15) / 16,
            (hdrNoisyOutputImage->height() + 15) / 16,
            1);

        // Barrier: compute writes → next stage reads
        VkMemoryBarrier postSpatialBarrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        postSpatialBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        postSpatialBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
            0, 1, &postSpatialBarrier, 0, nullptr, 0, nullptr);
    }
}
