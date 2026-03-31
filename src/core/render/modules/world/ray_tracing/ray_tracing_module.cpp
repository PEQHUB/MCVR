#include "core/render/modules/world/ray_tracing/ray_tracing_module.hpp"

#include "core/render/buffers.hpp"
#include "core/render/gpu_diagnostics.hpp"
#include "core/render/lights.hpp"
#include "core/render/modules/world/ray_tracing/submodules/atmosphere.hpp"
#include "core/render/modules/world/ray_tracing/submodules/world_prepare.hpp"
#include "core/render/pipeline.hpp"
#include "core/render/render_framework.hpp"
#include "core/render/crash_ring_buffer.hpp"
#include "core/render/radiance_logger.hpp"
#include "core/render/renderer.hpp"
#include "core/render/modules/world/svgf/blue_noise.hpp"

#include <cmath>
#include <cstring>

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
    reflectionMvImages_.resize(size);
    animatedTexMaskImages_.resize(size);
    particleMaskImages_.resize(size);
    biasMaskImages_.resize(size);
    rtHitDistImages_.resize(size);
    motionVectors3DImages_.resize(size);
    gbufferMetallicImages_.resize(size);
    gbufferShadingModelIdImages_.resize(size);
    gbufferMaterialIdImages_.resize(size);
    positionViewSpaceImages_.resize(size);

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
    if (!framework) return false;
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
    reflectionMvImages_[frameIndex] = images[16];
    animatedTexMaskImages_[frameIndex] = images[17];
    particleMaskImages_[frameIndex] = images[18];
    biasMaskImages_[frameIndex] = images[19];
    rtHitDistImages_[frameIndex] = images[20];
    motionVectors3DImages_[frameIndex] = images[21];
    gbufferMetallicImages_[frameIndex] = images[22];
    gbufferShadingModelIdImages_[frameIndex] = images[23];
    gbufferMaterialIdImages_[frameIndex] = images[24];
    positionViewSpaceImages_[frameIndex] = images[25];

    // Publish depth and motion vectors for frame generation resource tagging
    if (Renderer::frameGenDepthImages.size() <= frameIndex) {
        Renderer::frameGenDepthImages.resize(frameIndex + 1);
        Renderer::frameGenMotionVectorImages.resize(frameIndex + 1);
    }
    Renderer::frameGenDepthImages[frameIndex] = linearDepthImages_[frameIndex];
    Renderer::frameGenMotionVectorImages[frameIndex] = motionVectorImages_[frameIndex];

    // Create reservoir images for ReSTIR DI (only once, shared across frames)
    if (Renderer::options.restirEnabled) {
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
    if (!framework) return;
    auto worldPipeline = worldPipeline_.lock();
    if (!worldPipeline) return;
    uint32_t size = framework->swapchain()->imageCount();

    contexts_.resize(size);

    initDescriptorTables();
    initEnergyLUT();
    initImages();
    initPipeline();
    initSBT();
    initSpatialPipeline();
    initClusterPipeline();
    sharcCapacity_ = 1u << static_cast<uint32_t>(Renderer::options.sharcCapacityExponent);
    if (Renderer::options.sharcEnabled) {
        initSharcBuffers();
        initSharcUpdatePipeline();
        initSharcResolvePipeline();
    }
    initAccumulationPipeline();

    // Initialize blue noise buffers (Owen-scrambled Sobol + spatial scrambling tile)
    // Upload immediately via important-upload path (synchronous staging→device before first RT dispatch)
    blueNoise_ = std::make_shared<BlueNoise>(framework->device(), framework->vma());

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
    if (!framework) return;

    uint32_t size = framework->swapchain()->imageCount();
    for (int i = 0; i < size; i++) {
        if (rayTracingDescriptorTables_[i] != nullptr)
            rayTracingDescriptorTables_[i]->bindSamplerImage(sampler, image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                                             0, 0, index);
    }
}

void RayTracingModule::initAccumulationPipeline() {
    auto framework = framework_.lock();
    if (!framework) return;
    auto device = framework->device();
    VkDevice dev = device->vkDevice();
    uint32_t size = framework->swapchain()->imageCount();

    std::filesystem::path shaderPath = Renderer::folderPath / "shaders";
    accumShader_ = vk::Shader::create(device, (shaderPath / "world/ray_tracing/accumulate_comp.spv").string());
    if (!accumShader_) {
        std::cerr << "[Offline] Failed to load accumulate_comp.spv — accumulation disabled" << std::endl;
        return;
    }

    // Create RGBA32F accumulation buffer at render resolution
    uint32_t w = hdrNoisyOutputImages_[0]->width();
    uint32_t h = hdrNoisyOutputImages_[0]->height();
    Renderer::accumBufferImage = vk::DeviceLocalImage::create(
        device, framework->vma(), false, w, h, 1, VK_FORMAT_R32G32B32A32_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    // Transition from UNDEFINED to GENERAL (required before first storage access)
    Renderer::accumBufferImage->imageLayout() = VK_IMAGE_LAYOUT_GENERAL;

    // Descriptor set layout: 2 storage images (accumBuffer, noisyInput)
    std::vector<VkDescriptorSetLayoutBinding> bindings = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };
    VkDescriptorSetLayoutCreateInfo layoutInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layoutInfo.bindingCount = (uint32_t)bindings.size();
    layoutInfo.pBindings = bindings.data();
    vkCreateDescriptorSetLayout(dev, &layoutInfo, nullptr, &Renderer::accumDescSetLayout);

    // Pipeline layout with push constant (1 int = 4 bytes)
    VkPushConstantRange pushRange = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 4};
    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &Renderer::accumDescSetLayout;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushRange;
    vkCreatePipelineLayout(dev, &pipelineLayoutInfo, nullptr, &Renderer::accumPipelineLayout);

    // Compute pipeline
    VkComputePipelineCreateInfo pipelineInfo = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipelineInfo.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineInfo.stage.module = accumShader_->vkShaderModule();
    pipelineInfo.stage.pName = "main";
    pipelineInfo.layout = Renderer::accumPipelineLayout;
    vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &Renderer::accumPipeline);

    // Descriptor pool
    VkDescriptorPoolSize poolSizes[] = {{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2 * size}};
    VkDescriptorPoolCreateInfo poolInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfo.maxSets = size;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = poolSizes;
    vkCreateDescriptorPool(dev, &poolInfo, nullptr, &Renderer::accumDescPool);

    // Allocate descriptor sets
    std::vector<VkDescriptorSetLayout> layouts(size, Renderer::accumDescSetLayout);
    VkDescriptorSetAllocateInfo allocInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocInfo.descriptorPool = Renderer::accumDescPool;
    allocInfo.descriptorSetCount = size;
    allocInfo.pSetLayouts = layouts.data();
    Renderer::accumDescSets.resize(size);
    vkAllocateDescriptorSets(dev, &allocInfo, Renderer::accumDescSets.data());

    Renderer::accumPipelineReady = true;
    std::cout << "[Offline] Accumulation pipeline initialized (" << w << "x" << h << " RGBA32F)" << std::endl;

    // --- Emission compose pipeline (DLSS-RR: subtract emission before, add after) ---
    auto emissionShader = vk::Shader::create(device, (shaderPath / "world/ray_tracing/emission_compose_comp.spv").string());
    if (!emissionShader) {
        std::cerr << "[Offline] Failed to load emission_compose_comp.spv — emission preservation disabled" << std::endl;
        return;
    }

    // Same descriptor set layout: 2 storage images (target, emission)
    VkDescriptorSetLayoutCreateInfo ecLayoutInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    ecLayoutInfo.bindingCount = (uint32_t)bindings.size();
    ecLayoutInfo.pBindings = bindings.data();
    vkCreateDescriptorSetLayout(dev, &ecLayoutInfo, nullptr, &Renderer::emissionComposeDescSetLayout);

    // Pipeline layout with push constant: float preExposure + int mode = 8 bytes
    VkPushConstantRange ecPushRange = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 8};
    VkPipelineLayoutCreateInfo ecPipelineLayoutInfo = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    ecPipelineLayoutInfo.setLayoutCount = 1;
    ecPipelineLayoutInfo.pSetLayouts = &Renderer::emissionComposeDescSetLayout;
    ecPipelineLayoutInfo.pushConstantRangeCount = 1;
    ecPipelineLayoutInfo.pPushConstantRanges = &ecPushRange;
    vkCreatePipelineLayout(dev, &ecPipelineLayoutInfo, nullptr, &Renderer::emissionComposePipelineLayout);

    VkComputePipelineCreateInfo ecPipelineInfo = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    ecPipelineInfo.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    ecPipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    ecPipelineInfo.stage.module = emissionShader->vkShaderModule();
    ecPipelineInfo.stage.pName = "main";
    ecPipelineInfo.layout = Renderer::emissionComposePipelineLayout;
    vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &ecPipelineInfo, nullptr, &Renderer::emissionComposePipeline);

    VkDescriptorPoolSize ecPoolSizes[] = {{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2 * size}};
    VkDescriptorPoolCreateInfo ecPoolInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    ecPoolInfo.maxSets = size;
    ecPoolInfo.poolSizeCount = 1;
    ecPoolInfo.pPoolSizes = ecPoolSizes;
    vkCreateDescriptorPool(dev, &ecPoolInfo, nullptr, &Renderer::emissionComposeDescPool);

    std::vector<VkDescriptorSetLayout> ecLayouts(size, Renderer::emissionComposeDescSetLayout);
    VkDescriptorSetAllocateInfo ecAllocInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ecAllocInfo.descriptorPool = Renderer::emissionComposeDescPool;
    ecAllocInfo.descriptorSetCount = size;
    ecAllocInfo.pSetLayouts = ecLayouts.data();
    Renderer::emissionComposeDescSets.resize(size);
    vkAllocateDescriptorSets(dev, &ecAllocInfo, Renderer::emissionComposeDescSets.data());

    Renderer::emissionComposePipelineReady = true;
    std::cout << "[Offline] Emission compose pipeline initialized" << std::endl;
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
        if (sharcResolvePipeline_ != VK_NULL_HANDLE) vkDestroyPipeline(dev, sharcResolvePipeline_, nullptr);
        if (sharcResolvePipelineLayout_ != VK_NULL_HANDLE) vkDestroyPipelineLayout(dev, sharcResolvePipelineLayout_, nullptr);
        if (Renderer::accumPipeline != VK_NULL_HANDLE) vkDestroyPipeline(dev, Renderer::accumPipeline, nullptr);
        if (Renderer::accumPipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(dev, Renderer::accumPipelineLayout, nullptr);
        if (Renderer::accumDescSetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(dev, Renderer::accumDescSetLayout, nullptr);
        if (Renderer::accumDescPool != VK_NULL_HANDLE) vkDestroyDescriptorPool(dev, Renderer::accumDescPool, nullptr);
        Renderer::accumPipelineReady = false;
        if (Renderer::emissionComposePipeline != VK_NULL_HANDLE) vkDestroyPipeline(dev, Renderer::emissionComposePipeline, nullptr);
        if (Renderer::emissionComposePipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(dev, Renderer::emissionComposePipelineLayout, nullptr);
        if (Renderer::emissionComposeDescSetLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(dev, Renderer::emissionComposeDescSetLayout, nullptr);
        if (Renderer::emissionComposeDescPool != VK_NULL_HANDLE) vkDestroyDescriptorPool(dev, Renderer::emissionComposeDescPool, nullptr);
        Renderer::emissionComposePipelineReady = false;
    }
}

void RayTracingModule::initDescriptorTables() {
    auto framework = framework_.lock();
    if (!framework) return;

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
                    .descriptorCount = 4096, // bindless texture array
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
                .defineDescriptorLayoutSetBinding({
                    .binding = 3, // block sprite albedo sampler2DArray
                    .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                                  VK_SHADER_STAGE_INTERSECTION_BIT_KHR,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 4, // block sprite specular sampler2DArray
                    .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                                  VK_SHADER_STAGE_INTERSECTION_BIT_KHR,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 5, // block sprite normal sampler2DArray
                    .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                                  VK_SHADER_STAGE_INTERSECTION_BIT_KHR,
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
                .defineDescriptorLayoutSetBinding({
                    .binding = 10, // binding 10: Blender PBR texture mapping (cold path)
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 11, // binding 11: Material class mapping (unified material system)
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 12, // binding 12: Displaced face data SSBO (intersection shader DDA)
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_INTERSECTION_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 13, // binding 13: SpriteRegistry SSBO (texture array metadata)
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                                  VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                                  VK_SHADER_STAGE_INTERSECTION_BIT_KHR,
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
                .defineDescriptorLayoutSetBinding({
                    .binding = 4, // binding 4: energy compensation LUT (multi-scatter GGX + EON diffuse)
                    .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 9, // Blue noise Sobol (moved to set 2 to avoid set 1 binding count issue)
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 10, // Blue noise scrambling tile
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,
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
                .defineDescriptorLayoutSetBinding({
                    .binding = 21, // binding 21: reflectionMvImage (DLSS-RR specular reflection MVs)
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 22, // binding 22: animTexMaskImage (DLSS-RR animated texture mask)
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 23, // binding 23: particleMaskImage (DLSS-RR transparent/refractive surface mask)
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 24, // binding 24: biasMaskImage (DLSS-RR bias current color mask)
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 25, // binding 25: rtHitDistImage (DLSS-RR per-pixel noise level hint)
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 26, // binding 26: motionVectors3DImage (DLSS-RR world-space 3D velocity)
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 27, // binding 27: gbufferMetallicImage (GBuffer metallic factor)
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 28, // binding 28: gbufferShadingModelIdImage (GBuffer shading model ID)
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 29, // binding 29: gbufferMaterialIdImage (GBuffer material ID)
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR,
                })
                .defineDescriptorLayoutSetBinding({
                    .binding = 30, // binding 30: positionViewSpaceImage (view-space hit position)
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR,
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

// =============================================================================
// Energy Compensation LUT bake (Kulla-Conty GGX + EON FON)
// 64x64 RGBA16F: R=GGX_E, G=FON_E, B=GGX_Eavg, A=FON_Eavg
// Runs once at init. ~2ms on modern CPUs.
// =============================================================================

namespace {

// CPU versions of GGX utility functions (mirroring shader code)
struct Vec3 { float x, y, z; };

Vec3 normalize3(Vec3 v) {
    float len = sqrtf(v.x*v.x + v.y*v.y + v.z*v.z);
    if (len < 1e-8f) return {0, 0, 1};
    return {v.x/len, v.y/len, v.z/len};
}

Vec3 sampleGGXVNDF_CPU(Vec3 V, float ax, float ay, float r1, float r2) {
    Vec3 Vh = normalize3({ax * V.x, ay * V.y, V.z});
    float lensq = Vh.x*Vh.x + Vh.y*Vh.y;
    Vec3 T1 = lensq > 0 ? Vec3{-Vh.y / sqrtf(lensq), Vh.x / sqrtf(lensq), 0}
                         : Vec3{1, 0, 0};
    Vec3 T2 = {Vh.y*T1.z - Vh.z*T1.y, Vh.z*T1.x - Vh.x*T1.z, Vh.x*T1.y - Vh.y*T1.x};
    float r = sqrtf(r1);
    float phi = 6.28318530718f * r2;
    float t1 = r * cosf(phi);
    float t2 = r * sinf(phi);
    float s = 0.5f * (1.0f + Vh.z);
    t2 = (1.0f - s) * sqrtf(fmaxf(0.0f, 1.0f - t1*t1)) + s * t2;
    float nz = sqrtf(fmaxf(0.0f, 1.0f - t1*t1 - t2*t2));
    Vec3 Nh = {t1*T1.x + t2*T2.x + nz*Vh.x,
               t1*T1.y + t2*T2.y + nz*Vh.y,
               t1*T1.z + t2*T2.z + nz*Vh.z};
    return normalize3({ax * Nh.x, ay * Nh.y, fmaxf(0.0f, Nh.z)});
}

float smithG1_CPU(float NdotV, float alpha) {
    float a2 = alpha * alpha;
    float c = NdotV;
    return (2.0f * c) / (c + sqrtf(a2 + c*c - a2*c*c));
}

// Simple xorshift32 RNG
uint32_t xorshift32(uint32_t &state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

float randFloat(uint32_t &state) {
    return (float)(xorshift32(state) & 0xFFFFFF) / (float)0xFFFFFF;
}

// Integrate GGX directional albedo E(mu, alpha) for F0=1 via VNDF importance sampling
float bakeGGXDirectionalAlbedo(float NdotV, float alpha, int numSamples) {
    NdotV = fmaxf(NdotV, 1e-4f);
    alpha = fmaxf(alpha, 1e-4f);
    Vec3 V = {sqrtf(1.0f - NdotV*NdotV), 0.0f, NdotV};
    float sum = 0.0f;
    uint32_t seed = (uint32_t)(NdotV * 12345.0f + alpha * 67890.0f + 1);
    for (int i = 0; i < numSamples; i++) {
        float r1 = randFloat(seed);
        float r2 = randFloat(seed);
        Vec3 H = sampleGGXVNDF_CPU(V, alpha, alpha, r1, r2);
        // Reflect V around H
        float VdotH = V.x*H.x + V.y*H.y + V.z*H.z;
        Vec3 L = {2.0f*VdotH*H.x - V.x, 2.0f*VdotH*H.y - V.y, 2.0f*VdotH*H.z - V.z};
        if (L.z > 0.0f) {
            // With VNDF sampling + separable Smith, weight = G1(L)
            sum += smithG1_CPU(L.z, alpha);
        }
    }
    return fminf(fmaxf(sum / (float)numSamples, 0.0f), 1.0f);
}

// Integrate FON directional albedo E_F(mu, r) via cosine-weighted hemisphere sampling
float bakeFONDirectionalAlbedo(float mu_o, float r, int numSamples) {
    mu_o = fmaxf(mu_o, 1e-4f);
    float sinTheta_o = sqrtf(1.0f - mu_o*mu_o);
    // FON coefficients
    float A_F = 1.0f / (1.0f + (0.5f - 2.0f / (3.0f * 3.14159265f)) * r);
    float B_F = r * A_F;
    float sum = 0.0f;
    uint32_t seed = (uint32_t)(mu_o * 54321.0f + r * 98765.0f + 7);
    for (int i = 0; i < numSamples; i++) {
        float r1 = randFloat(seed);
        float r2 = randFloat(seed);
        // Cosine-weighted hemisphere sample
        float sqrtR1 = sqrtf(r1);
        float phi = 6.28318530718f * r2;
        float Lx = sqrtR1 * cosf(phi);
        float Ly = sqrtR1 * sinf(phi);
        float Lz = sqrtf(fmaxf(0.0f, 1.0f - r1));
        float mu_i = Lz;
        // s = dot(L, V) - mu_i * mu_o  (V is in the xz plane)
        float VdotL = Lx * sinTheta_o + Lz * mu_o;
        float s = VdotL - mu_i * mu_o;
        float t = (s > 0.0f) ? fmaxf(mu_i, mu_o) : 1.0f;
        float f_FON = (1.0f / 3.14159265f) * (A_F + B_F * s / t);
        // Cosine-weighted sampling: weight = f_FON * pi (PDF = cos/pi, integrand = f*cos)
        sum += f_FON * 3.14159265f;
    }
    return fminf(fmaxf(sum / (float)numSamples, 0.0f), 1.0f);
}

} // anonymous namespace

void RayTracingModule::initEnergyLUT() {
    auto framework = framework_.lock();
    if (!framework) return;

    constexpr int LUT_SIZE = 64;
    constexpr int GGX_SAMPLES = 1024;
    constexpr int FON_SAMPLES = 512;

    // Bake LUT on CPU
    // Layout: RGBA16F, R=GGX_E(NdotV,alpha), G=FON_E(NdotV,r), B=GGX_Eavg, A=FON_Eavg
    // UV: u=NdotV [0,1], v=perceptual roughness r [0,1]; alpha = r*r
    std::vector<uint16_t> lutData(LUT_SIZE * LUT_SIZE * 4); // RGBA16F = 4 x fp16 per texel

    // First pass: compute E values per texel
    std::vector<float> ggxE(LUT_SIZE * LUT_SIZE);
    std::vector<float> fonE(LUT_SIZE * LUT_SIZE);

    for (int y = 0; y < LUT_SIZE; y++) {
        float r = ((float)y + 0.5f) / (float)LUT_SIZE; // perceptual roughness
        float alpha = r * r; // GGX alpha
        for (int x = 0; x < LUT_SIZE; x++) {
            float NdotV = ((float)x + 0.5f) / (float)LUT_SIZE;
            int idx = y * LUT_SIZE + x;
            ggxE[idx] = bakeGGXDirectionalAlbedo(NdotV, alpha, GGX_SAMPLES);
            fonE[idx] = bakeFONDirectionalAlbedo(NdotV, r, FON_SAMPLES);
        }
    }

    // Second pass: compute E_avg per roughness row (cosine-weighted average over NdotV)
    std::vector<float> ggxEavg(LUT_SIZE);
    std::vector<float> fonEavg(LUT_SIZE);

    for (int y = 0; y < LUT_SIZE; y++) {
        float sumGGX = 0.0f, sumFON = 0.0f, sumWeight = 0.0f;
        for (int x = 0; x < LUT_SIZE; x++) {
            float NdotV = ((float)x + 0.5f) / (float)LUT_SIZE;
            float weight = NdotV; // cosine weight (mu * dmu, uniform spacing)
            int idx = y * LUT_SIZE + x;
            sumGGX += ggxE[idx] * weight;
            sumFON += fonE[idx] * weight;
            sumWeight += weight;
        }
        ggxEavg[y] = sumWeight > 0 ? sumGGX / sumWeight : 0.5f;
        fonEavg[y] = sumWeight > 0 ? sumFON / sumWeight : 0.5f;
    }

    // Convert to fp16 and pack into RGBA16F
    auto toFP16 = [](float f) -> uint16_t {
        // IEEE 754 float32 to float16 conversion
        uint32_t bits;
        memcpy(&bits, &f, 4);
        uint32_t sign = (bits >> 16) & 0x8000;
        int32_t exponent = ((bits >> 23) & 0xFF) - 127 + 15;
        uint32_t mantissa = bits & 0x7FFFFF;
        if (exponent <= 0) return (uint16_t)sign; // underflow to 0
        if (exponent >= 31) return (uint16_t)(sign | 0x7C00); // overflow to inf
        return (uint16_t)(sign | (exponent << 10) | (mantissa >> 13));
    };

    for (int y = 0; y < LUT_SIZE; y++) {
        for (int x = 0; x < LUT_SIZE; x++) {
            int idx = y * LUT_SIZE + x;
            int pixelBase = idx * 4;
            lutData[pixelBase + 0] = toFP16(ggxE[idx]);      // R: GGX E(NdotV, alpha)
            lutData[pixelBase + 1] = toFP16(fonE[idx]);       // G: FON E(NdotV, r)
            lutData[pixelBase + 2] = toFP16(ggxEavg[y]);      // B: GGX E_avg(alpha)
            lutData[pixelBase + 3] = toFP16(fonEavg[y]);      // A: FON E_avg(r)
        }
    }

    // Create GPU image
    energyLUT_ = vk::DeviceLocalImage::create(
        framework->device(), framework->vma(), true, // persistStaging
        LUT_SIZE, LUT_SIZE, 1,
        VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

    energyLUT_->uploadToStagingBuffer(lutData.data());

    // One-shot transfer: staging buffer -> GPU image
    auto cmdBuf = vk::CommandBuffer::create(framework->device(), framework->mainCommandPool());
    cmdBuf->begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
    cmdBuf->copyToDeviceLocalImage(energyLUT_);
    cmdBuf->end();
    cmdBuf->submitMainQueueIndividual(framework->device());
    vkQueueWaitIdle(framework->device()->mainVkQueue());

    // Create sampler (bilinear, clamp to edge)
    energyLUTSampler_ = vk::Sampler::create(
        framework->device(), VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_NEAREST,
        VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);

    RadianceLogger::log("RayTracing", "INFO", "Energy compensation LUT baked (%dx%d, GGX=%d FON=%d samples)",
        LUT_SIZE, LUT_SIZE, GGX_SAMPLES, FON_SAMPLES);
}

void RayTracingModule::initImages() {
    auto framework = framework_.lock();
    if (!framework) return;

    uint32_t size = framework->swapchain()->imageCount();

    for (int i = 0; i < size; i++) {
        rayTracingDescriptorTables_[i]->bindSamplerImageForShader(atmosphere_->atmLUTImageSampler_,
                                                                  atmosphere_->atmLUTImage_, 0, 1);
        rayTracingDescriptorTables_[i]->bindSamplerImageForShader(atmosphere_->atmCubeMapImageSamplers_[i],
                                                                  atmosphere_->atmCubeMapImages_[i], 0, 2, 7);

        // Batch all RT output image bindings into a single vkUpdateDescriptorSets call
        using IB = vk::DescriptorTable::ImageBinding;
        std::vector<IB> imageBindings = {
            {hdrNoisyOutputImages_[i],                 VK_IMAGE_LAYOUT_GENERAL, 3,  0},
            {diffuseAlbedoImages_[i],                  VK_IMAGE_LAYOUT_GENERAL, 3,  1},
            {specularAlbedoImages_[i],                 VK_IMAGE_LAYOUT_GENERAL, 3,  2},
            {normalRoughnessImages_[i],                VK_IMAGE_LAYOUT_GENERAL, 3,  3},
            {motionVectorImages_[i],                   VK_IMAGE_LAYOUT_GENERAL, 3,  4},
            {linearDepthImages_[i],                    VK_IMAGE_LAYOUT_GENERAL, 3,  5},
            {specularHitDepthImages_[i],               VK_IMAGE_LAYOUT_GENERAL, 3,  6},
            {firstHitDepthImages_[i],                  VK_IMAGE_LAYOUT_GENERAL, 3,  7},
            {firstHitDiffuseDirectLightImages_[i],     VK_IMAGE_LAYOUT_GENERAL, 3,  8},
            {firstHitDiffuseIndirectLightImages_[i],   VK_IMAGE_LAYOUT_GENERAL, 3,  9},
            {firstHitSpecularImages_[i],               VK_IMAGE_LAYOUT_GENERAL, 3, 10},
            {firstHitClearImages_[i],                  VK_IMAGE_LAYOUT_GENERAL, 3, 11},
            {firstHitBaseEmissionImages_[i],           VK_IMAGE_LAYOUT_GENERAL, 3, 12},
            {directLightDepthImages_[i],               VK_IMAGE_LAYOUT_GENERAL, 3, 13},
            {diffuseRayDirHitDistImages_[i],           VK_IMAGE_LAYOUT_GENERAL, 3, 16},
            {specularRayDirHitDistImages_[i],          VK_IMAGE_LAYOUT_GENERAL, 3, 17},
            {reflectionMvImages_[i],                   VK_IMAGE_LAYOUT_GENERAL, 3, 21},
        };
        // Conditionally add optional output images
        if (animatedTexMaskImages_[i])
            imageBindings.push_back({animatedTexMaskImages_[i],       VK_IMAGE_LAYOUT_GENERAL, 3, 22});
        if (particleMaskImages_[i])
            imageBindings.push_back({particleMaskImages_[i],          VK_IMAGE_LAYOUT_GENERAL, 3, 23});
        if (biasMaskImages_[i])
            imageBindings.push_back({biasMaskImages_[i],              VK_IMAGE_LAYOUT_GENERAL, 3, 24});
        if (rtHitDistImages_[i])
            imageBindings.push_back({rtHitDistImages_[i],             VK_IMAGE_LAYOUT_GENERAL, 3, 25});
        if (motionVectors3DImages_[i])
            imageBindings.push_back({motionVectors3DImages_[i],       VK_IMAGE_LAYOUT_GENERAL, 3, 26});
        if (gbufferMetallicImages_[i])
            imageBindings.push_back({gbufferMetallicImages_[i],       VK_IMAGE_LAYOUT_GENERAL, 3, 27});
        if (gbufferShadingModelIdImages_[i])
            imageBindings.push_back({gbufferShadingModelIdImages_[i], VK_IMAGE_LAYOUT_GENERAL, 3, 28});
        if (gbufferMaterialIdImages_[i])
            imageBindings.push_back({gbufferMaterialIdImages_[i],     VK_IMAGE_LAYOUT_GENERAL, 3, 29});
        if (positionViewSpaceImages_[i])
            imageBindings.push_back({positionViewSpaceImages_[i],     VK_IMAGE_LAYOUT_GENERAL, 3, 30});

        // ReSTIR DI reservoir images (initial binding, rebound each frame in render)
        if (reservoirImages_[0]) {
            imageBindings.push_back({reservoirImages_[0], VK_IMAGE_LAYOUT_GENERAL, 3, 14});
            imageBindings.push_back({reservoirImages_[1], VK_IMAGE_LAYOUT_GENERAL, 3, 15});
        }

        rayTracingDescriptorTables_[i]->bindImages(imageBindings);

        // Energy compensation LUT (set 2, binding 4) — sampler image, stays individual
        if (energyLUT_ && energyLUTSampler_) {
            rayTracingDescriptorTables_[i]->bindSamplerImageForShader(energyLUTSampler_, energyLUT_, 2, 4);
        }

        // Pre-bind blue noise SSBOs to ALL descriptor tables at init time.
        // The render() path re-binds per-frame, but this ensures no descriptor
        // is left dangling at set=1 bindings 13/14 before the first render.
        if (blueNoise_) {
            using BB = vk::DescriptorTable::BufferBinding;
            rayTracingDescriptorTables_[i]->bindBufferBatch({
                BB{blueNoise_->sobolBuffer(),      2, 9},
                BB{blueNoise_->scramblingBuffer(), 2, 10},
            });
        }
    }

    // Publish emission images so tone mapping can add them back after DLSS-RR denoising
    Renderer::emissionImages = firstHitBaseEmissionImages_;
}

void RayTracingModule::initPipeline() {
    auto framework = framework_.lock();
    if (!framework) return;
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

    // DDA displacement shaders
    displacedIntersectionShader_ =
        vk::Shader::create(device, (shaderPath / "world/ray_tracing/displaced_block_rint.spv").string());
    displacedClosestHitShader_ =
        vk::Shader::create(device, (shaderPath / "world/ray_tracing/displaced_block_rchit.spv").string());
    displacedShadowClosestHitShader_ =
        vk::Shader::create(device, (shaderPath / "world/ray_tracing/displaced_shadow_rchit.spv").string());

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
            .defineShaderStage(pointLightShadowMissShader_, VK_SHADER_STAGE_MISS_BIT_KHR)                   // 18
            .defineShaderStage(displacedIntersectionShader_, VK_SHADER_STAGE_INTERSECTION_BIT_KHR)      // 19
            .defineShaderStage(displacedClosestHitShader_, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR)         // 20
            .defineShaderStage(displacedShadowClosestHitShader_, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR)   // 21
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
            // Hit groups 0-7: triangle groups (existing)
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
            // Hit group 8: DDA displacement (procedural — rint + rchit)
            .defineShaderGroup(VK_RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_KHR, VK_SHADER_UNUSED_KHR, 20,
                               VK_SHADER_UNUSED_KHR, 19) // displaced block (CHS=20, rint=19)
            .endShaderGroup()
            .definePipelineLayout(rayTracingDescriptorTables_[0])
            .build(device);
}

void RayTracingModule::initSBT() {
    auto framework = framework_.lock();
    if (!framework) return;

    // 4 miss groups, 9 hit groups (8 triangle + 1 procedural for DDA displacement)
    sbts_.resize(framework->swapchain()->imageCount());
    for (int i = 0; i < framework->swapchain()->imageCount(); i++) {
        sbts_[i] = vk::SBT::create(framework->physicalDevice(), framework->device(), framework->vma(),
                                   rayTracingPipeline_, 4, 9);
    }
}

void RayTracingModule::initSpatialPipeline() {
    auto framework = framework_.lock();
    if (!framework) return;
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
    vkCreateComputePipelines(dev, device->pipelineCache(), 1, &pipelineInfo, nullptr, &spatialPipeline_);

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
    if (!framework) return;
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
    vkCreateComputePipelines(dev, device->pipelineCache(), 1, &pipelineInfo, nullptr, &clusterPipeline_);

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

void RayTracingModule::initSharcBuffers() {
    auto framework = framework_.lock();
    if (!framework) return;

    VkBufferUsageFlags sharcUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
        | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
        | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    sharcHashEntries_ = vk::DeviceLocalBuffer::create(
        framework->vma(), framework->device(), sharcCapacity_ * 8, sharcUsage);
    sharcAccumulation_ = vk::DeviceLocalBuffer::create(
        framework->vma(), framework->device(), sharcCapacity_ * 16, sharcUsage);
    sharcResolved_ = vk::DeviceLocalBuffer::create(
        framework->vma(), framework->device(), sharcCapacity_ * 16, sharcUsage);

    sharcBuffersInitialized_ = false; // Will zero-fill on first use
}

void RayTracingModule::initSharcUpdatePipeline() {
    auto framework = framework_.lock();
    if (!framework) return;
    auto device = framework->device();

    std::filesystem::path shaderPath = Renderer::folderPath / "shaders";
    sharcUpdateRayGenShader_ = vk::Shader::create(device, (shaderPath / "world/ray_tracing/sharc_update_rgen.spv").string());
    if (!sharcUpdateRayGenShader_) {
        std::cerr << "[SHARC] Failed to load sharc_update_rgen.spv" << std::endl;
        return;
    }

    // Build update RT pipeline with same CHS/AHS/miss shaders as main pipeline
    sharcUpdatePipeline_ =
        vk::RayTracingPipelineBuilder{}
            .beginShaderStage()
            .defineShaderStage(sharcUpdateRayGenShader_, VK_SHADER_STAGE_RAYGEN_BIT_KHR)                     // 0 (update rgen)
            .defineShaderStage(worldRayMissShader_, VK_SHADER_STAGE_MISS_BIT_KHR)                            // 1
            .defineShaderStage(handRayMissShader_, VK_SHADER_STAGE_MISS_BIT_KHR)                             // 2
            .defineShaderStage(shadowRayMissShader_, VK_SHADER_STAGE_MISS_BIT_KHR)                           // 3
            .defineShaderStage(worldSolidTransparentClosestHitShader_, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR)  // 4
            .defineShaderStage(worldNoReflectClosestHitShader_, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR)         // 5
            .defineShaderStage(worldCloudClosestHitShader_, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR)             // 6
            .defineShaderStage(worldTransparentAnyHitShader_, VK_SHADER_STAGE_ANY_HIT_BIT_KHR)               // 7
            .defineShaderStage(worldNoReflectAnyHitShader_, VK_SHADER_STAGE_ANY_HIT_BIT_KHR)                 // 8
            .defineShaderStage(worldCloudAnyHitShader_, VK_SHADER_STAGE_ANY_HIT_BIT_KHR)                     // 9
            .defineShaderStage(shadowRayClosestHitShader_, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR)              // 10
            .defineShaderStage(shadowAnyHitShader_, VK_SHADER_STAGE_ANY_HIT_BIT_KHR)                         // 11
            .defineShaderStage(boatWaterMaskClosestHitShader_, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR)          // 12
            .defineShaderStage(boatWaterMaskAnyHitShader_, VK_SHADER_STAGE_ANY_HIT_BIT_KHR)                  // 13
            .defineShaderStage(endPortalClosestHitShader_, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR)              // 14
            .defineShaderStage(endPortalAnyHitShader_, VK_SHADER_STAGE_ANY_HIT_BIT_KHR)                      // 15
            .defineShaderStage(endGatewayClosestHitShader_, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR)             // 16
            .defineShaderStage(endGatewayAnyHitShader_, VK_SHADER_STAGE_ANY_HIT_BIT_KHR)                     // 17
            .defineShaderStage(pointLightShadowMissShader_, VK_SHADER_STAGE_MISS_BIT_KHR)                    // 18
            .defineShaderStage(displacedIntersectionShader_, VK_SHADER_STAGE_INTERSECTION_BIT_KHR)       // 19
            .defineShaderStage(displacedClosestHitShader_, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR)          // 20
            .defineShaderStage(displacedShadowClosestHitShader_, VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR)    // 21
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
            // Hit group 8: DDA displacement (procedural)
            .defineShaderGroup(VK_RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_KHR, VK_SHADER_UNUSED_KHR, 20,
                               VK_SHADER_UNUSED_KHR, 19) // displaced block (CHS=20, rint=19)
            .endShaderGroup()
            .definePipelineLayout(rayTracingDescriptorTables_[0])
            .build(device);

    if (!sharcUpdatePipeline_) {
        std::cerr << "[SHARC] Failed to create update RT pipeline" << std::endl;
        return;
    }

    // Create SBTs for the update pipeline (same structure as main: 4 miss, 9 hit groups)
    sharcUpdateSbts_.resize(framework->swapchain()->imageCount());
    for (int i = 0; i < framework->swapchain()->imageCount(); i++) {
        sharcUpdateSbts_[i] = vk::SBT::create(framework->physicalDevice(), framework->device(), framework->vma(),
                                               sharcUpdatePipeline_, 4, 9);
    }
}

void RayTracingModule::initSharcResolvePipeline() {
    auto framework = framework_.lock();
    if (!framework) return;
    auto device = framework->device();
    VkDevice dev = device->vkDevice();

    // Destroy existing handles before recreating (prevents resource leak on rebuild)
    if (sharcResolvePipeline_ != VK_NULL_HANDLE) {
        vkDestroyPipeline(dev, sharcResolvePipeline_, nullptr);
        sharcResolvePipeline_ = VK_NULL_HANDLE;
    }
    if (sharcResolvePipelineLayout_ != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(dev, sharcResolvePipelineLayout_, nullptr);
        sharcResolvePipelineLayout_ = VK_NULL_HANDLE;
    }

    std::filesystem::path shaderPath = Renderer::folderPath / "shaders";
    sharcResolveShader_ = vk::Shader::create(device, (shaderPath / "world/ray_tracing/sharc_resolve_comp.spv").string());
    if (!sharcResolveShader_) {
        std::cerr << "[SHARC] Failed to load sharc_resolve_comp.spv" << std::endl;
        return;
    }

    // Pipeline layout with push constant only (no descriptor sets — uses BDA)
    VkPushConstantRange pushRange = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 76}; // 19 fields × 4 bytes (with uint64_t alignment)
    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipelineLayoutInfo.setLayoutCount = 0;
    pipelineLayoutInfo.pSetLayouts = nullptr;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushRange;
    vkCreatePipelineLayout(dev, &pipelineLayoutInfo, nullptr, &sharcResolvePipelineLayout_);

    VkComputePipelineCreateInfo pipelineInfo = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipelineInfo.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineInfo.stage.module = sharcResolveShader_->vkShaderModule();
    pipelineInfo.stage.pName = "main";
    pipelineInfo.layout = sharcResolvePipelineLayout_;
    VkResult result = vkCreateComputePipelines(dev, device->pipelineCache(), 1, &pipelineInfo, nullptr, &sharcResolvePipeline_);
    if (result != VK_SUCCESS) {
        std::cerr << "[SHARC] Failed to create resolve compute pipeline: " << result << std::endl;
        sharcResolvePipeline_ = VK_NULL_HANDLE;
    }
}

RayTracingModuleContext::RayTracingModuleContext(std::shared_ptr<FrameworkContext> frameworkContext,
                                                 std::shared_ptr<WorldPipelineContext> worldPipelineContext,
                                                 std::shared_ptr<RayTracingModule> rayTracingModule)
    : WorldModuleContext(frameworkContext, worldPipelineContext),
      rayTracingModule(rayTracingModule),
      rayTracingDescriptorTable(rayTracingModule->rayTracingDescriptorTables_[frameworkContext->frameIndex]),
      sbt(rayTracingModule->sbts_[frameworkContext->frameIndex]),
      sharcUpdateSbt(frameworkContext->frameIndex < rayTracingModule->sharcUpdateSbts_.size()
                     ? rayTracingModule->sharcUpdateSbts_[frameworkContext->frameIndex] : nullptr),
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
    auto ctx0 = frameworkContext.lock();
    if (ctx0) {
        ctx0->worldCommandBuffer->beginLabel("RT:Atmosphere", 0.3f, 0.3f, 0.9f);
    }
    atmosphereContext->render();
    if (ctx0) ctx0->worldCommandBuffer->endLabel();

    if (ctx0) ctx0->worldCommandBuffer->beginLabel("RT:BLAS/TLAS Build", 0.9f, 0.3f, 0.3f);
    worldPrepareContext->render();
    if (ctx0) ctx0->worldCommandBuffer->endLabel();

    if (worldPrepareContext->tlas == nullptr) {
        std::cout << "tlas is nullptr" << std::endl;
        return;
    }

    auto context = frameworkContext.lock();
    if (!context) return;
    auto framework = context->framework.lock();
    if (!framework) return;
    auto worldCommandBuffer = context->worldCommandBuffer;
    auto mainQueueIndex = framework->physicalDevice()->mainQueueIndex();

    auto module = rayTracingModule.lock();
    if (!module) return;

    rayTracingDescriptorTable->bindAS(worldPrepareContext->tlas, 1, 0);

    auto buffers = Renderer::instance().buffers();
    auto worldBuffer = buffers->worldUniformBuffer();

    // Batch all per-frame buffer bindings into a single vkUpdateDescriptorSets call
    using BB = vk::DescriptorTable::BufferBinding;
    std::vector<BB> bufferBindings = {
        {worldPrepareContext->blasOffsetsBuffer,     1,  1},
        {worldPrepareContext->vertexBufferAddr,      1,  2},
        {worldPrepareContext->indexBufferAddr,        1,  3},
        {worldPrepareContext->lastVertexBufferAddr,  1,  4},
        {worldPrepareContext->lastIndexBufferAddr,   1,  5},
        {worldPrepareContext->lastObjToWorldMat,     1,  6},
        {buffers->textureMappingBuffer(),            1,  7},
        {worldPrepareContext->areaLightBuffer,       1,  8},
        {worldBuffer,                                2,  0},
        {buffers->lastWorldUniformBuffer(),          2,  1},
        {buffers->skyUniformBuffer(),                2,  2},
    };
    if (module->tileLightBuffer_) {
        bufferBindings.push_back({module->tileLightBuffer_, 1, 9});
    }
    // Per-section biome colors for shader-side tinting
    if (worldPrepareContext->biomeColorBuffer) {
        bufferBindings.push_back({worldPrepareContext->biomeColorBuffer, 1, 10});
    }
    // Always bind Material Class Mapping buffer (may be real data or dummy)
    auto mcBuffer = buffers->materialClassMappingBuffer();
    if (mcBuffer) {
        bufferBindings.push_back({mcBuffer, 1, 11});
    }
    // SpriteRegistry SSBO for texture array metadata
    auto spriteRegBuffer = Renderer::textureSystem.registry().getBuffer();
    if (spriteRegBuffer) {
        bufferBindings.push_back({spriteRegBuffer, 1, 13});
    }
    // Blue noise buffers (already uploaded at init time via queueImportantWorldUpload)
    if (module->blueNoise_) {
        bufferBindings.push_back({module->blueNoise_->sobolBuffer(),      2, 9});
        bufferBindings.push_back({module->blueNoise_->scramblingBuffer(), 2, 10});
    }
    rayTracingDescriptorTable->bindBufferBatch(bufferBindings);

    // Batch per-frame image rebindings (ReSTIR reservoirs, bounce reservoirs)
    using IB = vk::DescriptorTable::ImageBinding;
    std::vector<IB> frameImageBindings;

    // ReSTIR DI: reservoir image binding
    // Binding 14 = CHS write (temporal output)
    // Binding 15 = CHS read (previous frame source)
    if (module->reservoirImages_[0]) {
        bool spatialEnabled = Renderer::options.restirEnabled && Renderer::options.restirSpatialEnabled && module->spatialPipeline_ != VK_NULL_HANDLE;
        if (spatialEnabled) {
            // Spatial path: CHS writes [0], spatial transforms [0]->[1], next CHS reads [1]
            // No self-aliasing: CHS reads [1] and writes [0]
            frameImageBindings.push_back({module->reservoirImages_[0], VK_IMAGE_LAYOUT_GENERAL, 3, 14});
            frameImageBindings.push_back({module->reservoirImages_[1], VK_IMAGE_LAYOUT_GENERAL, 3, 15});
        } else {
            // No spatial: ping-pong to prevent read-write race on same image
            uint32_t frameIdx = context->frameIndex;
            int writeIdx = frameIdx & 1;
            int readIdx  = 1 - writeIdx;
            frameImageBindings.push_back({module->reservoirImages_[writeIdx], VK_IMAGE_LAYOUT_GENERAL, 3, 14});
            frameImageBindings.push_back({module->reservoirImages_[readIdx],  VK_IMAGE_LAYOUT_GENERAL, 3, 15});
        }
    }

    // Bounce ReSTIR DI: per-bounce reservoir images (bindings 18-20)
    for (int b = 0; b < 3; b++) {
        if (module->bounceReservoirImages_[b]) {
            frameImageBindings.push_back(
                {module->bounceReservoirImages_[b], VK_IMAGE_LAYOUT_GENERAL, 3, static_cast<uint32_t>(18 + b)});
        }
    }
    rayTracingDescriptorTable->bindImages(frameImageBindings);

    // Bind block sprite texture arrays (set 0, bindings 3-5)
    auto& texSystem = Renderer::textureSystem;
    // Flush any pending texture array uploads (staged by animation tick, executed here on render thread)
    if (texSystem.isFinalized()) {
        auto vma = framework->vma();
        auto device = framework->device();
        texSystem.flushPendingUploads(vma, device, worldCommandBuffer, framework->gc());
    }
    // Block albedo array
    auto& texArrayMgr = texSystem.arrayManager();
    auto* albedoArray = texArrayMgr.getArray(texSystem.blockAlbedoArrayId());
    if (albedoArray && albedoArray->image) {
        rayTracingDescriptorTable->bindSamplerImage(
            albedoArray->sampler, albedoArray->image,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 0, 3, 0);
    }
    auto* specArray = texArrayMgr.getArray(texSystem.blockSpecularArrayId());
    if (specArray && specArray->image) {
        rayTracingDescriptorTable->bindSamplerImage(
            specArray->sampler, specArray->image,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 0, 4, 0);
    }
    auto* normArray = texArrayMgr.getArray(texSystem.blockNormalArrayId());
    if (normArray && normArray->image) {
        rayTracingDescriptorTable->bindSamplerImage(
            normArray->sampler, normArray->image,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, 0, 5, 0);
    }

    bool accumulating = Renderer::options.offlineState == 2;

    // Lazy SHARC init: create buffers + pipelines if enabled at runtime but not yet allocated.
    // Must happen BEFORE push constant population so SHARC BDAs are available on the same frame.
    if (Renderer::options.sharcEnabled && !module->sharcHashEntries_) {
        module->sharcCapacity_ = 1u << static_cast<uint32_t>(Renderer::options.sharcCapacityExponent);
        module->initSharcBuffers();
        module->initSharcUpdatePipeline();
        module->initSharcResolvePipeline();
        // Refresh ALL contexts' stale SBT pointers — were nullptr at construction time
        for (size_t ci = 0; ci < module->contexts_.size(); ci++) {
            auto rtCtx = std::static_pointer_cast<RayTracingModuleContext>(module->contexts_[ci]);
            if (ci < module->sharcUpdateSbts_.size()) {
                rtCtx->sharcUpdateSbt = module->sharcUpdateSbts_[ci];
            }
        }
    }

    RayTracingPushConstant pushConstant{};
    pushConstant.numRayBounces = accumulating
        ? static_cast<int>(Renderer::options.offlineBounces)
        : static_cast<int>(Renderer::options.rayBounces);
    pushConstant.flags = (Renderer::options.simplifiedIndirect ? 1 : 0)
                       | (Renderer::options.areaLightsEnabled ? 2 : 0)
                       | (Renderer::options.restirEnabled ? 4 : 0)
                       | (Renderer::options.restirSimplifiedBRDF ? 8 : 0)
                       | (Renderer::options.restirBounceEnabled ? 16 : 0)
                       | ((Renderer::options.sharcEnabled && !accumulating && module->sharcHashEntries_) ? 32 : 0)
                       | (Renderer::options.noiseLOD ? 64 : 0)
                       | (Renderer::options.multiScatterGGX ? 128 : 0)
                       | (Renderer::options.eonDiffuse ? 256 : 0)
                       | (Renderer::options.beerLawShadows ? 512 : 0)
                       | (Renderer::options.noEmissionClamp ? 1024 : 0)
                       | (Renderer::options.physicalSunDisk ? 2048 : 0)
                       | (Renderer::options.noHandAmbient ? 4096 : 0)
                       | ((framework->device()->hasSER() && Renderer::options.serEnabled) ? 8192 : 0)
                       | ((framework->device()->hasSER() && Renderer::options.serEnabled && Renderer::options.serHintsEnabled) ? 16384 : 0)
                       | (Renderer::options.entityNormalsEnabled ? 32768 : 0);
    pushConstant.areaLightCount = worldPrepareContext->areaLightCount;
    pushConstant.shadowSoftness = Renderer::options.shadowSoftness;
    pushConstant.risCandidates = Renderer::options.restirCandidates;
    pushConstant.temporalMClamp = Renderer::options.restirTemporalMClamp;
    pushConstant.wClamp = Renderer::options.restirWClamp;
    // Fixed pre-exposure compresses HDR radiance into fp16-friendly range for DLSS-RR.
    // Must be CONSTANT across frames — varying pre-exposure contaminates DLSS-RR's temporal
    // history (accumulated at different scales) and causes visible brightness oscillation.
    // 0.1 maps Minecraft's typical luminance range into fp16's sweet spot:
    //   Sun/lava (1000) → 100, torch (500) → 50, dark cave (0.01) → 0.001
    // DLSS-RR undoes it via InExposureScale = 1/0.1 = 10.
    // Only apply when DLSS-RR is active (denoiserMode == 1).
    pushConstant.preExposure = (Renderer::options.denoiserMode == 1) ? 0.1f : 1.0f;

    // POM
    pushConstant.pomHeightScale  = Renderer::options.pomEnabled ? Renderer::options.pomHeightScale : 0.0f;
    pushConstant.pomSteps        = Renderer::options.pomSteps;
    pushConstant.pomRefinement   = Renderer::options.pomRefinement;
    pushConstant.pomFadeDistance = Renderer::options.pomFadeDistance;

    // Color expansion
    pushConstant.colorExpansion = Renderer::options.colorExpansion;
    pushConstant.blueNoiseFrame = context->frameIndex;

    // Structured logging: push constants (every ~1 second)
    if (RadianceLogger::isEnabled()) {
        static uint64_t lastPCLog = 0;
        uint64_t curFrame = g_crashRing.frameCount();
        if (curFrame - lastPCLog >= 60) {
            lastPCLog = curFrame;
            RadianceLogger::log("RayTracing", "INFO",
                "pushConst: bounces=%d flags=0x%x lights=%d shadowSoft=%.2f pomH=%.4f pomSteps=%d pomFade=%.0f",
                pushConstant.numRayBounces, pushConstant.flags, pushConstant.areaLightCount,
                pushConstant.shadowSoftness, pushConstant.pomHeightScale,
                pushConstant.pomSteps, pushConstant.pomFadeDistance);
        }
    }

    // SHARC radiance cache
    if (Renderer::options.sharcEnabled && module->sharcHashEntries_) {
        auto worldUBO = static_cast<vk::Data::WorldUBO *>(buffers->worldUniformBuffer()->mappedPtr());
        pushConstant.sharcHashEntries = module->sharcHashEntries_->bufferAddress();
        pushConstant.sharcAccumulation = module->sharcAccumulation_->bufferAddress();
        pushConstant.sharcResolved = module->sharcResolved_->bufferAddress();
        pushConstant.sharcCameraX = static_cast<float>(worldUBO->cameraPos.x);
        pushConstant.sharcCameraY = static_cast<float>(worldUBO->cameraPos.y);
        pushConstant.sharcCameraZ = static_cast<float>(worldUBO->cameraPos.z);
        pushConstant.sharcSceneScale = Renderer::options.sharcSceneScale;
        pushConstant.sharcCapacity = module->sharcCapacity_;
        pushConstant.sharcRadianceScale = 1e3f;
        pushConstant.sharcFrameIndex = module->sharcFrameIndex_;
        pushConstant.sharcRoughnessThreshold = Renderer::options.sharcRoughnessThreshold;
        pushConstant.sharcUpdateBlockSize = Renderer::options.sharcUpdateBlockSize;
        pushConstant.sharcUpdateBounces = Renderer::options.sharcUpdateBounces;
    }

    // Offline accumulation
    // offlineDenoised: 0=Raw Fast (RR on), 1=Raw Accurate (RR off), 2=Denoised (epoch-based DLSS-RR)
    // Bit 1 (disableRR): auto-set by preset — Raw Accurate forces RR off
    bool disableRR = Renderer::options.offlineDisableRR
                     || Renderer::options.offlineDenoised == 1;
    pushConstant.offlineFlags = (accumulating ? 1 : 0)
                              | (disableRR ? 2 : 0)
                              | (Renderer::options.offlineDisableClamp ? 4 : 0)
                              | (Renderer::options.offlineNativeResActive ? 8 : 0);
    pushConstant.accumFrameCount = static_cast<int>(Renderer::accumFrameCount);
    // DOF: active in both FREE (preview) and ACCUMULATING modes
    bool dofEnabled = (accumulating || Renderer::options.offlineState == 1)
                      && Renderer::options.offlineAperture > 0.0f;
    float effectiveAperture = Renderer::options.offlineAperture * Renderer::options.dofStrength;
    pushConstant.aperture = dofEnabled ? effectiveAperture : 0.0f;
    pushConstant.focalDistance = Renderer::options.offlineFocalDistance;

    // Pre-exposure locked to 1.0 during accumulation (all presets).
    // Gives full fp16 precision for the Welford accumulator input (DLSS output at
    // scene-referred scale). DLSS-RR exposure params are neutral (§3.7: not supported),
    // so output stays in the same pre-exposed space as input — consistent with histogram.
    // All modes disable temporal reuse (each frame is independent).
    if (accumulating) {
        pushConstant.preExposure = 1.0f;
        pushConstant.temporalMClamp = 0;
    }

    // Material SSBO BDA: pass buffer address via push constant (avoids descriptor lookup overhead)
    pushConstant.materialClassAddr = mcBuffer ? mcBuffer->bufferAddress() : 0;

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

    // Reset SHARC buffers when disabled so re-enable starts fresh (prevents stale cache artifacts)
    if (!Renderer::options.sharcEnabled && module->sharcBuffersInitialized_) {
        module->sharcBuffersInitialized_ = false;
        module->sharcFrameIndex_ = 0;
    }

    // Deferred SHARC resize: process pending capacity change at frame boundary.
    // Previous frame's SHARC dispatch used old buffers via BDA — drain GPU before freeing them.
    // This is still vkDeviceWaitIdle, but deferred to the next frame boundary instead of inline,
    // so it doesn't stall mid-render and the wait is shorter (one frame of latency absorbed).
    if (module->sharcResizePending_ && module->sharcHashEntries_) {
        uint32_t desiredCapacity = 1u << static_cast<uint32_t>(
            std::max(18, std::min(26, Renderer::options.sharcCapacityExponent)));
        auto fw = module->framework_.lock();
        if (fw) {
            vkDeviceWaitIdle(fw->device()->vkDevice());

            VkBufferUsageFlags sharcUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            module->sharcCapacity_ = desiredCapacity;
            module->sharcHashEntries_ = vk::DeviceLocalBuffer::create(
                fw->vma(), fw->device(), desiredCapacity * 8, sharcUsage);
            module->sharcAccumulation_ = vk::DeviceLocalBuffer::create(
                fw->vma(), fw->device(), desiredCapacity * 16, sharcUsage);
            module->sharcResolved_ = vk::DeviceLocalBuffer::create(
                fw->vma(), fw->device(), desiredCapacity * 16, sharcUsage);
            module->sharcBuffersInitialized_ = false;
            module->sharcFrameIndex_ = 0;
            // Update push constant BDAs for this frame
            pushConstant.sharcHashEntries = module->sharcHashEntries_->bufferAddress();
            pushConstant.sharcAccumulation = module->sharcAccumulation_->bufferAddress();
            pushConstant.sharcResolved = module->sharcResolved_->bufferAddress();
            pushConstant.sharcCapacity = desiredCapacity;
        }
        module->sharcResizePending_ = false;
    }

    // Check if SHARC capacity exponent changed — mark for deferred resize at next frame boundary
    {
        uint32_t desiredCapacity = 1u << static_cast<uint32_t>(
            std::max(18, std::min(26, Renderer::options.sharcCapacityExponent)));
        if (desiredCapacity != module->sharcCapacity_ && module->sharcHashEntries_) {
            module->sharcResizePending_ = true;
        }
    }

    // SHARC 3-pass dispatch: update → resolve → main render (with cache query)
    // Skip entirely during offline accumulation — SHARC temporal cache conflicts with independent samples
    if (Renderer::options.sharcEnabled && !accumulating
        && module->sharcUpdatePipeline_ && module->sharcResolvePipeline_ != VK_NULL_HANDLE
        && module->sharcHashEntries_ && sharcUpdateSbt) {
        worldCommandBuffer->beginLabel("RT:SHARC Update", 0.9f, 0.6f, 0.1f);
        VkCommandBuffer cmd = worldCommandBuffer->vkCommandBuffer();
        GpuDiag::checkpoint(cmd, GpuDiag::SHARC_UPDATE_RT);

        // Zero-fill SHARC buffers on first use
        if (!module->sharcBuffersInitialized_) {
            vkCmdFillBuffer(cmd, module->sharcHashEntries_->vkBuffer(), 0, VK_WHOLE_SIZE, 0);
            vkCmdFillBuffer(cmd, module->sharcAccumulation_->vkBuffer(), 0, VK_WHOLE_SIZE, 0);
            vkCmdFillBuffer(cmd, module->sharcResolved_->vkBuffer(), 0, VK_WHOLE_SIZE, 0);

            VkMemoryBarrier fillBarrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            fillBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            fillBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier(cmd,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
                0, 1, &fillBarrier, 0, nullptr, 0, nullptr);
            module->sharcBuffersInitialized_ = true;
        }

        // Pass 1: SHARC Update (sparse RT — populates cache with bounce radiance)
        // Disable ReSTIR in update pass to prevent CHS reservoir writes.
        // Area lights (bit 1) stay ENABLED — CHS fallback paths (deterministic top-2 shadowed
        // for primary, unshadowed 8-light accumulation for bounce) do NOT write reservoirs.
        RayTracingPushConstant updatePC = pushConstant;
        updatePC.flags &= ~(4 | 16); // Clear restir (4), restir bounce (16) — keep area lights
        vkCmdPushConstants(cmd, rayTracingDescriptorTable->vkPipelineLayout(),
                           VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                               VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                               VK_SHADER_STAGE_INTERSECTION_BIT_KHR,
                           0, sizeof(RayTracingPushConstant), &updatePC);

        int ds = std::max(1, Renderer::options.sharcDownscale);
        worldCommandBuffer->bindDescriptorTable(rayTracingDescriptorTable, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR)
            ->bindRTPipeline(module->sharcUpdatePipeline_)
            ->raytracing(sharcUpdateSbt, hdrNoisyOutputImage->width() / ds, hdrNoisyOutputImage->height() / ds, 1);

        // Barrier: update RT writes → resolve compute reads
        VkMemoryBarrier updateBarrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        updateBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        updateBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 1, &updateBarrier, 0, nullptr, 0, nullptr);

        worldCommandBuffer->endLabel(); // end SHARC Update
        worldCommandBuffer->beginLabel("RT:SHARC Resolve", 0.9f, 0.7f, 0.2f);
        GpuDiag::checkpoint(cmd, GpuDiag::SHARC_RESOLVE);
        // Pass 2: SHARC Resolve (compute — temporal blend + stale eviction)
        struct SharcResolvePushConstant {
            float cameraPositionPrevX, cameraPositionPrevY, cameraPositionPrevZ;
            uint32_t accumulationFrameNum;
            uint32_t staleFrameNumMax;
            uint32_t frameIndex;
            uint64_t hashEntriesBDA;
            uint64_t accumulationBDA;
            uint64_t resolvedBDA;
            float cameraX, cameraY, cameraZ;
            float sceneScale;
            uint32_t capacity;
            float radianceScale;
            uint32_t enableAntiFirefly;
        } resolvePC = {};

        resolvePC.cameraPositionPrevX = module->sharcPrevCameraX_;
        resolvePC.cameraPositionPrevY = module->sharcPrevCameraY_;
        resolvePC.cameraPositionPrevZ = module->sharcPrevCameraZ_;
        resolvePC.accumulationFrameNum = static_cast<uint32_t>(Renderer::options.sharcAccumulationFrames);
        resolvePC.staleFrameNumMax = static_cast<uint32_t>(Renderer::options.sharcStaleFrames);
        resolvePC.frameIndex = module->sharcFrameIndex_;
        resolvePC.hashEntriesBDA = module->sharcHashEntries_->bufferAddress();
        resolvePC.accumulationBDA = module->sharcAccumulation_->bufferAddress();
        resolvePC.resolvedBDA = module->sharcResolved_->bufferAddress();
        resolvePC.cameraX = pushConstant.sharcCameraX;
        resolvePC.cameraY = pushConstant.sharcCameraY;
        resolvePC.cameraZ = pushConstant.sharcCameraZ;
        resolvePC.sceneScale = Renderer::options.sharcSceneScale;
        resolvePC.capacity = module->sharcCapacity_;
        resolvePC.radianceScale = 1e3f;
        resolvePC.enableAntiFirefly = 1;

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, module->sharcResolvePipeline_);
        vkCmdPushConstants(cmd, module->sharcResolvePipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT,
            0, 76, &resolvePC);
        vkCmdDispatch(cmd, (module->sharcCapacity_ + 63) / 64, 1, 1);

        // Barrier: resolve compute writes → main RT reads
        VkMemoryBarrier resolveBarrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        resolveBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        resolveBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
            0, 1, &resolveBarrier, 0, nullptr, 0, nullptr);

        // Update tracking for next frame
        module->sharcPrevCameraX_ = pushConstant.sharcCameraX;
        module->sharcPrevCameraY_ = pushConstant.sharcCameraY;
        module->sharcPrevCameraZ_ = pushConstant.sharcCameraZ;
        module->sharcFrameIndex_++;
        worldCommandBuffer->endLabel(); // end SHARC Resolve
    }

    // Re-push RT push constants (may have been invalidated by compute pipeline bind above)
    vkCmdPushConstants(worldCommandBuffer->vkCommandBuffer(), rayTracingDescriptorTable->vkPipelineLayout(),
                       VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                           VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                           VK_SHADER_STAGE_INTERSECTION_BIT_KHR,
                       0, sizeof(RayTracingPushConstant), &pushConstant);

    // Pass 3: Main Render (existing RT dispatch — now queries SHARC cache on bounces >= 1)
    worldCommandBuffer->beginLabel("RT:MainTrace", 1.0f, 0.2f, 0.2f);
    GpuDiag::checkpoint(worldCommandBuffer->vkCommandBuffer(), GpuDiag::RT_DISPATCH_MAIN);
    worldCommandBuffer->bindDescriptorTable(rayTracingDescriptorTable, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR)
        ->bindRTPipeline(module->rayTracingPipeline_)
        ->raytracing(sbt, hdrNoisyOutputImage->width(), hdrNoisyOutputImage->height(), 1);
    worldCommandBuffer->endLabel(); // end MainTrace

    // Spatial reuse compute pass (when ReSTIR and spatial reuse are both enabled)
    if (Renderer::options.restirEnabled && Renderer::options.restirSpatialEnabled && module->spatialPipeline_ != VK_NULL_HANDLE
        && module->reservoirImages_[0] && module->reservoirImages_[1]) {
        worldCommandBuffer->beginLabel("RT:ReSTIR Spatial", 0.2f, 0.8f, 0.8f);
        VkCommandBuffer cmd = worldCommandBuffer->vkCommandBuffer();
        GpuDiag::checkpoint(cmd, GpuDiag::RESTIR_SPATIAL);

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
        // Spatial reads CHS temporal output [0], writes spatial output [1]
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
        worldCommandBuffer->endLabel(); // end ReSTIR Spatial
    }

    // Offline accumulation: Welford running average into RGBA32F buffer
    // Raw Fast (0) and Raw Slow (1) accumulate here; DLSS-D (2) accumulates in DLSS module
    if (accumulating && Renderer::accumPipelineReady && Renderer::options.offlineDenoised != 2) {
        VkCommandBuffer cmd = worldCommandBuffer->vkCommandBuffer();
        uint32_t frameIdx = context->frameIndex;

        // Barrier: RT output → compute read
        VkMemoryBarrier accumBarrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        accumBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        accumBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 1, &accumBarrier, 0, nullptr, 0, nullptr);

        // Update descriptor set (only when images change — typically once at init)
        VkDescriptorSet accumSet = Renderer::accumDescSets[frameIdx];
        VkDescriptorImageInfo accumImgInfo{VK_NULL_HANDLE, Renderer::accumBufferImage->vkImageView(0), VK_IMAGE_LAYOUT_GENERAL};
        VkDescriptorImageInfo noisyImgInfo{VK_NULL_HANDLE, hdrNoisyOutputImage->vkImageView(0), VK_IMAGE_LAYOUT_GENERAL};
        VkWriteDescriptorSet accumWrites[] = {
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, accumSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &accumImgInfo, nullptr, nullptr},
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, accumSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &noisyImgInfo, nullptr, nullptr},
        };
        vkUpdateDescriptorSets(framework->device()->vkDevice(), 2, accumWrites, 0, nullptr);

        // Dispatch accumulation (writes averaged result to accumBuffer only)
        int32_t fc = static_cast<int32_t>(Renderer::accumFrameCount);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, Renderer::accumPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, Renderer::accumPipelineLayout,
            0, 1, &accumSet, 0, nullptr);
        vkCmdPushConstants(cmd, Renderer::accumPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, &fc);
        vkCmdDispatch(cmd,
            (hdrNoisyOutputImage->width() + 7) / 8,
            (hdrNoisyOutputImage->height() + 7) / 8,
            1);

        // Barrier: compute write → blit read
        VkMemoryBarrier postAccumBarrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        postAccumBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        postAccumBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 1, &postAccumBarrier, 0, nullptr, 0, nullptr);

        // Copy accumulated average from RGBA32F accumBuffer to RGBA16F hdrNoisyOutput
        VkImageBlit accumBlit{};
        accumBlit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        accumBlit.srcOffsets[1] = {(int)Renderer::accumBufferImage->width(), (int)Renderer::accumBufferImage->height(), 1};
        accumBlit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        accumBlit.dstOffsets[1] = {(int)hdrNoisyOutputImage->width(), (int)hdrNoisyOutputImage->height(), 1};
        vkCmdBlitImage(cmd,
            Renderer::accumBufferImage->vkImage(), VK_IMAGE_LAYOUT_GENERAL,
            hdrNoisyOutputImage->vkImage(), VK_IMAGE_LAYOUT_GENERAL,
            1, &accumBlit, VK_FILTER_NEAREST);

        // Barrier: blit write → next stage read
        VkMemoryBarrier postBlitBarrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        postBlitBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        postBlitBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 1, &postBlitBarrier, 0, nullptr, 0, nullptr);

        Renderer::accumFrameCount++;
        Renderer::accumOutputImage = hdrNoisyOutputImage;  // expose for denoiser/upscaler bypass
    }

    // (P4 removed — DLSS temporal mode replaced by DLSS-D Converge with per-frame reset)
}
