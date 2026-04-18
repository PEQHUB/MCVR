#include "cloud_adapter.hpp"
#include "app/engine_services.hpp"
#include "diagnostics/metrics_service.hpp"
#include "frame/frame_scheduler.hpp"
#include "scene/scene_resource_service.hpp"
#include "platform/vulkan/vk2_device.hpp"
#include "diagnostics/log.hpp"

#include <volk.h>

#include <algorithm>
#include <cmath>

namespace engine {

// ============================================================================
// Static image-barrier helper (mirrors atmosphere_adapter.cpp convention)
// ============================================================================

/*static*/ void CloudAdapter::imageBarrier(VkCommandBuffer cmd, VkImage image,
                                           VkImageLayout oldLayout, VkImageLayout newLayout,
                                           VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess,
                                           VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess,
                                           uint32_t layerCount, uint32_t depth) {
    VkImageSubresourceRange range{};
    range.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    range.baseMipLevel   = 0;
    range.levelCount     = 1;
    range.baseArrayLayer = 0;
    range.layerCount     = layerCount;

    // For 3D images depth > 1; Vulkan treats 3D image "layers" as 1 and depth separately.
    // The subresource range still uses layerCount = 1 for 3D images.
    (void)depth;

    VkImageMemoryBarrier2 b{};
    b.sType             = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    b.srcStageMask      = srcStage;
    b.srcAccessMask     = srcAccess;
    b.dstStageMask      = dstStage;
    b.dstAccessMask     = dstAccess;
    b.oldLayout         = oldLayout;
    b.newLayout         = newLayout;
    b.image             = image;
    b.subresourceRange  = range;

    VkDependencyInfo dep{};
    dep.sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.imageMemoryBarrierCount = 1;
    dep.pImageMemoryBarriers    = &b;
    vkCmdPipelineBarrier2(cmd, &dep);
}

// ============================================================================
// init
// ============================================================================

void CloudAdapter::init(EngineServices& services) {
    services_ = &services;
    device_   = services.device().device();

    std::string shaderBase = services.resourceDir() + "/shaders/world/cloud/";

    // --------------------------------------------------------------------------
    // Load shaders
    // --------------------------------------------------------------------------

    auto loadShader = [&](vk2::ShaderModule& out, const std::string& filename) -> bool {
        auto r = vk2::ShaderModule::fromFile(device_, shaderBase + filename);
        if (!r) {
            log::warn("cloud", "Failed to load cloud shader " + filename +
                      ": " + r.error().message + " — cloud adapter disabled");
            return false;
        }
        out = std::move(r.value());
        return true;
    };

    if (!loadShader(noiseGenShader_,   "cloud_noise_gen_comp.spv"))   return;
    if (!loadShader(weatherShader_,    "cloud_weather_comp.spv"))     return;
    if (!loadShader(raymarchShader_,   "cloud_raymarch_comp.spv"))    return;
    if (!loadShader(temporalShader_,   "cloud_temporal_comp.spv"))    return;
    // The composite pass SPV is named "cloud_coverage_comp.spv" in the resource dir.
    // (V1 source calls it composite; the compiled output was renamed for coverage blend.)
    if (!loadShader(compositeShader_,  "cloud_coverage_comp.spv"))    return;
    if (!loadShader(shadowShader_,     "cloud_shadow_comp.spv"))      return;

    // --------------------------------------------------------------------------
    // Descriptor set layouts
    // --------------------------------------------------------------------------

    // --- Set 0: 10 image bindings ---
    {
        VkDescriptorSetLayoutBinding bindings[10]{};

        // b0 radiance input — storage image (read by composite)
        bindings[0] = {0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        // b1 linear depth — storage image (read by raymarch / composite)
        bindings[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        // b2 cloud radiance output — storage image
        bindings[2] = {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        // b3 cloudColor (cloud-res HDR) — storage image
        bindings[3] = {3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        // b4 cloudHistory — storage image
        bindings[4] = {4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        // b5 weatherMap (written by weather pass) — storage image
        bindings[5] = {5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        // b6 noiseTexture3D (sampled by raymarch) — combined image sampler
        bindings[6] = {6, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        // b7 cloudShadowMap output — storage image
        bindings[7] = {7, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        // b8 weatherMap (sampled by raymarch) — combined image sampler
        bindings[8] = {8, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        // b9 noiseTexture3D (written by noise_gen) — storage image
        bindings[9] = {9, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};

        VkDescriptorSetLayoutCreateInfo ci{};
        ci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        ci.bindingCount = 10;
        ci.pBindings    = bindings;
        if (vkCreateDescriptorSetLayout(device_, &ci, nullptr, &descLayout0_) != VK_SUCCESS) {
            log::error("cloud", "Failed to create cloud descriptor set layout 0");
            return;
        }
    }

    // --- Set 1: 3 UBO bindings ---
    {
        VkDescriptorSetLayoutBinding bindings[3]{};
        // b0 WorldUBO
        bindings[0] = {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        // b1 SkyUBO
        bindings[1] = {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        // b2 LastWorldUBO
        bindings[2] = {2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};

        VkDescriptorSetLayoutCreateInfo ci{};
        ci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        ci.bindingCount = 3;
        ci.pBindings    = bindings;
        if (vkCreateDescriptorSetLayout(device_, &ci, nullptr, &descLayout1_) != VK_SUCCESS) {
            log::error("cloud", "Failed to create cloud descriptor set layout 1");
            return;
        }
    }

    // --------------------------------------------------------------------------
    // Descriptor allocators
    // --------------------------------------------------------------------------

    uint32_t fif = services.frame().framesInFlight();

    {
        std::vector<VkDescriptorPoolSize> sizes = {
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,        8},  // b0-b5, b7, b9
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2}, // b6, b8
        };
        auto r = descAllocator0_.init(device_, fif, sizes, 2);
        if (!r) {
            log::error("cloud", "Cloud desc allocator 0 init failed: " + r.error().message);
            return;
        }
    }

    {
        std::vector<VkDescriptorPoolSize> sizes = {
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 3},
        };
        auto r = descAllocator1_.init(device_, fif, sizes, 2);
        if (!r) {
            log::error("cloud", "Cloud desc allocator 1 init failed: " + r.error().message);
            return;
        }
    }

    // --------------------------------------------------------------------------
    // Pipeline layout (shared by all 6 passes)
    // --------------------------------------------------------------------------

    {
        VkPushConstantRange pcRange{};
        pcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcRange.offset     = 0;
        pcRange.size       = sizeof(CloudPushConstant);

        vk2::PipelineLayoutDesc pld;
        pld.setLayouts   = {descLayout0_, descLayout1_};
        pld.pushConstants = {pcRange};

        auto r = vk2::PipelineLayout::create(device_, pld);
        if (!r) {
            log::error("cloud", "Cloud pipeline layout failed: " + r.error().message);
            return;
        }
        pipelineLayout_ = std::move(r.value());
    }

    // --------------------------------------------------------------------------
    // Compute pipelines
    // --------------------------------------------------------------------------

    auto makePipeline = [&](vk2::ComputePipeline& out, const vk2::ShaderModule& shader,
                             const char* label) -> bool {
        vk2::ComputePipelineDesc cpd;
        cpd.shaderModule = shader.handle();
        cpd.layout       = pipelineLayout_.handle();
        auto r = vk2::ComputePipeline::create(device_, cpd);
        if (!r) {
            log::error("cloud", std::string("Cloud pipeline ") + label + " failed: " +
                       r.error().message);
            return false;
        }
        out = std::move(r.value());
        return true;
    };

    if (!makePipeline(noiseGenPipeline_,   noiseGenShader_,   "noise_gen"))   return;
    if (!makePipeline(weatherPipeline_,    weatherShader_,    "weather"))     return;
    if (!makePipeline(raymarchPipeline_,   raymarchShader_,   "raymarch"))    return;
    if (!makePipeline(temporalPipeline_,   temporalShader_,   "temporal"))    return;
    if (!makePipeline(compositePipeline_,  compositeShader_,  "composite"))   return;
    if (!makePipeline(shadowPipeline_,     shadowShader_,     "shadow"))      return;

    // --------------------------------------------------------------------------
    // Samplers
    // --------------------------------------------------------------------------

    // Noise sampler: REPEAT + linear (tiled 3D noise)
    {
        VkSamplerCreateInfo si{};
        si.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        si.magFilter    = VK_FILTER_LINEAR;
        si.minFilter    = VK_FILTER_LINEAR;
        si.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        si.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        si.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        si.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        if (vkCreateSampler(device_, &si, nullptr, &noiseSampler_) != VK_SUCCESS) {
            log::error("cloud", "Cloud noise sampler creation failed");
            return;
        }
    }

    // Weather sampler: CLAMP_TO_EDGE + linear
    {
        VkSamplerCreateInfo si{};
        si.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        si.magFilter    = VK_FILTER_LINEAR;
        si.minFilter    = VK_FILTER_LINEAR;
        si.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        if (vkCreateSampler(device_, &si, nullptr, &weatherSampler_) != VK_SUCCESS) {
            log::error("cloud", "Cloud weather sampler creation failed");
            return;
        }
    }

    // --------------------------------------------------------------------------
    // Static internal images (weather map, 3D noise, shadow map)
    // Resolution-dependent images (cloudColor per-frame, history) are created
    // lazily in createInternalImages() on the first execute() call.
    // --------------------------------------------------------------------------

    auto vmaAlloc = services.device().vma();

    // Weather map: 512×512 RGBA8
    {
        vk2::Image::Desc desc{};
        desc.width   = 512;
        desc.height  = 512;
        desc.format  = VK_FORMAT_R8G8B8A8_UNORM;
        desc.usage   = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        auto r = vk2::Image::create(device_, vmaAlloc, desc);
        if (!r) { log::error("cloud", "Weather map image: " + r.error().message); return; }
        weatherMapImage_ = std::move(r.value());
        weatherLayout_   = VK_IMAGE_LAYOUT_UNDEFINED;
    }

    // 3D noise texture: 128³ RGBA8 with VK_IMAGE_TYPE_3D
    {
        vk2::Image::Desc desc{};
        desc.width     = NOISE_RES;
        desc.height    = NOISE_RES;
        desc.depth     = NOISE_RES;
        desc.format    = VK_FORMAT_R8G8B8A8_UNORM;
        desc.imageType = VK_IMAGE_TYPE_3D;
        desc.usage     = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        auto r = vk2::Image::create(device_, vmaAlloc, desc);
        if (!r) { log::error("cloud", "3D noise image: " + r.error().message); return; }
        noiseTexture3D_ = std::move(r.value());
        noiseLayout_    = VK_IMAGE_LAYOUT_UNDEFINED;
    }

    // Shadow map: shadowMapSize × shadowMapSize R16F
    {
        vk2::Image::Desc desc{};
        desc.width   = shadowMapSize_;
        desc.height  = shadowMapSize_;
        desc.format  = VK_FORMAT_R16_SFLOAT;
        desc.usage   = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        auto r = vk2::Image::create(device_, vmaAlloc, desc);
        if (!r) { log::error("cloud", "Shadow map image: " + r.error().message); return; }
        cloudShadowImage_ = std::move(r.value());
        shadowLayout_     = VK_IMAGE_LAYOUT_UNDEFINED;
    }

    initialized_ = true;
    profileSlot_ = services.metrics().registerTimer("Cloud");

    log::info("cloud", "CloudAdapter initialized (noise " +
              std::to_string(NOISE_RES) + "^3, shadow " +
              std::to_string(shadowMapSize_) + "x" + std::to_string(shadowMapSize_) + ")");
}

// ============================================================================
// registerPass
// ============================================================================

ResourceHandle CloudAdapter::registerPass(GraphBuilder& builder,
                                          ResourceHandle radianceHandle,
                                          ResourceHandle depthHandle) {
    inputRadianceHandle_ = radianceHandle;
    inputDepthHandle_    = depthHandle;

    // Declare the cloud-composited radiance output (same format/size as radiance input).
    // We use RGBA16F to match the DLSS/denoiser output format.
    ImageResourceDesc outDesc{};
    outDesc.name       = "cloud_radiance";
    outDesc.format     = VK_FORMAT_R16G16B16A16_SFLOAT;
    outDesc.widthScale = 1.0f;
    outDesc.heightScale = 1.0f;
    cloudRadianceHandle_ = builder.addResource(outDesc);

    // Cloud shadow map is an internal image (not a graph resource) because it has a
    // fixed resolution independent of the render target. We expose it as a handle for
    // downstream consumers but it is not allocated by the graph resource pool.
    // Use an invalid handle here; downstream code that needs the VkImage accesses it
    // via services' resource pool indirectly, or the RT adapter binds it directly.
    cloudShadowHandle_ = ResourceHandle{};  // Not a graph resource; see cloudShadowImage_

    PassDescriptor pass;
    pass.name   = "cloud";
    pass.queue  = QueueAffinity::Graphics;
    pass.inputs.push_back(radianceHandle);
    pass.inputs.push_back(depthHandle);
    pass.outputs.push_back(cloudRadianceHandle_);
    pass.execute = [this](const FrameContext& ctx, const PassResources& res) {
        execute(ctx, res);
    };
    builder.addPass(std::move(pass));

    return cloudRadianceHandle_;
}

// ============================================================================
// configure
// ============================================================================

void CloudAdapter::configure(const ConfigSnapshot& config) {
    const auto& c = config.data;

    applyQualityPreset(c.cloudQuality);

    cloudBase_          = std::clamp(c.cloudAltitude,           64.0f,  320.0f);
    cloudThickness_     = std::clamp(c.cloudThickness,          16.0f,  256.0f);
    coverage_           = std::clamp(c.cloudCoverage,           0.0f,   1.0f);
    cloudType_          = std::clamp(c.cloudType,               0.0f,   1.0f);
    densityMultiplier_  = std::clamp(c.cloudDensity,            0.0f,   5.0f);
    windSpeed_          = c.cloudSpeed;
    detailStrength_     = std::clamp(c.cloudDetailStrength,     0.0f,   3.0f);
    scatterOctaves_     = std::clamp(c.cloudScatterOctaves,     1u,     8u);
    ambientStrength_    = std::clamp(c.cloudAmbientStrength,    0.0f,   3.0f);
    noiseScale_         = std::clamp(c.cloudNoiseScale,         16.0f,  4096.0f);
    cellFrequency_      = std::clamp(c.cloudCellFrequency,      1.0f,   32.0f);
    atmosphereFadeDist_ = std::clamp(c.cloudAtmosphereFadeDist, 100.0f, 4000.0f);
    windAngle_          = c.cloudWindAngle;
    debugMode_          = std::clamp(c.cloudDebugMode,          0u,     8u);

    // Override temporal blend if set explicitly in config (cloudTemporalBlend > 0).
    // If the schema stores a blend override (positive), use it; otherwise keep preset.
    if (c.cloudTemporalBlend > 0.0f) {
        temporalBlend_ = std::clamp(c.cloudTemporalBlend, 0.0f, 1.0f);
    }
}

// ============================================================================
// applyQualityPreset
// ============================================================================

void CloudAdapter::applyQualityPreset(uint32_t quality) {
    cloudQuality_ = quality;
    switch (quality) {
        case 0: // off
            resolutionDivisor_ = 2; marchSteps_ = 64; lightSteps_ = 4;
            temporalBlend_ = 0.95f; shadowMapSize_ = 256;
            break;
        case 1: // low
            resolutionDivisor_ = 4; marchSteps_ = 48; lightSteps_ = 3;
            temporalBlend_ = 0.97f; shadowMapSize_ = 128;
            break;
        case 2: // medium
            resolutionDivisor_ = 4; marchSteps_ = 64; lightSteps_ = 4;
            temporalBlend_ = 0.95f; shadowMapSize_ = 256;
            break;
        case 3: // high
            resolutionDivisor_ = 2; marchSteps_ = 64; lightSteps_ = 4;
            temporalBlend_ = 0.95f; shadowMapSize_ = 256;
            break;
        case 4: // ultra
            resolutionDivisor_ = 2; marchSteps_ = 96; lightSteps_ = 6;
            temporalBlend_ = 0.93f; shadowMapSize_ = 256;
            break;
        case 5: // extreme
            resolutionDivisor_ = 1; marchSteps_ = 96; lightSteps_ = 6;
            temporalBlend_ = 0.93f; shadowMapSize_ = 512;
            break;
        case 6: // insane
            resolutionDivisor_ = 1; marchSteps_ = 128; lightSteps_ = 8;
            temporalBlend_ = 0.90f; shadowMapSize_ = 512;
            break;
        default:
            resolutionDivisor_ = 2; marchSteps_ = 64; lightSteps_ = 4;
            temporalBlend_ = 0.95f; shadowMapSize_ = 256;
            break;
    }
}

// ============================================================================
// createInternalImages / destroyInternalImages
// ============================================================================

void CloudAdapter::createInternalImages(uint32_t renderW, uint32_t renderH) {
    if (!initialized_) return;

    uint32_t newCloudW = (renderW + resolutionDivisor_ - 1) / resolutionDivisor_;
    uint32_t newCloudH = (renderH + resolutionDivisor_ - 1) / resolutionDivisor_;

    // Check if we need to (re)create resolution-dependent images
    if (imagesReady_ && newCloudW == cloudW_ && newCloudH == cloudH_ &&
        renderW == renderW_ && renderH == renderH_) {
        return;  // Same resolution, no change needed
    }

    if (imagesReady_) {
        // Recreate: wait for device idle before destroying
        vkDeviceWaitIdle(device_);
        destroyInternalImages();
    }

    cloudW_ = newCloudW;
    cloudH_ = newCloudH;
    renderW_ = renderW;
    renderH_ = renderH;

    auto vma = services_->device().vma();
    uint32_t fif = services_->frame().framesInFlight();

    // Per-frame cloud color buffers (cloud-res RGBA16F)
    cloudColorImages_.resize(fif);
    for (uint32_t i = 0; i < fif; ++i) {
        vk2::Image::Desc desc{};
        desc.width  = cloudW_;
        desc.height = cloudH_;
        desc.format = VK_FORMAT_R16G16B16A16_SFLOAT;
        desc.usage  = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                      VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        auto r = vk2::Image::create(device_, vma, desc);
        if (!r) {
            log::error("cloud", "cloudColor[" + std::to_string(i) + "]: " + r.error().message);
            return;
        }
        cloudColorImages_[i] = std::move(r.value());
    }

    // Shared history image (cloud-res RGBA16F)
    {
        vk2::Image::Desc desc{};
        desc.width  = cloudW_;
        desc.height = cloudH_;
        desc.format = VK_FORMAT_R16G16B16A16_SFLOAT;
        desc.usage  = VK_IMAGE_USAGE_STORAGE_BIT |
                      VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                      VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        auto r = vk2::Image::create(device_, vma, desc);
        if (!r) {
            log::error("cloud", "cloudHistory: " + r.error().message);
            return;
        }
        cloudHistoryImage_ = std::move(r.value());
        historyLayout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    }

    imagesReady_     = true;
    noiseGenerated_  = false;  // Re-generate noise if images were recreated
    lastFrameTime_   = {};

    log::info("cloud", "CloudAdapter images created: render " +
              std::to_string(renderW) + "x" + std::to_string(renderH) +
              ", cloud " + std::to_string(cloudW_) + "x" + std::to_string(cloudH_));
}

void CloudAdapter::destroyInternalImages() {
    cloudColorImages_.clear();
    cloudHistoryImage_ = {};
    historyLayout_     = VK_IMAGE_LAYOUT_UNDEFINED;
    imagesReady_       = false;
}

// ============================================================================
// Descriptor set write helpers
// ============================================================================

VkDescriptorSet CloudAdapter::allocAndWriteDescSet0(uint32_t frameIndex,
                                                      VkImageView radianceView,
                                                      VkImageView depthView,
                                                      VkImageView cloudRadianceView) {
    auto setR = descAllocator0_.allocate(descLayout0_, frameIndex);
    if (!setR) return VK_NULL_HANDLE;
    VkDescriptorSet set = setR.value();

    // Build all 10 write descriptors.
    VkDescriptorImageInfo infos[10]{};

    // b0 radiance input (storage image, GENERAL)
    infos[0].imageView   = radianceView;
    infos[0].imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    // b1 linear depth (storage image, GENERAL)
    infos[1].imageView   = depthView;
    infos[1].imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    // b2 cloud radiance output (storage image, GENERAL)
    infos[2].imageView   = cloudRadianceView;
    infos[2].imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    // b3 cloudColor (this frame, storage image, GENERAL)
    infos[3].imageView   = cloudColorImages_[frameIndex].defaultView();
    infos[3].imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    // b4 cloudHistory (storage image, GENERAL — temporal reads last frame's color)
    infos[4].imageView   = cloudHistoryImage_.defaultView();
    infos[4].imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    // b5 weatherMap (storage image, GENERAL — written by weather pass)
    infos[5].imageView   = weatherMapImage_.defaultView();
    infos[5].imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    // b6 noiseTexture3D (combined image sampler, GENERAL — sampled by raymarch)
    infos[6].sampler     = noiseSampler_;
    infos[6].imageView   = noiseTexture3D_.defaultView();
    infos[6].imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    // b7 cloudShadowMap (storage image, GENERAL)
    infos[7].imageView   = cloudShadowImage_.defaultView();
    infos[7].imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    // b8 weatherMap (combined image sampler, GENERAL — sampled by raymarch)
    infos[8].sampler     = weatherSampler_;
    infos[8].imageView   = weatherMapImage_.defaultView();
    infos[8].imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    // b9 noiseTexture3D (storage image, GENERAL — written by noise_gen)
    infos[9].imageView   = noiseTexture3D_.defaultView();
    infos[9].imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkWriteDescriptorSet writes[10]{};
    VkDescriptorType types[10] = {
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,         // b0
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,         // b1
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,         // b2
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,         // b3
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,         // b4
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,         // b5
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,// b6
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,         // b7
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,// b8
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,         // b9
    };

    for (uint32_t i = 0; i < 10; ++i) {
        writes[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet          = set;
        writes[i].dstBinding      = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType  = types[i];
        writes[i].pImageInfo      = &infos[i];
    }

    vkUpdateDescriptorSets(device_, 10, writes, 0, nullptr);
    return set;
}

VkDescriptorSet CloudAdapter::allocAndWriteDescSet1(uint32_t frameIndex) {
    auto setR = descAllocator1_.allocate(descLayout1_, frameIndex);
    if (!setR) return VK_NULL_HANDLE;
    VkDescriptorSet set = setR.value();

    auto& sceneRes = services_->sceneRes();

    VkDescriptorBufferInfo bufInfos[3]{};
    bufInfos[0].buffer = sceneRes.worldUBO(frameIndex);
    bufInfos[0].offset = 0;
    bufInfos[0].range  = sceneRes.worldUBOSize();

    bufInfos[1].buffer = sceneRes.skyUBO(frameIndex);
    bufInfos[1].offset = 0;
    bufInfos[1].range  = sceneRes.skyUBOSize();

    // LastWorldUBO (same buffer as worldUBO if separate one isn't available;
    // SceneResourceService exposes lastWorldUBO())
    bufInfos[2].buffer = sceneRes.lastWorldUBO(frameIndex);
    bufInfos[2].offset = 0;
    bufInfos[2].range  = sceneRes.worldUBOSize();

    VkWriteDescriptorSet writes[3]{};
    for (uint32_t i = 0; i < 3; ++i) {
        writes[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet          = set;
        writes[i].dstBinding      = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[i].pBufferInfo     = &bufInfos[i];
    }

    vkUpdateDescriptorSets(device_, 3, writes, 0, nullptr);
    return set;
}

// ============================================================================
// execute
// ============================================================================

void CloudAdapter::execute(const FrameContext& ctx, const PassResources& resources) {
    if (!initialized_ || !resources.resolved || resources.cmd == VK_NULL_HANDLE) return;
    if (resources.inputs.size() < 2) return;
    if (resources.outputs.empty()) return;

    // Update config from the per-frame snapshot
    if (ctx.config) configure(*ctx.config);

    VkCommandBuffer cmd        = resources.cmd;
    uint32_t        frameIndex = ctx.frameIndex;

    if (profileSlot_ != UINT32_MAX)
        services_->metrics().beginNamedTimer(cmd, frameIndex, profileSlot_);

    uint32_t inRadianceIdx  = resources.inputs[0].index;
    uint32_t inDepthIdx     = resources.inputs[1].index;
    uint32_t outRadianceIdx = resources.outputs[0].index;

    auto& pool = services_->frame().resourcePool();
    uint32_t renderW = pool.image(inRadianceIdx).width();
    uint32_t renderH = pool.image(inRadianceIdx).height();
    if (renderW == 0 || renderH == 0) return;

    VkImage radianceVkImage  = resources.resolved->images[inRadianceIdx];
    VkImage depthVkImage     = resources.resolved->images[inDepthIdx];
    VkImage outRadianceVkImage = resources.resolved->images[outRadianceIdx];
    VkImageView radianceView = resources.resolved->views[inRadianceIdx];
    VkImageView depthView    = resources.resolved->views[inDepthIdx];
    VkImageView outRadianceView = resources.resolved->views[outRadianceIdx];

    // When cloudQuality == 0, pass the radiance through unchanged.
    // Copy radiance input → output with an image-to-image barrier and blit.
    if (cloudQuality_ == 0) {
        // Transition both images to suitable transfer layouts
        imageBarrier(cmd, radianceVkImage,
                     VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                     VK_ACCESS_2_SHADER_WRITE_BIT,
                     VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);

        imageBarrier(cmd, outRadianceVkImage,
                     VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                     VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                     VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);

        VkImageCopy copyRegion{};
        copyRegion.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        copyRegion.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        copyRegion.extent         = {renderW, renderH, 1};
        vkCmdCopyImage(cmd,
                       radianceVkImage,    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       outRadianceVkImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                       1, &copyRegion);

        // Restore both images to GENERAL for downstream passes
        imageBarrier(cmd, radianceVkImage,
                     VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                     VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);

        imageBarrier(cmd, outRadianceVkImage,
                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                     VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);
        if (profileSlot_ != UINT32_MAX)
            services_->metrics().endNamedTimer(cmd, frameIndex, profileSlot_);
        return;
    }

    // -------------------------------------------------------------------------
    // Ensure resolution-dependent internal images exist at the current size
    // -------------------------------------------------------------------------

    createInternalImages(renderW, renderH);
    if (!imagesReady_) return;

    // -------------------------------------------------------------------------
    // Advance wind time (FP32 precision preserved by wrapping at 86400 s)
    // -------------------------------------------------------------------------

    auto now = std::chrono::steady_clock::now();
    float deltaSeconds = 1.0f / 60.0f;
    if (lastFrameTime_.time_since_epoch().count() > 0) {
        deltaSeconds = std::chrono::duration<float>(now - lastFrameTime_).count();
        deltaSeconds = std::clamp(deltaSeconds, 0.001f, 0.1f);
    }
    lastFrameTime_ = now;
    windTime_ += windSpeed_ * deltaSeconds;
    windTime_  = std::fmod(windTime_, 86400.0f);

    // -------------------------------------------------------------------------
    // Build push constant
    // -------------------------------------------------------------------------

    CloudPushConstant pc{};
    pc.renderWidth       = renderW;
    pc.renderHeight      = renderH;
    pc.cloudWidth        = cloudW_;
    pc.cloudHeight       = cloudH_;
    pc.cloudBase         = cloudBase_;
    pc.cloudThickness    = cloudThickness_;
    pc.coverage          = coverage_;
    pc.cloudType         = cloudType_;
    pc.densityMultiplier = densityMultiplier_;
    pc.windTime          = windTime_;
    pc.frameIndex        = frameCounter_++;
    pc.marchSteps        = marchSteps_;
    pc.lightSteps        = lightSteps_;
    pc.temporalBlend     = temporalBlend_;
    pc.shadowMapSize     = shadowMapSize_;
    // Camera position from scene resource service (via WorldUBO camera pos)
    // Use the FrameContext camera if available
    pc.eyePosX           = ctx.camera.posX;
    pc.eyePosY           = ctx.camera.posY;
    pc.eyePosZ           = ctx.camera.posZ;
    pc.detailStrength    = detailStrength_;
    pc.scatterOctaves    = scatterOctaves_;
    pc.ambientStrength   = ambientStrength_;
    pc.noiseScale        = noiseScale_;
    pc.cellFrequency     = cellFrequency_;
    pc.atmosphereFadeDist = atmosphereFadeDist_;
    pc.windAngle         = windAngle_;
    pc.debugMode         = debugMode_;

    // -------------------------------------------------------------------------
    // Transition all images to GENERAL for compute shaders
    // -------------------------------------------------------------------------

    auto transitionToGeneral = [&](VkImage image, VkImageLayout& trackedLayout,
                                   VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess) {
        if (trackedLayout == VK_IMAGE_LAYOUT_GENERAL) return;
        VkPipelineStageFlags2 actualSrcStage = (trackedLayout == VK_IMAGE_LAYOUT_UNDEFINED)
            ? VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT : srcStage;
        VkAccessFlags2 actualSrcAccess = (trackedLayout == VK_IMAGE_LAYOUT_UNDEFINED)
            ? 0 : srcAccess;
        imageBarrier(cmd, image,
                     trackedLayout, VK_IMAGE_LAYOUT_GENERAL,
                     actualSrcStage, actualSrcAccess,
                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                     VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
        trackedLayout = VK_IMAGE_LAYOUT_GENERAL;
    };

    // Graph-managed images come in whatever layout the graph left them.
    // Ensure they are in GENERAL for compute.
    imageBarrier(cmd, radianceVkImage,
                 VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                 VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                 VK_ACCESS_2_SHADER_WRITE_BIT,
                 VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT);

    imageBarrier(cmd, depthVkImage,
                 VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                 VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                 VK_ACCESS_2_SHADER_WRITE_BIT,
                 VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT);

    imageBarrier(cmd, outRadianceVkImage,
                 VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                 VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                 VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

    // Internal images
    transitionToGeneral(weatherMapImage_.handle(),   weatherLayout_,
                        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    transitionToGeneral(noiseTexture3D_.handle(),    noiseLayout_,
                        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    transitionToGeneral(cloudShadowImage_.handle(),  shadowLayout_,
                        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    transitionToGeneral(cloudHistoryImage_.handle(), historyLayout_,
                        VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);

    // Cloud color (per-frame — always starts as UNDEFINED at frame start)
    imageBarrier(cmd, cloudColorImages_[frameIndex].handle(),
                 VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                 VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                 VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

    // -------------------------------------------------------------------------
    // Allocate and write descriptor sets
    // -------------------------------------------------------------------------

    descAllocator0_.resetFrame(frameIndex);
    descAllocator1_.resetFrame(frameIndex);

    VkDescriptorSet set0 = allocAndWriteDescSet0(frameIndex, radianceView, depthView, outRadianceView);
    VkDescriptorSet set1 = allocAndWriteDescSet1(frameIndex);
    if (set0 == VK_NULL_HANDLE || set1 == VK_NULL_HANDLE) return;

    VkDescriptorSet sets[2] = {set0, set1};
    VkPipelineLayout layout = pipelineLayout_.handle();

    auto bindSets = [&]() {
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout,
                                0, 2, sets, 0, nullptr);
    };

    auto pushPC = [&]() {
        vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(CloudPushConstant), &pc);
    };

    // -------------------------------------------------------------------------
    // Pass 0: Noise generation (first frame only)
    // -------------------------------------------------------------------------

    if (!noiseGenerated_) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, noiseGenPipeline_.handle());
        bindSets();
        pushPC();
        vkCmdDispatch(cmd, NOISE_RES / 4, NOISE_RES / 4, NOISE_RES / 4);

        // Barrier: noise written → readable by raymarch sampler
        imageBarrier(cmd, noiseTexture3D_.handle(),
                     VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                     VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT);

        noiseGenerated_ = true;
    }

    // -------------------------------------------------------------------------
    // Pass 1: Weather map update (512 × 512)
    // -------------------------------------------------------------------------

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, weatherPipeline_.handle());
    bindSets();
    pushPC();
    vkCmdDispatch(cmd, 512 / 8, 512 / 8, 1);

    // Barrier: weather written → readable by raymarch (both storage and sampler)
    imageBarrier(cmd, weatherMapImage_.handle(),
                 VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                 VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                 VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                 VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT);

    // -------------------------------------------------------------------------
    // Pass 2: Cloud ray march (cloud resolution)
    // -------------------------------------------------------------------------

    uint32_t marchX = (cloudW_ + 7) / 8;
    uint32_t marchY = (cloudH_ + 7) / 8;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, raymarchPipeline_.handle());
    bindSets();
    pushPC();
    vkCmdDispatch(cmd, marchX, marchY, 1);

    // Barrier: cloudColor written by raymarch → read+written by temporal
    imageBarrier(cmd, cloudColorImages_[frameIndex].handle(),
                 VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                 VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                 VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                 VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

    // -------------------------------------------------------------------------
    // Pass 3: Temporal reprojection (cloud resolution)
    // -------------------------------------------------------------------------

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, temporalPipeline_.handle());
    bindSets();
    pushPC();
    vkCmdDispatch(cmd, marchX, marchY, 1);

    // Barrier: cloudColor (now temporal output) → copy source + composite input
    imageBarrier(cmd, cloudColorImages_[frameIndex].handle(),
                 VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                 VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                 VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                 VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_TRANSFER_READ_BIT);

    // -------------------------------------------------------------------------
    // Copy cloudColor → history (for next frame's temporal blend)
    // -------------------------------------------------------------------------

    VkImageCopy copyRegion{};
    copyRegion.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copyRegion.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copyRegion.extent         = {cloudW_, cloudH_, 1};
    vkCmdCopyImage(cmd,
                   cloudColorImages_[frameIndex].handle(), VK_IMAGE_LAYOUT_GENERAL,
                   cloudHistoryImage_.handle(),            VK_IMAGE_LAYOUT_GENERAL,
                   1, &copyRegion);

    // Barrier: history written by copy → readable by next frame's temporal shader
    imageBarrier(cmd, cloudHistoryImage_.handle(),
                 VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                 VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                 VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT);

    // -------------------------------------------------------------------------
    // Pass 4: Composite clouds onto full-res radiance
    // -------------------------------------------------------------------------

    uint32_t compX = (renderW + 7) / 8;
    uint32_t compY = (renderH + 7) / 8;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, compositePipeline_.handle());
    bindSets();
    pushPC();
    vkCmdDispatch(cmd, compX, compY, 1);

    // Barrier: cloudRadiance written → readable downstream (tone mapping)
    imageBarrier(cmd, outRadianceVkImage,
                 VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                 VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                 VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT);

    // -------------------------------------------------------------------------
    // Pass 5: Cloud shadow map (independent of composite — fog/RT use it)
    // -------------------------------------------------------------------------

    uint32_t shadowGroups = (shadowMapSize_ + 7) / 8;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, shadowPipeline_.handle());
    bindSets();
    pushPC();
    vkCmdDispatch(cmd, shadowGroups, shadowGroups, 1);

    // Barrier: shadow map written → readable by downstream (RT shaders, post-process)
    imageBarrier(cmd, cloudShadowImage_.handle(),
                 VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                 VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                 VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                 VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT);

    if (profileSlot_ != UINT32_MAX)
        services_->metrics().endNamedTimer(cmd, frameIndex, profileSlot_);
}

// ============================================================================
// shutdown
// ============================================================================

void CloudAdapter::shutdown() {
    if (device_ == VK_NULL_HANDLE) return;

    // Destroy resolution-dependent images first
    destroyInternalImages();

    // Destroy pipelines
    noiseGenPipeline_  = {};
    weatherPipeline_   = {};
    raymarchPipeline_  = {};
    temporalPipeline_  = {};
    compositePipeline_ = {};
    shadowPipeline_    = {};

    // Destroy pipeline layout
    pipelineLayout_ = {};

    // Destroy descriptor allocators
    descAllocator0_.shutdown();
    descAllocator1_.shutdown();

    // Destroy descriptor set layouts
    if (descLayout0_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device_, descLayout0_, nullptr);
        descLayout0_ = VK_NULL_HANDLE;
    }
    if (descLayout1_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device_, descLayout1_, nullptr);
        descLayout1_ = VK_NULL_HANDLE;
    }

    // Destroy shader modules
    noiseGenShader_   = {};
    weatherShader_    = {};
    raymarchShader_   = {};
    temporalShader_   = {};
    compositeShader_  = {};
    shadowShader_     = {};

    // Destroy static internal images
    cloudShadowImage_  = {};
    noiseTexture3D_    = {};
    weatherMapImage_   = {};

    // Destroy samplers
    if (noiseSampler_ != VK_NULL_HANDLE) {
        vkDestroySampler(device_, noiseSampler_, nullptr);
        noiseSampler_ = VK_NULL_HANDLE;
    }
    if (weatherSampler_ != VK_NULL_HANDLE) {
        vkDestroySampler(device_, weatherSampler_, nullptr);
        weatherSampler_ = VK_NULL_HANDLE;
    }

    initialized_    = false;
    noiseGenerated_ = false;

    log::info("cloud", "CloudAdapter shut down");
}

} // namespace engine
