#include "atmosphere_adapter.hpp"

#include "app/engine_services.hpp"
#include "diagnostics/metrics_service.hpp"
#include "frame/frame_scheduler.hpp"
#include "scene/scene_resource_service.hpp"
#include "platform/vulkan/vk2_device.hpp"
#include "diagnostics/log.hpp"

#include <volk.h>

namespace engine {

// ---------------------------------------------------------------------------
// Image barrier helper
// ---------------------------------------------------------------------------

static void imageBarrier(VkCommandBuffer cmd, VkImage image,
                         VkImageLayout oldLayout, VkImageLayout newLayout,
                         VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess,
                         VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess,
                         uint32_t layerCount = 1) {
    VkImageMemoryBarrier2 b{};
    b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    b.srcStageMask  = srcStage;
    b.srcAccessMask = srcAccess;
    b.dstStageMask  = dstStage;
    b.dstAccessMask = dstAccess;
    b.oldLayout = oldLayout;
    b.newLayout = newLayout;
    b.image     = image;
    b.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    b.subresourceRange.baseMipLevel   = 0;
    b.subresourceRange.levelCount     = 1;
    b.subresourceRange.baseArrayLayer = 0;
    b.subresourceRange.layerCount     = layerCount;

    VkDependencyInfo dep{};
    dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.imageMemoryBarrierCount = 1;
    dep.pImageMemoryBarriers = &b;
    vkCmdPipelineBarrier2(cmd, &dep);
}

// ---------------------------------------------------------------------------
// init
// ---------------------------------------------------------------------------

void AtmosphereAdapter::init(EngineServices& services) {
    services_ = &services;
    device_   = services.device().device();
    auto vma  = services.device().vma();

    // --- Load shaders ---
    std::string base = services.resourceDir() + "/shaders/world/v2_atmosphere/";

    auto lutR = vk2::ShaderModule::fromFile(device_, base + "v2_trans_lut_comp.spv");
    if (!lutR) {
        log::error("atmosphere", "Failed to load v2_trans_lut_comp.spv: " + lutR.error().message);
        return;
    }
    lutShader_ = std::move(lutR.value());

    auto cubeR = vk2::ShaderModule::fromFile(device_, base + "v2_skycube_comp.spv");
    if (!cubeR) {
        log::error("atmosphere", "Failed to load v2_skycube_comp.spv: " + cubeR.error().message);
        return;
    }
    cubeShader_ = std::move(cubeR.value());

    // --- Create images ---

    // Transmittance LUT: 512x128 RGBA16F
    {
        vk2::Image::Desc desc{};
        desc.width  = LUT_WIDTH;
        desc.height = LUT_HEIGHT;
        desc.format = VK_FORMAT_R16G16B16A16_SFLOAT;
        desc.usage  = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        auto r = vk2::Image::create(device_, vma, desc);
        if (!r) { log::error("atmosphere", "LUT image: " + r.error().message); return; }
        lutImage_ = std::move(r.value());
    }

    // Sky cubemap: 512x512x6 RGBA16F
    {
        vk2::Image::Desc desc{};
        desc.width  = CUBE_DIM;
        desc.height = CUBE_DIM;
        desc.layers = 6;
        desc.format = VK_FORMAT_R16G16B16A16_SFLOAT;
        desc.usage  = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        desc.createFlags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
        auto r = vk2::Image::create(device_, vma, desc);
        if (!r) { log::error("atmosphere", "Cube image: " + r.error().message); return; }
        cubeImage_ = std::move(r.value());

        // The default view (index 0) is VK_IMAGE_VIEW_TYPE_2D_ARRAY — wrong for
        // imageCube shader bindings. Add an explicit CUBE view and record its index.
        // The underlying VkImage has VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT so this is legal.
        VkImageViewCreateInfo cvi{};
        cvi.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        cvi.image    = cubeImage_.handle();
        cvi.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
        cvi.format   = VK_FORMAT_R16G16B16A16_SFLOAT;
        cvi.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        cvi.subresourceRange.baseMipLevel   = 0;
        cvi.subresourceRange.levelCount     = 1;
        cvi.subresourceRange.baseArrayLayer = 0;
        cvi.subresourceRange.layerCount     = 6;
        auto vr = cubeImage_.addView(device_, cvi);
        if (!vr) { log::error("atmosphere", "Cube view (CUBE type): " + vr.error().message); return; }
        cubeViewIdx_ = vr.value();
        log::info("atmosphere", "Cube image: added VK_IMAGE_VIEW_TYPE_CUBE view at index " + std::to_string(cubeViewIdx_));
    }

    // --- Sampler ---
    {
        VkSamplerCreateInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        si.magFilter    = VK_FILTER_LINEAR;
        si.minFilter    = VK_FILTER_LINEAR;
        si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        if (vkCreateSampler(device_, &si, nullptr, &linearSampler_) != VK_SUCCESS) {
            log::error("atmosphere", "Sampler creation failed"); return;
        }
    }

    uint32_t fif = services.frame().framesInFlight();

    // --- LUT descriptor layout + pipeline ---
    // set 0: binding 0 = storage image (output LUT), binding 1 = uniform buffer (SkyUBO)
    {
        VkDescriptorSetLayoutBinding bindings[2]{};
        bindings[0] = {0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        bindings[1] = {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};

        VkDescriptorSetLayoutCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        ci.bindingCount = 2;
        ci.pBindings = bindings;
        if (vkCreateDescriptorSetLayout(device_, &ci, nullptr, &lutDescLayout_) != VK_SUCCESS) {
            log::error("atmosphere", "LUT desc layout failed"); return;
        }

        std::vector<VkDescriptorPoolSize> sizes = {
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1},
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1},
        };
        auto r = lutDescAllocator_.init(device_, fif, sizes, 1);
        if (!r) { log::error("atmosphere", "LUT desc alloc: " + r.error().message); return; }

        vk2::PipelineLayoutDesc pld;
        pld.setLayouts = {lutDescLayout_};
        auto plr = vk2::PipelineLayout::create(device_, pld);
        if (!plr) { log::error("atmosphere", "LUT pipeline layout: " + plr.error().message); return; }
        lutPipelineLayout_ = std::move(plr.value());

        vk2::ComputePipelineDesc cpd;
        cpd.shaderModule = lutShader_.handle();
        cpd.layout = lutPipelineLayout_.handle();
        auto cpr = vk2::ComputePipeline::create(device_, cpd);
        if (!cpr) { log::error("atmosphere", "LUT pipeline: " + cpr.error().message); return; }
        lutPipeline_ = std::move(cpr.value());
    }

    // --- Cubemap descriptor layout + pipeline ---
    // set 0: binding 0 = storage image (output cube), binding 1 = combined image sampler (LUT),
    //        binding 2 = uniform buffer (SkyUBO)
    {
        VkDescriptorSetLayoutBinding bindings[3]{};
        bindings[0] = {0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        bindings[1] = {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        bindings[2] = {2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};

        VkDescriptorSetLayoutCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        ci.bindingCount = 3;
        ci.pBindings = bindings;
        if (vkCreateDescriptorSetLayout(device_, &ci, nullptr, &cubeDescLayout_) != VK_SUCCESS) {
            log::error("atmosphere", "Cube desc layout failed"); return;
        }

        std::vector<VkDescriptorPoolSize> sizes = {
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1},
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1},
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1},
        };
        auto r = cubeDescAllocator_.init(device_, fif, sizes, 1);
        if (!r) { log::error("atmosphere", "Cube desc alloc: " + r.error().message); return; }

        VkPushConstantRange pcRange{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(CubePushConstant)};
        vk2::PipelineLayoutDesc pld;
        pld.setLayouts = {cubeDescLayout_};
        pld.pushConstants = {pcRange};
        auto plr = vk2::PipelineLayout::create(device_, pld);
        if (!plr) { log::error("atmosphere", "Cube pipeline layout: " + plr.error().message); return; }
        cubePipelineLayout_ = std::move(plr.value());

        vk2::ComputePipelineDesc cpd;
        cpd.shaderModule = cubeShader_.handle();
        cpd.layout = cubePipelineLayout_.handle();
        auto cpr = vk2::ComputePipeline::create(device_, cpd);
        if (!cpr) { log::error("atmosphere", "Cube pipeline: " + cpr.error().message); return; }
        cubePipeline_ = std::move(cpr.value());
    }

    initialized_ = true;
    profileSlot_ = services.metrics().registerTimer("Atmosphere");
    log::info("atmosphere", "AtmosphereAdapter initialized (LUT " +
              std::to_string(LUT_WIDTH) + "x" + std::to_string(LUT_HEIGHT) +
              ", cube " + std::to_string(CUBE_DIM) + ")");
}

// ---------------------------------------------------------------------------
// recordCommands
// ---------------------------------------------------------------------------

void AtmosphereAdapter::recordCommands(VkCommandBuffer cmd, uint32_t frameIndex) {
    if (!initialized_) return;

    auto& sceneRes = services_->sceneRes();
    if (!sceneRes.isInitialized()) return;

    if (profileSlot_ != UINT32_MAX)
        services_->metrics().beginNamedTimer(cmd, frameIndex, profileSlot_);

    // -----------------------------------------------------------------------
    // LUT pass (first frame only)
    // -----------------------------------------------------------------------
    if (!lutGenerated_) {
        // Transition LUT to GENERAL for compute write
        imageBarrier(cmd, lutImage_.handle(),
                     VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                     VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        lutImage_.setLayout(VK_IMAGE_LAYOUT_GENERAL);

        // Allocate and write LUT descriptor set
        lutDescAllocator_.resetFrame(frameIndex);
        auto setR = lutDescAllocator_.allocate(lutDescLayout_, frameIndex);
        if (!setR) return;
        VkDescriptorSet lutSet = setR.value();

        VkDescriptorImageInfo outInfo{};
        outInfo.imageView   = lutImage_.defaultView();
        outInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorBufferInfo skyInfo{};
        skyInfo.buffer = sceneRes.skyUBO(frameIndex);
        skyInfo.offset = 0;
        skyInfo.range  = sceneRes.skyUBOSize();

        VkWriteDescriptorSet writes[2]{};
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = lutSet;
        writes[0].dstBinding = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[0].pImageInfo = &outInfo;

        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet = lutSet;
        writes[1].dstBinding = 1;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[1].pBufferInfo = &skyInfo;

        vkUpdateDescriptorSets(device_, 2, writes, 0, nullptr);

        // Dispatch LUT compute
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, lutPipeline_.handle());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, lutPipelineLayout_.handle(),
                                0, 1, &lutSet, 0, nullptr);
        vkCmdDispatch(cmd, (LUT_WIDTH + 15) / 16, (LUT_HEIGHT + 15) / 16, 1);

        // Transition LUT to SHADER_READ_ONLY for cubemap sampling
        imageBarrier(cmd, lutImage_.handle(),
                     VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        lutImage_.setLayout(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

        lutGenerated_ = true;
    }

    // -----------------------------------------------------------------------
    // Cubemap pass (every frame)
    // -----------------------------------------------------------------------

    // Transition cubemap to GENERAL for compute write
    imageBarrier(cmd, cubeImage_.handle(),
                 cubeImage_.layout(), VK_IMAGE_LAYOUT_GENERAL,
                 VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                 VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                 6);
    cubeImage_.setLayout(VK_IMAGE_LAYOUT_GENERAL);

    // Allocate cubemap descriptor set
    cubeDescAllocator_.resetFrame(frameIndex);
    auto cubeSetR = cubeDescAllocator_.allocate(cubeDescLayout_, frameIndex);
    if (!cubeSetR) return;
    VkDescriptorSet cubeSet = cubeSetR.value();

    VkDescriptorImageInfo cubeOutInfo{};
    cubeOutInfo.imageView   = cubeView();  // VK_IMAGE_VIEW_TYPE_CUBE — required for imageCube storage binding
    cubeOutInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkDescriptorImageInfo lutSamplerInfo{};
    lutSamplerInfo.sampler     = linearSampler_;
    lutSamplerInfo.imageView   = lutImage_.defaultView();
    lutSamplerInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDescriptorBufferInfo skyInfo{};
    skyInfo.buffer = sceneRes.skyUBO(frameIndex);
    skyInfo.offset = 0;
    skyInfo.range  = sceneRes.skyUBOSize();

    VkWriteDescriptorSet writes[3]{};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = cubeSet;
    writes[0].dstBinding = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[0].pImageInfo = &cubeOutInfo;

    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = cubeSet;
    writes[1].dstBinding = 1;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[1].pImageInfo = &lutSamplerInfo;

    writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[2].dstSet = cubeSet;
    writes[2].dstBinding = 2;
    writes[2].descriptorCount = 1;
    writes[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writes[2].pBufferInfo = &skyInfo;

    vkUpdateDescriptorSets(device_, 3, writes, 0, nullptr);

    // Bind cubemap pipeline and dispatch once per face
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cubePipeline_.handle());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cubePipelineLayout_.handle(),
                            0, 1, &cubeSet, 0, nullptr);

    uint32_t groups = (CUBE_DIM + 15) / 16;
    for (int face = 0; face < 6; ++face) {
        CubePushConstant pc{face};
        vkCmdPushConstants(cmd, cubePipelineLayout_.handle(), VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, groups, groups, 1);
    }

    // Transition cubemap to SHADER_READ_ONLY for RT sampling
    imageBarrier(cmd, cubeImage_.handle(),
                 VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                 VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                 VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                 6);
    cubeImage_.setLayout(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    if (profileSlot_ != UINT32_MAX)
        services_->metrics().endNamedTimer(cmd, frameIndex, profileSlot_);
}

// ---------------------------------------------------------------------------
// shutdown
// ---------------------------------------------------------------------------

void AtmosphereAdapter::shutdown() {
    if (device_ == VK_NULL_HANDLE) return;

    cubePipeline_       = {};
    cubePipelineLayout_  = {};
    cubeShader_         = {};
    cubeDescAllocator_.shutdown();
    if (cubeDescLayout_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device_, cubeDescLayout_, nullptr);
        cubeDescLayout_ = VK_NULL_HANDLE;
    }

    lutPipeline_        = {};
    lutPipelineLayout_   = {};
    lutShader_          = {};
    lutDescAllocator_.shutdown();
    if (lutDescLayout_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device_, lutDescLayout_, nullptr);
        lutDescLayout_ = VK_NULL_HANDLE;
    }

    cubeImage_ = {};
    lutImage_  = {};

    if (linearSampler_ != VK_NULL_HANDLE) {
        vkDestroySampler(device_, linearSampler_, nullptr);
        linearSampler_ = VK_NULL_HANDLE;
    }

    initialized_  = false;
    lutGenerated_ = false;
    log::info("atmosphere", "AtmosphereAdapter shut down");
}

} // namespace engine
