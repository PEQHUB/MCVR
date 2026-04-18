#include "temporal_denoiser_adapter.hpp"

#include "app/engine_services.hpp"
#include "diagnostics/log.hpp"
#include "diagnostics/metrics_service.hpp"
#include "frame/frame_scheduler.hpp"
#include "platform/vulkan/vk2_device.hpp"
#include "rendergraph/graph_builder.hpp"
#include "rendergraph/graph_resource_pool.hpp"

#include <volk.h>
#include <array>
#include <cstring>

namespace engine {

void TemporalDenoiserAdapter::registerPass(GraphBuilder& builder,
                                           ResourceHandle radianceIn,
                                           ResourceHandle normalIn,
                                           ResourceHandle depthIn,
                                           ResourceHandle motionIn) {
    radianceInput_ = radianceIn;
    normalInput_ = normalIn;
    depthInput_ = depthIn;
    motionInput_ = motionIn;

    // Persistent history image. Same format as radiance so it can carry the
    // denoised result across frames. Declared as an output so the BarrierPlanner
    // routes it to GENERAL layout (matches imageLoad/imageStore in the shader).
    ImageResourceDesc historyDesc;
    historyDesc.name = "denoise_history";
    historyDesc.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    historyHandle_ = builder.addResource(historyDesc);

    // Denoised output that flows to ToneMapping.
    ImageResourceDesc outDesc;
    outDesc.name = "denoise_output";
    outDesc.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    outputHandle_ = builder.addResource(outDesc);

    PassDescriptor pass;
    pass.name = "temporal_denoiser";
    pass.queue = QueueAffinity::Graphics;
    pass.inputs.push_back(radianceIn);
    pass.inputs.push_back(normalIn);
    pass.inputs.push_back(depthIn);
    pass.inputs.push_back(motionIn);
    pass.outputs.push_back(historyHandle_);
    pass.outputs.push_back(outputHandle_);
    pass.execute = [this](const FrameContext& ctx, const PassResources& res) {
        execute(ctx, res);
    };
    builder.addPass(std::move(pass));
}

void TemporalDenoiserAdapter::init(EngineServices& services) {
    services_ = &services;
    device_ = services.device().device();

    // Load compute shader
    std::string path = services.resourceDir() + "/shaders/world/v2_denoise/temporal_denoise_comp.spv";
    auto shaderR = vk2::ShaderModule::fromFile(device_, path);
    if (!shaderR) {
        log::error("denoise", "Failed to load temporal_denoise_comp.spv: " + shaderR.error().message);
        return;
    }
    shader_ = std::move(shaderR.value());

    // Linear sampler for sampled inputs
    {
        VkSamplerCreateInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        si.magFilter = VK_FILTER_LINEAR;
        si.minFilter = VK_FILTER_LINEAR;
        si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        if (vkCreateSampler(device_, &si, nullptr, &linearSampler_) != VK_SUCCESS) {
            log::error("denoise", "Sampler creation failed");
            return;
        }
    }

    // Descriptor set layout: matches temporal_denoise.comp
    //   binding 0..3: combined image samplers (radiance, normal, depth, motion)
    //   binding 4: storage image (history, read+write)
    //   binding 5: storage image (denoised output, write)
    {
        std::array<VkDescriptorSetLayoutBinding, 6> bindings{};
        for (uint32_t i = 0; i < 4; ++i) {
            bindings[i] = {i, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                           VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        }
        bindings[4] = {4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1,
                       VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        bindings[5] = {5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1,
                       VK_SHADER_STAGE_COMPUTE_BIT, nullptr};

        VkDescriptorSetLayoutCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        ci.bindingCount = static_cast<uint32_t>(bindings.size());
        ci.pBindings = bindings.data();
        if (vkCreateDescriptorSetLayout(device_, &ci, nullptr, &descSetLayout_) != VK_SUCCESS) {
            log::error("denoise", "Descriptor set layout failed");
            return;
        }
    }

    // Per-frame descriptor allocator
    {
        std::vector<VkDescriptorPoolSize> sizes = {
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4},
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2},
        };
        uint32_t fif = services.frame().framesInFlight();
        auto r = descAllocator_.init(device_, fif, sizes, 1);
        if (!r) {
            log::error("denoise", "Descriptor allocator: " + r.error().message);
            return;
        }
    }

    // Pipeline layout: 1 set + push constant
    {
        VkPushConstantRange pcRange{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(TemporalDenoisePush)};
        vk2::PipelineLayoutDesc pld;
        pld.setLayouts = {descSetLayout_};
        pld.pushConstants = {pcRange};
        auto r = vk2::PipelineLayout::create(device_, pld);
        if (!r) {
            log::error("denoise", "Pipeline layout: " + r.error().message);
            return;
        }
        pipelineLayout_ = std::move(r.value());
    }

    // Compute pipeline
    {
        vk2::ComputePipelineDesc d;
        d.shaderModule = shader_.handle();
        d.layout = pipelineLayout_.handle();
        auto r = vk2::ComputePipeline::create(device_, d);
        if (!r) {
            log::error("denoise", "Pipeline: " + r.error().message);
            return;
        }
        pipeline_ = std::move(r.value());
    }

    initialized_ = true;
    profileSlot_ = services.metrics().registerTimer("Denoise");
    log::info("denoise", "TemporalDenoiserAdapter initialized");
}

void TemporalDenoiserAdapter::configure(const ConfigSnapshot& config) {
    // Reuse SVGF alpha config knob if present; otherwise sane default.
    // Future: add a dedicated v2.temporalDenoise.alpha option.
    (void)config;
    // alpha_ stays at default 0.1 for now
}

void TemporalDenoiserAdapter::execute(const FrameContext& ctx, const PassResources& resources) {
    if (!initialized_ || !resources.resolved || resources.cmd == VK_NULL_HANDLE) return;
    if (resources.inputs.size() < 4) return;
    if (resources.outputs.size() < 2) return;
    if (!resources.inputs[0].valid() || !resources.inputs[1].valid() ||
        !resources.inputs[2].valid() || !resources.inputs[3].valid()) return;
    if (!resources.outputs[0].valid() || !resources.outputs[1].valid()) return;

    if (ctx.config) configure(*ctx.config);

    VkCommandBuffer cmd = resources.cmd;

    if (profileSlot_ != UINT32_MAX)
        services_->metrics().beginNamedTimer(cmd, ctx.frameIndex, profileSlot_);

    auto& pool = services_->frame().resourcePool();
    uint32_t outIdx = resources.outputs[1].index;
    uint32_t outW = pool.image(outIdx).width();
    uint32_t outH = pool.image(outIdx).height();
    if (outW == 0 || outH == 0) return;

    // Allocate descriptor set for this frame
    descAllocator_.resetFrame(ctx.frameIndex);
    auto setResult = descAllocator_.allocate(descSetLayout_, ctx.frameIndex);
    if (!setResult) return;
    VkDescriptorSet descSet = setResult.value();

    // Sampled inputs (radiance, normal, depth, motion)
    std::array<VkDescriptorImageInfo, 4> sampledInfos{};
    for (uint32_t i = 0; i < 4; ++i) {
        uint32_t idx = resources.inputs[i].index;
        sampledInfos[i].sampler = linearSampler_;
        sampledInfos[i].imageView = resources.resolved->views[idx];
        sampledInfos[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }

    // Storage outputs (history, denoised)
    VkDescriptorImageInfo historyInfo{};
    historyInfo.imageView = resources.resolved->views[resources.outputs[0].index];
    historyInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkDescriptorImageInfo outInfo{};
    outInfo.imageView = resources.resolved->views[resources.outputs[1].index];
    outInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    std::array<VkWriteDescriptorSet, 6> writes{};
    for (uint32_t i = 0; i < 4; ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = descSet;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[i].pImageInfo = &sampledInfos[i];
    }
    writes[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[4].dstSet = descSet;
    writes[4].dstBinding = 4;
    writes[4].descriptorCount = 1;
    writes[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[4].pImageInfo = &historyInfo;

    writes[5].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[5].dstSet = descSet;
    writes[5].dstBinding = 5;
    writes[5].descriptorCount = 1;
    writes[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[5].pImageInfo = &outInfo;

    vkUpdateDescriptorSets(device_, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

    // Push constants
    TemporalDenoisePush pc{};
    pc.alpha = alpha_;
    pc.normalThreshold = normalThreshold_;
    pc.depthThreshold = depthThreshold_;
    pc.frameNumber = frameNumber_;
    pc.imageWidth = static_cast<int32_t>(outW);
    pc.imageHeight = static_cast<int32_t>(outH);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_.handle());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout_.handle(),
                            0, 1, &descSet, 0, nullptr);
    vkCmdPushConstants(cmd, pipelineLayout_.handle(), VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(pc), &pc);

    // 16x16 workgroup, one thread per pixel
    uint32_t groupsX = (outW + 15) / 16;
    uint32_t groupsY = (outH + 15) / 16;
    vkCmdDispatch(cmd, groupsX, groupsY, 1);

    if (profileSlot_ != UINT32_MAX)
        services_->metrics().endNamedTimer(cmd, ctx.frameIndex, profileSlot_);

    ++frameNumber_;
}

void TemporalDenoiserAdapter::shutdown() {
    if (device_ == VK_NULL_HANDLE) return;

    pipeline_ = {};
    pipelineLayout_ = {};
    shader_ = {};

    descAllocator_.shutdown();
    if (descSetLayout_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device_, descSetLayout_, nullptr);
        descSetLayout_ = VK_NULL_HANDLE;
    }
    if (linearSampler_ != VK_NULL_HANDLE) {
        vkDestroySampler(device_, linearSampler_, nullptr);
        linearSampler_ = VK_NULL_HANDLE;
    }

    initialized_ = false;
    frameNumber_ = 0;
    log::info("denoise", "TemporalDenoiserAdapter shut down");
}

} // namespace engine
