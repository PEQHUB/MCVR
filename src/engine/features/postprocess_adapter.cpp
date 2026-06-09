#include "postprocess_adapter.hpp"
#include "app/engine_services.hpp"
#include "frame/frame_scheduler.hpp"
#include "platform/vulkan/vk2_device.hpp"
#include "rendergraph/graph_builder.hpp"
#include "diagnostics/log.hpp"

#include <volk.h>
#include <algorithm>
#include <cmath>
#include <cstring>

namespace engine {

namespace {

// Pack two 16-bit floats into a uint32 (lower 16 = a, upper 16 = b).
// Adapted from FFX SDK ffxPackHalf2x16.
uint32_t packHalf2x16(float a, float b) {
    auto floatToHalf = [](float f) -> uint16_t {
        uint32_t bits;
        std::memcpy(&bits, &f, sizeof(bits));
        uint32_t sign = (bits >> 31) & 0x1;
        int32_t exp = static_cast<int32_t>((bits >> 23) & 0xFF) - 127;
        uint32_t mantissa = bits & 0x7FFFFF;

        if (exp >= 16) {
            // Inf or NaN
            return static_cast<uint16_t>((sign << 15) | 0x7C00 | (mantissa ? 1 : 0));
        }
        if (exp >= -14) {
            // Normalized
            uint32_t hexp = static_cast<uint32_t>(exp + 15);
            return static_cast<uint16_t>((sign << 15) | (hexp << 10) | (mantissa >> 13));
        }
        if (exp >= -24) {
            // Subnormal
            uint32_t m = (mantissa | 0x800000) >> (-exp - 1);
            return static_cast<uint16_t>((sign << 15) | (m >> 13));
        }
        // Underflow → ±0
        return static_cast<uint16_t>(sign << 15);
    };
    uint16_t ha = floatToHalf(a);
    uint16_t hb = floatToHalf(b);
    return static_cast<uint32_t>(ha) | (static_cast<uint32_t>(hb) << 16);
}

// Mirrors FFX's ffxCasSetup. Sharpening only (input == output size).
// See extern/FidelityFX-SDK/sdk/include/FidelityFX/gpu/cas/ffx_cas.h:66.
CasPushConstants buildCasConstants(float sharpness, float inW, float inH, float outW, float outH) {
    CasPushConstants pc{};
    auto asUint = [](float f) {
        uint32_t u;
        std::memcpy(&u, &f, sizeof(u));
        return u;
    };
    auto lerpf = [](float a, float b, float t) { return a + (b - a) * t; };
    auto sat = [](float v) { return std::max(0.0f, std::min(1.0f, v)); };

    pc.const0[0] = asUint(inW / outW);
    pc.const0[1] = asUint(inH / outH);
    pc.const0[2] = asUint(0.5f * inW / outW - 0.5f);
    pc.const0[3] = asUint(0.5f * inH / outH - 0.5f);

    float sharp = -1.0f / lerpf(8.0f, 5.0f, sat(sharpness));
    pc.const1[0] = asUint(sharp);
    pc.const1[1] = packHalf2x16(sharp, 0.0f);
    pc.const1[2] = asUint(8.0f * inW / outW);
    pc.const1[3] = 0;

    return pc;
}

} // namespace

void PostProcessAdapter::registerPass(GraphBuilder& builder, ResourceHandle inputHandle) {
    inputHandle_ = inputHandle;

    // CAS shader uses `rgba16` storage qualifier (R16G16B16A16_UNORM).
    // The composite blit will downconvert to swapchain BGRA8 with format conversion.
    ImageResourceDesc outDesc;
    outDesc.name = "post_sharpened";
    outDesc.format = VK_FORMAT_R16G16B16A16_UNORM;
    outputHandle_ = builder.addResource(outDesc);

    PassDescriptor pass;
    pass.name = "post_process";
    pass.queue = QueueAffinity::Graphics;
    pass.inputs.push_back(inputHandle);
    pass.outputs.push_back(outputHandle_);
    pass.execute = [this](const FrameContext& ctx, const PassResources& res) {
        execute(ctx, res);
    };
    builder.addPass(std::move(pass));

    // Take over as the final output — composite blit will pick this up.
    builder.setFinalOutput(outputHandle_);
}

void PostProcessAdapter::init(EngineServices& services) {
    services_ = &services;
    device_ = services.device().device();

    // Load CAS compute shader (V1's existing cas.comp, compiled by the shared shader build)
    std::string base = services.resourceDir() + "/shaders/world/post_render/";
    auto casR = vk2::ShaderModule::fromFile(device_, base + "cas_comp.spv");
    if (!casR) {
        log::error("postprocess", "Failed to load cas_comp.spv: " + casR.error().message);
        return;
    }
    casShader_ = std::move(casR.value());

    // Linear sampler for input
    {
        VkSamplerCreateInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        si.magFilter = VK_FILTER_LINEAR;
        si.minFilter = VK_FILTER_LINEAR;
        si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        if (vkCreateSampler(device_, &si, nullptr, &linearSampler_) != VK_SUCCESS) {
            log::error("postprocess", "Sampler creation failed"); return;
        }
    }

    // Descriptor set layout: matches cas.comp
    //   binding 0: combined image sampler (input)
    //   binding 1: storage image rgba16 (output)
    {
        VkDescriptorSetLayoutBinding bindings[2]{};
        bindings[0] = {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                       VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        bindings[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1,
                       VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        ci.bindingCount = 2;
        ci.pBindings = bindings;
        if (vkCreateDescriptorSetLayout(device_, &ci, nullptr, &descSetLayout_) != VK_SUCCESS) {
            log::error("postprocess", "Descriptor set layout failed"); return;
        }
    }

    // Per-frame descriptor allocator
    {
        std::vector<VkDescriptorPoolSize> sizes = {
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1},
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1},
        };
        uint32_t fif = services.frame().framesInFlight();
        auto r = descAllocator_.init(device_, fif, sizes, 1);
        if (!r) { log::error("postprocess", "Descriptor allocator: " + r.error().message); return; }
    }

    // Pipeline layout: 1 set + 32-byte push constant (2 uvec4)
    {
        VkPushConstantRange pcRange{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(CasPushConstants)};
        vk2::PipelineLayoutDesc pld;
        pld.setLayouts = {descSetLayout_};
        pld.pushConstants = {pcRange};
        auto r = vk2::PipelineLayout::create(device_, pld);
        if (!r) { log::error("postprocess", "Pipeline layout: " + r.error().message); return; }
        pipelineLayout_ = std::move(r.value());
    }

    // Compute pipeline
    {
        vk2::ComputePipelineDesc d;
        d.shaderModule = casShader_.handle();
        d.layout = pipelineLayout_.handle();
        auto r = vk2::ComputePipeline::create(device_, d);
        if (!r) { log::error("postprocess", "CAS pipeline: " + r.error().message); return; }
        casPipeline_ = std::move(r.value());
    }

    initialized_ = true;
    log::info("postprocess", "PostProcessAdapter (CAS) initialized");
}

void PostProcessAdapter::configure(const ConfigSnapshot& config) {
    sharpness_ = config.data.casSharpness;
    if (sharpness_ < 0.0f) sharpness_ = 0.0f;
    if (sharpness_ > 1.0f) sharpness_ = 1.0f;
}

void PostProcessAdapter::execute(const FrameContext& ctx, const PassResources& resources) {
    if (!initialized_ || !resources.resolved || resources.cmd == VK_NULL_HANDLE) return;
    if (resources.inputs.empty() || !resources.inputs[0].valid()) return;
    if (resources.outputs.empty() || !resources.outputs[0].valid()) return;

    if (ctx.config) configure(*ctx.config);

    VkCommandBuffer cmd = resources.cmd;
    uint32_t inIdx = resources.inputs[0].index;
    uint32_t outIdx = resources.outputs[0].index;

    auto& pool = services_->frame().resourcePool();
    uint32_t inW = pool.image(inIdx).width();
    uint32_t inH = pool.image(inIdx).height();
    uint32_t outW = pool.image(outIdx).width();
    uint32_t outH = pool.image(outIdx).height();
    if (outW == 0 || outH == 0) return;

    // Allocate per-frame descriptor set
    descAllocator_.resetFrame(ctx.frameIndex);
    auto setResult = descAllocator_.allocate(descSetLayout_, ctx.frameIndex);
    if (!setResult) return;
    VkDescriptorSet descSet = setResult.value();

    // Write bindings
    VkDescriptorImageInfo inInfo{};
    inInfo.sampler = linearSampler_;
    inInfo.imageView = resources.resolved->views[inIdx];
    inInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDescriptorImageInfo outInfo{};
    outInfo.imageView = resources.resolved->views[outIdx];
    outInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkWriteDescriptorSet writes[2]{};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = descSet;
    writes[0].dstBinding = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[0].pImageInfo = &inInfo;

    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = descSet;
    writes[1].dstBinding = 1;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[1].pImageInfo = &outInfo;

    vkUpdateDescriptorSets(device_, 2, writes, 0, nullptr);

    // Build CAS push constants for the current frame size + sharpness
    CasPushConstants pc = buildCasConstants(sharpness_,
        static_cast<float>(inW), static_cast<float>(inH),
        static_cast<float>(outW), static_cast<float>(outH));

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, casPipeline_.handle());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout_.handle(),
                            0, 1, &descSet, 0, nullptr);
    vkCmdPushConstants(cmd, pipelineLayout_.handle(), VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(pc), &pc);

    // FFX CAS dispatches one workgroup per 16x16 pixel tile.
    // 64-thread workgroup × 4 pixels per thread = 256 pixels = 16×16 tile.
    uint32_t groupsX = (outW + 15) / 16;
    uint32_t groupsY = (outH + 15) / 16;
    vkCmdDispatch(cmd, groupsX, groupsY, 1);
}

void PostProcessAdapter::shutdown() {
    if (device_ == VK_NULL_HANDLE) return;

    casPipeline_ = {};
    pipelineLayout_ = {};
    casShader_ = {};

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
    log::info("postprocess", "PostProcessAdapter shut down");
}

} // namespace engine
