#include "postprocess_adapter.hpp"
#include "app/engine_services.hpp"
#include "diagnostics/metrics_service.hpp"
#include "frame/frame_scheduler.hpp"
#include "platform/vulkan/vk2_device.hpp"
#include "rendergraph/graph_builder.hpp"
#include "diagnostics/log.hpp"
#include "config/engine_config.hpp"

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

// Helper: insert a full pipeline barrier for a compute-to-compute image transition.
void imageBarrier(VkCommandBuffer cmd, VkImage image,
                  VkImageLayout oldLayout, VkImageLayout newLayout,
                  VkAccessFlags2 srcAccess, VkPipelineStageFlags2 srcStage,
                  VkAccessFlags2 dstAccess, VkPipelineStageFlags2 dstStage) {
    VkImageMemoryBarrier2 barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    barrier.srcStageMask = srcStage;
    barrier.srcAccessMask = srcAccess;
    barrier.dstStageMask = dstStage;
    barrier.dstAccessMask = dstAccess;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.image = image;
    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    VkDependencyInfo dep{};
    dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.imageMemoryBarrierCount = 1;
    dep.pImageMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(cmd, &dep);
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

    std::string shaderBase = services.resourceDir() + "/shaders/";

    // --- Load CAS shader ---
    {
        auto casR = vk2::ShaderModule::fromFile(device_, shaderBase + "world/post_render/cas_comp.spv");
        if (!casR) {
            log::error("postprocess", "Failed to load cas_comp.spv: " + casR.error().message);
            return;
        }
        casShader_ = std::move(casR.value());
    }

    // --- Load bloom shaders ---
    {
        auto r = vk2::ShaderModule::fromFile(device_, shaderBase + "world/v2_post/v2_bloom_downsample_comp.spv");
        if (!r) {
            log::error("postprocess", "Failed to load v2_bloom_downsample_comp.spv: " + r.error().message);
            return;
        }
        bloomDownShader_ = std::move(r.value());
    }
    {
        auto r = vk2::ShaderModule::fromFile(device_, shaderBase + "world/v2_post/v2_bloom_composite_comp.spv");
        if (!r) {
            log::error("postprocess", "Failed to load v2_bloom_composite_comp.spv: " + r.error().message);
            return;
        }
        bloomCompShader_ = std::move(r.value());
    }

    // --- Linear sampler for all passes ---
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

    // --- CAS descriptor set layout (binding 0: sampler, binding 1: storage image) ---
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
        if (vkCreateDescriptorSetLayout(device_, &ci, nullptr, &casDescSetLayout_) != VK_SUCCESS) {
            log::error("postprocess", "CAS descriptor set layout failed"); return;
        }
    }

    // --- Bloom downsample descriptor set layout ---
    //   binding 0: combined image sampler (input full-res)
    //   binding 1: storage image (bloom half-res output, rgba16f)
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
        if (vkCreateDescriptorSetLayout(device_, &ci, nullptr, &bloomDownDescSetLayout_) != VK_SUCCESS) {
            log::error("postprocess", "Bloom downsample descriptor set layout failed"); return;
        }
    }

    // --- Bloom composite descriptor set layout ---
    //   binding 0: combined image sampler (original full-res)
    //   binding 1: combined image sampler (bloom half-res)
    //   binding 2: storage image (output full-res, rgba16)
    {
        VkDescriptorSetLayoutBinding bindings[3]{};
        bindings[0] = {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                       VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        bindings[1] = {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                       VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        bindings[2] = {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1,
                       VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        ci.bindingCount = 3;
        ci.pBindings = bindings;
        if (vkCreateDescriptorSetLayout(device_, &ci, nullptr, &bloomCompDescSetLayout_) != VK_SUCCESS) {
            log::error("postprocess", "Bloom composite descriptor set layout failed"); return;
        }
    }

    // --- Per-frame descriptor allocator (up to 3 sets/frame: bloom_down + bloom_comp + CAS) ---
    {
        // Pool sizes cover the worst case: 3 combined-image-samplers + 3 storage-images per frame
        std::vector<VkDescriptorPoolSize> sizes = {
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4},
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 3},
        };
        uint32_t fif = services.frame().framesInFlight();
        auto r = descAllocator_.init(device_, fif, sizes, 3);
        if (!r) { log::error("postprocess", "Descriptor allocator: " + r.error().message); return; }
    }

    // --- CAS pipeline layout (1 set + CasPushConstants) ---
    {
        VkPushConstantRange pcRange{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(CasPushConstants)};
        vk2::PipelineLayoutDesc pld;
        pld.setLayouts = {casDescSetLayout_};
        pld.pushConstants = {pcRange};
        auto r = vk2::PipelineLayout::create(device_, pld);
        if (!r) { log::error("postprocess", "CAS pipeline layout: " + r.error().message); return; }
        casPipelineLayout_ = std::move(r.value());
    }

    // --- Bloom downsample pipeline layout (1 set + BloomPushConstants) ---
    {
        VkPushConstantRange pcRange{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(BloomPushConstants)};
        vk2::PipelineLayoutDesc pld;
        pld.setLayouts = {bloomDownDescSetLayout_};
        pld.pushConstants = {pcRange};
        auto r = vk2::PipelineLayout::create(device_, pld);
        if (!r) { log::error("postprocess", "Bloom downsample pipeline layout: " + r.error().message); return; }
        bloomDownPipelineLayout_ = std::move(r.value());
    }

    // --- Bloom composite pipeline layout (1 set + BloomPushConstants) ---
    {
        VkPushConstantRange pcRange{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(BloomPushConstants)};
        vk2::PipelineLayoutDesc pld;
        pld.setLayouts = {bloomCompDescSetLayout_};
        pld.pushConstants = {pcRange};
        auto r = vk2::PipelineLayout::create(device_, pld);
        if (!r) { log::error("postprocess", "Bloom composite pipeline layout: " + r.error().message); return; }
        bloomCompPipelineLayout_ = std::move(r.value());
    }

    // --- CAS compute pipeline ---
    {
        vk2::ComputePipelineDesc d;
        d.shaderModule = casShader_.handle();
        d.layout = casPipelineLayout_.handle();
        auto r = vk2::ComputePipeline::create(device_, d);
        if (!r) { log::error("postprocess", "CAS pipeline: " + r.error().message); return; }
        casPipeline_ = std::move(r.value());
    }

    // --- Bloom downsample compute pipeline ---
    {
        vk2::ComputePipelineDesc d;
        d.shaderModule = bloomDownShader_.handle();
        d.layout = bloomDownPipelineLayout_.handle();
        auto r = vk2::ComputePipeline::create(device_, d);
        if (!r) { log::error("postprocess", "Bloom downsample pipeline: " + r.error().message); return; }
        bloomDownPipeline_ = std::move(r.value());
    }

    // --- Bloom composite compute pipeline ---
    {
        vk2::ComputePipelineDesc d;
        d.shaderModule = bloomCompShader_.handle();
        d.layout = bloomCompPipelineLayout_.handle();
        auto r = vk2::ComputePipeline::create(device_, d);
        if (!r) { log::error("postprocess", "Bloom composite pipeline: " + r.error().message); return; }
        bloomCompPipeline_ = std::move(r.value());
    }

    initialized_ = true;
    bloomInitialized_ = true;
    profileSlot_ = services.metrics().registerTimer("PostProcess");
    log::info("postprocess", "PostProcessAdapter (bloom + CAS) initialized");
}

void PostProcessAdapter::configure(const ConfigSnapshot& config) {
    sharpness_ = config.data.casSharpness;
    if (sharpness_ < 0.0f) sharpness_ = 0.0f;
    if (sharpness_ > 1.0f) sharpness_ = 1.0f;

    sharpenerMode_ = config.data.sharpenerMode;
    bloomEnabled_ = config.data.bloomEnabled;
    bloomIntensity_ = config.data.bloomIntensity;
    bloomThreshold_ = config.data.bloomThreshold;
}

void PostProcessAdapter::ensureBloomImages(uint32_t fullWidth, uint32_t fullHeight) {
    if (lastRenderWidth_ == fullWidth && lastRenderHeight_ == fullHeight) return;

    // Destroy old images by assigning default-constructed (triggers RAII destruction)
    bloomHalfRes_ = {};
    bloomIntermediate_ = {};

    uint32_t halfW = std::max(1u, fullWidth / 2);
    uint32_t halfH = std::max(1u, fullHeight / 2);

    // Half-res bloom buffer (rgba16f for HDR-range bright pass)
    {
        vk2::Image::Desc desc{};
        desc.width = halfW;
        desc.height = halfH;
        desc.format = VK_FORMAT_R16G16B16A16_SFLOAT;
        desc.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        auto r = vk2::Image::create(device_, services_->device().vma(), desc);
        if (!r) {
            log::error("postprocess", "Failed to create bloom half-res image: " + r.error().message);
            return;
        }
        bloomHalfRes_ = std::move(r.value());
    }

    // Full-res intermediate (rgba16 unorm, same as CAS output format)
    {
        vk2::Image::Desc desc{};
        desc.width = fullWidth;
        desc.height = fullHeight;
        desc.format = VK_FORMAT_R16G16B16A16_UNORM;
        desc.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        auto r = vk2::Image::create(device_, services_->device().vma(), desc);
        if (!r) {
            log::error("postprocess", "Failed to create bloom intermediate image: " + r.error().message);
            bloomHalfRes_ = {};
            return;
        }
        bloomIntermediate_ = std::move(r.value());
    }

    lastRenderWidth_ = fullWidth;
    lastRenderHeight_ = fullHeight;

    log::info("postprocess", "Bloom images created: " + std::to_string(halfW) + "x" +
              std::to_string(halfH) + " half, " + std::to_string(fullWidth) + "x" +
              std::to_string(fullHeight) + " intermediate");
}

void PostProcessAdapter::executeBloom(VkCommandBuffer cmd, const PassResources& resources,
                                       uint32_t inIdx, uint32_t outIdx,
                                       uint32_t frameIndex) {
    auto& pool = services_->frame().resourcePool();
    uint32_t inW = pool.image(inIdx).width();
    uint32_t inH = pool.image(inIdx).height();
    uint32_t outW = pool.image(outIdx).width();
    uint32_t outH = pool.image(outIdx).height();

    ensureBloomImages(inW, inH);
    if (bloomHalfRes_.handle() == VK_NULL_HANDLE ||
        bloomIntermediate_.handle() == VK_NULL_HANDLE) return;

    uint32_t halfW = bloomHalfRes_.width();
    uint32_t halfH = bloomHalfRes_.height();
    bool casEnabled = (sharpenerMode_ != 0);

    // Transition bloom images to writable layouts for the first use each frame.
    // bloomHalfRes_: UNDEFINED -> GENERAL (storage write target)
    imageBarrier(cmd, bloomHalfRes_.handle(),
                 VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                 0, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                 VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
    bloomHalfRes_.setLayout(VK_IMAGE_LAYOUT_GENERAL);

    // ========== PASS 1: Bloom Downsample ==========
    {
        auto setResult = descAllocator_.allocate(bloomDownDescSetLayout_, frameIndex);
        if (!setResult) return;
        VkDescriptorSet descSet = setResult.value();

        // Binding 0: input (full-res tone-mapped LDR) as combined image sampler
        VkDescriptorImageInfo inInfo{};
        inInfo.sampler = linearSampler_;
        inInfo.imageView = resources.resolved->views[inIdx];
        inInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        // Binding 1: bloom half-res as storage image (write)
        VkDescriptorImageInfo bloomInfo{};
        bloomInfo.imageView = bloomHalfRes_.defaultView();
        bloomInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

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
        writes[1].pImageInfo = &bloomInfo;

        vkUpdateDescriptorSets(device_, 2, writes, 0, nullptr);

        BloomPushConstants pc{};
        pc.threshold = bloomThreshold_;
        pc.intensity = bloomIntensity_;
        pc.texelSizeX = 1.0f / static_cast<float>(inW);
        pc.texelSizeY = 1.0f / static_cast<float>(inH);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, bloomDownPipeline_.handle());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, bloomDownPipelineLayout_.handle(),
                                0, 1, &descSet, 0, nullptr);
        vkCmdPushConstants(cmd, bloomDownPipelineLayout_.handle(), VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(pc), &pc);

        uint32_t groupsX = (halfW + 7) / 8;
        uint32_t groupsY = (halfH + 7) / 8;
        vkCmdDispatch(cmd, groupsX, groupsY, 1);
    }

    // Barrier: bloom half-res GENERAL (storage write) -> SHADER_READ_ONLY_OPTIMAL (sampler read)
    imageBarrier(cmd, bloomHalfRes_.handle(),
                 VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                 VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                 VK_ACCESS_2_SHADER_SAMPLED_READ_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
    bloomHalfRes_.setLayout(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    // ========== PASS 2: Bloom Composite ==========
    // If CAS is enabled, composite writes to bloomIntermediate_ so CAS can read it.
    // If CAS is disabled, composite writes directly to the graph output.
    {
        // Transition the write target
        if (casEnabled) {
            imageBarrier(cmd, bloomIntermediate_.handle(),
                         VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                         0, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                         VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
            bloomIntermediate_.setLayout(VK_IMAGE_LAYOUT_GENERAL);
        }
        // If !casEnabled, the output image is already in GENERAL layout from the graph.

        auto setResult = descAllocator_.allocate(bloomCompDescSetLayout_, frameIndex);
        if (!setResult) return;
        VkDescriptorSet descSet = setResult.value();

        // Binding 0: original full-res (combined image sampler)
        VkDescriptorImageInfo origInfo{};
        origInfo.sampler = linearSampler_;
        origInfo.imageView = resources.resolved->views[inIdx];
        origInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        // Binding 1: bloom half-res (combined image sampler)
        VkDescriptorImageInfo bloomSamplerInfo{};
        bloomSamplerInfo.sampler = linearSampler_;
        bloomSamplerInfo.imageView = bloomHalfRes_.defaultView();
        bloomSamplerInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        // Binding 2: output (storage image)
        VkDescriptorImageInfo outInfo{};
        if (casEnabled) {
            outInfo.imageView = bloomIntermediate_.defaultView();
        } else {
            outInfo.imageView = resources.resolved->views[outIdx];
        }
        outInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkWriteDescriptorSet writes[3]{};
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = descSet;
        writes[0].dstBinding = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[0].pImageInfo = &origInfo;

        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet = descSet;
        writes[1].dstBinding = 1;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[1].pImageInfo = &bloomSamplerInfo;

        writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[2].dstSet = descSet;
        writes[2].dstBinding = 2;
        writes[2].descriptorCount = 1;
        writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[2].pImageInfo = &outInfo;

        vkUpdateDescriptorSets(device_, 3, writes, 0, nullptr);

        BloomPushConstants pc{};
        pc.threshold = bloomThreshold_;
        pc.intensity = bloomIntensity_;
        pc.texelSizeX = 1.0f / static_cast<float>(halfW);
        pc.texelSizeY = 1.0f / static_cast<float>(halfH);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, bloomCompPipeline_.handle());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, bloomCompPipelineLayout_.handle(),
                                0, 1, &descSet, 0, nullptr);
        vkCmdPushConstants(cmd, bloomCompPipelineLayout_.handle(), VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(pc), &pc);

        uint32_t groupsX = (outW + 7) / 8;
        uint32_t groupsY = (outH + 7) / 8;
        vkCmdDispatch(cmd, groupsX, groupsY, 1);
    }

    // If CAS is enabled, transition the intermediate to sampler-read for CAS
    if (casEnabled) {
        imageBarrier(cmd, bloomIntermediate_.handle(),
                     VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                     VK_ACCESS_2_SHADER_SAMPLED_READ_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
        bloomIntermediate_.setLayout(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }
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

    // Begin per-pass GPU timer
    if (profileSlot_ != UINT32_MAX)
        services_->metrics().beginNamedTimer(cmd, ctx.frameIndex, profileSlot_);

    // NV checkpoint before the first compute dispatch
    nvInsertCheckpoint(cmd,
                       ctx.config && (ctx.config->data.diagFlags & DiagFlags::CRASH) != 0,
                       services_->device().caps().nvCheckpoints,
                       "POSTPROCESS_START");

    // Reset the descriptor allocator for this frame
    descAllocator_.resetFrame(ctx.frameIndex);

    bool casEnabled = (sharpenerMode_ != 0);
    bool bloomActive = bloomEnabled_ && bloomInitialized_ && bloomIntensity_ > 0.0f;

    if (bloomActive) {
        // Execute bloom passes (downsample + composite)
        executeBloom(cmd, resources, inIdx, outIdx, ctx.frameIndex);

        if (casEnabled) {
            // CAS reads from bloomIntermediate_, writes to graph output
            auto setResult = descAllocator_.allocate(casDescSetLayout_, ctx.frameIndex);
            if (!setResult) {
                if (profileSlot_ != UINT32_MAX)
                    services_->metrics().endNamedTimer(cmd, ctx.frameIndex, profileSlot_);
                return;
            }
            VkDescriptorSet descSet = setResult.value();

            VkDescriptorImageInfo casInInfo{};
            casInInfo.sampler = linearSampler_;
            casInInfo.imageView = bloomIntermediate_.defaultView();
            casInInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            VkDescriptorImageInfo casOutInfo{};
            casOutInfo.imageView = resources.resolved->views[outIdx];
            casOutInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

            VkWriteDescriptorSet writes[2]{};
            writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[0].dstSet = descSet;
            writes[0].dstBinding = 0;
            writes[0].descriptorCount = 1;
            writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[0].pImageInfo = &casInInfo;

            writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[1].dstSet = descSet;
            writes[1].dstBinding = 1;
            writes[1].descriptorCount = 1;
            writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            writes[1].pImageInfo = &casOutInfo;

            vkUpdateDescriptorSets(device_, 2, writes, 0, nullptr);

            CasPushConstants pc = buildCasConstants(sharpness_,
                static_cast<float>(outW), static_cast<float>(outH),
                static_cast<float>(outW), static_cast<float>(outH));

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, casPipeline_.handle());
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, casPipelineLayout_.handle(),
                                    0, 1, &descSet, 0, nullptr);
            vkCmdPushConstants(cmd, casPipelineLayout_.handle(), VK_SHADER_STAGE_COMPUTE_BIT,
                               0, sizeof(pc), &pc);

            uint32_t groupsX = (outW + 15) / 16;
            uint32_t groupsY = (outH + 15) / 16;
            vkCmdDispatch(cmd, groupsX, groupsY, 1);
        }
        // If CAS is disabled, bloom composite already wrote to the graph output.
    } else {
        // No bloom — CAS reads directly from graph input (original behavior)
        auto setResult = descAllocator_.allocate(casDescSetLayout_, ctx.frameIndex);
        if (!setResult) {
            if (profileSlot_ != UINT32_MAX)
                services_->metrics().endNamedTimer(cmd, ctx.frameIndex, profileSlot_);
            return;
        }
        VkDescriptorSet descSet = setResult.value();

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

        CasPushConstants pc = buildCasConstants(sharpness_,
            static_cast<float>(inW), static_cast<float>(inH),
            static_cast<float>(outW), static_cast<float>(outH));

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, casPipeline_.handle());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, casPipelineLayout_.handle(),
                                0, 1, &descSet, 0, nullptr);
        vkCmdPushConstants(cmd, casPipelineLayout_.handle(), VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(pc), &pc);

        uint32_t groupsX = (outW + 15) / 16;
        uint32_t groupsY = (outH + 15) / 16;
        vkCmdDispatch(cmd, groupsX, groupsY, 1);
    }

    // End per-pass GPU timer
    if (profileSlot_ != UINT32_MAX)
        services_->metrics().endNamedTimer(cmd, ctx.frameIndex, profileSlot_);
}

void PostProcessAdapter::shutdown() {
    if (device_ == VK_NULL_HANDLE) return;

    // Destroy adapter-owned images
    bloomHalfRes_ = {};
    bloomIntermediate_ = {};
    lastRenderWidth_ = 0;
    lastRenderHeight_ = 0;

    // Destroy pipelines
    casPipeline_ = {};
    casPipelineLayout_ = {};
    casShader_ = {};
    bloomDownPipeline_ = {};
    bloomDownPipelineLayout_ = {};
    bloomDownShader_ = {};
    bloomCompPipeline_ = {};
    bloomCompPipelineLayout_ = {};
    bloomCompShader_ = {};

    // Destroy descriptor allocator and layouts
    descAllocator_.shutdown();
    if (casDescSetLayout_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device_, casDescSetLayout_, nullptr);
        casDescSetLayout_ = VK_NULL_HANDLE;
    }
    if (bloomDownDescSetLayout_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device_, bloomDownDescSetLayout_, nullptr);
        bloomDownDescSetLayout_ = VK_NULL_HANDLE;
    }
    if (bloomCompDescSetLayout_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device_, bloomCompDescSetLayout_, nullptr);
        bloomCompDescSetLayout_ = VK_NULL_HANDLE;
    }
    if (linearSampler_ != VK_NULL_HANDLE) {
        vkDestroySampler(device_, linearSampler_, nullptr);
        linearSampler_ = VK_NULL_HANDLE;
    }

    initialized_ = false;
    bloomInitialized_ = false;
    log::info("postprocess", "PostProcessAdapter shut down");
}

} // namespace engine
