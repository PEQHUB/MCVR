#include "tonemapping_adapter.hpp"
#include "app/engine_services.hpp"
#include "frame/frame_scheduler.hpp"
#include "platform/vulkan/vk2_device.hpp"
#include "rendergraph/graph_builder.hpp"
#include "diagnostics/log.hpp"

#include <volk.h>

namespace engine {

static constexpr uint32_t HISTOGRAM_BINS = 256;
static constexpr VkDeviceSize HISTOGRAM_SIZE = HISTOGRAM_BINS * sizeof(uint32_t);
static constexpr VkDeviceSize EXPOSURE_SIZE = 58 * sizeof(float); // ExposureBuffer in exposure.comp

void ToneMappingAdapter::init(EngineServices& services) {
    services_ = &services;
    device_ = services.device().device();
    auto vma = services.device().vma();

    // Load shaders
    std::string base = services.resourceDir() + "/shaders/world/tone_mapping/";
    auto histR = vk2::ShaderModule::fromFile(device_, base + "hist_comp.spv");
    auto expR = vk2::ShaderModule::fromFile(device_, base + "exposure_comp.spv");
    auto vertR = vk2::ShaderModule::fromFile(device_, base + "tone_mapping_vert.spv");
    auto fragR = vk2::ShaderModule::fromFile(device_, base + "tone_mapping_frag.spv");
    if (!histR || !expR || !vertR || !fragR) {
        log::error("tonemapping", "Failed to load shaders");
        return;
    }
    histShader_ = std::move(histR.value());
    exposureShader_ = std::move(expR.value());
    vertShader_ = std::move(vertR.value());
    fragShader_ = std::move(fragR.value());

    // Create histogram buffer (device-local, storage + transfer)
    {
        vk2::Buffer::Desc bd{};
        bd.size = HISTOGRAM_SIZE;
        bd.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
                 | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        auto r = vk2::Buffer::create(device_, vma, bd);
        if (!r) { log::error("tonemapping", "Histogram buffer: " + r.error().message); return; }
        histogramBuffer_ = std::move(r.value());
    }

    // Create exposure buffer (device-local, storage + transfer)
    {
        vk2::Buffer::Desc bd{};
        bd.size = EXPOSURE_SIZE;
        bd.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
                 | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        auto r = vk2::Buffer::create(device_, vma, bd);
        if (!r) { log::error("tonemapping", "Exposure buffer: " + r.error().message); return; }
        exposureBuffer_ = std::move(r.value());
    }

    // Create dummy 1x1 image for unused sampler bindings (0, 3)
    {
        vk2::Image::Desc id{};
        id.width = 1;
        id.height = 1;
        id.format = VK_FORMAT_R16G16B16A16_SFLOAT;
        id.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        auto r = vk2::Image::create(device_, vma, id);
        if (!r) { log::error("tonemapping", "Dummy image: " + r.error().message); return; }
        dummyImage_ = std::move(r.value());
    }

    // Create linear sampler
    {
        VkSamplerCreateInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        si.magFilter = VK_FILTER_LINEAR;
        si.minFilter = VK_FILTER_LINEAR;
        si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        if (vkCreateSampler(device_, &si, nullptr, &linearSampler_) != VK_SUCCESS) {
            log::error("tonemapping", "Sampler creation failed"); return;
        }
    }

    // Descriptor set layout: matches hist.comp / exposure.comp
    // binding 0: combined image sampler (display-res HDR — unused by hist, needed for layout)
    // binding 1: storage buffer (histogram)
    // binding 2: storage buffer (exposure)
    // binding 3: combined image sampler (emission — unused, layout parity)
    // binding 4: combined image sampler (render-res HDR — metering source)
    {
        VkDescriptorSetLayoutBinding bindings[5]{};
        bindings[0] = {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                       VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
        bindings[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                       VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
        bindings[2] = {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                       VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
        bindings[3] = {3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                       VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
        bindings[4] = {4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                       VK_SHADER_STAGE_COMPUTE_BIT, nullptr};

        VkDescriptorSetLayoutCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        ci.bindingCount = 5;
        ci.pBindings = bindings;
        if (vkCreateDescriptorSetLayout(device_, &ci, nullptr, &descSetLayout_) != VK_SUCCESS) {
            log::error("tonemapping", "Descriptor set layout failed"); return;
        }
    }

    // Per-frame descriptor allocator
    {
        std::vector<VkDescriptorPoolSize> sizes = {
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 3},
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2},
        };
        uint32_t fif = services.frame().framesInFlight();
        auto r = descAllocator_.init(device_, fif, sizes, 1);
        if (!r) { log::error("tonemapping", "Descriptor allocator: " + r.error().message); return; }
    }

    // Pipeline layout: 1 set + push constants (shared by compute + graphics)
    {
        VkPushConstantRange pcRange{
            VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            0, sizeof(ToneMappingPushConstants)};
        vk2::PipelineLayoutDesc pld;
        pld.setLayouts = {descSetLayout_};
        pld.pushConstants = {pcRange};
        auto r = vk2::PipelineLayout::create(device_, pld);
        if (!r) { log::error("tonemapping", "Pipeline layout: " + r.error().message); return; }
        pipelineLayout_ = std::move(r.value());
    }

    // Compute pipelines
    {
        vk2::ComputePipelineDesc d;
        d.layout = pipelineLayout_.handle();
        d.shaderModule = histShader_.handle();
        auto r = vk2::ComputePipeline::create(device_, d);
        if (!r) { log::error("tonemapping", "Hist pipeline: " + r.error().message); return; }
        histPipeline_ = std::move(r.value());
    }
    {
        vk2::ComputePipelineDesc d;
        d.layout = pipelineLayout_.handle();
        d.shaderModule = exposureShader_.handle();
        auto r = vk2::ComputePipeline::create(device_, d);
        if (!r) { log::error("tonemapping", "Exposure pipeline: " + r.error().message); return; }
        exposurePipeline_ = std::move(r.value());
    }

    // Graphics pipeline (fullscreen triangle tone curve, dynamic rendering)
    {
        vk2::GraphicsPipelineDesc d;
        d.vertexShader = vertShader_.handle();
        d.fragmentShader = fragShader_.handle();
        d.layout = pipelineLayout_.handle();
        d.colorAttachmentFormat = VK_FORMAT_R8G8B8A8_SRGB;
        auto r = vk2::GraphicsPipeline::create(device_, d);
        if (!r) { log::error("tonemapping", "Graphics pipeline: " + r.error().message); return; }
        graphicsPipeline_ = std::move(r.value());
    }

    initialized_ = true;
    log::info("tonemapping", "ToneMappingAdapter initialized (histogram + exposure compute)");
}

void ToneMappingAdapter::configure(const ConfigSnapshot& config) {
    pc_.tonemapMode = static_cast<float>(config.data.tonemappingMode);
    pc_.saturation = config.data.saturation;
    pc_.Lwhite = config.data.Lwhite;
    pc_.psychoPeakSDR = config.data.psychoPeakSDR;
    pc_.psychoEnabled = config.data.psychoEnabled ? 1.0f : 0.0f;
}

void ToneMappingAdapter::execute(const FrameContext& ctx, const PassResources& resources) {
    if (!initialized_ || !resources.resolved || resources.cmd == VK_NULL_HANDLE) return;

    configure(*ctx.config);
    pc_.dt = ctx.deltaTime;

    VkCommandBuffer cmd = resources.cmd;

    // Allocate a fresh descriptor set for this frame (avoids race with in-flight frames)
    descAllocator_.resetFrame(ctx.frameIndex);
    auto setResult = descAllocator_.allocate(descSetLayout_, ctx.frameIndex);
    if (!setResult) return;
    VkDescriptorSet descSet = setResult.value();

    // Write all bindings to the fresh set
    {
        VkDescriptorImageInfo dummyInfo{linearSampler_, dummyImage_.defaultView(),
                                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        VkDescriptorImageInfo inputInfo = dummyInfo; // fallback
        if (!resources.inputs.empty() && resources.inputs[0].valid()) {
            inputInfo = {linearSampler_,
                         resources.resolved->views[resources.inputs[0].index],
                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        }
        VkDescriptorBufferInfo histInfo{histogramBuffer_.handle(), 0, HISTOGRAM_SIZE};
        VkDescriptorBufferInfo expInfo{exposureBuffer_.handle(), 0, EXPOSURE_SIZE};

        VkWriteDescriptorSet writes[5]{};
        writes[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descSet, 0, 0, 1,
                     VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &inputInfo, nullptr, nullptr};
        writes[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descSet, 1, 0, 1,
                     VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &histInfo, nullptr};
        writes[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descSet, 2, 0, 1,
                     VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &expInfo, nullptr};
        writes[3] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descSet, 3, 0, 1,
                     VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &dummyInfo, nullptr, nullptr};
        writes[4] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descSet, 4, 0, 1,
                     VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &inputInfo, nullptr, nullptr};
        vkUpdateDescriptorSets(device_, 5, writes, 0, nullptr);
    }

    // Zero exposure buffer on first frame (device-local memory is undefined)
    bool needExposureBarrier = !exposureInitialized_;
    if (!exposureInitialized_) {
        vkCmdFillBuffer(cmd, exposureBuffer_.handle(), 0, EXPOSURE_SIZE, 0);
        exposureInitialized_ = true;
    }

    // 1. Zero histogram buffer
    vkCmdFillBuffer(cmd, histogramBuffer_.handle(), 0, HISTOGRAM_SIZE, 0);

    // Barrier: transfer → compute (histogram buffer, + exposure buffer on first frame)
    {
        VkBufferMemoryBarrier2 barriers[2]{};
        uint32_t barrierCount = 0;

        barriers[barrierCount].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
        barriers[barrierCount].srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        barriers[barrierCount].srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        barriers[barrierCount].dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        barriers[barrierCount].dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        barriers[barrierCount].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[barrierCount].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[barrierCount].buffer = histogramBuffer_.handle();
        barriers[barrierCount].offset = 0;
        barriers[barrierCount].size = HISTOGRAM_SIZE;
        ++barrierCount;

        if (needExposureBarrier) {
            barriers[barrierCount].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
            barriers[barrierCount].srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            barriers[barrierCount].srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            barriers[barrierCount].dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            barriers[barrierCount].dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            barriers[barrierCount].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barriers[barrierCount].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barriers[barrierCount].buffer = exposureBuffer_.handle();
            barriers[barrierCount].offset = 0;
            barriers[barrierCount].size = EXPOSURE_SIZE;
            ++barrierCount;
        }

        VkDependencyInfo dep{};
        dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.bufferMemoryBarrierCount = barrierCount;
        dep.pBufferMemoryBarriers = barriers;
        vkCmdPipelineBarrier2(cmd, &dep);
    }

    // 2. Dispatch histogram
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, histPipeline_.handle());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout_.handle(),
                            0, 1, &descSet, 0, nullptr);
    vkCmdPushConstants(cmd, pipelineLayout_.handle(), VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(ToneMappingPushConstants), &pc_);

    // Get metering image dimensions
    uint32_t w = 0, h = 0;
    if (!resources.inputs.empty() && resources.inputs[0].valid()) {
        auto& pool = services_->frame().resourcePool();
        w = pool.image(resources.inputs[0].index).width();
        h = pool.image(resources.inputs[0].index).height();
    }

    if (w > 0 && h > 0) {
        vkCmdDispatch(cmd, (w + 15) / 16, (h + 15) / 16, 1);
    }

    // Barrier: compute → compute (histogram buffer RAW)
    {
        VkBufferMemoryBarrier2 barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
        barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        barrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.buffer = histogramBuffer_.handle();
        barrier.offset = 0;
        barrier.size = HISTOGRAM_SIZE;

        VkDependencyInfo dep{};
        dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.bufferMemoryBarrierCount = 1;
        dep.pBufferMemoryBarriers = &barrier;
        vkCmdPipelineBarrier2(cmd, &dep);
    }

    // 3. Dispatch exposure solve
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, exposurePipeline_.handle());
    vkCmdDispatch(cmd, 1, 1, 1);

    // --- Stage 3: Fullscreen triangle tone curve ---

    // Check we have an output image to render to
    if (resources.outputs.empty() || !resources.outputs[0].valid()) return;

    uint32_t outIdx = resources.outputs[0].index;
    VkImageView outView = resources.resolved->views[outIdx];
    auto& outImg = services_->frame().resourcePool().image(outIdx);
    uint32_t outW = outImg.width();
    uint32_t outH = outImg.height();

    // Barrier: exposure buffer compute → fragment read
    {
        VkBufferMemoryBarrier2 barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
        barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        barrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.buffer = exposureBuffer_.handle();
        barrier.offset = 0;
        barrier.size = EXPOSURE_SIZE;

        VkDependencyInfo dep{};
        dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.bufferMemoryBarrierCount = 1;
        dep.pBufferMemoryBarriers = &barrier;
        vkCmdPipelineBarrier2(cmd, &dep);
    }

    // Barrier planner already transitioned output to GENERAL.
    // Use GENERAL for rendering (valid per Vulkan spec for dynamic rendering).

    // Begin dynamic rendering
    VkRenderingAttachmentInfo colorAttachment{};
    colorAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    colorAttachment.imageView = outView;
    colorAttachment.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

    VkRenderingInfo renderInfo{};
    renderInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    renderInfo.renderArea = {{0, 0}, {outW, outH}};
    renderInfo.layerCount = 1;
    renderInfo.colorAttachmentCount = 1;
    renderInfo.pColorAttachments = &colorAttachment;

    vkCmdBeginRendering(cmd, &renderInfo);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline_.handle());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout_.handle(),
                            0, 1, &descSet, 0, nullptr);

    VkViewport viewport{0.0f, 0.0f, static_cast<float>(outW), static_cast<float>(outH),
                        0.0f, 1.0f};
    VkRect2D scissor{{0, 0}, {outW, outH}};
    vkCmdSetViewport(cmd, 0, 1, &viewport);
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    vkCmdDraw(cmd, 3, 1, 0, 0);  // Fullscreen triangle

    vkCmdEndRendering(cmd);
}

void ToneMappingAdapter::shutdown() {
    if (device_ == VK_NULL_HANDLE) return;

    graphicsPipeline_ = {};
    histPipeline_ = {};
    exposurePipeline_ = {};
    pipelineLayout_ = {};
    histShader_ = {};
    exposureShader_ = {};
    vertShader_ = {};
    fragShader_ = {};
    histogramBuffer_ = {};
    exposureBuffer_ = {};
    dummyImage_ = {};

    if (linearSampler_ != VK_NULL_HANDLE) {
        vkDestroySampler(device_, linearSampler_, nullptr);
        linearSampler_ = VK_NULL_HANDLE;
    }
    descAllocator_.shutdown();
    if (descSetLayout_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device_, descSetLayout_, nullptr);
        descSetLayout_ = VK_NULL_HANDLE;
    }

    initialized_ = false;
    log::info("tonemapping", "ToneMappingAdapter shut down");
}

void ToneMappingAdapter::registerPass(GraphBuilder& builder, ResourceHandle inputHandle) {
    inputHandle_ = inputHandle;

    // Output: tone-mapped LDR image
    ImageResourceDesc outDesc;
    outDesc.name = "mapped_output";
    outDesc.format = VK_FORMAT_R8G8B8A8_SRGB;
    outputHandle_ = builder.addResource(outDesc);

    PassDescriptor pass;
    pass.name = "tonemapping";
    pass.queue = QueueAffinity::Graphics;
    pass.inputs.push_back(inputHandle);
    pass.outputs.push_back(outputHandle_);
    pass.execute = [this](const FrameContext& ctx, const PassResources& res) {
        execute(ctx, res);
    };
    builder.addPass(std::move(pass));
    builder.setFinalOutput(outputHandle_);
}

} // namespace engine
