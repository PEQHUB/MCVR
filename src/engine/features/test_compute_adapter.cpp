#include "test_compute_adapter.hpp"
#include "app/engine_services.hpp"
#include "frame/frame_scheduler.hpp"
#include "platform/vulkan/vk2_device.hpp"
#include "rendergraph/graph_builder.hpp"
#include "diagnostics/log.hpp"

#include <volk.h>
#include <cmath>

namespace engine {

ResourceHandle TestComputeAdapter::registerPass(GraphBuilder& builder) {
    ImageResourceDesc outDesc;
    outDesc.name = "test_compute_output";
    outDesc.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    outputHandle_ = builder.addResource(outDesc);

    PassDescriptor pass;
    pass.name = "test_compute";
    pass.queue = QueueAffinity::Compute;
    pass.outputs.push_back(outputHandle_);
    pass.execute = [this](const FrameContext& ctx, const PassResources& res) {
        execute(ctx, res);
    };
    builder.addPass(std::move(pass));
    builder.setFinalOutput(outputHandle_);

    return outputHandle_;
}

void TestComputeAdapter::init(EngineServices& services) {
    services_ = &services;
    device_ = services.device().device();

    // Load shader
    std::string shaderPath = services.resourceDir() + "/shaders/world/v2_test/clear_image_comp.spv";
    auto shaderResult = vk2::ShaderModule::fromFile(device_, shaderPath);
    if (!shaderResult) {
        log::error("test-compute", "Failed to load shader: " + shaderResult.error().message);
        return;
    }
    shader_ = std::move(shaderResult.value());

    // Descriptor set layout: 1 storage image at binding 0
    VkDescriptorSetLayoutBinding binding{};
    binding.binding = 0;
    binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    binding.descriptorCount = 1;
    binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo layoutCI{};
    layoutCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutCI.bindingCount = 1;
    layoutCI.pBindings = &binding;

    if (vkCreateDescriptorSetLayout(device_, &layoutCI, nullptr, &descSetLayout_) != VK_SUCCESS) {
        log::error("test-compute", "Failed to create descriptor set layout");
        return;
    }

    // Per-frame descriptor allocator
    {
        std::vector<VkDescriptorPoolSize> sizes = {
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1},
        };
        uint32_t fif = services.frame().framesInFlight();
        auto allocR = descAllocator_.init(device_, fif, sizes, 1);
        if (!allocR) { log::error("test-compute", "Descriptor allocator: " + allocR.error().message); return; }
    }

    // Pipeline layout: 1 set + push constant (vec4 color)
    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushRange.offset = 0;
    pushRange.size = sizeof(float) * 4;

    vk2::PipelineLayoutDesc layoutDesc;
    layoutDesc.setLayouts = {descSetLayout_};
    layoutDesc.pushConstants = {pushRange};

    auto layoutResult = vk2::PipelineLayout::create(device_, layoutDesc);
    if (!layoutResult) {
        log::error("test-compute", "Failed to create pipeline layout: " + layoutResult.error().message);
        return;
    }
    pipelineLayout_ = std::move(layoutResult.value());

    // Compute pipeline
    vk2::ComputePipelineDesc pipeDesc;
    pipeDesc.shaderModule = shader_.handle();
    pipeDesc.layout = pipelineLayout_.handle();

    auto pipeResult = vk2::ComputePipeline::create(device_, pipeDesc);
    if (!pipeResult) {
        log::error("test-compute", "Failed to create compute pipeline: " + pipeResult.error().message);
        return;
    }
    pipeline_ = std::move(pipeResult.value());

    initialized_ = true;
    log::info("test-compute", "TestComputeAdapter initialized");
}

void TestComputeAdapter::configure(const ConfigSnapshot&) {
    // Nothing to configure
}

void TestComputeAdapter::execute(const FrameContext& ctx, const PassResources& resources) {
    if (!initialized_ || !resources.resolved || resources.cmd == VK_NULL_HANDLE) return;

    VkCommandBuffer cmd = resources.cmd;

    // Allocate a fresh descriptor set for this frame
    descAllocator_.resetFrame(ctx.frameIndex);
    auto setResult = descAllocator_.allocate(descSetLayout_, ctx.frameIndex);
    if (!setResult) return;
    VkDescriptorSet descSet = setResult.value();

    // Write output image binding
    if (!resources.outputs.empty() && resources.outputs[0].valid()) {
        VkDescriptorImageInfo imgInfo{};
        imgInfo.imageView = resources.resolved->views[resources.outputs[0].index];
        imgInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkWriteDescriptorSet write{};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = descSet;
        write.dstBinding = 0;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        write.pImageInfo = &imgInfo;

        vkUpdateDescriptorSets(device_, 1, &write, 0, nullptr);
    }

    // Cycling color
    float t = static_cast<float>(ctx.frameNumber % 360) / 360.0f;
    float color[4] = {
        0.5f * (1.0f + std::sin(t * 6.28318f)),
        0.5f * (1.0f + std::sin(t * 6.28318f + 2.094f)),
        0.5f * (1.0f + std::sin(t * 6.28318f + 4.189f)),
        1.0f
    };

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_.handle());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout_.handle(),
                            0, 1, &descSet, 0, nullptr);
    vkCmdPushConstants(cmd, pipelineLayout_.handle(), VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(color), color);

    // Dispatch — image size from resolved resources
    uint32_t outIdx = resources.outputs[0].index;
    // We don't have direct width/height from ResolvedResources, so use push constants
    // The shader reads imageSize() to bounds-check
    uint32_t w = 0, h = 0;
    // Get dimensions from the resource pool via services
    auto& pool = services_->frame().resourcePool();
    w = pool.image(outIdx).width();
    h = pool.image(outIdx).height();

    vkCmdDispatch(cmd, (w + 15) / 16, (h + 15) / 16, 1);
}

void TestComputeAdapter::shutdown() {
    if (device_ == VK_NULL_HANDLE) return;

    pipeline_ = {};
    pipelineLayout_ = {};
    shader_ = {};

    descAllocator_.shutdown();
    if (descSetLayout_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device_, descSetLayout_, nullptr);
        descSetLayout_ = VK_NULL_HANDLE;
    }

    initialized_ = false;
    log::info("test-compute", "TestComputeAdapter shut down");
}

} // namespace engine
