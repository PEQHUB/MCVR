#include "raytracing_adapter.hpp"
#include "app/engine_services.hpp"
#include "frame/frame_scheduler.hpp"
#include "platform/vulkan/vk2_device.hpp"
#include "scene/tlas_service.hpp"
#include "scene/scene_resource_service.hpp"
#include "rendergraph/graph_builder.hpp"
#include "diagnostics/log.hpp"

#include <volk.h>
#include <cstring>

namespace engine {

ResourceHandle RayTracingAdapter::registerPass(GraphBuilder& builder) {
    ImageResourceDesc outDesc;
    outDesc.name = "rt_radiance";
    outDesc.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    outputHandle_ = builder.addResource(outDesc);

    PassDescriptor pass;
    pass.name = "ray_tracing";
    pass.queue = QueueAffinity::Graphics;
    pass.outputs.push_back(outputHandle_);
    pass.execute = [this](const FrameContext& ctx, const PassResources& res) {
        execute(ctx, res);
    };
    builder.addPass(std::move(pass));

    return outputHandle_;
}

void RayTracingAdapter::init(EngineServices& services) {
    services_ = &services;
    device_ = services.device().device();

    if (!services.device().caps().rayTracingPipeline) {
        log::warn("rt-adapter", "RT pipeline not supported — adapter disabled");
        return;
    }

    // Load MVP shaders (PR39 prep — minimum viable real RT)
    std::string base = services.resourceDir() + "/shaders/world/v2_min/";
    auto rgenR = vk2::ShaderModule::fromFile(device_, base + "v2_world_rgen.spv");
    auto rchitR = vk2::ShaderModule::fromFile(device_, base + "v2_world_rchit.spv");
    auto rmissR = vk2::ShaderModule::fromFile(device_, base + "v2_world_rmiss.spv");
    if (!rgenR || !rchitR || !rmissR) {
        log::error("rt-adapter", "Failed to load v2_world RT shaders");
        return;
    }
    rgenShader_ = std::move(rgenR.value());
    rchitShader_ = std::move(rchitR.value());
    rmissShader_ = std::move(rmissR.value());

    // V1-compatible descriptor layout (4 sets + 160B push constant)
    {
        auto r = layout_.init(device_);
        if (!r) { log::error("rt-adapter", "Layout init: " + r.error().message); return; }
    }

    // Per-frame descriptor allocator sized for the full 4-set layout
    {
        auto sizes = RtDescriptorLayout::poolSizes();
        uint32_t fif = services.frame().framesInFlight();
        // 4 sets per frame (one per layout)
        auto r = descAllocator_.init(device_, fif, sizes, 4);
        if (!r) { log::error("rt-adapter", "Descriptor allocator: " + r.error().message); return; }
    }

    // RT pipeline using the V1-compatible layout
    {
        vk2::RtPipelineDesc desc;

        VkPipelineShaderStageCreateInfo rgenStage{};
        rgenStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        rgenStage.stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
        rgenStage.module = rgenShader_.handle();
        rgenStage.pName = "main";

        VkPipelineShaderStageCreateInfo missStage{};
        missStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        missStage.stage = VK_SHADER_STAGE_MISS_BIT_KHR;
        missStage.module = rmissShader_.handle();
        missStage.pName = "main";

        VkPipelineShaderStageCreateInfo chitStage{};
        chitStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        chitStage.stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
        chitStage.module = rchitShader_.handle();
        chitStage.pName = "main";

        desc.stages = {rgenStage, missStage, chitStage};
        desc.groups.push_back({VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR, 0});
        desc.groups.push_back({VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR, 1});
        desc.groups.push_back({VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR,
                               VK_SHADER_UNUSED_KHR, 2});

        desc.layout = layout_.pipelineLayout();
        desc.maxRecursionDepth = 1;

        auto r = vk2::RtPipeline::create(device_, services.device().physicalDevice(), desc);
        if (!r) { log::error("rt-adapter", "RT pipeline: " + r.error().message); return; }
        rtPipeline_ = std::move(r.value());
    }

    // Build SBT: 1 rgen, 1 miss, 1 hit
    {
        auto r = sbt_.build(device_, services.device().vma(), services.device().physicalDevice(),
                            rtPipeline_, 1, 1, 1);
        if (!r) { log::error("rt-adapter", "SBT: " + r.error().message); return; }
    }

    initialized_ = true;
    log::info("rt-adapter", "RayTracingAdapter initialized (V1-compatible 4-set layout)");
}

void RayTracingAdapter::configure(const ConfigSnapshot&) {}

void RayTracingAdapter::execute(const FrameContext& ctx, const PassResources& resources) {
    if (!resources.resolved || resources.cmd == VK_NULL_HANDLE) return;
    if (resources.outputs.empty() || !resources.outputs[0].valid()) return;

    VkCommandBuffer cmd = resources.cmd;
    uint32_t outIdx = resources.outputs[0].index;
    auto& pool = services_->frame().resourcePool();
    uint32_t w = pool.image(outIdx).width();
    uint32_t h = pool.image(outIdx).height();

    // If not initialized or no valid TLAS, clear output to sky color in GENERAL layout
    bool canTrace = initialized_ && services_->tlas().isValid()
                    && services_->sceneRes().isInitialized();
    if (!canTrace) {
        VkClearColorValue clearColor = {{0.4f, 0.6f, 0.9f, 1.0f}};
        VkImageSubresourceRange range = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdClearColorImage(cmd, resources.resolved->images[outIdx],
                             VK_IMAGE_LAYOUT_GENERAL, &clearColor, 1, &range);
        return;
    }

    // Allocate fresh descriptor sets for all 4 layouts
    descAllocator_.resetFrame(ctx.frameIndex);
    VkDescriptorSet sets[4] = {VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE};
    for (uint32_t i = 0; i < 4; ++i) {
        auto r = descAllocator_.allocate(layout_.setLayout(i), ctx.frameIndex);
        if (!r) {
            log::error("rt-adapter", "set " + std::to_string(i) + " alloc: " + r.error().message);
            return;
        }
        sets[i] = r.value();
    }

    // Write the bindings the v2_world MVP shader reads:
    //   set 1 binding 0: TLAS
    //   set 1 binding 2: vertex BDA SSBO (TlasService)
    //   set 1 binding 3: index BDA SSBO  (TlasService)
    //   set 2 binding 0: WorldUBO
    //   set 2 binding 2: SkyUBO
    //   set 3 binding 0: output radiance image
    // PARTIALLY_BOUND lets us leave everything else unwritten.
    auto& sceneRes = services_->sceneRes();
    auto& tlasService = services_->tlas();

    VkAccelerationStructureKHR tlas = tlasService.handle();
    VkWriteDescriptorSetAccelerationStructureKHR asWrite{};
    asWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
    asWrite.accelerationStructureCount = 1;
    asWrite.pAccelerationStructures = &tlas;

    VkDescriptorBufferInfo vtxBdaInfo{};
    vtxBdaInfo.buffer = tlasService.vertexBdaBuffer();
    vtxBdaInfo.offset = 0;
    vtxBdaInfo.range = tlasService.bdaArraySize();
    if (vtxBdaInfo.range == 0) vtxBdaInfo.range = VK_WHOLE_SIZE;

    VkDescriptorBufferInfo idxBdaInfo{};
    idxBdaInfo.buffer = tlasService.indexBdaBuffer();
    idxBdaInfo.offset = 0;
    idxBdaInfo.range = tlasService.bdaArraySize();
    if (idxBdaInfo.range == 0) idxBdaInfo.range = VK_WHOLE_SIZE;

    VkDescriptorBufferInfo worldUboInfo{};
    worldUboInfo.buffer = sceneRes.worldUBO(ctx.frameIndex);
    worldUboInfo.offset = 0;
    worldUboInfo.range = sceneRes.worldUBOSize();

    VkDescriptorBufferInfo skyUboInfo{};
    skyUboInfo.buffer = sceneRes.skyUBO(ctx.frameIndex);
    skyUboInfo.offset = 0;
    skyUboInfo.range = sceneRes.skyUBOSize();

    VkDescriptorImageInfo outImgInfo{};
    outImgInfo.imageView = resources.resolved->views[outIdx];
    outImgInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkWriteDescriptorSet writes[6]{};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].pNext = &asWrite;
    writes[0].dstSet = sets[rt_layout::SET_GEOMETRY];
    writes[0].dstBinding = rt_layout::GEO_BINDING_TLAS;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;

    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = sets[rt_layout::SET_GEOMETRY];
    writes[1].dstBinding = rt_layout::GEO_BINDING_VTX_ADDRS;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[1].pBufferInfo = &vtxBdaInfo;

    writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[2].dstSet = sets[rt_layout::SET_GEOMETRY];
    writes[2].dstBinding = rt_layout::GEO_BINDING_IDX_ADDRS;
    writes[2].descriptorCount = 1;
    writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[2].pBufferInfo = &idxBdaInfo;

    writes[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[3].dstSet = sets[rt_layout::SET_UNIFORMS];
    writes[3].dstBinding = rt_layout::UBO_BINDING_WORLD;
    writes[3].descriptorCount = 1;
    writes[3].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writes[3].pBufferInfo = &worldUboInfo;

    writes[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[4].dstSet = sets[rt_layout::SET_UNIFORMS];
    writes[4].dstBinding = rt_layout::UBO_BINDING_SKY;
    writes[4].descriptorCount = 1;
    writes[4].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writes[4].pBufferInfo = &skyUboInfo;

    writes[5].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[5].dstSet = sets[rt_layout::SET_OUTPUTS];
    writes[5].dstBinding = rt_layout::OUT_BINDING_HDR_NOISY;
    writes[5].descriptorCount = 1;
    writes[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[5].pImageInfo = &outImgInfo;

    // Skip vertex/index BDA writes if there's nothing to write yet (no chunks)
    uint32_t writeCount = 6;
    if (vtxBdaInfo.buffer == VK_NULL_HANDLE || idxBdaInfo.buffer == VK_NULL_HANDLE) {
        // BDA SSBOs not allocated yet (no chunks built TLAS yet) — skip those two writes
        // by shuffling: keep TLAS at 0, World/Sky/Out at 1/2/3, drop vtx/idx writes.
        writes[1] = writes[3];
        writes[2] = writes[4];
        writes[3] = writes[5];
        writeCount = 4;
    }

    vkUpdateDescriptorSets(device_, writeCount, writes, 0, nullptr);

    // Push constants — V1-compatible 160 bytes. Test shader ignores everything,
    // but the layout must match the C++ struct or vkCmdPushConstants validates wrong.
    RtPushConstants pc{};
    pc.numRayBounces = 1;
    pc.flags = 0;
    pc.preExposure = 1.0f;
    pc.colorExpansion = 1.0f;
    pc.sharcSceneScale = 4.0f;
    pc.sharcCapacity = 1u << 21;
    pc.sharcRadianceScale = 1.0f;
    pc.sharcRoughnessThreshold = 0.5f;
    pc.sharcUpdateBlockSize = 4;
    pc.sharcUpdateBounces = 4;
    pc.aperture = 0.0f;
    pc.focalDistance = 10.0f;
    pc.blueNoiseFrame = static_cast<uint32_t>(ctx.frameNumber);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, rtPipeline_.handle());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, layout_.pipelineLayout(),
                            0, 4, sets, 0, nullptr);
    vkCmdPushConstants(cmd, layout_.pipelineLayout(),
                       VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                       VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                       VK_SHADER_STAGE_INTERSECTION_BIT_KHR,
                       0, sizeof(pc), &pc);

    vkCmdTraceRaysKHR(cmd,
                      &sbt_.rgenRegion(), &sbt_.missRegion(),
                      &sbt_.hitRegion(), &sbt_.callableRegion(),
                      w, h, 1);
}

void RayTracingAdapter::shutdown() {
    if (device_ == VK_NULL_HANDLE) return;

    sbt_ = {};
    rtPipeline_ = {};
    layout_.shutdown();
    rgenShader_ = {};
    rchitShader_ = {};
    rmissShader_ = {};

    descAllocator_.shutdown();

    initialized_ = false;
    log::info("rt-adapter", "RayTracingAdapter shut down");
}

} // namespace engine
