#include "raytracing_adapter.hpp"
#include "atmosphere_adapter.hpp"
#include "app/engine_services.hpp"
#include "diagnostics/metrics_service.hpp"
#include "frame/frame_scheduler.hpp"
#include "platform/vulkan/vk2_device.hpp"
#include "scene/tlas_service.hpp"
#include "scene/scene_resource_service.hpp"
#include "scene/texture_service.hpp"
#include "rendergraph/graph_builder.hpp"
#include "diagnostics/log.hpp"
#include "diagnostics/diag_flags.hpp"
#include "config/engine_config.hpp"

#include <volk.h>
#include <cstdlib>
#include <cstring>

namespace engine {

namespace {

bool envVarEnabled(const char* name) {
    const char* value = std::getenv(name);
    if (!value || !*value) return false;
    return value[0] == '1' || value[0] == 't' || value[0] == 'T' ||
           value[0] == 'y' || value[0] == 'Y';
}

} // namespace

// --------------------------------------------------------------------------
// Static output specification table — one entry per set 3 binding.
// Order matches binding index; the binding field is explicit for clarity.
// --------------------------------------------------------------------------
const std::array<RayTracingAdapter::OutputSpec, RayTracingAdapter::OUTPUT_COUNT>
RayTracingAdapter::kOutputSpecs = {{
    // binding  name                        format                                   binding
    { "rt_radiance",              VK_FORMAT_R16G16B16A16_SFLOAT,  rt_layout::OUT_BINDING_HDR_NOISY },
    { "rt_diffuse_albedo",        VK_FORMAT_R8G8B8A8_UNORM,       rt_layout::OUT_BINDING_DIFFUSE_ALBEDO },
    { "rt_specular_albedo",       VK_FORMAT_R16G16B16A16_SFLOAT,  rt_layout::OUT_BINDING_SPECULAR_ALBEDO },
    { "rt_normal",                VK_FORMAT_R32G32B32A32_SFLOAT,  rt_layout::OUT_BINDING_NORMAL_ROUGHNESS },
    { "rt_motion",                VK_FORMAT_R32G32_SFLOAT,        rt_layout::OUT_BINDING_MOTION_VECTOR },
    { "rt_linear_depth",          VK_FORMAT_R32_SFLOAT,           rt_layout::OUT_BINDING_LINEAR_DEPTH },
    { "rt_specular_hit_depth",    VK_FORMAT_R16_SFLOAT,           rt_layout::OUT_BINDING_SPECULAR_HIT_DEPTH },
    { "rt_first_hit_depth",       VK_FORMAT_R32_SFLOAT,           rt_layout::OUT_BINDING_FIRST_HIT_DEPTH },
    { "rt_fh_diffuse_direct",     VK_FORMAT_R16G16B16A16_SFLOAT,  rt_layout::OUT_BINDING_FH_DIFFUSE_DIRECT },
    { "rt_fh_diffuse_indirect",   VK_FORMAT_R16G16B16A16_SFLOAT,  rt_layout::OUT_BINDING_FH_DIFFUSE_INDIRECT },
    { "rt_fh_specular",           VK_FORMAT_R16G16B16A16_SFLOAT,  rt_layout::OUT_BINDING_FH_SPECULAR },
    { "rt_fh_clear",              VK_FORMAT_R16G16B16A16_SFLOAT,  rt_layout::OUT_BINDING_FH_CLEAR },
    { "rt_fh_base_emission",      VK_FORMAT_R16G16B16A16_SFLOAT,  rt_layout::OUT_BINDING_FH_BASE_EMISSION },
    { "rt_direct_light_depth",    VK_FORMAT_R16_SFLOAT,           rt_layout::OUT_BINDING_DIRECT_LIGHT_DEPTH },
    { "rt_reservoir_current",     VK_FORMAT_R32G32B32A32_SFLOAT,  rt_layout::OUT_BINDING_RESERVOIR_CURRENT },
    { "rt_reservoir_previous",    VK_FORMAT_R32G32B32A32_SFLOAT,  rt_layout::OUT_BINDING_RESERVOIR_PREVIOUS },
    { "rt_diffuse_ray_dir",       VK_FORMAT_R16G16B16A16_SFLOAT,  rt_layout::OUT_BINDING_DIFFUSE_RAY_DIR },
    { "rt_specular_ray_dir",      VK_FORMAT_R16G16B16A16_SFLOAT,  rt_layout::OUT_BINDING_SPECULAR_RAY_DIR },
    { "rt_bounce_reservoir_1",    VK_FORMAT_R32G32B32A32_SFLOAT,  rt_layout::OUT_BINDING_BOUNCE_RESERVOIR_1 },
    { "rt_bounce_reservoir_2",    VK_FORMAT_R32G32B32A32_SFLOAT,  rt_layout::OUT_BINDING_BOUNCE_RESERVOIR_2 },
    { "rt_bounce_reservoir_3",    VK_FORMAT_R32G32B32A32_SFLOAT,  rt_layout::OUT_BINDING_BOUNCE_RESERVOIR_3 },
    { "rt_reflection_mv",         VK_FORMAT_R32G32_SFLOAT,        rt_layout::OUT_BINDING_REFLECTION_MV },
    { "rt_anim_tex_mask",         VK_FORMAT_R16_SFLOAT,           rt_layout::OUT_BINDING_ANIM_TEX_MASK },
    { "rt_particle_mask",         VK_FORMAT_R16_SFLOAT,           rt_layout::OUT_BINDING_PARTICLE_MASK },
    { "rt_bias_mask",             VK_FORMAT_R16_SFLOAT,           rt_layout::OUT_BINDING_BIAS_MASK },
    { "rt_hit_dist",              VK_FORMAT_R16_SFLOAT,           rt_layout::OUT_BINDING_RT_HIT_DIST },
    { "rt_motion_vec_3d",         VK_FORMAT_R16G16B16A16_SFLOAT,  rt_layout::OUT_BINDING_MOTION_VEC_3D },
    { "rt_gbuffer_metallic",      VK_FORMAT_R16_SFLOAT,           rt_layout::OUT_BINDING_GBUFFER_METALLIC },
    { "rt_gbuffer_shading_model_id", VK_FORMAT_R16_SFLOAT,        rt_layout::OUT_BINDING_GBUFFER_SHADING_MODEL_ID },
    { "rt_gbuffer_material_id",   VK_FORMAT_R16_SFLOAT,           rt_layout::OUT_BINDING_GBUFFER_MATERIAL_ID },
    { "rt_position_view_space",   VK_FORMAT_R32G32B32A32_SFLOAT,  rt_layout::OUT_BINDING_POSITION_VIEW_SPACE },
    { "rt_transparency_layer",    VK_FORMAT_R16G16B16A16_SFLOAT,  rt_layout::OUT_BINDING_TRANSPARENCY_LAYER },
    { "rt_transparency_opacity",  VK_FORMAT_R16G16B16A16_SFLOAT,  rt_layout::OUT_BINDING_TRANSPARENCY_OPACITY },
    { "rt_transparency_mvecs",    VK_FORMAT_R32G32_SFLOAT,        rt_layout::OUT_BINDING_TRANSPARENCY_MVECS },
}};

ResourceHandle RayTracingAdapter::registerPass(GraphBuilder& builder) {
    // Declare all 34 RT output images from the specification table.
    PassDescriptor pass;
    pass.name = "ray_tracing";
    pass.queue = QueueAffinity::Graphics;

    for (uint32_t i = 0; i < OUTPUT_COUNT; ++i) {
        const auto& spec = kOutputSpecs[i];
        ImageResourceDesc desc;
        desc.name   = spec.name;
        desc.format = spec.format;
        outputHandles_[spec.binding] = builder.addResource(desc);
        pass.outputs.push_back(outputHandles_[spec.binding]);
    }

    pass.execute = [this](const FrameContext& ctx, const PassResources& res) {
        execute(ctx, res);
    };
    builder.addPass(std::move(pass));

    return outputHandles_[rt_layout::OUT_BINDING_HDR_NOISY];
}

void RayTracingAdapter::init(EngineServices& services) {
    services_ = &services;
    device_ = services.device().device();

    if (!services.device().caps().rayTracingPipeline) {
        log::warn("rt-adapter", "RT pipeline not supported — adapter disabled");
        return;
    }

    // --- Shader loading: default to the minimal 3-stage path.
    // The 22-stage v2_full suite is still experimental and has repeatedly
    // produced GPU faults in-game when auto-enabled just because the SPIR-V
    // files exist. Opt in explicitly with RADIANCE_V2_ENABLE_FULL_RT=1.
    // ---
    fullShadersLoaded_ = false;
    const bool allowExperimentalFullRt = envVarEnabled("RADIANCE_V2_ENABLE_FULL_RT");

    // V1's full 22-stage shader suite (order matches V1 SBT layout exactly)
    static const char* kFullShaderNames[22] = {
        "world_rgen.spv",                    // 0  RAYGEN
        "world_rmiss.spv",                   // 1  MISS
        "hand_rmiss.spv",                    // 2  MISS
        "shadow_rmiss.spv",                  // 3  MISS
        "world_solid_transparent_rchit.spv", // 4  CLOSEST_HIT
        "world_no_reflect_rchit.spv",        // 5  CLOSEST_HIT
        "world_cloud_rchit.spv",             // 6  CLOSEST_HIT
        "world_transparent_rahit.spv",       // 7  ANY_HIT
        "world_no_reflect_rahit.spv",        // 8  ANY_HIT
        "world_cloud_rahit.spv",             // 9  ANY_HIT
        "shadow_rchit.spv",                  // 10 CLOSEST_HIT
        "shadow_rahit.spv",                  // 11 ANY_HIT
        "boat_water_mask_rchit.spv",         // 12 CLOSEST_HIT
        "boat_water_mask_rahit.spv",         // 13 ANY_HIT
        "end_portal_rchit.spv",              // 14 CLOSEST_HIT
        "end_portal_rahit.spv",              // 15 ANY_HIT
        "end_gateway_rchit.spv",             // 16 CLOSEST_HIT
        "end_gateway_rahit.spv",             // 17 ANY_HIT
        "point_light_shadow_rmiss.spv",      // 18 MISS
        "displaced_block_rint.spv",          // 19 INTERSECTION
        "displaced_block_rchit.spv",         // 20 CLOSEST_HIT
        "displaced_shadow_rchit.spv",        // 21 CLOSEST_HIT
    };

    static const VkShaderStageFlagBits kFullShaderStages[22] = {
        VK_SHADER_STAGE_RAYGEN_BIT_KHR,        // 0
        VK_SHADER_STAGE_MISS_BIT_KHR,          // 1
        VK_SHADER_STAGE_MISS_BIT_KHR,          // 2
        VK_SHADER_STAGE_MISS_BIT_KHR,          // 3
        VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,   // 4
        VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,   // 5
        VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,   // 6
        VK_SHADER_STAGE_ANY_HIT_BIT_KHR,       // 7
        VK_SHADER_STAGE_ANY_HIT_BIT_KHR,       // 8
        VK_SHADER_STAGE_ANY_HIT_BIT_KHR,       // 9
        VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,   // 10
        VK_SHADER_STAGE_ANY_HIT_BIT_KHR,       // 11
        VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,   // 12
        VK_SHADER_STAGE_ANY_HIT_BIT_KHR,       // 13
        VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,   // 14
        VK_SHADER_STAGE_ANY_HIT_BIT_KHR,       // 15
        VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,   // 16
        VK_SHADER_STAGE_ANY_HIT_BIT_KHR,       // 17
        VK_SHADER_STAGE_MISS_BIT_KHR,          // 18
        VK_SHADER_STAGE_INTERSECTION_BIT_KHR,  // 19
        VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,   // 20
        VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,   // 21
    };

    if (allowExperimentalFullRt) {
        std::string fullBase = services.resourceDir() + "/shaders/world/v2_full/";
        bool allLoaded = true;
        std::vector<vk2::ShaderModule> modules;
        modules.reserve(22);
        for (int i = 0; i < 22; ++i) {
            auto r = vk2::ShaderModule::fromFile(device_, fullBase + kFullShaderNames[i]);
            if (!r) {
                log::warn("rt-adapter", std::string("v2_full shader missing: ") + kFullShaderNames[i]
                          + " — falling back to v2_min");
                allLoaded = false;
                break;
            }
            modules.push_back(std::move(r.value()));
        }
        if (allLoaded) {
            shaderModules_ = std::move(modules);
            fullShadersLoaded_ = true;
            log::warn("rt-adapter", "Loaded experimental v2_full shader suite");
        }
    } else {
        log::warn("rt-adapter",
                  "Experimental v2_full auto-load disabled; using stable v2_min path "
                  "(set RADIANCE_V2_ENABLE_FULL_RT=1 to override)");
    }

    // Fallback: load v2_min 3-shader set
    if (!fullShadersLoaded_) {
        std::string minBase = services.resourceDir() + "/shaders/world/v2_min/";
        auto rgenR  = vk2::ShaderModule::fromFile(device_, minBase + "v2_world_rgen.spv");
        auto rmissR = vk2::ShaderModule::fromFile(device_, minBase + "v2_world_rmiss.spv");
        auto rchitR = vk2::ShaderModule::fromFile(device_, minBase + "v2_world_rchit.spv");
        if (!rgenR || !rmissR || !rchitR) {
            log::error("rt-adapter", "Failed to load v2_min RT shaders");
            return;
        }
        shaderModules_.clear();
        shaderModules_.push_back(std::move(rgenR.value()));
        shaderModules_.push_back(std::move(rmissR.value()));
        shaderModules_.push_back(std::move(rchitR.value()));
        log::info("rt-adapter", "Loaded v2_min shaders (3-stage fallback)");
    }

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

    // RT pipeline + SBT using the V1-compatible layout
    if (fullShadersLoaded_) {
        // --- Full 22-stage, 14-group pipeline (V1 SBT layout) ---
        vk2::RtPipelineDesc desc;
        desc.stages.resize(22);
        for (int i = 0; i < 22; ++i) {
            auto& s = desc.stages[i];
            s.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            s.stage = kFullShaderStages[i];
            s.module = shaderModules_[i].handle();
            s.pName = "main";
        }

        // 14 shader groups matching V1's SBT layout
        desc.groups.resize(14);

        // Group 0: GENERAL — raygen (stage 0)
        desc.groups[0] = {VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR, 0};
        // Group 1: GENERAL — world miss (stage 1)
        desc.groups[1] = {VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR, 1};
        // Group 2: GENERAL — hand miss (stage 2)
        desc.groups[2] = {VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR, 2};
        // Group 3: GENERAL — shadow miss (stage 3)
        desc.groups[3] = {VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR, 3};
        // Group 4: GENERAL — point light shadow miss (stage 18)
        desc.groups[4] = {VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR, 18};

        // Group 5: shadow hit — CHS=10(shadow_rchit), AHS=11(shadow_rahit)
        desc.groups[5] = {VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR,
                          VK_SHADER_UNUSED_KHR, 10, 11};
        // Group 6: world solid (opaque) — CHS=4(world_solid_transparent_rchit), AHS=UNUSED
        desc.groups[6] = {VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR,
                          VK_SHADER_UNUSED_KHR, 4};
        // Group 7: world transparent — CHS=4(world_solid_transparent_rchit), AHS=7(world_transparent_rahit)
        desc.groups[7] = {VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR,
                          VK_SHADER_UNUSED_KHR, 4, 7};
        // Group 8: world no-reflect — CHS=5(world_no_reflect_rchit), AHS=8(world_no_reflect_rahit)
        desc.groups[8] = {VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR,
                          VK_SHADER_UNUSED_KHR, 5, 8};
        // Group 9: world cloud — CHS=6(world_cloud_rchit), AHS=9(world_cloud_rahit)
        desc.groups[9] = {VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR,
                          VK_SHADER_UNUSED_KHR, 6, 9};
        // Group 10: boat water mask — CHS=12, AHS=13
        desc.groups[10] = {VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR,
                           VK_SHADER_UNUSED_KHR, 12, 13};
        // Group 11: end portal — CHS=14, AHS=15
        desc.groups[11] = {VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR,
                           VK_SHADER_UNUSED_KHR, 14, 15};
        // Group 12: end gateway — CHS=16, AHS=17
        desc.groups[12] = {VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR,
                           VK_SHADER_UNUSED_KHR, 16, 17};
        // Group 13: displaced block (procedural) — CHS=20, INT=19
        desc.groups[13] = {VK_RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_KHR,
                           VK_SHADER_UNUSED_KHR, 20, VK_SHADER_UNUSED_KHR, 19};

        desc.layout = layout_.pipelineLayout();
        desc.maxRecursionDepth = 3;

        auto r = vk2::RtPipeline::create(device_, services.device().physicalDevice(), desc);
        if (!r) { log::error("rt-adapter", "RT pipeline (full): " + r.error().message); return; }
        rtPipeline_ = std::move(r.value());

        // Build SBT: 1 rgen, 4 miss, 9 hit groups
        auto sr = sbt_.build(device_, services.device().vma(), services.device().physicalDevice(),
                             rtPipeline_, 1, 4, 9);
        if (!sr) { log::error("rt-adapter", "SBT (full): " + sr.error().message); return; }

    } else {
        // --- Minimal 3-stage, 3-group pipeline (v2_min fallback) ---
        vk2::RtPipelineDesc desc;

        VkPipelineShaderStageCreateInfo rgenStage{};
        rgenStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        rgenStage.stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
        rgenStage.module = shaderModules_[0].handle();
        rgenStage.pName = "main";

        VkPipelineShaderStageCreateInfo missStage{};
        missStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        missStage.stage = VK_SHADER_STAGE_MISS_BIT_KHR;
        missStage.module = shaderModules_[1].handle();
        missStage.pName = "main";

        VkPipelineShaderStageCreateInfo chitStage{};
        chitStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        chitStage.stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
        chitStage.module = shaderModules_[2].handle();
        chitStage.pName = "main";

        desc.stages = {rgenStage, missStage, chitStage};
        desc.groups.push_back({VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR, 0});
        desc.groups.push_back({VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR, 1});
        desc.groups.push_back({VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR,
                               VK_SHADER_UNUSED_KHR, 2});

        desc.layout = layout_.pipelineLayout();
        desc.maxRecursionDepth = 1;

        auto r = vk2::RtPipeline::create(device_, services.device().physicalDevice(), desc);
        if (!r) { log::error("rt-adapter", "RT pipeline (min): " + r.error().message); return; }
        rtPipeline_ = std::move(r.value());

        // Build SBT: 1 rgen, 1 miss, 1 hit
        auto sr = sbt_.build(device_, services.device().vma(), services.device().physicalDevice(),
                             rtPipeline_, 1, 1, 1);
        if (!sr) { log::error("rt-adapter", "SBT (min): " + sr.error().message); return; }
    }

    // --- SHARC resolve compute pipeline (push-constant-only, no descriptor sets) ---
    if (fullShadersLoaded_) {
        std::string spvPath = services.resourceDir() + "/shaders/world/v2_full/sharc_resolve_comp.spv";
        auto shaderR = vk2::ShaderModule::fromFile(device_, spvPath);
        if (shaderR) {
            sharcResolveShader_ = std::move(shaderR.value());

            // Pipeline layout: no descriptor sets, push constant = SharcResolvePC
            vk2::PipelineLayoutDesc pld;
            VkPushConstantRange pcr{};
            pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            pcr.offset = 0;
            pcr.size = sizeof(SharcResolvePC);
            pld.pushConstants.push_back(pcr);
            auto layoutR = vk2::PipelineLayout::create(device_, pld);
            if (layoutR) {
                sharcResolveLayout_ = std::move(layoutR.value());

                vk2::ComputePipelineDesc cpd;
                cpd.shaderModule = sharcResolveShader_.handle();
                cpd.layout = sharcResolveLayout_.handle();
                auto pipR = vk2::ComputePipeline::create(device_, cpd);
                if (pipR) {
                    sharcResolvePipeline_ = std::move(pipR.value());
                    sharcResolveReady_ = true;
                    log::info("rt-adapter", "SHARC resolve compute pipeline created");
                } else {
                    log::warn("rt-adapter", "SHARC resolve pipeline: " + pipR.error().message);
                }
            } else {
                log::warn("rt-adapter", "SHARC resolve layout: " + layoutR.error().message);
            }
        } else {
            log::info("rt-adapter", "SHARC resolve shader not found — compute pass disabled");
        }
    }

    // --- ReSTIR spatial reuse compute pipeline ---
    if (fullShadersLoaded_) {
        std::string spvPath = services.resourceDir() + "/shaders/world/v2_full/restir_spatial_comp.spv";
        auto shaderR = vk2::ShaderModule::fromFile(device_, spvPath);
        if (shaderR) {
            restirSpatialShader_ = std::move(shaderR.value());

            // Descriptor set layout: 4 storage image bindings matching the shader
            // binding 0: reservoirTemporalImage (readonly rgba32f)
            // binding 1: reservoirSpatialImage  (writeonly rgba32f)
            // binding 2: normalRoughnessImage   (readonly rgba16f)
            // binding 3: linearDepthImage        (readonly r16f)
            std::vector<VkDescriptorSetLayoutBinding> bindings(4);
            for (uint32_t i = 0; i < 4; ++i) {
                bindings[i].binding = i;
                bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                bindings[i].descriptorCount = 1;
                bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            }
            auto dslR = vk2::DescriptorSetLayout::create(device_, bindings);
            if (dslR) {
                restirSpatialDescLayout_ = std::move(dslR.value());

                // Pipeline layout: one descriptor set + RestirSpatialPC push constant
                vk2::PipelineLayoutDesc pld;
                pld.setLayouts.push_back(restirSpatialDescLayout_.handle());
                VkPushConstantRange pcr{};
                pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
                pcr.offset = 0;
                pcr.size = sizeof(RestirSpatialPC);
                pld.pushConstants.push_back(pcr);
                auto layoutR = vk2::PipelineLayout::create(device_, pld);
                if (layoutR) {
                    restirSpatialLayout_ = std::move(layoutR.value());

                    vk2::ComputePipelineDesc cpd;
                    cpd.shaderModule = restirSpatialShader_.handle();
                    cpd.layout = restirSpatialLayout_.handle();
                    auto pipR = vk2::ComputePipeline::create(device_, cpd);
                    if (pipR) {
                        restirSpatialPipeline_ = std::move(pipR.value());

                        // Descriptor allocator for the ReSTIR spatial pass
                        VkDescriptorPoolSize poolSize{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 4};
                        uint32_t fif = services.frame().framesInFlight();
                        auto allocR = restirSpatialDescAlloc_.init(device_, fif, {poolSize}, 1);
                        if (allocR) {
                            restirSpatialReady_ = true;
                            log::info("rt-adapter", "ReSTIR spatial compute pipeline created");
                        } else {
                            log::warn("rt-adapter", "ReSTIR spatial desc alloc: " + allocR.error().message);
                        }
                    } else {
                        log::warn("rt-adapter", "ReSTIR spatial pipeline: " + pipR.error().message);
                    }
                } else {
                    log::warn("rt-adapter", "ReSTIR spatial layout: " + layoutR.error().message);
                }
            } else {
                log::warn("rt-adapter", "ReSTIR spatial desc layout: " + dslR.error().message);
            }
        } else {
            log::info("rt-adapter", "ReSTIR spatial shader not found — compute pass disabled");
        }
    }

    // Create 1x1 white fallback texture for bindless array (set 0 binding 0).
    // When V1 shaders are loaded, they access textures[textureID] from binding 0.
    // We fill all 4096 slots with this fallback to prevent GPU faults until real
    // textures are available.
    {
        vk2::Image::Desc imgDesc{};
        imgDesc.width = 1;
        imgDesc.height = 1;
        imgDesc.format = VK_FORMAT_R8G8B8A8_UNORM;
        imgDesc.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        auto imgR = vk2::Image::create(device_, services.device().vma(), imgDesc);
        if (imgR) {
            fallbackWhiteImage_ = std::move(imgR.value());

            // Upload 1x1 white pixel via one-shot command buffer
            VkCommandBuffer oneShot = services.frame().beginOneShotCommandBuffer();
            if (oneShot != VK_NULL_HANDLE) {
                // Transition to TRANSFER_DST
                VkImageMemoryBarrier2 bar{};
                bar.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
                bar.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
                bar.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
                bar.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
                bar.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                bar.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                bar.image = fallbackWhiteImage_.handle();
                bar.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                VkDependencyInfo dep{};
                dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                dep.imageMemoryBarrierCount = 1;
                dep.pImageMemoryBarriers = &bar;
                vkCmdPipelineBarrier2(oneShot, &dep);

                // Clear to white
                VkClearColorValue white{};
                white.float32[0] = white.float32[1] = white.float32[2] = white.float32[3] = 1.0f;
                VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                vkCmdClearColorImage(oneShot, fallbackWhiteImage_.handle(),
                                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &white, 1, &range);

                // Transition to SHADER_READ_ONLY
                bar.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
                bar.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
                bar.dstStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
                bar.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
                bar.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                bar.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                vkCmdPipelineBarrier2(oneShot, &dep);

                services.frame().endOneShotCommandBuffer(oneShot);
                fallbackWhiteImage_.setLayout(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            }
        }

        // Create nearest sampler for fallback
        VkSamplerCreateInfo sci{};
        sci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sci.magFilter = VK_FILTER_NEAREST;
        sci.minFilter = VK_FILTER_NEAREST;
        sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        vkCreateSampler(device_, &sci, nullptr, &fallbackSampler_);
    }

    initialized_ = true;
    profileSlot_ = services.metrics().registerTimer("RayTracing");
    log::info("rt-adapter", std::string("RayTracingAdapter initialized (")
              + (fullShadersLoaded_ ? "full 22-stage" : "minimal 3-stage")
              + ", V1-compatible 4-set layout"
              + (sharcResolveReady_ ? ", SHARC-resolve" : "")
              + (restirSpatialReady_ ? ", ReSTIR-spatial" : "")
              + ")");
}

void RayTracingAdapter::configure(const ConfigSnapshot& config) {
    cachedConfig_ = config;
}

void RayTracingAdapter::execute(const FrameContext& ctx, const PassResources& resources) {
    if (!resources.resolved || resources.cmd == VK_NULL_HANDLE) return;
    if (resources.outputs.size() < OUTPUT_COUNT) return;
    for (uint32_t i = 0; i < OUTPUT_COUNT; ++i) {
        if (!resources.outputs[i].valid()) return;
    }

    VkCommandBuffer cmd = resources.cmd;

    // Begin per-pass GPU timer (no-op if not registered or GPU timing unavailable)
    if (profileSlot_ != UINT32_MAX)
        services_->metrics().beginNamedTimer(cmd, ctx.frameIndex, profileSlot_);

    // Resolve output indices (pass outputs are in the same order as kOutputSpecs).
    uint32_t outputIdx[OUTPUT_COUNT];
    for (uint32_t i = 0; i < OUTPUT_COUNT; ++i) {
        outputIdx[i] = resources.outputs[i].index;
    }

    auto& pool = services_->frame().resourcePool();
    uint32_t w = pool.image(outputIdx[0]).width();
    uint32_t h = pool.image(outputIdx[0]).height();

    // If not initialized or no valid TLAS, clear radiance to sky color and zero
    // all other output images (all in GENERAL layout per barrier planner).
    // Also block tracing if the BLAS-offsets buffer for this frame slot is not yet
    // populated — this can happen on the very first frame for a given slot, and
    // falling back to the 64-byte stub would cause out-of-bounds SSBO reads when
    // the TLAS has more than 16 instances, producing GPU page faults / DEVICE_LOST.
    //
    // Also block if the blasOffsetsBuffer for this slot was built in a prior frame
    // with FEWER instances than the current TLAS now has. In that window the shader
    // would read blasOffsets.offsets[instanceID] out of bounds, get a garbage BDA,
    // and dereference it — producing the recurring GPU page fault at fixed shader
    // offsets 0x12add00 / 0x12add60. Wait one more frame for tlas.rebuild() to
    // refresh this slot's buffers with the updated count before dispatching.
    uint32_t tlasInstanceCount = services_->tlas().instanceCount();
    uint32_t slotBdaCount    = services_->tlas().bdaArraySize(ctx.frameIndex)    / sizeof(uint64_t);
    uint32_t slotOffsetCount = services_->tlas().blasOffsetsSize(ctx.frameIndex) / sizeof(uint32_t);
    bool blasOffsetsFresh = (slotBdaCount    >= tlasInstanceCount)
                         && (slotOffsetCount >= tlasInstanceCount);

    // Grow biomeColorSSBO_ to cover all current TLAS instances before testing canTrace.
    // A 64-byte stub (4 entries) with 2050+ instances causes the shader to read
    // biomeColors.data[instanceID] far past the buffer end — GPU page fault / DEVICE_LOST.
    // resizeBiomeColors() is a no-op when the buffer is already large enough.
    if (tlasInstanceCount > 0) {
        services_->sceneRes().resizeBiomeColors(tlasInstanceCount);
    }
    bool biomeColorReady = (services_->sceneRes().biomeColorSSBOSize(ctx.frameIndex)
                            >= static_cast<VkDeviceSize>(tlasInstanceCount) * 16u);

    bool canTrace = initialized_ && services_->tlas().isValid()
                    && services_->tlas().handle(ctx.frameIndex) != VK_NULL_HANDLE
                    && services_->sceneRes().isInitialized()
                    && services_->tlas().blasOffsetsBuffer(ctx.frameIndex) != VK_NULL_HANDLE
                    && blasOffsetsFresh
                    && biomeColorReady;
    if (!canTrace) {
        if (!blasOffsetsFresh && services_->tlas().blasOffsetsBuffer(ctx.frameIndex) != VK_NULL_HANDLE) {
            log::info("rt-adapter",
                "Skipping RT dispatch: slot " + std::to_string(ctx.frameIndex & 1)
                + " bdaCount=" + std::to_string(slotBdaCount)
                + " offsetCount=" + std::to_string(slotOffsetCount)
                + " tlasInstances=" + std::to_string(tlasInstanceCount)
                + " — waiting for next rebuild");
        }
        VkImageSubresourceRange range = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        VkClearColorValue skyColor = {{0.4f, 0.6f, 0.9f, 1.0f}};
        VkClearColorValue zeroColor = {{0.0f, 0.0f, 0.0f, 0.0f}};
        for (uint32_t i = 0; i < OUTPUT_COUNT; ++i) {
            const VkClearColorValue& clr = (i == 0) ? skyColor : zeroColor;
            vkCmdClearColorImage(cmd, resources.resolved->images[outputIdx[i]],
                                 VK_IMAGE_LAYOUT_GENERAL, &clr, 1, &range);
        }
        if (profileSlot_ != UINT32_MAX)
            services_->metrics().endNamedTimer(cmd, ctx.frameIndex, profileSlot_);
        return;
    }

    // Allocate fresh descriptor sets for all 4 layouts
    descAllocator_.resetFrame(ctx.frameIndex);
    VkDescriptorSet sets[4] = {VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE};
    for (uint32_t i = 0; i < 4; ++i) {
        auto r = descAllocator_.allocate(layout_.setLayout(i), ctx.frameIndex);
        if (!r) {
            log::error("rt-adapter", "set " + std::to_string(i) + " alloc: " + r.error().message);
            if (profileSlot_ != UINT32_MAX)
                services_->metrics().endNamedTimer(cmd, ctx.frameIndex, profileSlot_);
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

    VkAccelerationStructureKHR tlas = tlasService.handle(ctx.frameIndex);
    VkWriteDescriptorSetAccelerationStructureKHR asWrite{};
    asWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
    asWrite.accelerationStructureCount = 1;
    asWrite.pAccelerationStructures = &tlas;

    VkDescriptorBufferInfo vtxBdaInfo{};
    vtxBdaInfo.buffer = tlasService.vertexBdaBuffer(ctx.frameIndex);
    vtxBdaInfo.offset = 0;
    vtxBdaInfo.range = tlasService.bdaArraySize(ctx.frameIndex);
    if (vtxBdaInfo.range == 0) vtxBdaInfo.range = VK_WHOLE_SIZE;

    VkDescriptorBufferInfo idxBdaInfo{};
    idxBdaInfo.buffer = tlasService.indexBdaBuffer(ctx.frameIndex);
    idxBdaInfo.offset = 0;
    idxBdaInfo.range = tlasService.bdaArraySize(ctx.frameIndex);
    if (idxBdaInfo.range == 0) idxBdaInfo.range = VK_WHOLE_SIZE;

    // Previous-frame BDA and transforms for motion vectors (set 1, bindings 4-6).
    // lastXxx buffers are now double-buffered by slot — pass ctx.frameIndex so we
    // read the slot that was written this frame (safe: fence[slot] was waited above).
    VkDescriptorBufferInfo lastVtxBdaInfo{};
    lastVtxBdaInfo.buffer = tlasService.lastVertexBdaBuffer(ctx.frameIndex);
    lastVtxBdaInfo.offset = 0;
    lastVtxBdaInfo.range = tlasService.lastBdaArraySize(ctx.frameIndex);
    if (lastVtxBdaInfo.range == 0) lastVtxBdaInfo.range = VK_WHOLE_SIZE;

    VkDescriptorBufferInfo lastIdxBdaInfo{};
    lastIdxBdaInfo.buffer = tlasService.lastIndexBdaBuffer(ctx.frameIndex);
    lastIdxBdaInfo.offset = 0;
    lastIdxBdaInfo.range = tlasService.lastBdaArraySize(ctx.frameIndex);
    if (lastIdxBdaInfo.range == 0) lastIdxBdaInfo.range = VK_WHOLE_SIZE;

    VkDescriptorBufferInfo lastXformInfo{};
    lastXformInfo.buffer = tlasService.lastTransformsBuffer(ctx.frameIndex);
    lastXformInfo.offset = 0;
    lastXformInfo.range = tlasService.lastTransformsSize(ctx.frameIndex);
    if (lastXformInfo.range == 0) lastXformInfo.range = VK_WHOLE_SIZE;

    VkDescriptorBufferInfo worldUboInfo{};
    worldUboInfo.buffer = sceneRes.worldUBO(ctx.frameIndex);
    worldUboInfo.offset = 0;
    worldUboInfo.range = sceneRes.worldUBOSize();

    VkDescriptorBufferInfo lastWorldUboInfo{};
    lastWorldUboInfo.buffer = sceneRes.lastWorldUBO(ctx.frameIndex);
    lastWorldUboInfo.offset = 0;
    lastWorldUboInfo.range = sceneRes.worldUBOSize();

    VkDescriptorBufferInfo skyUboInfo{};
    skyUboInfo.buffer = sceneRes.skyUBO(ctx.frameIndex);
    skyUboInfo.offset = 0;
    skyUboInfo.range = sceneRes.skyUBOSize();

    // --- Set 1 (geometry) additional buffer infos ---
    // Prefer real BLAS offsets from TlasService; fall back to SceneRes stub if not yet built.
    VkDescriptorBufferInfo blasOffsetsInfo{};
    if (tlasService.blasOffsetsBuffer(ctx.frameIndex) != VK_NULL_HANDLE) {
        blasOffsetsInfo.buffer = tlasService.blasOffsetsBuffer(ctx.frameIndex);
        blasOffsetsInfo.offset = 0;
        blasOffsetsInfo.range = tlasService.blasOffsetsSize(ctx.frameIndex);
        if (blasOffsetsInfo.range == 0) blasOffsetsInfo.range = VK_WHOLE_SIZE;
    } else {
        blasOffsetsInfo.buffer = sceneRes.blasOffsetsSSBO();
        blasOffsetsInfo.offset = 0;
        blasOffsetsInfo.range = VK_WHOLE_SIZE;
    }

    VkDescriptorBufferInfo texMapInfo{};
    texMapInfo.buffer = sceneRes.textureMappingSSBO(ctx.frameIndex);
    texMapInfo.offset = 0;
    texMapInfo.range = VK_WHOLE_SIZE;

    VkDescriptorBufferInfo areaLightInfo{};
    areaLightInfo.buffer = sceneRes.areaLightSSBO(ctx.frameIndex);
    areaLightInfo.offset = 0;
    areaLightInfo.range = VK_WHOLE_SIZE;

    VkDescriptorBufferInfo tileLightInfo{};
    tileLightInfo.buffer = sceneRes.tileLightSSBO(ctx.frameIndex);
    tileLightInfo.offset = 0;
    tileLightInfo.range = VK_WHOLE_SIZE;

    VkDescriptorBufferInfo blenderPbrInfo{};
    blenderPbrInfo.buffer = sceneRes.blenderPbrSSBO(ctx.frameIndex);
    blenderPbrInfo.offset = 0;
    blenderPbrInfo.range = VK_WHOLE_SIZE;

    VkDescriptorBufferInfo materialClassInfo{};
    materialClassInfo.buffer = sceneRes.materialClassSSBO();
    materialClassInfo.offset = 0;
    materialClassInfo.range = VK_WHOLE_SIZE;

    VkDescriptorBufferInfo displacedFaceInfo{};
    displacedFaceInfo.buffer = sceneRes.displacedFaceSSBO(ctx.frameIndex);
    displacedFaceInfo.offset = 0;
    displacedFaceInfo.range = VK_WHOLE_SIZE;

    VkDescriptorBufferInfo spriteRegistryInfo{};
    // Use TextureService's real SSBO when finalized, fall back to SceneResourceService stub.
    auto& texServiceForRegistry = services_->texture();
    if (texServiceForRegistry.isFinalized()) {
        spriteRegistryInfo.buffer = texServiceForRegistry.spriteRegistrySSBO();
    } else {
        spriteRegistryInfo.buffer = sceneRes.spriteRegistrySSBO(ctx.frameIndex);
    }
    spriteRegistryInfo.offset = 0;
    spriteRegistryInfo.range = VK_WHOLE_SIZE;

    // --- Set 2 (uniforms) additional infos ---
    VkDescriptorImageInfo energyLutInfo{};
    energyLutInfo.sampler = sceneRes.energyLUTSampler();
    energyLutInfo.imageView = sceneRes.energyLUTView();
    energyLutInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDescriptorBufferInfo blueNoiseSobolInfo{};
    blueNoiseSobolInfo.buffer = sceneRes.blueNoiseSobol();
    blueNoiseSobolInfo.offset = 0;
    blueNoiseSobolInfo.range = VK_WHOLE_SIZE;

    VkDescriptorBufferInfo blueNoiseScramblingInfo{};
    blueNoiseScramblingInfo.buffer = sceneRes.blueNoiseScrambling();
    blueNoiseScramblingInfo.offset = 0;
    blueNoiseScramblingInfo.range = VK_WHOLE_SIZE;

    // All 34 output image descriptors
    VkDescriptorImageInfo outImgInfos[OUTPUT_COUNT]{};
    for (uint32_t i = 0; i < OUTPUT_COUNT; ++i) {
        outImgInfos[i].imageView = resources.resolved->views[outputIdx[i]];
        outImgInfos[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    }

    // Writes:
    //   Set 0: atmo LUT, atmo cube, sprite albedo, specular, normal (cond.)  (up to 5)
    //   Set 1: TLAS, blasOffsets, vtxBDA, idxBDA, lastVtxBDA, lastIdxBDA,
    //          lastXform, texMap, areaLight, tileLight, biomeColors, blenderPbr,
    //          materialClass, displacedFace, spriteRegistry                   (15)
    //   Set 2: WorldUBO, LastWorldUBO, SkyUBO, energyLUT, blueNoiseSobol,
    //          blueNoiseScrambling                                            (6)
    //   Set 3: all 34 output images                                          (34)
    uint32_t wi = 0;
    VkWriteDescriptorSet writes[60]{};  // 5 + 15 + 6 + 34 = 60 max

    // --- Set 0 (textures) — atmosphere LUT + cubemap (conditional on adapter) ---
    VkDescriptorImageInfo atmoLutInfo{};
    VkDescriptorImageInfo atmoCubeInfo{};
    if (atmosphere_ && atmosphere_->isInitialized()) {
        atmoLutInfo.sampler     = atmosphere_->sampler();
        atmoLutInfo.imageView   = atmosphere_->lutView();
        atmoLutInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        atmoCubeInfo.sampler     = atmosphere_->sampler();
        atmoCubeInfo.imageView   = atmosphere_->cubeView();
        atmoCubeInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        writes[wi].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[wi].dstSet = sets[rt_layout::SET_TEXTURES];
        writes[wi].dstBinding = rt_layout::TEX_BINDING_ATMO_LUT;
        writes[wi].descriptorCount = 1;
        writes[wi].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[wi].pImageInfo = &atmoLutInfo;
        ++wi;

        writes[wi].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[wi].dstSet = sets[rt_layout::SET_TEXTURES];
        writes[wi].dstBinding = rt_layout::TEX_BINDING_ATMO_CUBE;
        writes[wi].descriptorCount = 1;
        writes[wi].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[wi].pImageInfo = &atmoCubeInfo;
        ++wi;
    }

    // --- Set 0 (textures) — sprite array bindings (conditional on TextureService finalized) ---
    auto& texService = services_->texture();
    VkDescriptorImageInfo spriteAlbedoImgInfo{};
    VkDescriptorImageInfo spriteSpecularImgInfo{};
    VkDescriptorImageInfo spriteNormalImgInfo{};
    if (texService.isFinalized()) {
        spriteAlbedoImgInfo.sampler = texService.arrayNearestSampler();
        spriteAlbedoImgInfo.imageView = texService.albedoArrayView();
        spriteAlbedoImgInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        spriteSpecularImgInfo.sampler = texService.arrayLinearSampler();
        spriteSpecularImgInfo.imageView = texService.specularArrayView();
        spriteSpecularImgInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        spriteNormalImgInfo.sampler = texService.arrayLinearSampler();
        spriteNormalImgInfo.imageView = texService.normalArrayView();
        spriteNormalImgInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        writes[wi].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[wi].dstSet = sets[rt_layout::SET_TEXTURES];
        writes[wi].dstBinding = rt_layout::TEX_BINDING_SPRITE_ALBEDO;
        writes[wi].descriptorCount = 1;
        writes[wi].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[wi].pImageInfo = &spriteAlbedoImgInfo;
        ++wi;

        writes[wi].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[wi].dstSet = sets[rt_layout::SET_TEXTURES];
        writes[wi].dstBinding = rt_layout::TEX_BINDING_SPRITE_SPECULAR;
        writes[wi].descriptorCount = 1;
        writes[wi].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[wi].pImageInfo = &spriteSpecularImgInfo;
        ++wi;

        writes[wi].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[wi].dstSet = sets[rt_layout::SET_TEXTURES];
        writes[wi].dstBinding = rt_layout::TEX_BINDING_SPRITE_NORMAL;
        writes[wi].descriptorCount = 1;
        writes[wi].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[wi].pImageInfo = &spriteNormalImgInfo;
        ++wi;
    }

    // --- Set 1 (geometry) ---
    writes[wi].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[wi].pNext = &asWrite;
    writes[wi].dstSet = sets[rt_layout::SET_GEOMETRY];
    writes[wi].dstBinding = rt_layout::GEO_BINDING_TLAS;
    writes[wi].descriptorCount = 1;
    writes[wi].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    ++wi;

    writes[wi].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[wi].dstSet = sets[rt_layout::SET_GEOMETRY];
    writes[wi].dstBinding = rt_layout::GEO_BINDING_BLAS_OFFSETS;
    writes[wi].descriptorCount = 1;
    writes[wi].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[wi].pBufferInfo = &blasOffsetsInfo;
    ++wi;

    // vtxBDA / idxBDA — record their indices so we can skip them if null
    uint32_t vtxBdaWriteIdx = wi;
    writes[wi].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[wi].dstSet = sets[rt_layout::SET_GEOMETRY];
    writes[wi].dstBinding = rt_layout::GEO_BINDING_VTX_ADDRS;
    writes[wi].descriptorCount = 1;
    writes[wi].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[wi].pBufferInfo = &vtxBdaInfo;
    ++wi;

    writes[wi].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[wi].dstSet = sets[rt_layout::SET_GEOMETRY];
    writes[wi].dstBinding = rt_layout::GEO_BINDING_IDX_ADDRS;
    writes[wi].descriptorCount = 1;
    writes[wi].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[wi].pBufferInfo = &idxBdaInfo;
    ++wi;

    // Previous-frame BDA and transforms (bindings 4, 5, 6).
    // On frame 1 (first world frame) lastXxx buffers are null — no previous frame exists.
    // We MUST always write these bindings: descriptor sets are freshly allocated each frame,
    // so unwritten bindings contain uninitialized GPU memory. The shader unconditionally reads
    // them for motion vector computation → GPU page fault / DEVICE_LOST.
    // Fix: substitute the sceneRes stub SSBO (64 bytes, zeroed) when the real buffer is null.
    // The shader sees all-zero motion vectors for one frame, which is correct and harmless.
    VkBuffer stubSSBO = sceneRes.blasOffsetsSSBO(); // any valid 64-byte zeroed stub buffer
    if (lastVtxBdaInfo.buffer == VK_NULL_HANDLE) {
        lastVtxBdaInfo.buffer = stubSSBO;
        lastVtxBdaInfo.range  = VK_WHOLE_SIZE;
    }
    if (lastIdxBdaInfo.buffer == VK_NULL_HANDLE) {
        lastIdxBdaInfo.buffer = stubSSBO;
        lastIdxBdaInfo.range  = VK_WHOLE_SIZE;
    }
    if (lastXformInfo.buffer == VK_NULL_HANDLE) {
        lastXformInfo.buffer = stubSSBO;
        lastXformInfo.range  = VK_WHOLE_SIZE;
    }

    writes[wi].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[wi].dstSet = sets[rt_layout::SET_GEOMETRY];
    writes[wi].dstBinding = rt_layout::GEO_BINDING_LAST_VTX_ADDRS;
    writes[wi].descriptorCount = 1;
    writes[wi].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[wi].pBufferInfo = &lastVtxBdaInfo;
    ++wi;

    writes[wi].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[wi].dstSet = sets[rt_layout::SET_GEOMETRY];
    writes[wi].dstBinding = rt_layout::GEO_BINDING_LAST_IDX_ADDRS;
    writes[wi].descriptorCount = 1;
    writes[wi].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[wi].pBufferInfo = &lastIdxBdaInfo;
    ++wi;

    writes[wi].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[wi].dstSet = sets[rt_layout::SET_GEOMETRY];
    writes[wi].dstBinding = rt_layout::GEO_BINDING_LAST_TRANSFORMS;
    writes[wi].descriptorCount = 1;
    writes[wi].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[wi].pBufferInfo = &lastXformInfo;
    ++wi;

    writes[wi].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[wi].dstSet = sets[rt_layout::SET_GEOMETRY];
    writes[wi].dstBinding = rt_layout::GEO_BINDING_TEXTURE_MAPPING;
    writes[wi].descriptorCount = 1;
    writes[wi].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[wi].pBufferInfo = &texMapInfo;
    ++wi;

    writes[wi].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[wi].dstSet = sets[rt_layout::SET_GEOMETRY];
    writes[wi].dstBinding = rt_layout::GEO_BINDING_AREA_LIGHT;
    writes[wi].descriptorCount = 1;
    writes[wi].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[wi].pBufferInfo = &areaLightInfo;
    ++wi;

    writes[wi].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[wi].dstSet = sets[rt_layout::SET_GEOMETRY];
    writes[wi].dstBinding = rt_layout::GEO_BINDING_TILE_LIGHT;
    writes[wi].descriptorCount = 1;
    writes[wi].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[wi].pBufferInfo = &tileLightInfo;
    ++wi;

    VkDescriptorBufferInfo biomeColorInfo{};
    biomeColorInfo.buffer = sceneRes.biomeColorSSBO(ctx.frameIndex);
    biomeColorInfo.offset = 0;
    biomeColorInfo.range = VK_WHOLE_SIZE;
    writes[wi].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[wi].dstSet = sets[rt_layout::SET_GEOMETRY];
    writes[wi].dstBinding = rt_layout::GEO_BINDING_BIOME_COLORS;
    writes[wi].descriptorCount = 1;
    writes[wi].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[wi].pBufferInfo = &biomeColorInfo;
    ++wi;

    writes[wi].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[wi].dstSet = sets[rt_layout::SET_GEOMETRY];
    writes[wi].dstBinding = rt_layout::GEO_BINDING_BLENDER_PBR;
    writes[wi].descriptorCount = 1;
    writes[wi].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[wi].pBufferInfo = &blenderPbrInfo;
    ++wi;

    writes[wi].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[wi].dstSet = sets[rt_layout::SET_GEOMETRY];
    writes[wi].dstBinding = rt_layout::GEO_BINDING_MATERIAL_CLASS;
    writes[wi].descriptorCount = 1;
    writes[wi].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[wi].pBufferInfo = &materialClassInfo;
    ++wi;

    writes[wi].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[wi].dstSet = sets[rt_layout::SET_GEOMETRY];
    writes[wi].dstBinding = rt_layout::GEO_BINDING_DISPLACED_FACE;
    writes[wi].descriptorCount = 1;
    writes[wi].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[wi].pBufferInfo = &displacedFaceInfo;
    ++wi;

    writes[wi].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[wi].dstSet = sets[rt_layout::SET_GEOMETRY];
    writes[wi].dstBinding = rt_layout::GEO_BINDING_SPRITE_REGISTRY;
    writes[wi].descriptorCount = 1;
    writes[wi].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[wi].pBufferInfo = &spriteRegistryInfo;
    ++wi;

    // --- Set 2 (uniforms) ---
    writes[wi].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[wi].dstSet = sets[rt_layout::SET_UNIFORMS];
    writes[wi].dstBinding = rt_layout::UBO_BINDING_WORLD;
    writes[wi].descriptorCount = 1;
    writes[wi].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writes[wi].pBufferInfo = &worldUboInfo;
    ++wi;

    writes[wi].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[wi].dstSet = sets[rt_layout::SET_UNIFORMS];
    writes[wi].dstBinding = rt_layout::UBO_BINDING_LAST_WORLD;
    writes[wi].descriptorCount = 1;
    writes[wi].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writes[wi].pBufferInfo = &lastWorldUboInfo;
    ++wi;

    writes[wi].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[wi].dstSet = sets[rt_layout::SET_UNIFORMS];
    writes[wi].dstBinding = rt_layout::UBO_BINDING_SKY;
    writes[wi].descriptorCount = 1;
    writes[wi].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writes[wi].pBufferInfo = &skyUboInfo;
    ++wi;

    writes[wi].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[wi].dstSet = sets[rt_layout::SET_UNIFORMS];
    writes[wi].dstBinding = rt_layout::UBO_BINDING_ENERGY_LUT;
    writes[wi].descriptorCount = 1;
    writes[wi].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[wi].pImageInfo = &energyLutInfo;
    ++wi;

    writes[wi].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[wi].dstSet = sets[rt_layout::SET_UNIFORMS];
    writes[wi].dstBinding = rt_layout::UBO_BINDING_BLUE_NOISE_SOBOL;
    writes[wi].descriptorCount = 1;
    writes[wi].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[wi].pBufferInfo = &blueNoiseSobolInfo;
    ++wi;

    writes[wi].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[wi].dstSet = sets[rt_layout::SET_UNIFORMS];
    writes[wi].dstBinding = rt_layout::UBO_BINDING_BLUE_NOISE_SCRAMBLING;
    writes[wi].descriptorCount = 1;
    writes[wi].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[wi].pBufferInfo = &blueNoiseScramblingInfo;
    ++wi;

    // --- Set 3 (all 34 output images) ---
    for (uint32_t i = 0; i < OUTPUT_COUNT; ++i) {
        writes[wi].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[wi].dstSet = sets[rt_layout::SET_OUTPUTS];
        writes[wi].dstBinding = kOutputSpecs[i].binding;
        writes[wi].descriptorCount = 1;
        writes[wi].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[wi].pImageInfo = &outImgInfos[i];
        ++wi;
    }

    // Skip current-frame vertex/index BDA writes if there's nothing to write yet (no chunks).
    // Compact the array by removing the 2 current-frame BDA entries.
    uint32_t writeCount = wi;
    if (vtxBdaInfo.buffer == VK_NULL_HANDLE || idxBdaInfo.buffer == VK_NULL_HANDLE) {
        uint32_t dst = vtxBdaWriteIdx;
        for (uint32_t src = vtxBdaWriteIdx + 2; src < writeCount; ++src) {
            writes[dst++] = writes[src];
        }
        writeCount -= 2;
    }

    vkUpdateDescriptorSets(device_, writeCount, writes, 0, nullptr);

    // Set 0 is freshly allocated every frame, so binding 0 must be rewritten for
    // each new descriptor set. Leaving it unwritten makes the next frame trace
    // against uninitialized bindless texture descriptors.
    if (fallbackWhiteImage_.handle() != VK_NULL_HANDLE && fallbackSampler_ != VK_NULL_HANDLE) {
        std::vector<VkDescriptorImageInfo> bindlessInfos(rt_layout::TEX_BINDLESS_COUNT);
        for (auto& info : bindlessInfos) {
            info.sampler = fallbackSampler_;
            info.imageView = fallbackWhiteImage_.defaultView();
            info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        }
        VkWriteDescriptorSet bindlessWrite{};
        bindlessWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        bindlessWrite.dstSet = sets[rt_layout::SET_TEXTURES];
        bindlessWrite.dstBinding = rt_layout::TEX_BINDING_BINDLESS;
        bindlessWrite.dstArrayElement = 0;
        bindlessWrite.descriptorCount = rt_layout::TEX_BINDLESS_COUNT;
        bindlessWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindlessWrite.pImageInfo = bindlessInfos.data();
        vkUpdateDescriptorSets(device_, 1, &bindlessWrite, 0, nullptr);
        if (!bindlessFallbackWritten_) {
            bindlessFallbackWritten_ = true;
            log::info("rt-adapter", "Initialized bindless fallback textures for RT descriptor set 0");
        }
    }

    // Push constants — V1-compatible 160 bytes, populated from ConfigSnapshot.
    if (ctx.config) configure(*ctx.config);
    RtPushConstants pc{};
    buildPushConstants(pc, ctx);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, rtPipeline_.handle());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, layout_.pipelineLayout(),
                            0, 4, sets, 0, nullptr);
    vkCmdPushConstants(cmd, layout_.pipelineLayout(),
                       VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR |
                       VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR |
                       VK_SHADER_STAGE_INTERSECTION_BIT_KHR,
                       0, sizeof(pc), &pc);

    // Structured diagnostic event: rt_dispatch (gated on DiagFlags::RT).
    // Read g_effectiveDiagFlags (diagFlags | level-implied flags) so diagLevel=3
    // activates this event without requiring an explicit diagFlags value.
    if ((engine::g_effectiveDiagFlags.load(std::memory_order_relaxed) & DiagFlags::RT) != 0) {
        log::event("rt-adapter", "rt_dispatch", {
            {"frame",          std::to_string(ctx.frameNumber)},
            {"width",          std::to_string(w)},
            {"height",         std::to_string(h)},
            {"numBounces",     std::to_string(pc.numRayBounces)},
            {"flags",          std::to_string(pc.flags)},
            {"areaLightCount", std::to_string(pc.areaLightCount)},
            {"shadowSoftness", std::to_string(pc.shadowSoftness)},
            {"risCandidates",  std::to_string(pc.risCandidates)},
            {"preExposure",    std::to_string(pc.preExposure)},
            {"temporalMClamp", std::to_string(pc.temporalMClamp)},
            {"wClamp",         std::to_string(pc.wClamp)}
        });
    }

    // Pre-dispatch diagnostic: log counts that could cause OOB shader reads.
    // slotBdaCount, slotOffsetCount, biomeColorSSBOSize are all checked above in
    // canTrace, but logging here confirms the values that actually reach the GPU.
    log::info("rt-adapter",
        "RT dispatch slot=" + std::to_string(ctx.frameIndex & 1)
        + " tlas=" + std::to_string(tlasInstanceCount)
        + " bda=" + std::to_string(slotBdaCount)
        + " offsets=" + std::to_string(slotOffsetCount)
        + " biomeBytes=" + std::to_string(services_->sceneRes().biomeColorSSBOSize(ctx.frameIndex))
        + " " + std::to_string(w) + "x" + std::to_string(h));

    nvInsertCheckpoint(cmd,
                       (engine::g_effectiveDiagFlags.load(std::memory_order_relaxed) & DiagFlags::CRASH) != 0,
                       services_->device().caps().nvCheckpoints,
                       "RT_TRACE_START");
    vkCmdTraceRaysKHR(cmd,
                      &sbt_.rgenRegion(), &sbt_.missRegion(),
                      &sbt_.hitRegion(), &sbt_.callableRegion(),
                      w, h, 1);

    // --- Post-RT compute passes (SHARC resolve + ReSTIR spatial) ---
    dispatchComputePasses(cmd, resources, w, h, ctx);

    // End per-pass GPU timer
    if (profileSlot_ != UINT32_MAX)
        services_->metrics().endNamedTimer(cmd, ctx.frameIndex, profileSlot_);
}

void RayTracingAdapter::buildPushConstants(RtPushConstants& pc, const FrameContext& ctx) const {
    const auto& cfg = cachedConfig_.data;

    // --- Core RT fields ---
    pc.numRayBounces = fullShadersLoaded_ ? static_cast<int32_t>(cfg.rayBounces) : 1;

    // Flags bitfield — mirrors V1 RayTracingModule flag assignment
    int32_t flags = 0;
    if (cfg.simplifiedIndirect)  flags |= (1 << 0);
    if (cfg.areaLightsEnabled)   flags |= (1 << 1);
    if (cfg.restirEnabled)       flags |= (1 << 2);
    if (cfg.restirSimplifiedBRDF) flags |= (1 << 3);
    if (cfg.restirBounceEnabled) flags |= (1 << 4);
    if (cfg.sharcEnabled && services_->sceneRes().sharcBuffersValid())
                                 flags |= (1 << 5);
    if (cfg.noiseLOD)            flags |= (1 << 6);
    if (cfg.multiScatterGGX)     flags |= (1 << 7);
    if (cfg.eonDiffuse)          flags |= (1 << 8);
    // bits 9-15: V1 booleans not yet in V2 config (beerLawShadows, noEmissionClamp,
    //   physicalSunDisk, noHandAmbient, SER, serHints, entityNormals) — left off
    pc.flags = flags;

    // Area lights — read count from scene resource service (updated by CmdAreaLightUpload)
    pc.areaLightCount = static_cast<int32_t>(services_->sceneRes().areaLightCount());

    pc.shadowSoftness      = cfg.shadowSoftness;
    pc.risCandidates       = static_cast<int32_t>(cfg.restirCandidates);
    pc.temporalMClamp      = static_cast<int32_t>(cfg.restirTemporalMClamp);
    pc.wClamp              = static_cast<int32_t>(cfg.restirWClamp);

    // Pre-exposure: auto-exposure feedback from ToneMappingAdapter (1-frame delayed)
    pc.preExposure = preExposure_;

    // --- POM fields ---
    pc.pomHeightScale    = cfg.pomEnabled ? cfg.pomHeightScale : 0.0f;
    pc.pomSteps          = static_cast<int32_t>(cfg.pomSteps);
    pc.pomRefinement     = static_cast<int32_t>(cfg.pomRefinement);
    pc.pomFadeDistance    = cfg.pomFadeDistance;

    // --- Color / Noise ---
    pc.colorExpansion = cfg.colorExpansion;
    pc.blueNoiseFrame = static_cast<uint32_t>(ctx.frameNumber);

    // --- SHARC radiance cache push constants ---
    auto& scRes = services_->sceneRes();
    if (cfg.sharcEnabled && scRes.sharcBuffersValid()) {
        pc.sharcHashEntries       = scRes.sharcHashEntriesBDA();
        pc.sharcAccumulation      = scRes.sharcAccumulationBDA();
        pc.sharcResolved          = scRes.sharcResolvedBDA();
        pc.sharcCameraX           = ctx.camera.posX;
        pc.sharcCameraY           = ctx.camera.posY;
        pc.sharcCameraZ           = ctx.camera.posZ;
        pc.sharcSceneScale        = cfg.sharcSceneScale;
        pc.sharcCapacity          = SceneResourceService::SHARC_CAPACITY;
        pc.sharcRadianceScale     = 1.0f;
        pc.sharcFrameIndex        = sharcFrameCounter_;
        pc.sharcRoughnessThreshold = cfg.sharcRoughnessThreshold;
        pc.sharcUpdateBlockSize   = 4;
        pc.sharcUpdateBounces     = 4;
    } else {
        pc.sharcHashEntries       = 0;
        pc.sharcAccumulation      = 0;
        pc.sharcResolved          = 0;
        pc.sharcCameraX           = 0.0f;
        pc.sharcCameraY           = 0.0f;
        pc.sharcCameraZ           = 0.0f;
        pc.sharcSceneScale        = 4.0f;
        pc.sharcCapacity          = SceneResourceService::SHARC_CAPACITY;
        pc.sharcRadianceScale     = 1.0f;
        pc.sharcFrameIndex        = 0;
        pc.sharcRoughnessThreshold = 0.5f;
        pc.sharcUpdateBlockSize   = 4;
        pc.sharcUpdateBounces     = 4;
    }

    // --- Offline accumulation ---
    // accumFrameCount_ and offlineEnabled_ are set each frame by EngineApp::processScene()
    // via setAccumState(). When enabled, the RT shader performs Welford accumulation
    // into its accumBufferImage using N = accumFrameCount_ as the running frame index.
    // offlineBounces_ overrides rayBounces when non-zero (higher quality offline path).
    pc.offlineFlags    = offlineEnabled_ ? 1u : 0u;
    pc.accumFrameCount = offlineEnabled_ ? accumFrameCount_ : 0u;
    pc.aperture        = offlineEnabled_ ? aperture_        : 0.0f;
    pc.focalDistance    = offlineEnabled_ ? focalDistance_   : 10.0f;
    // Override bounce depth for offline mode when a dedicated offline bounce count is set.
    if (offlineEnabled_ && offlineBounces_ > 0u) {
        pc.numRayBounces = static_cast<int32_t>(offlineBounces_);
    }

    // --- Material / displacement BDA ---
    // Wire buffer device addresses for material class mapping and displaced face data.
    // The buffers are stub-sized (64 bytes) until real data is uploaded via bridge,
    // but the BDAs must be non-zero so shaders can safely read the (empty) SSBO.
    pc.materialClassAddr     = services_->sceneRes().materialClassMappingBDA();
    pc.displacedFaceDataAddr = services_->sceneRes().displacedFaceBDA(ctx.frameIndex);
    // V2 does not yet feed displaced face geometry through the upload pipeline;
    // count stays 0 and shaders will skip displacement code paths.
    pc.displacedFaceCount    = 0;
    // displacementQuality: 0=Off. No V2 config field yet — displacement is off by default.
    pc.displacementQuality   = 0;
}

void RayTracingAdapter::dispatchComputePasses(VkCommandBuffer cmd, const PassResources& resources,
                                               uint32_t w, uint32_t h, const FrameContext& ctx) {
    if (!fullShadersLoaded_) {
        return;
    }

    const auto& cfg = cachedConfig_.data;
    auto& scRes = services_->sceneRes();

    // --------------- SHARC Resolve ---------------
    // Runs after main RT trace. Resolves accumulated radiance in the SHARC hash map.
    // The main RT trace writes into the accumulation buffer via BDA; this pass reads
    // accumulation and writes resolved data. Without the SHARC update RT pass, the
    // accumulation buffer stays zeroed so resolve is a safe no-op (all entries stale).
    if (sharcResolveReady_ && cfg.sharcEnabled && scRes.sharcBuffersValid()) {
        // Barrier: RT writes to SHARC buffers → compute reads
        VkMemoryBarrier2 memBar{};
        memBar.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        memBar.srcStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
        memBar.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
        memBar.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        memBar.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
        VkDependencyInfo dep{};
        dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.memoryBarrierCount = 1;
        dep.pMemoryBarriers = &memBar;
        vkCmdPipelineBarrier2(cmd, &dep);

        SharcResolvePC pc{};
        pc.cameraPositionPrevX = prevCameraX_;
        pc.cameraPositionPrevY = prevCameraY_;
        pc.cameraPositionPrevZ = prevCameraZ_;
        pc.accumulationFrameNum = 16;   // frames to accumulate before resolve
        pc.staleFrameNumMax = 128;      // evict entries not updated for this many frames
        pc.frameIndex = sharcFrameCounter_;
        pc.hashEntriesBDA = scRes.sharcHashEntriesBDA();
        pc.accumulationBDA = scRes.sharcAccumulationBDA();
        pc.resolvedBDA = scRes.sharcResolvedBDA();
        pc.cameraX = ctx.camera.posX;
        pc.cameraY = ctx.camera.posY;
        pc.cameraZ = ctx.camera.posZ;
        pc.sceneScale = cfg.sharcSceneScale;
        pc.capacity = SceneResourceService::SHARC_CAPACITY;
        pc.radianceScale = 1.0f;
        pc.enableAntiFirefly = 1;

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, sharcResolvePipeline_.handle());
        vkCmdPushConstants(cmd, sharcResolveLayout_.handle(), VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(pc), &pc);

        // One thread per hash entry, workgroup size = 64
        uint32_t groups = (SceneResourceService::SHARC_CAPACITY + 63) / 64;
        vkCmdDispatch(cmd, groups, 1, 1);

        // Update prev camera for next frame
        prevCameraX_ = ctx.camera.posX;
        prevCameraY_ = ctx.camera.posY;
        prevCameraZ_ = ctx.camera.posZ;
        sharcFrameCounter_++;
    }

    // --------------- ReSTIR Spatial Reuse ---------------
    // Reads reservoir_current (binding 14), writes reservoir_previous (binding 15).
    // Also reads normal+roughness (binding 3) and linear depth (binding 5).
    if (restirSpatialReady_ && cfg.restirEnabled && cfg.restirSpatialTaps > 0) {
        // Barrier: RT writes to storage images → compute reads
        VkMemoryBarrier2 memBar{};
        memBar.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        memBar.srcStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
        memBar.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
        memBar.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        memBar.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
        VkDependencyInfo dep{};
        dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.memoryBarrierCount = 1;
        dep.pMemoryBarriers = &memBar;
        vkCmdPipelineBarrier2(cmd, &dep);

        // Allocate descriptor set for this frame
        restirSpatialDescAlloc_.resetFrame(ctx.frameIndex);
        auto dsR = restirSpatialDescAlloc_.allocate(restirSpatialDescLayout_.handle(), ctx.frameIndex);
        if (!dsR) {
            log::warn("rt-adapter", "ReSTIR spatial desc alloc failed: " + dsR.error().message);
            return;
        }
        VkDescriptorSet ds = dsR.value();

        // Resolve output image indices from the pass resources
        uint32_t reservoirCurrentIdx = resources.outputs[rt_layout::OUT_BINDING_RESERVOIR_CURRENT].index;
        uint32_t reservoirPrevIdx    = resources.outputs[rt_layout::OUT_BINDING_RESERVOIR_PREVIOUS].index;
        uint32_t normalIdx           = resources.outputs[rt_layout::OUT_BINDING_NORMAL_ROUGHNESS].index;
        uint32_t depthIdx            = resources.outputs[rt_layout::OUT_BINDING_LINEAR_DEPTH].index;

        // Write 4 storage image descriptors
        VkDescriptorImageInfo imgInfos[4]{};
        imgInfos[0].imageView = resources.resolved->views[reservoirCurrentIdx];
        imgInfos[0].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        imgInfos[1].imageView = resources.resolved->views[reservoirPrevIdx];
        imgInfos[1].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        imgInfos[2].imageView = resources.resolved->views[normalIdx];
        imgInfos[2].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        imgInfos[3].imageView = resources.resolved->views[depthIdx];
        imgInfos[3].imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkWriteDescriptorSet writes[4]{};
        for (uint32_t i = 0; i < 4; ++i) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = ds;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            writes[i].pImageInfo = &imgInfos[i];
        }
        vkUpdateDescriptorSets(device_, 4, writes, 0, nullptr);

        // Push constants
        RestirSpatialPC pc{};
        pc.width = static_cast<int32_t>(w);
        pc.height = static_cast<int32_t>(h);
        pc.spatialTaps = static_cast<int32_t>(cfg.restirSpatialTaps);
        pc.spatialRadius = static_cast<int32_t>(cfg.restirSpatialRadius);
        pc.temporalMClamp = static_cast<int32_t>(cfg.restirTemporalMClamp);
        pc.wClamp = static_cast<int32_t>(cfg.restirWClamp);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, restirSpatialPipeline_.handle());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, restirSpatialLayout_.handle(),
                                0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, restirSpatialLayout_.handle(), VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(pc), &pc);

        // Dispatch: workgroup size = 16x16
        uint32_t gx = (w + 15) / 16;
        uint32_t gy = (h + 15) / 16;
        vkCmdDispatch(cmd, gx, gy, 1);
    }
}

void RayTracingAdapter::shutdown() {
    if (device_ == VK_NULL_HANDLE) return;

    sbt_ = {};
    rtPipeline_ = {};
    layout_.shutdown();
    shaderModules_.clear();
    fullShadersLoaded_ = false;

    descAllocator_.shutdown();

    // Compute pipeline cleanup
    sharcResolvePipeline_ = {};
    sharcResolveLayout_ = {};
    sharcResolveShader_ = {};
    sharcResolveReady_ = false;

    restirSpatialPipeline_ = {};
    restirSpatialLayout_ = {};
    restirSpatialShader_ = {};
    restirSpatialDescLayout_ = {};
    restirSpatialDescAlloc_.shutdown();
    restirSpatialReady_ = false;

    // Bindless fallback cleanup
    fallbackWhiteImage_ = {};
    if (fallbackSampler_ != VK_NULL_HANDLE) {
        vkDestroySampler(device_, fallbackSampler_, nullptr);
        fallbackSampler_ = VK_NULL_HANDLE;
    }
    bindlessFallbackWritten_ = false;

    initialized_ = false;
    log::info("rt-adapter", "RayTracingAdapter shut down");
}

} // namespace engine
