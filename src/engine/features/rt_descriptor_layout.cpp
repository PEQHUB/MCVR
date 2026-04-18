#include "rt_descriptor_layout.hpp"
#include "diagnostics/log.hpp"

#include <volk.h>
#include <array>
#include <vector>

namespace engine {

namespace {

using namespace rt_layout;

// Standard RT stage flags reused for most bindings.
constexpr VkShaderStageFlags kAllRtStages =
    VK_SHADER_STAGE_RAYGEN_BIT_KHR |
    VK_SHADER_STAGE_MISS_BIT_KHR |
    VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
    VK_SHADER_STAGE_ANY_HIT_BIT_KHR;

constexpr VkShaderStageFlags kRtPlusIntersection = kAllRtStages | VK_SHADER_STAGE_INTERSECTION_BIT_KHR;

// Helper: build a single binding descriptor.
VkDescriptorSetLayoutBinding mkBinding(uint32_t binding, VkDescriptorType type, uint32_t count,
                                       VkShaderStageFlags stages) {
    VkDescriptorSetLayoutBinding b{};
    b.binding = binding;
    b.descriptorType = type;
    b.descriptorCount = count;
    b.stageFlags = stages;
    return b;
}

// Helper: create one set layout with all bindings PARTIALLY_BOUND so the test
// shaders can leave most bindings unwritten.
vk2::Result<VkDescriptorSetLayout> createSetLayout(
    VkDevice device,
    const std::vector<VkDescriptorSetLayoutBinding>& bindings) {

    std::vector<VkDescriptorBindingFlags> flags(bindings.size(),
        VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT |
        VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT);

    VkDescriptorSetLayoutBindingFlagsCreateInfo bf{};
    bf.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
    bf.bindingCount = static_cast<uint32_t>(flags.size());
    bf.pBindingFlags = flags.data();

    VkDescriptorSetLayoutCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    ci.pNext = &bf;
    ci.bindingCount = static_cast<uint32_t>(bindings.size());
    ci.pBindings = bindings.data();

    VkDescriptorSetLayout layout = VK_NULL_HANDLE;
    VkResult r = vkCreateDescriptorSetLayout(device, &ci, nullptr, &layout);
    if (r != VK_SUCCESS) return vk2::makeError(r, "vkCreateDescriptorSetLayout (rt set)");
    return layout;
}

} // namespace

vk2::Result<void> RtDescriptorLayout::init(VkDevice device) {
    if (device_ != VK_NULL_HANDLE) return {}; // already initialized
    device_ = device;

    // ----- Set 0: textures -----
    {
        std::vector<VkDescriptorSetLayoutBinding> b;
        b.push_back(mkBinding(TEX_BINDING_BINDLESS,
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, TEX_BINDLESS_COUNT,
            kAllRtStages | VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT));
        b.push_back(mkBinding(TEX_BINDING_ATMO_LUT,
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
            kAllRtStages | VK_SHADER_STAGE_FRAGMENT_BIT));
        b.push_back(mkBinding(TEX_BINDING_ATMO_CUBE,
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
            kAllRtStages | VK_SHADER_STAGE_FRAGMENT_BIT));
        b.push_back(mkBinding(TEX_BINDING_SPRITE_ALBEDO,
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, kRtPlusIntersection));
        b.push_back(mkBinding(TEX_BINDING_SPRITE_SPECULAR,
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, kRtPlusIntersection));
        b.push_back(mkBinding(TEX_BINDING_SPRITE_NORMAL,
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, kRtPlusIntersection));

        auto r = createSetLayout(device_, b);
        if (!r) { shutdown(); return vk2::makeError(r.error().vkResult, "set 0: " + r.error().message); }
        setLayouts_[SET_TEXTURES] = r.value();
    }

    // ----- Set 1: geometry/buffers -----
    {
        std::vector<VkDescriptorSetLayoutBinding> b;
        b.push_back(mkBinding(GEO_BINDING_TLAS,
            VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1,
            VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR));
        for (uint32_t bi : {GEO_BINDING_BLAS_OFFSETS, GEO_BINDING_VTX_ADDRS, GEO_BINDING_IDX_ADDRS,
                             GEO_BINDING_LAST_VTX_ADDRS, GEO_BINDING_LAST_IDX_ADDRS,
                             GEO_BINDING_LAST_TRANSFORMS}) {
            b.push_back(mkBinding(bi, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, kAllRtStages));
        }
        b.push_back(mkBinding(GEO_BINDING_TEXTURE_MAPPING,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
            kAllRtStages | VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT));
        b.push_back(mkBinding(GEO_BINDING_AREA_LIGHT,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
            VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR));
        b.push_back(mkBinding(GEO_BINDING_TILE_LIGHT,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
            VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR));
        b.push_back(mkBinding(GEO_BINDING_BIOME_COLORS,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
            VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR));
        b.push_back(mkBinding(GEO_BINDING_BLENDER_PBR,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, kAllRtStages));
        b.push_back(mkBinding(GEO_BINDING_MATERIAL_CLASS,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, kAllRtStages));
        b.push_back(mkBinding(GEO_BINDING_DISPLACED_FACE,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
            VK_SHADER_STAGE_INTERSECTION_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR));
        b.push_back(mkBinding(GEO_BINDING_SPRITE_REGISTRY,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, kRtPlusIntersection));

        auto r = createSetLayout(device_, b);
        if (!r) { shutdown(); return vk2::makeError(r.error().vkResult, "set 1: " + r.error().message); }
        setLayouts_[SET_GEOMETRY] = r.value();
    }

    // ----- Set 2: uniforms + blue noise + energy LUT -----
    // Note: V1 has gaps at bindings 3, 5-8 — vkCreateDescriptorSetLayout allows skipping.
    {
        std::vector<VkDescriptorSetLayoutBinding> b;
        b.push_back(mkBinding(UBO_BINDING_WORLD,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1,
            kAllRtStages | VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT));
        b.push_back(mkBinding(UBO_BINDING_LAST_WORLD,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, kAllRtStages));
        b.push_back(mkBinding(UBO_BINDING_SKY,
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1,
            kAllRtStages | VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT));
        b.push_back(mkBinding(UBO_BINDING_ENERGY_LUT,
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
            VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR));
        b.push_back(mkBinding(UBO_BINDING_BLUE_NOISE_SOBOL,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
            VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR));
        b.push_back(mkBinding(UBO_BINDING_BLUE_NOISE_SCRAMBLING,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
            VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR));

        auto r = createSetLayout(device_, b);
        if (!r) { shutdown(); return vk2::makeError(r.error().vkResult, "set 2: " + r.error().message); }
        setLayouts_[SET_UNIFORMS] = r.value();
    }

    // ----- Set 3: 34 storage image outputs -----
    {
        std::vector<VkDescriptorSetLayoutBinding> b;
        for (uint32_t i = 0; i < OUT_BINDING_COUNT; ++i) {
            // Bindings 16-33 (DLSS-RR guides) are raygen-only in V1.
            VkShaderStageFlags stages = (i >= 16) ? VK_SHADER_STAGE_RAYGEN_BIT_KHR : kAllRtStages;
            b.push_back(mkBinding(i, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, stages));
        }

        auto r = createSetLayout(device_, b);
        if (!r) { shutdown(); return vk2::makeError(r.error().vkResult, "set 3: " + r.error().message); }
        setLayouts_[SET_OUTPUTS] = r.value();
    }

    // ----- Pipeline layout (4 sets + push constant) -----
    {
        VkPushConstantRange pc{};
        pc.stageFlags = kRtPlusIntersection;
        pc.offset = 0;
        pc.size = sizeof(RtPushConstants);

        vk2::PipelineLayoutDesc pld;
        pld.setLayouts = {setLayouts_[0], setLayouts_[1], setLayouts_[2], setLayouts_[3]};
        pld.pushConstants = {pc};

        auto r = vk2::PipelineLayout::create(device_, pld);
        if (!r) { shutdown(); return vk2::makeError(r.error().vkResult, "rt pipeline layout: " + r.error().message); }
        pipelineLayout_ = std::move(r.value());
    }

    log::info("rt-layout", "RtDescriptorLayout initialized: 4 sets (textures, geometry, uniforms, outputs), 160B push constant");
    return {};
}

void RtDescriptorLayout::shutdown() {
    if (device_ == VK_NULL_HANDLE) return;
    pipelineLayout_ = {};
    for (auto& l : setLayouts_) {
        if (l != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(device_, l, nullptr);
            l = VK_NULL_HANDLE;
        }
    }
    device_ = VK_NULL_HANDLE;
}

std::vector<VkDescriptorPoolSize> RtDescriptorLayout::poolSizes() {
    // Per descriptor set worst-case sums (multiply by maxSetsPerFrame in caller).
    return {
        // Set 0: 4096 bindless + 5 textures = 4101 combined image samplers
        // Set 2: 1 energy LUT
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, rt_layout::TEX_BINDLESS_COUNT + 5 + 1},
        // Set 1: 1 TLAS
        {VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1},
        // Set 1: 14 storage buffers + Set 2: 2 storage buffers (blue noise)
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 14 + 2},
        // Set 2: 3 uniform buffers
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 3},
        // Set 3: 34 storage images
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, rt_layout::OUT_BINDING_COUNT},
    };
}

} // namespace engine
