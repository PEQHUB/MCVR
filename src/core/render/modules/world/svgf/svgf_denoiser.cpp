#include "svgf_denoiser.hpp"
#include "core/render/renderer.hpp"
#include "core/render/buffers.hpp"
#include "core/render/render_framework.hpp"
#include <iostream>
#include <algorithm>

SvgfDenoiser::SvgfDenoiser() = default;

SvgfDenoiser::~SvgfDenoiser() {
    VkDevice dev = m_device->vkDevice();
    vkDestroySampler(dev, m_linearSampler, nullptr);
    vkDestroySampler(dev, m_pointSampler, nullptr);
    if (m_descriptorPool) vkDestroyDescriptorPool(dev, m_descriptorPool, nullptr);

    auto destroyPipeline = [&](SvgfPipeline& p) {
        if (p.pipeline) vkDestroyPipeline(dev, p.pipeline, nullptr);
        if (p.pipelineLayout) vkDestroyPipelineLayout(dev, p.pipelineLayout, nullptr);
        if (p.descriptorSetLayout) vkDestroyDescriptorSetLayout(dev, p.descriptorSetLayout, nullptr);
    };

    destroyPipeline(m_reprojectPipeline);
    destroyPipeline(m_giVariancePipeline);
    destroyPipeline(m_combinePipeline);
    destroyPipeline(m_extractRoughnessPipeline);
    for(auto& p : m_giAtrousPipelines) destroyPipeline(p);

    // Specular pipelines
    destroyPipeline(m_specReprojectPipeline);
    destroyPipeline(m_specVariancePipeline);
    for(auto& p : m_specAtrousPipelines) destroyPipeline(p);

    // m_ffxDenoiser.shutdown();
}

bool SvgfDenoiser::init(std::shared_ptr<vk::Instance> instance,
                        std::shared_ptr<vk::PhysicalDevice> physicalDevice,
                        std::shared_ptr<vk::Device> device,
                        std::shared_ptr<vk::VMA> vma,
                        uint32_t width,
                        uint32_t height,
                        uint32_t contextCount) {
    m_device = device;
    m_vma = vma;
    m_physicalDevice = physicalDevice;
    m_width = width;
    m_height = height;
    m_contextCount = contextCount;

    createSamplers();
    createResources();
    createPipelines();

    // Init FFX Denoiser
    // fsr::FfxDenoiserInitInfo ffxInfo{};
    // ffxInfo.device = device->vkDevice();
    // ffxInfo.physicalDevice = physicalDevice->vkPhysicalDevice();
    // ffxInfo.instance = instance->vkInstance();
    // ffxInfo.renderWidth = width;
    // ffxInfo.renderHeight = height;
    
    // if (!m_ffxDenoiser.init(ffxInfo)) {
    //     std::cerr << "Failed to initialize FFX Denoiser, continuing without it." << std::endl;
    // }

    return true;
}

void SvgfDenoiser::createSamplers() {
    VkSamplerCreateInfo samplerInfo = {VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    vkCreateSampler(m_device->vkDevice(), &samplerInfo, nullptr, &m_linearSampler);

    samplerInfo.magFilter = VK_FILTER_NEAREST;
    samplerInfo.minFilter = VK_FILTER_NEAREST;
    vkCreateSampler(m_device->vkDevice(), &samplerInfo, nullptr, &m_pointSampler);
}

void SvgfDenoiser::createResources() {
    // 1. SVGF GI History
    for (int i = 0; i < 2; ++i) {
        auto createImg = [&](VkFormat fmt) {
            return vk::DeviceLocalImage::create(m_device, m_vma, false, m_width, m_height, 1, fmt,
                VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
        };
        m_history[i].giRadiance = createImg(VK_FORMAT_R16G16B16A16_SFLOAT);
        m_history[i].giMoments = createImg(VK_FORMAT_R16G16_SFLOAT);  // Shader expects rg16f
        m_history[i].giDepth = createImg(VK_FORMAT_R16_SFLOAT);
        m_history[i].giNormal = createImg(VK_FORMAT_R16G16B16A16_SFLOAT);
        m_history[i].giHistoryLength = createImg(VK_FORMAT_R16_SFLOAT);  // Shader expects r16f
        // Direct lighting history (shadows)
        m_history[i].directRadiance = createImg(VK_FORMAT_R16G16B16A16_SFLOAT);
        m_history[i].directMoments = createImg(VK_FORMAT_R16G16_SFLOAT);  // Shader expects rg16f
    }

    // 2. SVGF Specular History
    for (int i = 0; i < 2; ++i) {
        auto createImg = [&](VkFormat fmt) {
            return vk::DeviceLocalImage::create(m_device, m_vma, false, m_width, m_height, 1, fmt,
                VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
        };
        m_specHistory[i].specRadiance = createImg(VK_FORMAT_R16G16B16A16_SFLOAT);
        m_specHistory[i].specMoments = createImg(VK_FORMAT_R32G32_SFLOAT);  // Shader expects rg32f
        m_specHistory[i].specDepth = createImg(VK_FORMAT_R16_SFLOAT);
        m_specHistory[i].specNormal = createImg(VK_FORMAT_R16G16B16A16_SFLOAT);
        m_specHistory[i].specHistoryLength = createImg(VK_FORMAT_R16_SFLOAT);  // Shader expects r16f
    }

    m_needsHistoryClear = true;  // Flag to clear history on first use

    // 2.5. MCVR-2 Style: Demodulated GI images
    m_demodulatedGiImages.resize(m_contextCount);
    for (uint32_t i = 0; i < m_contextCount; ++i) {
        m_demodulatedGiImages[i] = vk::DeviceLocalImage::create(m_device, m_vma, false, m_width, m_height, 1, 
            VK_FORMAT_R16G16B16A16_SFLOAT,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    }

    // 3. SVGF Ping-Pong (GI + Specular)
    m_framePingPong.resize(m_contextCount);
    for (uint32_t i = 0; i < m_contextCount; ++i) {
        auto createImg = [&](VkFormat fmt) {
            return vk::DeviceLocalImage::create(m_device, m_vma, false, m_width, m_height, 1, fmt,
                VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
        };
        // GI
        m_framePingPong[i].giColorPing = createImg(VK_FORMAT_R16G16B16A16_SFLOAT);
        m_framePingPong[i].giColorPong = createImg(VK_FORMAT_R16G16B16A16_SFLOAT);
        m_framePingPong[i].giVariancePing = createImg(VK_FORMAT_R16_SFLOAT);
        m_framePingPong[i].giVariancePong = createImg(VK_FORMAT_R16_SFLOAT);
        // Specular
        m_framePingPong[i].specColorPing = createImg(VK_FORMAT_R16G16B16A16_SFLOAT);
        m_framePingPong[i].specColorPong = createImg(VK_FORMAT_R16G16B16A16_SFLOAT);
        m_framePingPong[i].specVariancePing = createImg(VK_FORMAT_R16_SFLOAT);
        m_framePingPong[i].specVariancePong = createImg(VK_FORMAT_R16_SFLOAT);
        // Direct light
        m_framePingPong[i].directColorPing = createImg(VK_FORMAT_R16G16B16A16_SFLOAT);
        m_framePingPong[i].directColorPong = createImg(VK_FORMAT_R16G16B16A16_SFLOAT);
        m_framePingPong[i].directVariancePing = createImg(VK_FORMAT_R16_SFLOAT);
        m_framePingPong[i].directVariancePong = createImg(VK_FORMAT_R16_SFLOAT);
    }

    // 4. FFX Reflection Resources (kept for potential future use)
    uint32_t tileCount = ((m_width + 7) / 8) * ((m_height + 7) / 8);
    {
        auto createImg = [&](VkFormat fmt) {
            return vk::DeviceLocalImage::create(m_device, m_vma, false, m_width, m_height, 1, fmt,
                VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
        };
        // m_ffxResources.radianceB = createImg(VK_FORMAT_R16G16B16A16_SFLOAT);
        // m_ffxResources.varianceA = createImg(VK_FORMAT_R16_SFLOAT);
        // m_ffxResources.varianceB = createImg(VK_FORMAT_R16_SFLOAT);
        // m_ffxResources.extractedRoughness = createImg(VK_FORMAT_R16_SFLOAT);
        // m_ffxResources.output = createImg(VK_FORMAT_R16G16B16A16_SFLOAT);
        
        // m_ffxResources.denoiserTileList = vk::HostVisibleBuffer::create(m_vma, m_device, tileCount * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        // m_ffxResources.indirectArgs = vk::HostVisibleBuffer::create(m_vma, m_device, 256, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT);
    }

    // Descriptor Pool
    std::vector<VkDescriptorPoolSize> poolSizes = {
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 500},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 500},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 50} 
    };
    VkDescriptorPoolCreateInfo poolInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfo.maxSets = 500;
    poolInfo.poolSizeCount = (uint32_t)poolSizes.size();
    poolInfo.pPoolSizes = poolSizes.data();
    vkCreateDescriptorPool(m_device->vkDevice(), &poolInfo, nullptr, &m_descriptorPool);
}

void SvgfDenoiser::createPipelines() {
    VkDevice dev = m_device->vkDevice();

    auto createPipe = [&](const std::string& shaderName, const std::vector<VkDescriptorSetLayoutBinding>& bindings, SvgfPipeline& p) {
        VkDescriptorSetLayoutCreateInfo layoutInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        layoutInfo.bindingCount = (uint32_t)bindings.size();
        layoutInfo.pBindings = bindings.data();
        vkCreateDescriptorSetLayout(dev, &layoutInfo, nullptr, &p.descriptorSetLayout);

        VkPushConstantRange pushRange = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 128}; 
        VkPipelineLayoutCreateInfo pipelineLayoutInfo = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &p.descriptorSetLayout;
        pipelineLayoutInfo.pushConstantRangeCount = 1; 
        pipelineLayoutInfo.pPushConstantRanges = &pushRange;
        vkCreatePipelineLayout(dev, &pipelineLayoutInfo, nullptr, &p.pipelineLayout);

        auto shader = vk::Shader::create(m_device, (Renderer::folderPath / "shaders/world/svgf/" / shaderName).string());
        if (!shader) {
            std::cerr << "Failed to load shader: " << shaderName << std::endl;
            return;
        }
        
        VkComputePipelineCreateInfo pipelineInfo = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pipelineInfo.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipelineInfo.stage.module = shader->vkShaderModule();
        pipelineInfo.stage.pName = "main";
        pipelineInfo.layout = p.pipelineLayout;
        vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &p.pipeline);

        p.descriptorPool = m_descriptorPool;
        std::vector<VkDescriptorSetLayout> layouts(m_contextCount, p.descriptorSetLayout);
        VkDescriptorSetAllocateInfo allocInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        allocInfo.descriptorPool = p.descriptorPool;
        allocInfo.descriptorSetCount = m_contextCount;
        allocInfo.pSetLayouts = layouts.data();
        p.descriptorSets.resize(m_contextCount);
        vkAllocateDescriptorSets(dev, &allocInfo, p.descriptorSets.data());
    };

    // GI Reproject (gi_reproject.comp)
    std::vector<VkDescriptorSetLayoutBinding> reprojectBindings = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, 
        {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, 
        {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, 
        {3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, 
        {4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, 
        {5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, 
        {6, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, 
        {7, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, 
        {8, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, 
        {9, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, 
        {10, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, 
        {11, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, 
        {12, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, 
        {13, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, 
        {14, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, 
        {15, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, 
        {16, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, 
        {17, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, 
        {18, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, 
        {19, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, 
        {20, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, 
        {21, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, 
    };
    createPipe("gi_reproject_comp.spv", reprojectBindings, m_reprojectPipeline);

    // === MCVR-2 Style GI Pipelines ===
    
    // GI Prepare (gi_prepare.comp) - Demodulate + firefly clamp
    std::vector<VkDescriptorSetLayoutBinding> prepareBindings = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // giRadianceImage (input)
        {11, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // diffuseAlbedoImage
        {22, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // demodulatedGiImage (output)
        {19, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // giMomentsOut
    };
    createPipe("gi_prepare_comp.spv", prepareBindings, m_giPreparePipeline);

    // GI Temporal (gi_temporal.comp) - Temporal accumulation with history length
    std::vector<VkDescriptorSetLayoutBinding> temporalBindings = {
        {22, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // demodulatedGiImage (input)
        {19, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // giMomentsIn
        {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // motionVectorImage
        {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // linearDepthImage
        {3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // normalRoughnessImage
        {4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // giHistoryPrev
        {18, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // giMomentsPrev
        {12, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // giHistoryLengthPrev
        {5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // giDepthPrev
        {6, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // giNormalPrev
        {7, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // giHistoryOut
        {13, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // giHistoryLengthOut
        {10, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // worldUBO
        {23, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // lastWorldUBO
    };
    createPipe("gi_temporal_comp.spv", temporalBindings, m_giTemporalPipeline);

    // GI Variance (gi_variance.comp) - Variance estimation + dilation
    std::vector<VkDescriptorSetLayoutBinding> newVarianceBindings = {
        {7, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // giHistoryOut (input)
        {19, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // giMomentsOut
        {13, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // giHistoryLengthOut
        {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // linearDepthImage
        {3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // normalRoughnessImage
        {24, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // atrousStepImage0 (output)
        {25, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // atrousVarianceImage0
    };
    createPipe("gi_variance_comp.spv", newVarianceBindings, m_giVariancePipeline);

    // GI Modulate (gi_modulate.comp) - Remodulate (L * A)
    std::vector<VkDescriptorSetLayoutBinding> modulateBindings = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // denoisedGiImage (input)
        {11, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // diffuseAlbedoImage
        {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // modulatedGiImage (output)
    };
    createPipe("gi_modulate_comp.spv", modulateBindings, m_giModulatePipeline);

    // === Legacy GI Pipelines (will be phased out) ===

    // GI Variance (gi_filter_moments.comp) - DISABLED: Using gi_variance.comp instead
    // std::vector<VkDescriptorSetLayoutBinding> varianceBindings = {
    //     {0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    //     {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    //     {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    //     {3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    //     {4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    //     {5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    // };
    // createPipe("gi_filter_moments_comp.spv", varianceBindings, m_giVariancePipeline);

    // GI Atrous
    std::vector<VkDescriptorSetLayoutBinding> atrousBindings = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {6, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };
    for(int i=0; i<5; ++i) {
        int steps[] = {1, 2, 4, 8, 16};
        createPipe("gi_atrous_step" + std::to_string(steps[i]) + "_comp.spv", atrousBindings, m_giAtrousPipelines[i]);
    }

    // Combine (gi_combine.comp)
    std::vector<VkDescriptorSetLayoutBinding> combineBindings = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},           // outputImage
        {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // directLightingSampler
        {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // denoisedDiffuseSampler
        {3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // denoisedSpecularSampler
        {4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // diffuseAlbedoSampler
        {5, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // normalRoughnessSampler
        {6, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // linearDepthSampler
        {7, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},         // worldUBO
        {8, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // specularAlbedoSampler
        {9, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // clearRadianceSampler
        {10, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // baseEmissionSampler
    };
    createPipe("gi_combine_comp.spv", combineBindings, m_combinePipeline);

    // Extract Roughness (extract_roughness.comp)
    std::vector<VkDescriptorSetLayoutBinding> extractRoughnessBindings = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // normalRoughness input
        {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // extractedRoughness output
    };
    createPipe("extract_roughness_comp.spv", extractRoughnessBindings, m_extractRoughnessPipeline);

    // --- Specular Pipelines ---

    // Specular Reproject (spec_reproject.comp)
    std::vector<VkDescriptorSetLayoutBinding> specReprojectBindings = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},   // specRadianceImage
        {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},   // motionVectorImage
        {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},   // linearDepthImage
        {3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},   // normalRoughnessImage
        {4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},   // specHistoryPrev
        {5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},   // specDepthPrev
        {6, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},   // specNormalPrev
        {7, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},   // specHistoryOut
        {8, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},   // specDepthOut
        {9, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},   // specNormalOut
        {10, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // worldUBO
        {11, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // lastWorldUBO
        {12, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // specHistoryLengthPrev
        {13, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // specHistoryLengthOut
        {14, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // specMomentsPrev
        {15, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // specMomentsOut
    };
    createPipe("spec_reproject_comp.spv", specReprojectBindings, m_specReprojectPipeline);

    // Specular Variance (spec_filter_moments.comp)
    std::vector<VkDescriptorSetLayoutBinding> specVarianceBindings = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };
    createPipe("spec_filter_moments_comp.spv", specVarianceBindings, m_specVariancePipeline);

    // Specular Atrous
    std::vector<VkDescriptorSetLayoutBinding> specAtrousBindings = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {6, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    };
    for(int i=0; i<5; ++i) {
        int steps[] = {1, 2, 4, 8, 16};
        createPipe("spec_atrous_step" + std::to_string(steps[i]) + "_comp.spv", specAtrousBindings, m_specAtrousPipelines[i]);
    }

    // --- Direct Light Pipelines ---
    
    // Direct Temporal Clamp (direct_temporal_clamp.comp)
    std::vector<VkDescriptorSetLayoutBinding> directTemporalBindings = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // directInputImage
        {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // directOutputImage
        {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // linearDepthImage
        {3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // normalRoughnessImage
        {4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // shadowHistoryImage
    };
    createPipe("direct_temporal_clamp_comp.spv", directTemporalBindings, m_directTemporalClampPipeline);

    // Direct Moments Filter (direct_moments.comp)
    std::vector<VkDescriptorSetLayoutBinding> directMomentsBindings = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // directHistoryImage
        {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // directMomentsImage
        {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // directVarianceImage
        {3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // directMomentsPrev
        {4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // giHistoryLength (shared)
    };
    createPipe("direct_moments_comp.spv", directMomentsBindings, m_directMomentsFilterPipeline);

    // Direct Spatial Filter (direct_spatial.comp)
    std::vector<VkDescriptorSetLayoutBinding> directSpatialBindings = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // directInputImage
        {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // directOutputImage
        {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // linearDepthImage
        {3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // normalRoughnessImage
        {4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // shadowHistoryImage
        {5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // directCurrentImage
    };
    createPipe("direct_spatial_comp.spv", directSpatialBindings, m_directSpatialPipeline);

    // Direct Atrous Filters (direct_atrous_step*.comp)
    std::vector<VkDescriptorSetLayoutBinding> directAtrousBindings = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // directInputImage
        {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // directOutputImage
        {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // linearDepthImage
        {3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // normalRoughnessImage
        {4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // directVarianceInput
        {5, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // diffuseAlbedoImage
        {6, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},  // directVarianceOutput
    };
    for(int i=0; i<5; ++i) {
        int steps[] = {1, 2, 4, 8, 16};
        createPipe("direct_atrous_step" + std::to_string(steps[i]) + "_comp.spv", directAtrousBindings, m_directAtrousPipelines[i]);
    }
}

void SvgfDenoiser::denoise(std::shared_ptr<vk::CommandBuffer> commandBuffer,
                           const SvgfInputs &inputs,
                           const SvgfOutputs &outputs,
                           uint32_t frameIndex) {
    static int logCounter = 0;

    if (frameIndex >= m_framePingPong.size()) {
        std::cerr << "[SVGF] Error: frameIndex " << frameIndex << " out of bounds (size " << m_framePingPong.size() << ")" << std::endl;
        return;
    }

    VkCommandBuffer cmd = commandBuffer->vkCommandBuffer();
    auto& historyRead = m_history[m_historyReadIdx];
    auto& historyWrite = m_history[m_historyWriteIdx];
    auto& specHistoryRead = m_specHistory[m_historyReadIdx];
    auto& specHistoryWrite = m_specHistory[m_historyWriteIdx];
    auto& pp = m_framePingPong[frameIndex];

    // Clear history buffers on first frame to avoid garbage data
    if (m_needsHistoryClear) {
        m_needsHistoryClear = false;
        VkClearColorValue clearColor = {{0.0f, 0.0f, 0.0f, 0.0f}};
        VkImageSubresourceRange range = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

        auto clearImg = [&](const std::shared_ptr<vk::DeviceLocalImage>& img) {
            if (!img) return;
            VkImageMemoryBarrier barrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrier.subresourceRange = range;
            barrier.image = img->vkImage();
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
            vkCmdClearColorImage(cmd, img->vkImage(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clearColor, 1, &range);
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
        };

        for (int i = 0; i < 2; ++i) {
            clearImg(m_history[i].giRadiance);
            clearImg(m_history[i].giMoments);
            clearImg(m_history[i].giDepth);
            clearImg(m_history[i].giNormal);
            clearImg(m_history[i].giHistoryLength);
            clearImg(m_history[i].directRadiance);
            clearImg(m_history[i].directMoments);
            clearImg(m_specHistory[i].specRadiance);
            clearImg(m_specHistory[i].specMoments);
            clearImg(m_specHistory[i].specDepth);
            clearImg(m_specHistory[i].specNormal);
            clearImg(m_specHistory[i].specHistoryLength);
        }
        std::cerr << "[SVGF] History buffers cleared" << std::endl;
    }

    auto transition = [&](const std::shared_ptr<vk::DeviceLocalImage>& img, bool discard = false) {
        if(!img) return;
        VkImageMemoryBarrier barrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        barrier.oldLayout = discard ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_GENERAL; 
        barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        barrier.image = img->vkImage();
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
    };

    // Transition Resources (Keep content for inputs)
    transition(inputs.noisyDiffuseRadiance, false);
    transition(inputs.noisySpecularRadiance, false);
    transition(inputs.directRadiance, false);
    transition(inputs.motionVectors, false);
    transition(inputs.linearDepth, false);
    transition(inputs.normalRoughness, false);
    transition(inputs.diffuseAlbedo, false);
    transition(inputs.specularAlbedo, false);
    
    // Internal history usually needs to be kept
    transition(historyRead.giRadiance, false); 
    transition(historyRead.giDepth, false); 
    transition(historyRead.giNormal, false);
    transition(historyRead.giMoments, false); 
    transition(historyRead.giHistoryLength, false);
    
    // Output history and ping-pong can be undefined as they are overwritten
    transition(historyWrite.giRadiance, true);
    transition(historyWrite.giDepth, true);
    transition(historyWrite.giNormal, true);
    transition(historyWrite.giMoments, true);
    transition(historyWrite.giHistoryLength, true);

    // Direct lighting history transitions
    transition(historyRead.directRadiance, false);
    transition(historyRead.directMoments, false);
    transition(historyWrite.directRadiance, true);
    transition(historyWrite.directMoments, true);

    transition(pp.giColorPing, true);
    transition(pp.giColorPong, true);
    transition(pp.giVariancePing, true);
    transition(pp.giVariancePong, true);

    // Demodulated GI image transition
    transition(m_demodulatedGiImages[frameIndex], true);

    // Specular ping-pong transitions
    transition(specHistoryRead.specRadiance, false);
    transition(specHistoryRead.specDepth, false);
    transition(specHistoryRead.specNormal, false);
    transition(specHistoryRead.specMoments, false);
    transition(specHistoryRead.specHistoryLength, false);

    transition(specHistoryWrite.specRadiance, true);
    transition(specHistoryWrite.specDepth, true);
    transition(specHistoryWrite.specNormal, true);
    transition(specHistoryWrite.specMoments, true);
    transition(specHistoryWrite.specHistoryLength, true);

    // Specular ping-pong transitions
    transition(pp.specColorPing, true);
    transition(pp.specColorPong, true);
    transition(pp.specVariancePing, true);
    transition(pp.specVariancePong, true);

    transition(outputs.hdrOutput, true);

    // --- 1. SVGF GI PASS ---
    {
        // Helper lambda for adding image descriptors
        auto addImg = [&](VkDescriptorSet set, uint32_t binding, std::shared_ptr<vk::DeviceLocalImage> img,
                          std::vector<VkWriteDescriptorSet>& writes, 
                          std::vector<std::unique_ptr<VkDescriptorImageInfo>>& infos) {
            auto info = std::make_unique<VkDescriptorImageInfo>();
            info->imageView = img->vkImageView(0);
            info->imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            info->sampler = VK_NULL_HANDLE;
            writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, binding, 0, 1, 
                             VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, info.get(), nullptr, nullptr});
            infos.push_back(std::move(info));
        };

        // Step 1: gi_prepare - Demodulate + firefly clamp
        {
            VkDescriptorSet set = m_giPreparePipeline.descriptorSets[frameIndex];
            std::vector<VkWriteDescriptorSet> writes;
            std::vector<std::unique_ptr<VkDescriptorImageInfo>> infos;
            
            addImg(set, 0, inputs.noisyDiffuseRadiance, writes, infos);  // giRadianceImage
            addImg(set, 11, inputs.diffuseAlbedo, writes, infos);        // diffuseAlbedoImage
            addImg(set, 22, m_demodulatedGiImages[frameIndex], writes, infos); // demodulatedGiImage
            addImg(set, 19, historyWrite.giMoments, writes, infos);      // giMomentsOut (initial)
            
            vkUpdateDescriptorSets(m_device->vkDevice(), (uint32_t)writes.size(), writes.data(), 0, nullptr);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_giPreparePipeline.pipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_giPreparePipeline.pipelineLayout, 0, 1, &set, 0, nullptr);
            
            struct { int32_t width, height; float fireflyK; int32_t padding; } pc = {(int32_t)m_width, (int32_t)m_height, 3.0f, 0};
            vkCmdPushConstants(cmd, m_giPreparePipeline.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
            vkCmdDispatch(cmd, (m_width+15)/16, (m_height+15)/16, 1);
        }

        // Memory barrier after prepare
        {
            VkMemoryBarrier barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                0, 1, &barrier, 0, nullptr, 0, nullptr);
        }

        // Step 2: gi_temporal - Temporal accumulation with history length
        {
            VkDescriptorSet set = m_giTemporalPipeline.descriptorSets[frameIndex];
            std::vector<VkWriteDescriptorSet> writes;
            std::vector<std::unique_ptr<VkDescriptorImageInfo>> infos;
            
            addImg(set, 22, m_demodulatedGiImages[frameIndex], writes, infos); // demodulatedGiImage
            addImg(set, 19, historyWrite.giMoments, writes, infos);            // giMomentsIn
            addImg(set, 1, inputs.motionVectors, writes, infos);               // motionVectorImage
            addImg(set, 2, inputs.linearDepth, writes, infos);                 // linearDepthImage
            addImg(set, 3, inputs.normalRoughness, writes, infos);             // normalRoughnessImage
            addImg(set, 4, historyRead.giRadiance, writes, infos);             // giHistoryPrev
            addImg(set, 18, historyRead.giMoments, writes, infos);             // giMomentsPrev
            addImg(set, 12, historyRead.giHistoryLength, writes, infos);       // giHistoryLengthPrev
            addImg(set, 5, historyRead.giDepth, writes, infos);                // giDepthPrev
            addImg(set, 6, historyRead.giNormal, writes, infos);               // giNormalPrev
            addImg(set, 7, historyWrite.giRadiance, writes, infos);            // giHistoryOut
            addImg(set, 13, historyWrite.giHistoryLength, writes, infos);      // giHistoryLengthOut
            
            VkDescriptorBufferInfo uboInfo = {inputs.worldUniformBuffer->vkBuffer(), 0, VK_WHOLE_SIZE};
            writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, 10, 0, 1, 
                             VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &uboInfo, nullptr});
            
            VkDescriptorBufferInfo lastUboInfo = {Renderer::instance().buffers()->lastWorldUniformBuffer()->vkBuffer(), 0, VK_WHOLE_SIZE};
            writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, 23, 0, 1, 
                             VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &lastUboInfo, nullptr});
            
            vkUpdateDescriptorSets(m_device->vkDevice(), (uint32_t)writes.size(), writes.data(), 0, nullptr);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_giTemporalPipeline.pipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_giTemporalPipeline.pipelineLayout, 0, 1, &set, 0, nullptr);
            
            struct { int32_t width, height; float normalThresh; uint32_t maxHistory; int32_t padding; } pc = 
                {(int32_t)m_width, (int32_t)m_height, 0.5f, 16, 0};  // Reduced from 32 to 16 to reduce ghosting
            vkCmdPushConstants(cmd, m_giTemporalPipeline.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
            vkCmdDispatch(cmd, (m_width+15)/16, (m_height+15)/16, 1);
        }

        // Memory barrier after temporal
        {
            VkMemoryBarrier barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                0, 1, &barrier, 0, nullptr, 0, nullptr);
        }

        // Step 3: gi_variance - Variance estimation + dilation
        {
            VkDescriptorSet set = m_giVariancePipeline.descriptorSets[frameIndex];
            std::vector<VkWriteDescriptorSet> writes;
            std::vector<std::unique_ptr<VkDescriptorImageInfo>> infos;
            
            addImg(set, 7, historyWrite.giRadiance, writes, infos);       // giHistoryOut (accumulated)
            addImg(set, 19, historyWrite.giMoments, writes, infos);       // giMomentsOut
            addImg(set, 13, historyWrite.giHistoryLength, writes, infos); // giHistoryLengthOut
            addImg(set, 2, inputs.linearDepth, writes, infos);            // linearDepthImage
            addImg(set, 3, inputs.normalRoughness, writes, infos);        // normalRoughnessImage
            addImg(set, 24, pp.giColorPing, writes, infos);               // atrousStepImage0 (reuse ping)
            addImg(set, 25, pp.giVariancePing, writes, infos);            // atrousVarianceImage0
            
            vkUpdateDescriptorSets(m_device->vkDevice(), (uint32_t)writes.size(), writes.data(), 0, nullptr);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_giVariancePipeline.pipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_giVariancePipeline.pipelineLayout, 0, 1, &set, 0, nullptr);
            
            struct { int32_t width, height; float phiDepth, phiNormal, phiLuma; int32_t padding; } pc = 
                {(int32_t)m_width, (int32_t)m_height, 0.1f, 128.0f, 4.0f, 0};
            vkCmdPushConstants(cmd, m_giVariancePipeline.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
            vkCmdDispatch(cmd, (m_width+15)/16, (m_height+15)/16, 1);
        }

        // Memory barrier after variance
        {
            VkMemoryBarrier barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                0, 1, &barrier, 0, nullptr, 0, nullptr);
        }

        // Step 4: gi_atrous - 5 iterations of variance-adaptive spatial filtering
        auto giColorIn = pp.giColorPing;
        auto giVarIn = pp.giVariancePing;
        auto giColorOut = pp.giColorPong;
        auto giVarOut = pp.giVariancePong;

        for (int i = 0; i < 5; ++i) {
            VkDescriptorSet aSet = m_giAtrousPipelines[i].descriptorSets[frameIndex];
            std::vector<VkWriteDescriptorSet> aWrites;
            std::vector<std::unique_ptr<VkDescriptorImageInfo>> aInfos;
            
            addImg(aSet, 0, giColorIn, aWrites, aInfos);              // giInputImage
            addImg(aSet, 1, giColorOut, aWrites, aInfos);             // giOutputImage
            addImg(aSet, 2, inputs.linearDepth, aWrites, aInfos);     // linearDepthImage
            addImg(aSet, 3, inputs.normalRoughness, aWrites, aInfos); // normalRoughnessImage
            addImg(aSet, 4, giVarIn, aWrites, aInfos);                // giVarianceInput
            addImg(aSet, 6, giVarOut, aWrites, aInfos);               // giVarianceOutput
            
            vkUpdateDescriptorSets(m_device->vkDevice(), (uint32_t)aWrites.size(), aWrites.data(), 0, nullptr);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_giAtrousPipelines[i].pipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_giAtrousPipelines[i].pipelineLayout, 0, 1, &aSet, 0, nullptr);
            
            struct { int32_t width, height; float phiDepth, phiNormal, phiLuma; } pc = 
                {(int32_t)m_width, (int32_t)m_height, 0.1f, 128.0f, 4.0f};
            vkCmdPushConstants(cmd, m_giAtrousPipelines[i].pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
            vkCmdDispatch(cmd, (m_width+15)/16, (m_height+15)/16, 1);

            // Memory barrier between atrous iterations
            VkMemoryBarrier barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                0, 1, &barrier, 0, nullptr, 0, nullptr);

            // Ping-pong swap
            std::swap(giColorIn, giColorOut);
            std::swap(giVarIn, giVarOut);
        }

        // Step 5: gi_modulate - Remodulate (L * A)
        {
            VkDescriptorSet set = m_giModulatePipeline.descriptorSets[frameIndex];
            std::vector<VkWriteDescriptorSet> writes;
            std::vector<std::unique_ptr<VkDescriptorImageInfo>> infos;
            
            addImg(set, 0, giColorIn, writes, infos);                // denoisedGiImage (final denoised L)
            addImg(set, 11, inputs.diffuseAlbedo, writes, infos);    // diffuseAlbedoImage (A)
            addImg(set, 1, giColorOut, writes, infos);               // modulatedGiImage (output C = L * A)
            
            vkUpdateDescriptorSets(m_device->vkDevice(), (uint32_t)writes.size(), writes.data(), 0, nullptr);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_giModulatePipeline.pipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_giModulatePipeline.pipelineLayout, 0, 1, &set, 0, nullptr);
            
            struct { int32_t width, height, padding1, padding2; } pc = {(int32_t)m_width, (int32_t)m_height, 0, 0};
            vkCmdPushConstants(cmd, m_giModulatePipeline.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
            vkCmdDispatch(cmd, (m_width+15)/16, (m_height+15)/16, 1);
        }

        {
            VkMemoryBarrier barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                0, 1, &barrier, 0, nullptr, 0, nullptr);
        }

        pp.giColorPing = giColorOut;

        {
            VkImageCopy copyRegion = {};
            copyRegion.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            copyRegion.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            copyRegion.extent = {m_width, m_height, 1};
            
            vkCmdCopyImage(cmd, inputs.linearDepth->vkImage(), VK_IMAGE_LAYOUT_GENERAL,
                          historyWrite.giDepth->vkImage(), VK_IMAGE_LAYOUT_GENERAL, 1, &copyRegion);
            vkCmdCopyImage(cmd, inputs.normalRoughness->vkImage(), VK_IMAGE_LAYOUT_GENERAL,
                          historyWrite.giNormal->vkImage(), VK_IMAGE_LAYOUT_GENERAL, 1, &copyRegion);
        }
    }

    // --- 2. SVGF SPECULAR PASS ---
    {
        // Specular Reproject
        VkDescriptorSet set = m_specReprojectPipeline.descriptorSets[frameIndex];
        std::vector<VkWriteDescriptorSet> writes;
        std::vector<std::unique_ptr<VkDescriptorImageInfo>> infos;
        auto addImg = [&](uint32_t b, std::shared_ptr<vk::DeviceLocalImage> img) {
            auto info = std::make_unique<VkDescriptorImageInfo>();
            info->imageView = img->vkImageView(0); info->imageLayout = VK_IMAGE_LAYOUT_GENERAL; info->sampler = VK_NULL_HANDLE;
            writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, b, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, info.get(), nullptr, nullptr});
            infos.push_back(std::move(info));
        };
        addImg(0, inputs.noisySpecularRadiance);
        addImg(1, inputs.motionVectors);
        addImg(2, inputs.linearDepth);
        addImg(3, inputs.normalRoughness);
        addImg(4, specHistoryRead.specRadiance);
        addImg(5, specHistoryRead.specDepth);
        addImg(6, specHistoryRead.specNormal);
        addImg(7, specHistoryWrite.specRadiance);
        addImg(8, specHistoryWrite.specDepth);
        addImg(9, specHistoryWrite.specNormal);

        VkDescriptorBufferInfo uboInfo = {inputs.worldUniformBuffer->vkBuffer(), 0, VK_WHOLE_SIZE};
        writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, 10, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &uboInfo, nullptr});

        VkDescriptorBufferInfo lastUboInfo = {Renderer::instance().buffers()->lastWorldUniformBuffer()->vkBuffer(), 0, VK_WHOLE_SIZE};
        writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, 11, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &lastUboInfo, nullptr});

        addImg(12, specHistoryRead.specHistoryLength);
        addImg(13, specHistoryWrite.specHistoryLength);
        addImg(14, specHistoryRead.specMoments);
        addImg(15, specHistoryWrite.specMoments);

        vkUpdateDescriptorSets(m_device->vkDevice(), (uint32_t)writes.size(), writes.data(), 0, nullptr);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_specReprojectPipeline.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_specReprojectPipeline.pipelineLayout, 0, 1, &set, 0, nullptr);
        vkCmdDispatch(cmd, (m_width+15)/16, (m_height+15)/16, 1);

        // Specular Variance
        VkDescriptorSet vSet = m_specVariancePipeline.descriptorSets[frameIndex];
        std::vector<VkWriteDescriptorSet> vWrites;
        std::vector<std::unique_ptr<VkDescriptorImageInfo>> vInfos;
        auto addV = [&](uint32_t b, std::shared_ptr<vk::DeviceLocalImage> img) {
            auto i = std::make_unique<VkDescriptorImageInfo>();
            i->imageView = img->vkImageView(0); i->imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            vWrites.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, vSet, b, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, i.get(), nullptr, nullptr});
            vInfos.push_back(std::move(i));
        };
        addV(0, specHistoryWrite.specRadiance);
        addV(1, specHistoryWrite.specMoments);
        addV(2, specHistoryWrite.specHistoryLength);
        addV(3, pp.specVariancePing);
        addV(4, inputs.normalRoughness);
        addV(5, inputs.linearDepth);
        vkUpdateDescriptorSets(m_device->vkDevice(), (uint32_t)vWrites.size(), vWrites.data(), 0, nullptr);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_specVariancePipeline.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_specVariancePipeline.pipelineLayout, 0, 1, &vSet, 0, nullptr);
        vkCmdDispatch(cmd, (m_width+15)/16, (m_height+15)/16, 1);

        // Specular Atrous
        auto specColorIn = specHistoryWrite.specRadiance;
        auto specVarIn = pp.specVariancePing;
        auto specColorOut = pp.specColorPing;
        auto specVarOut = pp.specVariancePong;

        for (int i = 0; i < 5; ++i) {
            VkDescriptorSet aSet = m_specAtrousPipelines[i].descriptorSets[frameIndex];
            std::vector<VkWriteDescriptorSet> aWrites;
            std::vector<std::unique_ptr<VkDescriptorImageInfo>> aInfos;
            auto addA = [&](uint32_t b, std::shared_ptr<vk::DeviceLocalImage> img) {
                auto inf = std::make_unique<VkDescriptorImageInfo>();
                inf->imageView = img->vkImageView(0); inf->imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                aWrites.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, aSet, b, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, inf.get(), nullptr, nullptr});
                aInfos.push_back(std::move(inf));
            };
            addA(0, specColorIn);
            addA(1, specColorOut);
            addA(2, inputs.linearDepth);
            addA(3, inputs.normalRoughness);
            addA(4, specVarIn);
            addA(5, inputs.specularAlbedo ? inputs.specularAlbedo : inputs.diffuseAlbedo);
            addA(6, specVarOut);
            vkUpdateDescriptorSets(m_device->vkDevice(), (uint32_t)aWrites.size(), aWrites.data(), 0, nullptr);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_specAtrousPipelines[i].pipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_specAtrousPipelines[i].pipelineLayout, 0, 1, &aSet, 0, nullptr);
            float phiColor = 4.0f;  // Balanced specular denoising for clear reflections
            vkCmdPushConstants(cmd, m_specAtrousPipelines[i].pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, &phiColor);
            vkCmdDispatch(cmd, (m_width+15)/16, (m_height+15)/16, 1);

            if (i == 0) {
                specColorIn = pp.specColorPing; specVarIn = pp.specVariancePong;
                specColorOut = pp.specColorPong; specVarOut = pp.specVariancePing;
            } else {
                std::swap(specColorIn, specColorOut); std::swap(specVarIn, specVarOut);
            }
        }
        pp.specColorPing = specColorIn;
    }

    // --- 3. DIRECT LIGHT DENOISING ---
    {
        // Skip direct light denoising if input is not provided
        if (!inputs.directRadiance) {
            pp.directColorPing = inputs.directRadiance;
        } else {
            auto addImg = [&](VkDescriptorSet set, uint32_t binding, std::shared_ptr<vk::DeviceLocalImage> img,
                              std::vector<VkWriteDescriptorSet>& writes, 
                              std::vector<std::unique_ptr<VkDescriptorImageInfo>>& infos) {
                auto info = std::make_unique<VkDescriptorImageInfo>();
                info->imageView = img->vkImageView(0);
                info->imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                info->sampler = VK_NULL_HANDLE;
                writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, binding, 0, 1, 
                                 VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, info.get(), nullptr, nullptr});
                infos.push_back(std::move(info));
            };

            // Transition direct light history buffers
            transition(historyRead.directRadiance, false);
            transition(historyRead.directMoments, false);
            transition(historyWrite.directRadiance, true);
            transition(historyWrite.directMoments, true);
            transition(pp.directColorPing, true);
            transition(pp.directColorPong, true);
            transition(pp.directVariancePing, true);
            transition(pp.directVariancePong, true);

            // Step 2: Direct Moments Filter
            {
                VkDescriptorSet set = m_directMomentsFilterPipeline.descriptorSets[frameIndex];
                std::vector<VkWriteDescriptorSet> writes;
                std::vector<std::unique_ptr<VkDescriptorImageInfo>> infos;

                addImg(set, 0, historyWrite.directRadiance, writes, infos);     // directHistoryImage (accumulated)
                addImg(set, 1, historyWrite.directMoments, writes, infos);      // directMomentsImage (output)
                addImg(set, 2, pp.directVariancePing, writes, infos);           // directVarianceImage (output)
                addImg(set, 3, historyRead.directMoments, writes, infos);       // directMomentsPrev
                addImg(set, 4, historyWrite.giHistoryLength, writes, infos);    // giHistoryLength (shared)
                
                vkUpdateDescriptorSets(m_device->vkDevice(), (uint32_t)writes.size(), writes.data(), 0, nullptr);
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_directMomentsFilterPipeline.pipeline);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_directMomentsFilterPipeline.pipelineLayout, 0, 1, &set, 0, nullptr);
                vkCmdDispatch(cmd, (m_width+15)/16, (m_height+15)/16, 1);
            }

            {
                VkMemoryBarrier barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
                barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                    0, 1, &barrier, 0, nullptr, 0, nullptr);
            }

            auto directColorIn = inputs.directRadiance;
            auto directVarIn = pp.directVariancePing;
            auto directColorOut = pp.directColorPing;
            auto directVarOut = pp.directVariancePong;

            for (int i = 0; i < 5; ++i) {
                VkDescriptorSet aSet = m_directAtrousPipelines[i].descriptorSets[frameIndex];
                std::vector<VkWriteDescriptorSet> aWrites;
                std::vector<std::unique_ptr<VkDescriptorImageInfo>> aInfos;
                
                addImg(aSet, 0, directColorIn, aWrites, aInfos);               // directInputImage
                addImg(aSet, 1, directColorOut, aWrites, aInfos);              // directOutputImage
                addImg(aSet, 2, inputs.linearDepth, aWrites, aInfos);          // linearDepthImage
                addImg(aSet, 3, inputs.normalRoughness, aWrites, aInfos);      // normalRoughnessImage
                addImg(aSet, 4, directVarIn, aWrites, aInfos);                 // directVarianceInput
                addImg(aSet, 5, inputs.diffuseAlbedo, aWrites, aInfos);        // diffuseAlbedoImage
                addImg(aSet, 6, directVarOut, aWrites, aInfos);                // directVarianceOutput
                
                vkUpdateDescriptorSets(m_device->vkDevice(), (uint32_t)aWrites.size(), aWrites.data(), 0, nullptr);
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_directAtrousPipelines[i].pipeline);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_directAtrousPipelines[i].pipelineLayout, 0, 1, &aSet, 0, nullptr);
                
                struct { float phiColor; } pc = {4.0f};  // Moderate filtering for direct light
                vkCmdPushConstants(cmd, m_directAtrousPipelines[i].pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
                vkCmdDispatch(cmd, (m_width+15)/16, (m_height+15)/16, 1);

                VkMemoryBarrier barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
                barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                    0, 1, &barrier, 0, nullptr, 0, nullptr);

                if (i == 0) {
                    directColorIn = pp.directColorPing;
                    directVarIn = pp.directVariancePong;
                    directColorOut = pp.directColorPong;
                    directVarOut = pp.directVariancePing;
                } else {
                    std::swap(directColorIn, directColorOut);
                    std::swap(directVarIn, directVarOut);
                }
            }

            // Store final denoised direct light
            pp.directColorPing = directColorIn;
            {
                VkImageCopy copyRegion = {};
                copyRegion.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                copyRegion.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                copyRegion.extent = {m_width, m_height, 1};
                
                vkCmdCopyImage(cmd, inputs.directRadiance->vkImage(), VK_IMAGE_LAYOUT_GENERAL,
                              historyWrite.directRadiance->vkImage(), VK_IMAGE_LAYOUT_GENERAL, 1, &copyRegion);
            }
        }
    }

    // --- 4. COMBINE ---
    {
        VkDescriptorSet set = m_combinePipeline.descriptorSets[frameIndex];
        std::vector<VkWriteDescriptorSet> writes;
        std::vector<std::unique_ptr<VkDescriptorImageInfo>> infos;

        // Helper for Storage Image (Output)
        auto addImg = [&](uint32_t b, std::shared_ptr<vk::DeviceLocalImage> img) {
            auto info = std::make_unique<VkDescriptorImageInfo>();
            info->imageView = img->vkImageView(0); 
            info->imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, b, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, info.get(), nullptr, nullptr});
            infos.push_back(std::move(info));
        };

        // Helper for Combined Image Sampler (Inputs)
        auto addImgSampler = [&](uint32_t b, std::shared_ptr<vk::DeviceLocalImage> img) {
            auto info = std::make_unique<VkDescriptorImageInfo>();
            info->imageView = img->vkImageView(0); 
            info->imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            info->sampler = m_linearSampler; 
            writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, b, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, info.get(), nullptr, nullptr});
            infos.push_back(std::move(info));
        };

        addImg(0, outputs.hdrOutput);          // binding 0: outputImage (Storage)

        addImgSampler(1, pp.directColorPing ? pp.directColorPing : inputs.directRadiance); // binding 1: directLightingSampler
        addImgSampler(2, pp.giColorPing);      // binding 2: denoisedDiffuseSampler
        addImgSampler(3, pp.specColorPing);    // binding 3: denoisedSpecularSampler
        addImgSampler(4, inputs.diffuseAlbedo); // binding 4: diffuseAlbedoSampler
        addImgSampler(5, inputs.normalRoughness); // binding 5: normalRoughnessSampler
        addImgSampler(6, inputs.linearDepth);   // binding 6: linearDepthSampler
        // binding 7: worldUBO
        VkDescriptorBufferInfo uboInfo = {inputs.worldUniformBuffer->vkBuffer(), 0, VK_WHOLE_SIZE};
        writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, 7, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &uboInfo, nullptr});

        addImgSampler(8, inputs.specularAlbedo ? inputs.specularAlbedo : inputs.diffuseAlbedo); // binding 8: specularAlbedoSampler
        addImgSampler(9, inputs.clearRadiance ? inputs.clearRadiance : inputs.directRadiance);  // binding 9: clearRadianceSampler
        addImgSampler(10, inputs.baseEmission ? inputs.baseEmission : inputs.directRadiance);   // binding 10: baseEmissionSampler

        vkUpdateDescriptorSets(m_device->vkDevice(), (uint32_t)writes.size(), writes.data(), 0, nullptr);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_combinePipeline.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_combinePipeline.pipelineLayout, 0, 1, &set, 0, nullptr);
        vkCmdDispatch(cmd, (m_width+15)/16, (m_height+15)/16, 1);
    }

    m_historyReadIdx = 1 - m_historyReadIdx;
    m_historyWriteIdx = 1 - m_historyWriteIdx;
}
