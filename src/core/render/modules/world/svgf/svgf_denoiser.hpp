#pragma once

#include "common/shared.hpp"
#include "core/vulkan/all_core_vulkan.hpp"
#include <memory>
#include <vector>
#include <array>

struct SvgfInputs {
    std::shared_ptr<vk::DeviceLocalImage> directRadiance;
    std::shared_ptr<vk::DeviceLocalImage> noisyDiffuseRadiance;
    std::shared_ptr<vk::DeviceLocalImage> noisySpecularRadiance;
    std::shared_ptr<vk::DeviceLocalImage> motionVectors;
    std::shared_ptr<vk::DeviceLocalImage> linearDepth;
    std::shared_ptr<vk::DeviceLocalImage> normalRoughness;
    std::shared_ptr<vk::DeviceLocalImage> diffuseAlbedo;
    std::shared_ptr<vk::DeviceLocalImage> specularAlbedo;
    std::shared_ptr<vk::DeviceLocalImage> clearRadiance;
    std::shared_ptr<vk::DeviceLocalImage> baseEmission;
    std::shared_ptr<vk::HostVisibleBuffer> worldUniformBuffer;
};

struct SvgfOutputs {
    std::shared_ptr<vk::DeviceLocalImage> hdrOutput;
};

class SvgfDenoiser {
public:
    SvgfDenoiser();
    ~SvgfDenoiser();

    bool init(std::shared_ptr<vk::Instance> instance,
              std::shared_ptr<vk::PhysicalDevice> physicalDevice,
              std::shared_ptr<vk::Device> device,
              std::shared_ptr<vk::VMA> vma,
              uint32_t width,
              uint32_t height,
              uint32_t contextCount);

    void denoise(std::shared_ptr<vk::CommandBuffer> commandBuffer,
                 const SvgfInputs &inputs,
                 const SvgfOutputs &outputs,
                 uint32_t frameIndex);

private:
    struct SvgfPipeline {
        VkPipeline pipeline = VK_NULL_HANDLE;
        VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
        VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
        VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
        std::vector<VkDescriptorSet> descriptorSets;
    };

    void createPipelines();
    void createResources();
    void createSamplers();
    
    std::shared_ptr<vk::Device> m_device;
    std::shared_ptr<vk::VMA> m_vma;
    std::shared_ptr<vk::PhysicalDevice> m_physicalDevice;
    uint32_t m_width = 0;
    uint32_t m_height = 0;
    uint32_t m_contextCount = 0;
    
    VkDescriptorPool m_descriptorPool = VK_NULL_HANDLE;

    // --- SVGF GI Resources (MCVR-2 style) ---
    struct FrameHistory {
        std::shared_ptr<vk::DeviceLocalImage> giRadiance;
        std::shared_ptr<vk::DeviceLocalImage> giMoments;
        std::shared_ptr<vk::DeviceLocalImage> giDepth;
        std::shared_ptr<vk::DeviceLocalImage> giNormal;
        std::shared_ptr<vk::DeviceLocalImage> giHistoryLength;
        // Direct lighting history (shadows)
        std::shared_ptr<vk::DeviceLocalImage> directRadiance;
        std::shared_ptr<vk::DeviceLocalImage> directMoments;
    };
    std::array<FrameHistory, 2> m_history;
    uint32_t m_historyReadIdx = 0;
    uint32_t m_historyWriteIdx = 1;
    bool m_needsHistoryClear = true;

    // MCVR-2 style: demodulated radiance for GI
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> m_demodulatedGiImages;

    // --- SVGF Specular Resources ---
    struct SpecularHistory {
        std::shared_ptr<vk::DeviceLocalImage> specRadiance;
        std::shared_ptr<vk::DeviceLocalImage> specMoments;
        std::shared_ptr<vk::DeviceLocalImage> specDepth;
        std::shared_ptr<vk::DeviceLocalImage> specNormal;
        std::shared_ptr<vk::DeviceLocalImage> specHistoryLength;
    };
    std::array<SpecularHistory, 2> m_specHistory;

    struct FramePingPong {
        std::shared_ptr<vk::DeviceLocalImage> giColorPing;
        std::shared_ptr<vk::DeviceLocalImage> giColorPong;
        std::shared_ptr<vk::DeviceLocalImage> giVariancePing;
        std::shared_ptr<vk::DeviceLocalImage> giVariancePong;
        // Specular ping-pong
        std::shared_ptr<vk::DeviceLocalImage> specColorPing;
        std::shared_ptr<vk::DeviceLocalImage> specColorPong;
        std::shared_ptr<vk::DeviceLocalImage> specVariancePing;
        std::shared_ptr<vk::DeviceLocalImage> specVariancePong;
        // Direct light ping-pong
        std::shared_ptr<vk::DeviceLocalImage> directColorPing;
        std::shared_ptr<vk::DeviceLocalImage> directColorPong;
        std::shared_ptr<vk::DeviceLocalImage> directVariancePing;
        std::shared_ptr<vk::DeviceLocalImage> directVariancePong;
    };
    std::vector<FramePingPong> m_framePingPong;

    // --- Pipelines (MCVR-2 style GI) ---
    SvgfPipeline m_giPreparePipeline;      
    SvgfPipeline m_giTemporalPipeline;     
    SvgfPipeline m_giVariancePipeline;     
    std::array<SvgfPipeline, 5> m_giAtrousPipelines;  
    SvgfPipeline m_giModulatePipeline;     
    
    // Legacy pipelines (kept for compatibility)
    SvgfPipeline m_reprojectPipeline;      
    SvgfPipeline m_combinePipeline;
    SvgfPipeline m_extractRoughnessPipeline;

    // Specular pipelines
    SvgfPipeline m_specReprojectPipeline;
    SvgfPipeline m_specVariancePipeline;
    std::array<SvgfPipeline, 5> m_specAtrousPipelines;

    // Direct light pipelines
    SvgfPipeline m_directTemporalPipeline;
    SvgfPipeline m_directMomentsFilterPipeline;
    SvgfPipeline m_directSpatialPipeline;
    SvgfPipeline m_directTemporalClampPipeline;
    std::array<SvgfPipeline, 5> m_directAtrousPipelines;

    VkSampler m_linearSampler = VK_NULL_HANDLE;
    VkSampler m_pointSampler = VK_NULL_HANDLE;
};
