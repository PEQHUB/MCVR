#pragma once

#include <volk.h>
#include <memory>
#include <vector>

namespace mcvr {

enum class UpscalerQualityMode {
    NativeAA = 0,
    Quality = 1,
    Balanced = 2,
    Performance = 3,
    UltraPerformance = 4
};

struct UpscalerConfig {
    VkDevice device;
    VkPhysicalDevice physicalDevice;
    VkCommandPool commandPool;
    VkQueue graphicsQueue;
    uint32_t graphicsQueueFamily;

    uint32_t maxRenderWidth;
    uint32_t maxRenderHeight;
    uint32_t maxDisplayWidth;
    uint32_t maxDisplayHeight;

    UpscalerQualityMode qualityMode;

    bool hdr;
    bool depthInverted;
    bool depthInfinite;
    bool autoExposure;
    bool enableSharpening;
    float sharpness;
};

struct UpscalerInput {
    VkCommandBuffer commandBuffer;

    VkImage colorImage;
    VkImageView colorImageView;
    VkImageLayout colorLayout;
    VkImage depthImage;
    VkImageView depthImageView;
    VkFormat depthFormat;
    VkImage motionVectorImage;
    VkImageView motionVectorImageView;

    VkImage reactiveImage;
    VkImageView reactiveImageView;
    VkImage exposureImage;
    VkImageView exposureImageView;

    VkImage outputImage;
    VkImageView outputImageView;
    VkImageLayout outputLayout;

    float jitterOffsetX;
    float jitterOffsetY;
    float motionVectorScaleX;
    float motionVectorScaleY;

    uint32_t renderWidth;
    uint32_t renderHeight;
    uint32_t displayWidth;
    uint32_t displayHeight;

    float cameraNear;
    float cameraFar;
    float cameraFovVertical;
    float frameTimeDelta;
    float preExposure;

    bool reset;
    bool enableSharpening;
    float sharpness;
};

class FSR3Upscaler {
public:
    FSR3Upscaler();
    ~FSR3Upscaler();

    bool initialize(const UpscalerConfig& config);
    void dispatch(const UpscalerInput& input);
    void resize(uint32_t renderWidth, uint32_t renderHeight,
               uint32_t displayWidth, uint32_t displayHeight);
    void destroy();

    const char* getName() const { return "AMD FSR 3"; }
    bool isAvailable() const;

private:
    bool createContext();
    void destroyContext();

    bool m_initialized = false;
    bool m_contextCreated = false;

    VkDevice m_device = VK_NULL_HANDLE;
    VkPhysicalDevice m_physicalDevice = VK_NULL_HANDLE;
    VkCommandPool m_commandPool = VK_NULL_HANDLE;
    VkQueue m_graphicsQueue = VK_NULL_HANDLE;
    uint32_t m_graphicsQueueFamily = 0;

    UpscalerConfig m_config;

    uint32_t m_renderWidth = 0;
    uint32_t m_renderHeight = 0;
    uint32_t m_displayWidth = 0;
    uint32_t m_displayHeight = 0;

    void* m_fsrContext = nullptr;

    int32_t m_jitterPhaseCount = 0;
    int32_t m_jitterIndex = 0;

    bool m_debugLogged = false;
    uint32_t m_lastRenderWidth = 0;
    uint32_t m_lastRenderHeight = 0;
    uint32_t m_lastDisplayWidth = 0;
    uint32_t m_lastDisplayHeight = 0;
};

} // namespace mcvr