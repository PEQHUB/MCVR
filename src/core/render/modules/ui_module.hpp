#pragma once

#include "common/shared.hpp"
#include "common/singleton.hpp"
#include "core/all_extern.hpp"
#include "core/vulkan/all_core_vulkan.hpp"

#include <map>
#include <array>

class Framework;
class FrameworkContext;

struct GraphicsPipelineShaderInfo {
    std::string vertexShaderFile;
    std::string fragmentShaderFile;
    VkPrimitiveTopology topology;
};

struct GraphicsPipelineShaders {
    std::shared_ptr<vk::Shader> vertexShader;
    std::shared_ptr<vk::Shader> fragmentShader;
};

enum OverlayDrawPipelineType {
    POSITION_TEX,
    POSITION_COLOR,
    POSITION_TEX_COLOR,
    POSITION_COLOR_TEX_LIGHT,
    POSITION_COLOR_TEXTURE_OVERLAY_LIGHT_NORMAL,
    POSITION_COLOR_TEXTURE_OVERLAY_LIGHT_NORMAL_NO_OUTLINE,
    POSITION_END_PORTAL,
    POSITION,
    MAX_OVERLAY_DRAW_PIPELINE_TYPE,
};

enum OverlayPostPipelineType {
    BLUR,
    MAX_OVERLAY_POST_PIPELINE_TYPE,
};

enum OverlayMode {
    NONE,
    DRAW,
    POST,
};

class UIModuleContext;

class UIModule : public SharedObject<UIModule> {
    friend UIModuleContext;

  public:
    UIModule();

    void init(std::shared_ptr<Framework> framework);
    std::vector<std::shared_ptr<UIModuleContext>> &contexts();
    std::vector<std::shared_ptr<vk::DescriptorTable>> &overlayDescriptorTables();

    void bindTexture(std::shared_ptr<vk::Sampler> sampler, std::shared_ptr<vk::DeviceLocalImage> image, int index);

  private:
    void initOverlayDescriptorTablesAndFrameSamplers();

    void initOverlayDrawImages();
    void initOverlayDrawRenderPass();
    void initOverlayDrawFrameBuffers();
    void initOverlayDrawPipelineTypes();
    void initOverlayDrawPipelines();

    void initOverlayPostImages();
    void initOverlayPostRenderPass();
    void initOverlayPostFrameBuffers();
    void initOverlayPostPipelineTypes();
    void initOverlayPostPipelines();

  private:
    std::weak_ptr<Framework> framework_;
    std::vector<std::shared_ptr<vk::DescriptorTable>> overlayDescriptorTables_;

    std::vector<std::shared_ptr<vk::DeviceLocalImage>> overlayDrawColorImages_;
    std::vector<std::shared_ptr<vk::DeviceLocalImage>> overlayDrawDepthStencilImages_;
    std::shared_ptr<vk::RenderPass> overlayDrawRenderPass_;
    std::vector<std::shared_ptr<vk::Framebuffer>> overlayDrawFramebuffers_;
    std::map<OverlayDrawPipelineType, GraphicsPipelineShaderInfo> overlayDrawPipelineInfos_;
    std::map<OverlayDrawPipelineType, GraphicsPipelineShaders> overlayDrawPipelineShaders_;
    std::map<OverlayDrawPipelineType, std::shared_ptr<vk::DynamicGraphicsPipeline>> overlayDrawPipelines_;

    std::vector<std::shared_ptr<vk::DeviceLocalImage>> overlayPostColorImages_;
    std::vector<std::shared_ptr<vk::Sampler>> overlayDrawColorImageSamplers_;
    std::shared_ptr<vk::RenderPass> overlayPostRenderPass_;
    std::vector<std::shared_ptr<vk::Framebuffer>> overlayPostFramebuffers_;
    std::map<OverlayPostPipelineType, GraphicsPipelineShaderInfo> overlayPostPipelineInfos_;
    std::map<OverlayPostPipelineType, GraphicsPipelineShaders> overlayPostPipelineShaders_;
    std::map<OverlayPostPipelineType, std::shared_ptr<vk::GraphicsPipeline>> overlayPostPipelines_;

    std::vector<std::shared_ptr<UIModuleContext>> contexts_;
};

struct UIModuleContext : public SharedObject<UIModuleContext> {
    std::weak_ptr<FrameworkContext> frameworkContext;
    std::weak_ptr<UIModule> uiModule;

    bool overlayScissorEnabled;
    VkRect2D overlayScissor;
    VkViewport overlayViewport;

    VkBool32 overlayBlendEnabled;
    VkColorBlendEquationEXT overlayColorBlendEquation;
    VkColorComponentFlags overlayColorWriteMask;
    bool overlayColorLogicOpEnable;
    VkLogicOp overlayColorLogicOp;
    std::array<float, 4> overlayBlendConstants;

    bool overlayDepthTestEnable;
    bool overlayDepthWriteEnable;
    VkCompareOp overlayDepthCompareOp;
    bool overlayStencilTestEnable;
    std::array<VkStencilOp, 2> overlayFailOp; // for front and back face
    std::array<VkStencilOp, 2> overlayPassOp;
    std::array<VkStencilOp, 2> overlayDepthFailOp;
    std::array<VkCompareOp, 2> overlayCompareOp;
    std::array<uint32_t, 2> overlayReference;
    std::array<uint32_t, 2> overlayCompareMask;
    std::array<uint32_t, 2> overlayWriteMask;

    VkCullModeFlags overlayCullMode;
    VkFrontFace overlayFrontFace;
    VkPolygonMode overlayPolygonMode;
    bool overlayDepthBiasEnable;
    std::array<float, 3> overlayDepthBiasConstantFactor; // for 3 types of polygon mode
    std::array<float, 3> overlayDepthBiasClamp;
    std::array<float, 3> overlayDepthBiasSlopeFactor;
    float overlayLineWidth;

    std::array<float, 4> overlayClearColors;
    float overlayClearDepth;
    uint32_t overlayClearStencil;

    OverlayMode overlayMode;

    std::shared_ptr<vk::DescriptorTable> overlayDescriptorTable;
    std::shared_ptr<vk::DeviceLocalImage> overlayDrawColorImage;
    std::shared_ptr<vk::DeviceLocalImage> overlayDrawDepthStencilImage;
    std::shared_ptr<vk::Framebuffer> overlayDrawFramebuffer;
    std::shared_ptr<vk::DeviceLocalImage> overlayPostColorImage;
    std::shared_ptr<vk::Sampler> overlayDrawColorImageSampler;
    std::shared_ptr<vk::Framebuffer> overlayPostFramebuffer;

    UIModuleContext(std::shared_ptr<FrameworkContext> context, std::shared_ptr<UIModule> uiModule);

    void syncToCommandBuffer();
    void syncFromContext(std::shared_ptr<UIModuleContext> other);

    void setOverlayScissorEnabled(bool enabled);
    void setOverlayScissor(int x, int y, int width, int height);
    void setOverlayViewport(int x, int y, int width, int height);

    void setOverlayBlendEnable(bool enable);
    void setOverlayColorBlendConstants(float const1, float const2, float const3, float const4);
    void setOverlayColorLogicOpEnable(bool enable);
    void setOverlayBlendFuncSeparate(int srcColorBlendFactor,
                                     int srcAlphaBlendFactor,
                                     int dstColorBlendFactor,
                                     int dstAlphaBlendFactor);
    void setOverlayBlendOpSeparate(int colorBlendOp, int alphaBlendOp);
    void setOverlayColorWriteMask(int colorWriteMask);
    void setOverlayColorLogicOp(int colorLogicOp);

    void setOverlayDepthTestEnable(bool enable);
    void setOverlayDepthWriteEnable(bool enable);
    void setOverlayStencilTestEnable(bool enable);
    void setOverlayDepthCompareOp(int depthCompareOp);
    void setOverlayStencilFrontFunc(int compareOp, int reference, int compareMask);
    void setOverlayStencilBackFunc(int compareOp, int reference, int compareMask);
    void setOverlayStencilFrontOp(int failOp, int depthFailOp, int passOp);
    void setOverlayStencilBackOp(int failOp, int depthFailOp, int passOp);
    void setOverlayStencilFrontWriteMask(int writeMask);
    void setOverlayStencilBackWriteMask(int writeMask);

    void setOverlayLineWidth(float lineWidth);
    void setOverlayPolygonMode(int polygonMode);
    void setOverlayCullMode(int cullMode);
    void setOverlayFrontFace(int frontFace);
    void setOverlayDepthBiasEnable(int polygonMode, bool enable);
    void setOverlayDepthBias(float depthBiasSlopeFactor, float depthBiasConstantFactor);

    void setOverlayClearColor(float red, float green, float blue, float alpha);
    void setOverlayClearDepth(double depth);
    void setOverlayClearStencil(int stencil);

    void switchOverlayDraw();
    void switchOverlayPost();

    void clearOverlayEntireColorAttachment();
    void clearOverlayEntireDepthStencilAttachment(int aspectMask);

    void drawIndexed(std::shared_ptr<vk::DeviceLocalBuffer> vertexBuffer,
                     std::shared_ptr<vk::DeviceLocalBuffer> indexBuffer,
                     OverlayDrawPipelineType pipelineType,
                     uint32_t indexCount,
                     VkIndexType indexType);

    void postBlur(int times = 1);

    void begin(std::shared_ptr<UIModuleContext> lastContext);
    void end();
};
