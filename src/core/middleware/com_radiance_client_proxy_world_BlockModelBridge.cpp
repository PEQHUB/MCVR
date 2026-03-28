#include <jni.h>
#include "core/render/renderer.hpp"
#include "core/render/render_framework.hpp"
#include "core/render/block_model_table.hpp"
#include "core/render/sprite_registry.hpp"
#include "core/render/texture_arrays.hpp"

#include <fstream>
#include <iostream>

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_proxy_world_BlockModelBridge_uploadBlockModelTable(
    JNIEnv *, jclass,
    jlong entryPtr, jint entryCount,
    jlong quadPtr, jint quadCount) {

    auto* entries = reinterpret_cast<const BlockModelEntry*>(entryPtr);
    auto* quads = reinterpret_cast<const BlockModelQuad*>(quadPtr);

    Renderer::blockModelTable.load(entries, static_cast<uint32_t>(entryCount),
                                   quads, static_cast<uint32_t>(quadCount));
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_proxy_world_BlockModelBridge_uploadBiomeTintTable(
    JNIEnv *, jclass,
    jlong tintPtr, jint tintCount) {

    auto* tints = reinterpret_cast<const BiomeTintEntry*>(tintPtr);
    Renderer::blockModelTable.loadBiomeTints(tints, static_cast<uint32_t>(tintCount));
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_proxy_world_BlockModelBridge_uploadSpriteBounds(
    JNIEnv *, jclass,
    jlong boundsPtr, jint boundsCount) {

    auto* bounds = reinterpret_cast<const SpriteUVBounds*>(boundsPtr);
    Renderer::blockModelTable.loadSpriteBounds(bounds, static_cast<uint32_t>(boundsCount));
}

// Texture array system — stages sprite pixels into TextureArrayManager

// Block albedo array ID (created during finalize, used for per-layer uploads before that)
static uint32_t pendingBlockAlbedoArrayId = UINT32_MAX;
static std::vector<std::pair<uint32_t, std::vector<uint8_t>>> pendingSpritePixels; // {layerIndex, pixels}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_proxy_world_BlockModelBridge_uploadSpritePixels(
    JNIEnv *, jclass,
    jlong pixelPtr, jint spriteId,
    jint width, jint height, jint mipLevel, jint frameIndex) {

    auto* pixels = reinterpret_cast<const uint8_t*>(pixelPtr);
    size_t pixelSize = static_cast<size_t>(width) * height * 4; // RGBA8

    // Buffer frame 0 pixels until finalizeTextureArrays creates the array.
    // Java now only sends frame 0 for each sprite (1 layer per sprite).
    std::vector<uint8_t> pixelData(pixels, pixels + pixelSize);
    pendingSpritePixels.push_back({static_cast<uint32_t>(spriteId), std::move(pixelData)});

    // Register sprite with frameCount=1. Animation is driven by Java-side
    // per-tick re-uploads via updateSpriteLayer(), not shader-side layer cycling.
    if (frameIndex == 0) {
        Renderer::spriteRegistry.registerSprite(
            static_cast<uint16_t>(spriteId),
            static_cast<uint32_t>(spriteId),
            1, 1, 0, -1, -1, -1, -1);
    }
}

extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_proxy_world_BlockModelBridge_finalizeTextureArrays(
    JNIEnv *, jclass,
    jint totalBlockLayers, jint specularLayers, jint normalLayers,
    jint spriteSize, jint albedoFormat) {

    if (totalBlockLayers <= 0 || spriteSize <= 0) {
        std::cerr << "[BlockModelBridge] Invalid finalize params: "
                  << totalBlockLayers << " layers, " << spriteSize << "x" << spriteSize << std::endl;
        return;
    }

    auto framework = Renderer::instance().framework();
    auto vma = framework->vma();
    auto device = framework->device();

    // Validate against hardware maxImageArrayLayers
    auto physDevice = framework->physicalDevice();
    VkPhysicalDeviceProperties props = physDevice->properties();
    uint32_t maxLayers = props.limits.maxImageArrayLayers;
    uint32_t requestedLayers = static_cast<uint32_t>(totalBlockLayers);

    std::cout << "[BlockModelBridge] finalizeTextureArrays: "
              << requestedLayers << " block layers (max=" << maxLayers << "), "
              << spriteSize << "x" << spriteSize
              << ", " << pendingSpritePixels.size() << " staged uploads"
              << std::endl;

    if (requestedLayers > maxLayers) {
        std::cerr << "[BlockModelBridge] ERROR: " << requestedLayers
                  << " layers exceeds maxImageArrayLayers=" << maxLayers
                  << ". Clamping." << std::endl;
        requestedLayers = maxLayers;
    }

    // Create the block albedo texture array (1 layer per sprite, static only)
    VkFormat format = static_cast<VkFormat>(albedoFormat);
    pendingBlockAlbedoArrayId = Renderer::textureArrayManager.createArray(
        vma, device,
        static_cast<uint32_t>(spriteSize),
        requestedLayers,
        format, true /* generateMips */);

    // Stage all buffered pixel data (only frame 0 per sprite from Java)
    for (auto& [layerIndex, pixels] : pendingSpritePixels) {
        if (layerIndex < requestedLayers) {
            Renderer::textureArrayManager.stageLayerPixels(
                pendingBlockAlbedoArrayId, layerIndex, 0,
                pixels.data(), pixels.size());
        }
    }

    std::cout << "[BlockModelBridge] Staged " << pendingSpritePixels.size()
              << " layers into array " << pendingBlockAlbedoArrayId << std::endl;

    // Clear pending pixel buffer
    pendingSpritePixels.clear();
    pendingSpritePixels.shrink_to_fit();

    // All sprites have frameCount=1. Animation is driven by Java re-uploading
    // the current frame's pixels each tick via updateSpriteLayer().

    // Upload SpriteRegistry SSBO
    Renderer::spriteRegistry.uploadSSBO(vma, device);

    // Diagnostic: dump first 5 sprite pixel samples and registry entries to a log file
    {
        std::ofstream diag("C:/RadSER/texture_array_diag.log", std::ios::trunc);
        diag << "=== Texture Array Diagnostic ===" << std::endl;
        diag << "Total layers: " << requestedLayers << " (max=" << maxLayers << ")" << std::endl;
        diag << "Sprite size: " << spriteSize << "x" << spriteSize << std::endl;
        diag << "Format: " << albedoFormat << " (37=R8G8B8A8_SRGB)" << std::endl;
        diag << "Array ID: " << pendingBlockAlbedoArrayId << std::endl;
        diag << std::endl;
        for (uint16_t i = 0; i < 10 && i < requestedLayers; i++) {
            auto* se = Renderer::spriteRegistry.getEntry(i);
            diag << "Sprite[" << i << "]: baseLayer=" << (se ? se->baseLayer : 9999)
                 << " frameCount=" << (se ? se->frameCount : 0)
                 << " specular=" << (se ? se->specularLayer : -2)
                 << " overlay=" << (se ? se->overlaySprite : -2)
                 << std::endl;
        }
        diag.close();
        std::cout << "[BlockModelBridge] Diagnostic written to C:/RadSER/texture_array_diag.log" << std::endl;
    }

    // Pixel data is staged CPU-side. flushUploads() will be called on the first
    // render frame by ray_tracing_module.cpp when a command buffer is available.
}

// Per-tick animated sprite layer update — called from Java each render frame
// for sprites whose displayed animation frame has changed.
extern "C" JNIEXPORT void JNICALL Java_com_radiance_client_proxy_world_BlockModelBridge_updateSpriteLayer(
    JNIEnv *, jclass,
    jlong pixelPtr, jint spriteId,
    jint width, jint height) {

    if (pendingBlockAlbedoArrayId == UINT32_MAX) return; // not yet finalized

    auto* pixels = reinterpret_cast<const uint8_t*>(pixelPtr);
    size_t pixelSize = static_cast<size_t>(width) * height * 4; // RGBA8

    Renderer::textureArrayManager.stageLayerPixels(
        pendingBlockAlbedoArrayId,
        static_cast<uint32_t>(spriteId),
        0, pixels, pixelSize);
}
