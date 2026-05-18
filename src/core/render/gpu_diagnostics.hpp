#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <volk.h>

// GPU-side diagnostic breadcrumbs for DEVICE_LOST debugging.
// Zero cost when disabled — checkpoint() is an inline early-return on a static bool.
// Ships in Release builds.

class GpuDiag {
public:
    enum Checkpoint : uint32_t {
        FRAME_BEGIN = 0,
        UPLOAD_TEX_BEGIN,
        UPLOAD_TEX_END,
        UPLOAD_BUF_BEGIN,
        UPLOAD_BUF_END,
        STAGING_UPLOADS,
        WORLD_PREPARE_BEGIN,
        CHUNK_SCHEDULE_DONE,
        BLAS_BUILD_IMPORTANT,
        BLAS_BUILD_BATCH,
        BLAS_BUILD_ENTITY,
        TLAS_UPDATE,
        TLAS_BUILD,
        SHARC_UPDATE_RT,
        SHARC_RESOLVE,
        RT_DISPATCH_MAIN,
        RESTIR_SPATIAL,
        OFFLINE_ACCUMULATE,
        SVGF_DENOISE,
        NRD_DENOISE,
        DLSS_EVALUATE,
        FSR_UPSCALE,
        TONE_MAP_HISTOGRAM,
        TONE_MAP_EXPOSURE,
        POST_RENDER,
        POST_RENDER_CAS,
        HDR_COMPOSITE,
        FRAME_COMPLETE,
        _COUNT
    };

    // Call once after device creation.
    static void init(VkDevice device, VkPhysicalDevice physDevice, bool checkpointsSupported, bool deviceFaultSupported);

    // Enable/disable at runtime (controlled by Options::gpuDiagnostics).
    static void setEnabled(bool on);
    static inline bool enabled() { return enabled_; }

    // Insert a GPU checkpoint into a command buffer. Inline no-op when disabled.
    static inline void checkpoint(VkCommandBuffer cmd, Checkpoint cp) {
        if (!enabled_ || !checkpointsSupported_) return;
        vkCmdSetCheckpointNV(cmd, reinterpret_cast<const void*>(static_cast<uintptr_t>(cp)));
    }

    // Call after VK_ERROR_DEVICE_LOST. Queries GPU checkpoints + device fault, dumps to file.
    static void onDeviceLost(VkQueue queue, const std::filesystem::path& logsDir, uint64_t frameNumber);

    // Query checkpoints from a specific queue and print them. Returns last completed checkpoint ID.
    static uint32_t queryAndPrintCheckpoints(VkQueue queue, const char* queueName,
                                              std::function<void(const std::string&)> writeToAll);

    // Human-readable name for a checkpoint.
    static const char* checkpointName(uint32_t cp);

private:
    static bool enabled_;
    static bool checkpointsSupported_;
    static bool deviceFaultSupported_;
    static VkDevice device_;
};
