#include "gpu_diagnostics.hpp"

#include <chrono>
#include <ctime>
#include <fstream>
#include <iostream>
#include <vector>

bool GpuDiag::enabled_ = false;
bool GpuDiag::checkpointsSupported_ = false;
bool GpuDiag::deviceFaultSupported_ = false;
VkDevice GpuDiag::device_ = VK_NULL_HANDLE;

static const char* CHECKPOINT_NAMES[] = {
    "FRAME_BEGIN",
    "UPLOAD_TEX_BEGIN",
    "UPLOAD_TEX_END",
    "UPLOAD_BUF_BEGIN",
    "UPLOAD_BUF_END",
    "STAGING_UPLOADS",
    "WORLD_PREPARE_BEGIN",
    "CHUNK_SCHEDULE_DONE",
    "BLAS_BUILD_IMPORTANT",
    "BLAS_BUILD_BATCH",
    "BLAS_BUILD_ENTITY",
    "TLAS_UPDATE",
    "TLAS_BUILD",
    "SHARC_UPDATE_RT",
    "SHARC_RESOLVE",
    "RT_DISPATCH_MAIN",
    "RESTIR_SPATIAL",
    "OFFLINE_ACCUMULATE",
    "SVGF_DENOISE",
    "NRD_DENOISE",
    "DLSS_EVALUATE",
    "FSR_UPSCALE",
    "TONE_MAP_HISTOGRAM",
    "TONE_MAP_EXPOSURE",
    "POST_RENDER",
    "POST_RENDER_CAS",
    "HDR_COMPOSITE",
    "FRAME_COMPLETE",
};
static_assert(sizeof(CHECKPOINT_NAMES) / sizeof(CHECKPOINT_NAMES[0]) == GpuDiag::_COUNT,
              "CHECKPOINT_NAMES must match Checkpoint enum count");

void GpuDiag::init(VkDevice device, VkPhysicalDevice /*physDevice*/,
                    bool checkpointsSupported, bool deviceFaultSupported) {
    device_ = device;
    checkpointsSupported_ = checkpointsSupported;
    deviceFaultSupported_ = deviceFaultSupported;

    if (checkpointsSupported_)
        std::cout << "[GpuDiag] NV diagnostic checkpoints available" << std::endl;
    if (deviceFaultSupported_)
        std::cout << "[GpuDiag] EXT device fault available" << std::endl;
}

void GpuDiag::setEnabled(bool on) {
    enabled_ = on;
    if (on && checkpointsSupported_)
        std::cout << "[GpuDiag] ENABLED — GPU checkpoints active" << std::endl;
}

const char* GpuDiag::checkpointName(uint32_t cp) {
    if (cp < _COUNT) return CHECKPOINT_NAMES[cp];
    return "UNKNOWN";
}

void GpuDiag::onDeviceLost(VkQueue queue, const std::filesystem::path& logsDir, uint64_t frameNumber) {
    std::filesystem::create_directories(logsDir);
    std::ofstream out(logsDir / "gpu_diagnostics.txt");
    auto writeToAll = [&](const std::string& line) {
        std::cerr << line << std::endl;
        if (out.is_open()) out << line << "\n";
    };

    // Timestamp
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    char timeBuf[64];
    std::strftime(timeBuf, sizeof(timeBuf), "%Y-%m-%d %H:%M:%S", std::localtime(&time));

    writeToAll("=== GPU Diagnostics (DEVICE_LOST) ===");
    writeToAll(std::string("Time: ") + timeBuf);
    writeToAll("Frame: " + std::to_string(frameNumber));
    writeToAll("");

    // --- NV Diagnostic Checkpoints ---
    if (checkpointsSupported_ && queue != VK_NULL_HANDLE) {
        queryAndPrintCheckpoints(queue, "Main Queue", writeToAll);
    } else if (!checkpointsSupported_) {
        writeToAll("GPU Checkpoints: extension not available");
    } else {
        writeToAll("GPU Checkpoints: queue was VK_NULL_HANDLE");
    }

    writeToAll("");

    // --- EXT Device Fault ---
    if (deviceFaultSupported_ && device_ != VK_NULL_HANDLE) {
        VkDeviceFaultCountsEXT faultCounts{};
        faultCounts.sType = VK_STRUCTURE_TYPE_DEVICE_FAULT_COUNTS_EXT;
        VkResult faultResult = vkGetDeviceFaultInfoEXT(device_, &faultCounts, nullptr);

        if (faultResult == VK_SUCCESS) {
            writeToAll("Device Fault Counts:");
            writeToAll("  Address faults: " + std::to_string(faultCounts.addressInfoCount));
            writeToAll("  Vendor faults:  " + std::to_string(faultCounts.vendorInfoCount));
            writeToAll("  Vendor binary:  " + std::to_string(faultCounts.vendorBinarySize) + " bytes");

            if (faultCounts.addressInfoCount > 0 || faultCounts.vendorInfoCount > 0) {
                VkDeviceFaultInfoEXT faultInfo{};
                faultInfo.sType = VK_STRUCTURE_TYPE_DEVICE_FAULT_INFO_EXT;

                std::vector<VkDeviceFaultAddressInfoEXT> addrInfos(faultCounts.addressInfoCount);
                std::vector<VkDeviceFaultVendorInfoEXT> vendorInfos(faultCounts.vendorInfoCount);
                faultInfo.pAddressInfos = addrInfos.empty() ? nullptr : addrInfos.data();
                faultInfo.pVendorInfos = vendorInfos.empty() ? nullptr : vendorInfos.data();

                faultResult = vkGetDeviceFaultInfoEXT(device_, &faultCounts, &faultInfo);
                if (faultResult == VK_SUCCESS) {
                    writeToAll("  Description: " + std::string(faultInfo.description));

                    auto addrTypeName = [](int t) -> const char* {
                        // VkDeviceFaultAddressTypeEXT values
                        if (t == 0) return "NONE";
                        if (t == 1) return "READ_AFTER_DESTROY";
                        if (t == 2) return "WRITE_AFTER_DESTROY";
                        if (t == 3) return "EXEC_AFTER_DESTROY";
                        if (t == 4) return "IP_UNKNOWN";
                        if (t == 5) return "IP_INVALID";
                        if (t == 6) return "IP_FAULT";
                        return "UNKNOWN";
                    };
                    for (uint32_t i = 0; i < faultCounts.addressInfoCount; i++) {
                        auto& a = addrInfos[i];
                        char addr[32];
                        snprintf(addr, sizeof(addr), "0x%llX", (unsigned long long)a.reportedAddress);
                        writeToAll("  Address[" + std::to_string(i) + "]: " + addr +
                                   " type=" + std::to_string(a.addressType) +
                                   " (" + addrTypeName(a.addressType) + ")" +
                                   " precision=" + std::to_string(a.addressPrecision));
                    }
                    for (uint32_t i = 0; i < faultCounts.vendorInfoCount; i++) {
                        auto& v = vendorInfos[i];
                        writeToAll("  Vendor[" + std::to_string(i) + "]: " + std::string(v.description) +
                                   " code=" + std::to_string(v.vendorFaultCode) +
                                   " data=" + std::to_string(v.vendorFaultData));
                    }
                }
            }
        } else {
            writeToAll("Device Fault: vkGetDeviceFaultInfoEXT returned " + std::to_string(faultResult));
        }
    } else {
        writeToAll("Device Fault: extension not available");
    }

    writeToAll("");
    writeToAll("=== End GPU Diagnostics ===");

    if (out.is_open()) {
        out.flush();
        out.close();
        std::cerr << "[GpuDiag] Diagnostics written to " << (logsDir / "gpu_diagnostics.txt").string() << std::endl;
    }
}

uint32_t GpuDiag::queryAndPrintCheckpoints(VkQueue queue, const char* queueName,
                                            std::function<void(const std::string&)> writeToAll) {
    uint32_t count = 0;
    vkGetQueueCheckpointDataNV(queue, &count, nullptr);
    if (count > 0) {
        std::vector<VkCheckpointDataNV> checkpoints(count);
        for (auto& cp : checkpoints) {
            cp.sType = VK_STRUCTURE_TYPE_CHECKPOINT_DATA_NV;
            cp.pNext = nullptr;
        }
        vkGetQueueCheckpointDataNV(queue, &count, checkpoints.data());

        writeToAll(std::string(queueName) + " Checkpoints (" + std::to_string(count) + " recovered):");
        uint32_t lastId = UINT32_MAX;
        for (uint32_t i = 0; i < count; i++) {
            uint32_t id = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(checkpoints[i].pCheckpointMarker));
            const char* name = checkpointName(id);
            std::string stage;
            switch (checkpoints[i].stage) {
                case VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT: stage = "TOP_OF_PIPE"; break;
                case VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT: stage = "BOTTOM_OF_PIPE"; break;
                default: stage = "stage=0x" + std::to_string(checkpoints[i].stage); break;
            }
            writeToAll("  [" + std::to_string(i) + "] " + name + " (id=" + std::to_string(id) + ", " + stage + ")");
            if (lastId == UINT32_MAX || id > lastId) lastId = id;
        }

        if (lastId < _COUNT && lastId + 1 < _COUNT) {
            writeToAll("");
            writeToAll(">>> " + std::string(queueName) + " completed: " + std::string(checkpointName(lastId)));
            writeToAll(">>> " + std::string(queueName) + " likely crash point: " + std::string(checkpointName(lastId + 1)));
        }
        return lastId;
    } else {
        writeToAll(std::string(queueName) + " Checkpoints: none recovered (count=0)");
        return UINT32_MAX;
    }
}
