#pragma once

#ifdef MCVR_ENABLE_OMM

#include <omm.h>
#include <cstdint>
#include <vector>

class OMMBaker {
  public:
    OMMBaker();
    ~OMMBaker();

    OMMBaker(const OMMBaker &) = delete;
    OMMBaker &operator=(const OMMBaker &) = delete;

    struct BakeInput {
        const uint8_t *alphaData;
        uint32_t texWidth, texHeight;
        const void *uvData;         // pointer to first UV (float2)
        uint32_t uvStrideBytes;     // stride between UV entries
        const uint32_t *indexData;  // triangle indices (3 per tri)
        uint32_t indexCount;        // total index count
        float alphaCutoff;          // e.g. 0.05
        uint32_t maxSubdivisionLevel; // OMM baker subdivision level (1-8)
    };

    struct BakeResult {
        std::vector<uint8_t> arrayData;     // raw OMM bit data
        uint32_t descArrayCount;
        std::vector<uint32_t> descOffsets;  // per-block byte offset into arrayData
        std::vector<uint16_t> descSubdivisionLevels;
        std::vector<uint16_t> descFormats;
        // Per-triangle index into desc array (or special negative index)
        std::vector<int32_t> indexBuffer;
        uint32_t indexCount;
        // Histograms for VkMicromapEXT build and BLAS attachment
        struct UsageCount {
            uint32_t count;
            uint16_t subdivisionLevel;
            uint16_t format;
        };
        std::vector<UsageCount> descArrayHistogram;
        std::vector<UsageCount> indexHistogram;
    };

    bool bake(const BakeInput &input, BakeResult &result);

  private:
    ommBaker baker_ = nullptr;
};

#endif // MCVR_ENABLE_OMM
