#ifdef MCVR_ENABLE_OMM

#include "core/render/omm_baker.hpp"
#include <iostream>
#include <cstring>

static std::ostream &ommCout() { return std::cout << "[OMM Baker] "; }
static std::ostream &ommCerr() { return std::cerr << "[OMM Baker] "; }

OMMBaker::OMMBaker() {
    ommBakerCreationDesc desc = ommBakerCreationDescDefault();
    desc.type = ommBakerType_CPU;
    ommResult res = ommCreateBaker(&desc, &baker_);
    if (res != ommResult_SUCCESS) {
        ommCerr() << "Failed to create OMM baker, result=" << res << std::endl;
        baker_ = nullptr;
    }
}

OMMBaker::~OMMBaker() {
    if (baker_) {
        ommDestroyBaker(baker_);
        baker_ = nullptr;
    }
}

bool OMMBaker::bake(const BakeInput &input, BakeResult &result) {
    if (!baker_) return false;
    if (!input.alphaData || input.texWidth == 0 || input.texHeight == 0) return false;
    if (!input.indexData || input.indexCount == 0) return false;

    // Create CPU texture from alpha data (single channel UNORM8)
    ommCpuTextureMipDesc mip = ommCpuTextureMipDescDefault();
    mip.width = input.texWidth;
    mip.height = input.texHeight;
    mip.rowPitch = input.texWidth; // 1 byte per texel
    mip.textureData = input.alphaData;

    ommCpuTextureDesc texDesc = ommCpuTextureDescDefault();
    texDesc.format = ommCpuTextureFormat_UNORM8;
    texDesc.flags = ommCpuTextureFlags_None;
    texDesc.mips = &mip;
    texDesc.mipCount = 1;
    texDesc.alphaCutoff = input.alphaCutoff;

    ommCpuTexture texture = nullptr;
    ommResult res = ommCpuCreateTexture(baker_, &texDesc, &texture);
    if (res != ommResult_SUCCESS) {
        ommCerr() << "Failed to create CPU texture, result=" << res << std::endl;
        return false;
    }

    // Configure bake input
    ommCpuBakeInputDesc bakeDesc = ommCpuBakeInputDescDefault();
    bakeDesc.bakeFlags = ommCpuBakeFlags_Force32BitIndices;
    bakeDesc.texture = texture;
    bakeDesc.runtimeSamplerDesc.addressingMode = ommTextureAddressMode_Wrap;
    bakeDesc.runtimeSamplerDesc.filter = ommTextureFilterMode_Nearest;
    bakeDesc.alphaMode = ommAlphaMode_Test;
    bakeDesc.texCoordFormat = ommTexCoordFormat_UV32_FLOAT;
    bakeDesc.texCoords = input.uvData;
    bakeDesc.texCoordStrideInBytes = input.uvStrideBytes;
    bakeDesc.indexFormat = ommIndexFormat_UINT_32;
    bakeDesc.indexBuffer = input.indexData;
    bakeDesc.indexCount = input.indexCount;
    bakeDesc.format = ommFormat_OC1_4_State;
    bakeDesc.unknownStatePromotion = ommUnknownStatePromotion_Nearest;
    bakeDesc.alphaCutoff = input.alphaCutoff;
    bakeDesc.maxSubdivisionLevel = input.maxSubdivisionLevel;
    bakeDesc.dynamicSubdivisionScale = 2.0f; // adapt per-triangle: each micro-triangle covers ~2x2 texels

    ommCpuBakeResult bakeResult = nullptr;
    res = ommCpuBake(baker_, &bakeDesc, &bakeResult);
    ommCpuDestroyTexture(baker_, texture);

    if (res != ommResult_SUCCESS) {
        ommCerr() << "Bake failed, result=" << res << std::endl;
        if (bakeResult) ommCpuDestroyBakeResult(bakeResult);
        return false;
    }

    // Extract results
    const ommCpuBakeResultDesc *desc = nullptr;
    res = ommCpuGetBakeResultDesc(bakeResult, &desc);
    if (res != ommResult_SUCCESS || !desc) {
        ommCerr() << "Failed to get bake result desc, result=" << res << std::endl;
        ommCpuDestroyBakeResult(bakeResult);
        return false;
    }

    // Copy array data (raw OMM bits)
    result.arrayData.resize(desc->arrayDataSize);
    std::memcpy(result.arrayData.data(), desc->arrayData, desc->arrayDataSize);

    // Copy descriptor array (per-OMM-block descriptors)
    result.descArrayCount = desc->descArrayCount;
    result.descOffsets.resize(desc->descArrayCount);
    result.descSubdivisionLevels.resize(desc->descArrayCount);
    result.descFormats.resize(desc->descArrayCount);
    for (uint32_t i = 0; i < desc->descArrayCount; i++) {
        result.descOffsets[i] = desc->descArray[i].offset;
        result.descSubdivisionLevels[i] = desc->descArray[i].subdivisionLevel;
        result.descFormats[i] = desc->descArray[i].format;
    }

    // Copy per-triangle index buffer
    result.indexCount = desc->indexCount;
    result.indexBuffer.resize(desc->indexCount);
    if (desc->indexFormat == ommIndexFormat_UINT_32) {
        std::memcpy(result.indexBuffer.data(), desc->indexBuffer, desc->indexCount * sizeof(int32_t));
    } else if (desc->indexFormat == ommIndexFormat_UINT_16) {
        const int16_t *src = static_cast<const int16_t *>(desc->indexBuffer);
        for (uint32_t i = 0; i < desc->indexCount; i++) {
            result.indexBuffer[i] = static_cast<int32_t>(src[i]);
        }
    }

    // Copy desc array histogram (for VkMicromapEXT build)
    result.descArrayHistogram.resize(desc->descArrayHistogramCount);
    for (uint32_t i = 0; i < desc->descArrayHistogramCount; i++) {
        result.descArrayHistogram[i] = {
            desc->descArrayHistogram[i].count,
            desc->descArrayHistogram[i].subdivisionLevel,
            desc->descArrayHistogram[i].format,
        };
    }

    // Copy index histogram (for BLAS OMM attachment)
    result.indexHistogram.resize(desc->indexHistogramCount);
    for (uint32_t i = 0; i < desc->indexHistogramCount; i++) {
        result.indexHistogram[i] = {
            desc->indexHistogram[i].count,
            desc->indexHistogram[i].subdivisionLevel,
            desc->indexHistogram[i].format,
        };
    }

    ommCpuDestroyBakeResult(bakeResult);
    return true;
}

#endif // MCVR_ENABLE_OMM
