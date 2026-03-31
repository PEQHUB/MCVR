#include "core/render/anvil/region_reader.hpp"

// Use miniz for zlib decompression (compiled in miniz_impl.cpp)
#define MINIZ_NO_STDIO
#define MINIZ_NO_ARCHIVE_APIS
#define MINIZ_NO_ZLIB_COMPATIBLE_NAMES
#include "miniz.h"

#include <cstring>
#include <fstream>
#include <iostream>

namespace anvil {

static auto& regionCout() {
    static auto& s = std::cout;
    return s;
}

std::filesystem::path RegionReader::regionPath(const std::filesystem::path& regionDir,
                                               int32_t chunkX, int32_t chunkZ) {
    int32_t regionX = chunkX >> 5;
    int32_t regionZ = chunkZ >> 5;
    // Handle negative coordinates correctly (arithmetic right shift)
    if (chunkX < 0 && (chunkX & 31)) regionX--;
    if (chunkZ < 0 && (chunkZ & 31)) regionZ--;
    regionX = chunkX >> 5;
    regionZ = chunkZ >> 5;
    return regionDir / ("r." + std::to_string(regionX) + "." + std::to_string(regionZ) + ".mca");
}

bool RegionReader::regionExists(const std::filesystem::path& regionDir,
                                int32_t chunkX, int32_t chunkZ) {
    return std::filesystem::exists(regionPath(regionDir, chunkX, chunkZ));
}

std::vector<uint8_t> RegionReader::readChunk(const std::filesystem::path& regionDir,
                                             int32_t chunkX, int32_t chunkZ) {
    auto path = regionPath(regionDir, chunkX, chunkZ);
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) return {};

    // Local coordinates within the 32×32 region grid
    int32_t localX = ((chunkX % 32) + 32) % 32;
    int32_t localZ = ((chunkZ % 32) + 32) % 32;
    int32_t index = localX + localZ * 32;

    // Read location table entry (4 bytes at offset index*4)
    file.seekg(index * 4);
    uint8_t locBytes[4];
    file.read(reinterpret_cast<char*>(locBytes), 4);
    if (!file) return {};

    // Location: upper 3 bytes = sector offset, lowest byte = sector count
    uint32_t sectorOffset = (static_cast<uint32_t>(locBytes[0]) << 16) |
                            (static_cast<uint32_t>(locBytes[1]) << 8) |
                             static_cast<uint32_t>(locBytes[2]);
    uint8_t sectorCount = locBytes[3];

    if (sectorOffset == 0 && sectorCount == 0) return {}; // Chunk not present

    // Seek to chunk data (sector offset × 4096 bytes)
    file.seekg(static_cast<std::streamoff>(sectorOffset) * 4096);

    // Read chunk header: 4-byte big-endian length + 1-byte compression type
    uint8_t header[5];
    file.read(reinterpret_cast<char*>(header), 5);
    if (!file) return {};

    uint32_t dataLength = (static_cast<uint32_t>(header[0]) << 24) |
                          (static_cast<uint32_t>(header[1]) << 16) |
                          (static_cast<uint32_t>(header[2]) << 8) |
                           static_cast<uint32_t>(header[3]);
    uint8_t compressionType = header[4];
    dataLength--; // Subtract the compression type byte

    if (dataLength == 0 || dataLength > 16 * 1024 * 1024) return {}; // Sanity check: max 16MB

    // Read compressed data
    std::vector<uint8_t> compressed(dataLength);
    file.read(reinterpret_cast<char*>(compressed.data()), dataLength);
    if (!file) return {};

    // Decompress based on compression type
    // 1 = gzip, 2 = zlib (most common), 3 = uncompressed, 4 = lz4
    switch (compressionType) {
        case 2: // zlib
            return zlibDecompress(compressed.data(), compressed.size());
        case 3: // uncompressed
            return compressed;
        case 1: // gzip — same as zlib but with gzip header (miniz handles both)
            return zlibDecompress(compressed.data(), compressed.size());
        default:
            regionCout() << "[RegionReader] Unsupported compression type "
                         << static_cast<int>(compressionType)
                         << " for chunk (" << chunkX << "," << chunkZ << ")" << std::endl;
            return {};
    }
}

std::vector<uint8_t> RegionReader::zlibDecompress(const uint8_t* data, size_t compressedSize) {
    // Start with 4x estimate, grow if needed
    size_t outSize = compressedSize * 4;
    std::vector<uint8_t> output(outSize);

    mz_stream stream{};
    // MZ_DEFAULT_WINDOW_BITS handles both zlib and gzip headers
    if (mz_inflateInit2(&stream, MZ_DEFAULT_WINDOW_BITS) != MZ_OK) return {};

    stream.next_in = data;
    stream.avail_in = static_cast<unsigned int>(compressedSize);
    stream.next_out = output.data();
    stream.avail_out = static_cast<unsigned int>(outSize);

    while (true) {
        int status = mz_inflate(&stream, MZ_FINISH);
        if (status == MZ_STREAM_END) {
            output.resize(stream.total_out);
            mz_inflateEnd(&stream);
            return output;
        }
        if (status == MZ_BUF_ERROR || (status == MZ_OK && stream.avail_out == 0)) {
            // Need more output space
            size_t written = stream.total_out;
            outSize *= 2;
            output.resize(outSize);
            stream.next_out = output.data() + written;
            stream.avail_out = static_cast<unsigned int>(outSize - written);
            continue;
        }
        // Error
        mz_inflateEnd(&stream);
        return {};
    }
}

} // namespace anvil
