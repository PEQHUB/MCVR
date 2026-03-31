#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace anvil {

/// Reads Minecraft Anvil region files (.mca).
/// Each region file covers a 32×32 grid of chunk columns.
/// File layout: 4KB location table + 4KB timestamp table + chunk sectors (4KB aligned).
class RegionReader {
  public:
    /// Read a single chunk's raw (decompressed) NBT data from a region file.
    /// chunkX/chunkZ are GLOBAL chunk coordinates (not region-local).
    /// Returns empty vector if chunk is not present in the region file.
    static std::vector<uint8_t> readChunk(const std::filesystem::path& regionDir,
                                          int32_t chunkX, int32_t chunkZ);

    /// Get the region file path for a given chunk coordinate.
    /// Region coords = chunk coords >> 5 (i.e., floor divide by 32).
    static std::filesystem::path regionPath(const std::filesystem::path& regionDir,
                                            int32_t chunkX, int32_t chunkZ);

    /// Check if a region file exists for the given chunk coordinates.
    static bool regionExists(const std::filesystem::path& regionDir,
                             int32_t chunkX, int32_t chunkZ);

  private:
    /// Decompress zlib-compressed data. Returns empty on failure.
    static std::vector<uint8_t> zlibDecompress(const uint8_t* data, size_t compressedSize);
};

} // namespace anvil
