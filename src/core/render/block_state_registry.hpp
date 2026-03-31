#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>
#include <mutex>

/// Maps block state string identifiers (as stored in Anvil region file palettes)
/// to Minecraft's runtime globalStateId (Block.getRawIdFromState()).
///
/// Region files store block palettes as strings like:
///   "minecraft:stone"
///   "minecraft:oak_stairs[facing=north,half=top,shape=straight]"
///
/// The C++ mesher (BlockMesher) and BlockModelTable are indexed by globalStateId.
/// This registry bridges the gap for C++-only chunk loading from disk.
///
/// Loaded once from Java at world init. Immutable after load — safe for concurrent reads.
class BlockStateRegistry {
  public:
    /// Load from serialized Java data.
    /// Binary format: [uint32_t count] then count × [uint32_t stateId][uint16_t strLen][char[] str]
    void load(const uint8_t* data, uint32_t dataSize);

    /// Look up a palette string → globalStateId. Returns 0 (air) if not found.
    uint32_t resolve(const std::string& paletteEntry) const;

    /// Look up by separate name + properties map (for NBT parsing convenience).
    /// name = "minecraft:stone", properties = {{"variant","granite"}}
    uint32_t resolve(const std::string& name,
                     const std::vector<std::pair<std::string, std::string>>& properties) const;

    /// Get the string for a globalStateId (reverse lookup, for debugging).
    const std::string* getName(uint32_t stateId) const;

    bool isLoaded() const { return !stringToId_.empty(); }
    uint32_t stateCount() const { return static_cast<uint32_t>(stringToId_.size()); }

  private:
    /// Canonical string → stateId. The string format is:
    ///   "namespace:block" (default state) or "namespace:block[prop=val,prop=val,...]"
    /// Properties are always sorted alphabetically (Minecraft's canonical ordering).
    std::unordered_map<std::string, uint32_t> stringToId_;

    /// Reverse: stateId → canonical string (for debugging).
    std::unordered_map<uint32_t, std::string> idToString_;

    /// Build canonical string from name + sorted properties.
    static std::string canonicalize(const std::string& name,
                                    const std::vector<std::pair<std::string, std::string>>& properties);
};
