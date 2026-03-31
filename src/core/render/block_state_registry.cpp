#include "core/render/block_state_registry.hpp"
#include "core/render/anvil/region_reader.hpp"
#include "core/render/anvil/chunk_decoder.hpp"
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <sstream>

static auto& registryCout() {
    static auto& s = std::cout;
    return s;
}

void BlockStateRegistry::load(const uint8_t* data, uint32_t dataSize) {
    stringToId_.clear();
    idToString_.clear();

    if (dataSize < 4) return;

    uint32_t count;
    std::memcpy(&count, data, 4);
    const uint8_t* ptr = data + 4;
    const uint8_t* end = data + dataSize;

    stringToId_.reserve(count);
    idToString_.reserve(count);

    for (uint32_t i = 0; i < count && ptr + 6 <= end; i++) {
        uint32_t stateId;
        std::memcpy(&stateId, ptr, 4);
        ptr += 4;

        uint16_t strLen;
        std::memcpy(&strLen, ptr, 2);
        ptr += 2;

        if (ptr + strLen > end) break;

        std::string name(reinterpret_cast<const char*>(ptr), strLen);
        ptr += strLen;

        stringToId_[name] = stateId;
        idToString_[stateId] = name;
    }

    registryCout() << "[BlockStateRegistry] Loaded " << stringToId_.size()
                   << " block state mappings" << std::endl;

    // Spot-check: log a few known entries for verification
    auto check = [&](const char* name) {
        auto it = stringToId_.find(name);
        if (it != stringToId_.end()) {
            registryCout() << "  " << name << " -> stateId " << it->second << std::endl;
        } else {
            registryCout() << "  " << name << " -> NOT FOUND" << std::endl;
        }
    };
    check("minecraft:air");
    check("minecraft:stone");
    check("minecraft:grass_block[snowy=false]");
    check("minecraft:oak_stairs[facing=north,half=bottom,shape=straight,waterlogged=false]");

    // ---- Anvil reader smoke test ----
    // Try to read chunk (0,0) from the first save directory found.
    namespace fs = std::filesystem;
    fs::path savesRoot = "C:/Users/Administrator/AppData/Roaming/PrismLauncher/instances/1.21.4/minecraft/saves";
    if (fs::exists(savesRoot)) {
        for (auto& entry : fs::directory_iterator(savesRoot)) {
            fs::path regionDir = entry.path() / "region";
            if (!fs::exists(regionDir)) continue;

            registryCout() << "[AnvilTest] Testing with save: " << entry.path().filename() << std::endl;
            auto nbtData = anvil::RegionReader::readChunk(regionDir, 0, 0);
            if (nbtData.empty()) {
                registryCout() << "[AnvilTest] Chunk (0,0) not found in save" << std::endl;
                break;
            }
            registryCout() << "[AnvilTest] Decompressed NBT: " << nbtData.size() << " bytes" << std::endl;

            auto decoded = anvil::ChunkDecoder::decode(nbtData.data(), nbtData.size(), 0, 0, *this);
            registryCout() << "[AnvilTest] Decoded " << decoded.sections.size() << " sections" << std::endl;

            int nonEmpty = 0;
            for (auto& sec : decoded.sections) {
                if (!sec.empty) {
                    nonEmpty++;
                    // Count unique block states in this section
                    std::unordered_map<uint32_t, int> counts;
                    for (int i = 0; i < 4096; i++) counts[sec.blockStates[i]]++;

                    registryCout() << "  Y=" << sec.sectionY << ": "
                                   << counts.size() << " unique states, top: ";
                    // Show top 3 most common
                    std::vector<std::pair<int, uint32_t>> sorted;
                    for (auto& [id, cnt] : counts) sorted.push_back({cnt, id});
                    std::sort(sorted.rbegin(), sorted.rend());
                    for (int i = 0; i < std::min(3, (int)sorted.size()); i++) {
                        auto* name = getName(sorted[i].second);
                        registryCout() << (name ? *name : "?") << "×" << sorted[i].first;
                        if (i < 2 && i + 1 < (int)sorted.size()) registryCout() << ", ";
                    }
                    registryCout() << std::endl;
                }
            }
            registryCout() << "[AnvilTest] " << nonEmpty << " non-empty sections out of "
                           << decoded.sections.size() << std::endl;
            break; // Only test first save
        }
    }
}

uint32_t BlockStateRegistry::resolve(const std::string& paletteEntry) const {
    auto it = stringToId_.find(paletteEntry);
    return it != stringToId_.end() ? it->second : 0;
}

uint32_t BlockStateRegistry::resolve(const std::string& name,
                                     const std::vector<std::pair<std::string, std::string>>& properties) const {
    if (properties.empty()) {
        // Try name as-is first (default state without brackets)
        auto it = stringToId_.find(name);
        if (it != stringToId_.end()) return it->second;
    }
    return resolve(canonicalize(name, properties));
}

const std::string* BlockStateRegistry::getName(uint32_t stateId) const {
    auto it = idToString_.find(stateId);
    return it != idToString_.end() ? &it->second : nullptr;
}

std::string BlockStateRegistry::canonicalize(const std::string& name,
                                             const std::vector<std::pair<std::string, std::string>>& properties) {
    if (properties.empty()) return name;

    // Sort properties alphabetically by key (Minecraft's canonical order)
    auto sorted = properties;
    std::sort(sorted.begin(), sorted.end());

    std::ostringstream ss;
    ss << name << '[';
    for (size_t i = 0; i < sorted.size(); i++) {
        if (i > 0) ss << ',';
        ss << sorted[i].first << '=' << sorted[i].second;
    }
    ss << ']';
    return ss.str();
}
