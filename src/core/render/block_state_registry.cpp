#include "core/render/block_state_registry.hpp"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <sstream>

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

    std::cout << "[BlockStateRegistry] Loaded " << stringToId_.size()
              << " block state mappings" << std::endl;
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
