#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <stdexcept>

/// Minimal NBT (Named Binary Tag) parser for Minecraft chunk data.
/// Supports all tag types needed for block_states/biomes palette decoding.
/// Read-only, zero-copy where possible, no external dependencies.

namespace anvil {

enum class TagType : uint8_t {
    End = 0,
    Byte = 1,
    Short = 2,
    Int = 3,
    Long = 4,
    Float = 5,
    Double = 6,
    ByteArray = 7,
    String = 8,
    List = 9,
    Compound = 10,
    IntArray = 11,
    LongArray = 12,
};

class NbtTag {
  public:
    TagType type;

    // Scalar values
    int8_t byteVal = 0;
    int16_t shortVal = 0;
    int32_t intVal = 0;
    int64_t longVal = 0;
    float floatVal = 0;
    double doubleVal = 0;
    std::string stringVal;

    // Array values
    std::vector<int8_t> byteArray;
    std::vector<int32_t> intArray;
    std::vector<int64_t> longArray;

    // List: homogeneous typed children
    TagType listElementType = TagType::End;
    std::vector<NbtTag> listItems;

    // Compound: named children
    std::unordered_map<std::string, NbtTag> compounds;

    NbtTag() : type(TagType::End) {}
    explicit NbtTag(TagType t) : type(t) {}

    // Convenience accessors
    const NbtTag* get(const std::string& key) const {
        auto it = compounds.find(key);
        return it != compounds.end() ? &it->second : nullptr;
    }

    int8_t getByte(const std::string& key, int8_t def = 0) const {
        auto t = get(key);
        return (t && t->type == TagType::Byte) ? t->byteVal : def;
    }

    int32_t getInt(const std::string& key, int32_t def = 0) const {
        auto t = get(key);
        return (t && t->type == TagType::Int) ? t->intVal : def;
    }

    const std::string& getString(const std::string& key) const {
        static const std::string empty;
        auto t = get(key);
        return (t && t->type == TagType::String) ? t->stringVal : empty;
    }

    const NbtTag* getList(const std::string& key) const {
        auto t = get(key);
        return (t && t->type == TagType::List) ? t : nullptr;
    }

    const NbtTag* getCompound(const std::string& key) const {
        auto t = get(key);
        return (t && t->type == TagType::Compound) ? t : nullptr;
    }

    const std::vector<int64_t>& getLongArray(const std::string& key) const {
        static const std::vector<int64_t> empty;
        auto t = get(key);
        return (t && t->type == TagType::LongArray) ? t->longArray : empty;
    }
};

/// Parse NBT from raw (already decompressed) bytes.
/// Returns the root compound tag. Throws on malformed data.
class NbtReader {
  public:
    /// Parse a complete NBT document (starts with tag type byte + root name + root compound).
    static NbtTag parse(const uint8_t* data, size_t size);

  private:
    const uint8_t* data_;
    size_t size_;
    size_t pos_;

    NbtReader(const uint8_t* data, size_t size) : data_(data), size_(size), pos_(0) {}

    uint8_t readByte();
    int16_t readShort();
    int32_t readInt();
    int64_t readLong();
    float readFloat();
    double readDouble();
    std::string readString();
    NbtTag readTag(TagType type);
    NbtTag readCompound();
    NbtTag readList();
};

} // namespace anvil
