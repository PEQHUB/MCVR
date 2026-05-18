#include "core/render/anvil/nbt.hpp"
#include <cstring>
#include <stdexcept>

namespace anvil {

uint8_t NbtReader::readByte() {
    if (pos_ >= size_) throw std::runtime_error("NBT: unexpected EOF reading byte");
    return data_[pos_++];
}

int16_t NbtReader::readShort() {
    if (pos_ + 2 > size_) throw std::runtime_error("NBT: unexpected EOF reading short");
    // Big-endian
    int16_t v = static_cast<int16_t>((data_[pos_] << 8) | data_[pos_ + 1]);
    pos_ += 2;
    return v;
}

int32_t NbtReader::readInt() {
    if (pos_ + 4 > size_) throw std::runtime_error("NBT: unexpected EOF reading int");
    int32_t v = (static_cast<int32_t>(data_[pos_]) << 24) |
                (static_cast<int32_t>(data_[pos_ + 1]) << 16) |
                (static_cast<int32_t>(data_[pos_ + 2]) << 8) |
                 static_cast<int32_t>(data_[pos_ + 3]);
    pos_ += 4;
    return v;
}

int64_t NbtReader::readLong() {
    if (pos_ + 8 > size_) throw std::runtime_error("NBT: unexpected EOF reading long");
    int64_t v = 0;
    for (int i = 0; i < 8; i++) {
        v = (v << 8) | data_[pos_++];
    }
    return v;
}

float NbtReader::readFloat() {
    int32_t bits = readInt();
    float v;
    std::memcpy(&v, &bits, 4);
    return v;
}

double NbtReader::readDouble() {
    int64_t bits = readLong();
    double v;
    std::memcpy(&v, &bits, 8);
    return v;
}

std::string NbtReader::readString() {
    uint16_t len = static_cast<uint16_t>(readShort());
    if (pos_ + len > size_) throw std::runtime_error("NBT: unexpected EOF reading string");
    std::string s(reinterpret_cast<const char*>(data_ + pos_), len);
    pos_ += len;
    return s;
}

NbtTag NbtReader::readTag(TagType type) {
    NbtTag tag(type);
    switch (type) {
        case TagType::Byte:
            tag.byteVal = static_cast<int8_t>(readByte());
            break;
        case TagType::Short:
            tag.shortVal = readShort();
            break;
        case TagType::Int:
            tag.intVal = readInt();
            break;
        case TagType::Long:
            tag.longVal = readLong();
            break;
        case TagType::Float:
            tag.floatVal = readFloat();
            break;
        case TagType::Double:
            tag.doubleVal = readDouble();
            break;
        case TagType::ByteArray: {
            int32_t len = readInt();
            if (len < 0 || pos_ + len > size_) throw std::runtime_error("NBT: bad byte array length");
            tag.byteArray.resize(len);
            std::memcpy(tag.byteArray.data(), data_ + pos_, len);
            pos_ += len;
            break;
        }
        case TagType::String:
            tag.stringVal = readString();
            break;
        case TagType::List:
            tag = readList();
            break;
        case TagType::Compound:
            tag = readCompound();
            break;
        case TagType::IntArray: {
            int32_t len = readInt();
            if (len < 0) throw std::runtime_error("NBT: bad int array length");
            tag.intArray.resize(len);
            for (int32_t i = 0; i < len; i++) {
                tag.intArray[i] = readInt();
            }
            break;
        }
        case TagType::LongArray: {
            int32_t len = readInt();
            if (len < 0) throw std::runtime_error("NBT: bad long array length");
            tag.longArray.resize(len);
            for (int32_t i = 0; i < len; i++) {
                tag.longArray[i] = readLong();
            }
            break;
        }
        default:
            throw std::runtime_error("NBT: unknown tag type " + std::to_string(static_cast<int>(type)));
    }
    return tag;
}

NbtTag NbtReader::readCompound() {
    NbtTag tag(TagType::Compound);
    while (true) {
        uint8_t typeId = readByte();
        if (typeId == 0) break; // TAG_End
        TagType childType = static_cast<TagType>(typeId);
        std::string name = readString();
        tag.compounds[std::move(name)] = readTag(childType);
    }
    return tag;
}

NbtTag NbtReader::readList() {
    NbtTag tag(TagType::List);
    uint8_t elemType = readByte();
    tag.listElementType = static_cast<TagType>(elemType);
    int32_t len = readInt();
    if (len <= 0) return tag;
    tag.listItems.reserve(len);
    for (int32_t i = 0; i < len; i++) {
        tag.listItems.push_back(readTag(tag.listElementType));
    }
    return tag;
}

NbtTag NbtReader::parse(const uint8_t* data, size_t size) {
    NbtReader reader(data, size);

    // Root tag: type byte + name + payload
    uint8_t rootType = reader.readByte();
    if (rootType != static_cast<uint8_t>(TagType::Compound)) {
        throw std::runtime_error("NBT: root tag must be Compound, got " + std::to_string(rootType));
    }
    reader.readString(); // root name (usually empty in modern MC)
    return reader.readCompound();
}

} // namespace anvil
