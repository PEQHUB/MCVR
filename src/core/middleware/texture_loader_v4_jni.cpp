#include <jni.h>
#include "core/render/texture_loader_v4.hpp"
#include "core/render/renderer.hpp"
#include "core/render/render_framework.hpp"
#include "core/build/build_info.hpp"
#include "mz.h"
#include "mz_strm.h"
#include "mz_zip.h"
#include "mz_zip_rw.h"
#define XXH_INLINE_ALL
#include "xxhash.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <cstring>
#include <climits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

using json = nlohmann::json;

jstring makeString(JNIEnv* env, const std::string& value) {
    return env->NewStringUTF(value.c_str());
}

std::string readJString(JNIEnv* env, jstring value) {
    if (!value) return {};
    const char* chars = env->GetStringUTFChars(value, nullptr);
    if (!chars) return {};
    std::string out(chars);
    env->ReleaseStringUTFChars(value, chars);
    return out;
}

std::string jsonEscape(const std::string& value) {
    std::ostringstream out;
    for (char c : value) {
        switch (c) {
            case '\\': out << "\\\\"; break;
            case '"': out << "\\\""; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    const auto uc = static_cast<unsigned char>(c);
                    out << "\\u";
                    static const char* HEX = "0123456789abcdef";
                    out << "00" << HEX[(uc >> 4) & 0xF] << HEX[uc & 0xF];
                } else {
                    out << c;
                }
                break;
        }
    }
    return out.str();
}

struct PackIndexSnapshotState {
    bool hasSnapshot = false;
    bool diffReady = false;
    uint64_t generation = 0;
    uint64_t submitCount = 0;
    size_t snapshotBytes = 0;
    uint32_t packCount = 0;
    uint32_t resourceCount = 0;
    uint32_t ruleFileCount = 0;
    uint32_t sidecarCount = 0;
    uint64_t javaCaptureMillis = 0;
    uint64_t nativeIndexMillis = 0;
    uint32_t activeDiskPackCount = 0;
    uint32_t nativeResourceCount = 0;
    uint32_t nativeSidecarCount = 0;
    uint32_t nativeRuleFileCount = 0;
    uint32_t virtualJavaOnlyResourceCount = 0;
    uint32_t missingCount = 0;
    uint32_t winnerMismatchCount = 0;
    uint32_t flagMismatchCount = 0;
    uint32_t extraNativeCount = 0;
    uint32_t nativeContentHashReadyCount = 0;
    uint32_t nativeContentHashMissingCount = 0;
    uint32_t diskBackedContentHashReadyCount = 0;
    uint32_t diskBackedContentHashMissingCount = 0;
    uint32_t virtualJavaOnlyContentHashMissingCount = 0;
    std::string packStackHash;
    std::string indexError;
    std::string diffJson = "{\"ok\":false,\"reason\":\"not_captured\"}";
};

std::mutex g_packIndexMutex;
PackIndexSnapshotState g_packIndexState;

struct PackIndexResource {
    std::string resource;
    std::string winnerPackId;
    bool sidecar = false;
    bool ruleFile = false;
    bool virtualJavaOnly = false;
    bool contentHashReady = false;
    uint64_t contentHash64 = 0;
    uint64_t contentSizeBytes = 0;
    int precedence = -1;
};

struct PackIndexBuildResult {
    bool ok = true;
    bool diffReady = false;
    uint64_t nativeIndexMillis = 0;
    uint32_t activeDiskPackCount = 0;
    uint32_t nativeResourceCount = 0;
    uint32_t nativeSidecarCount = 0;
    uint32_t nativeRuleFileCount = 0;
    uint32_t virtualJavaOnlyResourceCount = 0;
    uint32_t missingCount = 0;
    uint32_t winnerMismatchCount = 0;
    uint32_t flagMismatchCount = 0;
    uint32_t extraNativeCount = 0;
    uint32_t nativeContentHashReadyCount = 0;
    uint32_t nativeContentHashMissingCount = 0;
    uint32_t diskBackedContentHashReadyCount = 0;
    uint32_t diskBackedContentHashMissingCount = 0;
    uint32_t virtualJavaOnlyContentHashMissingCount = 0;
    std::string error;
    std::string diffJson;
};

bool startsWith(std::string_view value, std::string_view prefix) {
    return value.size() >= prefix.size() && value.substr(0, prefix.size()) == prefix;
}

bool endsWith(std::string_view value, std::string_view suffix) {
    return value.size() >= suffix.size() && value.substr(value.size() - suffix.size()) == suffix;
}

std::string toLowerAscii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
        [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return value;
}

std::string normalizeSlash(std::string value) {
    std::replace(value.begin(), value.end(), '\\', '/');
    return value;
}

std::string hex64(uint64_t value) {
    std::ostringstream out;
    out << std::hex << std::nouppercase << std::setfill('0') << std::setw(16) << value;
    return out.str();
}

class Xxh3Stream {
public:
    Xxh3Stream() : state_(XXH3_createState()) {
        if (!state_ || XXH3_64bits_reset(state_) == XXH_ERROR) {
            throw std::runtime_error("XXH3 state initialization failed");
        }
    }

    ~Xxh3Stream() {
        XXH3_freeState(state_);
    }

    void update(const void* data, size_t length) {
        if (length == 0) return;
        if (XXH3_64bits_update(state_, data, length) == XXH_ERROR) {
            throw std::runtime_error("XXH3 update failed");
        }
        size_ += static_cast<uint64_t>(length);
    }

    uint64_t digest() const {
        return static_cast<uint64_t>(XXH3_64bits_digest(state_));
    }

    uint64_t size() const {
        return size_;
    }

private:
    XXH3_state_t* state_ = nullptr;
    uint64_t size_ = 0;
};

void applyContentHash(PackIndexResource& resource, uint64_t hash64, uint64_t sizeBytes) {
    resource.contentHashReady = true;
    resource.contentHash64 = hash64;
    resource.contentSizeBytes = sizeBytes;
}

bool isSidecarResourcePath(std::string_view path) {
    return endsWith(path, "_n.png")
        || endsWith(path, "_normal.png")
        || endsWith(path, "_norm.png")
        || endsWith(path, "_s.png")
        || endsWith(path, "_spec.png")
        || endsWith(path, "_specular.png")
        || endsWith(path, "_e.png")
        || endsWith(path, "_emissive.png")
        || endsWith(path, "_f.png")
        || endsWith(path, "_roughness.png")
        || endsWith(path, "_rough.png")
        || endsWith(path, "_metallic.png")
        || endsWith(path, "_metalness.png")
        || endsWith(path, "_height.png")
        || endsWith(path, "_displacement.png")
        || endsWith(path, "_disp.png")
        || endsWith(path, "_ao.png")
        || endsWith(path, "_ambientocclusion.png")
        || endsWith(path, "_ambient_occlusion.png");
}

std::optional<PackIndexResource> resourceFromAssetPath(const std::string& rawPath,
    const std::string& packId, int precedence) {
    std::string path = toLowerAscii(normalizeSlash(rawPath));
    if (endsWith(path, "/")) {
        return std::nullopt;
    }
    static constexpr std::string_view ASSETS = "assets/";
    if (!startsWith(path, ASSETS)) {
        return std::nullopt;
    }
    size_t namespaceStart = ASSETS.size();
    size_t namespaceEnd = path.find('/', namespaceStart);
    if (namespaceEnd == std::string::npos || namespaceEnd == namespaceStart || namespaceEnd + 1 >= path.size()) {
        return std::nullopt;
    }
    std::string ns = path.substr(namespaceStart, namespaceEnd - namespaceStart);
    std::string resourcePath = path.substr(namespaceEnd + 1);
    bool interestingTexture = startsWith(resourcePath, "textures/")
        && (endsWith(resourcePath, ".png") || endsWith(resourcePath, ".png.mcmeta"));
    bool ruleFile = (startsWith(resourcePath, "optifine/") || startsWith(resourcePath, "mcpatcher/"))
        && endsWith(resourcePath, ".properties");
    if (!interestingTexture && !ruleFile) {
        return std::nullopt;
    }
    PackIndexResource out;
    out.resource = ns + ":" + resourcePath;
    out.winnerPackId = packId;
    out.sidecar = interestingTexture && isSidecarResourcePath(resourcePath);
    out.ruleFile = ruleFile;
    out.precedence = precedence;
    return out;
}

void mergeResource(std::unordered_map<std::string, PackIndexResource>& table, PackIndexResource resource) {
    auto it = table.find(resource.resource);
    if (it == table.end() || resource.precedence >= it->second.precedence) {
        table[resource.resource] = std::move(resource);
    }
}

void hashCurrentZipEntry(void* zipReader, const std::string& packId, const std::string& filename,
    PackIndexResource& resource, std::vector<std::string>& errors) {
    int32_t openResult = mz_zip_reader_entry_open(zipReader);
    if (openResult != MZ_OK) {
        errors.push_back(packId + ": open zip entry failed " + std::to_string(openResult) + " " + filename);
        return;
    }

    bool closeNeeded = true;
    try {
        Xxh3Stream stream;
        std::array<char, 64 * 1024> buffer{};
        for (;;) {
            int32_t read = mz_zip_reader_entry_read(zipReader, buffer.data(), static_cast<int32_t>(buffer.size()));
            if (read > 0) {
                stream.update(buffer.data(), static_cast<size_t>(read));
                continue;
            }
            if (read < 0) {
                errors.push_back(packId + ": read zip entry failed " + std::to_string(read) + " " + filename);
            }
            break;
        }

        int32_t closeResult = mz_zip_reader_entry_close(zipReader);
        closeNeeded = false;
        if (closeResult != MZ_OK) {
            errors.push_back(packId + ": close zip entry failed " + std::to_string(closeResult) + " " + filename);
            return;
        }
        applyContentHash(resource, stream.digest(), stream.size());
    } catch (const std::exception& ex) {
        errors.push_back(packId + ": hash zip entry failed " + std::string(ex.what()) + " " + filename);
    }

    if (closeNeeded) {
        mz_zip_reader_entry_close(zipReader);
    }
}

void hashDirectoryFile(const std::string& packId, const std::filesystem::path& path,
    PackIndexResource& resource, std::vector<std::string>& errors) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        errors.push_back(packId + ": open file for hash failed " + path.string());
        return;
    }

    try {
        Xxh3Stream stream;
        std::array<char, 64 * 1024> buffer{};
        while (file) {
            file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
            std::streamsize got = file.gcount();
            if (got > 0) {
                stream.update(buffer.data(), static_cast<size_t>(got));
            }
        }
        if (!file.eof()) {
            errors.push_back(packId + ": read file for hash failed " + path.string());
            return;
        }
        applyContentHash(resource, stream.digest(), stream.size());
    } catch (const std::exception& ex) {
        errors.push_back(packId + ": hash file failed " + std::string(ex.what()) + " " + path.string());
    }
}

uint32_t countSidecars(const std::unordered_map<std::string, PackIndexResource>& table) {
    uint32_t count = 0;
    for (const auto& [_, resource] : table) {
        if (resource.sidecar) count++;
    }
    return count;
}

uint32_t countRuleFiles(const std::unordered_map<std::string, PackIndexResource>& table) {
    uint32_t count = 0;
    for (const auto& [_, resource] : table) {
        if (resource.ruleFile) count++;
    }
    return count;
}

void countContentHashes(const std::unordered_map<std::string, PackIndexResource>& table,
    uint32_t& ready, uint32_t& missing, uint32_t& diskReady, uint32_t& diskMissing, uint32_t& virtualMissing) {
    ready = 0;
    missing = 0;
    diskReady = 0;
    diskMissing = 0;
    virtualMissing = 0;
    for (const auto& [_, resource] : table) {
        if (resource.contentHashReady) {
            ready++;
            if (!resource.virtualJavaOnly) {
                diskReady++;
            }
            continue;
        }
        missing++;
        if (resource.virtualJavaOnly) {
            virtualMissing++;
        } else {
            diskMissing++;
        }
    }
}

void scanZipPack(const std::string& packId, const std::filesystem::path& path, int precedence,
    std::unordered_map<std::string, PackIndexResource>& table, std::vector<std::string>& errors) {
    std::shared_ptr<void> zipReader(mz_zip_reader_create(), [](void* handle) {
        void* zipReaderHandle = handle;
        mz_zip_reader_delete(&zipReaderHandle);
    });
    if (!zipReader) {
        errors.push_back(packId + ": create zip reader failed");
        return;
    }
    int32_t openResult = mz_zip_reader_open_file(zipReader.get(), path.string().c_str());
    if (openResult != MZ_OK) {
        errors.push_back(packId + ": open zip failed " + std::to_string(openResult));
        return;
    }
    int32_t entryResult = mz_zip_reader_goto_first_entry(zipReader.get());
    while (entryResult == MZ_OK) {
        mz_zip_file* info = nullptr;
        if (mz_zip_reader_entry_get_info(zipReader.get(), &info) == MZ_OK
            && info != nullptr && info->filename != nullptr) {
            if (auto resource = resourceFromAssetPath(info->filename, packId, precedence)) {
                hashCurrentZipEntry(zipReader.get(), packId, info->filename, *resource, errors);
                mergeResource(table, std::move(*resource));
            }
        }
        entryResult = mz_zip_reader_goto_next_entry(zipReader.get());
    }
    int32_t closeResult = mz_zip_reader_close(zipReader.get());
    if (closeResult != MZ_OK) {
        errors.push_back(packId + ": close zip failed " + std::to_string(closeResult));
    }
}

void scanDirectoryPack(const std::string& packId, const std::filesystem::path& root, int precedence,
    std::unordered_map<std::string, PackIndexResource>& table, std::vector<std::string>& errors) {
    std::error_code ec;
    std::filesystem::recursive_directory_iterator it(root,
        std::filesystem::directory_options::skip_permission_denied, ec);
    std::filesystem::recursive_directory_iterator end;
    if (ec) {
        errors.push_back(packId + ": open directory failed " + ec.message());
        return;
    }
    for (; it != end; it.increment(ec)) {
        if (ec) {
            errors.push_back(packId + ": directory walk failed " + ec.message());
            ec.clear();
            continue;
        }
        if (!it->is_regular_file(ec) || ec) {
            ec.clear();
            continue;
        }
        std::filesystem::path relative = std::filesystem::relative(it->path(), root, ec);
        if (ec) {
            errors.push_back(packId + ": relative path failed " + ec.message());
            ec.clear();
            continue;
        }
        if (auto resource = resourceFromAssetPath(relative.string(), packId, precedence)) {
            hashDirectoryFile(packId, it->path(), *resource, errors);
            mergeResource(table, std::move(*resource));
        }
    }
}

std::string jsonString(const json& object, const char* key) {
    auto it = object.find(key);
    return it != object.end() && it->is_string() ? it->get<std::string>() : std::string();
}

bool jsonBool(const json& object, const char* key, bool fallback = false) {
    auto it = object.find(key);
    return it != object.end() && it->is_boolean() ? it->get<bool>() : fallback;
}

int jsonInt(const json& object, const char* key, int fallback = 0) {
    auto it = object.find(key);
    return it != object.end() && it->is_number_integer() ? it->get<int>() : fallback;
}

json resourceJson(const PackIndexResource& resource) {
    json out = json::object();
    out["resource"] = resource.resource;
    out["winnerPackId"] = resource.winnerPackId;
    out["sidecar"] = resource.sidecar;
    out["ruleFile"] = resource.ruleFile;
    out["virtualJavaOnly"] = resource.virtualJavaOnly;
    out["contentHashReady"] = resource.contentHashReady;
    if (resource.contentHashReady) {
        out["contentHashAlgorithm"] = "xxh3_64";
        out["contentHash64"] = hex64(resource.contentHash64);
        out["contentSizeBytes"] = resource.contentSizeBytes;
    }
    return out;
}

PackIndexBuildResult buildPackIndexFromSnapshot(const std::string& snapshotJson) {
    auto started = std::chrono::steady_clock::now();
    PackIndexBuildResult result;
    json snapshot = json::parse(snapshotJson);
    std::unordered_map<std::string, PackIndexResource> truth;
    std::unordered_map<std::string, PackIndexResource> native;
    std::vector<std::string> errors;

    const json* truthResources = nullptr;
    if (snapshot.contains("vanillaTruth") && snapshot["vanillaTruth"].is_object()
        && snapshot["vanillaTruth"].contains("resources")
        && snapshot["vanillaTruth"]["resources"].is_array()) {
        truthResources = &snapshot["vanillaTruth"]["resources"];
    }
    if (truthResources == nullptr) {
        throw std::runtime_error("snapshot missing vanillaTruth.resources");
    }

    for (const auto& item : *truthResources) {
        if (!item.is_object()) continue;
        PackIndexResource resource;
        resource.resource = jsonString(item, "resource");
        resource.winnerPackId = jsonString(item, "winnerPackId");
        resource.sidecar = jsonBool(item, "sidecar");
        resource.ruleFile = jsonBool(item, "ruleFile");
        resource.precedence = startsWith(resource.winnerPackId, "file/") ? -1 : 0;
        if (resource.resource.empty()) continue;
        truth[resource.resource] = resource;
        if (!startsWith(resource.winnerPackId, "file/")) {
            resource.virtualJavaOnly = true;
            native[resource.resource] = resource;
            result.virtualJavaOnlyResourceCount++;
        }
    }

    if (snapshot.contains("packs") && snapshot["packs"].is_array()) {
        for (const auto& item : snapshot["packs"]) {
            if (!item.is_object() || !jsonBool(item, "active")) {
                continue;
            }
            std::string sourceKind = jsonString(item, "sourceKind");
            if (sourceKind != "zip" && sourceKind != "directory") {
                continue;
            }
            std::string packId = jsonString(item, "id");
            std::string path = jsonString(item, "path");
            int precedence = jsonInt(item, "nativePrecedence", -1);
            if (packId.empty() || path.empty() || precedence < 0) {
                continue;
            }
            result.activeDiskPackCount++;
            if (sourceKind == "zip") {
                scanZipPack(packId, std::filesystem::path(path), precedence, native, errors);
            } else {
                scanDirectoryPack(packId, std::filesystem::path(path), precedence, native, errors);
            }
        }
    }

    json diff = json::object();
    diff["ok"] = true;
    diff["schema"] = "radser_pack_index_diff_v1";
    diff["generation"] = snapshot.value("generation", 0);
    diff["nativeAcceptedSnapshot"] = true;
    diff["diffReady"] = true;
    diff["indexMode"] = "native_l1_disk_plus_virtual_java";
    diff["nativeCentralDirectoryScan"] = true;

    json samples = json::array();
    for (const auto& [resourceId, expected] : truth) {
        auto actualIt = native.find(resourceId);
        if (actualIt == native.end()) {
            result.missingCount++;
            if (samples.size() < 32) {
                json sample = json::object();
                sample["kind"] = "missing_native_resource";
                sample["expected"] = resourceJson(expected);
                samples.push_back(sample);
            }
            continue;
        }
        const PackIndexResource& actual = actualIt->second;
        bool winnerMismatch = expected.winnerPackId != actual.winnerPackId;
        bool flagMismatch = expected.sidecar != actual.sidecar || expected.ruleFile != actual.ruleFile;
        if (winnerMismatch) result.winnerMismatchCount++;
        if (flagMismatch) result.flagMismatchCount++;
        if ((winnerMismatch || flagMismatch) && samples.size() < 32) {
            json sample = json::object();
            sample["kind"] = winnerMismatch ? "winner_mismatch" : "flag_mismatch";
            sample["expected"] = resourceJson(expected);
            sample["actual"] = resourceJson(actual);
            samples.push_back(sample);
        }
    }
    for (const auto& [resourceId, actual] : native) {
        if (truth.find(resourceId) == truth.end()) {
            result.extraNativeCount++;
            if (samples.size() < 32) {
                json sample = json::object();
                sample["kind"] = "extra_native_resource";
                sample["actual"] = resourceJson(actual);
                samples.push_back(sample);
            }
        }
    }

    result.nativeResourceCount = static_cast<uint32_t>(native.size());
    result.nativeSidecarCount = countSidecars(native);
    result.nativeRuleFileCount = countRuleFiles(native);
    countContentHashes(native,
        result.nativeContentHashReadyCount,
        result.nativeContentHashMissingCount,
        result.diskBackedContentHashReadyCount,
        result.diskBackedContentHashMissingCount,
        result.virtualJavaOnlyContentHashMissingCount);
    result.diffReady = true;
    result.ok = errors.empty();
    auto elapsed = std::chrono::steady_clock::now() - started;
    result.nativeIndexMillis =
        static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count());

    diff["diffCount"] = result.missingCount + result.winnerMismatchCount
        + result.flagMismatchCount + result.extraNativeCount;
    diff["missingCount"] = result.missingCount;
    diff["winnerMismatchCount"] = result.winnerMismatchCount;
    diff["flagMismatchCount"] = result.flagMismatchCount;
    diff["extraNativeCount"] = result.extraNativeCount;
    diff["contentHashReady"] = result.nativeContentHashMissingCount == 0;
    diff["contentHashAlgorithm"] = "xxh3_64";
    diff["nativeContentHashReadyCount"] = result.nativeContentHashReadyCount;
    diff["nativeContentHashMissingCount"] = result.nativeContentHashMissingCount;
    diff["diskBackedContentHashReady"] = result.diskBackedContentHashMissingCount == 0;
    diff["diskBackedContentHashReadyCount"] = result.diskBackedContentHashReadyCount;
    diff["diskBackedContentHashMissingCount"] = result.diskBackedContentHashMissingCount;
    diff["virtualJavaOnlyContentHashMissingCount"] = result.virtualJavaOnlyContentHashMissingCount;
    diff["nativeTotals"] = {
        {"resourceCount", result.nativeResourceCount},
        {"sidecarResources", result.nativeSidecarCount},
        {"ruleFiles", result.nativeRuleFileCount},
        {"activeDiskPackCount", result.activeDiskPackCount},
        {"virtualJavaOnlyResourceCount", result.virtualJavaOnlyResourceCount},
        {"contentHashReadyCount", result.nativeContentHashReadyCount},
        {"contentHashMissingCount", result.nativeContentHashMissingCount},
        {"diskBackedContentHashReadyCount", result.diskBackedContentHashReadyCount},
        {"diskBackedContentHashMissingCount", result.diskBackedContentHashMissingCount},
        {"virtualJavaOnlyContentHashMissingCount", result.virtualJavaOnlyContentHashMissingCount},
        {"nativeIndexMillis", result.nativeIndexMillis}
    };
    diff["vanillaTotals"] = snapshot.value("totals", json::object());
    diff["samples"] = samples;
    diff["errors"] = errors;
    result.error = errors.empty() ? "" : "pack_scan_errors";
    result.diffJson = diff.dump();
    return result;
}

std::string packIndexStatusJson() {
    std::lock_guard<std::mutex> lock(g_packIndexMutex);
    std::ostringstream out;
    out << "{";
    out << "\"ok\":true";
    out << ",\"schema\":\"radser_native_pack_index_status_v1\"";
    out << ",\"indexMode\":\"native_l1_disk_plus_virtual_java\"";
    out << ",\"hasSnapshot\":" << (g_packIndexState.hasSnapshot ? "true" : "false");
    out << ",\"diffReady\":" << (g_packIndexState.diffReady ? "true" : "false");
    out << ",\"generation\":" << g_packIndexState.generation;
    out << ",\"submitCount\":" << g_packIndexState.submitCount;
    out << ",\"snapshotBytes\":" << g_packIndexState.snapshotBytes;
    out << ",\"packCount\":" << g_packIndexState.packCount;
    out << ",\"resourceCount\":" << g_packIndexState.resourceCount;
    out << ",\"ruleFileCount\":" << g_packIndexState.ruleFileCount;
    out << ",\"sidecarCount\":" << g_packIndexState.sidecarCount;
    out << ",\"javaCaptureMillis\":" << g_packIndexState.javaCaptureMillis;
    out << ",\"nativeIndexMillis\":" << g_packIndexState.nativeIndexMillis;
    out << ",\"activeDiskPackCount\":" << g_packIndexState.activeDiskPackCount;
    out << ",\"nativeResourceCount\":" << g_packIndexState.nativeResourceCount;
    out << ",\"nativeSidecarCount\":" << g_packIndexState.nativeSidecarCount;
    out << ",\"nativeRuleFileCount\":" << g_packIndexState.nativeRuleFileCount;
    out << ",\"virtualJavaOnlyResourceCount\":" << g_packIndexState.virtualJavaOnlyResourceCount;
    out << ",\"missingCount\":" << g_packIndexState.missingCount;
    out << ",\"winnerMismatchCount\":" << g_packIndexState.winnerMismatchCount;
    out << ",\"flagMismatchCount\":" << g_packIndexState.flagMismatchCount;
    out << ",\"extraNativeCount\":" << g_packIndexState.extraNativeCount;
    out << ",\"contentHashReady\":" << (g_packIndexState.nativeContentHashMissingCount == 0 ? "true" : "false");
    out << ",\"contentHashAlgorithm\":\"xxh3_64\"";
    out << ",\"nativeContentHashReadyCount\":" << g_packIndexState.nativeContentHashReadyCount;
    out << ",\"nativeContentHashMissingCount\":" << g_packIndexState.nativeContentHashMissingCount;
    out << ",\"diskBackedContentHashReady\":"
        << (g_packIndexState.diskBackedContentHashMissingCount == 0 ? "true" : "false");
    out << ",\"diskBackedContentHashReadyCount\":" << g_packIndexState.diskBackedContentHashReadyCount;
    out << ",\"diskBackedContentHashMissingCount\":" << g_packIndexState.diskBackedContentHashMissingCount;
    out << ",\"virtualJavaOnlyContentHashMissingCount\":"
        << g_packIndexState.virtualJavaOnlyContentHashMissingCount;
    out << ",\"packStackHash\":\"" << jsonEscape(g_packIndexState.packStackHash) << "\"";
    out << ",\"indexError\":\"" << jsonEscape(g_packIndexState.indexError) << "\"";
    out << "}";
    return out.str();
}

std::string packIndexDiffJson() {
    std::lock_guard<std::mutex> lock(g_packIndexMutex);
    return g_packIndexState.diffJson;
}

void logNativeException(const char* method, jlong generation, const std::exception& ex) {
    std::cerr << "[TextureLoaderV4JNI] " << method << " caught native exception";
    if (generation > 0) {
        std::cerr << " generation=" << generation;
    }
    std::cerr << " what=\"" << ex.what() << "\"" << std::endl;
}

void logNativeException(const char* method, jlong generation) {
    std::cerr << "[TextureLoaderV4JNI] " << method << " caught unknown native exception";
    if (generation > 0) {
        std::cerr << " generation=" << generation;
    }
    std::cerr << std::endl;
}

jstring makeNativeExceptionJson(JNIEnv* env, const char* method) {
    std::ostringstream out;
    out << "{\"error\":\"native_exception\",\"method\":\"" << method << "\"}";
    return makeString(env, out.str());
}

template <typename Fn>
jboolean guardedJniBool(const char* method, jlong generation, Fn fn) {
    try {
        return fn();
    } catch (const std::exception& ex) {
        logNativeException(method, generation, ex);
    } catch (...) {
        logNativeException(method, generation);
    }
    return JNI_FALSE;
}

template <typename Fn>
jstring guardedJniString(JNIEnv* env, const char* method, Fn fn) {
    try {
        return fn();
    } catch (const std::exception& ex) {
        logNativeException(method, 0, ex);
    } catch (...) {
        logNativeException(method, 0);
    }
    return makeNativeExceptionJson(env, method);
}

static constexpr uint32_t CHANNEL_ALBEDO   = 1u << 0;
static constexpr uint32_t CHANNEL_SPECULAR = 1u << 1;
static constexpr uint32_t CHANNEL_NORMAL   = 1u << 2;
static constexpr uint32_t CHANNEL_FLAG     = 1u << 3;

uint32_t tierSizePixels(uint32_t tier) {
    static const uint32_t SIZES[] = {16, 32, 64, 128, 256, 512, 1024};
    if (tier >= 7) return 0;
    return SIZES[tier];
}

uint32_t descriptorPageForV4(uint32_t namespaceId, uint32_t tier,
    uint32_t page, uint32_t startLayer, uint32_t* nativeStartLayer,
    uint32_t* nativeCapacity) {
    if (nativeStartLayer) *nativeStartLayer = startLayer;
    if (nativeCapacity) *nativeCapacity = 0;

    if (namespaceId == 1u) {
        const uint32_t capacity = TexturePagePool::pageLayerCapacityStatic(tier);
        if (capacity == 0) return UINT32_MAX;
        const uint32_t nativePage = startLayer / capacity;
        if (nativeStartLayer) *nativeStartLayer = startLayer % capacity;
        if (nativeCapacity) *nativeCapacity = capacity;
        return 1u + tier * 8u + nativePage;
    }

    if (namespaceId == 2u) {
        const uint32_t capacity = TexturePagePool::pageLayerCapacityStatic(tier);
        if (capacity == 0) return UINT32_MAX;
        const uint32_t basePage = page >= 8u ? page - 8u : page;
        const uint32_t nativePage = basePage + startLayer / capacity;
        if (nativeStartLayer) *nativeStartLayer = startLayer % capacity;
        if (nativeCapacity) *nativeCapacity = capacity;
        return 64u + nativePage;
    }

    return page;
}

uint32_t nullPlaneMask(jlong albedoPtr, jlong specularPtr, jlong normalPtr, jlong flagPtr) {
    uint32_t mask = 0;
    if (albedoPtr == 0) mask |= CHANNEL_ALBEDO;
    if (specularPtr == 0) mask |= CHANNEL_SPECULAR;
    if (normalPtr == 0) mask |= CHANNEL_NORMAL;
    if (flagPtr == 0) mask |= CHANNEL_FLAG;
    return mask;
}

uint32_t clampJintToU32(jint value) {
    return value < 0 ? 0u : static_cast<uint32_t>(value);
}

TextureLoaderV4::UploadRequest requestForRejection(
    jlong generation, jint namespaceId, jint tier, jint page,
    jint startLayer, jint layerCount, jint layerCapacity, jint width, jint height,
    jint vkFormat, jlong albedoPtr, jlong specularPtr, jlong normalPtr, jlong flagPtr,
    jlong bytesPerLayer, jint channelMask, jboolean visible) {
    TextureLoaderV4::UploadRequest req{};
    req.generation = generation <= 0 ? 0u : static_cast<uint64_t>(generation);
    req.namespaceId = clampJintToU32(namespaceId);
    req.tier = clampJintToU32(tier);
    req.page = clampJintToU32(page);
    req.startLayer = clampJintToU32(startLayer);
    req.layerCount = clampJintToU32(layerCount);
    req.layerCapacity = clampJintToU32(layerCapacity);
    req.width = clampJintToU32(width);
    req.height = clampJintToU32(height);
    req.format = static_cast<VkFormat>(vkFormat);
    req.albedoData = reinterpret_cast<const uint8_t*>(albedoPtr);
    req.specularData = reinterpret_cast<const uint8_t*>(specularPtr);
    req.normalData = reinterpret_cast<const uint8_t*>(normalPtr);
    req.flagData = reinterpret_cast<const uint8_t*>(flagPtr);
    req.bytesPerLayer = bytesPerLayer <= 0 ? 0u : static_cast<uint64_t>(bytesPerLayer);
    req.channelMask = clampJintToU32(channelMask);
    req.visible = visible == JNI_TRUE;
    return req;
}

/// Validate signed JNI inputs before any unsigned cast.
/// Accepts four-plane channel masks; requires a non-null pointer for every set bit.
/// page and startLayer must be non-negative (no -1 sentinel; Java provides explicit values).
bool validateLayerUploadSigned(jlong generation, jint namespaceId, jint tier,
    jint page, jint startLayer, jint layerCount, jint layerCapacity,
    jint width, jint height, jint vkFormat, jint channelMask,
    jlong albedoPtr, jlong specularPtr, jlong normalPtr, jlong flagPtr,
    jlong bytesPerLayer) {
    if (generation <= 0) return false;
    if (namespaceId < 0 || namespaceId > 3) return false;
    if (tier < 0 || tier >= 7) return false;
    if (page < 0) return false;
    if (startLayer < 0) return false;
    if (layerCount <= 0) return false;
    if (layerCapacity <= 0) return false;
    if (width <= 0 || height <= 0) return false;
    const uint32_t tierSize = tierSizePixels(static_cast<uint32_t>(tier));
    if (tierSize == 0) return false;
    if (width != static_cast<jint>(tierSize) || height != static_cast<jint>(tierSize)) return false;
    if (vkFormat != static_cast<jint>(VK_FORMAT_R8G8B8A8_UNORM)) return false;
    const uint64_t expectedBytes = uint64_t(static_cast<uint32_t>(width)) * uint64_t(static_cast<uint32_t>(height)) * 4ull;
    if (expectedBytes == 0 || expectedBytes > static_cast<uint64_t>(INT64_MAX)) return false;
    if (bytesPerLayer <= 0) return false;
    if (static_cast<uint64_t>(bytesPerLayer) != expectedBytes) return false;
    // Four-plane validation: require a non-null pointer for every set bit
    const uint32_t mask = static_cast<uint32_t>(channelMask);
    if ((mask & CHANNEL_ALBEDO) == 0) return false;
    if (albedoPtr == 0) return false;
    if ((mask & CHANNEL_SPECULAR) && specularPtr == 0) return false;
    if ((mask & CHANNEL_NORMAL) && normalPtr == 0) return false;
    if ((mask & CHANNEL_FLAG) && flagPtr == 0) return false;
    // Reject unknown bits
    static constexpr uint32_t ALL_CHANNELS = CHANNEL_ALBEDO | CHANNEL_SPECULAR | CHANNEL_NORMAL | CHANNEL_FLAG;
    if ((mask & ~ALL_CHANNELS) != 0) return false;
    return true;
}

} // anonymous namespace

// ---- V4 lifecycle ----

extern "C" JNIEXPORT jboolean JNICALL
Java_com_radiance_client_proxy_vulkan_TextureArrayBridgeV4_nativeBeginTextureLoaderV4(
    JNIEnv*, jclass, jlong generation, jlong manifestPtr, jint manifestBytes) {
    return guardedJniBool("nativeBeginTextureLoaderV4", generation, [&]() -> jboolean {
    if (generation <= 0) return JNI_FALSE;
    // manifestPtr/manifestBytes can be 0 for initial begin
    auto* renderer = Renderer::try_instance();
    if (!renderer || !renderer->framework()) return JNI_FALSE;
    auto& loader = renderer->textureLoaderV4();
    if (!loader.isInitialized()) {
        auto framework = renderer->framework();
        if (!loader.initialize(framework->device(), framework->vma())) {
            return JNI_FALSE;
        }
    }
    if (!loader.beginGeneration(static_cast<uint64_t>(generation))) {
        return JNI_FALSE;
    }
    const uint64_t nativeGeneration = static_cast<uint64_t>(generation);
    Renderer::textureSystem.beginV4MaterialPages(nativeGeneration);
    auto framework = renderer->framework();
    if (!Renderer::textureSystem.ensureV4ShaderFallbackResources(
            nativeGeneration, framework->vma(), framework->device())) {
        std::cerr << "[TextureLoaderV4JNI] nativeBeginTextureLoaderV4 rejected: V4 fallback resources unavailable"
                  << " generation=" << generation << std::endl;
        loader.cancelGeneration(nativeGeneration, -2);
        return JNI_FALSE;
    }
    return JNI_TRUE;
    });
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_radiance_client_proxy_vulkan_TextureArrayBridgeV4_nativeEnsureV4ShaderFallbackResources(
    JNIEnv*, jclass, jlong generation) {
    return guardedJniBool("nativeEnsureV4ShaderFallbackResources", generation, [&]() -> jboolean {
    if (generation <= 0) return JNI_FALSE;
    auto* renderer = Renderer::try_instance();
    if (!renderer || !renderer->framework()) return JNI_FALSE;
    auto framework = renderer->framework();
    return Renderer::textureSystem.ensureV4ShaderFallbackResources(
        static_cast<uint64_t>(generation), framework->vma(), framework->device())
        ? JNI_TRUE : JNI_FALSE;
    });
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_radiance_client_proxy_vulkan_TextureArrayBridgeV4_nativeUploadTexturePageV4(
    JNIEnv*, jclass, jlong generation, jint namespaceId, jint tier, jint page,
    jint startLayer, jint layerCount, jint layerCapacity, jint width, jint height, jint vkFormat,
    jlong albedoPtr, jlong specularPtr, jlong normalPtr, jlong flagPtr,
    jlong bytesPerLayer, jint channelMask, jboolean visible) {
    return guardedJniBool("nativeUploadTexturePageV4", generation, [&]() -> jboolean {
    // Validate all signed JNI inputs before any unsigned cast.
    // Negative values would become huge native values if cast without validation.
    if (!validateLayerUploadSigned(generation, namespaceId, tier, page, startLayer,
        layerCount, layerCapacity, width, height, vkFormat, channelMask,
        albedoPtr, specularPtr, normalPtr, flagPtr, bytesPerLayer)) {
        if (auto* renderer = Renderer::try_instance()) {
            auto rejected = requestForRejection(generation, namespaceId, tier, page, startLayer,
                layerCount, layerCapacity, width, height, vkFormat,
                albedoPtr, specularPtr, normalPtr, flagPtr, bytesPerLayer, channelMask, visible);
            const uint32_t planeMask = nullPlaneMask(albedoPtr, specularPtr, normalPtr, flagPtr);
            renderer->textureLoaderV4().recordUploadRejection(rejected,
                planeMask != 0 ? "invalid_plane_pointer" : "invalid_arguments",
                UINT32_MAX, UINT32_MAX, UINT32_MAX, 0, UINT32_MAX, planeMask, "validation");
        }
        std::cerr << "[TextureLoaderV4JNI] nativeUploadTexturePageV4 rejected invalid arguments"
                  << " generation=" << generation
                  << " namespace=" << namespaceId
                  << " tier=" << tier
                  << " page=" << page
                  << " startLayer=" << startLayer
                  << " layers=" << layerCount
                  << " capacity=" << layerCapacity
                  << " width=" << width
                  << " height=" << height
                  << " bytesPerLayer=" << bytesPerLayer
                  << " channelMask=" << channelMask << std::endl;
        return JNI_FALSE;
    }

    auto* renderer = Renderer::try_instance();
    if (!renderer || !renderer->framework()) {
        std::cerr << "[TextureLoaderV4JNI] nativeUploadTexturePageV4 rejected: renderer unavailable"
                  << " generation=" << generation
                  << " tier=" << tier
                  << " page=" << page
                  << " startLayer=" << startLayer
                  << " layers=" << layerCount
                  << " capacity=" << layerCapacity << std::endl;
        return JNI_FALSE;
    }

    // Build request only after validation — safe to cast now
    TextureLoaderV4::UploadRequest req{};
    req.generation = static_cast<uint64_t>(generation);
    req.namespaceId = static_cast<uint32_t>(namespaceId);
    req.tier = static_cast<uint32_t>(tier);
    req.page = static_cast<uint32_t>(page);
    req.startLayer = static_cast<uint32_t>(startLayer);
    req.layerCount = static_cast<uint32_t>(layerCount);
    req.layerCapacity = static_cast<uint32_t>(layerCapacity);
    req.width = static_cast<uint32_t>(width);
    req.height = static_cast<uint32_t>(height);
    req.format = static_cast<VkFormat>(vkFormat);
    req.albedoData = reinterpret_cast<const uint8_t*>(albedoPtr);
    req.specularData = (channelMask & CHANNEL_SPECULAR) ? reinterpret_cast<const uint8_t*>(specularPtr) : nullptr;
    req.normalData   = (channelMask & CHANNEL_NORMAL)   ? reinterpret_cast<const uint8_t*>(normalPtr)   : nullptr;
    req.flagData     = (channelMask & CHANNEL_FLAG)     ? reinterpret_cast<const uint8_t*>(flagPtr)     : nullptr;
    req.bytesPerLayer = static_cast<uint64_t>(bytesPerLayer);
    req.channelMask = static_cast<uint32_t>(channelMask);
    req.visible = visible == JNI_TRUE;

    bool queued = renderer->textureLoaderV4().enqueueUpload(req);
    if (!queued) {
        std::cerr << "[TextureLoaderV4JNI] nativeUploadTexturePageV4 rejected: queue failed"
                  << " generation=" << generation
                  << " tier=" << tier
                  << " page=" << page
                  << " startLayer=" << startLayer
                  << " layers=" << layerCount
                  << " capacity=" << layerCapacity << std::endl;
        return JNI_FALSE;
    }

    uint32_t nativeStartLayer = static_cast<uint32_t>(startLayer);
    uint32_t nativeCapacity = static_cast<uint32_t>(layerCapacity);
    uint32_t descriptorPage = descriptorPageForV4(
        static_cast<uint32_t>(namespaceId),
        static_cast<uint32_t>(tier),
        static_cast<uint32_t>(page),
        static_cast<uint32_t>(startLayer),
        &nativeStartLayer,
        &nativeCapacity);
    if (descriptorPage >= vk::Data::MATERIAL_TEXTURE_PAGE_MAX || nativeCapacity == 0) {
        renderer->textureLoaderV4().recordUploadRejection(req,
            "invalid_descriptor_page", UINT32_MAX,
            UINT32_MAX, nativeStartLayer, nativeCapacity,
            descriptorPage, nullPlaneMask(albedoPtr, specularPtr, normalPtr, flagPtr),
            "descriptor_map");
        std::cerr << "[TextureLoaderV4JNI] nativeUploadTexturePageV4 rejected invalid descriptor page"
                  << " generation=" << generation
                  << " namespace=" << namespaceId
                  << " tier=" << tier
                  << " page=" << page
                  << " descriptorPage=" << descriptorPage
                  << " startLayer=" << startLayer
                  << " nativeStartLayer=" << nativeStartLayer
                  << " layers=" << layerCount
                  << " capacity=" << layerCapacity
                  << " nativeCapacity=" << nativeCapacity << std::endl;
        return JNI_FALSE;
    }

    auto framework = renderer->framework();
    bool published = Renderer::textureSystem.uploadMaterialTextureLayersV4(
        descriptorPage,
        static_cast<uint32_t>(width),
        nativeStartLayer,
        static_cast<uint32_t>(layerCount),
        nativeCapacity,
        reinterpret_cast<const uint8_t*>(albedoPtr),
        reinterpret_cast<const uint8_t*>(specularPtr),
        reinterpret_cast<const uint8_t*>(normalPtr),
        reinterpret_cast<const uint8_t*>(flagPtr),
        static_cast<uint32_t>(channelMask),
        static_cast<uint64_t>(generation),
        framework->vma(),
        framework->device());
    if (!published) {
        renderer->textureLoaderV4().recordUploadRejection(req,
            "descriptor_publish_failed", UINT32_MAX,
            descriptorPage, nativeStartLayer, nativeCapacity,
            descriptorPage, nullPlaneMask(albedoPtr, specularPtr, normalPtr, flagPtr),
            "descriptor_publish");
        std::cerr << "[TextureLoaderV4JNI] nativeUploadTexturePageV4 rejected: descriptor/material page publication failed"
                  << " generation=" << generation
                  << " namespace=" << namespaceId
                  << " tier=" << tier
                  << " page=" << page
                  << " descriptorPage=" << descriptorPage
                  << " startLayer=" << startLayer
                  << " nativeStartLayer=" << nativeStartLayer
                  << " layers=" << layerCount
                  << " capacity=" << layerCapacity
                  << " nativeCapacity=" << nativeCapacity << std::endl;
        return JNI_FALSE;
    }

    return JNI_TRUE;
    });
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_radiance_client_proxy_vulkan_TextureArrayBridgeV4_nativeCommitTextureLoaderV4(
    JNIEnv*, jclass, jlong generation) {
    return guardedJniBool("nativeCommitTextureLoaderV4", generation, [&]() -> jboolean {
    if (generation <= 0) return JNI_FALSE;
    auto* renderer = Renderer::try_instance();
    if (!renderer || !renderer->framework()) return JNI_FALSE;
    auto framework = renderer->framework();
    if (!Renderer::textureSystem.ensureV4ShaderFallbackResources(
            static_cast<uint64_t>(generation), framework->vma(), framework->device())) {
        std::cerr << "[TextureLoaderV4JNI] nativeCommitTextureLoaderV4 rejected: V4 fallback contract unavailable"
                  << " generation=" << generation << std::endl;
        return JNI_FALSE;
    }
    return renderer->textureLoaderV4().commitGeneration(static_cast<uint64_t>(generation))
        ? JNI_TRUE : JNI_FALSE;
    });
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_radiance_client_proxy_vulkan_TextureArrayBridgeV4_nativeCancelTextureLoaderV4(
    JNIEnv*, jclass, jlong generation, jint reasonCode) {
    return guardedJniBool("nativeCancelTextureLoaderV4", generation, [&]() -> jboolean {
    if (generation <= 0) return JNI_FALSE;
    auto* renderer = Renderer::try_instance();
    if (!renderer || !renderer->framework()) return JNI_FALSE;
    return renderer->textureLoaderV4().cancelGeneration(
        static_cast<uint64_t>(generation), static_cast<int>(reasonCode))
        ? JNI_TRUE : JNI_FALSE;
    });
}

// ---- Sparse registry updates ----

extern "C" JNIEXPORT jboolean JNICALL
Java_com_radiance_client_proxy_vulkan_TextureArrayBridgeV4_nativeUpdateMaterialTableSparseV4(
    JNIEnv*, jclass, jlong generation, jlong entriesPtr, jint entryCount) {
    return guardedJniBool("nativeUpdateMaterialTableSparseV4", generation, [&]() -> jboolean {
    if (generation <= 0 || entriesPtr == 0 || entryCount <= 0) return JNI_FALSE;
    auto* renderer = Renderer::try_instance();
    if (!renderer || !renderer->framework()) return JNI_FALSE;
    auto framework = renderer->framework();
    return Renderer::textureSystem.updateMaterialTableSparse(
        reinterpret_cast<const vk::Data::MaterialEntry*>(entriesPtr),
        static_cast<uint32_t>(entryCount),
        static_cast<uint64_t>(generation),
        framework->vma(), framework->device()) ? JNI_TRUE : JNI_FALSE;
    });
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_radiance_client_proxy_vulkan_TextureArrayBridgeV4_nativeUpdateSpriteRegistrySparseV4(
    JNIEnv*, jclass, jlong generation, jlong entriesPtr, jint entryCount) {
    return guardedJniBool("nativeUpdateSpriteRegistrySparseV4", generation, [&]() -> jboolean {
    if (generation <= 0 || entriesPtr == 0 || entryCount <= 0) return JNI_FALSE;
    auto* renderer = Renderer::try_instance();
    if (!renderer || !renderer->framework()) return JNI_FALSE;
    auto framework = renderer->framework();
    return Renderer::textureSystem.updateSpriteRegistrySparse(
        reinterpret_cast<const vk::Data::SpriteEntry*>(entriesPtr),
        static_cast<uint32_t>(entryCount),
        static_cast<uint64_t>(generation),
        framework->vma(), framework->device()) ? JNI_TRUE : JNI_FALSE;
    });
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_radiance_client_proxy_vulkan_TextureArrayBridgeV4_nativeUpdateTextureRulesV4(
    JNIEnv*, jclass, jlong generation, jlong entriesPtr, jint entryCount) {
    return guardedJniBool("nativeUpdateTextureRulesV4", generation, [&]() -> jboolean {
    if (generation <= 0 || entryCount < 0) return JNI_FALSE;
    auto* renderer = Renderer::try_instance();
    if (!renderer || !renderer->framework()) return JNI_FALSE;
    auto framework = renderer->framework();
    const auto* entries = entriesPtr == 0
        ? nullptr
        : reinterpret_cast<const vk::Data::TextureRuleEntry*>(entriesPtr);
    const uint32_t count = entryCount <= 0 ? 0u : static_cast<uint32_t>(entryCount);
    return Renderer::textureSystem.uploadTextureRules(
        entries,
        count,
        static_cast<uint64_t>(generation),
        framework->vma(), framework->device()) ? JNI_TRUE : JNI_FALSE;
    });
}

// ---- Status JSON ----

extern "C" JNIEXPORT jboolean JNICALL
Java_com_radiance_client_proxy_vulkan_TextureArrayBridgeV4_nativeSubmitPackStackSnapshotV1(
    JNIEnv* env, jclass, jlong generation, jstring snapshotJson,
    jint packCount, jint resourceCount, jint ruleFileCount, jint sidecarCount,
    jstring packStackHash, jlong javaCaptureMillis) {
    return guardedJniBool("nativeSubmitPackStackSnapshotV1", generation, [&]() -> jboolean {
    if (generation <= 0 || !snapshotJson || packCount < 0 || resourceCount < 0
        || ruleFileCount < 0 || sidecarCount < 0 || javaCaptureMillis < 0) {
        return JNI_FALSE;
    }
    std::string snapshot = readJString(env, snapshotJson);
    if (snapshot.empty()) {
        return JNI_FALSE;
    }
    std::string hash = readJString(env, packStackHash);
    PackIndexBuildResult index = buildPackIndexFromSnapshot(snapshot);
    {
        std::lock_guard<std::mutex> lock(g_packIndexMutex);
        g_packIndexState.hasSnapshot = true;
        g_packIndexState.diffReady = index.diffReady;
        g_packIndexState.generation = static_cast<uint64_t>(generation);
        g_packIndexState.submitCount++;
        g_packIndexState.snapshotBytes = snapshot.size();
        g_packIndexState.packCount = static_cast<uint32_t>(packCount);
        g_packIndexState.resourceCount = static_cast<uint32_t>(resourceCount);
        g_packIndexState.ruleFileCount = static_cast<uint32_t>(ruleFileCount);
        g_packIndexState.sidecarCount = static_cast<uint32_t>(sidecarCount);
        g_packIndexState.javaCaptureMillis = static_cast<uint64_t>(javaCaptureMillis);
        g_packIndexState.nativeIndexMillis = index.nativeIndexMillis;
        g_packIndexState.activeDiskPackCount = index.activeDiskPackCount;
        g_packIndexState.nativeResourceCount = index.nativeResourceCount;
        g_packIndexState.nativeSidecarCount = index.nativeSidecarCount;
        g_packIndexState.nativeRuleFileCount = index.nativeRuleFileCount;
        g_packIndexState.virtualJavaOnlyResourceCount = index.virtualJavaOnlyResourceCount;
        g_packIndexState.missingCount = index.missingCount;
        g_packIndexState.winnerMismatchCount = index.winnerMismatchCount;
        g_packIndexState.flagMismatchCount = index.flagMismatchCount;
        g_packIndexState.extraNativeCount = index.extraNativeCount;
        g_packIndexState.nativeContentHashReadyCount = index.nativeContentHashReadyCount;
        g_packIndexState.nativeContentHashMissingCount = index.nativeContentHashMissingCount;
        g_packIndexState.diskBackedContentHashReadyCount = index.diskBackedContentHashReadyCount;
        g_packIndexState.diskBackedContentHashMissingCount = index.diskBackedContentHashMissingCount;
        g_packIndexState.virtualJavaOnlyContentHashMissingCount = index.virtualJavaOnlyContentHashMissingCount;
        g_packIndexState.packStackHash = std::move(hash);
        g_packIndexState.indexError = index.error;
        g_packIndexState.diffJson = std::move(index.diffJson);
    }
    return index.ok ? JNI_TRUE : JNI_FALSE;
    });
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_radiance_client_proxy_vulkan_TextureArrayBridgeV4_nativePackIndexStatusJson(
    JNIEnv* env, jclass) {
    return guardedJniString(env, "nativePackIndexStatusJson", [&]() -> jstring {
    return makeString(env, packIndexStatusJson());
    });
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_radiance_client_proxy_vulkan_TextureArrayBridgeV4_nativePackIndexDiffAgainstVanillaJson(
    JNIEnv* env, jclass) {
    return guardedJniString(env, "nativePackIndexDiffAgainstVanillaJson", [&]() -> jstring {
    return makeString(env, packIndexDiffJson());
    });
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_radiance_client_proxy_vulkan_TextureArrayBridgeV4_nativeTextureLoaderV4StatusJson(
    JNIEnv* env, jclass) {
    return guardedJniString(env, "nativeTextureLoaderV4StatusJson", [&]() -> jstring {
    auto* renderer = Renderer::try_instance();
    if (!renderer) return makeString(env, "{\"error\":\"no_renderer\"}");
    return makeString(env, renderer->textureLoaderV4().statusJson());
    });
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_radiance_client_proxy_vulkan_TextureArrayBridgeV4_nativeTextureTierStatusJsonV4(
    JNIEnv* env, jclass) {
    return guardedJniString(env, "nativeTextureTierStatusJsonV4", [&]() -> jstring {
    auto* renderer = Renderer::try_instance();
    if (!renderer) return makeString(env, "{\"error\":\"no_renderer\"}");
    return makeString(env, renderer->textureLoaderV4().tierStatusJson());
    });
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_radiance_client_proxy_vulkan_TextureArrayBridgeV4_nativeGpuUploadQueueStatusJsonV4(
    JNIEnv* env, jclass) {
    return guardedJniString(env, "nativeGpuUploadQueueStatusJsonV4", [&]() -> jstring {
    auto* renderer = Renderer::try_instance();
    if (!renderer) return makeString(env, "{\"error\":\"no_renderer\"}");
    return makeString(env, renderer->textureLoaderV4().uploadQueueStatusJson());
    });
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_radiance_client_proxy_vulkan_TextureArrayBridgeV4_nativeMaterialPagePoolStatusJsonV4(
    JNIEnv* env, jclass) {
    return guardedJniString(env, "nativeMaterialPagePoolStatusJsonV4", [&]() -> jstring {
    auto* renderer = Renderer::try_instance();
    if (!renderer) return makeString(env, "{\"error\":\"no_renderer\"}");
    return makeString(env, renderer->textureLoaderV4().pagePoolStatusJson());
    });
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_radiance_client_proxy_vulkan_TextureArrayBridgeV4_nativeV4FrameResourceStatusJson(
    JNIEnv* env, jclass) {
    return guardedJniString(env, "nativeV4FrameResourceStatusJson", [&]() -> jstring {
    auto* renderer = Renderer::try_instance();
    if (!renderer) return makeString(env, "{\"error\":\"no_renderer\"}");
    return makeString(env, Renderer::textureSystem.v4FrameResourceStatusJson());
    });
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_radiance_client_proxy_vulkan_TextureArrayBridgeV4_nativeFirstFrameNativeReadinessJsonV4(
    JNIEnv* env, jclass, jlong generation) {
    return guardedJniString(env, "nativeFirstFrameNativeReadinessJsonV4", [&]() -> jstring {
    auto* renderer = Renderer::try_instance();
    if (!renderer) return makeString(env, "{\"error\":\"no_renderer\"}");
    return makeString(env, renderer->textureLoaderV4().firstFrameReadinessJson(
        static_cast<uint64_t>(generation)));
    });
}

extern "C" JNIEXPORT jint JNICALL
Java_com_radiance_client_proxy_vulkan_TextureArrayBridgeV4_nativePageLayerCapacityForTier(
    JNIEnv*, jclass, jint tier) {
    if (tier < 0 || tier >= static_cast<jint>(TexturePagePool::kMaxTiers)) return 0;
    return static_cast<jint>(TexturePagePool::pageLayerCapacityStatic(static_cast<uint32_t>(tier)));
}
