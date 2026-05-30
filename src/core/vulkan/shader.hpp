#pragma once

#include "core/all_extern.hpp"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <shaderc/shaderc.hpp>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace vk {
class Device;
shaderc_shader_kind shaderKindFromStage(VkShaderStageFlagBits stage);

class ShaderSpirvCache {
  public:
    static std::optional<uint64_t> computeDependencyHash(const std::string &sourcePath,
                                                         const std::vector<std::string> &includeDirectories);

    static std::string computeCompiledHash(const std::string &sourcePath,
                                           VkShaderStageFlagBits stage,
                                           const std::unordered_map<std::string, std::string> &definitions,
                                           const std::vector<std::string> &includeDirectories,
                                           const std::string &injectedSource,
                                           uint64_t dependencyHash);

    static std::vector<uint32_t> readCachedSpirv(const std::filesystem::path &path);

    static void writeCachedSpirv(const std::filesystem::path &path, const std::vector<uint32_t> &spirv);

  private:
    static constexpr uint64_t hashSeed_ = 14695981039346656037ULL;

    static void updateHashBytes(uint64_t &hash, const void *source, size_t byteCount);
    static void updateHashString(uint64_t &hash, std::string_view value);

    template <typename T>
    static void updateHashValue(uint64_t &hash, const T &value) {
        updateHashBytes(hash, &value, sizeof(T));
    }

    static std::filesystem::path makeCanonical(const std::filesystem::path &path);

    static std::optional<uint64_t> computeFileDependencyHash(
        const std::filesystem::path &shaderPath,
        const std::vector<std::string> &includeDirectories,
        std::unordered_map<std::string, std::optional<uint64_t>> &hashCache,
        std::unordered_set<std::string> &visitingPaths);

    static void trimCacheDirectory(const std::filesystem::path &cacheDir, size_t maxFiles = 500);
};

class ShaderIncluder : public shaderc::CompileOptions::IncluderInterface {
  public:
    explicit ShaderIncluder(std::vector<std::filesystem::path> includeDirectories);

    shaderc_include_result *GetInclude(const char *requestedSource,
                                       shaderc_include_type type,
                                       const char *requestingSource,
                                       size_t includeDepth) override;
    void ReleaseInclude(shaderc_include_result *data) override;

  private:
    struct IncludeData;

    std::vector<std::filesystem::path> includeDirectories_;
};

class Shader : public SharedObject<Shader> {
  public:
    struct CompileResult {
        std::string sourcePath;
        VkShaderStageFlagBits stage;
        std::vector<uint32_t> spirv;
#ifdef DEBUG
        bool cacheHit = false;
        bool cacheReadFailed = false;
        std::string cacheFilePath;

        CompileResult clone() const { return {sourcePath, stage, spirv, cacheHit, cacheReadFailed, cacheFilePath}; }
#else
        CompileResult clone() const { return {sourcePath, stage, spirv}; }
#endif
    };

    Shader(std::shared_ptr<Device> device, std::string path);
    Shader(std::shared_ptr<Device> device,
           std::string sourcePath,
           VkShaderStageFlagBits stage,
           std::unordered_map<std::string, std::string> definitions = {},
           std::vector<std::string> includeDirectories = {},
           std::string injectedSource = "");
    Shader(std::shared_ptr<Device> device, CompileResult compileResult);
    ~Shader();

    static CompileResult compileGlslToSpv(std::string sourcePath,
                                          VkShaderStageFlagBits stage,
                                          std::unordered_map<std::string, std::string> definitions = {},
                                          std::vector<std::string> includeDirectories = {},
                                          std::string injectedSource = "",
                                          const std::filesystem::path &cacheDir = {});

    VkShaderModule &vkShaderModule();

  private:
    void createModule(const std::vector<uint32_t> &spirv, const std::string &sourcePath);

    std::shared_ptr<Device> device_;

    std::string path_;
    VkShaderModule module_;
};
}; // namespace vk
