#include "core/vulkan/shader.hpp"

#include "core/vulkan/debug_utils.hpp"
#include "core/vulkan/device.hpp"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

std::ostream &shaderCout() {
    return std::cout << "[Shader] ";
}

std::ostream &shaderCerr() {
    return std::cerr << "[Shader] ";
}

std::string injectSourceAfterVersion(std::string sourceText, const std::string &injectedSource) {
    if (injectedSource.empty()) { return sourceText; }

    size_t insertPos = 0;
    if (sourceText.rfind("#version", 0) == 0) {
        size_t lineEnd = sourceText.find('\n');
        insertPos = lineEnd == std::string::npos ? sourceText.size() : lineEnd + 1;
        while (insertPos < sourceText.size()) {
            size_t nextLineEnd = sourceText.find('\n', insertPos);
            size_t lineLength =
                nextLineEnd == std::string::npos ? sourceText.size() - insertPos : nextLineEnd - insertPos;
            std::string_view line(sourceText.data() + insertPos, lineLength);
            size_t contentOffset = line.find_first_not_of(" \t\r");
            if (contentOffset == std::string::npos) {
                insertPos = nextLineEnd == std::string::npos ? sourceText.size() : nextLineEnd + 1;
                continue;
            }
            if (line.compare(contentOffset, sizeof("#extension") - 1, "#extension") != 0) { break; }
            insertPos = nextLineEnd == std::string::npos ? sourceText.size() : nextLineEnd + 1;
        }
    }

    sourceText.insert(insertPos, injectedSource + "\n");
    return sourceText;
}

shaderc_shader_kind vk::shaderKindFromStage(VkShaderStageFlagBits stage) {
    switch (stage) {
        case VK_SHADER_STAGE_VERTEX_BIT: return shaderc_glsl_vertex_shader;
        case VK_SHADER_STAGE_FRAGMENT_BIT: return shaderc_glsl_fragment_shader;
        case VK_SHADER_STAGE_COMPUTE_BIT: return shaderc_glsl_compute_shader;
        case VK_SHADER_STAGE_RAYGEN_BIT_KHR: return shaderc_glsl_raygen_shader;
        case VK_SHADER_STAGE_MISS_BIT_KHR: return shaderc_glsl_miss_shader;
        case VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR: return shaderc_glsl_closesthit_shader;
        case VK_SHADER_STAGE_ANY_HIT_BIT_KHR: return shaderc_glsl_anyhit_shader;
        case VK_SHADER_STAGE_INTERSECTION_BIT_KHR: return shaderc_glsl_intersection_shader;
        case VK_SHADER_STAGE_CALLABLE_BIT_KHR: return shaderc_glsl_callable_shader;
        default: shaderCerr() << "unsupported shader stage " << stage << std::endl; exit(EXIT_FAILURE);
    }
}

struct vk::ShaderIncluder::IncludeData {
    shaderc_include_result result{};
    std::string sourceName;
    std::string content;
};

vk::ShaderIncluder::ShaderIncluder(std::vector<std::filesystem::path> includeDirectories)
    : includeDirectories_(std::move(includeDirectories)) {}

shaderc_include_result *vk::ShaderIncluder::GetInclude(const char *requestedSource,
                                                       shaderc_include_type type,
                                                       const char *requestingSource,
                                                       size_t includeDepth) {
    (void)includeDepth;
    std::filesystem::path requested(requestedSource);
    std::filesystem::path resolved;

    if (type == shaderc_include_type_relative && requestingSource != nullptr && requestingSource[0] != '\0') {
        std::filesystem::path requesting(requestingSource);
        std::filesystem::path candidate = requesting.parent_path() / requested;
        if (std::filesystem::exists(candidate) && std::filesystem::is_regular_file(candidate)) {
            resolved = std::filesystem::weakly_canonical(candidate);
        }
    }

    if (resolved.empty()) {
        for (const std::filesystem::path &dir : includeDirectories_) {
            std::filesystem::path candidate = dir / requested;
            if (std::filesystem::exists(candidate) && std::filesystem::is_regular_file(candidate)) {
                resolved = std::filesystem::weakly_canonical(candidate);
                break;
            }
        }
    }

    IncludeData *include = new IncludeData{};
    if (resolved.empty()) {
        include->content = "Failed to resolve include: " + std::string(requestedSource);
        include->sourceName = requestedSource;
        include->result.source_name = include->sourceName.c_str();
        include->result.source_name_length = include->sourceName.size();
        include->result.content = include->content.c_str();
        include->result.content_length = include->content.size();
        include->result.user_data = include;
        return &include->result;
    }

    std::ifstream file(resolved, std::ios::binary);
    if (!file.is_open()) {
        include->content = "Failed to open include: " + resolved.string();
        include->sourceName = resolved.string();
        include->result.source_name = include->sourceName.c_str();
        include->result.source_name_length = include->sourceName.size();
        include->result.content = include->content.c_str();
        include->result.content_length = include->content.size();
        include->result.user_data = include;
        return &include->result;
    }

    include->sourceName = resolved.string();
    include->content.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
    include->result.source_name = include->sourceName.c_str();
    include->result.source_name_length = include->sourceName.size();
    include->result.content = include->content.c_str();
    include->result.content_length = include->content.size();
    include->result.user_data = include;
    return &include->result;
}

void vk::ShaderIncluder::ReleaseInclude(shaderc_include_result *data) {
    if (data == nullptr || data->user_data == nullptr) { return; }
    delete static_cast<IncludeData *>(data->user_data);
}

vk::Shader::Shader(std::shared_ptr<Device> device, std::string path) : device_(device), path_(path) {
    std::ifstream file(path, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        shaderCerr() << "Cannot open file: " << path << std::endl;
        exit(EXIT_FAILURE);
    }
    std::vector<char> fileBytes(file.tellg());
    file.seekg(0, std::ios::beg);
    file.read(fileBytes.data(), fileBytes.size());
    file.close();

    VkShaderModuleCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = fileBytes.size();
    createInfo.pCode = (uint32_t *)fileBytes.data();

    if (vkCreateShaderModule(device_->vkDevice(), &createInfo, nullptr, &module_) != VK_SUCCESS) {
        shaderCerr() << "failed to create shader module for " << path << std::endl;
        exit(EXIT_FAILURE);
    }
    auto shaderName = std::string("Shader ") + std::filesystem::path(path_).filename().string();
    vk::DebugUtils::setObjectName(device_->vkDevice(), VK_OBJECT_TYPE_SHADER_MODULE, module_, shaderName);

#ifdef DEBUG
    shaderCout() << "created shader module for " << path << std::endl;
#endif
}

vk::Shader::Shader(std::shared_ptr<Device> device,
                   std::string sourcePath,
                   VkShaderStageFlagBits stage,
                   std::unordered_map<std::string, std::string> definitions,
                   std::vector<std::string> includeDirectories,
                   std::string injectedSource)
    : device_(device), path_(sourcePath) {
    auto compileResult =
        compileGlslToSpv(std::move(sourcePath), stage, std::move(definitions), std::move(includeDirectories),
                         std::move(injectedSource));
    createModule(compileResult.spirv, compileResult.sourcePath);
}

vk::Shader::Shader(std::shared_ptr<Device> device, vk::Shader::CompileResult compileResult)
    : device_(device), path_(compileResult.sourcePath) {
    createModule(compileResult.spirv, compileResult.sourcePath);
}

void vk::ShaderSpirvCache::updateHashBytes(uint64_t &hash, const void *source, size_t byteCount) {
    const auto *bytes = static_cast<const unsigned char *>(source);
    for (size_t i = 0; i < byteCount; i++) { hash = (hash ^ bytes[i]) * 1099511628211ULL; }
}

void vk::ShaderSpirvCache::updateHashString(uint64_t &hash, std::string_view value) {
    updateHashBytes(hash, value.data(), value.size());
    const char nullTerminator = '\0';
    updateHashBytes(hash, &nullTerminator, sizeof(nullTerminator));
}

std::filesystem::path vk::ShaderSpirvCache::makeCanonical(const std::filesystem::path &path) {
    std::error_code errorCode;
    std::filesystem::path result = std::filesystem::weakly_canonical(path, errorCode);
    return errorCode ? path.lexically_normal() : result;
}

std::optional<uint64_t> vk::ShaderSpirvCache::computeDependencyHash(const std::string &sourcePath,
                                                                    const std::vector<std::string> &includeDirectories) {
    std::unordered_map<std::string, std::optional<uint64_t>> hashCache;
    std::unordered_set<std::string> visitingPaths;
    return computeFileDependencyHash(sourcePath, includeDirectories, hashCache, visitingPaths);
}

std::string vk::ShaderSpirvCache::computeCompiledHash(const std::string &sourcePath,
                                                      VkShaderStageFlagBits stage,
                                                      const std::unordered_map<std::string, std::string> &definitions,
                                                      const std::vector<std::string> &includeDirectories,
                                                      const std::string &injectedSource,
                                                      uint64_t dependencyHash) {
    uint64_t hash = hashSeed_;
    updateHashString(hash, makeCanonical(sourcePath).string());
    updateHashValue(hash, stage);
    updateHashValue(hash, dependencyHash);
    std::vector<std::pair<std::string, std::string>> sortedDefinitions(definitions.begin(), definitions.end());
    std::sort(sortedDefinitions.begin(), sortedDefinitions.end());
    for (const auto &[name, value] : sortedDefinitions) {
        updateHashString(hash, name);
        updateHashString(hash, value);
    }
    for (const auto &includeDirectory : includeDirectories) { updateHashString(hash, includeDirectory); }
    updateHashString(hash, injectedSource);
    std::ostringstream stream;
    stream << std::hex << hash;
    return stream.str();
}

std::vector<uint32_t> vk::ShaderSpirvCache::readCachedSpirv(const std::filesystem::path &path) {
    std::ifstream file(path, std::ios::ate | std::ios::binary);
    if (!file.is_open()) return {};
    size_t fileSize = file.tellg();
    if (fileSize % sizeof(uint32_t) != 0) return {};
    file.seekg(0, std::ios::beg);
    std::vector<uint32_t> spirv(fileSize / sizeof(uint32_t));
    file.read(reinterpret_cast<char *>(spirv.data()), static_cast<std::streamsize>(fileSize));
    return spirv;
}

void vk::ShaderSpirvCache::writeCachedSpirv(const std::filesystem::path &path, const std::vector<uint32_t> &spirv) {
    std::error_code errorCode;
    std::filesystem::create_directories(path.parent_path(), errorCode);
    if (errorCode) return;
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) return;
    file.write(reinterpret_cast<const char *>(spirv.data()),
               static_cast<std::streamsize>(spirv.size() * sizeof(uint32_t)));
    trimCacheDirectory(path.parent_path());
}

std::optional<uint64_t>
vk::ShaderSpirvCache::computeFileDependencyHash(const std::filesystem::path &shaderPath,
                                                const std::vector<std::string> &includeDirectories,
                                                std::unordered_map<std::string, std::optional<uint64_t>> &hashCache,
                                                std::unordered_set<std::string> &visitingPaths) {
    std::filesystem::path canonicalPath = makeCanonical(shaderPath);
    const std::string cacheKey = canonicalPath.string();

    if (auto iter = hashCache.find(cacheKey); iter != hashCache.end()) { return iter->second; }
    if (!visitingPaths.insert(cacheKey).second) { return 0ull; }

    std::ifstream file(canonicalPath, std::ios::binary);
    if (!file.is_open()) {
        visitingPaths.erase(cacheKey);
        hashCache[cacheKey] = std::nullopt;
        return std::nullopt;
    }

    std::string content{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
    uint64_t hash = hashSeed_;
    updateHashString(hash, cacheKey);
    updateHashString(hash, content);

    auto stripComments = [](std::string_view line, bool &isInsideBlockComment) -> std::string {
        std::string result;
        result.reserve(line.size());
        for (size_t i = 0; i < line.size();) {
            if (isInsideBlockComment) {
                if (i + 1 < line.size() && line[i] == '*' && line[i + 1] == '/') {
                    isInsideBlockComment = false;
                    i += 2;
                } else {
                    i++;
                }
                continue;
            }
            if (i + 1 < line.size() && line[i] == '/' && line[i + 1] == '*') {
                isInsideBlockComment = true;
                i += 2;
                continue;
            }
            if (i + 1 < line.size() && line[i] == '/' && line[i + 1] == '/') { break; }
            result.push_back(line[i]);
            i++;
        }
        return result;
    };

    auto isIdentifierCharacter = [](char ch) -> bool {
        unsigned char uch = static_cast<unsigned char>(ch);
        return std::isalnum(uch) != 0 || ch == '_';
    };

    auto parseIncludeDirective = [&](std::string_view line) -> std::optional<std::string> {
        size_t i = 0;
        while (i < line.size() && std::isspace(static_cast<unsigned char>(line[i])) != 0) { i++; }
        if (i >= line.size() || line[i] != '#') { return std::nullopt; }
        i++;
        while (i < line.size() && std::isspace(static_cast<unsigned char>(line[i])) != 0) { i++; }
        size_t directiveStart = i;
        while (i < line.size() && isIdentifierCharacter(line[i])) { i++; }
        if (line.substr(directiveStart, i - directiveStart) != "include") { return std::nullopt; }
        while (i < line.size() && std::isspace(static_cast<unsigned char>(line[i])) != 0) { i++; }
        if (i >= line.size() || line[i] != '"') { return std::nullopt; }
        i++;
        size_t pathStart = i;
        while (i < line.size() && line[i] != '"') { i++; }
        if (i >= line.size()) { return std::nullopt; }
        return std::string(line.substr(pathStart, i - pathStart));
    };

    auto resolveIncludePath =
        [&](const std::filesystem::path &requestingPath,
            std::string_view includePath) -> std::optional<std::filesystem::path> {
        std::error_code errorCode;
        std::filesystem::path relativeCandidate = requestingPath.parent_path() / std::filesystem::path(includePath);
        if (std::filesystem::exists(relativeCandidate, errorCode) &&
            std::filesystem::is_regular_file(relativeCandidate, errorCode)) {
            return makeCanonical(relativeCandidate);
        }
        for (const auto &includeDirectory : includeDirectories) {
            std::filesystem::path candidate =
                std::filesystem::path(includeDirectory) / std::filesystem::path(includePath);
            if (std::filesystem::exists(candidate, errorCode) &&
                std::filesystem::is_regular_file(candidate, errorCode)) {
                return makeCanonical(candidate);
            }
        }
        return std::nullopt;
    };

    std::istringstream contentStream(content);
    std::string line;
    bool isInsideBlockComment = false;
    while (std::getline(contentStream, line)) {
        std::string stripped = stripComments(line, isInsideBlockComment);
        auto includePath = parseIncludeDirective(stripped);
        if (!includePath.has_value()) { continue; }

        auto resolvedPath = resolveIncludePath(canonicalPath, *includePath);
        if (!resolvedPath.has_value()) {
            visitingPaths.erase(cacheKey);
            hashCache[cacheKey] = std::nullopt;
            return std::nullopt;
        }

        auto includedHash = computeFileDependencyHash(*resolvedPath, includeDirectories, hashCache, visitingPaths);
        if (!includedHash.has_value()) {
            visitingPaths.erase(cacheKey);
            hashCache[cacheKey] = std::nullopt;
            return std::nullopt;
        }
        updateHashValue(hash, *includedHash);
    }

    visitingPaths.erase(cacheKey);
    hashCache[cacheKey] = hash;
    return hash;
}

void vk::ShaderSpirvCache::trimCacheDirectory(const std::filesystem::path &cacheDir, size_t maxFiles) {
    std::error_code errorCode;
    std::vector<std::pair<std::filesystem::file_time_type, std::filesystem::path>> entries;
    for (const auto &entry : std::filesystem::directory_iterator(cacheDir, errorCode)) {
        if (errorCode) break;
        if (entry.path().extension() == ".spv" && entry.is_regular_file()) {
            entries.emplace_back(entry.last_write_time(errorCode), entry.path());
        }
    }
    if (entries.size() <= maxFiles) return;
    std::sort(entries.begin(), entries.end());
    for (size_t i = 0; i < entries.size() - maxFiles; i++) {
        std::filesystem::remove(entries[i].second, errorCode);
    }
}

vk::Shader::CompileResult
vk::Shader::compileGlslToSpv(std::string sourcePath,
                             VkShaderStageFlagBits stage,
                             std::unordered_map<std::string, std::string> definitions,
                             std::vector<std::string> includeDirectories,
                             std::string injectedSource,
                             const std::filesystem::path &cacheDir) {
    std::optional<uint64_t> dependencyHash = vk::ShaderSpirvCache::computeDependencyHash(sourcePath, includeDirectories);
#ifdef DEBUG
    bool cacheReadFailed = false;
    std::string cacheFilePath;
#endif

    if (!cacheDir.empty() && dependencyHash.has_value()) {
        std::string hash = vk::ShaderSpirvCache::computeCompiledHash(sourcePath, stage, definitions, includeDirectories,
                                                                 injectedSource, *dependencyHash);
        std::error_code errorCode;
        std::filesystem::path cacheFile = cacheDir / (hash + ".spv");
#ifdef DEBUG
        cacheFilePath = cacheFile.string();
#endif
        if (std::filesystem::is_regular_file(cacheFile, errorCode)) {
            std::vector<uint32_t> cachedSpirv = vk::ShaderSpirvCache::readCachedSpirv(cacheFile);
            if (!cachedSpirv.empty()) {
                return {
                    .sourcePath = std::move(sourcePath),
                    .stage = stage,
                    .spirv = std::move(cachedSpirv),
#ifdef DEBUG
                    .cacheHit = true,
                    .cacheReadFailed = false,
                    .cacheFilePath = std::move(cacheFilePath),
#endif
                };
            }
#ifdef DEBUG
            cacheReadFailed = true;
#endif
        }
    }

    std::ifstream sourceFile(sourcePath, std::ios::binary);
    if (!sourceFile.is_open()) {
        shaderCerr() << "Cannot open source file: " << sourcePath << std::endl;
        exit(EXIT_FAILURE);
    }
    std::string sourceText{std::istreambuf_iterator<char>(sourceFile), std::istreambuf_iterator<char>()};
    sourceText = injectSourceAfterVersion(std::move(sourceText), injectedSource);

    std::vector<std::filesystem::path> includePaths;
    std::filesystem::path sourceFsPath(sourcePath);
    includePaths.emplace_back(sourceFsPath.parent_path());
    for (const std::string &directory : includeDirectories) {
        includePaths.emplace_back(std::filesystem::path(directory));
    }

    shaderc::Compiler compiler;
    shaderc::CompileOptions options;
    for (const auto &[name, value] : definitions) { options.AddMacroDefinition(name, value); }
    options.SetTargetEnvironment(shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_4);
    options.SetSourceLanguage(shaderc_source_language_glsl);
    options.SetOptimizationLevel(shaderc_optimization_level_performance);
    options.SetIncluder(std::make_unique<ShaderIncluder>(includePaths));

    shaderc::SpvCompilationResult result =
        compiler.CompileGlslToSpv(sourceText, shaderKindFromStage(stage), sourcePath.c_str(), options);
    if (result.GetCompilationStatus() != shaderc_compilation_status_success) {
        shaderCerr() << "failed to compile shader source " << sourcePath << "\n"
                     << result.GetErrorMessage() << std::endl;
        exit(EXIT_FAILURE);
    }

    std::vector<uint32_t> spirv(result.cbegin(), result.cend());

    if (!cacheDir.empty() && dependencyHash.has_value()) {
        std::string hash = vk::ShaderSpirvCache::computeCompiledHash(sourcePath, stage, definitions, includeDirectories,
                                                                    injectedSource, *dependencyHash);
        vk::ShaderSpirvCache::writeCachedSpirv(cacheDir / (hash + ".spv"), spirv);
    }

    return {
        .sourcePath = std::move(sourcePath),
        .stage = stage,
        .spirv = std::move(spirv),
#ifdef DEBUG
        .cacheHit = false,
        .cacheReadFailed = cacheReadFailed,
        .cacheFilePath = std::move(cacheFilePath),
#endif
    };
}

void vk::Shader::createModule(const std::vector<uint32_t> &spirv, const std::string &sourcePath) {
    VkShaderModuleCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = spirv.size() * sizeof(uint32_t);
    createInfo.pCode = spirv.data();

    if (vkCreateShaderModule(device_->vkDevice(), &createInfo, nullptr, &module_) != VK_SUCCESS) {
        shaderCerr() << "failed to create shader module for " << sourcePath << std::endl;
        exit(EXIT_FAILURE);
    }
    auto shaderName = std::string("Shader ") + std::filesystem::path(sourcePath).filename().string();
    vk::DebugUtils::setObjectName(device_->vkDevice(), VK_OBJECT_TYPE_SHADER_MODULE, module_, shaderName);

#ifdef DEBUG
    shaderCout() << "created runtime shader module for " << sourcePath << std::endl;
#endif
}

vk::Shader::~Shader() {
    vkDestroyShaderModule(device_->vkDevice(), module_, nullptr);
}

VkShaderModule &vk::Shader::vkShaderModule() {
    return module_;
}
