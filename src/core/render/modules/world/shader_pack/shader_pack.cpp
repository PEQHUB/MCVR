#include "core/render/modules/world/shader_pack/shader_pack.hpp"

#include "core/render/renderer.hpp"
#include "core/util/parallel.hpp"

#include "mz.h"
#include "mz_strm.h"
#include "mz_zip.h"
#include "mz_zip_rw.h"
#include <tinyexpr.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cstring>
#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

using json = nlohmann::json;

namespace {

std::string trimCopy(std::string_view value) {
    size_t start = 0;
    while (start < value.size() && std::isspace(static_cast<unsigned char>(value[start])) != 0) { start++; }

    size_t end = value.size();
    while (end > start && std::isspace(static_cast<unsigned char>(value[end - 1])) != 0) { end--; }

    return std::string(value.substr(start, end - start));
}

std::optional<std::string> unwrapScalarConstructor(std::string_view expression, std::string_view constructorName) {
    std::string_view trimmed = expression;
    while (!trimmed.empty() && std::isspace(static_cast<unsigned char>(trimmed.front())) != 0) {
        trimmed.remove_prefix(1);
    }
    while (!trimmed.empty() && std::isspace(static_cast<unsigned char>(trimmed.back())) != 0) {
        trimmed.remove_suffix(1);
    }

    if (trimmed.size() <= constructorName.size() + 2 ||
        trimmed.substr(0, constructorName.size()) != constructorName) {
        return std::nullopt;
    }

    trimmed.remove_prefix(constructorName.size());
    while (!trimmed.empty() && std::isspace(static_cast<unsigned char>(trimmed.front())) != 0) {
        trimmed.remove_prefix(1);
    }
    if (trimmed.empty() || trimmed.front() != '(' || trimmed.back() != ')') { return std::nullopt; }

    int depth = 0;
    for (size_t i = 0; i < trimmed.size(); ++i) {
        if (trimmed[i] == '(') {
            depth++;
        } else if (trimmed[i] == ')') {
            depth--;
            if (depth == 0 && i + 1 != trimmed.size()) { return std::nullopt; }
            if (depth < 0) { return std::nullopt; }
        }
    }
    if (depth != 0) { return std::nullopt; }

    return trimCopy(trimmed.substr(1, trimmed.size() - 2));
}

std::string parseAttributeValueForDefineExpression(const ShaderPackLoader::AttributeConfig &attribute,
                                                   const std::string &currentValue) {
    if (attribute.type == "bool") {
        return currentValue == "render_pipeline.true" || currentValue == "true" || currentValue == "1" ? "1" : "0";
    }
    if (attribute.type.rfind("enum:", 0) == 0) {
        std::string options = attribute.type.substr(5);
        std::stringstream stream(options);
        std::string option;
        int index = 0;
        while (std::getline(stream, option, '-')) {
            if (option == currentValue) { return std::to_string(index); }
            index++;
        }
    }
    return currentValue;
}

std::string substitutePlaceholderInDefineExpression(const std::string &expression, const std::string &value) {
    auto isSymbolChar = [](char ch) { return std::isalnum(static_cast<unsigned char>(ch)) != 0 || ch == '_'; };
    std::string resolved;
    resolved.reserve(expression.size() + value.size());
    for (size_t i = 0; i < expression.size(); i++) {
        if (expression[i] == 'X') {
            bool leftIsSymbol = i > 0 && isSymbolChar(expression[i - 1]);
            bool rightIsSymbol = i + 1 < expression.size() && isSymbolChar(expression[i + 1]);
            if (!leftIsSymbol && !rightIsSymbol) {
                resolved += value;
                continue;
            }
        }
        resolved += expression[i];
    }
    return resolved;
}

std::optional<double> tryEvaluateScalarDefineExpression(const ShaderPackLoader::AttributeConfig &attribute,
                                                        const std::string &currentValue,
                                                        const std::string &expression) {
    std::string resolved = substitutePlaceholderInDefineExpression(
        expression, parseAttributeValueForDefineExpression(attribute, currentValue));

    bool stripped = true;
    while (stripped) {
        stripped = false;
        for (std::string_view constructorName : {"uint", "int", "float", "double"}) {
            auto inner = unwrapScalarConstructor(resolved, constructorName);
            if (!inner.has_value()) { continue; }
            resolved = std::move(*inner);
            stripped = true;
            break;
        }
    }

    try {
        return ExpressionEvaluator::evaluate(resolved, {});
    } catch (...) { return std::nullopt; }
}

}

double ExpressionEvaluator::trueValue() {
    return 1.0;
}

double ExpressionEvaluator::falseValue() {
    return 0.0;
}

double ExpressionEvaluator::ifValue(double condition, double ifTrueValue, double ifFalseValue) {
    return condition != 0.0 ? ifTrueValue : ifFalseValue;
}

bool ExpressionEvaluator::isValidIdentifier(std::string_view name) {
    if (name.empty()) { return false; }
    if (std::isalpha(static_cast<unsigned char>(name.front())) == 0) { return false; }
    for (char c : name) {
        if (std::isalnum(static_cast<unsigned char>(c)) == 0 && c != '_') { return false; }
    }
    return true;
}

bool ExpressionEvaluator::isReservedIdentifier(std::string_view name) {
    return name == "if" || name == "true" || name == "false";
}

double ExpressionEvaluator::evaluate(const std::string &expression, const std::vector<Variable> &variables) {
    std::vector<std::string> names;
    std::vector<double> values;
    std::vector<te_variable> teVariables;

    names.reserve(variables.size() + 2);
    values.reserve(variables.size() + 2);
    teVariables.reserve(variables.size() + 3);

    auto appendVariable = [&](const std::string &name, double value) {
        names.push_back(name);
        values.push_back(value);
        teVariables.push_back({
            .name = names.back().c_str(),
            .address = &values.back(),
            .type = TE_VARIABLE,
            .context = nullptr,
        });
    };

    appendVariable("true", trueValue());
    appendVariable("false", falseValue());

    for (const auto &variable : variables) {
        if (!isValidIdentifier(variable.name)) {
            throw std::runtime_error("invalid execution variable name: " + variable.name);
        }
        if (isReservedIdentifier(variable.name)) {
            throw std::runtime_error("reserved execution variable name: " + variable.name);
        }
        appendVariable(variable.name, variable.value);
    }

    teVariables.push_back({
        .name = "if",
        .address = reinterpret_cast<const void *>(&ExpressionEvaluator::ifValue),
        .type = TE_FUNCTION3 | TE_FLAG_PURE,
        .context = nullptr,
    });

    int error = 0;
    te_expr *compiled =
        te_compile(expression.c_str(), teVariables.data(), static_cast<int>(teVariables.size()), &error);
    if (compiled == nullptr) {
        throw std::runtime_error("invalid execution expression: " + expression + " at " + std::to_string(error));
    }

    double value = te_eval(compiled);
    te_free(compiled);
    return value;
}

std::string ShaderPackLoader::toLower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return value;
}

std::string ShaderPackLoader::trim(std::string value) {
    auto isWhitespace = [](unsigned char ch) { return std::isspace(ch) != 0; };
    value.erase(value.begin(),
                std::find_if(value.begin(), value.end(), [&](unsigned char ch) { return !isWhitespace(ch); }));
    value.erase(std::find_if(value.rbegin(), value.rend(), [&](unsigned char ch) { return !isWhitespace(ch); }).base(),
                value.end());
    return value;
}

static uint64_t fnv1a64(std::string_view data) {
    uint64_t hash = 14695981039346656037ULL;
    for (unsigned char c : data) hash = (hash ^ c) * 1099511628211ULL;
    return hash;
}

std::string ShaderPackLoader::stableFolderName(const fs::path &path) {
    std::error_code ec;
    fs::path normalizedPath = fs::weakly_canonical(path, ec);
    if (ec) { normalizedPath = path.lexically_normal(); }
    std::string key = normalizedPath.string();
    size_t hashValue = static_cast<size_t>(fnv1a64(key));
    std::ostringstream stream;
    stream << normalizedPath.stem().string() << "_" << std::hex << hashValue;
    return stream.str();
}

bool ShaderPackLoader::extractZip(const fs::path &zipPath,
                                  const fs::path &destinationPath,
                                  std::string &error) {
    std::error_code ec;
    fs::remove_all(destinationPath, ec);
    fs::create_directories(destinationPath, ec);
    if (ec) {
        error = "failed to prepare temp dir: " + destinationPath.string();
        return false;
    }

    std::shared_ptr<void> zipReader(mz_zip_reader_create(), [](void *handle) {
        if (handle == nullptr) { return; }
        void *zipReaderHandle = handle;
        mz_zip_reader_delete(&zipReaderHandle);
    });
    if (zipReader == nullptr) {
        error = "failed to create zip reader";
        return false;
    }

    int32_t openResult = mz_zip_reader_open_file(zipReader.get(), zipPath.string().c_str());
    if (openResult != MZ_OK) {
        error = "failed to open zip file";
        return false;
    }

    int32_t extractResult = mz_zip_reader_save_all(zipReader.get(), destinationPath.string().c_str());
    int32_t closeResult = mz_zip_reader_close(zipReader.get());

    if (extractResult != MZ_OK) {
        error = "failed to extract zip file";
        return false;
    }
    if (closeResult != MZ_OK) {
        error = "failed to close zip file";
        return false;
    }

    return true;
}

fs::path ShaderPackLoader::resolveShaderPackRoot(const fs::path &path, std::string &error) {
    if (path.empty()) {
        error = "shader pack path is empty";
        return {};
    }

    std::error_code ec;
    if (!fs::exists(path, ec)) {
        error = "shader pack path does not exist: " + path.string();
        return {};
    }

    if (fs::is_directory(path, ec)) { return fs::weakly_canonical(path, ec); }

    if (toLower(path.extension().string()) != ".zip") {
        error = "shader pack must be a directory or zip: " + path.string();
        return {};
    }

    const fs::path extractRoot = Renderer::folderPath / "temp/shaders/world/ray_tracing";
    const fs::path destination = extractRoot / stableFolderName(path);
    if (!extractZip(path, destination, error)) { return {}; }

    return destination;
}

std::optional<std::reference_wrapper<const json>>
ShaderPackLoader::findKey(const json &value, std::initializer_list<std::string_view> keys) {
    for (std::string_view key : keys) {
        auto iter = value.find(std::string(key));
        if (iter != value.end()) { return std::cref(*iter); }
    }
    return std::nullopt;
}

const json &ShaderPackLoader::requireObject(const json &value, std::string_view key, const std::string &context) {
    std::string keyString(key);
    auto iter = value.find(keyString);
    if (iter == value.end() || !iter->is_object()) {
        throw std::runtime_error(context + "." + keyString + " must be an object");
    }
    return *iter;
}

const json &ShaderPackLoader::requireArray(const json &value, std::string_view key, const std::string &context) {
    std::string keyString(key);
    auto iter = value.find(keyString);
    if (iter == value.end() || !iter->is_array()) {
        throw std::runtime_error(context + "." + keyString + " must be an array");
    }
    return *iter;
}

std::string ShaderPackLoader::requireString(const json &value, std::string_view key, const std::string &context) {
    std::string keyString(key);
    auto iter = value.find(keyString);
    if (iter == value.end() || !iter->is_string()) {
        throw std::runtime_error(context + "." + keyString + " must be a string");
    }
    return iter->get<std::string>();
}

std::optional<std::string>
ShaderPackLoader::optionalString(const json &value, std::string_view key, const std::string &context) {
    std::string keyString(key);
    auto iter = value.find(keyString);
    if (iter == value.end()) { return std::nullopt; }
    if (!iter->is_string()) { throw std::runtime_error(context + "." + keyString + " must be a string"); }
    return iter->get<std::string>();
}

bool ShaderPackLoader::optionalBool(const json &value, std::string_view key, bool fallback, const std::string &context) {
    std::string keyString(key);
    auto iter = value.find(keyString);
    if (iter == value.end()) { return fallback; }
    if (!iter->is_boolean()) { throw std::runtime_error(context + "." + keyString + " must be a boolean"); }
    return iter->get<bool>();
}

VkFormat ShaderPackLoader::parseFormat(const std::string &value) {
    const std::string lower = toLower(value);
    if (lower == "r8_unorm") return VK_FORMAT_R8_UNORM;
    if (lower == "r8g8_unorm") return VK_FORMAT_R8G8_UNORM;
    if (lower == "r8g8b8a8_unorm") return VK_FORMAT_R8G8B8A8_UNORM;
    if (lower == "r16_sfloat") return VK_FORMAT_R16_SFLOAT;
    if (lower == "r16g16_sfloat") return VK_FORMAT_R16G16_SFLOAT;
    if (lower == "r16g16b16a16_sfloat") return VK_FORMAT_R16G16B16A16_SFLOAT;
    if (lower == "r32_sfloat") return VK_FORMAT_R32_SFLOAT;
    if (lower == "r32g32_sfloat") return VK_FORMAT_R32G32_SFLOAT;
    if (lower == "r32g32_uint") return VK_FORMAT_R32G32_UINT;
    if (lower == "r32g32b32a32_sfloat") return VK_FORMAT_R32G32B32A32_SFLOAT;
    if (lower == "r32g32b32a32_uint") return VK_FORMAT_R32G32B32A32_UINT;
    throw std::runtime_error("unsupported format: " + value);
}

VkFilter ShaderPackLoader::parseFilter(const std::string &value) {
    const std::string lower = toLower(value);
    if (lower == "nearest") return VK_FILTER_NEAREST;
    if (lower == "linear") return VK_FILTER_LINEAR;
    throw std::runtime_error("unsupported filter: " + value);
}

VkSamplerAddressMode ShaderPackLoader::parseAddressMode(const std::string &value) {
    const std::string lower = toLower(value);
    if (lower == "repeat") return VK_SAMPLER_ADDRESS_MODE_REPEAT;
    if (lower == "clamp_to_edge") return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    if (lower == "mirrored_repeat") return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
    throw std::runtime_error("unsupported address mode: " + value);
}

ShaderPackLoader::TextureDimension ShaderPackLoader::parseTextureDimension(const std::string &value) {
    const std::string lower = toLower(value);
    if (lower == ShaderPackLoader::VALUE_2D) return ShaderPackLoader::TextureDimension::Texture2D;
    if (lower == ShaderPackLoader::VALUE_2D_ARRAY) return ShaderPackLoader::TextureDimension::Texture2DArray;
    if (lower == ShaderPackLoader::VALUE_3D) return ShaderPackLoader::TextureDimension::Texture3D;
    if (lower == ShaderPackLoader::VALUE_CUBE) return ShaderPackLoader::TextureDimension::Cube;
    throw std::runtime_error("unsupported texture dimension: " + value);
}

std::unordered_map<std::string, std::string> ShaderPackLoader::parseTranslations(const fs::path &langPath) {
    std::unordered_map<std::string, std::string> translations;
    std::ifstream stream(langPath);
    if (!stream.is_open()) { return translations; }

    std::string line;
    while (std::getline(stream, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') { continue; }
        size_t separator = line.find('=');
        if (separator == std::string::npos) { continue; }
        std::string key = trim(line.substr(0, separator));
        std::string value = trim(line.substr(separator + 1));
        if (!key.empty()) { translations[key] = value; }
    }

    return translations;
}

std::unordered_map<std::string, std::string> ShaderPackLoader::loadLanguage(const fs::path &rootPath,
                                                                            const std::string &language) {
    std::unordered_map<std::string, std::string> translations;
    const fs::path langRoot = rootPath / ShaderPackLoader::KEY_LANG;
    if (!fs::exists(langRoot) || !fs::is_directory(langRoot)) { return translations; }

    auto normalizeLanguageCode = [&](std::string code) {
        code = trim(toLower(std::move(code)));
        std::replace(code.begin(), code.end(), '\\', '/');

        size_t separator = code.find_last_of('/');
        if (separator != std::string::npos) { code = code.substr(separator + 1); }

        auto stripSuffix = [&](std::string_view suffix) {
            if (code.size() < suffix.size()) { return false; }
            if (code.compare(code.size() - suffix.size(), suffix.size(), suffix) != 0) { return false; }
            code.resize(code.size() - suffix.size());
            return true;
        };
        while (stripSuffix(".json") || stripSuffix(".lang")) {}

        std::string normalized;
        normalized.reserve(code.size());
        bool previousUnderscore = false;
        for (char ch : code) {
            unsigned char uch = static_cast<unsigned char>(ch);
            if (std::isalnum(uch) != 0) {
                normalized.push_back(static_cast<char>(std::tolower(uch)));
                previousUnderscore = false;
                continue;
            }
            if (ch == '@' || ch == '#') { break; }
            if ((ch == '-' || ch == '_' || ch == '.' || std::isspace(uch) != 0) &&
                !normalized.empty() && !previousUnderscore) {
                normalized.push_back('_');
                previousUnderscore = true;
            }
        }

        while (!normalized.empty() && normalized.back() == '_') {
            normalized.pop_back();
        }
        return normalized;
    };

    std::unordered_map<std::string, fs::path> availableTranslations;
    for (const auto &entry : fs::directory_iterator(langRoot)) {
        if (!entry.is_regular_file()) { continue; }
        if (toLower(entry.path().extension().string()) != ".lang") { continue; }

        std::string normalizedCode = normalizeLanguageCode(entry.path().stem().string());
        if (!normalizedCode.empty()) {
            availableTranslations.emplace(std::move(normalizedCode), entry.path());
        }
    }

    auto mergeFrom = [&](const std::string &code) {
        std::string normalizedCode = normalizeLanguageCode(code);
        if (normalizedCode.empty()) { return; }

        auto pathIter = availableTranslations.find(normalizedCode);
        if (pathIter == availableTranslations.end()) { return; }

        auto loaded = parseTranslations(pathIter->second);
        for (const auto &[key, value] : loaded) { translations[key] = value; }
    };

    mergeFrom("en_us");
    if (!language.empty()) {
        std::vector<std::string> languageCandidates;
        auto addCandidate = [&](const std::string &candidate) {
            if (candidate.empty()) { return; }
            if (std::find(languageCandidates.begin(), languageCandidates.end(), candidate) != languageCandidates.end()) {
                return;
            }
            languageCandidates.push_back(candidate);
        };

        std::string normalizedLanguage = normalizeLanguageCode(language);
        addCandidate(normalizedLanguage);

        std::string reducedLanguage = normalizedLanguage;
        while (true) {
            size_t underscore = reducedLanguage.rfind('_');
            if (underscore == std::string::npos) { break; }
            reducedLanguage.resize(underscore);
            addCandidate(reducedLanguage);
        }

        for (const std::string &candidate : languageCandidates) {
            if (candidate != "en_us") { mergeFrom(candidate); }
        }
    }

    return translations;
}

std::vector<std::string> ShaderPackLoader::parseStringArray(const json &value,
                                                            std::string_view key,
                                                            const std::string &context) {
    std::vector<std::string> values;
    auto iter = value.find(std::string(key));
    if (iter == value.end()) { return values; }
    if (!iter->is_array()) { throw std::runtime_error(context + "." + std::string(key) + " must be an array"); }
    values.reserve(iter->size());
    for (size_t i = 0; i < iter->size(); i++) {
        if (!(*iter)[i].is_string()) {
            throw std::runtime_error(context + "." + std::string(key) + "[" + std::to_string(i) + "] must be a string");
        }
        values.push_back((*iter)[i].get<std::string>());
    }
    return values;
}

fs::path ShaderPackLoader::resolveRelativeFile(const fs::path &rootPath,
                                               const std::string &reference,
                                               const std::string &context) {
    if (reference.empty()) { throw std::runtime_error(context + " cannot be empty"); }
    const fs::path relative(reference);
    if (relative.is_absolute()) { throw std::runtime_error(context + " must be relative to shader pack root"); }
    const fs::path resolved = rootPath / relative;
    if (!fs::exists(resolved)) {
        throw std::runtime_error(context + " references missing file: " + reference);
    }
    return resolved;
}

std::vector<fs::path> ShaderPackLoader::collectShaderFiles(const fs::path &rootPath) {
    std::vector<fs::path> shaderFiles;
    for (const auto &entry : fs::recursive_directory_iterator(rootPath)) {
        if (entry.is_regular_file()) { shaderFiles.push_back(entry.path()); }
    }
    std::sort(shaderFiles.begin(), shaderFiles.end(),
              [](const fs::path &lhs, const fs::path &rhs) { return lhs.generic_string() < rhs.generic_string(); });
    return shaderFiles;
}

std::unordered_map<std::string, ShaderPackLoader::ClassifiedHitShaderPaths>
ShaderPackLoader::classifyHitShaders(const std::vector<fs::path> &shaderFiles) {
    std::unordered_map<std::string, ShaderPackLoader::ClassifiedHitShaderPaths> classified;
    for (const auto &shaderPath : shaderFiles) {
        std::string extension = toLower(shaderPath.extension().string());
        if (extension == ".rahit" || extension == ".rchit" || extension == ".rint") {
            auto &group = classified[shaderPath.stem().string()];
            if (extension == ".rahit" && !group.anyHit.has_value()) {
                group.anyHit = shaderPath;
            } else if (extension == ".rchit" && !group.closestHit.has_value()) {
                group.closestHit = shaderPath;
            } else if (extension == ".rint" && !group.intersection.has_value()) {
                group.intersection = shaderPath;
            }
        }
    }
    return classified;
}

std::unordered_map<std::string, std::string> ShaderPackLoader::parseDefinitions(const json &jsonValue,
                                                                                const std::string &context) {
    std::unordered_map<std::string, std::string> definitions;

    std::optional<std::reference_wrapper<const json>> iter;
    int present = 0;
    for (std::string_view key : {std::string_view(ShaderPackLoader::KEY_DEFINITIONS),
                                 std::string_view(ShaderPackLoader::KEY_DEFINES),
                                 std::string_view(ShaderPackLoader::KEY_DEFINE)}) {
        auto candidate = findKey(jsonValue, {key});
        if (candidate.has_value()) {
            iter = candidate;
            present++;
        }
    }
    if (present > 1) { throw std::runtime_error(context + " cannot define multiple define objects"); }
    if (!iter.has_value()) { return definitions; }
    if (!iter->get().is_object()) {
        throw std::runtime_error(context + "." + ShaderPackLoader::KEY_DEFINITIONS + " must be an object");
    }
    for (auto defIter = iter->get().begin(); defIter != iter->get().end(); ++defIter) {
        if (!defIter.value().is_string()) {
            throw std::runtime_error(context + "." + ShaderPackLoader::KEY_DEFINITIONS + "." + defIter.key() +
                                     " must be a string");
        }
        definitions[defIter.key()] = defIter.value().get<std::string>();
    }
    return definitions;
}

std::optional<std::string> ShaderPackLoader::inferScalarType(const json &value, bool allowString) {
    if (value.is_boolean()) { return std::string("bool"); }
    if (value.is_number_integer() || value.is_number_unsigned()) { return std::string("int"); }
    if (value.is_number_float()) { return std::string("float"); }
    if (allowString && value.is_string()) { return std::string("string"); }
    return std::nullopt;
}

std::string ShaderPackLoader::parseScalarValue(const json &value, const std::string &context) {
    if (value.is_string()) { return value.get<std::string>(); }
    if (value.is_boolean()) { return value.get<bool>() ? "true" : "false"; }
    if (value.is_number()) { return value.dump(); }
    throw std::runtime_error(context + " must be a string, boolean, or number");
}

std::string ShaderPackLoader::parseNumericExpressionValue(const json &value, const std::string &context) {
    if (value.is_string()) { return value.get<std::string>(); }
    if (value.is_number()) { return value.dump(); }
    throw std::runtime_error(context + " must be a string or number");
}

ShaderPackLoader::DefineConfig ShaderPackLoader::parseAttributeDefinition(const json &attributeJson,
                                                                          const std::string &context) {
    ShaderPackLoader::DefineConfig define;

    auto defineNode = findKey(attributeJson, {ShaderPackLoader::KEY_DEFINE});
    if (!defineNode.has_value()) { return define; }

    if (defineNode->get().is_string()) {
        define.direct = defineNode->get().get<std::string>();
        return define;
    }

    if (!defineNode->get().is_object()) {
        throw std::runtime_error(context + "." + ShaderPackLoader::KEY_DEFINE + " must be a string or object");
    }

    bool hasStringValues = false;
    bool hasObjectValues = false;
    for (auto iter = defineNode->get().begin(); iter != defineNode->get().end(); ++iter) {
        hasStringValues |= iter.value().is_string();
        hasObjectValues |= iter.value().is_object();
        if (!iter.value().is_string() && !iter.value().is_object()) {
            throw std::runtime_error(context + "." + ShaderPackLoader::KEY_DEFINE + "." + iter.key() +
                                     " must be a string or object");
        }
    }

    if (hasStringValues && hasObjectValues) {
        throw std::runtime_error(context + "." + ShaderPackLoader::KEY_DEFINE + " cannot mix string and object entries");
    }

    if (hasObjectValues) {
        for (auto valueIter = defineNode->get().begin(); valueIter != defineNode->get().end(); ++valueIter) {
            auto &definitions = define.variants[valueIter.key()];
            for (auto defIter = valueIter.value().begin(); defIter != valueIter.value().end(); ++defIter) {
                if (!defIter.value().is_string()) {
                    throw std::runtime_error(context + "." + ShaderPackLoader::KEY_DEFINE + "." + valueIter.key() + "." +
                                             defIter.key() + " must be a string");
                }
                definitions[defIter.key()] = defIter.value().get<std::string>();
            }
        }
        return define;
    }

    for (auto defIter = defineNode->get().begin(); defIter != defineNode->get().end(); ++defIter) {
        define.expressions[defIter.key()] = defIter.value().get<std::string>();
    }
    return define;
}

ShaderPackLoader::AttributeConfig ShaderPackLoader::parseAttributeConfig(const json &attributeJson,
                                                                         const std::string &context) {
    if (!attributeJson.is_object()) { throw std::runtime_error(context + " must be an object"); }

    ShaderPackLoader::AttributeConfig attribute;
    attribute.name = requireString(attributeJson, ShaderPackLoader::KEY_NAME, context);
    attribute.type = requireString(attributeJson, ShaderPackLoader::KEY_TYPE, context);
    attribute.defaultValue = requireString(attributeJson, ShaderPackLoader::KEY_DEFAULT_VALUE, context);
    attribute.define = parseAttributeDefinition(attributeJson, context);
    return attribute;
}

ShaderPackLoader::VariableConfig ShaderPackLoader::parseVariableConfig(const json &variableJson, const std::string &context) {
    if (!variableJson.is_object() || variableJson.size() != 1) {
        throw std::runtime_error(context + " must be an object with exactly one variable");
    }

    auto variableIter = variableJson.begin();
    ShaderPackLoader::VariableConfig variable;
    variable.name = variableIter.key();
    if (!ExpressionEvaluator::isValidIdentifier(variable.name)) {
        throw std::runtime_error(context + "." + variable.name + " has an invalid variable name");
    }
    if (ExpressionEvaluator::isReservedIdentifier(variable.name)) {
        throw std::runtime_error(context + "." + variable.name + " uses a reserved variable name");
    }
    if (variableIter.value().is_object()) {
        auto defaultValueIter = variableIter.value().find(ShaderPackLoader::KEY_DEFAULT_VALUE);
        if (defaultValueIter == variableIter.value().end()) {
            throw std::runtime_error(context + "." + variable.name + "." + ShaderPackLoader::KEY_DEFAULT_VALUE +
                                     " must be present");
        }
        auto typeIter = variableIter.value().find(ShaderPackLoader::KEY_TYPE);
        if (typeIter != variableIter.value().end()) {
            if (!typeIter->is_string()) {
                throw std::runtime_error(context + "." + variable.name + "." + ShaderPackLoader::KEY_TYPE +
                                         " must be a string");
            }
            variable.type = typeIter->get<std::string>();
        } else {
            auto inferredType = inferScalarType(*defaultValueIter, true);
            if (!inferredType.has_value()) {
                throw std::runtime_error(context + "." + variable.name + "." + ShaderPackLoader::KEY_TYPE +
                                         " could not be inferred");
            }
            variable.type = *inferredType;
        }
        variable.defaultValue = parseScalarValue(*defaultValueIter,
                                                 context + "." + variable.name + "." +
                                                     ShaderPackLoader::KEY_DEFAULT_VALUE);
        return variable;
    }

    auto inferredType = inferScalarType(variableIter.value(), true);
    if (!inferredType.has_value()) {
        throw std::runtime_error(context + "." + variable.name + " must be an object or scalar");
    }
    variable.type = *inferredType;
    variable.defaultValue = parseScalarValue(variableIter.value(), context + "." + variable.name);
    return variable;
}

ShaderPackLoader::TextureConfig
ShaderPackLoader::parseTextureConfig(const json &textureJson,
                                     const fs::path &rootPath,
                                     const std::string &context) {
    if (!textureJson.is_object()) { throw std::runtime_error(context + " must be an object"); }

    ShaderPackLoader::TextureConfig texture;
    texture.name = requireString(textureJson, ShaderPackLoader::KEY_NAME, context);
    std::string type = requireString(textureJson, ShaderPackLoader::KEY_TYPE, context);
    std::string loweredType = toLower(type);
    if (loweredType == ShaderPackLoader::VALUE_FILE) {
        texture.imported = true;
    } else if (loweredType == ShaderPackLoader::VALUE_INTERMEDIATE) {
        texture.imported = false;
    } else {
        throw std::runtime_error("unsupported texture type: " + type);
    }
    texture.dimension =
        parseTextureDimension(optionalString(textureJson, ShaderPackLoader::KEY_DIMENSION, context)
                                  .value_or(ShaderPackLoader::VALUE_2D));
    texture.filter =
        parseFilter(optionalString(textureJson, ShaderPackLoader::KEY_FILTER, context).value_or("linear"));
    texture.addressMode =
        parseAddressMode(optionalString(textureJson, ShaderPackLoader::KEY_ADDRESS_MODE, context).value_or("repeat"));
    texture.shared = optionalBool(textureJson, ShaderPackLoader::KEY_SHARED, false, context);

    auto sampledBindingIter = textureJson.find(ShaderPackLoader::KEY_SAMPLED_BINDING);
    if (sampledBindingIter != textureJson.end()) {
        if (!sampledBindingIter->is_number_unsigned()) {
            throw std::runtime_error(context + "." + ShaderPackLoader::KEY_SAMPLED_BINDING + " must be an unsigned integer");
        }
        texture.sampledBinding = sampledBindingIter->get<uint32_t>();
    }

    auto storageBindingIter = textureJson.find(ShaderPackLoader::KEY_STORAGE_BINDING);
    if (storageBindingIter != textureJson.end()) {
        if (!storageBindingIter->is_number_unsigned()) {
            throw std::runtime_error(context + "." + ShaderPackLoader::KEY_STORAGE_BINDING + " must be an unsigned integer");
        }
        texture.storageBinding = storageBindingIter->get<uint32_t>();
    }
    if (texture.imported) {
        texture.sourcePath = resolveRelativeFile(rootPath, requireString(textureJson, ShaderPackLoader::KEY_PATH, context),
                                                 context + "." + ShaderPackLoader::KEY_PATH);
        texture.format =
            parseFormat(optionalString(textureJson, ShaderPackLoader::KEY_FORMAT, context).value_or("R8G8B8A8_UNORM"));
        if (!texture.sampledBinding.has_value()) { throw std::runtime_error(context + " requires sampled_binding"); }
        if (texture.dimension == ShaderPackLoader::TextureDimension::Texture3D ||
            texture.dimension == ShaderPackLoader::TextureDimension::Texture2DArray) {
            auto parseImportedExtent = [&](std::string_view key) {
                const std::string keyString(key);
                auto iter = textureJson.find(keyString);
                if (iter == textureJson.end() || !iter->is_number_unsigned()) {
                    throw std::runtime_error(context + "." + keyString +
                                             " must be an unsigned integer for imported layered textures");
                }
                return iter->get<uint32_t>();
            };
            texture.importedWidth = parseImportedExtent(ShaderPackLoader::KEY_WIDTH);
            texture.importedHeight = parseImportedExtent(ShaderPackLoader::KEY_HEIGHT);
            texture.importedDepth = parseImportedExtent(ShaderPackLoader::KEY_DEPTH);
            if (texture.importedWidth == 0 || texture.importedHeight == 0 || texture.importedDepth <= 1) {
                throw std::runtime_error(context +
                                         " imported layered textures require positive width/height and depth > 1");
            }
        } else if (texture.dimension != ShaderPackLoader::TextureDimension::Texture2D) {
            throw std::runtime_error(context + " currently only supports imported 2d, 2d_array, or 3d textures");
        }
    } else {
        texture.format = parseFormat(requireString(textureJson, ShaderPackLoader::KEY_FORMAT, context));
        auto widthIter = textureJson.find(ShaderPackLoader::KEY_WIDTH);
        auto heightIter = textureJson.find(ShaderPackLoader::KEY_HEIGHT);
        auto depthIter = textureJson.find(ShaderPackLoader::KEY_DEPTH);
        auto scaleIter = textureJson.find(ShaderPackLoader::KEY_SCALE);
        if (widthIter != textureJson.end() && heightIter != textureJson.end()) {
            texture.widthExpression = parseNumericExpressionValue(*widthIter, context + "." + ShaderPackLoader::KEY_WIDTH);
            texture.heightExpression = parseNumericExpressionValue(*heightIter, context + "." + ShaderPackLoader::KEY_HEIGHT);
            if (texture.dimension == ShaderPackLoader::TextureDimension::Texture3D ||
                texture.dimension == ShaderPackLoader::TextureDimension::Texture2DArray) {
                if (depthIter == textureJson.end()) {
                    throw std::runtime_error(context + "." + ShaderPackLoader::KEY_DEPTH +
                                             " must be present for intermediate layered textures");
                }
                texture.depthExpression =
                    parseNumericExpressionValue(*depthIter, context + "." + ShaderPackLoader::KEY_DEPTH);
            } else if (depthIter != textureJson.end()) {
                throw std::runtime_error(context + "." + ShaderPackLoader::KEY_DEPTH +
                                         " is only supported for intermediate 2d_array or 3d textures");
            }
        } else if (scaleIter != textureJson.end()) {
            if (texture.dimension == ShaderPackLoader::TextureDimension::Texture3D ||
                texture.dimension == ShaderPackLoader::TextureDimension::Texture2DArray) {
                throw std::runtime_error(context +
                                         " intermediate 2d_array and 3d textures require explicit width/height/depth");
            }
            if (!scaleIter->is_number()) {
                throw std::runtime_error(context + "." + ShaderPackLoader::KEY_SCALE + " must be numeric");
            }
            texture.scale = scaleIter->get<float>();
        } else {
            throw std::runtime_error(context + " requires width/height or scale");
        }
    }

    return texture;
}

ShaderPackLoader::BufferConfig ShaderPackLoader::parseBufferConfig(const json &bufferJson, const std::string &context) {
    if (!bufferJson.is_object()) { throw std::runtime_error(context + " must be an object"); }

    ShaderPackLoader::BufferConfig buffer;
    buffer.name = requireString(bufferJson, ShaderPackLoader::KEY_NAME, context);
    std::string type = requireString(bufferJson, ShaderPackLoader::KEY_TYPE, context);
    if (toLower(type) != ShaderPackLoader::VALUE_INTERMEDIATE) {
        throw std::runtime_error("unsupported buffer type: " + type);
    }

    buffer.shared = optionalBool(bufferJson, ShaderPackLoader::KEY_SHARED, false, context);
    auto sizeIter = bufferJson.find(ShaderPackLoader::KEY_SIZE);
    if (sizeIter == bufferJson.end()) { throw std::runtime_error(context + "." + ShaderPackLoader::KEY_SIZE + " must be present"); }
    buffer.sizeExpression = parseNumericExpressionValue(*sizeIter, context + "." + ShaderPackLoader::KEY_SIZE);

    auto bindingIter = bufferJson.find(ShaderPackLoader::KEY_BINDING);
    if (bindingIter == bufferJson.end() || !bindingIter->is_number_unsigned()) {
        throw std::runtime_error(context + "." + ShaderPackLoader::KEY_BINDING + " must be an unsigned integer");
    }
    buffer.binding = bindingIter->get<uint32_t>();
    return buffer;
}

ShaderPackLoader::Stage ShaderPackLoader::parseStage(const json &value, const std::string &context) {
    if (!value.is_string()) { throw std::runtime_error(context + " must be a string"); }

    const std::string stage = toLower(value.get<std::string>());
    if (stage == ShaderPackLoader::KEY_RAY_TRACING) { return ShaderPackLoader::Stage::RayTracing; }
    if (stage == ShaderPackLoader::KEY_POST_RENDER) { return ShaderPackLoader::Stage::PostRender; }
    throw std::runtime_error("unsupported stage: " + value.get<std::string>());
}

ShaderPackLoader::ResourceList ShaderPackLoader::parsePassResourceList(const json &value,
                                                                       std::string_view key,
                                                                       const std::string &context) {
    ShaderPackLoader::ResourceList resources;

    auto iter = value.find(std::string(key));
    if (iter == value.end()) { return resources; }

    const std::string keyContext = context + "." + std::string(key);
    if (!iter->is_object()) { throw std::runtime_error(keyContext + " must be an object"); }

    for (auto fieldIter = iter->begin(); fieldIter != iter->end(); ++fieldIter) {
        if (fieldIter.key() != ShaderPackLoader::KEY_IMAGES && fieldIter.key() != ShaderPackLoader::KEY_BUFFERS) {
            throw std::runtime_error(keyContext + " only supports " + ShaderPackLoader::KEY_IMAGES + " and " +
                                     ShaderPackLoader::KEY_BUFFERS);
        }
    }

    resources.images = parseStringArray(*iter, ShaderPackLoader::KEY_IMAGES, keyContext);
    resources.buffers = parseStringArray(*iter, ShaderPackLoader::KEY_BUFFERS, keyContext);
    return resources;
}

ShaderPackLoader::RayTracingPassConfig
ShaderPackLoader::parseRayTracingPassConfig(
    const json &passJson,
    const fs::path &rootPath,
    const std::unordered_map<std::string, ShaderPackLoader::ClassifiedHitShaderPaths> &classifiedHitShaders,
    const std::string &context) {
    ShaderPackLoader::RayTracingPassConfig pass;
    pass.name = requireString(passJson, ShaderPackLoader::KEY_NAME, context);
    pass.querySharc = optionalBool(passJson, ShaderPackLoader::KEY_QUERY_SHARC, false, context);
    pass.inputs = parsePassResourceList(passJson, ShaderPackLoader::KEY_INPUTS, context);
    pass.outputs = parsePassResourceList(passJson, ShaderPackLoader::KEY_OUTPUTS, context);
    if (auto widthIter = passJson.find(ShaderPackLoader::KEY_WIDTH); widthIter != passJson.end()) {
        pass.width = parseNumericExpressionValue(*widthIter, context + "." + ShaderPackLoader::KEY_WIDTH);
    }
    if (auto heightIter = passJson.find(ShaderPackLoader::KEY_HEIGHT); heightIter != passJson.end()) {
        pass.height = parseNumericExpressionValue(*heightIter, context + "." + ShaderPackLoader::KEY_HEIGHT);
    }
    if (auto depthIter = passJson.find(ShaderPackLoader::KEY_DEPTH); depthIter != passJson.end()) {
        pass.depth = parseNumericExpressionValue(*depthIter, context + "." + ShaderPackLoader::KEY_DEPTH);
    }
    pass.definitions = parseDefinitions(passJson, context);

    pass.rayGenShaderPath = resolveRelativeFile(rootPath, requireString(passJson, ShaderPackLoader::KEY_RGEN, context),
                                                context + "." + ShaderPackLoader::KEY_RGEN);
    pass.defaultHitGroupName = requireString(passJson, ShaderPackLoader::KEY_DEFAULT_HIT_GROUP, context);

    const json &hitGroupsJson = requireObject(passJson, ShaderPackLoader::KEY_HIT_GROUPS, context);
    if (hitGroupsJson.empty()) {
        throw std::runtime_error(context + "." + ShaderPackLoader::KEY_HIT_GROUPS + " must not be empty");
    }
    if (!hitGroupsJson.contains(pass.defaultHitGroupName)) {
        throw std::runtime_error(context + "." + ShaderPackLoader::KEY_DEFAULT_HIT_GROUP + " is missing in hit_groups");
    }

    std::vector<std::string> groupNames;
    groupNames.reserve(hitGroupsJson.size());
    groupNames.push_back(pass.defaultHitGroupName);
    for (auto iter = hitGroupsJson.begin(); iter != hitGroupsJson.end(); ++iter) {
        if (iter.key() != pass.defaultHitGroupName) { groupNames.push_back(iter.key()); }
    }
    std::sort(groupNames.begin() + 1, groupNames.end());

    auto parseGroupType = [&](const json &groupJson, const std::string &groupName) {
        const std::string typeValue =
            requireString(groupJson, ShaderPackLoader::KEY_TYPE,
                          context + "." + ShaderPackLoader::KEY_HIT_GROUPS + "." + groupName);
        if (typeValue == ShaderPackLoader::VALUE_TRIANGLE) return VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
        if (typeValue == ShaderPackLoader::VALUE_AABB) return VK_RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_KHR;
        throw std::runtime_error("unsupported hit group type: " + typeValue);
    };

    auto parseGroup = [&](const std::string &groupName,
                          const std::optional<fs::path> &fallbackClosestHit) {
        const json &groupJson = hitGroupsJson.at(groupName);
        if (!groupJson.is_object()) {
            throw std::runtime_error(context + "." + ShaderPackLoader::KEY_HIT_GROUPS + "." + groupName +
                                     " must be an object");
        }

        const json &shadersJson =
            requireObject(groupJson, ShaderPackLoader::KEY_SHADERS,
                          context + "." + ShaderPackLoader::KEY_HIT_GROUPS + "." + groupName);
        auto classifiedIter = classifiedHitShaders.find(groupName);
        std::optional<std::reference_wrapper<const ShaderPackLoader::ClassifiedHitShaderPaths>> classified;
        if (classifiedIter != classifiedHitShaders.end()) { classified = std::cref(classifiedIter->second); }

        ShaderPackLoader::HitGroupConfig group;
        group.name = groupName;
        if (auto closestRef =
                optionalString(shadersJson, ShaderPackLoader::KEY_RCHIT,
                               context + "." + ShaderPackLoader::KEY_HIT_GROUPS + "." + groupName + "." +
                                   ShaderPackLoader::KEY_SHADERS);
            closestRef.has_value()) {
            group.closestHit = resolveRelativeFile(rootPath, *closestRef,
                                                   context + "." + ShaderPackLoader::KEY_HIT_GROUPS + "." + groupName +
                                                       "." + ShaderPackLoader::KEY_SHADERS + "." +
                                                       ShaderPackLoader::KEY_RCHIT);
        } else if (classified.has_value() && classified->get().closestHit.has_value()) {
            group.closestHit = classified->get().closestHit;
        }

        if (auto anyRef =
                optionalString(shadersJson, ShaderPackLoader::KEY_RAHIT,
                               context + "." + ShaderPackLoader::KEY_HIT_GROUPS + "." + groupName + "." +
                                   ShaderPackLoader::KEY_SHADERS);
            anyRef.has_value()) {
            group.anyHit = resolveRelativeFile(rootPath, *anyRef,
                                               context + "." + ShaderPackLoader::KEY_HIT_GROUPS + "." + groupName +
                                                   "." + ShaderPackLoader::KEY_SHADERS + "." +
                                                   ShaderPackLoader::KEY_RAHIT);
        } else if (classified.has_value() && classified->get().anyHit.has_value()) {
            group.anyHit = classified->get().anyHit;
        }

        if (auto intersectionRef =
                optionalString(shadersJson, ShaderPackLoader::KEY_RINT,
                               context + "." + ShaderPackLoader::KEY_HIT_GROUPS + "." + groupName + "." +
                                   ShaderPackLoader::KEY_SHADERS);
            intersectionRef.has_value()) {
            group.intersection = resolveRelativeFile(rootPath, *intersectionRef,
                                                     context + "." + ShaderPackLoader::KEY_HIT_GROUPS + "." +
                                                         groupName + "." + ShaderPackLoader::KEY_SHADERS + "." +
                                                         ShaderPackLoader::KEY_RINT);
        } else if (classified.has_value() && classified->get().intersection.has_value()) {
            group.intersection = classified->get().intersection;
        }

        if (!group.closestHit.has_value() && !group.intersection.has_value() && fallbackClosestHit.has_value()) {
            group.closestHit = fallbackClosestHit;
        }

        VkRayTracingShaderGroupTypeKHR requestedType = parseGroupType(groupJson, groupName);
        if (group.intersection.has_value()) {
            group.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_KHR;
        } else if (requestedType == VK_RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_KHR) {
            throw std::runtime_error("aabb hit group requires rint: " + groupName);
        } else {
            group.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
        }

        if (!group.closestHit.has_value() && !group.intersection.has_value()) {
            throw std::runtime_error("hit group must provide rchit or rint: " + groupName);
        }

        return group;
    };

    ShaderPackLoader::HitGroupConfig defaultGroup = parseGroup(pass.defaultHitGroupName, std::nullopt);
    pass.hitGroups.push_back(defaultGroup);
    for (size_t i = 1; i < groupNames.size(); i++) {
        pass.hitGroups.push_back(parseGroup(groupNames[i], defaultGroup.closestHit));
    }

    const json &missShadersJson = requireArray(passJson, ShaderPackLoader::KEY_MISS, context);
    if (missShadersJson.empty()) {
        throw std::runtime_error(context + "." + ShaderPackLoader::KEY_MISS + " must not be empty");
    }
    for (size_t i = 0; i < missShadersJson.size(); i++) {
        const json &missJson = missShadersJson[i];
        if (!missJson.is_object()) {
            throw std::runtime_error(context + "." + ShaderPackLoader::KEY_MISS + "[" + std::to_string(i) +
                                     "] must be an object");
        }
        pass.missShaders.push_back({
            .name = requireString(missJson, ShaderPackLoader::KEY_NAME,
                                  context + "." + ShaderPackLoader::KEY_MISS + "[" + std::to_string(i) + "]"),
            .shaderPath = resolveRelativeFile(
                rootPath,
                requireString(missJson, ShaderPackLoader::KEY_SHADER,
                              context + "." + ShaderPackLoader::KEY_MISS + "[" + std::to_string(i) + "]"),
                context + "." + ShaderPackLoader::KEY_MISS + "[" + std::to_string(i) + "]." +
                    ShaderPackLoader::KEY_SHADER),
        });
    }

    return pass;
}

ShaderPackLoader::FullScreenPassConfig
ShaderPackLoader::parseFullScreenPassConfig(const json &passJson, const fs::path &rootPath, const std::string &context) {
    ShaderPackLoader::FullScreenPassConfig pass;
    pass.name = requireString(passJson, ShaderPackLoader::KEY_NAME, context);
    pass.fragmentShaderPath = resolveRelativeFile(rootPath, requireString(passJson, ShaderPackLoader::KEY_FRAGMENT, context),
                                                  context + "." + ShaderPackLoader::KEY_FRAGMENT);
    pass.target = requireString(passJson, ShaderPackLoader::KEY_TARGET, context);
    pass.inputs = parsePassResourceList(passJson, ShaderPackLoader::KEY_INPUTS, context);
    pass.outputs = parsePassResourceList(passJson, ShaderPackLoader::KEY_OUTPUTS, context);
    if (std::find(pass.outputs.images.begin(), pass.outputs.images.end(), pass.target) == pass.outputs.images.end()) {
        pass.outputs.images.push_back(pass.target);
    }
    auto faceIter = passJson.find(ShaderPackLoader::KEY_FACE);
    if (faceIter != passJson.end()) {
        if (!faceIter->is_number_integer()) {
            throw std::runtime_error(context + "." + ShaderPackLoader::KEY_FACE + " must be an integer");
        }
        int face = faceIter->get<int>();
        if (face < 0 || face > 5) {
            throw std::runtime_error(context + "." + ShaderPackLoader::KEY_FACE + " must be between 0 and 5");
        }
        pass.face = static_cast<uint32_t>(face);
    }
    auto layerIter = passJson.find(ShaderPackLoader::KEY_LAYER);
    if (layerIter != passJson.end()) {
        if (!layerIter->is_number_unsigned()) {
            throw std::runtime_error(context + "." + ShaderPackLoader::KEY_LAYER + " must be an unsigned integer");
        }
        pass.layer = layerIter->get<uint32_t>();
    }
    if (pass.face.has_value() && pass.layer.has_value()) {
        throw std::runtime_error(context + " cannot specify both face and layer");
    }
    pass.definitions = parseDefinitions(passJson, context);
    return pass;
}

ShaderPackLoader::RenderShaderConfig
ShaderPackLoader::parseRenderShaderConfig(const json &shaderJson, const fs::path &rootPath, const std::string &context) {
    if (!shaderJson.is_object()) { throw std::runtime_error(context + " must be an object"); }

    ShaderPackLoader::RenderShaderConfig shader;
    shader.vertexShaderPath =
        resolveRelativeFile(rootPath, requireString(shaderJson, ShaderPackLoader::KEY_VERTEX, context),
                            context + "." + ShaderPackLoader::KEY_VERTEX);
    shader.fragmentShaderPath =
        resolveRelativeFile(rootPath, requireString(shaderJson, ShaderPackLoader::KEY_FRAGMENT, context),
                            context + "." + ShaderPackLoader::KEY_FRAGMENT);
    return shader;
}

ShaderPackLoader::RenderPassConfig
ShaderPackLoader::parseRenderPassConfig(const json &passJson, const fs::path &rootPath, const std::string &context) {
    ShaderPackLoader::RenderPassConfig pass;
    pass.name = requireString(passJson, ShaderPackLoader::KEY_NAME, context);
    if (auto shaderIter = passJson.find(ShaderPackLoader::KEY_SHADER); shaderIter != passJson.end()) {
        if (!shaderIter->is_object() || shaderIter->empty()) {
            throw std::runtime_error(context + "." + ShaderPackLoader::KEY_SHADER + " must be a non-empty object");
        }

        for (auto shaderEntry = shaderIter->begin(); shaderEntry != shaderIter->end(); ++shaderEntry) {
            pass.shaderConfigs.emplace(
                shaderEntry.key(),
                parseRenderShaderConfig(shaderEntry.value(), rootPath,
                                        context + "." + ShaderPackLoader::KEY_SHADER + "." + shaderEntry.key()));
        }
    } else {
        pass.vertexShaderPath =
            resolveRelativeFile(rootPath, requireString(passJson, ShaderPackLoader::KEY_VERTEX, context),
                                context + "." + ShaderPackLoader::KEY_VERTEX);
        pass.fragmentShaderPath =
            resolveRelativeFile(rootPath, requireString(passJson, ShaderPackLoader::KEY_FRAGMENT, context),
                                context + "." + ShaderPackLoader::KEY_FRAGMENT);
        pass.shaderConfigs.emplace("default", ShaderPackLoader::RenderShaderConfig{
                                                  .vertexShaderPath = pass.vertexShaderPath,
                                                  .fragmentShaderPath = pass.fragmentShaderPath,
                                              });
    }
    pass.content = requireString(passJson, ShaderPackLoader::KEY_CONTENT, context);
    pass.inputs = parsePassResourceList(passJson, ShaderPackLoader::KEY_INPUTS, context);
    pass.outputs = parsePassResourceList(passJson, ShaderPackLoader::KEY_OUTPUTS, context);
    if (pass.outputs.images.empty()) { pass.outputs.images.push_back("out:ldr"); }

    auto optionalBoolField = [&](std::string_view key) -> std::optional<bool> {
        auto iter = passJson.find(std::string(key));
        if (iter == passJson.end()) { return std::nullopt; }
        if (!iter->is_boolean()) { throw std::runtime_error(context + "." + std::string(key) + " must be a boolean"); }
        return iter->get<bool>();
    };
    pass.depthTest = optionalBoolField(ShaderPackLoader::KEY_DEPTH_TEST);
    pass.depthWrite = optionalBoolField(ShaderPackLoader::KEY_DEPTH_WRITE);
    pass.depthCompare = optionalString(passJson, ShaderPackLoader::KEY_DEPTH_COMPARE, context);
    pass.colorBlend = optionalBool(passJson, ShaderPackLoader::KEY_COLOR_BLEND, true, context);
    pass.definitions = parseDefinitions(passJson, context);
    return pass;
}

ShaderPackLoader::ComputePassConfig
ShaderPackLoader::parseComputePassConfig(const json &passJson, const fs::path &rootPath, const std::string &context) {
    ShaderPackLoader::ComputePassConfig pass;
    pass.name = requireString(passJson, ShaderPackLoader::KEY_NAME, context);
    pass.computeShaderPath = resolveRelativeFile(rootPath, requireString(passJson, ShaderPackLoader::KEY_COMPUTE, context),
                                                 context + "." + ShaderPackLoader::KEY_COMPUTE);
    pass.inputs = parsePassResourceList(passJson, ShaderPackLoader::KEY_INPUTS, context);
    pass.outputs = parsePassResourceList(passJson, ShaderPackLoader::KEY_OUTPUTS, context);
    if (auto gxIter = passJson.find(ShaderPackLoader::KEY_GX); gxIter != passJson.end()) {
        pass.gx = parseNumericExpressionValue(*gxIter, context + "." + ShaderPackLoader::KEY_GX);
    }
    if (auto gyIter = passJson.find(ShaderPackLoader::KEY_GY); gyIter != passJson.end()) {
        pass.gy = parseNumericExpressionValue(*gyIter, context + "." + ShaderPackLoader::KEY_GY);
    }
    if (auto gzIter = passJson.find(ShaderPackLoader::KEY_GZ); gzIter != passJson.end()) {
        pass.gz = parseNumericExpressionValue(*gzIter, context + "." + ShaderPackLoader::KEY_GZ);
    }
    pass.definitions = parseDefinitions(passJson, context);
    return pass;
}

ShaderPackLoader::PassConfig
ShaderPackLoader::parsePassConfig(
    const json &passJson,
    const fs::path &rootPath,
    const std::unordered_map<std::string, ShaderPackLoader::ClassifiedHitShaderPaths> &classifiedHitShaders,
    const std::string &context) {
    if (!passJson.is_object()) { throw std::runtime_error(context + " must be an object"); }

    const std::string type = requireString(passJson, ShaderPackLoader::KEY_TYPE, context);
    ShaderPackLoader::PassConfig pass;
    if (type == ShaderPackLoader::VALUE_FULL_SCREEN) {
        pass.type = ShaderPackLoader::PassConfig::Type::FullScreen;
    } else if (type == ShaderPackLoader::VALUE_RENDER) {
        pass.type = ShaderPackLoader::PassConfig::Type::Render;
    } else if (type == ShaderPackLoader::VALUE_RAY_TRACING) {
        pass.type = ShaderPackLoader::PassConfig::Type::RayTracing;
    } else if (type == ShaderPackLoader::VALUE_COMPUTE) {
        pass.type = ShaderPackLoader::PassConfig::Type::Compute;
    } else {
        throw std::runtime_error("unsupported pass type: " + type);
    }

    if (auto stageIter = passJson.find(ShaderPackLoader::KEY_STAGE); stageIter != passJson.end()) {
        pass.stage = parseStage(*stageIter, context + "." + ShaderPackLoader::KEY_STAGE);
    } else {
        pass.stage = ShaderPackLoader::Stage::RayTracing;
    }

    if (pass.type == ShaderPackLoader::PassConfig::Type::RayTracing &&
        pass.stage != ShaderPackLoader::Stage::RayTracing) {
        throw std::runtime_error("ray_tracing pass must use ray_tracing stage: " + context);
    }
    if (pass.type == ShaderPackLoader::PassConfig::Type::Render &&
        pass.stage != ShaderPackLoader::Stage::PostRender) {
        throw std::runtime_error("render pass must use post_render stage: " + context);
    }

    switch (pass.type) {
        case ShaderPackLoader::PassConfig::Type::FullScreen:
            pass.fullScreen = parseFullScreenPassConfig(passJson, rootPath, context);
            return pass;
        case ShaderPackLoader::PassConfig::Type::Render:
            pass.render = parseRenderPassConfig(passJson, rootPath, context);
            return pass;
        case ShaderPackLoader::PassConfig::Type::RayTracing:
            pass.rayTracing = parseRayTracingPassConfig(passJson, rootPath, classifiedHitShaders, context);
            return pass;
        case ShaderPackLoader::PassConfig::Type::Compute:
            pass.compute = parseComputePassConfig(passJson, rootPath, context);
            return pass;
    }

    throw std::runtime_error("unsupported pass type: " + type);
}

ShaderPackLoader::ExecutionCommand ShaderPackLoader::parseExecutionCommand(const json &commandJson,
                                                                           const std::string &context) {
    if (!commandJson.is_object() || commandJson.size() != 1) {
        throw std::runtime_error(context + " must be an object with exactly one command");
    }

    ShaderPackLoader::ExecutionCommand command;
    if (auto assignIter = commandJson.find(ShaderPackLoader::KEY_ASSIGN); assignIter != commandJson.end()) {
        if (!assignIter->is_object() || assignIter->empty()) {
            throw std::runtime_error(context + "." + ShaderPackLoader::KEY_ASSIGN + " must be a non-empty object");
        }
        command.type = ShaderPackLoader::ExecutionCommand::Type::Assign;
        for (auto valueIter = assignIter->begin(); valueIter != assignIter->end(); ++valueIter) {
            ShaderPackLoader::ExecutionCommand::Assignment assignment;
            if (valueIter.value().is_object()) {
                auto typeIter = valueIter.value().find(ShaderPackLoader::KEY_TYPE);
                if (typeIter != valueIter.value().end()) {
                    if (!typeIter->is_string()) {
                        throw std::runtime_error(context + "." + ShaderPackLoader::KEY_ASSIGN + "." + valueIter.key() +
                                                 "." + ShaderPackLoader::KEY_TYPE + " must be a string");
                    }
                    assignment.type = typeIter->get<std::string>();
                    assignment.explicitType = true;
                }

                auto assignmentValueIter = valueIter.value().find(ShaderPackLoader::KEY_VALUE);
                if (assignmentValueIter == valueIter.value().end()) {
                    throw std::runtime_error(context + "." + ShaderPackLoader::KEY_ASSIGN + "." + valueIter.key() +
                                             "." + ShaderPackLoader::KEY_VALUE + " must be present");
                }

                for (auto fieldIter = valueIter.value().begin(); fieldIter != valueIter.value().end(); ++fieldIter) {
                    if (fieldIter.key() != ShaderPackLoader::KEY_TYPE &&
                        fieldIter.key() != ShaderPackLoader::KEY_VALUE) {
                        throw std::runtime_error(context + "." + ShaderPackLoader::KEY_ASSIGN + "." + valueIter.key() +
                                                 " only supports " + ShaderPackLoader::KEY_TYPE + " and " +
                                                 ShaderPackLoader::KEY_VALUE);
                    }
                }
                assignment.value = *assignmentValueIter;
            } else {
                assignment.type = inferScalarType(valueIter.value(), false);
                assignment.value = valueIter.value();
            }
            command.assignments[valueIter.key()] = std::move(assignment);
        }
        return command;
    }

    if (auto passIter = commandJson.find(ShaderPackLoader::KEY_PASS); passIter != commandJson.end()) {
        if (!passIter->is_string()) {
            throw std::runtime_error(context + "." + ShaderPackLoader::KEY_PASS + " must be a string");
        }
        command.type = ShaderPackLoader::ExecutionCommand::Type::Pass;
        command.passName = passIter->get<std::string>();
        return command;
    }

    if (auto ifElseIter = commandJson.find(ShaderPackLoader::KEY_IF_ELSE); ifElseIter != commandJson.end()) {
        if (!ifElseIter->is_object()) {
            throw std::runtime_error(context + "." + ShaderPackLoader::KEY_IF_ELSE + " must be an object");
        }
        command.type = ShaderPackLoader::ExecutionCommand::Type::IfElse;
        auto conditionIter = ifElseIter->find(ShaderPackLoader::KEY_CONDITION);
        if (conditionIter == ifElseIter->end()) {
            throw std::runtime_error(context + "." + ShaderPackLoader::KEY_IF_ELSE + "." +
                                     ShaderPackLoader::KEY_CONDITION + " must be present");
        }
        command.condition = *conditionIter;
        command.thenCommands = parseExecutionCommands(
            requireArray(*ifElseIter, ShaderPackLoader::KEY_THEN, context + "." + ShaderPackLoader::KEY_IF_ELSE),
            context + "." + ShaderPackLoader::KEY_IF_ELSE + "." + ShaderPackLoader::KEY_THEN);
        auto elseIter = ifElseIter->find(ShaderPackLoader::KEY_ELSE);
        if (elseIter != ifElseIter->end()) {
            command.elseCommands = parseExecutionCommands(
                requireArray(*ifElseIter, ShaderPackLoader::KEY_ELSE, context + "." + ShaderPackLoader::KEY_IF_ELSE),
                context + "." + ShaderPackLoader::KEY_IF_ELSE + "." + ShaderPackLoader::KEY_ELSE);
        }
        return command;
    }

    if (auto whileIter = commandJson.find(ShaderPackLoader::KEY_WHILE); whileIter != commandJson.end()) {
        if (!whileIter->is_object()) {
            throw std::runtime_error(context + "." + ShaderPackLoader::KEY_WHILE + " must be an object");
        }
        command.type = ShaderPackLoader::ExecutionCommand::Type::While;
        auto conditionIter = whileIter->find(ShaderPackLoader::KEY_CONDITION);
        if (conditionIter == whileIter->end()) {
            throw std::runtime_error(context + "." + ShaderPackLoader::KEY_WHILE + "." +
                                     ShaderPackLoader::KEY_CONDITION + " must be present");
        }
        command.condition = *conditionIter;
        command.thenCommands = parseExecutionCommands(
            requireArray(*whileIter, ShaderPackLoader::KEY_THEN, context + "." + ShaderPackLoader::KEY_WHILE),
            context + "." + ShaderPackLoader::KEY_WHILE + "." + ShaderPackLoader::KEY_THEN);
        return command;
    }

    throw std::runtime_error(context + " must contain assign, pass, if_else, or while");
}

std::vector<ShaderPackLoader::ExecutionCommand>
ShaderPackLoader::parseExecutionCommands(const json &commandsJson, const std::string &context) {
    if (!commandsJson.is_array()) { throw std::runtime_error(context + " must be an array"); }

    std::vector<ShaderPackLoader::ExecutionCommand> commands;
    commands.reserve(commandsJson.size());
    for (size_t i = 0; i < commandsJson.size(); i++) {
        commands.push_back(parseExecutionCommand(commandsJson[i], context + "[" + std::to_string(i) + "]"));
    }
    return commands;
}

ShaderPackLoader::ExecutionConfig ShaderPackLoader::parseExecutionConfig(const json &configJson, const std::string &context) {
    ShaderPackLoader::ExecutionConfig execution;
    if (!configJson.is_object()) { throw std::runtime_error(context + " must be an object"); }

    auto parseVariables = [&](std::vector<ShaderPackLoader::VariableConfig> &variables,
                              std::string_view key,
                              const std::string &variablesContext) {
        auto variablesIter = configJson.find(std::string(key));
        if (variablesIter == configJson.end()) { return; }
        if (!variablesIter->is_array()) { throw std::runtime_error(variablesContext + " must be an array"); }
        variables.reserve(variablesIter->size());
        for (size_t i = 0; i < variablesIter->size(); i++) {
            variables.push_back(parseVariableConfig((*variablesIter)[i], variablesContext + "[" + std::to_string(i) + "]"));
        }
    };

    parseVariables(execution.globalVariables, ShaderPackLoader::KEY_GLOBAL_VARIABLES,
                   context + "." + ShaderPackLoader::KEY_GLOBAL_VARIABLES);

    auto commandsIter = configJson.find(ShaderPackLoader::KEY_COMMANDS);
    if (commandsIter != configJson.end()) {
        execution.commands = parseExecutionCommands(*commandsIter,
                                                    context + "." + ShaderPackLoader::KEY_COMMANDS);
    }

    std::unordered_map<std::string, std::string> names;
    auto checkVariables = [&](const std::vector<ShaderPackLoader::VariableConfig> &variables, const std::string &scope) {
        for (const auto &variable : variables) {
            auto [iter, inserted] = names.emplace(variable.name, scope);
            if (!inserted) {
                throw std::runtime_error("duplicate execution variable " + variable.name + " in " + iter->second +
                                         " and " + scope);
            }
        }
    };
    checkVariables(execution.globalVariables, ShaderPackLoader::KEY_GLOBAL_VARIABLES);

    std::unordered_map<std::string, size_t> variableIndices;
    std::unordered_map<std::string, bool> placeholderTypes;
    execution.variables = execution.globalVariables;
    variableIndices.reserve(execution.variables.size());
    placeholderTypes.reserve(execution.variables.size());
    for (size_t i = 0; i < execution.variables.size(); i++) {
        variableIndices[execution.variables[i].name] = i;
        placeholderTypes[execution.variables[i].name] = false;
    }
    collectExecutionVariables(execution.commands, execution.variables, variableIndices, placeholderTypes);

    return execution;
}

ShaderPackLoader::ShaderPack ShaderPackLoader::parseConfigFile(const fs::path &rootPath, const std::string &language) {
    const fs::path configPath = rootPath / "configs.json";
    if (!fs::exists(configPath)) {
        throw std::runtime_error("missing shader config: " + configPath.string());
    }

    std::ifstream stream(configPath);
    if (!stream.is_open()) { throw std::runtime_error("failed to open shader config: " + configPath.string()); }

    json configJson;
    stream >> configJson;
    if (!configJson.is_object()) { throw std::runtime_error("shader config root must be an object"); }

    ShaderPackLoader::ShaderPack shaderPack;
    shaderPack.rootPath = rootPath;
    shaderPack.requiresEmission = optionalBool(configJson, ShaderPackLoader::KEY_REQUIRES_EMISSION, false, "root");
    shaderPack.includeDirectories = {rootPath.string()};
    shaderPack.translations = loadLanguage(rootPath, language);
    auto classifiedHitShaders = classifyHitShaders(collectShaderFiles(rootPath));

    auto attributesIter = configJson.find(ShaderPackLoader::KEY_ATTRIBUTES);
    if (attributesIter != configJson.end()) {
        if (!attributesIter->is_array()) { throw std::runtime_error("root.attributes must be an array"); }
        const json &attributesJson = *attributesIter;
        shaderPack.attributes.resize(attributesJson.size());
        mcvr::parallelFor(attributesJson.size(), [&](size_t i) {
            shaderPack.attributes[i] =
                parseAttributeConfig(attributesJson[i], "root.attributes[" + std::to_string(i) + "]");
        });
    }

    auto texturesIter = configJson.find(ShaderPackLoader::KEY_TEXTURES);
    if (texturesIter != configJson.end()) {
        if (!texturesIter->is_array()) { throw std::runtime_error("root.textures must be an array"); }
        const json &texturesJson = *texturesIter;
        shaderPack.textures.resize(texturesJson.size());
        mcvr::parallelFor(texturesJson.size(), [&](size_t i) {
            shaderPack.textures[i] =
                parseTextureConfig(texturesJson[i], rootPath, "root.textures[" + std::to_string(i) + "]");
        });
    }

    auto buffersIter = configJson.find(ShaderPackLoader::KEY_BUFFERS);
    if (buffersIter != configJson.end()) {
        if (!buffersIter->is_array()) { throw std::runtime_error("root.buffers must be an array"); }
        const json &buffersJson = *buffersIter;
        shaderPack.buffers.resize(buffersJson.size());
        mcvr::parallelFor(buffersJson.size(), [&](size_t i) {
            shaderPack.buffers[i] = parseBufferConfig(buffersJson[i], "root.buffers[" + std::to_string(i) + "]");
        });
    }

    auto passesIter = configJson.find(ShaderPackLoader::KEY_PASSES);
    if (passesIter == configJson.end() || !passesIter->is_array() || passesIter->empty()) {
        throw std::runtime_error("root.passes must be a non-empty array");
    }
    const json &passesJson = *passesIter;
    shaderPack.passes.resize(passesJson.size());
    mcvr::parallelFor(passesJson.size(), [&](size_t i) {
        shaderPack.passes[i] =
            parsePassConfig(passesJson[i], rootPath, classifiedHitShaders, "root.passes[" + std::to_string(i) + "]");
    });

    auto sharcIter = configJson.find(ShaderPackLoader::KEY_SHARC);
    if (sharcIter != configJson.end()) {
        if (!sharcIter->is_object()) {
            throw std::runtime_error("root." + std::string(ShaderPackLoader::KEY_SHARC) + " must be an object");
        }

        ShaderPackLoader::SharcConfig sharc;
        sharc.updatePassName =
            requireString(*sharcIter, ShaderPackLoader::KEY_UPDATE_PASS, "root." + std::string(ShaderPackLoader::KEY_SHARC));
        sharc.resolveCompShaderPath =
            resolveRelativeFile(rootPath,
                                requireString(*sharcIter, ShaderPackLoader::KEY_RESOLVE_COMP,
                                              "root." + std::string(ShaderPackLoader::KEY_SHARC)),
                                "root." + std::string(ShaderPackLoader::KEY_SHARC) + "." +
                                    std::string(ShaderPackLoader::KEY_RESOLVE_COMP));

        bool foundUpdatePass = false;
        for (const auto &pass : shaderPack.passes) {
            if (pass.type != ShaderPackLoader::PassConfig::Type::RayTracing) { continue; }
            if (pass.rayTracing.name == sharc.updatePassName) {
                foundUpdatePass = true;
                break;
            }
        }
        if (!foundUpdatePass) {
            throw std::runtime_error("root." + std::string(ShaderPackLoader::KEY_SHARC) + "." +
                                     std::string(ShaderPackLoader::KEY_UPDATE_PASS) +
                                     " must reference a ray_tracing pass");
        }

        shaderPack.sharc = std::move(sharc);
    }

    auto executionIter = configJson.find(ShaderPackLoader::KEY_EXECUTION);
    if (executionIter != configJson.end()) {
        if (!executionIter->is_object()) {
            throw std::runtime_error("root." + std::string(ShaderPackLoader::KEY_EXECUTION) + " must be an object");
        }

        if (executionIter->contains(ShaderPackLoader::KEY_RAY_TRACING)) {
            if (auto rayTracingIter = executionIter->find(ShaderPackLoader::KEY_RAY_TRACING);
                rayTracingIter != executionIter->end()) {
                shaderPack.rayTracingExecution =
                    parseExecutionConfig(*rayTracingIter, "root.execution." + std::string(ShaderPackLoader::KEY_RAY_TRACING));
            }
        } else {
            shaderPack.rayTracingExecution = parseExecutionConfig(*executionIter, "root.execution");
        }
    }

    auto executionPostIter = configJson.find(ShaderPackLoader::KEY_EXECUTION_POST);
    if (executionPostIter != configJson.end()) {
        if (!executionPostIter->is_object()) {
            throw std::runtime_error("root." + std::string(ShaderPackLoader::KEY_EXECUTION_POST) + " must be an object");
        }
        shaderPack.postRenderExecution = parseExecutionConfig(*executionPostIter, "root.execution_post");
    }

    auto appendDefaultExecutionCommands = [&](ShaderPackLoader::ExecutionConfig &execution,
                                              ShaderPackLoader::Stage stage) {
        if (!execution.commands.empty()) { return; }

        for (const auto &pass : shaderPack.passes) {
            if (pass.stage != stage) { continue; }
            execution.commands.push_back({
                .type = ShaderPackLoader::ExecutionCommand::Type::Pass,
                .passName = pass.type == ShaderPackLoader::PassConfig::Type::FullScreen ? pass.fullScreen.name
                                                                 : pass.type == ShaderPackLoader::PassConfig::Type::Render
                                                                     ? pass.render.name
                                                                 : pass.type == ShaderPackLoader::PassConfig::Type::RayTracing
                                                                     ? pass.rayTracing.name
                                                                     : pass.compute.name,
            });
        }
    };
    appendDefaultExecutionCommands(shaderPack.rayTracingExecution, ShaderPackLoader::Stage::RayTracing);
    appendDefaultExecutionCommands(shaderPack.postRenderExecution, ShaderPackLoader::Stage::PostRender);

    std::unordered_map<uint32_t, std::string> runtimeBindings;
    auto registerRuntimeBinding = [&](uint32_t binding, const std::string &owner) {
        auto [iter, inserted] = runtimeBindings.emplace(binding, owner);
        if (!inserted) {
            throw std::runtime_error("duplicate runtime binding " + std::to_string(binding) + ": " + iter->second +
                                     " and " + owner);
        }
    };
    for (const auto &texture : shaderPack.textures) {
        if (texture.sampledBinding.has_value()) {
            registerRuntimeBinding(*texture.sampledBinding, "texture(sampled):" + texture.name);
        }
        if (texture.storageBinding.has_value()) {
            registerRuntimeBinding(*texture.storageBinding, "texture(storage):" + texture.name);
        }
    }
    for (const auto &buffer : shaderPack.buffers) { registerRuntimeBinding(buffer.binding, "buffer:" + buffer.name); }

    return shaderPack;
}

void ShaderPackLoader::collectExecutionVariables(const std::vector<ExecutionCommand> &commands,
                                                 std::vector<VariableConfig> &variables,
                                                 std::unordered_map<std::string, size_t> &variableIndices,
                                                 std::unordered_map<std::string, bool> &placeholderTypes) {
    for (const auto &command : commands) {
        switch (command.type) {
            case ShaderPackLoader::ExecutionCommand::Type::Assign:
                for (const auto &[name, assignment] : command.assignments) {
                    auto type = inferAssignmentType(assignment);
                    auto variableIter = variableIndices.find(name);
                    if (variableIter == variableIndices.end()) {
                        variables.push_back({
                            .name = name,
                            .type = type.value_or("float"),
                            .defaultValue = defaultValueForType(type.value_or("float")),
                        });
                        variableIndices[name] = variables.size() - 1;
                        placeholderTypes[name] = !type.has_value();
                        continue;
                    }

                    if (!type.has_value()) { continue; }

                    VariableConfig &variable = variables[variableIter->second];
                    bool &placeholderType = placeholderTypes[name];
                    if (placeholderType) {
                        variable.type = *type;
                        variable.defaultValue = defaultValueForType(*type);
                        placeholderType = false;
                        continue;
                    }
                    if (variable.type != *type) {
                        throw std::runtime_error("execution variable type mismatch for " + name + ": " + variable.type +
                                                 " and " + *type);
                    }
                }
                break;
            case ShaderPackLoader::ExecutionCommand::Type::IfElse:
                collectExecutionVariables(command.thenCommands, variables, variableIndices, placeholderTypes);
                collectExecutionVariables(command.elseCommands, variables, variableIndices, placeholderTypes);
                break;
            case ShaderPackLoader::ExecutionCommand::Type::While:
                collectExecutionVariables(command.thenCommands, variables, variableIndices, placeholderTypes);
                break;
            case ShaderPackLoader::ExecutionCommand::Type::Pass:
                break;
        }
    }
}

std::optional<std::string>
ShaderPackLoader::inferAssignmentType(const ShaderPackLoader::ExecutionCommand::Assignment &assignment) {
    if (assignment.type.has_value()) { return assignment.type; }
    return inferScalarType(assignment.value, false);
}

std::string ShaderPackLoader::defaultValueForType(const std::string &type) {
    if (type == "bool") { return "false"; }
    if (type == "int" || type.rfind("int_range:", 0) == 0 || type.rfind("enum:", 0) == 0) { return "0"; }
    return "0.0";
}

std::string ShaderPackLoader::buildExecutionSource(const std::vector<VariableConfig> &variables, uint32_t executionSet) {
    if (variables.empty()) { return ""; }

    auto variableAccessor = [](const VariableConfig &variable) {
        const std::string bufferValue = "rtExecutionVariables." + variable.name + "_";
        if (variable.type == "bool") { return "(" + bufferValue + " != 0.0)"; }
        if (variable.type == "int" || variable.type.rfind("int_range:", 0) == 0 ||
            variable.type.rfind("enum:", 0) == 0) {
            return "int(" + bufferValue + ")";
        }
        return bufferValue;
    };

    std::ostringstream stream;
    stream << "layout(set = " << executionSet << ", binding = " << ShaderPackLoader::EXECUTION_BINDING
           << ", std430) readonly buffer RtExecutionVariables {\n";
    for (const auto &variable : variables) { stream << "    float " << variable.name << "_;\n"; }
    stream << "} rtExecutionVariables;\n";
    for (const auto &variable : variables) {
        stream << "#define " << variable.name << " " << variableAccessor(variable) << "\n";
    }
    return stream.str();
}

std::string ShaderPackLoader::parseValue(const std::string &type, const std::string &currentValue) {
    if (type == "bool") {
        return currentValue == "render_pipeline.true" || currentValue == "true" || currentValue == "1" ? "1" : "0";
    }
    else if (type.rfind("enum:", 0) == 0) {
        std::string options = type.substr(5);
        std::stringstream stream(options);
        std::string option;
        int index = 0;
        while (std::getline(stream, option, '-')) {
            if (option == currentValue) { return std::to_string(index); }
            index++;
        }
    }
    return currentValue;
}

std::string ShaderPackLoader::substitutePlaceholder(const std::string &expression, const std::string &value) {
    auto isSymbolChar = [](char ch) { return std::isalnum(static_cast<unsigned char>(ch)) != 0 || ch == '_'; };
    std::string resolved;
    resolved.reserve(expression.size() + value.size());
    for (size_t i = 0; i < expression.size(); i++) {
        if (expression[i] == 'X') {
            bool leftIsSymbol = i > 0 && isSymbolChar(expression[i - 1]);
            bool rightIsSymbol = i + 1 < expression.size() && isSymbolChar(expression[i + 1]);
            if (!leftIsSymbol && !rightIsSymbol) {
                resolved += value;
                continue;
            }
        }
        resolved += expression[i];
    }
    return resolved;
}

void ShaderPackLoader::applyDefinitionExpressions(std::unordered_map<std::string, std::string> &definitions,
                                                  const std::unordered_map<std::string, std::string> &expressions,
                                                  const std::string &value) {
    for (const auto &[name, expression] : expressions) {
        definitions[name] = substitutePlaceholder(expression, value);
    }
}

ShaderPackLoader::LoadResult ShaderPackLoader::load(const fs::path &path,
                                                    const fs::path &builtInPath,
                                                    const std::string &language) {
    LoadResult result;

    auto tryLoad = [&](const fs::path &shaderPackPath)
        -> std::optional<ShaderPackLoader::ShaderPack> {
        std::string error;
        fs::path rootPath = resolveShaderPackRoot(shaderPackPath, error);
        if (rootPath.empty()) {
            result.error = error;
            return std::nullopt;
        }
        try {
            return parseConfigFile(rootPath, language);
        } catch (const std::exception &exception) {
            result.error = exception.what();
            return std::nullopt;
        }
    };

    auto resolvePackPath = [](const fs::path &shaderPackPath) {
        if (shaderPackPath.empty() || shaderPackPath.is_absolute()) {
            return shaderPackPath;
        }
        return Renderer::folderPath / shaderPackPath;
    };

    fs::path fallbackPath = resolvePackPath(builtInPath);
    fs::path requestedPath = resolvePackPath(path.empty() ? builtInPath : path);
    if (auto shaderPack = tryLoad(requestedPath); shaderPack.has_value()) {
        result.success = true;
        result.shaderPack = std::move(*shaderPack);
        return result;
    }

    if (!path.empty() && requestedPath != fallbackPath) {
        if (auto fallback = tryLoad(fallbackPath); fallback.has_value()) {
            result.success = true;
            result.shaderPack = std::move(*fallback);
            return result;
        }
    }

    return result;
}

std::unordered_map<std::string, std::string>
ShaderPackLoader::buildAttributeDefinitions(const std::vector<ShaderPackLoader::AttributeConfig> &attributes,
                                            const std::unordered_map<std::string, std::string> &currentValues) {
    std::unordered_map<std::string, std::string> definitions;
    for (const auto &attribute : attributes) {
        auto valueIter = currentValues.find(attribute.name);
        const std::string &currentValue = valueIter == currentValues.end() ? attribute.defaultValue : valueIter->second;
        std::string value = parseValue(attribute.type, currentValue);

        if (attribute.define.direct.has_value()) { definitions[*attribute.define.direct] = value; }

        applyDefinitionExpressions(definitions, attribute.define.expressions, value);

        auto valueDefinesIter = attribute.define.variants.find(currentValue);
        if (valueDefinesIter != attribute.define.variants.end()) {
            applyDefinitionExpressions(definitions, valueDefinesIter->second, value);
        }
    }
    return definitions;
}

#include "core/render/render_framework.hpp"
#include "core/render/renderer.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

std::shared_ptr<vk::DeviceLocalBuffer>
ShaderPack::createPassExecutionBuffer(std::shared_ptr<vk::Device> device,
                                       std::shared_ptr<vk::VMA> vma,
                                       size_t executionBufferSize) {
    if (executionBufferSize == 0) { return nullptr; }
    return vk::DeviceLocalBuffer::create(vma, device, true, executionBufferSize,
                                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 0, VMA_MEMORY_USAGE_GPU_ONLY);
}

ShaderPack::ShaderPack(std::shared_ptr<Framework> framework) : framework_(framework) {}

std::filesystem::path ShaderPack::builtInShaderPackPath() {
    return Renderer::folderPath / "shaders/world/ray_tracing/vanilla-pt.zip";
}

ShaderPack::BuildConfig
ShaderPack::buildConfigFromRayTracingAttributes(const std::vector<std::string> &attributeKVs) {
    BuildConfig config;
    for (size_t i = 0; i + 1 < attributeKVs.size(); i += 2) {
        const std::string &key = attributeKVs[i];
        const std::string &value = attributeKVs[i + 1];
        config.staticAttributes[key] = value;
        if (key == "render_pipeline.module.ray_tracing.attribute.shader_pack_path") {
            config.shaderPackPath = value;
        } else if (key == "render_pipeline.module.ray_tracing.attribute.use_sharc") {
            config.shouldUseSharc = value == "render_pipeline.true";
        }
    }
    return config;
}

bool ShaderPack::initialize(const BuildConfig &config, std::string &error) {
    ShaderPackLoader::LoadResult loadResult = ShaderPackLoader::load(
        config.shaderPackPath.empty() ? std::filesystem::path{} : std::filesystem::path(config.shaderPackPath),
        builtInShaderPackPath(), config.language);
    if (!loadResult.success) {
        error = loadResult.error;
        return false;
    }
    shaderPack_ = std::move(loadResult.shaderPack);
    staticAttributeValues_ = config.staticAttributes;
    shaderAttributes_ = ShaderPackLoader::buildAttributeDefinitions(shaderPack_.attributes, config.staticAttributes);

    initStageRuntime(rayTracingStageRuntime_, shaderPack_.rayTracingExecution);
    initStageRuntime(postRenderStageRuntime_, shaderPack_.postRenderExecution);

    hasSharcRuntime_ = false;
    if (config.shouldUseSharc && shaderPack_.sharc.has_value()) { hasSharcRuntime_ = true; }

    return true;
}

void ShaderPack::ensureRuntimeResources(uint32_t referenceWidth, uint32_t referenceHeight) {
    if (runtimeResourcesReady_ && referenceWidth_ == referenceWidth && referenceHeight_ == referenceHeight) {
        refreshRuntimeBuffers();
        return;
    }

    runtimeTextures_.clear();
    runtimeTextureIndices_.clear();
    runtimeBuffers_.clear();
    runtimeBufferIndices_.clear();

    referenceWidth_ = referenceWidth;
    referenceHeight_ = referenceHeight;

    initRuntimeTextures();
    initRuntimeBuffers();
    loadRuntimeResources();
    runtimeResourcesReady_ = true;
}

void ShaderPack::refreshRuntimeBuffers() {
    if (!runtimeResourcesReady_ || runtimeBuffers_.empty()) { return; }

    auto framework = framework_.lock();
    auto device = framework->device();
    auto vma = framework->vma();
    auto &gc = framework->gc();
    uint32_t frameCount = framework->swapchain()->imageCount();

    for (auto &runtimeBuffer : runtimeBuffers_) {
        const size_t expectedSize =
            std::max(static_cast<size_t>(1), static_cast<size_t>(std::ceil(evaluateNumericExpression(runtimeBuffer.config.sizeExpression))));

        bool needsResize = runtimeBuffer.frameBuffers.empty();
        if (!needsResize) {
            for (const auto &buffer : runtimeBuffer.frameBuffers) {
                if (buffer == nullptr || buffer->size() != expectedSize) {
                    needsResize = true;
                    break;
                }
            }
        }
        if (!needsResize) { continue; }

        const uint32_t bufferFrameCount = runtimeBuffer.config.shared ? 1 : frameCount;
        runtimeBuffer.frameBuffers.resize(bufferFrameCount);
        for (uint32_t frameIndex = 0; frameIndex < bufferFrameCount; ++frameIndex) {
            gc.collect(runtimeBuffer.frameBuffers[frameIndex]);
            runtimeBuffer.frameBuffers[frameIndex] =
                vk::DeviceLocalBuffer::create(vma, device, false, expectedSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 0,
                                              VMA_MEMORY_USAGE_GPU_ONLY);
        }
    }
}

void ShaderPack::setRuntimeResourceExpressionVariables(std::vector<ExpressionEvaluator::Variable> variables) {
    runtimeResourceExpressionVariables_ = std::move(variables);
}

void ShaderPack::bindRuntimeResources(const std::shared_ptr<vk::DescriptorTable> &descriptorTable,
                                           uint32_t setIndex,
                                           uint32_t frameIndex) {
    if (!runtimeResourcesReady_ || descriptorTable == nullptr) { return; }

    for (const auto &texture : runtimeTextures_) {
        if (texture.config.sampledBinding.has_value()) {
            std::shared_ptr<vk::DeviceLocalImage> image =
                texture.config.imported ? texture.importedImage
                                        : texture.frameImages[texture.config.shared ? 0 : frameIndex];
            VkImageLayout layout = texture.config.imported || !texture.config.storageBinding.has_value() ?
                                       VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL :
                                       VK_IMAGE_LAYOUT_GENERAL;
            descriptorTable->bindSamplerImage(texture.sampler, image, layout, setIndex, *texture.config.sampledBinding, 0,
                                              texture.sampledViewIndex);
        }
        if (texture.config.storageBinding.has_value() && !texture.config.imported) {
            descriptorTable->bindImage(texture.frameImages[texture.config.shared ? 0 : frameIndex], VK_IMAGE_LAYOUT_GENERAL,
                                       setIndex, *texture.config.storageBinding);
        }
    }
    for (const auto &buffer : runtimeBuffers_) {
        descriptorTable->bindBuffer(buffer.frameBuffers[buffer.config.shared ? 0 : frameIndex], setIndex,
                                    buffer.config.binding);
    }
}

void ShaderPack::defineRuntimeResourceDescriptorSet(vk::DescriptorTableBuilder &builder,
                                                    VkShaderStageFlags sampledImageStageFlags,
                                                    VkShaderStageFlags storageImageStageFlags,
                                                    VkShaderStageFlags storageBufferStageFlags) const {
    if (!hasRuntimeResources()) { return; }

    auto &set = builder.beginDescriptorLayoutSet();
    auto &bindings = set.beginDescriptorLayoutSetBinding();
    for (const auto &texture : shaderPack_.textures) {
        if (texture.sampledBinding.has_value()) {
            bindings.defineDescriptorLayoutSetBinding({
                .binding = *texture.sampledBinding,
                .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .descriptorCount = 1,
                .stageFlags = sampledImageStageFlags,
            });
        }
        if (texture.storageBinding.has_value()) {
            bindings.defineDescriptorLayoutSetBinding({
                .binding = *texture.storageBinding,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                .descriptorCount = 1,
                .stageFlags = storageImageStageFlags,
            });
        }
    }
    for (const auto &buffer : shaderPack_.buffers) {
        bindings.defineDescriptorLayoutSetBinding({
            .binding = buffer.binding,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = storageBufferStageFlags,
        });
    }
    bindings.endDescriptorLayoutSetBinding();
    set.endDescriptorLayoutSet();
}

void ShaderPack::defineExecutionDescriptorSet(vk::DescriptorTableBuilder &builder,
                                              ShaderPackLoader::Stage stage,
                                              VkShaderStageFlags stageFlags) const {
    if (execution(stage).variables.empty()) { return; }

    builder.beginDescriptorLayoutSet()
        .beginDescriptorLayoutSetBinding()
        .defineDescriptorLayoutSetBinding({
            .binding = ShaderPackLoader::EXECUTION_BINDING,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = stageFlags,
        })
        .endDescriptorLayoutSetBinding()
        .endDescriptorLayoutSet();
}

void ShaderPack::copyStageExecutionState(
    ShaderPackLoader::Stage stage,
    std::unordered_map<std::string, ShaderPackLoader::VariableConfig> &variableConfigs,
    std::unordered_map<std::string, std::string> &globalVariables) const {
    const auto &runtime = stageRuntime(stage);
    variableConfigs = runtime.executionVariableConfigs;
    globalVariables = runtime.globalVariables;
}

uint32_t ShaderPack::executionSet(uint32_t runtimeResourceSet) const {
    return runtimeResourceSet + (hasRuntimeResources() ? 1u : 0u);
}

bool ShaderPack::hasRuntimeResources() const {
    return !shaderPack_.textures.empty() || !shaderPack_.buffers.empty();
}

void ShaderPack::preClose() {
    runtimeTextures_.clear();
    runtimeTextureIndices_.clear();
    runtimeBuffers_.clear();
    runtimeBufferIndices_.clear();
    runtimeResourcesReady_ = false;
    referenceWidth_ = 0;
    referenceHeight_ = 0;
}

std::shared_ptr<vk::Shader> ShaderPack::createShader(
    std::shared_ptr<vk::Device> device,
    const std::filesystem::path &path,
    VkShaderStageFlagBits stage,
    const std::unordered_map<std::string, std::string> &definitions,
    ShaderPackLoader::Stage executionStage,
    uint32_t executionSet) const {
    const std::filesystem::path cacheDir = Renderer::folderPath / "cache/shaders";
    auto compileResult = vk::Shader::compileGlslToSpv(
        path.string(), stage, mergeDefinitions(shaderAttributes_, definitions),
        shaderPack_.includeDirectories, executionSource(executionStage, executionSet), cacheDir);
    return vk::Shader::create(device, std::move(compileResult));
}

static bool isShaderIdentifierStart(char ch) {
    unsigned char uch = static_cast<unsigned char>(ch);
    return std::isalpha(uch) != 0 || ch == '_';
}

static bool isShaderIdentifierContinue(char ch) {
    unsigned char uch = static_cast<unsigned char>(ch);
    return std::isalnum(uch) != 0 || ch == '_';
}

static std::string stripShaderComments(std::string_view line, bool &insideBlockComment) {
    std::string stripped;
    stripped.reserve(line.size());
    for (size_t i = 0; i < line.size();) {
        if (insideBlockComment) {
            if (i + 1 < line.size() && line[i] == '*' && line[i + 1] == '/') {
                insideBlockComment = false;
                i += 2;
            } else {
                i++;
            }
            continue;
        }
        if (i + 1 < line.size() && line[i] == '/' && line[i + 1] == '*') {
            insideBlockComment = true;
            i += 2;
            continue;
        }
        if (i + 1 < line.size() && line[i] == '/' && line[i + 1] == '/') { break; }
        stripped.push_back(line[i]);
        i++;
    }
    return stripped;
}

static void collectShaderAttributeIdentifiers(std::string_view source,
                                              const std::unordered_set<std::string> &attributeKeys,
                                              std::set<std::string> &used) {
    for (size_t i = 0; i < source.size();) {
        if (!isShaderIdentifierStart(source[i])) {
            i++;
            continue;
        }
        size_t start = i++;
        while (i < source.size() && isShaderIdentifierContinue(source[i])) { i++; }
        std::string identifier(source.substr(start, i - start));
        if (attributeKeys.find(identifier) != attributeKeys.end()) { used.insert(std::move(identifier)); }
    }
}

static std::optional<std::string> parseShaderInclude(std::string_view line) {
    size_t i = 0;
    while (i < line.size() && std::isspace(static_cast<unsigned char>(line[i])) != 0) { i++; }
    if (i >= line.size() || line[i] != '#') { return std::nullopt; }
    i++;
    while (i < line.size() && std::isspace(static_cast<unsigned char>(line[i])) != 0) { i++; }
    size_t directiveStart = i;
    while (i < line.size() && isShaderIdentifierContinue(line[i])) { i++; }
    if (line.substr(directiveStart, i - directiveStart) != "include") { return std::nullopt; }
    while (i < line.size() && std::isspace(static_cast<unsigned char>(line[i])) != 0) { i++; }
    if (i >= line.size() || line[i] != '"') { return std::nullopt; }
    i++;
    size_t pathStart = i;
    while (i < line.size() && line[i] != '"') { i++; }
    if (i >= line.size()) { return std::nullopt; }
    return std::string(line.substr(pathStart, i - pathStart));
}

static std::optional<std::filesystem::path>
resolveShaderInclude(const std::filesystem::path &requestingPath,
                     std::string_view includePath,
                     const std::vector<std::filesystem::path> &includeDirectories) {
    std::error_code ec;
    std::filesystem::path relativeCandidate = requestingPath.parent_path() / std::filesystem::path(includePath);
    if (std::filesystem::exists(relativeCandidate, ec) && std::filesystem::is_regular_file(relativeCandidate, ec)) {
        return std::filesystem::weakly_canonical(relativeCandidate, ec);
    }
    for (const auto &includeDirectory : includeDirectories) {
        std::filesystem::path candidate = includeDirectory / std::filesystem::path(includePath);
        if (std::filesystem::exists(candidate, ec) && std::filesystem::is_regular_file(candidate, ec)) {
            return std::filesystem::weakly_canonical(candidate, ec);
        }
    }
    return std::nullopt;
}

static std::optional<std::set<std::string>>
collectShaderAttributeDependencies(const std::filesystem::path &shaderPath,
                                   const std::unordered_set<std::string> &attributeKeys,
                                   const std::vector<std::filesystem::path> &includeDirectories,
                                   std::unordered_map<std::string, std::optional<std::set<std::string>>> &cache,
                                   std::unordered_set<std::string> &activeFiles) {
    std::error_code ec;
    std::filesystem::path canonicalPath = std::filesystem::weakly_canonical(shaderPath, ec);
    if (ec) { canonicalPath = shaderPath.lexically_normal(); }
    const std::string cacheKey = canonicalPath.string();

    if (auto iter = cache.find(cacheKey); iter != cache.end()) { return iter->second; }
    if (!activeFiles.insert(cacheKey).second) { return std::set<std::string>{}; }

    std::ifstream file(canonicalPath);
    if (!file.is_open()) {
        activeFiles.erase(cacheKey);
        cache[cacheKey] = std::nullopt;
        return std::nullopt;
    }

    std::set<std::string> used;
    std::string line;
    bool insideBlockComment = false;
    while (std::getline(file, line)) {
        std::string stripped = stripShaderComments(line, insideBlockComment);
        collectShaderAttributeIdentifiers(stripped, attributeKeys, used);

        auto includePath = parseShaderInclude(stripped);
        if (!includePath.has_value()) { continue; }
        auto resolvedInclude = resolveShaderInclude(canonicalPath, *includePath, includeDirectories);
        if (!resolvedInclude.has_value()) {
            activeFiles.erase(cacheKey);
            cache[cacheKey] = std::nullopt;
            return std::nullopt;
        }
        auto includeUsed =
            collectShaderAttributeDependencies(*resolvedInclude, attributeKeys, includeDirectories, cache, activeFiles);
        if (!includeUsed.has_value()) {
            activeFiles.erase(cacheKey);
            cache[cacheKey] = std::nullopt;
            return std::nullopt;
        }
        used.insert(includeUsed->begin(), includeUsed->end());
    }

    activeFiles.erase(cacheKey);
    cache[cacheKey] = used;
    return used;
}

static std::unordered_map<std::string, std::string>
filterShaderAttributes(const std::unordered_map<std::string, std::string> &allAttrs,
                       const std::filesystem::path &shaderPath,
                       const std::unordered_set<std::string> &attributeKeys,
                       const std::vector<std::filesystem::path> &includeDirectories,
                       std::unordered_map<std::string, std::optional<std::set<std::string>>> &cache) {
    std::unordered_set<std::string> activeFiles;
    auto used =
        collectShaderAttributeDependencies(shaderPath, attributeKeys, includeDirectories, cache, activeFiles);
    if (!used.has_value()) { return allAttrs; }

    std::unordered_map<std::string, std::string> filtered;
    for (const auto &[k, v] : allAttrs) {
        if (used->find(k) != used->end()) { filtered.emplace(k, v); }
    }
    return filtered;
}

static std::unordered_map<std::string, std::string>
stripAttributeDefinitions(const std::unordered_map<std::string, std::string> &definitions,
                          const std::unordered_set<std::string> &attributeKeys) {
    std::unordered_map<std::string, std::string> stripped;
    for (const auto &[key, value] : definitions) {
        if (attributeKeys.find(key) == attributeKeys.end()) { stripped.emplace(key, value); }
    }
    return stripped;
}

std::vector<std::shared_ptr<vk::Shader>>
ShaderPack::createShaders(std::shared_ptr<vk::Device> device,
                          const std::vector<ShaderCreateInfo> &requests
#ifdef DEBUG
                          ,
                          ShaderBatchStats *stats
#endif
                          ) const {
    if (requests.empty()) { return {}; }

    const std::filesystem::path cacheDir = Renderer::folderPath / "cache/shaders";
    std::vector<std::filesystem::path> includeDirectories;
    includeDirectories.reserve(shaderPack_.includeDirectories.size());
    for (const auto &includeDirectory : shaderPack_.includeDirectories) {
        includeDirectories.emplace_back(includeDirectory);
    }
    std::unordered_set<std::string> attributeKeys;
    attributeKeys.reserve(shaderAttributes_.size());
    for (const auto &[name, value] : shaderAttributes_) {
        (void)value;
        attributeKeys.insert(name);
    }
    std::unordered_map<std::string, std::optional<std::set<std::string>>> dependencyCache;

    const auto buildKey = [&](const ShaderCreateInfo &r,
                              const std::unordered_map<std::string, std::string> &attrDefs,
                              const std::string &execSrc) {
        std::ostringstream key;
        key << r.path.string() << '|' << static_cast<int>(r.stage) << '|';
        auto merged = mergeDefinitions(attrDefs, r.definitions);
        std::vector<std::pair<std::string, std::string>> sortedDefinitions(merged.begin(), merged.end());
        std::sort(sortedDefinitions.begin(), sortedDefinitions.end());
        for (const auto &[k, v] : sortedDefinitions) key << k << '=' << v << ';';
        key << '|' << static_cast<int>(r.executionStage) << '|' << r.executionSet << '|' << execSrc;
        return key.str();
    };

    std::vector<size_t> uniqueIndices;
    std::vector<size_t> requestToUniqueIndex(requests.size());
    std::unordered_map<std::string, size_t> keyToIndex;
    std::vector<std::unordered_map<std::string, std::string>> filteredRequestAttributes(requests.size());
    std::vector<std::unordered_map<std::string, std::string>> requestDefinitions(requests.size());
    std::vector<std::string> executionSources(requests.size());

    for (size_t i = 0; i < requests.size(); i++) {
        filteredRequestAttributes[i] = filterShaderAttributes(
            shaderAttributes_, requests[i].path, attributeKeys, includeDirectories, dependencyCache);
        requestDefinitions[i] = stripAttributeDefinitions(requests[i].definitions, attributeKeys);
        executionSources[i] = executionSource(requests[i].executionStage, requests[i].executionSet);
        ShaderCreateInfo request = requests[i];
        request.definitions = requestDefinitions[i];
        std::string key = buildKey(request, filteredRequestAttributes[i], executionSources[i]);
        auto [iter, inserted] = keyToIndex.emplace(std::move(key), uniqueIndices.size());
        if (inserted) { uniqueIndices.push_back(i); }
        requestToUniqueIndex[i] = iter->second;
    }

    std::vector<vk::Shader::CompileResult> compileResults(uniqueIndices.size());
    mcvr::parallelFor(uniqueIndices.size(), [&](size_t ui) {
        const size_t requestIndex = uniqueIndices[ui];
        const auto &request = requests[requestIndex];
        compileResults[ui] = vk::Shader::compileGlslToSpv(
            request.path.string(), request.stage,
            mergeDefinitions(filteredRequestAttributes[requestIndex], requestDefinitions[requestIndex]),
            shaderPack_.includeDirectories, executionSources[requestIndex], cacheDir);
    });

#ifdef DEBUG
    if (stats != nullptr) {
        stats->requestCount = requests.size();
        stats->uniqueShaderCount = uniqueIndices.size();
        for (const auto &compileResult : compileResults) {
            if (compileResult.cacheHit) {
                stats->cacheHitCount++;
            } else {
                stats->cacheMissCount++;
            }
            if (compileResult.cacheReadFailed) { stats->cacheReadFailureCount++; }
        }
    }
#endif

    std::vector<std::shared_ptr<vk::Shader>> shaders(requests.size());
    mcvr::parallelFor(requests.size(), [&](size_t i) {
        shaders[i] = vk::Shader::create(device, compileResults[requestToUniqueIndex[i]].clone());
    });
    return shaders;
}

const ShaderPackLoader::ShaderPack &ShaderPack::shaderPack() const {
    return shaderPack_;
}

const std::unordered_map<std::string, std::string> &ShaderPack::shaderAttributes() const {
    return shaderAttributes_;
}

const ShaderPack::StageRuntime &ShaderPack::stageRuntime(ShaderPackLoader::Stage stage) const {
    return stage == ShaderPackLoader::Stage::PostRender ? postRenderStageRuntime_ : rayTracingStageRuntime_;
}

const ShaderPackLoader::ExecutionConfig &ShaderPack::execution(ShaderPackLoader::Stage stage) const {
    return stage == ShaderPackLoader::Stage::PostRender ? shaderPack_.postRenderExecution : shaderPack_.rayTracingExecution;
}

std::string ShaderPack::executionSource(ShaderPackLoader::Stage stage, uint32_t executionSet) const {
    return ShaderPackLoader::buildExecutionSource(execution(stage).variables, executionSet);
}

double ShaderPack::evaluateNumericExpression(
    ShaderPackLoader::Stage stage,
    const std::string &expression,
    const ExecutionVariables &variables,
    const std::vector<ExpressionEvaluator::Variable> &additionalVariables,
    bool includeAttributeDefines) const {
    auto parseBoolValue = [](const std::string &value) {
        return value == "true" || value == "1" || value == "render_pipeline.true";
    };
    auto parseNumber = [&](const ShaderPackLoader::VariableConfig &config, const std::string &value) {
        if (config.type == "bool") { return parseBoolValue(value) ? 1.0 : 0.0; }
        if (config.type.rfind("enum:", 0) == 0) {
            std::string options = config.type.substr(5);
            std::stringstream stream(options);
            std::string option;
            int index = 0;
            while (std::getline(stream, option, '-')) {
                if (option == value) { return static_cast<double>(index); }
                index++;
            }
        }
        return std::stod(value);
    };

    std::vector<ExpressionEvaluator::Variable> expressionVariables;
    expressionVariables.reserve(variables.size() + shaderPack_.attributes.size() + additionalVariables.size());
    std::unordered_map<std::string, size_t> variableIndices;
    variableIndices.reserve(expressionVariables.capacity());
    auto appendExpressionVariable = [&](const std::string &name, double value) {
        if (variableIndices.find(name) != variableIndices.end()) { return; }
        variableIndices[name] = expressionVariables.size();
        expressionVariables.push_back({
            .name = name,
            .value = value,
        });
    };
    auto appendDynamicVariable = [&](const std::string &name, const std::string &value, const std::string &type) {
        if (type == "bool") {
            appendExpressionVariable(name, parseBoolValue(value) ? 1.0 : 0.0);
            return;
        }
        if (type.rfind("enum:", 0) == 0) {
            appendExpressionVariable(name, parseNumber({
                                     .name = name,
                                     .type = type,
                                     .defaultValue = value,
                                 }, value));
            return;
        }

        size_t parsed = 0;
        try {
            double numericValue = std::stod(value, &parsed);
            if (parsed == value.size()) { appendExpressionVariable(name, numericValue); }
        } catch (...) {}
    };

    for (const auto &[name, variable] : variables) {
        appendDynamicVariable(variable.name, variable.value, variable.type);
    }

    if (includeAttributeDefines) {
        for (const auto &attribute : shaderPack_.attributes) {
            const std::string &currentValue = [&]() -> const std::string & {
                auto iter = staticAttributeValues_.find(attribute.name);
                return iter == staticAttributeValues_.end() ? attribute.defaultValue : iter->second;
            }();

            if (attribute.define.direct.has_value()) {
                if (attribute.type == "bool") {
                    appendExpressionVariable(*attribute.define.direct, parseBoolValue(currentValue) ? 1.0 : 0.0);
                } else if (attribute.type == "int" || attribute.type.rfind("int_range:", 0) == 0 ||
                           attribute.type.rfind("float_range:", 0) == 0 || attribute.type.rfind("enum:", 0) == 0) {
                    appendExpressionVariable(*attribute.define.direct,
                                             parseNumber({
                                                             .name = attribute.name,
                                                             .type = attribute.type,
                                                             .defaultValue = currentValue,
                                                         },
                                                         currentValue));
                }
            }

            for (const auto &[name, expression] : attribute.define.expressions) {
                auto value = tryEvaluateScalarDefineExpression(attribute, currentValue, expression);
                if (value.has_value()) { appendExpressionVariable(name, *value); }
            }
        }
    }

    for (const auto &variable : additionalVariables) { appendExpressionVariable(variable.name, variable.value); }

    return ExpressionEvaluator::evaluate(expression, expressionVariables);
}

bool ShaderPack::evaluateCondition(ShaderPackLoader::Stage stage,
                                   const nlohmann::json &condition,
                                   const ExecutionVariables &variables,
                                   const std::vector<ExpressionEvaluator::Variable> &additionalVariables,
                                   bool includeAttributeDefines) const {
    if (condition.is_boolean()) { return condition.get<bool>(); }
    if (condition.is_number()) { return condition.get<double>() != 0.0; }
    if (condition.is_string()) {
        return evaluateNumericExpression(stage, condition.get<std::string>(), variables, additionalVariables,
                                         includeAttributeDefines) != 0.0;
    }
    throw std::runtime_error("execution condition must be a string, boolean, or number");
}

std::string ShaderPack::evaluateAssignment(
    ShaderPackLoader::Stage stage,
    ExecutionVariable &variable,
    const ShaderPackLoader::ExecutionCommand::Assignment &assignment,
    const ExecutionVariables &variables,
    const std::vector<ExpressionEvaluator::Variable> &additionalVariables,
    bool includeAttributeDefines) const {
    const nlohmann::json &value = assignment.value;
    const auto &configs = stageRuntime(stage).executionVariableConfigs;
    auto configIter = configs.find(variable.name);
    const bool hasConfig = configIter != configs.end();
    auto formatNumber = [](double numericValue) {
        std::ostringstream stream;
        stream << numericValue;
        return stream.str();
    };
    auto inferType = [&]() {
        if (assignment.type.has_value()) { return *assignment.type; }
        if (!variable.type.empty()) { return variable.type; }
        if (value.is_boolean()) { return std::string("bool"); }
        if (value.is_number_integer() || value.is_number_unsigned()) { return std::string("int"); }
        if (value.is_number_float()) { return std::string("float"); }
        if (value.is_string()) {
            try {
                evaluateNumericExpression(stage, value.get<std::string>(), variables, additionalVariables,
                                          includeAttributeDefines);
                return std::string("float");
            } catch (...) { return std::string("string"); }
        }
        return std::string("string");
    };

    if (hasConfig && assignment.explicitType && assignment.type.has_value() && *assignment.type != configIter->second.type) {
        throw std::runtime_error("execution assignment type for " + variable.name +
                                 " does not match global variable type");
    }

    const std::string type = hasConfig ? configIter->second.type : inferType();
    variable.type = type;
    auto isBoolType = [&]() { return type == "bool"; };
    auto isIntType = [&]() { return type == "int" || type.rfind("int_range:", 0) == 0; };
    auto isFloatType = [&]() { return type == "float" || type.rfind("float_range:", 0) == 0; };

    if (value.is_string()) {
        const std::string stringValue = value.get<std::string>();
        if (isBoolType()) {
            return evaluateNumericExpression(stage, stringValue, variables, additionalVariables,
                                             includeAttributeDefines) != 0.0 ?
                       "true" :
                       "false";
        }
        if (isIntType()) {
            return std::to_string(static_cast<int64_t>(
                evaluateNumericExpression(stage, stringValue, variables, additionalVariables, includeAttributeDefines)));
        }
        if (isFloatType()) {
            return formatNumber(
                evaluateNumericExpression(stage, stringValue, variables, additionalVariables, includeAttributeDefines));
        }
        return stringValue;
    }

    if (value.is_boolean()) { return value.get<bool>() ? "true" : "false"; }
    if (value.is_number()) {
        double numericValue = value.get<double>();
        if (isBoolType()) { return numericValue != 0.0 ? "true" : "false"; }
        if (isIntType()) { return std::to_string(static_cast<int64_t>(numericValue)); }
        if (isFloatType()) { return formatNumber(numericValue); }
        return formatNumber(numericValue);
    }
    throw std::runtime_error("execution assignment for " + variable.name + " must be a string, boolean, or number");
}

void ShaderPack::uploadExecutionBuffer(ShaderPackLoader::Stage stage,
                                       const std::shared_ptr<vk::DeviceLocalBuffer> &executionBuffer,
                                       const ExecutionVariables &variables,
                                       const std::shared_ptr<vk::CommandBuffer> &commandBuffer,
                                       const std::shared_ptr<vk::DescriptorTable> &descriptorTable,
                                       uint32_t executionSet,
                                       uint32_t queueIndex,
                                       VkPipelineStageFlags2 dstStageMask) const {
    const auto &execution = this->execution(stage);
    if (executionBuffer == nullptr || execution.variables.empty()) { return; }

    auto parseBoolValue = [](const std::string &value) {
        return value == "true" || value == "1" || value == "render_pipeline.true";
    };
    auto parseFloatValue = [&](const ShaderPackLoader::VariableConfig &config, const std::string &value) {
        if (config.type == "bool") { return parseBoolValue(value) ? 1.0f : 0.0f; }
        if (config.type == "int" || config.type.rfind("int_range:", 0) == 0) {
            try {
                return static_cast<float>(std::stoi(value));
            } catch (...) { return 0.0f; }
        }
        if (config.type.rfind("enum:", 0) == 0) {
            std::string options = config.type.substr(5);
            std::stringstream stream(options);
            std::string option;
            int index = 0;
            while (std::getline(stream, option, '-')) {
                if (option == value) { return static_cast<float>(index); }
                index++;
            }
            try {
                return static_cast<float>(std::stoi(value));
            } catch (...) { return 0.0f; }
        }
        try {
            return std::stof(value);
        } catch (...) { return 0.0f; }
    };

    std::vector<float> values(execution.variables.size(), 0.0f);
    for (size_t i = 0; i < execution.variables.size(); i++) {
        const auto &variable = execution.variables[i];
        auto valueIter = variables.find(variable.name);
        values[i] = parseFloatValue(variable, valueIter != variables.end() ? valueIter->second.value : variable.defaultValue);
    }

    executionBuffer->uploadToStagingBuffer(values.data(), values.size() * sizeof(float), 0);
    executionBuffer->uploadToBuffer(commandBuffer);
    descriptorTable->bindBuffer(executionBuffer, executionSet, ShaderPackLoader::EXECUTION_BINDING);
    commandBuffer->barriersBufferImage(
        {{
            .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
            .dstStageMask = dstStageMask,
            .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
            .srcQueueFamilyIndex = queueIndex,
            .dstQueueFamilyIndex = queueIndex,
            .buffer = executionBuffer,
        }},
        {});
}

void ShaderPack::executeCommands(ShaderPackLoader::Stage stage,
                                 const std::vector<ShaderPackLoader::ExecutionCommand> &commands,
                                 ExecutionVariables &variables,
                                 const std::vector<ExpressionEvaluator::Variable> &additionalVariables,
                                 bool includeAttributeDefines,
                                 uint32_t loopLimit,
                                 const ExecutePassCallback &executePass) const {
    auto findOrCreateVariable = [&](std::string_view name) -> ExecutionVariable & {
        auto iter = variables.find(std::string(name));
        if (iter != variables.end()) { return iter->second; }

        const auto &configs = stageRuntime(stage).executionVariableConfigs;
        auto configIter = configs.find(std::string(name));
        ExecutionVariable variable = {
            .name = std::string(name),
            .value = configIter != configs.end() ? configIter->second.defaultValue : "",
            .type = configIter != configs.end() ? configIter->second.type : "",
        };
        return variables.emplace(variable.name, std::move(variable)).first->second;
    };

    std::function<void(const ShaderPackLoader::ExecutionCommand &)> executeCommand;
    executeCommand = [&](const ShaderPackLoader::ExecutionCommand &command) {
        switch (command.type) {
            case ShaderPackLoader::ExecutionCommand::Type::Assign:
                for (const auto &[name, value] : command.assignments) {
                    ExecutionVariable &variable = findOrCreateVariable(name);
                    variable.value = evaluateAssignment(stage, variable, value, variables, additionalVariables,
                                                        includeAttributeDefines);
                }
                return;
            case ShaderPackLoader::ExecutionCommand::Type::Pass:
                executePass(command.passName, variables);
                return;
            case ShaderPackLoader::ExecutionCommand::Type::IfElse:
                if (evaluateCondition(stage, command.condition, variables, additionalVariables,
                                      includeAttributeDefines)) {
                    executeCommands(stage, command.thenCommands, variables, additionalVariables,
                                    includeAttributeDefines, loopLimit, executePass);
                } else {
                    executeCommands(stage, command.elseCommands, variables, additionalVariables,
                                    includeAttributeDefines, loopLimit, executePass);
                }
                return;
            case ShaderPackLoader::ExecutionCommand::Type::While: {
                uint32_t iteration = 0;
                while (evaluateCondition(stage, command.condition, variables, additionalVariables,
                                         includeAttributeDefines)) {
                    if (iteration++ >= loopLimit) { throw std::runtime_error("execution while exceeded iteration limit"); }
                    executeCommands(stage, command.thenCommands, variables, additionalVariables,
                                    includeAttributeDefines, loopLimit, executePass);
                }
                return;
            }
        }
    };

    for (const auto &command : commands) { executeCommand(command); }
}

bool ShaderPack::hasSharcRuntime() const {
    return hasSharcRuntime_;
}

bool ShaderPack::runtimeResourcesReady() const {
    return runtimeResourcesReady_;
}

std::optional<std::reference_wrapper<ShaderPack::RuntimeTexture>>
ShaderPack::findRuntimeTexture(std::string_view name) {
    auto iter = runtimeTextureIndices_.find(std::string(name));
    if (iter == runtimeTextureIndices_.end()) { return std::nullopt; }
    return runtimeTextures_[iter->second];
}

std::optional<std::reference_wrapper<const ShaderPack::RuntimeTexture>>
ShaderPack::findRuntimeTexture(std::string_view name) const {
    auto iter = runtimeTextureIndices_.find(std::string(name));
    if (iter == runtimeTextureIndices_.end()) { return std::nullopt; }
    return runtimeTextures_[iter->second];
}

std::optional<std::reference_wrapper<ShaderPack::RuntimeBuffer>>
ShaderPack::findRuntimeBuffer(std::string_view name) {
    auto iter = runtimeBufferIndices_.find(std::string(name));
    if (iter == runtimeBufferIndices_.end()) { return std::nullopt; }
    return runtimeBuffers_[iter->second];
}

std::optional<std::reference_wrapper<const ShaderPack::RuntimeBuffer>>
ShaderPack::findRuntimeBuffer(std::string_view name) const {
    auto iter = runtimeBufferIndices_.find(std::string(name));
    if (iter == runtimeBufferIndices_.end()) { return std::nullopt; }
    return runtimeBuffers_[iter->second];
}

std::shared_ptr<vk::DeviceLocalImage> ShaderPack::findRuntimeVKTexture(RuntimeTexture &runtimeTexture,
                                                                       uint32_t frameIndex) {
    return static_cast<const ShaderPack *>(this)->findRuntimeVKTexture(
        const_cast<const RuntimeTexture &>(runtimeTexture), frameIndex);
}

std::shared_ptr<vk::DeviceLocalImage> ShaderPack::findRuntimeVKTexture(const RuntimeTexture &runtimeTexture,
                                                                       uint32_t frameIndex) const {
    if (runtimeTexture.config.imported) { return runtimeTexture.importedImage; }
    return runtimeTexture.frameImages[runtimeTexture.config.shared ? 0 : frameIndex];
}

uint32_t ShaderPack::runtimeTextureSingleLayerViewIndex(const RuntimeTexture &runtimeTexture, uint32_t layer) const {
    if (runtimeTexture.singleLayerViewBaseIndex == 0) {
        throw std::runtime_error("runtime texture does not expose per-layer views: " + runtimeTexture.config.name);
    }
    if (runtimeTexture.config.dimension == ShaderPackLoader::TextureDimension::Cube) {
        if (layer >= 6) { throw std::runtime_error("cube face index out of range: " + runtimeTexture.config.name); }
        return runtimeTexture.singleLayerViewBaseIndex + layer;
    }
    if (runtimeTexture.config.dimension != ShaderPackLoader::TextureDimension::Texture2DArray) {
        throw std::runtime_error("runtime texture is not a 2d_array: " + runtimeTexture.config.name);
    }

    uint32_t layerCount = 0;
    if (runtimeTexture.config.imported) {
        layerCount = runtimeTexture.config.importedDepth;
    } else if (!runtimeTexture.frameImages.empty() && runtimeTexture.frameImages[0] != nullptr) {
        layerCount = runtimeTexture.frameImages[0]->layer();
    }
    if (layer >= layerCount) {
        throw std::runtime_error("2d_array layer index out of range: " + runtimeTexture.config.name);
    }
    return runtimeTexture.singleLayerViewBaseIndex + layer;
}

std::shared_ptr<vk::DeviceLocalBuffer> ShaderPack::findRuntimeVKBuffer(RuntimeBuffer &runtimeBuffer,
                                                                       uint32_t frameIndex) {
    return static_cast<const ShaderPack *>(this)->findRuntimeVKBuffer(
        const_cast<const RuntimeBuffer &>(runtimeBuffer), frameIndex);
}

std::shared_ptr<vk::DeviceLocalBuffer> ShaderPack::findRuntimeVKBuffer(const RuntimeBuffer &runtimeBuffer,
                                                                       uint32_t frameIndex) const {
    return runtimeBuffer.frameBuffers[runtimeBuffer.config.shared ? 0 : frameIndex];
}

std::unordered_map<std::string, std::string>
ShaderPack::mergeDefinitions(const std::unordered_map<std::string, std::string> &lhs,
                                  const std::unordered_map<std::string, std::string> &rhs) {
    std::unordered_map<std::string, std::string> merged = lhs;
    for (const auto &[key, value] : rhs) { merged[key] = value; }
    return merged;
}

double ShaderPack::evaluateNumericExpression(const std::string &expression) const {
    auto parseBoolValue = [](const std::string &value) {
        return value == "true" || value == "1" || value == "render_pipeline.true";
    };
    auto parseNumber = [&](const ShaderPackLoader::AttributeConfig &config, const std::string &value) {
        if (config.type == "bool") { return parseBoolValue(value) ? 1.0 : 0.0; }
        if (config.type.rfind("enum:", 0) == 0) {
            std::string options = config.type.substr(5);
            std::stringstream stream(options);
            std::string option;
            int index = 0;
            while (std::getline(stream, option, '-')) {
                if (option == value) { return static_cast<double>(index); }
                index++;
            }
        }
        return std::stod(value);
    };

    std::vector<ExpressionEvaluator::Variable> expressionVariables = runtimeResourceExpressionVariables_;
    expressionVariables.reserve(expressionVariables.size() + shaderPack_.attributes.size());
    std::unordered_map<std::string, size_t> variableIndices;
    variableIndices.reserve(expressionVariables.size() + shaderPack_.attributes.size());

    for (size_t i = 0; i < expressionVariables.size(); ++i) {
        variableIndices.emplace(expressionVariables[i].name, i);
    }

    auto appendExpressionVariable = [&](const std::string &name, double value) {
        if (variableIndices.find(name) != variableIndices.end()) { return; }
        variableIndices[name] = expressionVariables.size();
        expressionVariables.push_back({
            .name = name,
            .value = value,
        });
    };

    for (const auto &attribute : shaderPack_.attributes) {
        const std::string &currentValue = [&]() -> const std::string & {
            auto iter = staticAttributeValues_.find(attribute.name);
            return iter == staticAttributeValues_.end() ? attribute.defaultValue : iter->second;
        }();

        if (attribute.define.direct.has_value()) {
            if (attribute.type == "bool") {
                appendExpressionVariable(*attribute.define.direct, parseBoolValue(currentValue) ? 1.0 : 0.0);
            } else if (attribute.type == "int" || attribute.type.rfind("int_range:", 0) == 0 ||
                       attribute.type.rfind("float_range:", 0) == 0 || attribute.type.rfind("enum:", 0) == 0) {
                appendExpressionVariable(*attribute.define.direct, parseNumber(attribute, currentValue));
            }
        }

        for (const auto &[name, expression] : attribute.define.expressions) {
            auto value = tryEvaluateScalarDefineExpression(attribute, currentValue, expression);
            if (value.has_value()) { appendExpressionVariable(name, *value); }
        }
    }

    return ExpressionEvaluator::evaluate(expression, expressionVariables);
}

void ShaderPack::initStageRuntime(StageRuntime &stageRuntime, const ShaderPackLoader::ExecutionConfig &execution) {
    stageRuntime.executionVariableConfigs.clear();
    stageRuntime.globalVariables.clear();

    for (const auto &variable : execution.globalVariables) {
        stageRuntime.globalVariables[variable.name] = variable.defaultValue;
    }
    for (const auto &variable : execution.variables) {
        stageRuntime.executionVariableConfigs[variable.name] = variable;
    }
}

void ShaderPack::initRuntimeTextures() {
    auto framework = framework_.lock();
    auto device = framework->device();
    auto vma = framework->vma();
    uint32_t frameCount = framework->swapchain()->imageCount();

    std::vector<RuntimeTexture> runtimeTextures(shaderPack_.textures.size());
    for (size_t textureIndex = 0; textureIndex < shaderPack_.textures.size(); ++textureIndex) {
        const auto &textureConfig = shaderPack_.textures[textureIndex];
        RuntimeTexture runtimeTexture;
        runtimeTexture.config = textureConfig;
        if (textureConfig.sampledBinding.has_value()) {
            runtimeTexture.sampler = vk::Sampler::create(device, textureConfig.filter, VK_SAMPLER_MIPMAP_MODE_LINEAR,
                                                         textureConfig.addressMode);
        }
        if (textureConfig.imported) {
            runtimeTexture.sampledViewIndex = 0;
        } else {
            uint32_t textureFrameCount = textureConfig.shared ? 1 : frameCount;
            runtimeTexture.frameImages.resize(textureFrameCount);
            for (uint32_t frameIndex = 0; frameIndex < textureFrameCount; frameIndex++) {
                uint32_t width = 0;
                uint32_t height = 0;
                uint32_t depth = 1;
                if (textureConfig.scale > 0.0f) {
                    width = std::max<uint32_t>(1, static_cast<uint32_t>(referenceWidth_ * textureConfig.scale));
                    height = std::max<uint32_t>(1, static_cast<uint32_t>(referenceHeight_ * textureConfig.scale));
                } else if (!textureConfig.widthExpression.empty() && !textureConfig.heightExpression.empty()) {
                    double widthValue = evaluateNumericExpression(textureConfig.widthExpression);
                    double heightValue = evaluateNumericExpression(textureConfig.heightExpression);
                    if (!std::isfinite(widthValue) || !std::isfinite(heightValue) || widthValue <= 0.0 || heightValue <= 0.0) {
                        throw std::runtime_error("invalid runtime texture extent: " + textureConfig.name);
                    }
                    width = std::max<uint32_t>(1, static_cast<uint32_t>(std::ceil(widthValue)));
                    height = std::max<uint32_t>(1, static_cast<uint32_t>(std::ceil(heightValue)));
                    if (textureConfig.dimension == ShaderPackLoader::TextureDimension::Texture3D ||
                        textureConfig.dimension == ShaderPackLoader::TextureDimension::Texture2DArray) {
                        if (textureConfig.depthExpression.empty()) {
                            throw std::runtime_error("runtime layered texture requires depth: " + textureConfig.name);
                        }
                        double depthValue = evaluateNumericExpression(textureConfig.depthExpression);
                        if (!std::isfinite(depthValue) || depthValue <= 0.0) {
                            throw std::runtime_error("invalid runtime texture depth: " + textureConfig.name);
                        }
                        depth = std::max<uint32_t>(1, static_cast<uint32_t>(std::ceil(depthValue)));
                    }
                } else {
                    throw std::runtime_error("runtime texture requires width/height or scale: " + textureConfig.name);
                }

                const bool isCube = textureConfig.dimension == ShaderPackLoader::TextureDimension::Cube;
                const bool is3D = textureConfig.dimension == ShaderPackLoader::TextureDimension::Texture3D;
                const bool is2DArray = textureConfig.dimension == ShaderPackLoader::TextureDimension::Texture2DArray;
                VkImageUsageFlags usage = VK_IMAGE_USAGE_SAMPLED_BIT;
                if (!is3D) { usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT; }
                if (textureConfig.storageBinding.has_value()) { usage |= VK_IMAGE_USAGE_STORAGE_BIT; }

                if (isCube) {
                    runtimeTexture.frameImages[frameIndex] = vk::DeviceLocalImage::create(
                        device, vma, false, width, height, 6, textureConfig.format, usage, 0,
                        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT);

                    for (int faceIndex = 0; faceIndex < 6; faceIndex++) {
                        runtimeTexture.frameImages[frameIndex]->addImageView(VkImageViewCreateInfo{
                            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                            .image = runtimeTexture.frameImages[frameIndex]->vkImage(),
                            .viewType = VK_IMAGE_VIEW_TYPE_2D,
                            .format = runtimeTexture.frameImages[frameIndex]->vkFormat(),
                            .components =
                                {
                                    .r = VK_COMPONENT_SWIZZLE_IDENTITY,
                                    .g = VK_COMPONENT_SWIZZLE_IDENTITY,
                                    .b = VK_COMPONENT_SWIZZLE_IDENTITY,
                                    .a = VK_COMPONENT_SWIZZLE_IDENTITY,
                                },
                            .subresourceRange =
                                {
                                    .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                    .baseMipLevel = 0,
                                    .levelCount = 1,
                                    .baseArrayLayer = static_cast<uint32_t>(faceIndex),
                                    .layerCount = 1,
                                },
                        });
                    }
                    runtimeTexture.frameImages[frameIndex]->addImageView(VkImageViewCreateInfo{
                        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                        .image = runtimeTexture.frameImages[frameIndex]->vkImage(),
                        .viewType = VK_IMAGE_VIEW_TYPE_CUBE,
                        .format = runtimeTexture.frameImages[frameIndex]->vkFormat(),
                        .components =
                            {
                                .r = VK_COMPONENT_SWIZZLE_IDENTITY,
                                .g = VK_COMPONENT_SWIZZLE_IDENTITY,
                                .b = VK_COMPONENT_SWIZZLE_IDENTITY,
                                .a = VK_COMPONENT_SWIZZLE_IDENTITY,
                            },
                        .subresourceRange =
                            {
                                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                .baseMipLevel = 0,
                                .levelCount = 1,
                                .baseArrayLayer = 0,
                                .layerCount = 6,
                            },
                    });
                    runtimeTexture.singleLayerViewBaseIndex = 1;
                    runtimeTexture.sampledViewIndex = 7;
                } else if (is3D) {
                    runtimeTexture.frameImages[frameIndex] =
                        vk::DeviceLocalImage::create3D(device, vma, width, height, depth, textureConfig.format, usage);
                    runtimeTexture.sampledViewIndex = 0;
                } else if (is2DArray) {
                    runtimeTexture.frameImages[frameIndex] =
                        vk::DeviceLocalImage::create(device, vma, false, width, height, depth, textureConfig.format,
                                                    usage);
                    for (uint32_t layerIndex = 0; layerIndex < depth; layerIndex++) {
                        runtimeTexture.frameImages[frameIndex]->addImageView(VkImageViewCreateInfo{
                            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                            .image = runtimeTexture.frameImages[frameIndex]->vkImage(),
                            .viewType = VK_IMAGE_VIEW_TYPE_2D,
                            .format = runtimeTexture.frameImages[frameIndex]->vkFormat(),
                            .components =
                                {
                                    .r = VK_COMPONENT_SWIZZLE_IDENTITY,
                                    .g = VK_COMPONENT_SWIZZLE_IDENTITY,
                                    .b = VK_COMPONENT_SWIZZLE_IDENTITY,
                                    .a = VK_COMPONENT_SWIZZLE_IDENTITY,
                                },
                            .subresourceRange =
                                {
                                    .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                    .baseMipLevel = 0,
                                    .levelCount = 1,
                                    .baseArrayLayer = layerIndex,
                                    .layerCount = 1,
                                },
                        });
                    }
                    runtimeTexture.singleLayerViewBaseIndex = 1;
                    runtimeTexture.sampledViewIndex = 0;
                } else {
                    runtimeTexture.frameImages[frameIndex] =
                        vk::DeviceLocalImage::create(device, vma, false, width, height, 1, textureConfig.format, usage);
                    runtimeTexture.sampledViewIndex = 0;
                }
            }
        }

        runtimeTextures[textureIndex] = std::move(runtimeTexture);
    }

    runtimeTextures_ = std::move(runtimeTextures);
    runtimeTextureIndices_.clear();
    runtimeTextureIndices_.reserve(runtimeTextures_.size());
    for (size_t i = 0; i < runtimeTextures_.size(); i++) {
        runtimeTextureIndices_[runtimeTextures_[i].config.name] = i;
    }
}

void ShaderPack::initRuntimeBuffers() {
    auto framework = framework_.lock();
    auto device = framework->device();
    auto vma = framework->vma();
    uint32_t frameCount = framework->swapchain()->imageCount();

    std::vector<RuntimeBuffer> runtimeBuffers(shaderPack_.buffers.size());
    for (size_t bufferIndex = 0; bufferIndex < shaderPack_.buffers.size(); ++bufferIndex) {
        const auto &bufferConfig = shaderPack_.buffers[bufferIndex];
        RuntimeBuffer runtimeBuffer;
        runtimeBuffer.config = bufferConfig;

        uint32_t bufferFrameCount = bufferConfig.shared ? 1 : frameCount;
        runtimeBuffer.frameBuffers.resize(bufferFrameCount);
        const size_t bufferSize =
            std::max(static_cast<size_t>(1), static_cast<size_t>(std::ceil(evaluateNumericExpression(bufferConfig.sizeExpression))));
        for (uint32_t frameIndex = 0; frameIndex < bufferFrameCount; frameIndex++) {
            runtimeBuffer.frameBuffers[frameIndex] =
                vk::DeviceLocalBuffer::create(vma, device, false, bufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 0,
                                              VMA_MEMORY_USAGE_GPU_ONLY);
        }

        runtimeBuffers[bufferIndex] = std::move(runtimeBuffer);
    }

    runtimeBuffers_ = std::move(runtimeBuffers);
    runtimeBufferIndices_.clear();
    runtimeBufferIndices_.reserve(runtimeBuffers_.size());
    for (size_t i = 0; i < runtimeBuffers_.size(); i++) {
        runtimeBufferIndices_[runtimeBuffers_[i].config.name] = i;
    }
}

void ShaderPack::loadRuntimeResources() {
    auto framework = framework_.lock();
    auto device = framework->device();
    auto vma = framework->vma();
    auto physicalDevice = framework->physicalDevice();

    bool hasImportedTextures = false;
    for (const auto &texture : runtimeTextures_) {
        if (texture.config.imported) {
            hasImportedTextures = true;
            break;
        }
    }
    if (!hasImportedTextures) { return; }

    auto readBinaryTexture = [](const fs::path &path) {
        std::ifstream stream(path, std::ios::binary | std::ios::ate);
        if (!stream.is_open()) {
            throw std::runtime_error("failed to open imported texture: " + path.string());
        }

        std::streamsize size = stream.tellg();
        if (size < 0) {
            throw std::runtime_error("failed to read imported texture size: " + path.string());
        }
        stream.seekg(0, std::ios::beg);

        std::vector<uint8_t> data(static_cast<size_t>(size));
        if (size > 0 && !stream.read(reinterpret_cast<char *>(data.data()), size)) {
            throw std::runtime_error("failed to read imported texture data: " + path.string());
        }
        return data;
    };

    for (size_t textureIndex = 0; textureIndex < runtimeTextures_.size(); ++textureIndex) {
        auto &texture = runtimeTextures_[textureIndex];
        if (!texture.config.imported) { continue; }
        if (texture.config.dimension == ShaderPackLoader::TextureDimension::Texture2D) {
            if (texture.config.format != VK_FORMAT_R8G8B8A8_UNORM) {
                throw std::runtime_error("Imported 2d textures currently require R8G8B8A8_UNORM: " + texture.config.name);
            }

            int width = 0;
            int height = 0;
            int channels = 0;
            stbi_uc *pixels = stbi_load(texture.config.sourcePath.string().c_str(), &width, &height, &channels, 4);
            if (pixels == nullptr || width <= 0 || height <= 0) {
                throw std::runtime_error("failed to load imported texture: " + texture.config.sourcePath.string());
            }
            texture.importedImage = vk::DeviceLocalImage::create(device, vma, true, static_cast<uint32_t>(width),
                                                                 static_cast<uint32_t>(height), 1,
                                                                 texture.config.format, VK_IMAGE_USAGE_SAMPLED_BIT);
            texture.importedImage->uploadToStagingBuffer(pixels);
            stbi_image_free(pixels);
            continue;
        }

        if (texture.config.dimension == ShaderPackLoader::TextureDimension::Texture2DArray) {
            const size_t expectedSize =
                static_cast<size_t>(texture.config.importedWidth) * static_cast<size_t>(texture.config.importedHeight) *
                static_cast<size_t>(texture.config.importedDepth) * vk::formatToByte(texture.config.format);
            std::vector<uint8_t> raw = readBinaryTexture(texture.config.sourcePath);
            if (raw.size() != expectedSize) {
                throw std::runtime_error("imported 2d_array texture size mismatch for " + texture.config.name +
                                         ": expected " + std::to_string(expectedSize) + " bytes but got " +
                                         std::to_string(raw.size()));
            }

            texture.importedImage = vk::DeviceLocalImage::create(device, vma, true, texture.config.importedWidth,
                                                                 texture.config.importedHeight, texture.config.importedDepth,
                                                                 texture.config.format, VK_IMAGE_USAGE_SAMPLED_BIT);
            texture.importedImage->uploadToStagingBuffer(raw.data());
            continue;
        }

        if (texture.config.dimension != ShaderPackLoader::TextureDimension::Texture3D) {
            throw std::runtime_error("unsupported imported texture dimension: " + texture.config.name);
        }

        const size_t expectedSize =
            static_cast<size_t>(texture.config.importedWidth) * static_cast<size_t>(texture.config.importedHeight) *
            static_cast<size_t>(texture.config.importedDepth) * vk::formatToByte(texture.config.format);
        std::vector<uint8_t> raw = readBinaryTexture(texture.config.sourcePath);
        if (raw.size() != expectedSize) {
            throw std::runtime_error("imported 3d texture size mismatch for " + texture.config.name + ": expected " +
                                     std::to_string(expectedSize) + " bytes but got " + std::to_string(raw.size()));
        }

        texture.importedImage = vk::DeviceLocalImage::create3D(device, vma, texture.config.importedWidth,
                                                               texture.config.importedHeight, texture.config.importedDepth,
                                                               texture.config.format, VK_IMAGE_USAGE_SAMPLED_BIT, true);
        texture.importedImage->uploadToStagingBuffer(raw.data());
    }

    auto commandPool = vk::CommandPool::create(physicalDevice, device);
    auto commandBuffer = vk::CommandBuffer::create(device, commandPool);
    commandBuffer->begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

    bool hasUploads = false;
    for (auto &texture : runtimeTextures_) {
        if (!texture.config.imported) { continue; }

        commandBuffer->barriersBufferImage({}, {{
                                                   .srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                                                   .srcAccessMask = 0,
                                                   .dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                                   .dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                                   .oldLayout = texture.importedImage->imageLayout(),
                                                   .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                                   .srcQueueFamilyIndex = physicalDevice->mainQueueIndex(),
                                                   .dstQueueFamilyIndex = physicalDevice->mainQueueIndex(),
                                                   .image = texture.importedImage,
                                                   .subresourceRange = vk::wholeColorSubresourceRange,
                                               }});
        texture.importedImage->imageLayout() = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        texture.importedImage->uploadToImage(commandBuffer);
        commandBuffer->barriersBufferImage({}, {{
                                                   .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                                   .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                                   .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
                                                                   VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR |
                                                                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                                                   .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
                                                   .oldLayout = texture.importedImage->imageLayout(),
                                                   .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                                   .srcQueueFamilyIndex = physicalDevice->mainQueueIndex(),
                                                   .dstQueueFamilyIndex = physicalDevice->mainQueueIndex(),
                                                   .image = texture.importedImage,
                                                   .subresourceRange = vk::wholeColorSubresourceRange,
                                               }});
        texture.importedImage->imageLayout() = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        hasUploads = true;
    }

    commandBuffer->end();
    if (hasUploads) {
        commandBuffer->submitMainQueueIndividual(device);
        vkQueueWaitIdle(device->mainVkQueue());
    }
}
