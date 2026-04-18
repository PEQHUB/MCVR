#include "config_service.hpp"
#include "diagnostics/log.hpp"

#include <fstream>
#include <sstream>

namespace engine {

ConfigService::ConfigService() {
    // Start with defaults from the generated struct
    live_ = EngineConfig{};
    currentSnapshot_ = std::make_shared<const ConfigSnapshot>(ConfigSnapshot{live_});
}

ConfigService::~ConfigService() = default;

bool ConfigService::load(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        if (log::isInitialized()) {
            log::warn("config", std::string("Could not open config file: ") + path);
        }
        return false;
    }

    // Parse Java-style properties: key=value, # comments, blank lines
    std::string line;
    std::unordered_map<std::string, std::string> props;
    while (std::getline(file, line)) {
        // Strip \r from Windows line endings
        if (!line.empty() && line.back() == '\r') line.pop_back();
        // Skip comments and blanks
        if (line.empty() || line[0] == '#' || line[0] == '!') continue;
        auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string key = line.substr(0, eq);
        std::string val = line.substr(eq + 1);
        props[key] = val;
    }

    // Apply known keys to live config.
    // This is intentionally verbose per-field rather than reflection-based,
    // matching the generated struct field names to property keys.
    auto getBool = [&](const std::string& k, bool& field) {
        auto it = props.find(k);
        if (it != props.end()) field = (it->second == "true");
    };
    auto getUint = [&](const std::string& k, uint32_t& field) {
        auto it = props.find(k);
        if (it != props.end()) {
            try { field = static_cast<uint32_t>(std::stoul(it->second)); } catch (...) {}
        }
    };
    auto getFloat = [&](const std::string& k, float& field) {
        auto it = props.find(k);
        if (it != props.end()) {
            try { field = std::stof(it->second); } catch (...) {}
        }
    };
    auto getInt = [&](const std::string& k, int32_t& field) {
        auto it = props.find(k);
        if (it != props.end()) {
            try { field = std::stoi(it->second); } catch (...) {}
        }
    };

    // display
    getBool("vsync", live_.vsync);
    getUint("maxFps", live_.maxFps);
    getUint("inactivityFpsLimit", live_.inactivityFpsLimit);
    getBool("outputScale2x", live_.outputScale2x);

    // upscaler
    getUint("upscalerMode", live_.upscalerMode);
    getUint("upscalerQuality", live_.upscalerQuality);
    getUint("upscalerResOverride", live_.upscalerResOverride);
    getUint("dlssQuality", live_.dlssQuality);

    // rayTracing
    getUint("rayBounces", live_.rayBounces);
    getBool("simplifiedIndirect", live_.simplifiedIndirect);
    getBool("serEnabled", live_.serEnabled);
    getBool("serHintsEnabled", live_.serHintsEnabled);
    getBool("noiseLOD", live_.noiseLOD);
    getBool("multiScatterGGX", live_.multiScatterGGX);
    getBool("eonDiffuse", live_.eonDiffuse);
    getBool("greedyMeshingEnabled", live_.greedyMeshingEnabled);
    getBool("ommEnabled", live_.ommEnabled);
    getUint("ommBakerLevel", live_.ommBakerLevel);
    getBool("areaLightsEnabled", live_.areaLightsEnabled);
    getFloat("shadowSoftness", live_.shadowSoftness);
    getBool("restirEnabled", live_.restirEnabled);
    getBool("restirBounceEnabled", live_.restirBounceEnabled);
    getBool("restirSimplifiedBRDF", live_.restirSimplifiedBRDF);
    getUint("restirCandidates", live_.restirCandidates);
    getUint("restirTemporalMClamp", live_.restirTemporalMClamp);
    getUint("restirWClamp", live_.restirWClamp);
    getBool("pomEnabled", live_.pomEnabled);
    getFloat("pomHeightScale", live_.pomHeightScale);
    getUint("pomSteps", live_.pomSteps);
    getUint("pomRefinement", live_.pomRefinement);
    getFloat("pomFadeDistance", live_.pomFadeDistance);
    getBool("sharcEnabled", live_.sharcEnabled);
    getFloat("sharcSceneScale", live_.sharcSceneScale);
    getFloat("sharcRoughnessThreshold", live_.sharcRoughnessThreshold);
    getUint("restirSpatialTaps", live_.restirSpatialTaps);
    getUint("restirSpatialRadius", live_.restirSpatialRadius);

    // exposure (Java stores tenths/percent — but v2 config stores native floats)
    getFloat("exposureCompensation", live_.exposureCompensation);
    getBool("manualExposureEnabled", live_.manualExposureEnabled);
    getFloat("manualExposure", live_.manualExposure);
    getFloat("brightAdaptSpeed", live_.brightAdaptSpeed);
    getFloat("darkAdaptSpeed", live_.darkAdaptSpeed);
    getFloat("sceneChangeThreshold", live_.sceneChangeThreshold);
    getFloat("centerWeightStrength", live_.centerWeightStrength);
    getFloat("highlightWeight", live_.highlightWeight);

    // toneMapping
    getUint("tonemappingMode", live_.tonemappingMode);
    getBool("psychoEnabled", live_.psychoEnabled);
    getFloat("saturation", live_.saturation);
    getBool("saturationAdaptive", live_.saturationAdaptive);
    getFloat("Lwhite", live_.Lwhite);
    getFloat("colorExpansion", live_.colorExpansion);
    getUint("sharpenerMode", live_.sharpenerMode);
    getFloat("casSharpness", live_.casSharpness);
    getFloat("psychoPeakSDR", live_.psychoPeakSDR);

    // bloom
    getBool("bloomEnabled", live_.bloomEnabled);
    getFloat("bloomIntensity", live_.bloomIntensity);
    getFloat("bloomThreshold", live_.bloomThreshold);

    // chunks
    getUint("chunkBuildingBatchSize", live_.chunkBuildingBatchSize);
    getUint("chunkBuildingTotalBatches", live_.chunkBuildingTotalBatches);
    getFloat("chunkCullDistance", live_.chunkCullDistance);
    getFloat("chunkLodDistance", live_.chunkLodDistance);

    // debug
    getBool("gpuDiagnostics", live_.gpuDiagnostics);
    getBool("validationLayers", live_.validationLayers);

    // frameGen
    getBool("frameGenEnabled", live_.frameGenEnabled);
    getUint("frameGenMultiplier", live_.frameGenMultiplier);

    // reflex
    getBool("reflexEnabled", live_.reflexEnabled);
    getBool("reflexBoost", live_.reflexBoost);
    getUint("maxFpsLimit", live_.maxFpsLimit);

    // clouds
    getUint("cloudQuality", live_.cloudQuality);
    getFloat("cloudDensity", live_.cloudDensity);
    getFloat("cloudCoverage", live_.cloudCoverage);
    getFloat("cloudType", live_.cloudType);
    getFloat("cloudSpeed", live_.cloudSpeed);
    getFloat("cloudAltitude", live_.cloudAltitude);
    getFloat("cloudThickness", live_.cloudThickness);
    getFloat("cloudDetailStrength", live_.cloudDetailStrength);
    getUint("cloudScatterOctaves", live_.cloudScatterOctaves);
    getFloat("cloudAmbientStrength", live_.cloudAmbientStrength);
    getFloat("cloudTemporalBlend", live_.cloudTemporalBlend);
    getFloat("cloudNoiseScale", live_.cloudNoiseScale);
    getFloat("cloudCellFrequency", live_.cloudCellFrequency);
    getFloat("cloudAtmosphereFadeDist", live_.cloudAtmosphereFadeDist);
    getUint("cloudDebugMode", live_.cloudDebugMode);
    getFloat("cloudWindAngle", live_.cloudWindAngle);

    // offlineAccum
    getBool("offlineAccumEnabled", live_.offlineAccumEnabled);
    getUint("offlineBounces", live_.offlineBounces);
    getBool("offlineDenoised", live_.offlineDenoised);
    getFloat("offlineAperture", live_.offlineAperture);
    getFloat("offlineFocalDistance", live_.offlineFocalDistance);
    getUint("offlineDlssEpochLength", live_.offlineDlssEpochLength);

    // displacement
    getUint("displacementQuality", live_.displacementQuality);
    getUint("tessMaxLevel", live_.tessMaxLevel);
    getFloat("tessNearDist", live_.tessNearDist);
    getFloat("tessMidDist", live_.tessMidDist);
    getFloat("tessFarDist", live_.tessFarDist);

    validate();

    if (log::isInitialized()) {
        log::info("config", std::string("Loaded ") + std::to_string(props.size()) +
                  " properties from " + path);
    }
    return true;
}

bool ConfigService::save(const std::string& path) const {
    std::ofstream file(path);
    if (!file.is_open()) {
        if (log::isInitialized()) {
            log::error("config", std::string("Could not write config file: ") + path);
        }
        return false;
    }

    file << "# RadSER Engine Configuration (v2)\n";

    auto putBool = [&](const char* k, bool v) { file << k << '=' << (v ? "true" : "false") << '\n'; };
    auto putUint = [&](const char* k, uint32_t v) { file << k << '=' << v << '\n'; };
    auto putFloat = [&](const char* k, float v) { file << k << '=' << v << '\n'; };
    auto putInt = [&](const char* k, int32_t v) { file << k << '=' << v << '\n'; };

    putBool("vsync", live_.vsync);
    putUint("maxFps", live_.maxFps);
    putUint("inactivityFpsLimit", live_.inactivityFpsLimit);
    putBool("outputScale2x", live_.outputScale2x);
    putUint("upscalerMode", live_.upscalerMode);
    putUint("upscalerQuality", live_.upscalerQuality);
    putUint("upscalerResOverride", live_.upscalerResOverride);
    putUint("dlssQuality", live_.dlssQuality);
    putUint("rayBounces", live_.rayBounces);
    putBool("simplifiedIndirect", live_.simplifiedIndirect);
    putBool("serEnabled", live_.serEnabled);
    putBool("serHintsEnabled", live_.serHintsEnabled);
    putBool("noiseLOD", live_.noiseLOD);
    putBool("multiScatterGGX", live_.multiScatterGGX);
    putBool("eonDiffuse", live_.eonDiffuse);
    putBool("greedyMeshingEnabled", live_.greedyMeshingEnabled);
    putBool("ommEnabled", live_.ommEnabled);
    putUint("ommBakerLevel", live_.ommBakerLevel);
    putBool("areaLightsEnabled", live_.areaLightsEnabled);
    putFloat("shadowSoftness", live_.shadowSoftness);
    putBool("restirEnabled", live_.restirEnabled);
    putBool("restirBounceEnabled", live_.restirBounceEnabled);
    putBool("restirSimplifiedBRDF", live_.restirSimplifiedBRDF);
    putUint("restirCandidates", live_.restirCandidates);
    putUint("restirTemporalMClamp", live_.restirTemporalMClamp);
    putUint("restirWClamp", live_.restirWClamp);
    putBool("pomEnabled", live_.pomEnabled);
    putFloat("pomHeightScale", live_.pomHeightScale);
    putUint("pomSteps", live_.pomSteps);
    putUint("pomRefinement", live_.pomRefinement);
    putFloat("pomFadeDistance", live_.pomFadeDistance);
    putBool("sharcEnabled", live_.sharcEnabled);
    putFloat("sharcSceneScale", live_.sharcSceneScale);
    putFloat("sharcRoughnessThreshold", live_.sharcRoughnessThreshold);
    putUint("restirSpatialTaps", live_.restirSpatialTaps);
    putUint("restirSpatialRadius", live_.restirSpatialRadius);
    putFloat("exposureCompensation", live_.exposureCompensation);
    putBool("manualExposureEnabled", live_.manualExposureEnabled);
    putFloat("manualExposure", live_.manualExposure);
    putFloat("brightAdaptSpeed", live_.brightAdaptSpeed);
    putFloat("darkAdaptSpeed", live_.darkAdaptSpeed);
    putFloat("sceneChangeThreshold", live_.sceneChangeThreshold);
    putFloat("centerWeightStrength", live_.centerWeightStrength);
    putFloat("highlightWeight", live_.highlightWeight);
    putUint("tonemappingMode", live_.tonemappingMode);
    putBool("psychoEnabled", live_.psychoEnabled);
    putFloat("saturation", live_.saturation);
    putBool("saturationAdaptive", live_.saturationAdaptive);
    putFloat("Lwhite", live_.Lwhite);
    putFloat("colorExpansion", live_.colorExpansion);
    putUint("sharpenerMode", live_.sharpenerMode);
    putFloat("casSharpness", live_.casSharpness);
    putFloat("psychoPeakSDR", live_.psychoPeakSDR);
    putBool("bloomEnabled", live_.bloomEnabled);
    putFloat("bloomIntensity", live_.bloomIntensity);
    putFloat("bloomThreshold", live_.bloomThreshold);
    putUint("chunkBuildingBatchSize", live_.chunkBuildingBatchSize);
    putUint("chunkBuildingTotalBatches", live_.chunkBuildingTotalBatches);
    putFloat("chunkCullDistance", live_.chunkCullDistance);
    putFloat("chunkLodDistance", live_.chunkLodDistance);
    putBool("gpuDiagnostics", live_.gpuDiagnostics);
    putBool("validationLayers", live_.validationLayers);
    putBool("frameGenEnabled", live_.frameGenEnabled);
    putUint("frameGenMultiplier", live_.frameGenMultiplier);
    putBool("reflexEnabled", live_.reflexEnabled);
    putBool("reflexBoost", live_.reflexBoost);
    putUint("maxFpsLimit", live_.maxFpsLimit);
    // clouds
    putUint("cloudQuality", live_.cloudQuality);
    putFloat("cloudDensity", live_.cloudDensity);
    putFloat("cloudCoverage", live_.cloudCoverage);
    putFloat("cloudType", live_.cloudType);
    putFloat("cloudSpeed", live_.cloudSpeed);
    putFloat("cloudAltitude", live_.cloudAltitude);
    putFloat("cloudThickness", live_.cloudThickness);
    putFloat("cloudDetailStrength", live_.cloudDetailStrength);
    putUint("cloudScatterOctaves", live_.cloudScatterOctaves);
    putFloat("cloudAmbientStrength", live_.cloudAmbientStrength);
    putFloat("cloudTemporalBlend", live_.cloudTemporalBlend);
    putFloat("cloudNoiseScale", live_.cloudNoiseScale);
    putFloat("cloudCellFrequency", live_.cloudCellFrequency);
    putFloat("cloudAtmosphereFadeDist", live_.cloudAtmosphereFadeDist);
    putUint("cloudDebugMode", live_.cloudDebugMode);
    putFloat("cloudWindAngle", live_.cloudWindAngle);
    // offlineAccum
    putBool("offlineAccumEnabled", live_.offlineAccumEnabled);
    putUint("offlineBounces", live_.offlineBounces);
    putBool("offlineDenoised", live_.offlineDenoised);
    putFloat("offlineAperture", live_.offlineAperture);
    putFloat("offlineFocalDistance", live_.offlineFocalDistance);
    putUint("offlineDlssEpochLength", live_.offlineDlssEpochLength);
    // displacement
    putUint("displacementQuality", live_.displacementQuality);
    putUint("tessMaxLevel", live_.tessMaxLevel);
    putFloat("tessNearDist", live_.tessNearDist);
    putFloat("tessMidDist", live_.tessMidDist);
    putFloat("tessFarDist", live_.tessFarDist);

    return true;
}

std::shared_ptr<const ConfigSnapshot> ConfigService::snapshot() {
    std::lock_guard lock(snapshotMutex_);
    currentSnapshot_ = std::make_shared<const ConfigSnapshot>(ConfigSnapshot{live_});
    return currentSnapshot_;
}

ConfigSubscriptionId ConfigService::subscribe(ConfigKey key, ConfigCallback callback) {
    std::lock_guard lock(subMutex_);
    auto id = nextSubId_++;
    subscriptions_.push_back({id, key, std::move(callback)});
    return id;
}

void ConfigService::unsubscribe(ConfigSubscriptionId id) {
    std::lock_guard lock(subMutex_);
    subscriptions_.erase(
        std::remove_if(subscriptions_.begin(), subscriptions_.end(),
            [id](const auto& s) { return s.id == id; }),
        subscriptions_.end());
}

void ConfigService::notifyChange(ConfigKey key) {
    // Copy matching callbacks under lock, invoke outside — prevents deadlock
    // if a subscriber calls subscribe/unsubscribe during the callback.
    std::vector<ConfigCallback> toCall;
    {
        std::lock_guard lock(subMutex_);
        for (const auto& sub : subscriptions_) {
            if (sub.key == key) {
                toCall.push_back(sub.callback);
            }
        }
    }
    for (const auto& cb : toCall) {
        cb(key);
    }
}

void ConfigService::validate() {
    validateConfig(live_);
}

} // namespace engine
