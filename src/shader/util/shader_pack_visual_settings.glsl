#ifndef RARSER_SHADER_PACK_VISUAL_SETTINGS_GLSL
#define RARSER_SHADER_PACK_VISUAL_SETTINGS_GLSL

layout(set = 4, binding = 0, std430) readonly buffer ShaderPackVisualSettingsBlock {
    int cloudMode;
    int captureVolumetricCloudIndirect;
    int volumetricCloudTemporalAccumulation;
    int volumetricCloudCastShadow;
    int waterSurfaceMode;
    int waterCausticsEnabled;
    int volumetricLightMode;
    int volumetricCloudClearAmount;

    int indirectVolumetricCloudViewSteps;
    int indirectVolumetricCloudLightSteps;
    int indirectVolumetricCloudAmbientSteps;
    int volumetricCloudViewSteps;
    int volumetricCloudLightSteps;
    int volumetricCloudAmbientSteps;
    int volumetricLightSamples;
    int reservedVisual0;

    float indirectVolumetricCloudReflectionMaxRoughness;
    float volumetricCloudBottomHeight;
    float volumetricCloudTopHeight;
    float volumetricCloudBaseScale;
    float volumetricCloudDetailScale;
    float volumetricCloudCoverage;
    float volumetricCloudDensity;
    float volumetricCloudShadowSoftness;

    float volumetricCloudAmbientStrength;
    float volumetricCloudPowderStrength;
    float volumetricCloudWeatherScale;
    float volumetricLightScatteringStrength;
    float volumetricLightMaxDistance;
    float volumetricLightNearStepSize;
    float volumetricLightFarStepSize;
    float volumetricLightLuminanceLimit;
} shaderPackVisualSettings;

#endif
