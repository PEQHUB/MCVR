#ifndef SHARED_HPP
#define SHARED_HPP

#include "common/mapping.hpp"

#define INF_DISTANCE 65504.0
#define PI 3.14159265358979323
#define INV_PI 0.31830988618379067
#define TWO_PI 6.28318530717958648
#define INV_TWO_PI 0.15915494309189533
#define INV_4_PI 0.07957747154594766

#ifdef __cplusplus
namespace vk {
#endif
#ifdef __cplusplus
namespace VertexFormat {
#endif
    struct Triangle {
        T_VEC3 pos;
        T_VEC3 color;
    };

    struct TexturedTriangle {
        T_VEC3 pos;
        T_VEC2 uv;
    };

    struct ArrayTexturedTriangle {
        T_VEC3 pos;
        T_FLOAT metallic;
        T_VEC3 norm;
        T_FLOAT roughness;
        T_VEC2 uv;
        T_FLOAT textureLayer;
        T_FLOAT pad0;
        T_VEC3 color;
        T_FLOAT intensity;
    };

    struct PositionOnly {
        T_VEC3 position;
    };

    struct PositionTexColor {
        T_VEC3 position;
        T_VEC2 uv;
        T_UINT color;
    };

    struct PositionColor {
        T_VEC3 position;
        T_UINT color;
    };

    struct PositionColorNormal {
        T_VEC3 position;
        T_UINT color;
        T_UINT normal; // first 3 bytes
    };

    struct PositionTex {
        T_VEC3 position;
        T_VEC2 uv;
    };

    struct PositionColorTexLight {
        T_VEC3 position;
        T_UINT color;
        T_VEC2 uv0;
        T_UINT uv2;
    };

    struct PositionColorLight {
        T_VEC3 position;
        T_UINT color;
        T_UINT uv2;
    };

    struct PositionTexColorLight {
        T_VEC3 position;
        T_VEC2 uv0;
        T_UINT color;
        T_UINT uv2;
    };

    struct PositionTexColorNormal {
        T_VEC3 position;
        T_VEC2 uv0;
        T_UINT color;
        T_UINT normal; // first 3 bytes
    };

    struct PositionTexLightColor {
        T_VEC3 position;
        T_VEC2 uv0;
        T_UINT uv2;
        T_UINT color;
    };

    struct PositionColorTexLightNormal {
        T_VEC3 position;
        T_UINT color;
        T_VEC2 uv0;    // texture
        T_UINT uv2;    // lightmap
        T_UINT normal; // first 3 bytes
    };

    struct PositionColorTexOverlayLightNormal {
        T_VEC3 position;
        T_UINT color;
        T_VEC2 uv0;    // texture
        T_UINT uv1;    // overlay
        T_UINT uv2;    // lightmap
        T_UINT normal; // first 3 bytes
    };

    struct PBRTriangle {
        T_VEC3 pos;
        T_UINT useNorm;

        T_VEC3 norm;
        T_UINT useColorLayer;

        T_VEC4 colorLayer;

        T_UINT useTexture;
        T_UINT useOverlay;
        T_VEC2 textureUV;

        T_IVEC2 overlayUV;
        T_UINT useGlint;
        T_UINT textureID;

        T_VEC2 glintUV;
        T_UINT glintTexture;
        T_UINT useLight;

        T_IVEC2 lightUV;
        T_UINT coordinate;
        T_FLOAT albedoEmission;

        T_VEC3 postBase;
        T_UINT emissiveBlockType; // EmissiveBlock ordinal (0-39), 255 = none/LabPBR
    };
#ifdef __cplusplus
}; // namespace VertexFormat
#endif

#ifdef __cplusplus
namespace Data {
#endif
    struct Camera {
        T_MAT4 viewMatrix;
        T_MAT4 projMatrix;
        T_MAT4 viewMatrixInv;
        T_MAT4 projMatrixInv;
        T_VEC2 jitter;
        T_VEC2 pad0;
    };

    struct DirectionalLight {
        T_VEC3 direction;  // 12 bytes
        T_FLOAT pad0;      // 4 bytes
        T_VEC3 color;      // 12 bytes
        T_FLOAT intensity; // 4 bytes
    };

    struct World {
        DirectionalLight directionalLight; // 32 bytes
        T_FLOAT time;                      // 4 bytes
        T_UINT seed;                       // 4 bytes
    };

    struct OverlayUBO {
        T_UINT texIndices[12];

        T_MAT4 modelViewMat;

        T_MAT4 projectionMat;

        T_VEC4 colorModulator;

        T_FLOAT glintAlpha;
        T_FLOAT fogStart;
        T_FLOAT fogEnd;
        T_UINT fogShape;

        T_VEC4 fogColor;

        T_MAT4 textureMat;

        T_FLOAT gameTime;
        T_FLOAT lineWidth;
        T_VEC2 screenSize;

        T_VEC3 light0Direction;
        T_FLOAT pad0;

        T_VEC3 light1Direction;
        T_FLOAT pad1;
    };

    struct OverlayPostUBO {
        T_MAT4 projectionMat;
        T_VEC2 inSize;
        T_VEC2 outSize;
        T_VEC2 blurDir;
        T_FLOAT radius;
        T_FLOAT radiusMultiplier;
    };

    struct WorldUBO {
        T_MAT4 cameraViewMat;

        T_MAT4 cameraEffectedViewMat;

        T_MAT4 cameraProjMat;

        T_MAT4 cameraViewMatInv;

        T_MAT4 cameraEffectedViewMatInv;

        T_MAT4 cameraProjMatInv;

        T_VEC2 cameraJitter;
        T_FLOAT gameTime;
        T_UINT seed;

        T_MAT4 textureMat;

        T_UINT overlayTextureID;
        T_UINT isFirstPerson;
        T_FLOAT fogStart;
        T_FLOAT fogEnd;

        T_VEC4 fogColor;

        T_UINT fogType;
        T_UINT skyType;
        T_UINT rayBounces;
        T_FLOAT pad3;

        T_DVEC4 cameraPos; // w for padding

        T_UINT endSkyTextureID;
        T_UINT endPortalTextureID;
        T_UINT pad4;
        T_UINT pad5;

        T_VEC4 emissionData[50]; // Per-block: .rgb = BT.2020 color override (0,0,0 = use texture), .a = scalar multiplier

        T_VEC4 emissiveGamut[13]; // Per-emissive-block gamut boost, indexed as [i/4][i%4], 1.0 = neutral

    };

    struct SkyUBO {
        T_VEC3 baseColor;
        T_UINT skyType;

        T_VEC4 horizonColor;

        T_VEC3 sunDirection;
        T_UINT isSunRisingOrSetting;

        T_VEC3 moonDirection;
        T_FLOAT moonDirPad;

        T_UINT isSkyDark;
        T_UINT hasBlindnessOrDarkness;
        T_UINT cameraSubmersionType;
        T_UINT moonPhase;

        T_FLOAT rainGradient;
        T_FLOAT hdrRadianceScale;
        T_FLOAT pad1;
        T_FLOAT pad2;

        // AtmosphereParams

        T_FLOAT Rg;
        T_FLOAT Rt;
        T_FLOAT Hr;
        T_FLOAT Hm;

        T_VEC3 betaR;
        T_FLOAT mieG;

        T_VEC3 betaM;
        T_FLOAT minViewCos;

        T_VEC3 sunRadiance;
        T_UINT sunTextureID;

        T_VEC3 moonRadiance;
        T_UINT moonTextureID;

        T_VEC4 envCelestial;
        T_VEC4 envWaterTintFog;
        T_VEC4 envSky;

        // Clouds (procedural volumetric slab)
        // x: base height (world Y)
        // y: thickness
        // z: density scale (extinction multiplier)
        // w: albedo/brightness scale
        T_VEC4 envCloud;

        // Volumetric clouds driven by vanilla Fancy CloudCells
        // cloudTile.x: textures[] index for the 65x65 R8 mask
        // cloudTile.y: center cell X (j)
        // cloudTile.z: center cell Z (k)
        // cloudTile.w: reserved
        T_IVEC4 cloudTile;

        // cloudWrap.x: intra-cell offset X (vanilla l, in blocks [0, 12))
        // cloudWrap.y: intra-cell offset Z (vanilla m, in blocks [0, 12))
        // cloudWrap.z: ticks (vanilla ticks + tickDelta)
        // cloudWrap.w: reserved
        T_VEC4 cloudWrap;

        // cloudShape.x: puffiness (edge softness / roundness)
        // cloudShape.y: detailScale
        // cloudShape.z: detailStrength
        // cloudShape.w: anisotropy (HG g)
        T_VEC4 cloudShape;

        // cloudLighting.x: shadowStrength
        // cloudLighting.y: ambientStrength
        // cloudLighting.z: sunOcclusionStrength
        // cloudLighting.w: noiseAffectsShadows (0=off, 1=on)
        T_VEC4 cloudLighting;
    };

    // Hot path: 20 bytes per entry, read by every ray hit
    struct TextureMapEntry {
        T_INT specular;
        T_INT normal;
        T_INT flag;
        T_INT properties;    // bit 0: has height map
        T_INT maskTexture;   // bindless index of R8_UNORM material class mask, -1 = none
    };

#ifdef __cplusplus
    static constexpr int TEX_PROP_HAS_HEIGHT_MAP = 1;
#else
    #define TEX_PROP_HAS_HEIGHT_MAP 1
#endif

    struct TextureMapping {
        TextureMapEntry entries[4096];
    };

    // Unified material class: full Disney BRDF parameters for a material category.
    // ~32 classes (IRON, GOLD, DIAMOND, WOOD, GLASS, etc.), 128 bytes each.
    // Indexed by material class ID from the material mask atlas.
    struct MaterialClassEntry {
        // Pack 0: base BRDF
        T_VEC3 f0;              // Fresnel reflectance at normal incidence (RGB)
        T_FLOAT roughness;      // Perceptual roughness [0,1] (squared in shader for GGX alpha)

        // Pack 1: extended BRDF
        T_FLOAT metallic;       // [0,1]
        T_FLOAT transmission;   // [0,1], -1.0 = keep LabPBR value
        T_FLOAT ior;            // Index of refraction (>= 1.0)
        T_FLOAT subsurface;     // [0,1]

        // Pack 2: Disney extended
        T_FLOAT anisotropic;    // [0,1]
        T_FLOAT sheenWeight;    // [0,1]
        T_FLOAT sheenTint;      // [0,1]
        T_FLOAT coatWeight;     // [0,1]

        // Pack 3: coat + noise base
        T_FLOAT coatRoughness;  // [0,1]
        T_FLOAT noiseScale;     // World-space noise scale
        T_FLOAT noiseStrength;  // [0,1]
        T_UINT  noisePacked;    // Bit-packed: octaves(0-3), type(4-8), seed(9-17), target(20-23)

        // Pack 4: POM + height field + normal controls (16 bytes)
        T_UINT  pomPacked0;   // filterMode(3) | pomMode(2) | heightSource(3) | pomSteps(8) | pomRefinement(4) | filterRadius(4) | mipBias(4) | flags(4)
        T_UINT  pomPacked1;   // normalClamp(8) | geometricBlend(8) | pomAOStrength(8) | heightContrast(8)
        T_UINT  pomPacked2;   // heightRemapMin(8) | heightRemapMax(8) | heightOffset(8) | normalDistanceFade(8)
        T_FLOAT pomDepth;     // [0.00-0.50] per-block POM depth in blocks (0 = disabled)

        // Pack 5: gamut + noise mask + normal strength
        T_FLOAT gamutBoost;     // Oklab chroma scale (1.0 = neutral)
        T_FLOAT noiseMaskThreshold; // [0,1]
        T_UINT  noiseMaskPacked;    // Bit-packed mask params
        T_FLOAT normalStrength; // Normal map intensity (1.0 = neutral)

        // Pack 6: noise transform
        T_FLOAT noiseRotation;  // Radians
        T_FLOAT noiseAspect;    // Y/X ratio
        T_FLOAT noiseLacunarity;// FBM lacunarity
        T_FLOAT noiseContrast;  // S-curve contrast

        // Pack 7: emission + classification
        T_FLOAT emissionNits;   // Surface luminance in nits (0 = no emission)
        T_UINT  emissionType;   // EmissiveBlock ordinal (255 = none)
        T_UINT  classId;        // Material class ID (for cross-reference)
        T_UINT  flags;          // Bit 0: has override, Bit 1: area light, Bit 2: vivid color,
                                // Bit 3: AutoPBR (shader-side roughness+normal from albedo),
                                // Bit 4: invertRoughness, Bit 5: invertNormal, Bit 6: invertHeight

        // Pack 8: AutoPBR shader-side params (GPU-computed roughness + normal from albedo)
        T_FLOAT lumMin;         // Precomputed per-block min luminance [0,1] (linear)
        T_FLOAT lumMax;         // Precomputed per-block max luminance [0,1] (linear)
        T_UINT  autoPBRPacked0; // rMin_u8 | rMax_u8<<8 | center_u8<<16 | spread_u8<<24
        T_UINT  autoPBRPacked1; // normalStrength_u16 | heightGamma_u16<<16
    }; // 144 bytes (9 x vec4), std430 aligned

#ifdef __cplusplus
    static constexpr int MAX_MATERIAL_CLASSES = 512;
#else
    #define MAX_MATERIAL_CLASSES 512
#endif

    struct MaterialClassMapping {
        MaterialClassEntry entries[MAX_MATERIAL_CLASSES];
    }; // 72 KB (512 × 144 bytes)

    struct ExposureData {
        T_INT width;
        T_INT height;
        T_INT stride;
        T_FLOAT exposure;

        T_FLOAT minL;
        T_FLOAT maxL;
        T_UINT total;
        T_UINT pad0;

        T_UINT bins[256];
    };

    struct LightMapUBO {
        T_FLOAT ambientLightFactor;
        T_FLOAT skyFactor;
        T_FLOAT blockFactor;
        T_INT useBrightLightmap;

        T_VEC3 skyLightColor;
        T_FLOAT nightVisionFactor;

        T_FLOAT darknessScale;
        T_FLOAT darkenWorldFactor;
        T_FLOAT brightnessFactor;
        T_FLOAT pad0;
    };

    struct AreaLight {
        T_VEC3 position;     // camera-relative world position
        T_FLOAT halfExtent;  // cube half-size (0.5=full block, 0.05=point-like)
        T_VEC3 color;        // pre-computed emissive RGB
        T_FLOAT intensity;   // brightness scale
        T_VEC3 _unused;      // available for future use
        T_FLOAT radius;      // max range in blocks
    }; // 48 bytes, std430 aligned (3 x vec4)
#ifdef __cplusplus
}; // namespace Data
#endif
#ifdef __cplusplus
}; // namespace vk
#endif

#endif
