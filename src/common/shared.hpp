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

    // PBRTriangle flags bitfield (bits 0-5: boolean enables, bits 8-10: coordinate)
#ifdef __cplusplus
    static constexpr uint32_t PBR_FLAG_USE_NORM        = 1u << 0;
    static constexpr uint32_t PBR_FLAG_USE_COLOR_LAYER  = 1u << 1;
    static constexpr uint32_t PBR_FLAG_USE_TEXTURE      = 1u << 2;
    static constexpr uint32_t PBR_FLAG_USE_OVERLAY      = 1u << 3;
    static constexpr uint32_t PBR_FLAG_USE_GLINT        = 1u << 4;
    static constexpr uint32_t PBR_FLAG_USE_LIGHT        = 1u << 5;
    static constexpr uint32_t PBR_FLAG_GREEDY_MERGED    = 1u << 6; // UV tiling needed (greedy-merged quad)
    static constexpr uint32_t PBR_FLAG_OVERLAY_ALPHA_MASK = 1u << 7; // colorLayer holds overlay sprite bounds for alpha-masked biome tinting
    static constexpr uint32_t PBR_FLAG_COORD_SHIFT      = 8u;
    static constexpr uint32_t PBR_FLAG_COORD_MASK       = 0x7u << 8u; // 3 bits
    // Compact format: vivid flag relocated from emissiveBlockType bit 16 to flags bit 11
    static constexpr uint32_t PBR_FLAG_COMPACT_VIVID    = 1u << 11;
    // Biome tint type in bits 12-13: 0=none, 1=grass, 2=foliage, 3=water
    // Resolved in shader from per-section SSBO — NOT baked into vertex colorLayer
    static constexpr uint32_t PBR_FLAG_BIOME_TINT_SHIFT = 12u;
    static constexpr uint32_t PBR_FLAG_BIOME_TINT_MASK  = 0x3u << 12u;
    static constexpr uint32_t PBR_FLAG_BLOCK_GEOMETRY   = 1u << 14; // block chunk: use texture array, not bindless atlas
    static constexpr uint32_t PBR_FLAG_FLUID_GEOMETRY   = 1u << 15; // fluid surface: alpha is not a cutout mask
    static constexpr uint32_t PBR_FLAG_COMPACT_THIN_CUTOUT_PLANT = 1u << 15; // compact-only carrier for emissiveBlockType bit 31
    static constexpr uint32_t PBR_PACKED_THIN_CUTOUT_PLANT = 1u << 31; // emissiveBlockType bit: exact Minecraft plant cards
#else
    #define PBR_FLAG_USE_NORM        (1u << 0)
    #define PBR_FLAG_USE_COLOR_LAYER (1u << 1)
    #define PBR_FLAG_USE_TEXTURE     (1u << 2)
    #define PBR_FLAG_USE_OVERLAY     (1u << 3)
    #define PBR_FLAG_USE_GLINT       (1u << 4)
    #define PBR_FLAG_USE_LIGHT       (1u << 5)
    #define PBR_FLAG_GREEDY_MERGED   (1u << 6)
    #define PBR_FLAG_OVERLAY_ALPHA_MASK (1u << 7)
    #define PBR_FLAG_COORD_SHIFT     8u
    #define PBR_FLAG_COORD_MASK      (0x7u << 8u)
    #define PBR_FLAG_COMPACT_VIVID   (1u << 11)
    #define PBR_FLAG_BIOME_TINT_SHIFT 12u
    #define PBR_FLAG_BIOME_TINT_MASK  (0x3u << 12u)
    #define PBR_FLAG_BLOCK_GEOMETRY   (1u << 14)
    #define PBR_FLAG_FLUID_GEOMETRY   (1u << 15)
    #define PBR_FLAG_COMPACT_THIN_CUTOUT_PLANT (1u << 15)
    #define PBR_PACKED_THIN_CUTOUT_PLANT (1u << 31)
#endif

    // 96 bytes per vertex, std430 aligned (6 x vec4)
    // 96 bytes per vertex, std430 aligned (6 x vec4)
    struct PBRTriangle {
        T_VEC3 pos;                 // 0..11
        T_UINT flags;               // 12..15  (booleans + coordinate packed)

        T_VEC3 norm;                // 16..27
        T_FLOAT albedoEmission;     // 28..31

        T_VEC4 colorLayer;          // 32..47

        T_VEC3 postBase;            // 48..59
        T_UINT emissiveBlockType;   // 60..63  EmissiveBlock ordinal (0-39), 255 = none/LabPBR

        T_VEC2 textureUV;           // 64..71
        T_VEC2 glintUV;             // 72..79

        T_UINT textureID;           // 80..83
        T_UINT glintTexture;        // 84..87
        T_UINT overlayPacked;       // 88..91  (overlay_u & 0xFFFF) | (overlay_v << 16)
        T_UINT lightPacked;         // 92..95  (light_u & 0xFFFF) | (light_v << 16)
    };
#ifdef __cplusplus
    static_assert(sizeof(PBRTriangle) == 96, "PBRTriangle must be exactly 96 bytes");
    static_assert(offsetof(PBRTriangle, flags) == 12, "flags offset mismatch");
    static_assert(offsetof(PBRTriangle, norm) == 16, "norm offset mismatch");
    static_assert(offsetof(PBRTriangle, albedoEmission) == 28, "albedoEmission offset mismatch");
    static_assert(offsetof(PBRTriangle, colorLayer) == 32, "colorLayer offset mismatch");
    static_assert(offsetof(PBRTriangle, postBase) == 48, "postBase offset mismatch");
    static_assert(offsetof(PBRTriangle, textureUV) == 64, "textureUV offset mismatch");
    static_assert(offsetof(PBRTriangle, textureID) == 80, "textureID offset mismatch");
    static_assert(offsetof(PBRTriangle, overlayPacked) == 88, "overlayPacked offset mismatch");
    static_assert(offsetof(PBRTriangle, lightPacked) == 92, "lightPacked offset mismatch");
#endif

    // 32 bytes per vertex, std430 aligned (2 x vec4)
    // Compact format for world chunk geometry. Drops norm, postBase, lightPacked,
    // glintUV, glintTexture, overlayPacked. Compresses colorLayer to RGBA8,
    // albedoEmission to fp16. Entities keep full PBRTriangle.
    struct PBRTriangleCompact {
        T_VEC3 pos;            // 0..11   float32 world position (required by VK AS)
        T_UINT packed0;        // 12..15  flags/carriers:16 | textureID:16
                               //         bits 0-10 = flags, 11 = vivid, 15 = thin plant carrier
        T_VEC2 textureUV;      // 16..23  float32 atlas UVs
        T_UINT colorPacked;    // 24..27  R:8 | G:8 | B:8 | A:8
        T_UINT packed1;        // 28..31  albedoEmission_half:16 | emissiveBlockType:16
    };
#ifdef __cplusplus
    static_assert(sizeof(PBRTriangleCompact) == 32, "PBRTriangleCompact must be exactly 32 bytes");
    static_assert(offsetof(PBRTriangleCompact, pos) == 0, "compact pos offset mismatch");
    static_assert(offsetof(PBRTriangleCompact, packed0) == 12, "compact packed0 offset mismatch");
    static_assert(offsetof(PBRTriangleCompact, textureUV) == 16, "compact textureUV offset mismatch");
    static_assert(offsetof(PBRTriangleCompact, colorPacked) == 24, "compact colorPacked offset mismatch");
    static_assert(offsetof(PBRTriangleCompact, packed1) == 28, "compact packed1 offset mismatch");
#endif

    // 64 bytes per vertex, std430 aligned (4 x vec4)
    // Lossless format: drops only dead fields (norm, postBase, lightPacked).
    // Every field the shader reads is at full precision — bit-identical output.
    // Packs textureID + glintTexture into one uint32 (16 bits each).
    struct PBRTriangleLossless {
        T_VEC3 pos;                // 0..11   float32 world position
        T_UINT flags;              // 12..15  full uint32 (all flag bits intact)
        T_VEC4 colorLayer;         // 16..31  full vec4 (alpha preserved for glass)
        T_VEC2 textureUV;          // 32..39  float32 atlas UVs
        T_VEC2 glintUV;            // 40..47  float32 (enchantment shimmer preserved)
        T_FLOAT albedoEmission;    // 48..51  float32 (full precision, area lights exact)
        T_UINT emissiveBlockType;  // 52..55  full uint32 (vivid bit 16 in place)
        T_UINT textureID_glint;    // 56..59  textureID:16 | glintTexture:16
        T_UINT overlayPacked;      // 60..63  full uint32 (mining cracks preserved)
    };
#ifdef __cplusplus
    static_assert(sizeof(PBRTriangleLossless) == 64, "PBRTriangleLossless must be exactly 64 bytes");
    static_assert(offsetof(PBRTriangleLossless, pos) == 0, "lossless pos offset mismatch");
    static_assert(offsetof(PBRTriangleLossless, flags) == 12, "lossless flags offset mismatch");
    static_assert(offsetof(PBRTriangleLossless, colorLayer) == 16, "lossless colorLayer offset mismatch");
    static_assert(offsetof(PBRTriangleLossless, textureUV) == 32, "lossless textureUV offset mismatch");
    static_assert(offsetof(PBRTriangleLossless, glintUV) == 40, "lossless glintUV offset mismatch");
    static_assert(offsetof(PBRTriangleLossless, albedoEmission) == 48, "lossless albedoEmission offset mismatch");
    static_assert(offsetof(PBRTriangleLossless, emissiveBlockType) == 52, "lossless emissiveBlockType offset mismatch");
    static_assert(offsetof(PBRTriangleLossless, textureID_glint) == 56, "lossless textureID_glint offset mismatch");
    static_assert(offsetof(PBRTriangleLossless, overlayPacked) == 60, "lossless overlayPacked offset mismatch");
#endif
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
        T_FLOAT cameraTmin;  // Min ray distance for primary rays (skips enclosing block geometry)

        T_DVEC4 cameraPos; // w for padding

        T_UINT endSkyTextureID;
        T_UINT endPortalTextureID;
        T_UINT animTick;    // Global animation tick counter (incremented per game tick)
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
        T_FLOAT thunderGradient;
        T_FLOAT wetSurfaceStrength;

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

    // 36 bytes per entry, read by every ray hit
    struct TextureMapEntry {
        T_INT specular;
        T_INT normal;
        T_INT flag;
        T_INT properties;    // bit 0: has height map
        T_INT maskTexture;   // bindless index of R8_UNORM material class mask, -1 = none
        T_FLOAT _reserved0;  // padding (sprite bounds moved to per-vertex data)
        T_FLOAT _reserved1;
        T_FLOAT _reserved2;
        T_FLOAT _reserved3;
    };

#ifdef __cplusplus
    static constexpr int TEX_PROP_HAS_HEIGHT_MAP = 1;
    static constexpr int TEX_PROP_TEXTURE_ARRAY = 1 << 8;
#else
    #define TEX_PROP_HAS_HEIGHT_MAP 1
    #define TEX_PROP_TEXTURE_ARRAY (1 << 8)
#endif

    struct TextureMapping {
        TextureMapEntry entries[4096];
    };

    // Per-sprite metadata for texture array lookups.
    // 32 bytes, std430 aligned. Indexed by sequential spriteId assigned at resource load.
    struct SpriteEntry {
        T_UINT  baseLayer;       // First albedo layer in the texture array (animation frames start here)
        T_UINT  frameCount;      // 1 = static, N = animated (N consecutive layers)
        T_UINT  tickRate;        // Game ticks per animation frame (1 = every tick)
        T_UINT  flags;           // bit 0: hasSpecular, bit 1: hasNormal, bit 2: hasHeight
        T_INT   specularLayer;   // Layer in specular array (-1 = no specular)
        T_INT   normalLayer;     // Layer in normal array (-1 = no normal)
        T_INT   overlaySprite;   // spriteId of overlay texture (-1 = none) [grass block sides]
        T_INT   maskLayer;       // Block height range pack: minAlpha | (maxAlpha << 8), -1 = none
    };

#ifdef __cplusplus
    static constexpr uint32_t SPRITE_FLAG_HAS_SPECULAR = 1u << 0;
    static constexpr uint32_t SPRITE_FLAG_HAS_NORMAL   = 1u << 1;
    static constexpr uint32_t SPRITE_FLAG_HAS_HEIGHT   = 1u << 2;
    static constexpr uint32_t SPRITE_FLAG_SPEC_SOURCE_SHIFT = 3u;
    static constexpr uint32_t SPRITE_FLAG_NORMAL_SOURCE_SHIFT = 5u;
    static constexpr uint32_t SPRITE_FLAG_SOURCE_MASK = 0x3u;
    static constexpr uint32_t SPRITE_FLAG_SPEC_SOURCE_MASK = SPRITE_FLAG_SOURCE_MASK << SPRITE_FLAG_SPEC_SOURCE_SHIFT;
    static constexpr uint32_t SPRITE_FLAG_NORMAL_SOURCE_MASK = SPRITE_FLAG_SOURCE_MASK << SPRITE_FLAG_NORMAL_SOURCE_SHIFT;
    static constexpr uint32_t SPRITE_SOURCE_GENERATED = 0u;
    static constexpr uint32_t SPRITE_SOURCE_PACK_AUTHORED = 1u;
    static constexpr uint32_t SPRITE_SOURCE_USER_CUSTOM = 2u;
    static constexpr uint32_t SPRITE_SOURCE_FLAT = 3u;
    static constexpr uint32_t SPRITE_MAX_ENTRIES        = 4096u;
    static_assert(sizeof(SpriteEntry) == 32, "SpriteEntry must be exactly 32 bytes");
#else
    #define SPRITE_FLAG_HAS_SPECULAR (1u << 0)
    #define SPRITE_FLAG_HAS_NORMAL   (1u << 1)
    #define SPRITE_FLAG_HAS_HEIGHT   (1u << 2)
    #define SPRITE_FLAG_SPEC_SOURCE_SHIFT 3u
    #define SPRITE_FLAG_NORMAL_SOURCE_SHIFT 5u
    #define SPRITE_FLAG_SOURCE_MASK 0x3u
    #define SPRITE_FLAG_SPEC_SOURCE_MASK (SPRITE_FLAG_SOURCE_MASK << SPRITE_FLAG_SPEC_SOURCE_SHIFT)
    #define SPRITE_FLAG_NORMAL_SOURCE_MASK (SPRITE_FLAG_SOURCE_MASK << SPRITE_FLAG_NORMAL_SOURCE_SHIFT)
    #define SPRITE_SOURCE_GENERATED 0u
    #define SPRITE_SOURCE_PACK_AUTHORED 1u
    #define SPRITE_SOURCE_USER_CUSTOM 2u
    #define SPRITE_SOURCE_FLAT 3u
    #define SPRITE_MAX_ENTRIES       4096u
#endif

    struct SpriteRegistry {
        SpriteEntry entries[SPRITE_MAX_ENTRIES];
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

        // Pack 4: displacement + height field + normal controls (16 bytes)
        T_UINT  pomPacked0;   // heightFilter(3) | displacementMode(2) | heightSource(3) | filterRadius(4) | mipBias(4) | selfShadow(1)
        T_UINT  pomPacked1;   // normalClamp(8) | geometricBlend(8) | legacyAO(8) | heightContrast(8)
        T_UINT  pomPacked2;   // heightRemapMin(8) | heightRemapMax(8) | heightOffset(8) | normalDistanceFade(8)
        T_FLOAT pomDepth;     // [0.00-2.00] per-block displacement depth in blocks (0 = disabled)

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
        T_UINT  autoPBRPacked1; // heightGamma_u16 | reserved_u16<<16
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
