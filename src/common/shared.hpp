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
    static constexpr uint32_t PBR_FLAG_OVERLAY_ALPHA_MASK = 1u << 7; // colorLayer holds tint for folded block overlay sprites
    static constexpr uint32_t PBR_FLAG_COORD_SHIFT      = 8u;
    static constexpr uint32_t PBR_FLAG_COORD_MASK       = 0x7u << 8u; // 3 bits
    // Biome tint type in bits 12-13: 0=none, 1=grass, 2=foliage, 3=water
    // Resolved in shader from per-section SSBO - NOT baked into vertex colorLayer
    static constexpr uint32_t PBR_FLAG_BIOME_TINT_SHIFT = 12u;
    static constexpr uint32_t PBR_FLAG_BIOME_TINT_MASK  = 0x3u << 12u;
    static constexpr uint32_t PBR_FLAG_BLOCK_GEOMETRY   = 1u << 14; // block chunk: use texture array, not bindless atlas
    static constexpr uint32_t PBR_FLAG_FLUID_GEOMETRY   = 1u << 15; // fluid surface: alpha is not a cutout mask
    static constexpr uint32_t PBR_FLAG_ALPHA_MODE_SHIFT = 16u;
    static constexpr uint32_t PBR_FLAG_ALPHA_MODE_MASK  = 0x3u << 16u; // 0=opaque, 1=cutout, 2=transparent
    static constexpr uint32_t PBR_ALPHA_MODE_OPAQUE      = 0u;
    static constexpr uint32_t PBR_ALPHA_MODE_CUTOUT      = 1u;
    static constexpr uint32_t PBR_ALPHA_MODE_TRANSPARENT = 2u;
    static constexpr uint32_t PBR_FLAG_TEXT_MODE_SHIFT = 18u;
    static constexpr uint32_t PBR_FLAG_TEXT_MODE_MASK  = 0xFu << 18u; // vanilla text modes 1..8
    static constexpr uint32_t PBR_FLAG_WATER_GEOMETRY  = 1u << 22; // water fluid surface, distinct from lava/other fluids
    static constexpr uint32_t PBR_FLAG_GREEDY_MERGED = 1u << 23; // greedy-meshed quad (skip per-face AO)
    static constexpr uint32_t PBR_FLAG_COMPACT_VIVID    = 1u << 11;
    static constexpr uint32_t PBR_TEXT_MODE_BACKGROUND = 1u;
    static constexpr uint32_t PBR_TEXT_MODE_INTENSITY = 2u;
    static constexpr uint32_t PBR_TEXT_MODE_RGBA = 3u;
    static constexpr uint32_t PBR_TEXT_MODE_BACKGROUND_SEE_THROUGH = 4u;
    static constexpr uint32_t PBR_TEXT_MODE_INTENSITY_SEE_THROUGH = 5u;
    static constexpr uint32_t PBR_TEXT_MODE_RGBA_SEE_THROUGH = 6u;
    static constexpr uint32_t PBR_TEXT_MODE_INTENSITY_POLYGON_OFFSET = 7u;
    static constexpr uint32_t PBR_TEXT_MODE_RGBA_POLYGON_OFFSET = 8u;
    static constexpr uint32_t PBR_PACKED_EMISSIVE_TYPE_MASK = 0xFFu;
    static constexpr uint32_t PBR_PACKED_SHADER_BLOCK_ID_SHIFT = 17u;
    static constexpr uint32_t PBR_PACKED_SHADER_BLOCK_ID_MASK = 0x3FFFu << PBR_PACKED_SHADER_BLOCK_ID_SHIFT;
    static constexpr uint32_t PBR_PACKED_THIN_CUTOUT_PLANT = 1u << 31; // emissiveBlockType bit: exact Minecraft plant cards
#else
    #define PBR_FLAG_USE_NORM        (1u << 0)
    #define PBR_FLAG_USE_COLOR_LAYER (1u << 1)
    #define PBR_FLAG_USE_TEXTURE     (1u << 2)
    #define PBR_FLAG_USE_OVERLAY     (1u << 3)
    #define PBR_FLAG_USE_GLINT       (1u << 4)
    #define PBR_FLAG_USE_LIGHT       (1u << 5)
    #define PBR_FLAG_OVERLAY_ALPHA_MASK (1u << 7)
    #define PBR_FLAG_COORD_SHIFT     8u
    #define PBR_FLAG_COORD_MASK      (0x7u << 8u)
    #define PBR_FLAG_BIOME_TINT_SHIFT 12u
    #define PBR_FLAG_BIOME_TINT_MASK  (0x3u << 12u)
    #define PBR_FLAG_BLOCK_GEOMETRY   (1u << 14)
    #define PBR_FLAG_FLUID_GEOMETRY   (1u << 15)
    #define PBR_FLAG_ALPHA_MODE_SHIFT 16u
    #define PBR_FLAG_ALPHA_MODE_MASK  (0x3u << 16u)
    #define PBR_ALPHA_MODE_OPAQUE      0u
    #define PBR_ALPHA_MODE_CUTOUT      1u
    #define PBR_ALPHA_MODE_TRANSPARENT 2u
    #define PBR_FLAG_TEXT_MODE_SHIFT 18u
    #define PBR_FLAG_TEXT_MODE_MASK  (0xFu << 18u)
    #define PBR_FLAG_WATER_GEOMETRY  (1u << 22)
    #define PBR_FLAG_GREEDY_MERGED (1u << 23)
    #define PBR_FLAG_COMPACT_VIVID   (1u << 11)
    #define PBR_TEXT_MODE_BACKGROUND 1u
    #define PBR_TEXT_MODE_INTENSITY 2u
    #define PBR_TEXT_MODE_RGBA 3u
    #define PBR_TEXT_MODE_BACKGROUND_SEE_THROUGH 4u
    #define PBR_TEXT_MODE_INTENSITY_SEE_THROUGH 5u
    #define PBR_TEXT_MODE_RGBA_SEE_THROUGH 6u
    #define PBR_TEXT_MODE_INTENSITY_POLYGON_OFFSET 7u
    #define PBR_TEXT_MODE_RGBA_POLYGON_OFFSET 8u
    #define PBR_PACKED_EMISSIVE_TYPE_MASK 0xFFu
    #define PBR_PACKED_SHADER_BLOCK_ID_SHIFT 17u
    #define PBR_PACKED_SHADER_BLOCK_ID_MASK (0x3FFFu << PBR_PACKED_SHADER_BLOCK_ID_SHIFT)
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

        T_VEC3 postBase;            // 48..59  block origin for block geometry, post base otherwise
        T_UINT emissiveBlockType;   // 60..63  low 8=EmissiveBlock, bits 17..30=shader block id, bit 31=thin plant

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
    static_assert(offsetof(PBRTriangle, emissiveBlockType) == 60, "emissiveBlockType offset mismatch");
    static_assert(offsetof(PBRTriangle, textureUV) == 64, "textureUV offset mismatch");
    static_assert(offsetof(PBRTriangle, textureID) == 80, "textureID offset mismatch");
    static_assert(offsetof(PBRTriangle, overlayPacked) == 88, "overlayPacked offset mismatch");
    static_assert(offsetof(PBRTriangle, lightPacked) == 92, "lightPacked offset mismatch");
    static_assert(PBR_FLAG_FLUID_GEOMETRY == (1u << 15), "fluid geometry flag bit mismatch");
    static_assert(PBR_FLAG_WATER_GEOMETRY == (1u << 22), "water geometry flag bit mismatch");
    static_assert(PBR_PACKED_EMISSIVE_TYPE_MASK == 0xFFu, "emissive block type mask mismatch");
    static_assert(PBR_PACKED_SHADER_BLOCK_ID_SHIFT == 17u, "shader block id shift mismatch");
    static_assert(PBR_PACKED_SHADER_BLOCK_ID_MASK == (0x3FFFu << 17u), "shader block id mask mismatch");
#endif

    // 32 bytes per vertex, compact far-field format.
    struct PBRTriangleCompact {
        T_VEC3 pos;            // 0..11
        T_UINT packed0;        // 12..15  flags:12 | pad:4 | textureID:16
        T_VEC2 textureUV;      // 16..23
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

    // 64 bytes per vertex, lossless near-field format.
    struct PBRTriangleLossless {
        T_VEC3 pos;                // 0..11
        T_UINT flags;              // 12..15
        T_VEC4 colorLayer;         // 16..31
        T_VEC2 textureUV;          // 32..39
        T_VEC2 glintUV;            // 40..47
        T_FLOAT albedoEmission;    // 48..51
        T_UINT emissiveBlockType;  // 52..55
        T_UINT textureID_glint;    // 56..59
        T_UINT overlayPacked;      // 60..63
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
        T_INT _reservedTexRule; // reserved; old material-class mask slot removed
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

    // Texture-primary scalar material rules. These are keyed by spriteId and are
    // deliberately not broad material classes. LabPBR texture pixels remain the
    // source for roughness, F0/metal, emission, normal, AO, and height.
    struct TextureRuleEntry {
        T_UINT flags;             // bit 0: enabled; later bits mark explicit channels
        T_FLOAT transmission;     // 0 opaque, 1 fully transmissive
        T_FLOAT ior;
        T_FLOAT metallic;
        T_FLOAT f0;
        T_FLOAT roughness;
        T_FLOAT anisotropic;
        T_FLOAT sheenWeight;
        T_FLOAT sheenTint;
        T_FLOAT coatWeight;
        T_FLOAT coatRoughness;
        T_FLOAT emission;
        T_FLOAT conductorF0R;
        T_FLOAT conductorF0G;
        T_FLOAT conductorF0B;
        T_UINT modeFlags;         // bits 0-1 volume, 2-3 thickness, 4-5 coat mask, 6-7 diffuse model
        T_FLOAT absorptionR;
        T_FLOAT absorptionG;
        T_FLOAT absorptionB;
        T_FLOAT absorptionDistance;
        T_FLOAT thicknessAmount;
        T_FLOAT refractionRoughness;
        T_FLOAT emissionTintR;
        T_FLOAT emissionTintG;
        T_FLOAT emissionTintB;
        T_FLOAT emissionNits;
        T_FLOAT anisotropicRotation;
        T_FLOAT sheenRoughness;
        T_FLOAT coatIor;
        T_FLOAT coatTintR;
        T_FLOAT coatTintG;
        T_FLOAT coatTintB;
        T_FLOAT coatMask;
        T_FLOAT uvScaleU;
        T_FLOAT uvScaleV;
        T_FLOAT uvOffsetU;
        T_FLOAT uvOffsetV;
        T_FLOAT filterRadius;
        T_FLOAT mipBias;
        T_FLOAT displacementScale;
        T_FLOAT subSurfaceRadius;
        T_FLOAT subSurfaceThickness;
        T_FLOAT subSurfaceTintR;
        T_FLOAT subSurfaceTintG;
        T_FLOAT subSurfaceTintB;
        T_FLOAT reserved0;
        T_FLOAT reserved1;
        T_FLOAT reserved2;
    };

    // Global material-id table. PBRTriangle.textureID carries this material id
    // for block geometry; the first backend wraps current sprite-array layers.
    struct MaterialEntry {
        T_UINT materialId;
        T_INT  baseSpriteId;
        T_INT  fallbackMaterialId;
        T_UINT flags;
        T_UINT albedoPage;
        T_INT  albedoLayer;
        T_UINT specularPage;
        T_INT  specularLayer;
        T_UINT normalPage;
        T_INT  normalLayer;
        T_UINT flagPage;
        T_INT  flagLayer;
        T_INT  overlayMaterialId;
        T_UINT displacementPolicy;
        T_FLOAT displacementScale;
        T_INT  heightRangePacked;
        T_FLOAT uvScaleU;
        T_FLOAT uvScaleV;
        T_FLOAT uvOffsetU;
        T_FLOAT uvOffsetV;
    };

#ifdef __cplusplus
    static constexpr uint32_t SPRITE_FLAG_HAS_SPECULAR = 1u << 0;
    static constexpr uint32_t SPRITE_FLAG_HAS_NORMAL   = 1u << 1;
    static constexpr uint32_t SPRITE_FLAG_HAS_HEIGHT   = 1u << 2;
    static constexpr uint32_t SPRITE_FLAG_EMISSIVE_OVERLAY = 1u << 7;
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
    static constexpr uint32_t MATERIAL_MAX_ENTRIES      = 65536u;
    static constexpr uint32_t MATERIAL_TEXTURE_PAGE_MAX = 64u;
    static constexpr uint32_t MATERIAL_PAGE_NAMESPACE_MASK = 0xF0000000u;
    static constexpr uint32_t MATERIAL_PAGE_INDEX_MASK = 0x0FFFFFFFu;
    static constexpr uint32_t MATERIAL_PAGE_NAMESPACE_MATERIAL = 0x00000000u;
    static constexpr uint32_t MATERIAL_PAGE_NAMESPACE_VANILLA_TIER = 0x80000000u;
    static constexpr uint32_t MATERIAL_FLAG_VALID = 1u << 0;
    static constexpr uint32_t MATERIAL_FLAG_VANILLA_SPRITE = 1u << 1;
    static constexpr uint32_t MATERIAL_FLAG_COMPAT_VIRTUAL = 1u << 2;
    static constexpr uint32_t MATERIAL_FLAG_GPU_RESIDENT = 1u << 3;
    static constexpr uint32_t MATERIAL_FLAG_PENDING_RESIDENCY = 1u << 4;
    static constexpr uint32_t MATERIAL_FLAG_FALLBACK = 1u << 5;
    static constexpr uint32_t MATERIAL_FLAG_HAS_SPECULAR = 1u << 6;
    static constexpr uint32_t MATERIAL_FLAG_HAS_NORMAL = 1u << 7;
    static constexpr uint32_t MATERIAL_FLAG_DISPLACEMENT_ELIGIBLE = 1u << 8;
    static constexpr uint32_t MATERIAL_FLAG_CUTOUT_DISPLACEMENT_BLOCKED = 1u << 9;
    static constexpr uint32_t MATERIAL_DISPLACEMENT_DISABLED = 0u;
    static constexpr uint32_t MATERIAL_DISPLACEMENT_AUTHORED_HEIGHT = 1u;
    static constexpr uint32_t MATERIAL_DISPLACEMENT_BLOCKED_CUTOUT = 2u;
    static_assert(sizeof(SpriteEntry) == 32, "SpriteEntry must be exactly 32 bytes");
    static constexpr uint32_t TEXTURE_RULE_ENABLED = 1u << 0;
    static constexpr uint32_t TEXTURE_RULE_TRANSMISSION = 1u << 1;
    static constexpr uint32_t TEXTURE_RULE_IOR = 1u << 2;
    static constexpr uint32_t TEXTURE_RULE_METALLIC = 1u << 3;
    static constexpr uint32_t TEXTURE_RULE_F0 = 1u << 4;
    static constexpr uint32_t TEXTURE_RULE_ROUGHNESS = 1u << 5;
    static constexpr uint32_t TEXTURE_RULE_ANISOTROPIC = 1u << 6;
    static constexpr uint32_t TEXTURE_RULE_SHEEN = 1u << 7;
    static constexpr uint32_t TEXTURE_RULE_COAT = 1u << 8;
    static constexpr uint32_t TEXTURE_RULE_EMISSION = 1u << 9;
    static constexpr uint32_t TEXTURE_RULE_CONDUCTOR_F0_RGB = 1u << 10;
    static constexpr uint32_t TEXTURE_RULE_ABSORPTION = 1u << 11;
    static constexpr uint32_t TEXTURE_RULE_THICKNESS = 1u << 12;
    static constexpr uint32_t TEXTURE_RULE_VOLUME_MODE = 1u << 13;
    static constexpr uint32_t TEXTURE_RULE_REFRACTION_ROUGHNESS = 1u << 14;
    static constexpr uint32_t TEXTURE_RULE_EMISSION_TINT_NITS = 1u << 15;
    static constexpr uint32_t TEXTURE_RULE_ANISOTROPIC_ROTATION = 1u << 16;
    static constexpr uint32_t TEXTURE_RULE_COAT_EXT = 1u << 17;
    static constexpr uint32_t TEXTURE_RULE_SHEEN_ROUGHNESS = 1u << 18;
    static constexpr uint32_t TEXTURE_RULE_UV_TRANSFORM = 1u << 19;
    static constexpr uint32_t TEXTURE_RULE_FILTER_RADIUS = 1u << 20;
    static constexpr uint32_t TEXTURE_RULE_MIP_BIAS = 1u << 21;
    static constexpr uint32_t TEXTURE_RULE_DISPLACEMENT_SCALE = 1u << 22;
    static constexpr uint32_t TEXTURE_RULE_SUBSURFACE_EXT = 1u << 23;
    static constexpr uint32_t TEXTURE_RULE_DIFFUSE_MODEL = 1u << 24;
    static constexpr uint32_t TEXTURE_RULE_MODE_DIFFUSE_SHIFT = 6u;
    static constexpr uint32_t TEXTURE_RULE_MODE_DIFFUSE_MASK = 0x3u << TEXTURE_RULE_MODE_DIFFUSE_SHIFT;
    static constexpr uint32_t TEXTURE_RULE_DIFFUSE_GLOBAL = 0u;
    static constexpr uint32_t TEXTURE_RULE_DIFFUSE_EON = 1u;
    static constexpr uint32_t TEXTURE_RULE_DIFFUSE_VMF = 2u;
    static constexpr uint32_t TEXTURE_RULE_DIFFUSE_LEGACY = 3u;
    static_assert(sizeof(TextureRuleEntry) == 192, "TextureRuleEntry must be exactly 192 bytes");
    static_assert(sizeof(MaterialEntry) == 80, "MaterialEntry must be exactly 80 bytes");
    static_assert(offsetof(MaterialEntry, materialId) == 0);
    static_assert(offsetof(MaterialEntry, albedoPage) == 16);
    static_assert(offsetof(MaterialEntry, albedoLayer) == 20);
    static_assert(offsetof(MaterialEntry, specularPage) == 24);
    static_assert(offsetof(MaterialEntry, specularLayer) == 28);
    static_assert(offsetof(MaterialEntry, normalPage) == 32);
    static_assert(offsetof(MaterialEntry, normalLayer) == 36);
    static_assert(offsetof(MaterialEntry, flagPage) == 40);
    static_assert(offsetof(MaterialEntry, flagLayer) == 44);
    static_assert(offsetof(MaterialEntry, uvOffsetV) == 76);
#else
    #define SPRITE_FLAG_HAS_SPECULAR (1u << 0)
    #define SPRITE_FLAG_HAS_NORMAL   (1u << 1)
    #define SPRITE_FLAG_HAS_HEIGHT   (1u << 2)
    #define SPRITE_FLAG_EMISSIVE_OVERLAY (1u << 7)
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
    #define MATERIAL_MAX_ENTRIES     65536u
    #define MATERIAL_TEXTURE_PAGE_MAX 64u
    #define MATERIAL_PAGE_NAMESPACE_MASK 0xF0000000u
    #define MATERIAL_PAGE_INDEX_MASK 0x0FFFFFFFu
    #define MATERIAL_PAGE_NAMESPACE_MATERIAL 0x00000000u
    #define MATERIAL_PAGE_NAMESPACE_VANILLA_TIER 0x80000000u
    #define MATERIAL_FLAG_VALID (1u << 0)
    #define MATERIAL_FLAG_VANILLA_SPRITE (1u << 1)
    #define MATERIAL_FLAG_COMPAT_VIRTUAL (1u << 2)
    #define MATERIAL_FLAG_GPU_RESIDENT (1u << 3)
    #define MATERIAL_FLAG_PENDING_RESIDENCY (1u << 4)
    #define MATERIAL_FLAG_FALLBACK (1u << 5)
    #define MATERIAL_FLAG_HAS_SPECULAR (1u << 6)
    #define MATERIAL_FLAG_HAS_NORMAL (1u << 7)
    #define MATERIAL_FLAG_DISPLACEMENT_ELIGIBLE (1u << 8)
    #define MATERIAL_FLAG_CUTOUT_DISPLACEMENT_BLOCKED (1u << 9)
    #define MATERIAL_DISPLACEMENT_DISABLED 0u
    #define MATERIAL_DISPLACEMENT_AUTHORED_HEIGHT 1u
    #define MATERIAL_DISPLACEMENT_BLOCKED_CUTOUT 2u
    #define TEXTURE_RULE_ENABLED (1u << 0)
    #define TEXTURE_RULE_TRANSMISSION (1u << 1)
    #define TEXTURE_RULE_IOR (1u << 2)
    #define TEXTURE_RULE_METALLIC (1u << 3)
    #define TEXTURE_RULE_F0 (1u << 4)
    #define TEXTURE_RULE_ROUGHNESS (1u << 5)
    #define TEXTURE_RULE_ANISOTROPIC (1u << 6)
    #define TEXTURE_RULE_SHEEN (1u << 7)
    #define TEXTURE_RULE_COAT (1u << 8)
    #define TEXTURE_RULE_EMISSION (1u << 9)
    #define TEXTURE_RULE_CONDUCTOR_F0_RGB (1u << 10)
    #define TEXTURE_RULE_ABSORPTION (1u << 11)
    #define TEXTURE_RULE_THICKNESS (1u << 12)
    #define TEXTURE_RULE_VOLUME_MODE (1u << 13)
    #define TEXTURE_RULE_REFRACTION_ROUGHNESS (1u << 14)
    #define TEXTURE_RULE_EMISSION_TINT_NITS (1u << 15)
    #define TEXTURE_RULE_ANISOTROPIC_ROTATION (1u << 16)
    #define TEXTURE_RULE_COAT_EXT (1u << 17)
    #define TEXTURE_RULE_SHEEN_ROUGHNESS (1u << 18)
    #define TEXTURE_RULE_UV_TRANSFORM (1u << 19)
    #define TEXTURE_RULE_FILTER_RADIUS (1u << 20)
    #define TEXTURE_RULE_MIP_BIAS (1u << 21)
    #define TEXTURE_RULE_DISPLACEMENT_SCALE (1u << 22)
    #define TEXTURE_RULE_SUBSURFACE_EXT (1u << 23)
    #define TEXTURE_RULE_DIFFUSE_MODEL (1u << 24)
    #define TEXTURE_RULE_MODE_DIFFUSE_SHIFT 6u
    #define TEXTURE_RULE_MODE_DIFFUSE_MASK (0x3u << TEXTURE_RULE_MODE_DIFFUSE_SHIFT)
    #define TEXTURE_RULE_DIFFUSE_GLOBAL 0u
    #define TEXTURE_RULE_DIFFUSE_EON 1u
    #define TEXTURE_RULE_DIFFUSE_VMF 2u
    #define TEXTURE_RULE_DIFFUSE_LEGACY 3u
#endif

    struct SpriteRegistry {
        SpriteEntry entries[SPRITE_MAX_ENTRIES];
    };

    struct TextureRuleRegistry {
        TextureRuleEntry entries[SPRITE_MAX_ENTRIES];
    };

    struct MaterialRegistry {
        MaterialEntry entries[MATERIAL_MAX_ENTRIES];
    };

#ifdef __cplusplus
    static constexpr int MAX_MATERIAL_CLASSES = 512;
#else
    #define MAX_MATERIAL_CLASSES 512
#endif

    struct MaterialClassEntry {
        T_VEC3 f0;
        T_FLOAT roughness;
        T_FLOAT metallic;
        T_FLOAT transmission;
        T_FLOAT ior;
        T_FLOAT subsurface;
        T_FLOAT anisotropic;
        T_FLOAT sheenWeight;
        T_FLOAT sheenTint;
        T_FLOAT coatWeight;
        T_FLOAT coatRoughness;
        T_FLOAT noiseScale;
        T_FLOAT noiseStrength;
        T_UINT noisePacked;
        T_UINT pomPacked0;
        T_UINT pomPacked1;
        T_UINT pomPacked2;
        T_FLOAT pomDepth;
        T_FLOAT gamutBoost;
        T_FLOAT noiseMaskThreshold;
        T_UINT noiseMaskPacked;
        T_FLOAT normalStrength;
        T_FLOAT noiseRotation;
        T_FLOAT noiseAspect;
        T_FLOAT noiseLacunarity;
        T_FLOAT noiseContrast;
        T_FLOAT emissionNits;
        T_UINT emissionType;
        T_UINT classId;
        T_UINT flags;
        T_FLOAT lumMin;
        T_FLOAT lumMax;
        T_UINT autoPBRPacked0;
        T_UINT autoPBRPacked1;
    };

    struct MaterialClassMapping {
        MaterialClassEntry entries[MAX_MATERIAL_CLASSES];
    };

    struct DisplacedFaceData {
        T_VEC3 corner;
        T_UINT faceAxis;
        T_VEC3 edgeU;
        T_FLOAT heightScale;
        T_VEC3 edgeV;
        T_UINT textureID;
        T_INT normalTexID;
        T_INT specularTexID;
        T_VEC2 uvMin;
        T_VEC2 uvMax;
        T_UINT pomPacked0;
        T_UINT materialClassIdx;
        T_UINT emissiveBlockType;
        T_UINT properties;
        T_FLOAT lumMin;
        T_FLOAT lumMax;
        T_VEC4 colorLayer;
        T_UINT pomPacked1;
        T_UINT pomPacked2;
        T_UINT flags;
        T_UINT fadeEdgeMask;
    };

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
        T_INT customLightmapEnabled;
        T_INT customLightmapIncludesNightVision;
        T_INT pad0;
        T_INT pad1;
        T_INT pad2;

        T_VEC4 customSkyLight[16];
        T_VEC4 customBlockLight[16];
    };
#ifdef __cplusplus
}; // namespace Data
#endif
#ifdef __cplusplus
}; // namespace vk
#endif

#endif
