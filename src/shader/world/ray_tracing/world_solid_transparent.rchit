#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#ifndef RARSER_SHADER_DISPLACEMENT
#define RARSER_SHADER_DISPLACEMENT 1
#endif
#ifndef RARSER_THIN_PLANT_PRIMARY_FIX
#define RARSER_THIN_PLANT_PRIMARY_FIX 1
#endif

#include "../util/surface_bsdf.glsl"
#include "../util/random.glsl"
#include "../util/blue_noise.glsl"
#include "../util/ray_cone.glsl"
#include "../util/ray_payloads.glsl"
#include "../util/util.glsl"
#include "common/shared.hpp"
#include "../util/ray_offset.glsl"
#include "../util/colorspace.glsl"
#include "../util/noise.glsl"
#include "../util/texture_rules.glsl"

layout(set = 0, binding = 0) uniform sampler2D textures[];

#include "../util/sprite_fetch.glsl"

#include "../util/clouds.glsl"
#include "../util/fft_water.glsl"

layout(set = 1, binding = 0) uniform accelerationStructureEXT topLevelAS;

layout(set = 1, binding = 1) readonly buffer BLASOffsets {
    uint offsets[];
}
blasOffsets;

layout(set = 1, binding = 2) readonly buffer VertexBufferAddr {
    uint64_t addrs[];
}
vertexBufferAddrs;

layout(set = 1, binding = 3) readonly buffer IndexBufferAddr {
    uint64_t addrs[];
}
indexBufferAddrs;

layout(set = 1, binding = 4) readonly buffer LastVertexBufferAddr {
    uint64_t addrs[];
}
lastVertexBufferAddrs;

layout(set = 1, binding = 5) readonly buffer LastIndexBufferAddr {
    uint64_t addrs[];
}
lastIndexBufferAddrs;

layout(set = 1, binding = 6) readonly buffer LastObjToWorldMat {
    mat4 mat[];
}
lastObjToWorldMats;

layout(set = 1, binding = 7) readonly buffer TextureMappingBuffer {
    TextureMapping mapping;
};

#if RARSER_SHADER_DISPLACEMENT
#include "../util/displacement.glsl"
#endif

layout(set = 1, binding = 10) readonly buffer BiomeColorBuffer {
    uvec4 data[];  // .x=grass, .y=foliage, .z=water (packed 0x00RRGGBB)
} biomeColors;

#include "../util/vertex_fetch.glsl"

layout(set = 2, binding = 0) uniform WorldUniform {
    WorldUBO worldUbo;
};

layout(set = 2, binding = 1) uniform LastWorldUniform {
    WorldUBO lastWorldUbo;
};

layout(set = 2, binding = 2) uniform SkyUniform {
    SkyUBO skyUBO;
};

// VertexBuffer and IndexBuffer declared in vertex_fetch.glsl

layout(push_constant) uniform PushConstant {
    int numRayBounces;
    int flags;
    int reservedLighting0;
    float reservedLighting1;
    int reservedLighting2;
    int reservedLighting3;
    int reservedLighting4;
    float preExposure;
    float displacementDepthScale;
    int displacementPrimarySteps;
    int displacementRefinementSteps;
    float displacementFadeDistanceBlocks;
    uint blueNoiseFrame;
    uint rtDebugFlags;
    uint handInstanceCount;
    uint reservedPc0;
    uint64_t sharcHashEntries;
    uint64_t sharcAccumulation;
    uint64_t sharcResolved;
    float sharcCameraX;
    float sharcCameraY;
    float sharcCameraZ;
    float sharcSceneScale;
    uint sharcCapacity;
    float sharcRadianceScale;
    uint sharcFrameIndex;
    float sharcRoughnessThreshold;
    int sharcUpdateBlockSize;
    int sharcUpdateBounces;
    int sharcQueryMode;
    int sharcQueryReserved;
    int offlineFlags;
    int accumFrameCount;
    float aperture;
    float focalDistance;
    uint64_t reservedAddr;
} pc;
#define SIMPLIFIED_INDIRECT ((pc.flags & 1) != 0)
#define BEER_LAW_SHADOWS   ((pc.flags & 512) != 0)
#define NO_EMISSION_CLAMP  ((pc.flags & 1024) != 0)
#define PHYSICAL_SUN_DISK  ((pc.flags & 2048) != 0)
#define NO_HAND_AMBIENT    ((pc.flags & 4096) != 0)
#define ENTITY_NORMALS_ON  ((pc.flags & 32768) != 0)
#define RT_DEBUG_DISABLE_SECONDARY_SUN_SHADOW   ((pc.rtDebugFlags & 4u) != 0u)
#define RT_DEBUG_DISABLE_SECONDARY_CLOUD_SHADOW ((pc.rtDebugFlags & 8u) != 0u)
#define RT_DEBUG_DISABLE_DIRECT_LIGHTING        ((pc.rtDebugFlags & 16u) != 0u)
#define RT_DEBUG_MATERIAL_MODE_SHIFT 24u
#define RT_DEBUG_MATERIAL_MODE_MASK  15u
#define RT_DEBUG_MATERIAL_MODE       ((pc.rtDebugFlags >> RT_DEBUG_MATERIAL_MODE_SHIFT) & RT_DEBUG_MATERIAL_MODE_MASK)
#define RT_DEBUG_MATERIAL_ID         1u
#define RT_DEBUG_MATERIAL_PAGE_LAYER 2u
#define RT_DEBUG_MATERIAL_DISPLACE_ELIGIBLE 3u
#define RT_DEBUG_MATERIAL_DISPLACE_HEIGHT   4u

layout(set = 3, binding = 3, rgba32f) uniform readonly image2D normalRoughnessImage;
layout(set = 3, binding = 4, rg32f) uniform readonly image2D motionVectorImage;
layout(set = 3, binding = 5, r32f) uniform readonly image2D linearDepthImage;

layout(location = 0) rayPayloadInEXT PrimaryRay mainRay;
layout(location = 1) rayPayloadEXT ShadowRay shadowRay;
hitAttributeEXT vec2 attribs;

vec3 materialDebugHashColor(uint materialId) {
    uint h = materialId * 1664525u + 1013904223u;
    h ^= h >> 16u;
    return vec3(
        float((h >> 0u) & 255u),
        float((h >> 8u) & 255u),
        float((h >> 16u) & 255u)) / 255.0;
}

vec3 materialDebugPageLayer(uint materialId) {
    MaterialEntry material = safeMaterialEntry(materialId);
    uint albedoPage = materialTexturePage(material.albedoPage);
    uint normalPage = materialTexturePage(material.normalPage);
    uint albedoLayer = uint(max(material.albedoLayer, 0));
    return vec3(
        float(albedoPage & 255u) / 255.0,
        float(albedoLayer & 255u) / 255.0,
        float(normalPage & 255u) / 255.0);
}

#if RARSER_SHADER_DISPLACEMENT
vec3 materialDebugDisplacementEligible(uint materialId) {
    return displacementBlockHasAuthoredHeight(materialId) ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
}

vec3 materialDebugDisplacementHeight(uint materialId, vec2 uv, float lod) {
    if (!displacementBlockHasAuthoredHeight(materialId)) return vec3(0.0);
    int layer = displacementNormalLayer(materialId);
    if (layer < 0) return vec3(0.0);
    float sampleLod = displacementClampedRuleLod(materialId, lod);
    vec2 sampleUv = materialRuleUv(materialId, fract(uv));
    float h = displacementSampleNormalAlphaTrilinear(materialId, layer, sampleUv, sampleLod);
    return vec3(h);
}
#else
vec3 materialDebugDisplacementEligible(uint materialId) {
    return vec3(1.0, 0.0, 0.0);
}

vec3 materialDebugDisplacementHeight(uint materialId, vec2 uv, float lod) {
    return vec3(0.0);
}
#endif

vec3 materialDebugColor(uint materialId, vec2 uv, float lod) {
    uint mode = RT_DEBUG_MATERIAL_MODE;
    if (mode == RT_DEBUG_MATERIAL_ID) return materialDebugHashColor(materialId);
    if (mode == RT_DEBUG_MATERIAL_PAGE_LAYER) return materialDebugPageLayer(materialId);
    if (mode == RT_DEBUG_MATERIAL_DISPLACE_ELIGIBLE) return materialDebugDisplacementEligible(materialId);
    if (mode == RT_DEBUG_MATERIAL_DISPLACE_HEIGHT) return materialDebugDisplacementHeight(materialId, uv, lod);
    return vec3(0.0);
}

vec3 thinPlantRayOrigin(vec3 worldPos, vec3 rayDir, vec3 geometricNormal) {
    vec3 biasNormal = dot(rayDir, geometricNormal) >= 0.0 ? geometricNormal : -geometricNormal;
    return offset_ray(worldPos, biasNormal);
}

uint thinPlantBlockTypeFromPacked(uint packedBlockType) {
    return (packedBlockType & PBR_PACKED_SHADER_BLOCK_ID_MASK) >> PBR_PACKED_SHADER_BLOCK_ID_SHIFT;
}

uint thinPlantHashInt(int v, uint salt) {
    uint x = uint(v) ^ salt;
    x ^= x >> 16u;
    x *= 0x7feb352du;
    x ^= x >> 15u;
    x *= 0x846ca68bu;
    return x ^ (x >> 16u);
}

uint thinPlantColumnHash(vec3 cameraRelativeWorldPos) {
    ivec3 cell = ivec3(floor(cameraRelativeWorldPos + vec3(worldUbo.cameraPos.xyz) + vec3(0.0001)));
    uint h = thinPlantHashInt(cell.x, 0x9e3779b9u) ^ thinPlantHashInt(cell.z, 0x85ebca6bu);
    return h == 0u ? 1u : h;
}

float thinPlantFill() {
    return 0.18;
}

float thinPlantBrightness() {
    return 0.85;
}

vec3 thinPlantAlbedo(vec3 albedo) {
    float brightness = thinPlantBrightness();
    float cap = clamp(0.40 + 0.38 * brightness, 0.40, 0.88);
    return min(clamp(albedo, vec3(0.0), vec3(1.0)) * clamp(brightness, 0.35, 1.25), vec3(cap));
}

float thinPlantDiffuseFactor(vec3 lightDir) {
    float fill = thinPlantFill();
    float y = clamp(normalize(lightDir).y, -1.0, 1.0);
    float side = sqrt(max(1.0 - y * y, 0.0));
    float floorTerm = mix(0.06, 0.22, fill);
    float topTerm = mix(0.30, 0.52, fill);
    float sideTerm = mix(0.03, 0.09, fill);
    return clamp(floorTerm + topTerm * max(y, 0.0) + sideTerm * side, 0.04, 0.62);
}

vec3 thinPlantDiffuseEval(LabPBRMat mat, vec3 lightDir) {
    return thinPlantAlbedo(mat.albedo) * (thinPlantDiffuseFactor(lightDir) * INV_PI);
}

vec3 thinPlantDiffuseSample(LabPBRMat mat, out vec3 sampleDir, out float pdf, inout uint seed) {
    vec3 plantNormal = vec3(0.0, 1.0, 0.0);
    vec3 T, B;
    Onb(plantNormal, T, B);

    vec3 localDir = CosineSampleHemisphere(rand(seed), rand(seed));
    sampleDir = normalize(ToWorld(T, B, plantNormal, localDir));
    pdf = max(localDir.z * INV_PI, 1e-6);
    float sampleScale = mix(0.42, 0.72, thinPlantFill());
    return thinPlantAlbedo(mat.albedo) * (localDir.z * INV_PI) * sampleScale;
}
vec3 calculateNormal(vec3 p0, vec3 p1, vec3 p2, vec2 uv0, vec2 uv1, vec2 uv2, vec3 matNormal, vec3 viewDir, out vec3 geometricNormal, bool openglNormal) {
    vec3 edge1 = p1 - p0;
    vec3 edge2 = p2 - p0;
    vec3 geoNormalObj = normalize(cross(edge1, edge2));

    mat3 normalMatrix = transpose(mat3(gl_WorldToObject3x4EXT));
    vec3 geometricNormalWorld = normalize(normalMatrix * geoNormalObj);
    geometricNormal = geometricNormalWorld;

    if (any(isnan(matNormal))) { return geometricNormalWorld; }

    // TBN
    vec2 deltaUV1 = uv1 - uv0;
    vec2 deltaUV2 = uv2 - uv0;
    float det = deltaUV1.x * deltaUV2.y - deltaUV2.x * deltaUV1.y;

    vec3 tangentObj;
    if (abs(det) < 1e-6) {
        tangentObj = (abs(geoNormalObj.x) > 0.99) ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    } else {
        float f = 1.0 / det;
        tangentObj.x = f * (deltaUV2.y * edge1.x - deltaUV1.y * edge2.x);
        tangentObj.y = f * (deltaUV2.y * edge1.y - deltaUV1.y * edge2.y);
        tangentObj.z = f * (deltaUV2.y * edge1.z - deltaUV1.y * edge2.z);
    }

    // Gram-Schmidt
    vec3 TObj = normalize(tangentObj - geoNormalObj * dot(geoNormalObj, tangentObj));
    vec3 BObj = cross(geoNormalObj, TObj);

    vec3 T = normalize(normalMatrix * TObj);
    vec3 B = normalize(normalMatrix * BObj);
    vec3 N = geometricNormalWorld;

    // LabPBR uses DirectX (Y-) convention; Blender PBR uses OpenGL (Y+) — no flip needed
    vec3 correctedLocalNormal = matNormal;
    if (!openglNormal) {
        correctedLocalNormal.y = -correctedLocalNormal.y;
    }

    vec3 finalNormal = normalize(T * correctedLocalNormal.x + B * correctedLocalNormal.y + N * correctedLocalNormal.z);

    // unseenable faces
    if (dot(viewDir, finalNormal) < 0.0)
        return geometricNormalWorld;
    else
        return finalNormal;
}

vec3 applyWaterNormalToBasis(vec3 matNormal, vec3 tangent, vec3 bitangent, vec3 geometricNormal, vec3 viewDir) {
    if (any(isnan(matNormal)) || any(isinf(matNormal))) { return geometricNormal; }

    vec3 correctedLocalNormal = matNormal;
    correctedLocalNormal.y = -correctedLocalNormal.y;

    vec3 normal = tangent * correctedLocalNormal.x + bitangent * correctedLocalNormal.y +
                  geometricNormal * correctedLocalNormal.z;
    if (dot(normal, normal) <= 1e-10) { return geometricNormal; }
    normal = normalize(normal);

    float NdotV = dot(normal, viewDir);
    if (NdotV >= 0.99999) { return normal; }

    vec3 edgeNormal = normal - viewDir * NdotV;
    float edgeNormalLen2 = dot(edgeNormal, edgeNormal);
    if (edgeNormalLen2 <= 1e-10) { return geometricNormal; }

    float weight = 1.0 - NdotV;
    weight = sin(min(weight, PI * 0.5));
    weight = clamp(min(max(NdotV, dot(viewDir, geometricNormal)), 1.0 - weight), 0.0, 1.0);

    float tangentWeight2 = max(1.0 - weight * weight, 0.0);
    if (tangentWeight2 <= 1e-10) { return geometricNormal; }

    return viewDir * weight + edgeNormal * inversesqrt(edgeNormalLen2 / tangentWeight2);
}

void main() {
    vec3 viewDir = -mainRay.direction;

    uint instanceID = gl_InstanceCustomIndexEXT;
    uint geometryID = gl_GeometryIndexEXT;

    uint blasOffset;
    uint vertexFormat;
    UnpackedVertex v0, v1, v2;
    fetchTriangleVertices(instanceID, geometryID, gl_PrimitiveID, v0, v1, v2, blasOffset, vertexFormat);

    vec3 baryCoords = vec3(1.0 - (attribs.x + attribs.y), attribs.x, attribs.y);
    // Compute world-space hit position from ray parameters instead of vertex buffer positions.
    // This correctly handles per-geometry transforms (mega-BLAS) where vertex buffer positions
    // are in chunk-local space but gl_ObjectToWorldEXT reflects the mega-chunk TLAS transform.
    vec3 worldPos = gl_WorldRayOriginEXT + gl_HitTEXT * gl_WorldRayDirectionEXT;
    vec3 planeHitWorldPos = worldPos;
    float actualHitT = gl_HitTEXT;

    // Biome tint: shader-side resolution from per-section SSBO (bits 12-13 of flags).
    // 0=none, 1=grass, 2=foliage, 3=water. Fixed tints use per-vertex colorLayer.
    vec3 colorLayer = vec3(1.0);
    uint biomeTintType = (v0.flags >> PBR_FLAG_BIOME_TINT_SHIFT) & 0x3u;
    if (biomeTintType != 0u) {
        uint packed = biomeColors.data[instanceID][biomeTintType - 1u];
        colorLayer = vec3(
            float((packed >> 16u) & 0xFFu) / 255.0,
            float((packed >> 8u) & 0xFFu) / 255.0,
            float(packed & 0xFFu) / 255.0
        );
    } else if ((v0.flags & PBR_FLAG_USE_COLOR_LAYER) != 0u) {
        // Fixed-color tint (birch/spruce leaves, etc.) - per-vertex colorLayer.
        colorLayer = baryCoords.x * v0.colorLayer + baryCoords.y * v1.colorLayer + baryCoords.z * v2.colorLayer;
    }

    bool useTexture = (v0.flags & PBR_FLAG_USE_TEXTURE) != 0u;
    float albedoEmission =
        baryCoords.x * v0.albedoEmission + baryCoords.y * v1.albedoEmission + baryCoords.z * v2.albedoEmission;
    uint textureID = v0.textureID;
    uint materialRuleTextureID = textureID;
    vec4 albedoValue;
    vec4 specularValue;
    vec4 normalValue;
    ivec4 flagValue;
    vec2 textureUV = vec2(0.0);
    vec2 uvMin = vec2(0.0);
    vec2 uvMax = vec2(1.0);
    vec3 rawAlbedoLinear = vec3(1.0);
    bool isBlockGeometry = (v0.flags & PBR_FLAG_BLOCK_GEOMETRY) != 0u;
    bool foldBlockOverlay = (v0.flags & PBR_FLAG_OVERLAY_ALPHA_MASK) != 0u;
    bool blockOverlayComposited = false;
    uint coordinateMode = (v0.flags & PBR_FLAG_COORD_MASK) >> PBR_FLAG_COORD_SHIFT;
    bool heightFieldProxyGeometry = isBlockGeometry && coordinateMode == 1u;
    int specularTextureID = -1;
    int normalTextureID = -1;
    int flagTextureID = -1;
    vec2 textureUVRaw = vec2(0.0);

    if (useTexture) {
        textureUV = baryCoords.x * v0.textureUV + baryCoords.y * v1.textureUV + baryCoords.z * v2.textureUV;
        textureUVRaw = textureUV;

        if (isBlockGeometry) {
            // === BLOCK GEOMETRY: material-id sampling via MaterialRegistry ===
            uint materialId = textureID;

            uvMin = vec2(0.0);
            uvMax = vec2(1.0);

            float lod = 0;
            albedoValue = fetchBlockAlbedoLod(materialId, textureUV, worldUbo.animTick, lod);

            rawAlbedoLinear = albedoValue.rgb;
            specularValue = fetchBlockSpecularLod(materialId, textureUV, lod);
            normalValue = fetchBlockNormalLod(materialId, textureUV, lod);
            flagValue = fetchBlockFlagLod(materialId, textureUV, lod);

            if (foldBlockOverlay) {
                blockOverlayComposited = applyBlockOverlayMaterialLod(
                    albedoValue, specularValue, normalValue, flagValue,
                    materialId, textureUV, worldUbo.animTick, lod, colorLayer,
                    materialRuleTextureID);
                rawAlbedoLinear = albedoValue.rgb;
            }
        } else {
            // === ENTITY GEOMETRY: legacy atlas sampling via textures[] ===
            specularTextureID = mapping.entries[textureID].specular;
            normalTextureID = mapping.entries[textureID].normal;
            flagTextureID = mapping.entries[textureID].flag;

            uvMin = min(min(v0.textureUV, v1.textureUV), v2.textureUV);
            uvMax = max(max(v0.textureUV, v1.textureUV), v2.textureUV);

            float lod = 0;
            albedoValue = textureLod(textures[nonuniformEXT(textureID)], textureUV, lod);
            rawAlbedoLinear = albedoValue.rgb;
            if (specularTextureID >= 0) {
                specularValue = textureLod(textures[nonuniformEXT(specularTextureID)], textureUV, lod);
            } else {
                specularValue = vec4(0.0);
            }
            if (normalTextureID >= 0) {
                normalValue = textureLod(textures[nonuniformEXT(normalTextureID)], textureUV, lod);
            } else {
                normalValue = vec4(0.0);
            }
            if (flagTextureID >= 0) {
                vec4 floatFlagValue = textureLod(textures[nonuniformEXT(flagTextureID)], textureUV, ceil(lod));
                flagValue = ivec4(round(floatFlagValue * 255.0));
            } else {
                flagValue = ivec4(0);
            }
        }
    } else {
        albedoValue = vec4(1.0);
        specularValue = vec4(0.0);
        normalValue = vec4(0.0);
        flagValue = ivec4(0);
    }

    uint packedBlockType = v0.emissiveBlockType;
    bool thinCutoutPlant = false;
#if RARSER_THIN_PLANT_PRIMARY_FIX
    thinCutoutPlant = isBlockGeometry && ((packedBlockType & PBR_PACKED_THIN_CUTOUT_PLANT) != 0u);
#endif
    uint sourceThinBlockType = thinCutoutPlant ? thinPlantBlockTypeFromPacked(packedBlockType) : 0u;
    uint sourceThinColumnHash = thinCutoutPlant ? thinPlantColumnHash(worldPos) : 0u;
    bool fluidGeometry = (v0.flags & PBR_FLAG_FLUID_GEOMETRY) != 0u;
    bool waterGeometry = (v0.flags & PBR_FLAG_WATER_GEOMETRY) != 0u;
#if RARSER_SHADER_DISPLACEMENT
    bool displacementGlobalEligible = pc.displacementDepthScale > DISPLACEMENT_MIN_DEPTH &&
                                      actualHitT < pc.displacementFadeDistanceBlocks;
#else
    bool displacementGlobalEligible = false;
#endif

#if RARSER_SHADER_DISPLACEMENT
    DisplacementSource displacementSource;
    displacementInitSource(displacementSource);
    bool hasDisplacedSurface = false;
    bool displacementUsesWorldOffset = DISPLACEMENT_WORLD_OFFSET_SCALE > 0.0;
    float displacedDepth = 0.0;
    vec2 displacedUV = textureUV;
    vec3 displacedBaseNormal = vec3(0.0, 1.0, 0.0);
    vec3 displacedGeometricNormal = vec3(0.0, 1.0, 0.0);
    vec3 displacedDpu = vec3(0.0);
    vec3 displacedDpv = vec3(0.0);
    vec3 displacedPlaneAtUv = planeHitWorldPos;

    bool displacementGeometryEligible =
        displacementGlobalEligible && useTexture && isBlockGeometry &&
        coordinateMode == 0u && !prGetIsHand(mainRay) && !fluidGeometry &&
        !thinCutoutPlant;
    bool canDisplace = displacementGeometryEligible && displacementBlockHasAuthoredHeight(textureID);

    if (canDisplace) {
        vec3 wp0 = vec3(gl_ObjectToWorldEXT * vec4(v0.pos, 1.0));
        vec3 wp1 = vec3(gl_ObjectToWorldEXT * vec4(v1.pos, 1.0));
        vec3 wp2 = vec3(gl_ObjectToWorldEXT * vec4(v2.pos, 1.0));
        vec2 displacementChartOffset = vec2(0.0);
        if (displacementPrepareAuthoredBlockSource(v0.flags, packedBlockType, textureID,
                                                   wp0, wp1, wp2,
                                                   v0.textureUV, v1.textureUV, v2.textureUV,
                                                   v0.postBase, worldUbo.animTick, actualHitT,
                                                   pc.displacementDepthScale,
                                                   pc.displacementFadeDistanceBlocks, viewDir,
                                                   true, false,
                                                   displacementSource, displacedDpu, displacedDpv,
                                                   displacedBaseNormal, displacementChartOffset)) {
            DisplacementHit displacementHit;
            int primarySteps = clamp(pc.displacementPrimarySteps, 1, 512);
            vec2 planeTextureUV = textureUVRaw + displacementChartOffset;
            if (displacementTracePrimary(displacementSource, planeTextureUV, planeHitWorldPos, gl_WorldRayDirectionEXT,
                                         viewDir, displacedDpu, displacedDpv, displacedBaseNormal,
                                         primarySteps, pc.displacementRefinementSteps, displacementHit)) {
                hasDisplacedSurface = true;
                textureUV = displacementWrappedBlockUV(displacementHit.uv);
                displacedUV = displacementHit.uv;
                displacedDepth = displacementHit.depth;
                vec3 planeAtDisplacedUv = planeHitWorldPos +
                    displacedDpu * (displacementHit.uv.x - planeTextureUV.x) +
                    displacedDpv * (displacementHit.uv.y - planeTextureUV.y);
                worldPos = displacementHit.worldPos;
                float rayDirLen2 = max(dot(gl_WorldRayDirectionEXT, gl_WorldRayDirectionEXT), 1e-8);
                actualHitT = max(dot(worldPos - gl_WorldRayOriginEXT, gl_WorldRayDirectionEXT) / rayDirLen2,
                                 gl_HitTEXT - DISPLACEMENT_TRACE_BIAS);
                displacedGeometricNormal = displacementHit.geometricNormal;
                displacedPlaneAtUv = planeAtDisplacedUv;

                float lod = 0.0;
                albedoValue = fetchBlockAlbedoLod(textureID, textureUV, worldUbo.animTick, lod);
                rawAlbedoLinear = albedoValue.rgb;
                specularValue = fetchBlockSpecularLod(textureID, textureUV, lod);
                normalValue = fetchBlockNormalLod(textureID, textureUV, lod);
                flagValue = fetchBlockFlagLod(textureID, textureUV, lod);
                materialRuleTextureID = textureID;
                blockOverlayComposited = false;

                if (displacementUsesWorldOffset && mainRay.index == 0u) {
                    uint64_t lastIndexBufferAddr = lastIndexBufferAddrs.addrs[blasOffset + geometryID];
                    uint64_t lastVertexBufferAddr = lastVertexBufferAddrs.addrs[blasOffset + geometryID];
                    if (lastIndexBufferAddr > 0 && lastVertexBufferAddr > 0) {
                        vec3 prevLocalPos;
                        fetchTrianglePositions(vertexFormat, lastVertexBufferAddr, lastIndexBufferAddr,
                                               gl_PrimitiveID, baryCoords, prevLocalPos);
                        mat4 lastModelMat = lastObjToWorldMats.mat[instanceID];
                        vec3 prevBaseNormal = displacementNormalize(mat3(lastModelMat) *
                            normalize(cross(v1.pos - v0.pos, v2.pos - v0.pos)), displacedBaseNormal);
                        if (dot(prevBaseNormal, displacedBaseNormal) < 0.0) prevBaseNormal = -prevBaseNormal;
                        mainRay.prevWorldPos = mat3(lastModelMat) * prevLocalPos + lastModelMat[3].xyz -
                                               prevBaseNormal * displacedDepth;
                        mainRay.hasPrevWorldPos = 1u;
                    }
                }
            }
        }
    }
#endif

    vec3 glint = vec3(0.0);
    vec4 overlayColor = vec4(0.0);
    if (!SIMPLIFIED_INDIRECT || mainRay.index <= 1) {
        if (!isBlockGeometry) {
            // Entity glint and overlay (blocks don't use these)
            float useGlint = float((v0.flags & PBR_FLAG_USE_GLINT) != 0u);
            uint glintTexture = v0.glintTexture;
            vec2 glintUV = baryCoords.x * v0.glintUV + baryCoords.y * v1.glintUV + baryCoords.z * v2.glintUV;
            glintUV = (worldUbo.textureMat * vec4(glintUV, 0.0, 1.0)).xy;
            glint = useGlint * texture(textures[nonuniformEXT(glintTexture)], glintUV).rgb;
            glint = glint * glint;

            bool useOverlay = (v0.flags & PBR_FLAG_USE_OVERLAY) != 0u;
            ivec2 overlayUV = ivec2(int(v0.overlayPacked & 0xFFFFu), int(v0.overlayPacked >> 16u));
            overlayColor = texelFetch(textures[nonuniformEXT(worldUbo.overlayTextureID)], overlayUV, 0);
        }
    }

    vec3 tint;
    if ((v0.flags & PBR_FLAG_USE_OVERLAY) != 0u && !isBlockGeometry) {
        tint = mix(overlayColor.rgb, albedoValue.rgb * colorLayer, overlayColor.a) + glint;
    } else if (isBlockGeometry && blockOverlayComposited) {
        tint = albedoValue.rgb + glint;
    } else {
        tint = albedoValue.rgb * colorLayer + glint;
    }

    tint = CS_BT709_TO_BT2020 * tint;  // BT.709 -> BT.2020 working space

    albedoValue = vec4(tint, albedoValue.a);
    if (isBlockGeometry && RT_DEBUG_MATERIAL_MODE != 0u) {
        albedoValue = vec4(materialDebugColor(materialRuleTextureID, textureUV, 0.0), 1.0);
        tint = albedoValue.rgb;
    }
    LabPBRMat mat = convertLabPBRMaterial(albedoValue, specularValue, normalValue);
    if (isBlockGeometry) {
        applyTextureRule(materialRuleTextureID, mat);
    }

    if (isBlockGeometry && !fluidGeometry && !waterGeometry &&
        mat.metallic < 0.5 && mat.transmission <= EPS) {
        mat.roughness = max(mat.roughness, 0.35);
        mat.coatWeight = min(mat.coatWeight, 0.15);
    }

    // The normal blue channel carries legacy LabPBR/raster AO for compatibility and audit.
    // Full path tracing gets occlusion from geometry and visibility, so AO does not darken throughput.
    if (!isBlockGeometry && normalTextureID < 0) mat.ao = 1.0;

    if (!isBlockGeometry && !ENTITY_NORMALS_ON) {
        mat.normal = vec3(0.0, 0.0, 1.0);
    }

    if (heightFieldProxyGeometry) {
        mat.normal = vec3(0.0, 0.0, 1.0);
        mat.ao = 1.0;
    }

    // the provided normal is unreliable! (such as grass, etc.)
    // calculate on the fly for now
    vec3 geometricNormal;
    vec3 normal =
        calculateNormal(v0.pos, v1.pos, v2.pos, v0.textureUV, v1.textureUV, v2.textureUV, mat.normal, viewDir, geometricNormal, false);
#if RARSER_SHADER_DISPLACEMENT
    if (hasDisplacedSurface) {
        geometricNormal = displacedGeometricNormal;
        normal = geometricNormal;
        mat.normal = vec3(0.0, 0.0, 1.0);
    }
#endif

    bool useRealisticWaterSurface = waterGeometry && shaderPackVisualSettings.waterSurfaceMode == 1 && abs(geometricNormal.y) > 0.75;
    if (waterGeometry) {
        vec3 waterTint = CS_BT709_TO_BT2020 * clamp(skyUBO.envWaterTintFog.rgb, vec3(0.02), vec3(1.0));
        albedoValue.rgb = waterTint;
        tint = waterTint;
        mat.f0 = vec3(0.02);
        mat.albedo = waterTint;
        mat.roughness = 0.01;
        mat.metallic = 0.0;
        mat.transmission = 1.0;
        mat.ior = 1.333;
        mat.refractionRoughness = 0.0;
        mat.coatWeight = 0.0;
        mat.coatMask = 0.0;

        if (useRealisticWaterSurface) {
            vec3 waterCoordNormal = geometricNormal.y >= 0.0 ? geometricNormal : -geometricNormal;
            vec3 absWorldPos = worldPos + vec3(worldUbo.cameraPos.xyz);
            vec2 waterCoord = fftWaterSurfaceCoord(absWorldPos, vec3(0.0), vec3(0.0), waterCoordNormal);
            FftWaterSample waterSample = sampleFftWater(waterCoord, worldUbo.gameTime);
            vec3 tangent, bitangent;
            fftWaterStableBasis(geometricNormal, tangent, bitangent);
            vec3 localWaterNormal = normalize(vec3(-waterSample.slope.x, waterSample.slope.y, 1.0));
            mat.roughness = clamp(0.005 + 0.012 * min(length(waterSample.slope), 0.45), 0.005, 0.022);
            normal = applyWaterNormalToBasis(localWaterNormal, tangent, bitangent, geometricNormal, viewDir);
            mat.normal = vec3(0.0, 0.0, 1.0);
        }
    }

#if RARSER_THIN_PLANT_PRIMARY_FIX
    if (thinCutoutPlant) {
        normal = vec3(0.0, 1.0, 0.0);
        mat.normal = vec3(0.0, 0.0, 1.0);
        mat.roughness = max(mat.roughness, 0.92);
        mat.metallic = 0.0;
        mat.transmission = 0.0;
        mat.f0 = vec3(0.04);
        mat.coatWeight = 0.0;
    }
#endif

    // --- Wet Surface Effect (Lagarde "Water Drop" model) ---
    // Rain wets upward-facing porous surfaces: roughness drops, albedo darkens, F0 → water IOR.
    float wetStrength = skyUBO.wetSurfaceStrength;
    if (skyUBO.rainGradient > 0.001 && wetStrength > 0.001
        && mat.metallic < 0.5 && mat.transmission < 0.5) {
        float rainExposure = max(dot(geometricNormal, vec3(0.0, 1.0, 0.0)), 0.0);
        float porosity = smoothstep(0.20, 0.85, mat.roughness);
        float wetness = clamp(skyUBO.rainGradient * rainExposure * porosity * wetStrength, 0.0, 1.0);
        if (wetness > 0.001) {
            float wf = (1.0 - wetness);
            mat.roughness = max(mat.roughness * wf * wf, 0.01);
            float darken = mix(1.0, 0.3, wetness * porosity);
            mat.albedo *= darken;
            albedoValue.rgb *= darken;
            mat.f0 = mix(mat.f0, vec3(0.02), wetness);
        }
    }

    // Write post-override material to mainRay for rgen G-buffer / DLSS-RR guide buffers
    mainRay.roughness = mat.roughness;
    mainRay.metallic = mat.metallic;
    mainRay.f0 = mat.f0;
    mainRay.emission = mat.emission;
    mainRay.subSurface = mat.subSurface;

    // Scene-referred emission normalization: 1.0 = 200 cd/m² (ITU-R BT.2408 paper white).
    // Physical nit values from vertex data are divided by this to keep radiance in a range
    // that fits RGBA16F after pre-exposure multiplication (max ~60 × preExp 32 = 1920).
    const float EMISSION_REFERENCE_NITS = 200.0;

    // add glowing radiance from emissive surfaces only.
    albedoEmission = max(albedoEmission, 0.0);

    // Apply per-block emission data from UBO (color override + scalar multiplier)
    // Extract emissive type from lower 8 bits of packed field
    uint emBlockType = packedBlockType & 0xFFu;
    prSetEmBlockType(mainRay, emBlockType);
    vec3 emissionTint = tint; // default: texture albedo (BT.2020)
    bool uniformGlow = false;
    if (emBlockType < 50u && albedoEmission > 0.0) {
        vec4 emData = worldUbo.emissionData[emBlockType];
        uniformGlow = (emData.a < 0.0); // negative multiplier = uniform glow flag
        albedoEmission *= abs(emData.a); // strip sign, apply magnitude
        if (dot(emData.rgb, emData.rgb) > 0.0001) {
            // Flame mask: only bright texture pixels (actual flame) get the spectral color.
            // Dark pixels (wood stick, stone base) keep their texture albedo.
            float texLum = dot(tint, vec3(0.2627, 0.6780, 0.0593)); // BT.2020 luminance
            float flameMask = smoothstep(0.15, 0.5, texLum);
            emissionTint = mix(tint, emData.rgb, flameMask);
        }
    }

    // Per-emissive-block gamut boost (Oklab chroma scaling on emission tint)
    if (emBlockType < 50u) {
        float emGamut = worldUbo.emissiveGamut[emBlockType >> 2u][emBlockType & 3u];
        if (emGamut != 1.0) {
            const mat3 bt2020_to_lms = mat3(
                0.6167557871, 0.2651330639, 0.1001026342,
                0.3601983994, 0.6358393640, 0.2039065193,
                0.0230458134, 0.0990275718, 0.6959908464);
            const mat3 lms_to_lab = mat3(
                0.2104542553,  1.9779984951,  0.0259040371,
                0.7936177850, -2.4285922050,  0.7827717662,
               -0.0040720468,  0.4505937099, -0.8086757660);
            const mat3 lab_to_lms = mat3(
                1.0,           1.0,           1.0,
                0.3963377774, -0.1055613458, -0.0894841775,
                0.2158037573, -0.0638541728, -1.2914855480);
            const mat3 lms_to_bt2020 = mat3(
                 2.1399067359, -0.8847358624, -0.0485737581,
                -1.2463895090,  2.1632309822, -0.4545031427,
                 0.1064827729, -0.2784951194,  1.5030769008);

            vec3 lms = bt2020_to_lms * max(emissionTint, vec3(0.0));
            vec3 lms_g = sign(lms) * pow(abs(lms), vec3(1.0 / 3.0));
            vec3 lab = lms_to_lab * lms_g;
            lab.yz *= emGamut;
            vec3 lms_g2 = lab_to_lms * lab;
            vec3 lms2 = clamp(lms_g2 * lms_g2 * lms_g2, vec3(-10.0), vec3(10.0));
            emissionTint = lms_to_bt2020 * lms2;
            float minC = min(emissionTint.r, min(emissionTint.g, emissionTint.b));
            if (minC < 0.0) {
                float luma = dot(emissionTint, vec3(0.2627, 0.6780, 0.0593));
                float t = luma / max(luma - minC, 1e-6);
                emissionTint = mix(vec3(luma), emissionTint, t);
            }
        }
    }

    // Emission in scene-referred units (1.0 = EMISSION_REFERENCE_NITS cd/m²).
    // Vertex nits (albedoEmission) are authoritative when set; LabPBR is fallback only.
    if (mat.emissionNits > 0.0) {
        emissionTint = mat.emissionTint;
    }
    float materialEmissionNits = mat.emissionNits > 0.0 ? mat.emissionNits : EMISSION_REFERENCE_NITS;
    float combinedEmission = (albedoEmission != 0.0) ? albedoEmission : (mat.emission * materialEmissionNits);
    float sceneEmission = combinedEmission / EMISSION_REFERENCE_NITS;

    // Texture-based emission mask: only bright texels (flame head) emit.
    // Dark texels (wood stick, stone base) get zero emission.
    // Uniform glow blocks skip this mask (emit from all texels equally).
    if (!uniformGlow) {
        float maskLum = dot(tint, vec3(0.2627, 0.6780, 0.0593));
        sceneEmission *= max(smoothstep(0.10, 0.40, maskLum), 0.05);
    }

    float factor = 1.0;
    // Emission radiance: emissionTint is BT.2020 (texture albedo or spectral flame color).
    vec3 emissionRadiance = factor * emissionTint * sceneEmission * mainRay.throughput;

    // Per-sample contribution clamping to prevent fireflies (scene-referred range)
    // Ground truth: no clamping — accumulation converges naturally
    if (!NO_EMISSION_CLAMP) {
        float emissionLum = luminanceBT2020(emissionRadiance);
        float maxContribution = 1000.0;
        if (emissionLum > maxContribution) {
            emissionRadiance *= maxContribution / emissionLum;
        }
    }

    mainRay.radiance += emissionRadiance;
    mainRay.albedoEmission = albedoEmission; // Store abs value for bloom

    mainRay.hitT = actualHitT;
    mainRay.coneWidth += actualHitT * mainRay.coneSpread;

    // Perfect specular surfaces have a Dirac delta BRDF — they redirect light,
    // they don't scatter it. Direct lighting appears through the reflection chain.
    bool isPerfectSpecular = (max(mat.roughness, mat.refractionRoughness * mat.transmission) < 0.001);

    if (!isPerfectSpecular && !RT_DEBUG_DISABLE_DIRECT_LIGHTING) {
    // shadow ray for direct lighting
    vec3 sunDir = normalize(skyUBO.sunDirection);
    vec3 lightDir = sunDir;
    // Softer sun sampling for hand = wider penumbras (500 vs 3000)
    // Ground truth: physical sun disk half-angle 0.267° → kappa ≈ 46000
    float kappa = PHYSICAL_SUN_DISK ? 46000.0 : ((prGetIsHand(mainRay)) ? 500.0 : 3000.0);
    if (sunDir.y < 0) { lightDir = normalize(skyUBO.moonDirection); }
    vec2 vmfBN = vec2(-1.0); // A/B test: force PCG, disable blue noise
    vec3 sampledLightDir = SampleVMF(mainRay.seed, lightDir, kappa, vmfBN);
    vec3 shadowBiasN = dot(sampledLightDir, geometricNormal) > 0.0 ? geometricNormal : -geometricNormal;
    vec3 shadowRayOrigin = thinCutoutPlant ? thinPlantRayOrigin(worldPos, sampledLightDir, geometricNormal)
                                           : offset_ray(worldPos, shadowBiasN);
    bool shadowLocalVisible = true;
#if RARSER_SHADER_DISPLACEMENT
    if (hasDisplacedSurface && displacementUsesWorldOffset) {
        DisplacementOutgoingRayStart shadowStart;
        displacementResolveOutgoingRayStart(displacementSource, displacedUV, displacedDepth,
                                            worldPos, displacedPlaneAtUv, sampledLightDir,
                                            displacedDpu, displacedDpv, displacedBaseNormal,
                                            geometricNormal, max(8, pc.displacementPrimarySteps / 2),
                                            shadowStart);
        shadowLocalVisible = shadowStart.visible;
        shadowRayOrigin = offset_ray(shadowStart.origin, shadowStart.biasNormal);
    }
#endif

    bool skipSecondarySunShadow = RT_DEBUG_DISABLE_SECONDARY_SUN_SHADOW && mainRay.index > 0;
    if (worldUbo.skyType == 1 && !skipSecondarySunShadow) {
        float pdf; // not used
            vec3 lightBRDF = thinCutoutPlant ? thinPlantDiffuseEval(mat, sampledLightDir)
                                         : SurfaceBSDFEval(mat, viewDir, normal, sampledLightDir, pdf, pc.flags);

        shadowRay.radiance = vec3(0.0);
        shadowRay.throughput = vec3(1.0);
        shadowRay.seed = mainRay.seed;
        shadowRay.hitT = INF_DISTANCE;
        shadowRay.insideBoat = prGetInsideBoat(mainRay) ? 1u : 0u;
        shadowRay.bounceIndex = mainRay.index;
        shadowRay.originThinBlockType = sourceThinBlockType;
        shadowRay.originThinColumnHash = sourceThinColumnHash;
        shadowRay.lastThinPlantHitKey = 0u;
        shadowRay.mediumAbsorption = vec3(0.0);
        shadowRay.mediumEntryT = 0.0;

        uint shadowMask = WORLD_MASK;
        if (!prGetIsHand(mainRay)) {
            shadowMask |= PLAYER_MASK | PLAYER_HEAD_MASK;  // world surfaces see player shadows; hand does not (prevents self-shadowing)
        }

        if (shadowLocalVisible) {
            traceRayEXT(topLevelAS, gl_RayFlagsNoneEXT,
                        shadowMask,
                        0,                                     // sbtRecordOffset
                        0,                                     // sbtRecordStride
                        2,                                     // missIndex
                        shadowRayOrigin, 0.0, sampledLightDir, 1000, 1);
        }

        // Add direct lighting contribution
        vec3 lightContribution = shadowRay.radiance;

        // Apply cloud shadowing (procedural volumetric slab).
        // This is evaluated at the shading point so it works for primary and reflected paths.
        if (shaderPackVisualSettings.cloudMode == 2 && shaderPackVisualSettings.volumetricCloudCastShadow != 0 &&
            !(RT_DEBUG_DISABLE_SECONDARY_CLOUD_SHADOW && mainRay.index > 0)) {
            float cloudT = cloudTransmittance(worldPos + sampledLightDir * 0.01, sampledLightDir, 0.0, 24000.0, worldUbo, skyUBO, 0);
            float shadowStrength = max(skyUBO.cloudLighting.x, 0.0) *
                                   mix(1.0, 0.55, clamp(shaderPackVisualSettings.volumetricCloudShadowSoftness, 0.0, 1.0));
            cloudT = pow(max(cloudT, 1e-6), shadowStrength);
            lightContribution *= cloudT;
        }

        float progress = skyUBO.rainGradient;
        vec3 lightRadiance = lightContribution * mainRay.throughput * lightBRDF;
        vec3 finalLightRadiance = mix(lightRadiance, vec3(0.0), progress);

        // Hand shadow smoothing: apply ambient floor with smooth falloff
        // Prevents harsh black transitions on hand geometry in shadow
        // Ground truth: no fake ambient — accumulation provides correct indirect fill
        if (prGetIsHand(mainRay) && !NO_HAND_AMBIENT) {
            float shadowLum = dot(finalLightRadiance, vec3(0.2627, 0.6780, 0.0593));
            float ambientFloor = 0.08 * dot(mainRay.throughput, vec3(0.2627, 0.6780, 0.0593));
            float blend = smoothstep(0.0, ambientFloor * 2.0, shadowLum);
            vec3 handAmbient = ambientFloor * tint * mainRay.throughput;
            finalLightRadiance = mix(handAmbient, finalLightRadiance, blend);
        }

        mainRay.radiance += finalLightRadiance;

        mainRay.directLightRadiance = finalLightRadiance;
        mainRay.directLightHitT = shadowRay.hitT;
    }

    } // !isPerfectSpecular

    mainRay.instanceIndex = instanceID;
    mainRay.geometryIndex = geometryID;
    mainRay.primitiveIndex = gl_PrimitiveID;
    mainRay.baryCoords = baryCoords;
    mainRay.worldPos = worldPos;
    mainRay.normal = normal;
    mainRay.albedoValue = albedoValue;

    // PSR mirror/glass continuation: if roughness is near-zero, trace deterministically
    // instead of stochastic BSDF sampling. The rgen PSR loop will continue tracing.
    if (max(mat.roughness, mat.refractionRoughness * mat.transmission) < 0.001) {
        if (mat.transmission > 0.0) {
            // PSR glass: deterministic Snell refraction for transmissive materials
            bool entering = dot(gl_WorldRayDirectionEXT, geometricNormal) < 0.0;
            float eta = entering ? (1.0 / mat.ior) : mat.ior;
            vec3 faceNormal = entering ? normal : -normal;
            vec3 refractDir = refract(gl_WorldRayDirectionEXT, faceNormal, eta);

            // Fresnel reflectance (Schlick) to determine reflect vs refract
            float cosTheta = max(dot(viewDir, faceNormal), 0.0);
            float r0 = ((1.0 - mat.ior) / (1.0 + mat.ior));
            r0 = r0 * r0;
            float fresnelReflect = r0 + (1.0 - r0) * pow(1.0 - cosTheta, 5.0);
            // Blend reflection amount by (1 - transmission): full transmission → mostly refract
            fresnelReflect *= (1.0 - mat.transmission);

            if (length(refractDir) < 0.001 || fresnelReflect > 0.999) {
                // Total internal reflection fallback
                vec3 reflectDir = reflect(gl_WorldRayDirectionEXT, faceNormal);
                vec3 bounceOffsetN = dot(reflectDir, geometricNormal) > 0.0 ? geometricNormal : -geometricNormal;
                vec3 bounceWorldPos = worldPos;
#if RARSER_SHADER_DISPLACEMENT
                if (hasDisplacedSurface && displacementUsesWorldOffset) {
                    DisplacementOutgoingRayStart bounceStart;
                    if (displacementResolveOutgoingRayStart(displacementSource, displacedUV, displacedDepth,
                                                            worldPos, displacedPlaneAtUv, reflectDir,
                                                            displacedDpu, displacedDpv, displacedBaseNormal,
                                                            geometricNormal, max(8, pc.displacementPrimarySteps / 2),
                                                            bounceStart)) {
                        bounceWorldPos = bounceStart.origin;
                        bounceOffsetN = bounceStart.biasNormal;
                    } else {
                        mainRay.flags |= PR_STOP_BIT;
                        return;
                    }
                }
#endif
                mainRay.origin = offset_ray(bounceWorldPos, bounceOffsetN);
                mainRay.direction = reflectDir;
                mainRay.throughput *= mat.f0;
                prSetLobeType(mainRay, 1u); // specular
            } else {
                // Refraction path
                vec3 bounceOffsetN = dot(refractDir, geometricNormal) > 0.0 ? geometricNormal : -geometricNormal;
                vec3 bounceWorldPos = worldPos;
#if RARSER_SHADER_DISPLACEMENT
                if (hasDisplacedSurface && displacementUsesWorldOffset) {
                    DisplacementOutgoingRayStart bounceStart;
                    if (displacementResolveOutgoingRayStart(displacementSource, displacedUV, displacedDepth,
                                                            worldPos, displacedPlaneAtUv, refractDir,
                                                            displacedDpu, displacedDpv, displacedBaseNormal,
                                                            geometricNormal, max(8, pc.displacementPrimarySteps / 2),
                                                            bounceStart)) {
                        bounceWorldPos = bounceStart.origin;
                        bounceOffsetN = bounceStart.biasNormal;
                    } else {
                        mainRay.flags |= PR_STOP_BIT;
                        return;
                    }
                }
#endif
                mainRay.origin = offset_ray(bounceWorldPos, bounceOffsetN);
                mainRay.direction = refractDir;
                bool thinGlass = (mat.materialModeFlags & 0x3u) == 1u;
                float absorptionPath = thinGlass ? mat.thickness : mat.thickness * max(mat.absorptionDistance, 0.01);
                vec3 beerTint = exp(-max(mat.absorption, vec3(0.0)) * absorptionPath);
                mainRay.throughput *= (1.0 - fresnelReflect) * beerTint;
                prSetLobeType(mainRay, 2u); // transmission
            }
            // Clear noisy and stop (PSR continuation)
            mainRay.flags &= ~(PR_NOISY_BIT | PR_STOP_BIT);
            return;
        } else {
            // PSR mirror: opaque specular reflection
            vec3 reflectDir = reflect(gl_WorldRayDirectionEXT, normal);
            vec3 bounceOffsetN = dot(reflectDir, geometricNormal) > 0.0 ? geometricNormal : -geometricNormal;
            vec3 bounceWorldPos = worldPos;
#if RARSER_SHADER_DISPLACEMENT
            if (hasDisplacedSurface && displacementUsesWorldOffset) {
                DisplacementOutgoingRayStart bounceStart;
                if (displacementResolveOutgoingRayStart(displacementSource, displacedUV, displacedDepth,
                                                        worldPos, displacedPlaneAtUv, reflectDir,
                                                        displacedDpu, displacedDpv, displacedBaseNormal,
                                                        geometricNormal, max(8, pc.displacementPrimarySteps / 2),
                                                        bounceStart)) {
                    bounceWorldPos = bounceStart.origin;
                    bounceOffsetN = bounceStart.biasNormal;
                } else {
                    mainRay.flags |= PR_STOP_BIT;
                    return;
                }
            }
#endif
            mainRay.origin = offset_ray(bounceWorldPos, bounceOffsetN);
            mainRay.direction = reflectDir;

            // Fresnel reflectance (Schlick) — preserves mirror tint for metals
            float cosTheta = max(dot(viewDir, normal), 0.0);
            vec3 fresnel = mat.f0 + (1.0 - mat.f0) * pow(1.0 - cosTheta, 5.0);
            mainRay.throughput *= fresnel;

            // Clear noisy and stop, set lobeType=specular
            mainRay.flags &= ~(PR_NOISY_BIT | PR_STOP_BIT);
            prSetLobeType(mainRay, 1u);
            return;
        }
    }

    // sample next direction using Disney BSDF
    // First bounce: Owen-scrambled Sobol for lobe selection (dim 76) + GGX VNDF (dims 77-78)
    vec3 sampleDir;
    float pdf;
    uint lobeType;
    vec3 bsdfXi = vec3(-1.0); // A/B: PCG only
    vec3 bsdf;
    if (thinCutoutPlant) {
        bsdf = thinPlantDiffuseSample(mat, sampleDir, pdf, mainRay.seed);
        lobeType = 0u;
    } else {
        bsdf = SurfaceBSDFSample(mat, viewDir, normal, sampleDir, pdf, mainRay.seed, lobeType, pc.flags, bsdfXi);
    }

    prSetLobeType(mainRay, lobeType);
    mainRay.flags |= PR_NOISY_BIT;

    // early exit if sampling failed (check BEFORE updating throughput to avoid amplification)
    if (pdf <= 1e-6) {
        mainRay.flags |= PR_STOP_BIT;
        return;
    }

    // Prevent light leaks: perturbed normals can sample directions below the
    // geometric surface plane. Kill these for non-transmissive lobes (diffuse/specular).
    // Transmission (lobeType 2: glass/water) still needs through-surface rays.
    if (!thinCutoutPlant && lobeType != 2 && dot(sampleDir, geometricNormal) <= 0.0) {
        mainRay.flags |= PR_STOP_BIT;
        return;
    }

    mainRay.throughput *= bsdf / max(pdf, 1e-4);

    vec3 bounceOffsetN = dot(sampleDir, geometricNormal) > 0.0 ? geometricNormal : -geometricNormal;
    vec3 bounceWorldPos = worldPos;
#if RARSER_SHADER_DISPLACEMENT
    if (hasDisplacedSurface && displacementUsesWorldOffset) {
        DisplacementOutgoingRayStart bounceStart;
        if (displacementResolveOutgoingRayStart(displacementSource, displacedUV, displacedDepth,
                                                worldPos, displacedPlaneAtUv, sampleDir,
                                                displacedDpu, displacedDpv, displacedBaseNormal,
                                                geometricNormal, max(8, pc.displacementPrimarySteps / 2),
                                                bounceStart)) {
            bounceWorldPos = bounceStart.origin;
            bounceOffsetN = bounceStart.biasNormal;
        } else {
            mainRay.flags |= PR_STOP_BIT;
            return;
        }
    }
#endif
    mainRay.origin = thinCutoutPlant ? thinPlantRayOrigin(bounceWorldPos, sampleDir, geometricNormal)
                                     : offset_ray(bounceWorldPos, bounceOffsetN);

    mainRay.direction = sampleDir;
    mainRay.flags &= ~PR_STOP_BIT;
}
