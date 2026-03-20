#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#include "../util/disney.glsl"
#include "../util/random.glsl"
#include "../util/ray_payloads.glsl"
#include "../util/util.glsl"
#include "../util/area_light.glsl"
#include "../util/ray_offset.glsl"
#include "../util/colorspace.glsl"

layout(set = 0, binding = 0) uniform sampler2D textures[];

#include "../util/clouds.glsl"

layout(set = 1, binding = 0) uniform accelerationStructureEXT topLevelAS;

layout(set = 1, binding = 7) readonly buffer TextureMappingBuffer {
    TextureMapping mapping;
};

#include "../util/material_properties.glsl"

layout(set = 1, binding = 8) readonly buffer AreaLightBuffer {
    AreaLight lights[];
} areaLightBuffer;

layout(set = 1, binding = 12) readonly buffer DisplacedFaceDataBuffer {
    DisplacedFaceData faces[];
} displacedFaces;

layout(set = 2, binding = 0) uniform WorldUniform {
    WorldUBO worldUbo;
};

layout(set = 2, binding = 1) uniform LastWorldUniform {
    WorldUBO lastWorldUbo;
};

layout(set = 2, binding = 2) uniform SkyUniform {
    SkyUBO skyUBO;
};

layout(push_constant) uniform PushConstant {
    int numRayBounces;
    int flags;
    int areaLightCount;
    float shadowSoftness;
    int risCandidates;
    int temporalMClamp;
    int wClamp;
    float preExposure;
    float pomHeightScale;
    int   pomSteps;
    int   pomRefinement;
    float pomFadeDistance;
    float colorExpansion;
    uint _pad0;
    uint64_t _sharcBDA0;
    uint64_t _sharcBDA1;
    uint64_t _sharcBDA2;
    float _sharcPad0, _sharcPad1, _sharcPad2, _sharcPad3;
    uint _sharcPad4;
    float _sharcPad5;
    uint _sharcPad6;
    float _sharcPad7;
    int _sharcPad8;
    int _sharcPad9;
    int offlineFlags;
    int accumFrameCount;
    float aperture;
    float focalDistance;
    uint64_t materialClassAddr;
} pc;
#define SIMPLIFIED_INDIRECT ((pc.flags & 1) != 0)
#define AREA_LIGHTS_ON ((pc.flags & 2) != 0)
#define PHYSICAL_SUN_DISK  ((pc.flags & 2048) != 0)
#define NO_HAND_AMBIENT    ((pc.flags & 4096) != 0)
#define NO_EMISSION_CLAMP  ((pc.flags & 1024) != 0)

layout(location = 0) rayPayloadInEXT PrimaryRay mainRay;
layout(location = 1) rayPayloadEXT ShadowRay shadowRay;
hitAttributeEXT vec2 faceUV;

// Bilinear height sampling for normal computation
float sampleHeightBilinear(vec2 uv, int texID) {
    vec2 texSize = vec2(textureSize(textures[nonuniformEXT(texID)], 0));
    vec2 texelCoord = uv * texSize - 0.5;
    vec2 f = fract(texelCoord);
    ivec2 i = ivec2(floor(texelCoord));
    ivec2 size = ivec2(texSize);
    ivec2 i00 = clamp(i, ivec2(0), size - 1);
    ivec2 i10 = clamp(i + ivec2(1, 0), ivec2(0), size - 1);
    ivec2 i01 = clamp(i + ivec2(0, 1), ivec2(0), size - 1);
    ivec2 i11 = clamp(i + ivec2(1, 1), ivec2(0), size - 1);
    float h00 = texelFetch(textures[nonuniformEXT(texID)], i00, 0).w;
    float h10 = texelFetch(textures[nonuniformEXT(texID)], i10, 0).w;
    float h01 = texelFetch(textures[nonuniformEXT(texID)], i01, 0).w;
    float h11 = texelFetch(textures[nonuniformEXT(texID)], i11, 0).w;
    return mix(mix(h00, h10, f.x), mix(h01, h11, f.x), f.y);
}

void main() {
    vec3 viewDir = -mainRay.direction;
    DisplacedFaceData face = displacedFaces.faces[gl_PrimitiveID];

    // Compute atlas UV from face UV hit attribute
    vec2 atlasUV = mix(face.uvMin, face.uvMax, faceUV);

    // Compute world position from face data
    vec3 localPos = face.corner + faceUV.x * face.edgeU + faceUV.y * face.edgeV;

    // Displace along face normal by sampled height
    vec3 axisNormals[6] = vec3[6](
        vec3(1, 0, 0), vec3(-1, 0, 0),
        vec3(0, 1, 0), vec3(0, -1, 0),
        vec3(0, 0, 1), vec3(0, 0, -1)
    );
    vec3 faceNormal = axisNormals[face.faceAxis];

    float height = 1.0;
    if (face.normalTexID >= 0) {
        height = sampleHeightBilinear(atlasUV, face.normalTexID);
    }
    localPos -= faceNormal * (1.0 - height) * face.heightScale;

    vec3 worldPos = vec4(localPos, 1.0) * gl_ObjectToWorld3x4EXT;

    // Compute normal from height gradient (central differences)
    vec3 normal = faceNormal;
    if (face.normalTexID >= 0) {
        vec2 tileSize = face.uvMax - face.uvMin;
        vec2 texelSize = 1.0 / vec2(textureSize(textures[nonuniformEXT(face.normalTexID)], 0));
        vec2 eps = max(texelSize, tileSize * 0.02);

        float hL = sampleHeightBilinear(clamp(atlasUV - vec2(eps.x, 0), face.uvMin, face.uvMax), face.normalTexID);
        float hR = sampleHeightBilinear(clamp(atlasUV + vec2(eps.x, 0), face.uvMin, face.uvMax), face.normalTexID);
        float hD = sampleHeightBilinear(clamp(atlasUV - vec2(0, eps.y), face.uvMin, face.uvMax), face.normalTexID);
        float hU = sampleHeightBilinear(clamp(atlasUV + vec2(0, eps.y), face.uvMin, face.uvMax), face.normalTexID);

        float dhdU = (hR - hL) / (2.0 * eps.x / tileSize.x);
        float dhdV = (hU - hD) / (2.0 * eps.y / tileSize.y);

        // Face tangent vectors in world space
        vec3 edgeUNorm = normalize(face.edgeU);
        vec3 edgeVNorm = normalize(face.edgeV);
        mat3 normalMatrix = transpose(mat3(gl_WorldToObject3x4EXT));
        vec3 T = normalize(normalMatrix * edgeUNorm);
        vec3 B = normalize(normalMatrix * edgeVNorm);
        vec3 N = normalize(normalMatrix * faceNormal);

        // Height gradient to normal perturbation
        vec3 perturbedNormal = normalize(N + face.heightScale * (dhdU * T + dhdV * B));

        // Also apply normal map texture (LabPBR XY)
        vec4 normalValue = textureLod(textures[nonuniformEXT(face.normalTexID)], atlasUV, 0);
        vec3 matNormal = normalValue.xyz * 2.0 - 1.0;
        matNormal.y = -matNormal.y; // LabPBR DirectX convention
        if (length(matNormal.xy) > 0.01) {
            matNormal.z = sqrt(max(1.0 - dot(matNormal.xy, matNormal.xy), 0.0));
            normal = normalize(T * matNormal.x + B * matNormal.y + perturbedNormal * matNormal.z);
        } else {
            normal = perturbedNormal;
        }

        if (dot(viewDir, normal) < 0.0) normal = perturbedNormal;
    }
    vec3 geometricNormal = normalize(transpose(mat3(gl_WorldToObject3x4EXT)) * faceNormal);

    // Sample textures
    vec4 albedoValue = textureLod(textures[nonuniformEXT(face.textureID)], atlasUV, 0);
    vec4 specularValue = vec4(0.0);
    vec4 normalValue = vec4(0.0);
    if (face.specularTexID >= 0) {
        specularValue = textureLod(textures[nonuniformEXT(face.specularTexID)], atlasUV, 0);
    }
    if (face.normalTexID >= 0) {
        normalValue = textureLod(textures[nonuniformEXT(face.normalTexID)], atlasUV, 0);
    }

    // Apply color tint
    vec3 tint = albedoValue.rgb * face.colorLayer.rgb;
    tint = CS_BT709_TO_BT2020 * tint;
    albedoValue = vec4(tint, albedoValue.a);

    // Convert LabPBR material
    LabPBRMat mat = convertLabPBRMaterial(albedoValue, specularValue, normalValue);

    // Material class override via BDA (same as main CHS)
    uint packedBlockType = face.emissiveBlockType;
    uint materialType = (packedBlockType >> 8u) & 0xFFu;
    uint materialClassIdx = 0u;
    bool hasMaterialClass = false;
    if (materialType > 0u) {
        materialClassIdx = materialType - 1u;
        hasMaterialClass = true;
    } else {
        int maskTexID = mapping.entries[face.textureID].maskTexture;
        if (maskTexID >= 0) {
            materialClassIdx = uint(texture(textures[nonuniformEXT(maskTexID)], atlasUV).r * 255.0 + 0.5);
            hasMaterialClass = (materialClassIdx < 160u);
        }
    }

    if (hasMaterialClass && pc.materialClassAddr != 0) {
        MaterialClassBufferRef matBuf = MaterialClassBufferRef(pc.materialClassAddr);
        MaterialClassEntry mc = matBuf.materialClassMapping.entries[materialClassIdx];
        if (dot(mc.f0, mc.f0) > 0.0001) mat.f0 = mc.f0;
        else if (mc.metallic < 0.5) {
            float ior = max(mc.ior, 1.0);
            float f0 = ((ior - 1.0) * (ior - 1.0)) / ((ior + 1.0) * (ior + 1.0));
            mat.f0 = vec3(f0);
        }
        mat.roughness = max(mc.roughness * mc.roughness, 0.01);
        mat.metallic = mc.metallic;
        if (mc.transmission >= 0.0) mat.transmission = mc.transmission;
        mat.ior = max(mc.ior, 1.0);
        mat.subSurface = mc.subsurface;
        if (mat.metallic > 0.5) mat.albedo = mat.f0;
    }

    // Write post-override material
    mainRay.roughness = mat.roughness;
    mainRay.metallic = mat.metallic;
    mainRay.f0 = mat.f0;
    mainRay.emission = mat.emission;
    mainRay.subSurface = mat.subSurface;

    const float EMISSION_REFERENCE_NITS = 200.0;
    uint emBlockType = packedBlockType & 0xFFu;
    mainRay.emBlockTypeOut = emBlockType;
    float albedoEmission = 0.0; // Displaced blocks use LabPBR emission
    float sceneEmission = mat.emission;
    vec3 emissionTint = tint;
    vec3 emissionRadiance = sceneEmission * emissionTint * mainRay.throughput;

    if (!NO_EMISSION_CLAMP) {
        float emLum = luminanceBT2020(emissionRadiance);
        if (emLum > 1000.0) emissionRadiance *= 1000.0 / emLum;
    }

    mainRay.radiance += emissionRadiance;
    mainRay.albedoEmission = albedoEmission;
    mainRay.hitT = gl_HitTEXT;
    mainRay.coneWidth += gl_HitTEXT * mainRay.coneSpread;

    bool isPerfectSpecular = (mat.roughness < 0.001);

    if (!isPerfectSpecular) {
        // Shadow ray for direct lighting
        vec3 sunDir = normalize(skyUBO.sunDirection);
        vec3 lightDir = sunDir;
        float kappa = PHYSICAL_SUN_DISK ? 46000.0 : 3000.0;
        if (sunDir.y < 0) lightDir = normalize(skyUBO.moonDirection);
        vec3 sampledLightDir = SampleVMF(mainRay.seed, lightDir, kappa);
        vec3 shadowBiasN = dot(sampledLightDir, geometricNormal) > 0.0 ? geometricNormal : -geometricNormal;
        vec3 shadowRayOrigin = offset_ray(worldPos, shadowBiasN);

        if (worldUbo.skyType == 1) {
            float pdf;
            vec3 lightBRDF = DisneyEval(mat, viewDir, normal, sampledLightDir, pdf, pc.flags);

            shadowRay.radiance = vec3(0.0);
            shadowRay.throughput = vec3(1.0);
            shadowRay.seed = mainRay.seed;
            shadowRay.hitT = INF_DISTANCE;
            shadowRay.insideBoat = mainRay.insideBoat;
            shadowRay.bounceIndex = mainRay.index;
            shadowRay.mediumAbsorption = vec3(0.0);
            shadowRay.mediumEntryT = 0.0;

            traceRayEXT(topLevelAS, gl_RayFlagsNoneEXT,
                        WORLD_MASK | PLAYER_MASK | PLAYER_HEAD_MASK,
                        0, 0, 2,
                        shadowRayOrigin, 0.0, sampledLightDir, 1000, 1);

            vec3 lightContribution = shadowRay.radiance;
            float cloudT = cloudTransmittance(worldPos + sampledLightDir * 0.01, sampledLightDir, 0.0, 1000.0, worldUbo, skyUBO, 0);
            float shadowStr = max(skyUBO.cloudLighting.x, 0.0);
            cloudT = pow(max(cloudT, 1e-6), shadowStr);
            lightContribution *= cloudT;

            float progress = skyUBO.rainGradient;
            vec3 finalLightRadiance = mix(lightContribution * mainRay.throughput * lightBRDF, vec3(0.0), progress);
            mainRay.radiance += finalLightRadiance;
            mainRay.directLightRadiance = finalLightRadiance;
            mainRay.directLightHitT = shadowRay.hitT;
        }
    } else {
        mainRay.directLightRadiance = vec3(0.0);
        mainRay.directLightHitT = INF_DISTANCE;
    }

    // Bounce: sample Disney BRDF for next ray direction
    mainRay.worldPos = worldPos;
    mainRay.normal = normal;
    mainRay.instanceIndex = gl_InstanceCustomIndexEXT;
    mainRay.geometryIndex = gl_GeometryIndexEXT;
    mainRay.primitiveIndex = gl_PrimitiveID;
    mainRay.baryCoords = vec3(faceUV, 0.0);
    mainRay.albedoValue = albedoValue;
    mainRay.specularValue = specularValue;
    mainRay.normalValue = normalValue;
    mainRay.flagValue = ivec4(0);
    mainRay.noisy = 1;

    if (mainRay.index >= uint(pc.numRayBounces)) {
        mainRay.stop = 1;
        return;
    }

    float pdf;
    vec3 nextDir;
    vec3 brdfValue = DisneySample(mat, viewDir, normal, nextDir, pdf, mainRay.seed, mainRay.lobeType, pc.flags);
    if (pdf < 1e-7 || any(isnan(nextDir))) {
        mainRay.stop = 1;
        return;
    }

    mainRay.throughput *= brdfValue / pdf;

    // Russian roulette
    float maxT = max(mainRay.throughput.r, max(mainRay.throughput.g, mainRay.throughput.b));
    if (maxT < 0.01 && mainRay.index > 2u) {
        float q = max(0.05, 1.0 - maxT);
        if (rand(mainRay.seed) < q) {
            mainRay.stop = 1;
            return;
        }
        mainRay.throughput /= (1.0 - q);
    }

    // Set up next ray
    vec3 biasNormal = dot(nextDir, geometricNormal) > 0.0 ? geometricNormal : -geometricNormal;
    mainRay.origin = offset_ray(worldPos, biasNormal);
    mainRay.direction = nextDir;
    mainRay.stop = 0;
    mainRay.cont = 1;
}
