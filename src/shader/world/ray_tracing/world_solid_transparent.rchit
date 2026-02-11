#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#include "../util/disney.glsl"
#include "../util/random.glsl"
#include "../util/ray_cone.glsl"
#include "../util/ray_payloads.glsl"
#include "../util/util.glsl"
#include "common/shared.hpp"

layout(set = 0, binding = 0) uniform sampler2D textures[];

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

layout(set = 2, binding = 0) uniform WorldUniform {
    WorldUBO worldUbo;
};

layout(set = 2, binding = 1) uniform LastWorldUniform {
    WorldUBO lastWorldUbo;
};

layout(set = 2, binding = 2) uniform SkyUniform {
    SkyUBO skyUBO;
};

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer VertexBuffer {
    PBRTriangle vertices[];
}
vertexBuffer;

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer IndexBuffer {
    uint indices[];
}
indexBuffer;

layout(location = 0) rayPayloadInEXT PrimaryRay mainRay;
layout(location = 1) rayPayloadEXT ShadowRay shadowRay;
hitAttributeEXT vec2 attribs;

vec3 calculateNormal(vec3 p0, vec3 p1, vec3 p2, vec2 uv0, vec2 uv1, vec2 uv2, vec3 matNormal, vec3 viewDir) {
    vec3 edge1 = p1 - p0;
    vec3 edge2 = p2 - p0;
    vec3 geoNormalObj = normalize(cross(edge1, edge2));

    mat3 normalMatrix = transpose(mat3(gl_WorldToObject3x4EXT));
    vec3 geometricNormalWorld = normalize(normalMatrix * geoNormalObj);

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

    // LabPBR / DirectX (Y-)
    vec3 correctedLocalNormal = matNormal;
    correctedLocalNormal.y = -correctedLocalNormal.y;

    vec3 finalNormal = normalize(T * correctedLocalNormal.x + B * correctedLocalNormal.y + N * correctedLocalNormal.z);

    // unseenable faces
    if (dot(viewDir, finalNormal) < 0.0)
        return geometricNormalWorld;
    else
        return finalNormal;
}

void main() {
    vec3 viewDir = -mainRay.direction;

    uint instanceID = gl_InstanceCustomIndexEXT;
    uint geometryID = gl_GeometryIndexEXT;

    uint blasOffset = blasOffsets.offsets[instanceID];

    IndexBuffer indexBuffer = IndexBuffer(indexBufferAddrs.addrs[blasOffset + geometryID]);
    uint indexBaseID = 3 * gl_PrimitiveID;
    uint i0 = indexBuffer.indices[indexBaseID];
    uint i1 = indexBuffer.indices[indexBaseID + 1];
    uint i2 = indexBuffer.indices[indexBaseID + 2];

    VertexBuffer vertexBuffer = VertexBuffer(vertexBufferAddrs.addrs[blasOffset + geometryID]);
    PBRTriangle v0 = vertexBuffer.vertices[i0];
    PBRTriangle v1 = vertexBuffer.vertices[i1];
    PBRTriangle v2 = vertexBuffer.vertices[i2];

    vec3 baryCoords = vec3(1.0 - (attribs.x + attribs.y), attribs.x, attribs.y);
    vec3 localPos = baryCoords.x * v0.pos + baryCoords.y * v1.pos + baryCoords.z * v2.pos;
    vec3 worldPos = vec4(localPos, 1.0) * gl_ObjectToWorld3x4EXT;

    uint useColorLayer = v0.useColorLayer;
    vec3 colorLayer;
    if (useColorLayer > 0) {
        colorLayer = (baryCoords.x * v0.colorLayer + baryCoords.y * v1.colorLayer + baryCoords.z * v2.colorLayer).rgb;
    } else {
        colorLayer = vec3(1.0);
    }

    uint useTexture = v0.useTexture;
    float albedoEmission =
        baryCoords.x * v0.albedoEmission + baryCoords.y * v1.albedoEmission + baryCoords.z * v2.albedoEmission;
    uint textureID = v0.textureID;
    int specularTextureID = mapping.entries[textureID].specular;
    int normalTextureID = mapping.entries[textureID].normal;
    int flagTextureID = mapping.entries[textureID].flag;
    vec4 albedoValue;
    vec4 specularValue;
    vec4 normalValue;
    ivec4 flagValue;
    vec2 textureUV;
    if (useTexture > 0) {
        textureUV = baryCoords.x * v0.textureUV + baryCoords.y * v1.textureUV + baryCoords.z * v2.textureUV;

        // ray cone
        float coneRadiusWorld = mainRay.coneWidth + gl_HitTEXT * mainRay.coneSpread;
        vec3 dposdu, dposdv;
        computedposduDv(v0.pos, v1.pos, v2.pos, v0.textureUV, v1.textureUV, v2.textureUV, dposdu, dposdv);
        // lod still has issues, temporally disable
        float lod = 0; // lodWithCone(textures[nonuniformEXT(textureID)], textureUV, coneRadiusWorld, dposdu, dposdv);

        albedoValue = textureLod(textures[nonuniformEXT(textureID)], textureUV, lod);
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
    } else {
        albedoValue = vec4(1.0);
        specularValue = vec4(0.0);
        normalValue = vec4(0.0);
        flagValue = ivec4(0);
    }

    uint useGlint = v0.useGlint;
    uint glintTexture = v0.glintTexture;
    vec2 glintUV = baryCoords.x * v0.glintUV + baryCoords.y * v1.glintUV + baryCoords.z * v2.glintUV;
    glintUV = (worldUbo.textureMat * vec4(glintUV, 0.0, 1.0)).xy;
    vec3 glint = useGlint * texture(textures[nonuniformEXT(glintTexture)], glintUV).rgb;
    glint = glint * glint;

    uint useOverlay = v0.useOverlay;
    ivec2 overlayUV = v0.overlayUV;
    vec4 overlayColor = texelFetch(textures[nonuniformEXT(worldUbo.overlayTextureID)], overlayUV, 0);

    vec3 tint;
    if (useOverlay > 0) {
        tint = mix(overlayColor.rgb, albedoValue.rgb * colorLayer, overlayColor.a) + glint;
    } else {
        tint = albedoValue.rgb * colorLayer + glint;
    }

    albedoValue = vec4(tint, albedoValue.a);
    LabPBRMat mat = convertLabPBRMaterial(albedoValue, specularValue, normalValue);

    // the provided normal is unreliable! (such as grass, etc.)
    // calculate on the fly for now
    vec3 normal =
        calculateNormal(v0.pos, v1.pos, v2.pos, v0.textureUV, v1.textureUV, v2.textureUV, mat.normal, viewDir);

    // add glowing radiance
    float factor = mainRay.index == 0 ? 1.0 : 16.0;
    vec3 emissionRadiance = factor * tint * mat.emission * mainRay.throughput;
    emissionRadiance += tint * albedoEmission * mainRay.throughput;
    mainRay.radiance += emissionRadiance;

    mainRay.hitT = gl_HitTEXT;
    mainRay.coneWidth += gl_HitTEXT * mainRay.coneSpread;

    // shadow ray for direct lighting
    vec3 lightDir = normalize(skyUBO.sunDirection);
    // Softer sun sampling for hand = wider penumbras (500 vs 3000)
    float kappa = (mainRay.isHand > 0) ? 500.0 : 3000.0;
    if (lightDir.y < 0) { lightDir = -lightDir; }
    vec3 sampledLightDir = SampleVMF(mainRay.seed, lightDir, kappa);
    vec3 shadowRayOrigin = worldPos + (dot(sampledLightDir, normal) > 0.0 ? normal : -normal) * 0.001;

    if (worldUbo.skyType == 1) {
        float pdf; // not used
        vec3 lightBRDF = DisneyEval(mat, viewDir, normal, sampledLightDir, pdf);

        shadowRay.radiance = vec3(0.0);
        shadowRay.throughput = vec3(1.0);
        shadowRay.seed = mainRay.seed;
        shadowRay.hitT = INF_DISTANCE;
        shadowRay.insideBoat = mainRay.insideBoat;

        uint shadowMask = WORLD_MASK | CLOUD_MASK;
        if (mainRay.isHand == 0) {
            shadowMask |= PLAYER_MASK;  // world surfaces see player shadows; hand does not (prevents self-shadowing)
        }

        traceRayEXT(topLevelAS, gl_RayFlagsNoneEXT,
                    shadowMask,
                    0,                                     // sbtRecordOffset
                    0,                                     // sbtRecordStride
                    2,                                     // missIndex
                    shadowRayOrigin, 0.001, sampledLightDir, 1000, 1);

        // Add direct lighting contribution
        vec3 lightContribution = shadowRay.radiance;

        float progress = skyUBO.rainGradient;
        vec3 lightRadiance = lightContribution * mainRay.throughput * lightBRDF;
        vec3 finalLightRadiance = mix(lightRadiance, vec3(0.0), progress);

        // Hand shadow smoothing: apply ambient floor with smooth falloff
        // Prevents harsh black transitions on hand geometry in shadow
        if (mainRay.isHand > 0) {
            float shadowLum = dot(finalLightRadiance, vec3(0.2126, 0.7152, 0.0722));
            float ambientFloor = 0.08 * dot(mainRay.throughput, vec3(0.2126, 0.7152, 0.0722));
            float blend = smoothstep(0.0, ambientFloor * 2.0, shadowLum);
            vec3 handAmbient = ambientFloor * tint * mainRay.throughput;
            finalLightRadiance = mix(handAmbient, finalLightRadiance, blend);
        }

        mainRay.radiance += finalLightRadiance;
        mainRay.directLightRadiance = finalLightRadiance;

        mainRay.directLightHitT = shadowRay.hitT;
    }

    mainRay.instanceIndex = instanceID;
    mainRay.geometryIndex = geometryID;
    mainRay.primitiveIndex = gl_PrimitiveID;
    mainRay.baryCoords = baryCoords;
    mainRay.worldPos = worldPos;
    mainRay.normal = normal;
    mainRay.albedoValue = albedoValue;
    mainRay.specularValue = specularValue;
    mainRay.normalValue = normalValue;
    mainRay.flagValue = flagValue;

    // sample next direction using Disney BSDF
    vec3 sampleDir;
    float pdf;
    uint lobeType;
    vec3 bsdf = DisneySample(mat, viewDir, normal, sampleDir, pdf, mainRay.seed, lobeType);

    mainRay.throughput *= bsdf / max(pdf, 1e-4);
    mainRay.lobeType = lobeType;
    mainRay.noisy = 1;

    // early exit if sampling failed
    if (pdf <= 1e-6) {
        mainRay.stop = 1;
        return;
    }

    vec3 offsetDir = dot(sampleDir, normal) > 0.0 ? normal : -normal;
    mainRay.origin = worldPos + offsetDir * 0.001;

    mainRay.direction = sampleDir;
    mainRay.stop = 0;
}
