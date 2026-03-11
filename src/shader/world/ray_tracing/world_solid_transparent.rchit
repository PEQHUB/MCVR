#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_opacity_micromap : require

#include "../util/disney.glsl"
#include "../util/random.glsl"
#include "../util/ray_cone.glsl"
#include "../util/ray_payloads.glsl"
#include "../util/util.glsl"
#include "common/shared.hpp"
#include "../util/area_light.glsl"
#include "../util/restir.glsl"
#include "../util/ray_offset.glsl"

layout(set = 0, binding = 0) uniform sampler2D textures[];

#include "../util/clouds.glsl"

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

layout(set = 1, binding = 8) readonly buffer AreaLightBuffer {
    AreaLight lights[];
} areaLightBuffer;

layout(set = 1, binding = 9) readonly buffer TileLightBuffer {
    uint data[];
} tileLightBuffer;

const int TILE_SIZE = 16;
const int MAX_LIGHTS_PER_TILE = 512;

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

layout(push_constant) uniform PushConstant {
    int numRayBounces;
    int flags;
    int areaLightCount;
    float shadowSoftness;
    int risCandidates;
    int temporalMClamp;
    int wClamp;
    float preExposure;
} pc;
#define SIMPLIFIED_INDIRECT ((pc.flags & 1) != 0)
#define AREA_LIGHTS_ON ((pc.flags & 2) != 0)
#define RESTIR_ENABLED ((pc.flags & 4) != 0)
#define RESTIR_SIMPLIFIED_BRDF ((pc.flags & 8) != 0)
#define RESTIR_BOUNCE_ENABLED ((pc.flags & 16) != 0)

layout(set = 3, binding = 3, rgba16f) uniform readonly image2D normalRoughnessImage;
layout(set = 3, binding = 4, rg16f) uniform readonly image2D motionVectorImage;
layout(set = 3, binding = 5, r16f) uniform readonly image2D linearDepthImage;
layout(set = 3, binding = 14, rgba32f) uniform image2D reservoirCurrentImage;
layout(set = 3, binding = 15, rgba32f) uniform image2D reservoirPreviousImage;
layout(set = 3, binding = 18, rgba32f) uniform image2D bounceReservoirImage1;
layout(set = 3, binding = 19, rgba32f) uniform image2D bounceReservoirImage2;
layout(set = 3, binding = 20, rgba32f) uniform image2D bounceReservoirImage3;

// Bounce reservoir load/store helpers (per-bounce-level image selection)
vec4 loadBounceReservoir(uint bounceIdx, ivec2 p) {
    if (bounceIdx == 1u) return imageLoad(bounceReservoirImage1, p);
    if (bounceIdx == 2u) return imageLoad(bounceReservoirImage2, p);
    return imageLoad(bounceReservoirImage3, p);
}
void storeBounceReservoir(uint bounceIdx, ivec2 p, vec4 d) {
    if (bounceIdx == 1u) imageStore(bounceReservoirImage1, p, d);
    else if (bounceIdx == 2u) imageStore(bounceReservoirImage2, p, d);
    else imageStore(bounceReservoirImage3, p, d);
}

layout(location = 0) rayPayloadInEXT PrimaryRay mainRay;
layout(location = 1) rayPayloadEXT ShadowRay shadowRay;
hitAttributeEXT vec2 attribs;

vec3 calculateNormal(vec3 p0, vec3 p1, vec3 p2, vec2 uv0, vec2 uv1, vec2 uv2, vec3 matNormal, vec3 viewDir, out vec3 geometricNormal) {
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

    vec3 glint = vec3(0.0);
    vec4 overlayColor = vec4(0.0);
    if (!SIMPLIFIED_INDIRECT || mainRay.index <= 1) {
        uint useGlint = v0.useGlint;
        uint glintTexture = v0.glintTexture;
        vec2 glintUV = baryCoords.x * v0.glintUV + baryCoords.y * v1.glintUV + baryCoords.z * v2.glintUV;
        glintUV = (worldUbo.textureMat * vec4(glintUV, 0.0, 1.0)).xy;
        glint = useGlint * texture(textures[nonuniformEXT(glintTexture)], glintUV).rgb;
        glint = glint * glint;

        uint useOverlay = v0.useOverlay;
        ivec2 overlayUV = v0.overlayUV;
        overlayColor = texelFetch(textures[nonuniformEXT(worldUbo.overlayTextureID)], overlayUV, 0);
    }

    vec3 tint;
    if (v0.useOverlay > 0) {
        tint = mix(overlayColor.rgb, albedoValue.rgb * colorLayer, overlayColor.a) + glint;
    } else {
        tint = albedoValue.rgb * colorLayer + glint;
    }

    albedoValue = vec4(tint, albedoValue.a);
    LabPBRMat mat = convertLabPBRMaterial(albedoValue, specularValue, normalValue);

    // the provided normal is unreliable! (such as grass, etc.)
    // calculate on the fly for now
    vec3 geometricNormal;
    vec3 normal =
        calculateNormal(v0.pos, v1.pos, v2.pos, v0.textureUV, v1.textureUV, v2.textureUV, mat.normal, viewDir, geometricNormal);

    // Visible area light cube: if this hit point is inside a small light cube, render as solid emissive
    if (AREA_LIGHTS_ON && pc.areaLightCount > 0 && mainRay.index == 0) {
        int checkCount = min(pc.areaLightCount, 64);

        for (int i = 0; i < checkCount; i++) {
            int idx = i;
            AreaLight al = areaLightBuffer.lights[idx];
            if (al.halfExtent >= 0.3) continue;

            vec3 delta = abs(worldPos - al.position);
            if (all(lessThanEqual(delta, vec3(al.halfExtent)))) {
                // Fade cube brightness at close range to prevent clear/denoised boundary artifacts
                float cubeDist = length(al.position - gl_WorldRayOriginEXT);
                float cubeFade = smoothstep(0.3, 1.5, cubeDist);
                if (cubeFade < 0.01) continue;

                // Visible emissive cube — route through CLEAR path (bypasses NRD denoiser)
                mainRay.radiance = al.color * max(al.intensity, 1.0) * 200.0 * cubeFade * mainRay.throughput;
                mainRay.albedoValue = vec4(al.color, 1.0);
                mainRay.albedoEmission = 0.0;
                mainRay.specularValue = vec4(0.0);
                mainRay.normalValue = vec4(0.5, 0.5, 1.0, 1.0); // flat LabPBR normal (avoids NaN from vec4(0))
                mainRay.flagValue = ivec4(0);
                mainRay.directLightRadiance = vec3(0.0);
                mainRay.directLightHitT = INF_DISTANCE;
                mainRay.hitT = gl_HitTEXT;
                mainRay.coneWidth += gl_HitTEXT * mainRay.coneSpread;
                mainRay.worldPos = worldPos;
                mainRay.normal = normal;
                mainRay.instanceIndex = instanceID;
                mainRay.geometryIndex = geometryID;
                mainRay.primitiveIndex = gl_PrimitiveID;
                mainRay.baryCoords = baryCoords;
                mainRay.noisy = 0;
                mainRay.lobeType = 0;
                mainRay.stop = 1;
                return;
            }
        }
    }

    // add glowing radiance (with area light / emissive mutual exclusion)
    // albedoEmission sign convention:
    //   positive = emissive mode (full bounce emission)
    //   negative = area light mode (suppress bounce emission; abs = self-glow value)
    float rawAlbedoEmission = albedoEmission;
    albedoEmission = abs(rawAlbedoEmission);
    bool hasAreaLight = rawAlbedoEmission < -0.0001;

    float combinedEmission = max(mat.emission, albedoEmission);

    float factor;
    if (mainRay.index == 0) {
        factor = 1.0; // Primary hit: always show self-glow
    } else if (hasAreaLight) {
        factor = 0.0;
    } else {
        factor = 4.0 * skyUBO.hdrRadianceScale; // Emissive mode: reduced from 16x to 4x
    }

    vec3 emissionRadiance = factor * tint * combinedEmission * mainRay.throughput;

    // Per-sample contribution clamping to prevent fireflies
    float emissionLum = dot(emissionRadiance, vec3(0.2126, 0.7152, 0.0722));
    float maxContribution = 4.0;
    if (emissionLum > maxContribution) {
        emissionRadiance *= maxContribution / emissionLum;
    }

    mainRay.radiance += emissionRadiance;
    mainRay.albedoEmission = albedoEmission; // Store abs value for bloom

    mainRay.hitT = gl_HitTEXT;
    mainRay.coneWidth += gl_HitTEXT * mainRay.coneSpread;

    // shadow ray for direct lighting
    vec3 sunDir = normalize(skyUBO.sunDirection);
    vec3 lightDir = sunDir;
    // Softer sun sampling for hand = wider penumbras (500 vs 3000)
    float kappa = (mainRay.isHand > 0) ? 500.0 : 3000.0;
    if (sunDir.y < 0) { lightDir = normalize(skyUBO.moonDirection); }
    vec3 sampledLightDir = SampleVMF(mainRay.seed, lightDir, kappa);
    vec3 shadowBiasN = dot(sampledLightDir, geometricNormal) > 0.0 ? geometricNormal : -geometricNormal;
    vec3 shadowRayOrigin = offset_ray(worldPos, shadowBiasN);

    if (worldUbo.skyType == 1) {
        float pdf; // not used
        vec3 lightBRDF = DisneyEval(mat, viewDir, normal, sampledLightDir, pdf);

        shadowRay.radiance = vec3(0.0);
        shadowRay.throughput = vec3(1.0);
        shadowRay.seed = mainRay.seed;
        shadowRay.hitT = INF_DISTANCE;
        shadowRay.insideBoat = mainRay.insideBoat;
        shadowRay.bounceIndex = mainRay.index;

        uint shadowMask = WORLD_MASK;
        if (mainRay.isHand == 0) {
            shadowMask |= PLAYER_MASK;  // world surfaces see player shadows; hand does not (prevents self-shadowing)
        }

        traceRayEXT(topLevelAS, gl_RayFlagsNoneEXT,
                    shadowMask,
                    0,                                     // sbtRecordOffset
                    0,                                     // sbtRecordStride
                    2,                                     // missIndex
                    shadowRayOrigin, 0.0, sampledLightDir, 1000, 1);

        // Add direct lighting contribution
        vec3 lightContribution = shadowRay.radiance;

        // Apply cloud shadowing (procedural volumetric slab).
        // This is evaluated at the shading point so it works for primary and reflected paths.
        float cloudT = cloudTransmittance(worldPos + sampledLightDir * 0.01, sampledLightDir, 0.0, 1000.0, worldUbo, skyUBO, 0);
        float shadowStrength = max(skyUBO.cloudLighting.x, 0.0);
        cloudT = pow(max(cloudT, 1e-6), shadowStrength);
        lightContribution *= cloudT;

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

    // Area light illumination
    if (AREA_LIGHTS_ON && pc.areaLightCount > 0 && mainRay.index == 0) {
        int searchCount = min(pc.areaLightCount, 128);

        if (RESTIR_ENABLED) {
            // ======== Clean ReSTIR DI: RIS → Temporal → Shadow → Shade ========
            ivec2 pixel = ivec2(mainRay.pixelPacked & 0xFFFFu, mainRay.pixelPacked >> 16u);

            // --- Load previous frame reservoir via motion vector reprojection ---
            vec2 mv = imageLoad(motionVectorImage, pixel).xy;
            ivec2 prevPixel = ivec2(round(vec2(pixel) + mv));
            ivec2 imgSize = imageSize(reservoirPreviousImage);
            bool prevValid = (prevPixel.x >= 0 && prevPixel.y >= 0 &&
                              prevPixel.x < imgSize.x && prevPixel.y < imgSize.y);

            if (prevValid) {
                vec3 prevNorm = imageLoad(normalRoughnessImage, prevPixel).xyz;
                float prevDepth = imageLoad(linearDepthImage, prevPixel).r;
                float currDepth = imageLoad(linearDepthImage, pixel).r;
                if (dot(normal, prevNorm) < 0.9 ||
                    abs(currDepth - prevDepth) / max(currDepth, 0.001) > 0.1)
                    prevValid = false;
            }

            vec4 prevPacked = prevValid ? imageLoad(reservoirPreviousImage, prevPixel) : vec4(0.0);
            Reservoir prevRes = unpackReservoir(prevPacked);

            // --- Step 1: RIS candidates — uniform random from tile list ---
            Reservoir currentRes = createReservoir();
            int prevLightIdx = -1;
            uint prevStableId = prevRes.lightStableId;

            // Cap at 64: contribution-sorted global list makes top-64 sufficient.
            // Larger pool increases per-frame variance → noisy → denoiser lag + elevated blacks.
            int effectiveCount = min(pc.areaLightCount, 64);
            int numCandidates = min(pc.risCandidates, effectiveCount);
            float sourcePdf = 1.0 / float(effectiveCount);

            for (int c = 0; c < numCandidates; c++) {
                // Uniform random selection from available lights
                int tileSlot = int(rand(mainRay.seed) * float(effectiveCount));
                tileSlot = clamp(tileSlot, 0, effectiveCount - 1);
                int idx = tileSlot;
                AreaLight al = areaLightBuffer.lights[idx];


                // Opportunistic stableId match (O(1) — no separate scan)
                if (prevLightIdx < 0 && prevStableId != 0u) {
                    uint sid = floatBitsToUint(al._unused.x);
                    if (sid == prevStableId) prevLightIdx = idx;
                }

                vec3 toLight = al.position - worldPos;
                float dist2 = dot(toLight, toLight);
                float dist = sqrt(dist2);
                vec3 alDir = toLight / max(dist, 0.001);

                // Distance culling (Chebyshev or Euclidean)
                vec3 absToLight = abs(toLight);
                float chebyDist = max(absToLight.x, max(absToLight.y, absToLight.z));
                float cullDist = chebyDist;
                if (cullDist > al.radius) continue;

                float NdotL = dot(normal, alDir);
                if (NdotL < -0.1 || dist < 0.05) continue;

                // Windowed inverse-square attenuation
                float invDist2 = 1.0 / max(dist2, 0.01);
                float f = cullDist / al.radius;
                float window = max(1.0 - f * f, 0.0);
                window *= window;

                float atten;
                if (al.halfExtent >= 0.01) {
                    float faceArea = cubeFaceArea(al.halfExtent);
                    vec3 absDir = abs(alDir);
                    float cosLight = max(absDir.x, max(absDir.y, absDir.z));
                    atten = faceArea * cosLight * invDist2 * window;
                } else {
                    atten = invDist2 * window;
                }
                if (atten < 0.00001) continue;

                // Luminance-only target PDF (canonical ReSTIR — position-dependent, view-independent)
                vec3 unshadowed = al.color * al.intensity * atten;
                float targetPdf = computeTargetPdf(unshadowed);
                if (targetPdf <= 0.0) continue;

                float weight = targetPdf / sourcePdf;
                uint stableId = floatBitsToUint(al._unused.x);
                updateReservoir(currentRes, stableId, idx, weight, targetPdf, unshadowed, alDir, dist, mainRay.seed);
            }

            // Fallback: scan tile list for stableId match (uint comparison only)
            if (prevLightIdx < 0 && prevStableId != 0u) {
                int scanCount = min(effectiveCount, 32);
                for (int i = 0; i < scanCount; i++) {
                    int idx = i;
                    if (floatBitsToUint(areaLightBuffer.lights[idx]._unused.x) == prevStableId) {
                        prevLightIdx = idx;
                        break;
                    }
                }
            }

            // --- Step 2: Temporal reuse — merge previous reservoir ---
            if (prevLightIdx >= 0 && prevRes.M > 0.0) {
                AreaLight prevAL = areaLightBuffer.lights[prevLightIdx];

                vec3 toLight = prevAL.position - worldPos;
                float dist2 = dot(toLight, toLight);
                float dist = sqrt(dist2);

                vec3 absToLight = abs(toLight);
                float chebyDist = max(absToLight.x, max(absToLight.y, absToLight.z));
                float cullDist = chebyDist;

                if (cullDist <= prevAL.radius && dist >= 0.05) {
                    vec3 alDir = toLight / max(dist, 0.001);

                    if (dot(normal, alDir) > -0.1) {
                        float invDist2 = 1.0 / max(dist2, 0.01);
                        float f2 = cullDist / prevAL.radius;
                        float window = max(1.0 - f2 * f2, 0.0);
                        window *= window;

                        float atten;
                        if (prevAL.halfExtent >= 0.01) {
                            float faceArea = cubeFaceArea(prevAL.halfExtent);
                            vec3 absDir = abs(alDir);
                            float cosLight = max(absDir.x, max(absDir.y, absDir.z));
                            atten = faceArea * cosLight * invDist2 * window;
                        } else {
                            atten = invDist2 * window;
                        }

                        vec3 prevUnshadowed = prevAL.color * prevAL.intensity * atten;
                        float prevTargetPdf = computeTargetPdf(prevUnshadowed);

                        if (prevTargetPdf > 0.0) {
                            prevRes.M = min(prevRes.M, float(pc.temporalMClamp));
                            prevRes.unshadowed = prevUnshadowed;
                            prevRes.lightDir = alDir;
                            prevRes.lightDist = dist;
                            prevRes.lightIdx = prevLightIdx;
                            mergeReservoir(currentRes, prevRes, prevTargetPdf, mainRay.seed);
                        }
                    }
                }
            }

            finalizeReservoir(currentRes);
            currentRes.W = min(currentRes.W, float(pc.wClamp));

            // --- Step 3: Shadow ray + shade selected light ---
            vec3 alAccum = vec3(0.0);
            if (currentRes.lightIdx >= 0) {
                AreaLight al = areaLightBuffer.lights[currentRes.lightIdx];

                CubeSample cs = sampleCubeLight(al, worldPos, pc.shadowSoftness, mainRay.seed);
                vec3 toSample = cs.worldPos - worldPos;
                float sDist = length(toSample);
                vec3 sDir = toSample / sDist;
                vec3 shadowOrigin = offset_ray(worldPos, geometricNormal);

                shadowRay.radiance = vec3(0.0);
                shadowRay.throughput = vec3(0.0);
                shadowRay.seed = mainRay.seed;
                shadowRay.hitT = INF_DISTANCE;
                shadowRay.insideBoat = mainRay.insideBoat;
                shadowRay.bounceIndex = mainRay.index;

                float safeMargin = al.halfExtent + 0.02;
                float alTMax = sDist - safeMargin;
                if (alTMax > 0.01) {
                    traceRayEXT(topLevelAS,
                        gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT,
                        WORLD_MASK, 0, 0, 3,
                        shadowOrigin, 0.0, sDir, alTMax, 1);
                } else {
                    shadowRay.throughput = vec3(1.0);
                }

                float visibility = shadowRay.throughput.x;

                vec3 brdf;
                if (RESTIR_SIMPLIFIED_BRDF) {
                    float NdotL = max(dot(normal, currentRes.lightDir), 0.0);
                    brdf = NdotL / 3.14159265 * mat.albedo;
                } else {
                    float pdf;
                    brdf = DisneyEval(mat, viewDir, normal, currentRes.lightDir, pdf);
                }
                alAccum = currentRes.unshadowed * brdf * currentRes.W * visibility;
            }

            mainRay.directLightRadiance += alAccum * mainRay.throughput;
            mainRay.radiance += alAccum * mainRay.throughput;

            // --- Step 4: Store reservoir for next frame (spatial reuse reads this) ---
            imageStore(reservoirCurrentImage, pixel, packReservoir(currentRes));

        } else {
            // ======== Fallback: deterministic top-2 shadowed ========
            int bestIdx[2] = int[](-1, -1);
            float bestContrib[2] = float[](0.0, 0.0);
            vec3 bestUnshadowed[2] = vec3[](vec3(0.0), vec3(0.0));
            vec3 bestDir[2] = vec3[](vec3(0.0), vec3(0.0));
            float bestDist[2] = float[](0.0, 0.0);

            for (int i = 0; i < searchCount; i++) {
                AreaLight al = areaLightBuffer.lights[i];

                vec3 toLight = al.position - worldPos;
                float dist2 = dot(toLight, toLight);
                float dist = sqrt(dist2);
                vec3 alDir = toLight / max(dist, 0.001);

                // Distance culling (Chebyshev or Euclidean)
                vec3 absToLight = abs(toLight);
                float chebyDist = max(absToLight.x, max(absToLight.y, absToLight.z));
                float cullDist = chebyDist;
                if (cullDist > al.radius) continue;

                float NdotL = dot(normal, alDir);
                if (NdotL < -0.1 || dist < 0.05) continue;

                float invDist2 = 1.0 / max(dist2, 0.01);
                float f = cullDist / al.radius;
                float window = max(1.0 - f * f, 0.0);
                window *= window;

                float atten;
                if (al.halfExtent >= 0.01) {
                    float faceArea = cubeFaceArea(al.halfExtent);
                    vec3 absDir = abs(alDir);
                    float cosLight = max(absDir.x, max(absDir.y, absDir.z));
                    atten = faceArea * cosLight * invDist2 * window;
                } else {
                    atten = invDist2 * window;
                }
                if (atten < 0.00001) continue;

                // DisneyEval already includes cosine — no NdotL here
                vec3 unshadowed = al.color * al.intensity * atten;
                float contrib = dot(unshadowed, vec3(0.2126, 0.7152, 0.0722));

                if (contrib > bestContrib[0]) {
                    bestIdx[1] = bestIdx[0]; bestContrib[1] = bestContrib[0];
                    bestUnshadowed[1] = bestUnshadowed[0]; bestDir[1] = bestDir[0]; bestDist[1] = bestDist[0];
                    bestIdx[0] = i; bestContrib[0] = contrib;
                    bestUnshadowed[0] = unshadowed; bestDir[0] = alDir; bestDist[0] = dist;
                } else if (contrib > bestContrib[1]) {
                    bestIdx[1] = i; bestContrib[1] = contrib;
                    bestUnshadowed[1] = unshadowed; bestDir[1] = alDir; bestDist[1] = dist;
                }
            }

            vec3 alAccum = vec3(0.0);
            for (int k = 0; k < 2; k++) {
                if (bestIdx[k] < 0) continue;
                AreaLight al = areaLightBuffer.lights[bestIdx[k]];

                CubeSample cs = sampleCubeLight(al, worldPos, pc.shadowSoftness, mainRay.seed);
                vec3 toSample = cs.worldPos - worldPos;
                float sDist = length(toSample);
                vec3 sDir = toSample / sDist;
                vec3 shadowOrigin = offset_ray(worldPos, geometricNormal);

                shadowRay.radiance = vec3(0.0);
                shadowRay.throughput = vec3(0.0);
                shadowRay.seed = mainRay.seed;
                shadowRay.hitT = INF_DISTANCE;
                shadowRay.insideBoat = mainRay.insideBoat;
                shadowRay.bounceIndex = mainRay.index;

                float safeMargin = al.halfExtent + 0.02;
                float alTMax = sDist - safeMargin;
                if (alTMax > 0.01) {
                    traceRayEXT(topLevelAS,
                        gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT,
                        WORLD_MASK, 0, 0, 3,
                        shadowOrigin, 0.0, sDir, alTMax, 1);
                } else {
                    shadowRay.throughput = vec3(1.0);
                }

                float visibility = shadowRay.throughput.x;
                vec3 brdf;
                if (RESTIR_SIMPLIFIED_BRDF) {
                    float NdotL = max(dot(normal, bestDir[k]), 0.0);
                    brdf = NdotL / 3.14159265 * mat.albedo;
                } else {
                    float pdf;
                    brdf = DisneyEval(mat, viewDir, normal, bestDir[k], pdf);
                }
                alAccum += bestUnshadowed[k] * brdf * visibility;
            }

            mainRay.directLightRadiance += alAccum * mainRay.throughput;
            mainRay.radiance += alAccum * mainRay.throughput;
        }
    } else if (AREA_LIGHTS_ON && pc.areaLightCount > 0 && mainRay.index > 0 && RESTIR_ENABLED && RESTIR_BOUNCE_ENABLED) {
        // ======== Bounce 1-3 ReSTIR DI: RIS → Temporal → Shadow → Shade ========
        // Per-bounce reservoir image, same-pixel temporal reuse (no MV reprojection).
        // Uses global SSBO (camera-sorted) since bounce hits have no screen-space tile list.
        ivec2 pixel = ivec2(mainRay.pixelPacked & 0xFFFFu, mainRay.pixelPacked >> 16u);

        // --- Load previous bounce reservoir for THIS bounce level ---
        Reservoir prevRes = unpackReservoir(loadBounceReservoir(mainRay.index, pixel));

        // --- Step 1: RIS candidates from global SSBO ---
        Reservoir currentRes = createReservoir();
        int prevLightIdx = -1;
        uint prevStableId = prevRes.lightStableId;

        int searchCount = min(pc.areaLightCount, 128 / 4);
        int numCandidates = min(pc.risCandidates, searchCount);
        float sourcePdf = 1.0 / float(searchCount);

        for (int c = 0; c < numCandidates; c++) {
            int idx = clamp(int(rand(mainRay.seed) * float(searchCount)), 0, searchCount - 1);
            AreaLight al = areaLightBuffer.lights[idx];


            // Opportunistic stableId match for temporal reuse
            uint sid = floatBitsToUint(al._unused.x);
            if (prevLightIdx < 0 && prevStableId != 0u && sid == prevStableId)
                prevLightIdx = idx;

            vec3 toLight = al.position - worldPos;
            float dist2 = dot(toLight, toLight);
            float dist = sqrt(dist2);
            vec3 alDir = toLight / max(dist, 0.001);

            vec3 absToLight = abs(toLight);
            float chebyDist = max(absToLight.x, max(absToLight.y, absToLight.z));
            float cullDist = chebyDist;
            if (cullDist > al.radius) continue;
            float NdotL = dot(normal, alDir);
            if (NdotL < -0.1 || dist < 0.05) continue;

            float invDist2 = 1.0 / max(dist2, 0.01);
            float f = cullDist / al.radius;
            float window = max(1.0 - f * f, 0.0);
            window *= window;

            float atten;
            if (al.halfExtent >= 0.01) {
                float faceArea = cubeFaceArea(al.halfExtent);
                vec3 absDir = abs(alDir);
                float cosLight = max(absDir.x, max(absDir.y, absDir.z));
                atten = faceArea * cosLight * invDist2 * window;
            } else {
                atten = invDist2 * window;
            }
            if (atten < 0.00001) continue;

            vec3 unshadowed = al.color * al.intensity * atten;
            float targetPdf = computeTargetPdf(unshadowed);
            if (targetPdf <= 0.0) continue;

            float weight = targetPdf / sourcePdf;
            updateReservoir(currentRes, sid, idx, weight, targetPdf, unshadowed, alDir, dist, mainRay.seed);
        }

        // Fallback: scan global SSBO for stableId match (uint comparison only)
        if (prevLightIdx < 0 && prevStableId != 0u) {
            int scanCount = min(pc.areaLightCount, 32);
            for (int i = 0; i < scanCount; i++) {
                if (floatBitsToUint(areaLightBuffer.lights[i]._unused.x) == prevStableId) {
                    prevLightIdx = i;
                    break;
                }
            }
        }

        // --- Step 2: Temporal merge (halved M clamp for bounce stability) ---
        if (prevLightIdx >= 0 && prevRes.M > 0.0) {
            AreaLight prevAL = areaLightBuffer.lights[prevLightIdx];

            vec3 toLight = prevAL.position - worldPos;
            float dist2 = dot(toLight, toLight);
            float dist = sqrt(dist2);

            vec3 absToLight = abs(toLight);
            float chebyDist = max(absToLight.x, max(absToLight.y, absToLight.z));
            float cullDist = chebyDist;

            if (cullDist <= prevAL.radius && dist >= 0.05) {
                vec3 alDir = toLight / max(dist, 0.001);

                if (dot(normal, alDir) > -0.1) {
                    float invDist2 = 1.0 / max(dist2, 0.01);
                    float f2 = cullDist / prevAL.radius;
                    float window = max(1.0 - f2 * f2, 0.0);
                    window *= window;

                    float atten;
                    if (prevAL.halfExtent >= 0.01) {
                        float faceArea = cubeFaceArea(prevAL.halfExtent);
                        vec3 absDir = abs(alDir);
                        float cosLight = max(absDir.x, max(absDir.y, absDir.z));
                        atten = faceArea * cosLight * invDist2 * window;
                    } else {
                        atten = invDist2 * window;
                    }

                    vec3 prevUnshadowed = prevAL.color * prevAL.intensity * atten;
                    float prevTargetPdf = computeTargetPdf(prevUnshadowed);

                    if (prevTargetPdf > 0.0) {
                        prevRes.M = min(prevRes.M, float(pc.temporalMClamp / 2));
                        prevRes.unshadowed = prevUnshadowed;
                        prevRes.lightDir = alDir;
                        prevRes.lightDist = dist;
                        prevRes.lightIdx = prevLightIdx;
                        mergeReservoir(currentRes, prevRes, prevTargetPdf, mainRay.seed);
                    }
                }
            }
        }

        finalizeReservoir(currentRes);
        currentRes.W = min(currentRes.W, float(pc.wClamp));

        // --- Step 3: Shadow ray + shade selected light ---
        vec3 alAccum = vec3(0.0);
        if (currentRes.lightIdx >= 0) {
            AreaLight al = areaLightBuffer.lights[currentRes.lightIdx];

            CubeSample cs = sampleCubeLight(al, worldPos, pc.shadowSoftness, mainRay.seed);
            vec3 toSample = cs.worldPos - worldPos;
            float sDist = length(toSample);
            vec3 sDir = toSample / sDist;
            vec3 shadowOrigin = offset_ray(worldPos, geometricNormal);

            shadowRay.radiance = vec3(0.0);
            shadowRay.throughput = vec3(0.0);
            shadowRay.seed = mainRay.seed;
            shadowRay.hitT = INF_DISTANCE;
            shadowRay.insideBoat = mainRay.insideBoat;
            shadowRay.bounceIndex = mainRay.index;

            float safeMargin = al.halfExtent + 0.02;
            float alTMax = sDist - safeMargin;
            if (alTMax > 0.01) {
                traceRayEXT(topLevelAS,
                    gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT,
                    WORLD_MASK, 0, 0, 3,
                    shadowOrigin, 0.0, sDir, alTMax, 1);
            } else {
                shadowRay.throughput = vec3(1.0);
            }

            float visibility = shadowRay.throughput.x;

            // Simplified Lambertian BRDF for indirect bounces (specular invisible behind denoiser)
            float NdotL = max(dot(normal, currentRes.lightDir), 0.0);
            vec3 brdf = NdotL / 3.14159265 * mat.albedo;
            alAccum = currentRes.unshadowed * brdf * currentRes.W * visibility;
        }

        alAccum *= 1.0;
        mainRay.radiance += alAccum * mainRay.throughput;

        // Store reservoir for next frame's temporal reuse
        storeBounceReservoir(mainRay.index, pixel, packReservoir(currentRes));

    } else if (AREA_LIGHTS_ON && pc.areaLightCount > 0 && mainRay.index > 0) {
        // Fallback when ReSTIR disabled: original unshadowed 8-light accumulation
        int bounceSearchCount = min(pc.areaLightCount, max(128 / 16, 8));
        vec3 alAccum = vec3(0.0);

        for (int i = 0; i < bounceSearchCount; i++) {
            AreaLight al = areaLightBuffer.lights[i];

            vec3 toLight = al.position - worldPos;
            float dist2 = dot(toLight, toLight);
            float dist = sqrt(dist2);
            vec3 alDir = toLight / max(dist, 0.001);

            vec3 absToLight = abs(toLight);
            float chebyDist = max(absToLight.x, max(absToLight.y, absToLight.z));
            float cullDist = chebyDist;
            if (cullDist > al.radius) continue;

            float NdotL = dot(normal, alDir);
            if (NdotL < -0.1 || dist < 0.05) continue;

            float invDist2 = 1.0 / max(dist2, 0.01);
            float f = cullDist / al.radius;
            float window = max(1.0 - f * f, 0.0);
            window *= window;

            float atten;
            if (al.halfExtent >= 0.01) {
                float faceArea = cubeFaceArea(al.halfExtent);
                vec3 absDir = abs(alDir);
                float cosLight = max(absDir.x, max(absDir.y, absDir.z));
                atten = faceArea * cosLight * invDist2 * window;
            } else {
                atten = invDist2 * window;
            }
            if (atten < 0.00001) continue;

            vec3 unshadowed = al.color * al.intensity * atten;

            vec3 brdf;
            if (RESTIR_SIMPLIFIED_BRDF) {
                float NdotL_b = max(dot(normal, alDir), 0.0);
                brdf = NdotL_b / 3.14159265 * mat.albedo;
            } else {
                float pdf;
                brdf = DisneyEval(mat, viewDir, normal, alDir, pdf);
            }

            alAccum += unshadowed * brdf;
        }

        alAccum *= 1.0;
        mainRay.radiance += alAccum * mainRay.throughput;
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

    mainRay.lobeType = lobeType;
    mainRay.noisy = 1;

    // early exit if sampling failed (check BEFORE updating throughput to avoid amplification)
    if (pdf <= 1e-6) {
        mainRay.stop = 1;
        return;
    }

    mainRay.throughput *= bsdf / max(pdf, 1e-4);

    vec3 bounceOffsetN = dot(sampleDir, geometricNormal) > 0.0 ? geometricNormal : -geometricNormal;
    mainRay.origin = offset_ray(worldPos, bounceOffsetN);

    mainRay.direction = sampleDir;
    mainRay.stop = 0;
}
