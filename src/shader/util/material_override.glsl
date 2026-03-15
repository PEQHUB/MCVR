// material_override.glsl — Extracted from world_solid_transparent.rchit (Phase 2A)
//
// THREE-TIER MATERIAL OVERRIDE CONTRACT
// Tier 1 (LabPBR): convertLabPBRMaterial() in labpbr.glsl
//   Source: resource pack _s/_n textures or Auto-PBR equivalents
//   Sets: roughness, metallic, emission, f0, normal, ao, height
//
// Tier 2 (Blender PBR): overlayDirectPBR() in labpbr.glsl
//   Gate: TEX_PROP_DIRECT_PBR flag (bit 2) in TextureMapEntry.properties
//   Protocol: per-channel -1.0 sentinel = "keep Tier 1 value"
//
// Tier 3 (UBO overrides): applyMaterialOverride() below
//   Gate: materialType > 0 (vertex emissiveBlockType bits 8-15)
//   Protocol per property:
//     F0: override if dot(pack0.rgb) > 0.0001, else derive from IOR (dielectric) or albedo (metal)
//     Roughness: additive offset if specular texture exists, else direct or texture-blend
//     Transmission: override if pack1.y >= 0.0 (sentinel -1 = keep Tier 1)
//     All others: direct replacement from UBO

#ifndef MATERIAL_OVERRIDE_GLSL
#define MATERIAL_OVERRIDE_GLSL

struct NoiseParams {
    float scale;
    float strength;
    int octaves;
    int type;
    int seed;
    int target;  // bit 0=roughness, bit 1=normal, bit 2=metallic
};

struct EmissionResult {
    vec3 radiance;
    float sceneEmission;
    float absEmission;
    bool hasAreaLight;
    uint emBlockType;      // emissive type from lower 8 bits of packedBlockType
};

#define EMISSION_REFERENCE_NITS 200.0

// Oklab chroma scaling for BT.2020 colors (gamut boost/reduction).
// boostFactor > 1.0 = more saturated, < 1.0 = less, 1.0 = no change.
// Includes gamut clamp: desaturates toward grey if result falls outside BT.2020.
vec3 applyGamutBoost(vec3 colorBT2020, float boostFactor) {
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

    vec3 lms = bt2020_to_lms * max(colorBT2020, vec3(0.0));
    vec3 lms_g = sign(lms) * pow(abs(lms), vec3(1.0 / 3.0));
    vec3 lab = lms_to_lab * lms_g;
    lab.yz *= boostFactor;
    vec3 lms_g2 = lab_to_lms * lab;
    vec3 lms2 = lms_g2 * lms_g2 * lms_g2;
    vec3 result = lms_to_bt2020 * lms2;
    // Gamut clamp: desaturate toward grey if outside BT.2020
    float minC = min(result.r, min(result.g, result.b));
    if (minC < 0.0) {
        float luma = dot(result, vec3(0.2627, 0.6780, 0.0593));
        float t = luma / (luma - minC);
        result = mix(vec3(luma), result, t);
    }
    return result;
}

// Tier 3: UBO material overrides (5 x vec4 per block from worldUbo.materialData[])
// Requires: worldUbo (set 2 binding 0) in scope
void applyMaterialOverride(
    inout LabPBRMat mat,
    uint materialType,       // block ordinal (bits 8-15 of emissiveBlockType)
    vec3 rawAlbedoLinear,    // original albedo for texture-blend roughness
    vec4 specularValue,      // specular texture sample (for hasSpecularData check)
    out NoiseParams noise,   // extracted noise parameters
    out float gamutBoost     // extracted gamut boost factor
) {
        uint idx = materialType - 1u;
        vec4 pack0 = worldUbo.materialData[idx];          // F0.rgb, roughness
        {
        vec4 pack1 = worldUbo.materialData[idx + 160u];   // metallic, transmission, ior, subsurface
        vec4 pack2 = worldUbo.materialData[idx + 320u];   // anisotropic, sheenWeight, sheenTint, coatWeight
        vec4 pack3 = worldUbo.materialData[idx + 480u];   // coatRoughness, noiseScale, noiseStrength, noiseOctaves
        vec4 pack4 = worldUbo.materialData[idx + 640u];   // channelR, channelG, channelB, textureBlend
        vec4 pack5 = worldUbo.materialData[idx + 800u];   // gamutBoost, pomDepth, reserved, reserved

        {
            // Apply F0 override if set; for dielectrics with zero F0, derive from IOR
            if (dot(pack0.rgb, pack0.rgb) > 0.0001) {
                mat.f0 = pack0.rgb;
            } else if (pack1.x < 0.5) {
                // Dielectric: compute F0 from IOR using Fresnel equation
                float ior = max(pack1.z, 1.0);
                float f0 = ((ior - 1.0) * (ior - 1.0)) / ((ior + 1.0) * (ior + 1.0));
                mat.f0 = vec3(f0);
            } else {
                // Metal without explicit F0: use albedo (standard Disney metallic workflow)
                mat.f0 = mat.albedo;
            }
            float matRoughness = pack0.a * pack0.a;  // perceptual -> GGX alpha

            // Texture roughness channel routing: derive roughness from albedo channel mix.
            // textureBlend > 0 blends between slider value and albedo-derived roughness.
            // textureBlend = 0 uses slider value directly (default for most blocks).
            float textureBlend = pack4.w;
            if (textureBlend > 0.001) {
                float weightSum = pack4.x + pack4.y + pack4.z;
                float signal = dot(rawAlbedoLinear, pack4.xyz) / max(weightSum, 0.001);
                float texRoughness = (1.0 - signal) * (1.0 - signal);
                mat.roughness = mix(matRoughness, texRoughness, textureBlend);
            } else {
                mat.roughness = matRoughness;
            }

            mat.metallic = pack1.x;
            // Transmission override protocol:
            // pack1.y >= 0.0 -> explicit override value (0.0 = opaque, 1.0 = fully transmissive)
            // pack1.y < 0.0  -> sentinel: keep LabPBR auto-detected value (set by Java for unregistered blocks)
            if (pack1.y >= 0.0) mat.transmission = pack1.y;
            mat.ior = max(pack1.z, 1.0);
            mat.subSurface = pack1.w;
            mat.anisotropic = pack2.x;
            mat.sheenWeight = pack2.y;
            mat.sheenTint = pack2.z;
            mat.coatWeight = pack2.w;
            mat.coatRoughness = pack3.x;

            // Extract noise parameters from pack3
            noise.scale = pack3.y;
            noise.strength = pack3.z;
            // pack3.w packs: octaves (bits 0-3) | noiseType (bits 4-7) | seed (bits 8-17) | noiseTarget (bits 20-22)
            // Use floatBitsToInt to preserve exact bit pattern (Java side uses Float.intBitsToFloat)
            int noisePacked = floatBitsToInt(pack3.w);
            noise.octaves = noisePacked & 0xF;
            noise.type = (noisePacked >> 4) & 0xF;
            noise.seed = (noisePacked >> 8) & 0x3FF;
            noise.target = (noisePacked >> 20) & 0x7;

            gamutBoost = pack5.x; // per-material gamut boost

            if (mat.metallic > 0.5) {
                // Sync albedo to F0 for metals. Required because Disney BRDF uses
                // mat.albedo (not mat.f0) for the metal Fresnel term. When the user
                // explicitly overrides F0 via pack0.rgb (line ~465), albedo must be
                // updated to match. Redundant but harmless when F0 wasn't overridden.
                mat.albedo = mat.f0;
            }
        }
        }
}

// Procedural noise modulation — gated by noiseTarget bits
// bit 0 = roughness, bit 1 = normal perturbation, bit 2 = metallic
// Requires: fbmTyped(), noiseGradientTyped() from noise.glsl
// Requires: worldUbo.cameraPos from WorldUBO
void applyNoiseModulation(
    inout LabPBRMat mat,
    inout vec3 normal,
    vec3 worldPos,
    dvec3 cameraPos,
    NoiseParams noise
) {
        // Use absolute world coordinates so noise is stable (doesn't follow camera)
        vec3 noisePos = (worldPos + vec3(cameraPos.xyz)) * noise.scale;
        // Apply seed as spatial offset for variation
        if (noise.seed > 0) {
            noisePos += vec3(float(noise.seed) * 7.13, float(noise.seed) * 11.37, float(noise.seed) * 23.71);
        }
        float n = fbmTyped(noisePos, noise.octaves, noise.type);
        // Roughness modulation (bit 0)
        if ((noise.target & 1) != 0) {
            mat.roughness = clamp(mat.roughness + n * noise.strength, 0.0, 1.0);
        }
        // Metallic modulation (bit 2)
        if ((noise.target & 4) != 0) {
            mat.metallic = clamp(mat.metallic + n * noise.strength, 0.0, 1.0);
        }
        // Normal perturbation from noise gradient (bit 1)
        if ((noise.target & 2) != 0) {
            float eps = 0.01 / max(noise.scale, 0.1);
            vec3 grad = noiseGradientTyped(noisePos, eps, noise.octaves, noise.type) * noise.strength * 0.3;
            vec3 T = normalize(cross(normal, abs(normal.y) < 0.99 ? vec3(0, 1, 0) : vec3(1, 0, 0)));
            vec3 B = cross(normal, T);
            normal = normalize(normal + grad.x * T + grad.y * B);
        }
}

// Emission evaluation: sign convention, UBO data, flame mask, gamut boost,
// scene-referred normalization, texture masking, firefly clamping.
// Requires: worldUbo.emissionData[], worldUbo.emissiveGamut[] from WorldUBO
// Uses applyGamutBoost() defined above in this file
// Requires: luminanceBT2020() from colorspace.glsl
EmissionResult evaluateEmission(
    float rawAlbedoEmission,
    vec3 tint,
    uint packedBlockType,
    uint bounceIndex,
    vec3 throughput,
    float preExposure,
    float labpbrEmission   // mat.emission from LabPBR (fallback when vertex emission is 0)
) {
    EmissionResult result;

    // albedoEmission sign convention:
    //   positive = emissive mode (full bounce emission)
    //   negative = area light mode (suppress bounce emission; abs = self-glow value)
    float albedoEmission = abs(rawAlbedoEmission);
    result.hasAreaLight = rawAlbedoEmission < -0.0001;

    // Apply per-block emission data from UBO (color override + scalar multiplier)
    // Extract emissive type from lower 8 bits of packed field
    uint emBlockType = packedBlockType & 0xFFu;
    vec3 emissionTint = tint; // default: texture albedo (BT.2020)
    if (emBlockType < 50u && albedoEmission > 0.0) {
        vec4 emData = worldUbo.emissionData[emBlockType];
        albedoEmission *= emData.a; // scalar multiplier
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
            emissionTint = applyGamutBoost(emissionTint, emGamut);
        }
    }

    // Emission in scene-referred units (1.0 = EMISSION_REFERENCE_NITS cd/m2).
    // Vertex nits (albedoEmission) are authoritative when set; LabPBR is fallback only.
    float combinedEmission = (albedoEmission != 0.0) ? albedoEmission : (labpbrEmission * EMISSION_REFERENCE_NITS);
    float sceneEmission = combinedEmission / EMISSION_REFERENCE_NITS;

    // Texture-based emission mask: only bright texels (flame head) emit.
    // Dark texels (wood stick, stone base) get zero emission.
    float maskLum = dot(tint, vec3(0.2627, 0.6780, 0.0593));
    sceneEmission *= smoothstep(0.10, 0.40, maskLum);

    float factor;
    if (bounceIndex == 0u) {
        factor = 1.0; // Primary hit: always show self-glow
    } else if (result.hasAreaLight) {
        factor = 0.0;
    } else {
        factor = 1.0; // Emissive mode
    }

    // Emission radiance: emissionTint is BT.2020 (texture albedo or spectral flame color).
    vec3 emissionRadiance = factor * emissionTint * sceneEmission * throughput;

    // Per-sample contribution clamping to prevent fireflies (scene-referred range)
    float emissionLum = luminanceBT2020(emissionRadiance);
    float maxContribution = 1000.0;
    if (emissionLum > maxContribution) {
        emissionRadiance *= maxContribution / emissionLum;
    }

    result.radiance = emissionRadiance;
    result.sceneEmission = sceneEmission;
    result.absEmission = albedoEmission;
    result.emBlockType = emBlockType;

    return result;
}

#endif // MATERIAL_OVERRIDE_GLSL
