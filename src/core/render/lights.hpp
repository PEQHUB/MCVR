#pragma once

#include <array>
#include <glm/glm.hpp>

// Light type IDs — must match Java LightSourceRegistry.java
enum LightTypeId {
    LIGHT_TORCH = 0,
    LIGHT_SOUL_TORCH = 1,
    LIGHT_LANTERN = 2,
    LIGHT_SOUL_LANTERN = 3,
    LIGHT_CAMPFIRE = 4,
    LIGHT_SOUL_CAMPFIRE = 5,
    LIGHT_GLOWSTONE = 6,
    LIGHT_SEA_LANTERN = 7,
    LIGHT_SHROOMLIGHT = 8,
    LIGHT_JACK_O_LANTERN = 9,
    LIGHT_END_ROD = 10,
    LIGHT_BEACON = 11,
    LIGHT_OCHRE_FROGLIGHT = 12,
    LIGHT_VERDANT_FROGLIGHT = 13,
    LIGHT_PEARL_FROGLIGHT = 14,
    LIGHT_REDSTONE_TORCH = 15,
    LIGHT_REDSTONE_LAMP = 16,
    LIGHT_CANDLE = 17,
    // 18-20: UNUSED (formerly CANDLE_2/3/4)
    LIGHT_CAVE_VINES = 21,
    LIGHT_GLOW_LICHEN = 22,
    LIGHT_FURNACE = 23,
    LIGHT_BLAST_FURNACE = 24,
    LIGHT_SMOKER = 25,
    LIGHT_ENDER_CHEST = 26,
    LIGHT_CRYING_OBSIDIAN = 27,
    LIGHT_NETHER_PORTAL = 28,
    LIGHT_CONDUIT = 29,
    LIGHT_RESPAWN_ANCHOR_1 = 30,
    LIGHT_RESPAWN_ANCHOR_2 = 31,
    LIGHT_RESPAWN_ANCHOR_3 = 32,
    LIGHT_RESPAWN_ANCHOR_4 = 33,
    LIGHT_AMETHYST_CLUSTER = 34,
    LIGHT_LARGE_AMETHYST_BUD = 35,
    LIGHT_COPPER_BULB = 36,
    LIGHT_ENCHANTING_TABLE = 37,
    // --- New: formerly emissive-only blocks ---
    LIGHT_LAVA = 38,
    LIGHT_FIRE = 39,
    LIGHT_SOUL_FIRE = 40,
    LIGHT_MAGMA_BLOCK = 41,
    LIGHT_SCULK_SENSOR = 42,
    LIGHT_SCULK_CATALYST = 43,
    LIGHT_SCULK_VEIN = 44,
    LIGHT_SCULK = 45,
    LIGHT_SCULK_SHRIEKER = 46,
    LIGHT_BREWING_STAND = 47,
    LIGHT_END_PORTAL = 48,
    LIGHT_END_PORTAL_FRAME = 49,
    LIGHT_TYPE_COUNT = 50,
};

struct LightTypeDef {
    float halfExtent;      // cube half-size (0.5=full, 0.05=point, 0.15=small, 0.3=medium)
    float lumens;          // physically-based luminous power (total output in lumens)
    float radius;          // max effective range in blocks
    glm::vec3 color;       // BT.709 base/fallback color (converted to BT.2020 at SSBO upload)
    float yOffset;         // vertical offset from block center (0.35 = flame tip)
    float flickerStrength; // procedural flicker amplitude (0.0=none, 0.08=torch, 0.15=candle)
    float colorTemperature; // Kelvin for blackbody (0 = non-thermal, uses color field)
    float spectralPurity;   // BT.2020 chroma boost (0.0 = none, 0.5 = aggressive)
};

// Converts lumens to the internal intensity scale.
// Calibrated for HDR area lights with cubeFaceArea attenuation.
// Small lights (halfExtent=0.10, faceArea=0.04) need high intensity to compensate for
// the area factor. 500/150 ≈ 3.33 gives torch (3000 lm) intensity 10000,
// producing ~8000 nits at 4 blocks on stone walls — strong warm glow on HDR.
inline constexpr float LUMENS_TO_INTENSITY = 500.0f / 150.0f;  // ~3.333

// Indexed by LightTypeId. Lumens = real-world luminous power.
// Intensity is derived at runtime: intensity = lumens * LUMENS_TO_INTENSITY * globalMultiplier * perBlockMultiplier
// yOffset: vertical shift from block center (0.35 = flame tip for torches/candles)
// colorTemperature: Kelvin for blackbody emitters (XYZ -> BT.2020 directly). 0 = use color field.
// spectralPurity: chroma boost in BT.2020 space for non-thermal emitters.
inline constexpr std::array<LightTypeDef, LIGHT_TYPE_COUNT> LIGHT_DEFS = {{
    //                                                                                          temp    purity
    // LIGHT_TORCH          (3000 lm — standard torch, blackbody 1800K)
    { 0.10f, 3000.0f, 48.0f, {1.0f, 0.7f, 0.3f}, 0.07f, 0.08f,  1800.0f, 0.00f },
    // LIGHT_SOUL_TORCH     (1600 lm — fantasy blue fire, spectral uplift)
    { 0.10f, 1600.0f, 40.0f, {0.3f, 0.8f, 0.9f}, 0.12f, 0.04f,     0.0f, 0.35f },
    // LIGHT_LANTERN         (200 lm — oil lantern, blackbody 2200K + slight uplift)
    { 0.15f, 200.0f, 48.0f, {1.0f, 0.7f, 0.3f}, -0.1f, 0.03f,  2200.0f, 0.05f },
    // LIGHT_SOUL_LANTERN    (100 lm — soul variant, spectral uplift)
    { 0.15f, 100.0f, 40.0f, {0.3f, 0.8f, 0.9f}, -0.1f, 0.02f,     0.0f, 0.30f },
    // LIGHT_CAMPFIRE        (800 lm — open wood fire, blackbody 1500K)
    { 0.15f, 800.0f, 48.0f, {1.0f, 0.7f, 0.3f}, 0.15f, 0.05f,  1500.0f, 0.00f },
    // LIGHT_SOUL_CAMPFIRE   (400 lm — blue variant, spectral uplift)
    { 0.15f, 400.0f, 40.0f, {0.3f, 0.8f, 0.9f}, 0.15f, 0.04f,     0.0f, 0.30f },
    // LIGHT_GLOWSTONE       (300 lm — phosphorescent mineral, spectral uplift)
    { 0.50f, 300.0f, 48.0f, {1.0f, 0.85f, 0.5f}, 0.0f, 0.0f,      0.0f, 0.40f },
    // LIGHT_SEA_LANTERN     (250 lm — bioluminescent aqua, spectral uplift)
    { 0.50f, 250.0f, 48.0f, {0.7f, 0.85f, 1.0f}, 0.0f, 0.0f,      0.0f, 0.45f },
    // LIGHT_SHROOMLIGHT     (250 lm — bioluminescent fungus, spectral uplift)
    { 0.50f, 250.0f, 48.0f, {1.0f, 0.6f, 0.3f}, 0.0f, 0.0f,       0.0f, 0.35f },
    // LIGHT_JACK_O_LANTERN  (50 lm — candle inside pumpkin, blackbody 1900K)
    { 0.50f, 50.0f, 48.0f, {1.0f, 0.7f, 0.3f}, 0.0f, 0.0f,    1900.0f, 0.00f },
    // LIGHT_END_ROD         (400 lm — fluorescent, slight uplift)
    { 0.05f, 400.0f, 48.0f, {0.95f, 0.9f, 1.0f}, 0.0f, 0.0f,      0.0f, 0.20f },
    // LIGHT_BEACON          (2000 lm — powerful focused light)
    { 0.30f, 2000.0f, 64.0f, {0.9f, 0.95f, 1.0f}, 0.0f, 0.0f,     0.0f, 0.00f },
    // LIGHT_OCHRE_FROGLIGHT (200 lm — bioluminescent warm)
    { 0.50f, 200.0f, 48.0f, {1.0f, 0.9f, 0.5f}, 0.0f, 0.0f,       0.0f, 0.00f },
    // LIGHT_VERDANT_FROGLIGHT (200 lm — bioluminescent green)
    { 0.50f, 200.0f, 48.0f, {0.4f, 1.0f, 0.5f}, 0.0f, 0.0f,       0.0f, 0.00f },
    // LIGHT_PEARL_FROGLIGHT (200 lm — bioluminescent pink)
    { 0.50f, 200.0f, 48.0f, {0.9f, 0.6f, 0.8f}, 0.0f, 0.0f,       0.0f, 0.00f },
    // LIGHT_REDSTONE_TORCH  (20 lm — dim deep red, spectral uplift)
    { 0.05f, 20.0f, 32.0f, {1.0f, 0.2f, 0.1f}, 0.35f, 0.06f,      0.0f, 0.45f },
    // LIGHT_REDSTONE_LAMP   (800 lm — red incandescent, spectral uplift)
    { 0.50f, 800.0f, 48.0f, {1.0f, 0.2f, 0.1f}, 0.0f, 0.0f,       0.0f, 0.45f },
    // LIGHT_CANDLE           (13 lm — single candle, blackbody 1800K)
    { 0.05f, 13.0f, 40.0f, {1.0f, 0.75f, 0.35f}, 0.3f, 0.12f,  1800.0f, 0.00f },
    // [18] UNUSED — formerly CANDLE_2
    { 0.05f, 13.0f, 40.0f, {1.0f, 0.75f, 0.35f}, 0.3f, 0.12f,     0.0f, 0.00f },
    // [19] UNUSED — formerly CANDLE_3
    { 0.05f, 13.0f, 40.0f, {1.0f, 0.75f, 0.35f}, 0.3f, 0.12f,     0.0f, 0.00f },
    // [20] UNUSED — formerly CANDLE_4
    { 0.05f, 13.0f, 40.0f, {1.0f, 0.75f, 0.35f}, 0.3f, 0.12f,     0.0f, 0.00f },
    // LIGHT_CAVE_VINES      (10 lm — dim bioluminescent berries)
    { 0.05f, 10.0f, 48.0f, {1.0f, 0.75f, 0.35f}, 0.35f, 0.03f,    0.0f, 0.00f },
    // LIGHT_GLOW_LICHEN     (5 lm — very dim bioluminescence)
    { 0.05f, 5.0f, 32.0f, {0.4f, 0.8f, 0.6f}, 0.0f, 0.0f,         0.0f, 0.00f },
    // LIGHT_FURNACE         (300 lm — fire behind grate, blackbody 1400K)
    { 0.30f, 300.0f, 40.0f, {1.0f, 0.5f, 0.2f}, 0.0f, 0.04f,  1400.0f, 0.00f },
    // LIGHT_BLAST_FURNACE   (500 lm — hotter smelting fire, blackbody 1600K)
    { 0.30f, 500.0f, 40.0f, {1.0f, 0.5f, 0.2f}, 0.0f, 0.04f,  1600.0f, 0.00f },
    // LIGHT_SMOKER           (200 lm — cooking fire, blackbody 1400K)
    { 0.30f, 200.0f, 40.0f, {1.0f, 0.5f, 0.2f}, 0.0f, 0.04f,  1400.0f, 0.00f },
    // LIGHT_ENDER_CHEST     (15 lm — subtle magical glow)
    { 0.15f, 15.0f, 32.0f, {0.3f, 0.7f, 0.5f}, 0.0f, 0.02f,       0.0f, 0.00f },
    // LIGHT_CRYING_OBSIDIAN (40 lm — purple fluorescent)
    { 0.30f, 40.0f, 40.0f, {0.6f, 0.2f, 0.9f}, 0.0f, 0.03f,       0.0f, 0.00f },
    // LIGHT_NETHER_PORTAL   (80 lm — plasma purple, spectral uplift)
    { 0.30f, 80.0f, 40.0f, {0.5f, 0.2f, 0.8f}, 0.0f, 0.02f,       0.0f, 0.50f },
    // LIGHT_CONDUIT          (500 lm — powerful aquatic light)
    { 0.30f, 500.0f, 48.0f, {0.9f, 0.95f, 1.0f}, 0.0f, 0.0f,      0.0f, 0.00f },
    // LIGHT_RESPAWN_ANCHOR_1 (20 lm — low charge)
    { 0.30f, 20.0f, 28.0f, {1.0f, 0.6f, 0.2f}, 0.0f, 0.0f,        0.0f, 0.00f },
    // LIGHT_RESPAWN_ANCHOR_2 (50 lm — medium-low)
    { 0.30f, 50.0f, 32.0f, {1.0f, 0.6f, 0.2f}, 0.0f, 0.0f,        0.0f, 0.00f },
    // LIGHT_RESPAWN_ANCHOR_3 (100 lm — medium-high)
    { 0.30f, 100.0f, 40.0f, {1.0f, 0.6f, 0.2f}, 0.0f, 0.0f,       0.0f, 0.00f },
    // LIGHT_RESPAWN_ANCHOR_4 (200 lm — full charge)
    { 0.30f, 200.0f, 48.0f, {1.0f, 0.6f, 0.2f}, 0.0f, 0.0f,       0.0f, 0.00f },
    // LIGHT_AMETHYST_CLUSTER (8 lm — crystal fluorescence, spectral uplift)
    { 0.05f, 8.0f, 24.0f, {0.7f, 0.5f, 0.9f}, 0.0f, 0.0f,         0.0f, 0.45f },
    // LIGHT_LARGE_AMETHYST_BUD (5 lm — smaller crystal, spectral uplift)
    { 0.05f, 5.0f, 24.0f, {0.7f, 0.5f, 0.9f}, 0.0f, 0.0f,         0.0f, 0.45f },
    // LIGHT_COPPER_BULB     (600 lm — electric bulb, blackbody 2700K + slight uplift)
    { 0.50f, 600.0f, 48.0f, {1.0f, 0.7f, 0.4f}, 0.0f, 0.0f,   2700.0f, 0.05f },
    // LIGHT_ENCHANTING_TABLE (3 lm — faint magical glow)
    { 0.05f, 3.0f, 16.0f, {0.5f, 0.8f, 0.5f}, 0.0f, 0.02f,        0.0f, 0.00f },
    // --- Formerly emissive-only blocks ---
    // LIGHT_LAVA             (1500 lm — flowing lava, blackbody 1323K basaltic ~1050°C)
    { 0.50f, 1500.0f, 48.0f, {1.0f, 0.4f, 0.1f}, 0.0f, 0.0f,  1323.0f, 0.00f },
    // LIGHT_FIRE             (500 lm — open flame, blackbody 1500K)
    { 0.15f, 500.0f, 48.0f, {1.0f, 0.6f, 0.2f}, 0.15f, 0.12f, 1500.0f, 0.00f },
    // LIGHT_SOUL_FIRE        (250 lm — blue flame, spectral uplift)
    { 0.15f, 250.0f, 40.0f, {0.3f, 0.8f, 0.9f}, 0.15f, 0.08f,     0.0f, 0.30f },
    // LIGHT_MAGMA_BLOCK      (80 lm — cooling lava, blackbody 1200K)
    { 0.50f, 80.0f, 24.0f, {1.0f, 0.3f, 0.1f}, 0.0f, 0.0f,    1200.0f, 0.00f },
    // LIGHT_SCULK_SENSOR     (2 lm — deep-sea bioluminescence, spectral uplift)
    { 0.15f, 2.0f, 16.0f, {0.2f, 0.5f, 0.5f}, 0.0f, 0.02f,        0.0f, 0.40f },
    // LIGHT_SCULK_CATALYST   (3 lm — slightly brighter sculk, spectral uplift)
    { 0.30f, 3.0f, 16.0f, {0.2f, 0.5f, 0.5f}, 0.0f, 0.02f,        0.0f, 0.40f },
    // LIGHT_SCULK_VEIN       (1 lm — very dim, spectral uplift)
    { 0.05f, 1.0f, 16.0f, {0.2f, 0.5f, 0.5f}, 0.0f, 0.0f,         0.0f, 0.40f },
    // LIGHT_SCULK            (1 lm — very dim, spectral uplift)
    { 0.50f, 1.0f, 16.0f, {0.15f, 0.4f, 0.4f}, 0.0f, 0.0f,        0.0f, 0.40f },
    // LIGHT_SCULK_SHRIEKER   (3 lm — slightly brighter sculk, spectral uplift)
    { 0.30f, 3.0f, 16.0f, {0.2f, 0.5f, 0.5f}, 0.0f, 0.0f,         0.0f, 0.40f },
    // LIGHT_BREWING_STAND    (5 lm — small pilot flame, blackbody 2000K)
    { 0.15f, 5.0f, 16.0f, {1.0f, 0.6f, 0.2f}, 0.0f, 0.0f,     2000.0f, 0.00f },
    // LIGHT_END_PORTAL       (30 lm — void energy, spectral uplift)
    { 0.50f, 30.0f, 48.0f, {0.3f, 0.1f, 0.5f}, 0.0f, 0.0f,        0.0f, 0.50f },
    // LIGHT_END_PORTAL_FRAME (3 lm — faint eye glow)
    { 0.15f, 3.0f, 16.0f, {0.4f, 0.7f, 0.4f}, 0.0f, 0.02f,        0.0f, 0.00f },
}};
