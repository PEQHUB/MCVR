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
    float intensity;       // brightness factor derived from light level / 15.0
    float radius;          // max effective range in blocks
    glm::vec3 color;       // pre-computed emissive RGB
    float yOffset;         // vertical offset from block center (0.35 = flame tip)
    float flickerStrength; // procedural flicker amplitude (0.0=none, 0.08=torch, 0.15=candle)
};

// Indexed by LightTypeId. Intensity = lightLevel / 15.0, radius ~ lightLevel * 3.2.
// yOffset: vertical shift from block center (0.35 = flame tip for torches/candles)
inline constexpr std::array<LightTypeDef, LIGHT_TYPE_COUNT> LIGHT_DEFS = {{
    // LIGHT_TORCH          (level 14, spherical point light, warm orange)
    { 0.0f, 0.25f, 48.0f, {1.0f, 0.7f, 0.3f}, 0.07f, 0.08f },
    // LIGHT_SOUL_TORCH     (level 10, spherical point light, cyan)
    { 0.0f, 0.42f, 40.0f, {0.3f, 0.8f, 0.9f}, 0.12f, 0.04f },
    // LIGHT_LANTERN         (level 15, small, warm orange)
    { 0.15f, 1.000f, 48.0f, {1.0f, 0.7f, 0.3f}, -0.1f, 0.03f },
    // LIGHT_SOUL_LANTERN    (level 10, small, cyan)
    { 0.15f, 0.667f, 40.0f, {0.3f, 0.8f, 0.9f}, -0.1f, 0.02f },
    // LIGHT_CAMPFIRE        (level 15, small, warm orange)
    { 0.15f, 1.000f, 48.0f, {1.0f, 0.7f, 0.3f}, 0.15f, 0.05f },
    // LIGHT_SOUL_CAMPFIRE   (level 10, small, cyan)
    { 0.15f, 0.667f, 40.0f, {0.3f, 0.8f, 0.9f}, 0.15f, 0.04f },
    // LIGHT_GLOWSTONE       (level 15, full, golden)
    { 0.50f, 1.000f, 48.0f, {1.0f, 0.85f, 0.5f}, 0.0f, 0.0f },
    // LIGHT_SEA_LANTERN     (level 15, full, blue-white)
    { 0.50f, 1.000f, 48.0f, {0.7f, 0.85f, 1.0f}, 0.0f, 0.0f },
    // LIGHT_SHROOMLIGHT     (level 15, full, orange)
    { 0.50f, 1.000f, 48.0f, {1.0f, 0.6f, 0.3f}, 0.0f, 0.0f },
    // LIGHT_JACK_O_LANTERN  (level 15, full, warm orange)
    { 0.50f, 1.000f, 48.0f, {1.0f, 0.7f, 0.3f}, 0.0f, 0.0f },
    // LIGHT_END_ROD         (level 14, point, cool white)
    { 0.05f, 0.933f, 48.0f, {0.95f, 0.9f, 1.0f}, 0.0f, 0.0f },
    // LIGHT_BEACON          (level 15, medium, bright white)
    { 0.30f, 1.000f, 64.0f, {0.9f, 0.95f, 1.0f}, 0.0f, 0.0f },
    // LIGHT_OCHRE_FROGLIGHT (level 15, full, yellow)
    { 0.50f, 1.000f, 48.0f, {1.0f, 0.9f, 0.5f}, 0.0f, 0.0f },
    // LIGHT_VERDANT_FROGLIGHT (level 15, full, green)
    { 0.50f, 1.000f, 48.0f, {0.4f, 1.0f, 0.5f}, 0.0f, 0.0f },
    // LIGHT_PEARL_FROGLIGHT (level 15, full, pink)
    { 0.50f, 1.000f, 48.0f, {0.9f, 0.6f, 0.8f}, 0.0f, 0.0f },
    // LIGHT_REDSTONE_TORCH  (level 7, point, deep red)
    { 0.05f, 0.467f, 32.0f, {1.0f, 0.2f, 0.1f}, 0.35f, 0.06f },
    // LIGHT_REDSTONE_LAMP   (level 15, full, deep red)
    { 0.50f, 1.000f, 48.0f, {1.0f, 0.2f, 0.1f}, 0.0f, 0.0f },
    // LIGHT_CANDLE           (consolidated, point, warm — intensity boosted for visibility)
    { 0.05f, 0.600f, 40.0f, {1.0f, 0.75f, 0.35f}, 0.3f, 0.12f },
    // [18] UNUSED — formerly CANDLE_2
    { 0.05f, 0.600f, 40.0f, {1.0f, 0.75f, 0.35f}, 0.3f, 0.12f },
    // [19] UNUSED — formerly CANDLE_3
    { 0.05f, 0.600f, 40.0f, {1.0f, 0.75f, 0.35f}, 0.3f, 0.12f },
    // [20] UNUSED — formerly CANDLE_4
    { 0.05f, 0.600f, 40.0f, {1.0f, 0.75f, 0.35f}, 0.3f, 0.12f },
    // LIGHT_CAVE_VINES      (level 14, point, warm)
    { 0.05f, 0.933f, 48.0f, {1.0f, 0.75f, 0.35f}, 0.35f, 0.03f },
    // LIGHT_GLOW_LICHEN     (level 7, point, teal)
    { 0.05f, 0.467f, 32.0f, {0.4f, 0.8f, 0.6f}, 0.0f, 0.0f },
    // LIGHT_FURNACE         (level 13, medium, fire orange)
    { 0.30f, 0.867f, 40.0f, {1.0f, 0.5f, 0.2f}, 0.0f, 0.04f },
    // LIGHT_BLAST_FURNACE   (level 13, medium, fire orange)
    { 0.30f, 0.867f, 40.0f, {1.0f, 0.5f, 0.2f}, 0.0f, 0.04f },
    // LIGHT_SMOKER           (level 13, medium, fire orange)
    { 0.30f, 0.867f, 40.0f, {1.0f, 0.5f, 0.2f}, 0.0f, 0.04f },
    // LIGHT_ENDER_CHEST     (level 7, small, teal)
    { 0.15f, 0.467f, 32.0f, {0.3f, 0.7f, 0.5f}, 0.0f, 0.02f },
    // LIGHT_CRYING_OBSIDIAN (level 10, medium, purple)
    { 0.30f, 0.667f, 40.0f, {0.6f, 0.2f, 0.9f}, 0.0f, 0.03f },
    // LIGHT_NETHER_PORTAL   (level 11, medium, purple)
    { 0.30f, 0.733f, 40.0f, {0.5f, 0.2f, 0.8f}, 0.0f, 0.02f },
    // LIGHT_CONDUIT          (level 15, medium, bright white)
    { 0.30f, 1.000f, 48.0f, {0.9f, 0.95f, 1.0f}, 0.0f, 0.0f },
    // LIGHT_RESPAWN_ANCHOR_1 (level 4, medium, warm amber)
    { 0.30f, 0.267f, 28.0f, {1.0f, 0.6f, 0.2f}, 0.0f, 0.0f },
    // LIGHT_RESPAWN_ANCHOR_2 (level 7, medium, warm amber)
    { 0.30f, 0.467f, 32.0f, {1.0f, 0.6f, 0.2f}, 0.0f, 0.0f },
    // LIGHT_RESPAWN_ANCHOR_3 (level 10, medium, warm amber)
    { 0.30f, 0.667f, 40.0f, {1.0f, 0.6f, 0.2f}, 0.0f, 0.0f },
    // LIGHT_RESPAWN_ANCHOR_4 (level 15, medium, warm amber)
    { 0.30f, 1.000f, 48.0f, {1.0f, 0.6f, 0.2f}, 0.0f, 0.0f },
    // LIGHT_AMETHYST_CLUSTER (level 5, point, purple)
    { 0.05f, 0.333f, 24.0f, {0.7f, 0.5f, 0.9f}, 0.0f, 0.0f },
    // LIGHT_LARGE_AMETHYST_BUD (level 4, point, purple)
    { 0.05f, 0.267f, 24.0f, {0.7f, 0.5f, 0.9f}, 0.0f, 0.0f },
    // LIGHT_COPPER_BULB     (level 15, full, warm)
    { 0.50f, 1.000f, 48.0f, {1.0f, 0.7f, 0.4f}, 0.0f, 0.0f },
    // LIGHT_ENCHANTING_TABLE (level 1, point, green-teal)
    { 0.05f, 0.067f, 16.0f, {0.5f, 0.8f, 0.5f}, 0.0f, 0.02f },
    // --- New: formerly emissive-only blocks ---
    // LIGHT_LAVA             (level 15, full, deep orange)
    { 0.50f, 1.000f, 48.0f, {1.0f, 0.4f, 0.1f}, 0.0f, 0.0f },
    // LIGHT_FIRE             (level 15, small, orange)
    { 0.15f, 1.000f, 48.0f, {1.0f, 0.6f, 0.2f}, 0.15f, 0.12f },
    // LIGHT_SOUL_FIRE        (level 10, small, cyan)
    { 0.15f, 0.667f, 40.0f, {0.3f, 0.8f, 0.9f}, 0.15f, 0.08f },
    // LIGHT_MAGMA_BLOCK      (level 3, full, deep red-orange)
    { 0.50f, 0.200f, 24.0f, {1.0f, 0.3f, 0.1f}, 0.0f, 0.0f },
    // LIGHT_SCULK_SENSOR     (level 1, small, dark teal)
    { 0.15f, 0.067f, 16.0f, {0.2f, 0.5f, 0.5f}, 0.0f, 0.02f },
    // LIGHT_SCULK_CATALYST   (level 1, medium, dark teal)
    { 0.30f, 0.067f, 16.0f, {0.2f, 0.5f, 0.5f}, 0.0f, 0.02f },
    // LIGHT_SCULK_VEIN       (level 1, point, dark teal)
    { 0.05f, 0.067f, 16.0f, {0.2f, 0.5f, 0.5f}, 0.0f, 0.0f },
    // LIGHT_SCULK            (level 1, full, dark teal)
    { 0.50f, 0.067f, 16.0f, {0.15f, 0.4f, 0.4f}, 0.0f, 0.0f },
    // LIGHT_SCULK_SHRIEKER   (level 1, medium, dark teal)
    { 0.30f, 0.067f, 16.0f, {0.2f, 0.5f, 0.5f}, 0.0f, 0.0f },
    // LIGHT_BREWING_STAND    (level 1, small, warm amber)
    { 0.15f, 0.067f, 16.0f, {1.0f, 0.6f, 0.2f}, 0.0f, 0.0f },
    // LIGHT_END_PORTAL       (level 15, full, deep purple)
    { 0.50f, 1.000f, 48.0f, {0.3f, 0.1f, 0.5f}, 0.0f, 0.0f },
    // LIGHT_END_PORTAL_FRAME (level 1, small, green-teal)
    { 0.15f, 0.067f, 16.0f, {0.4f, 0.7f, 0.4f}, 0.0f, 0.02f },
}};
