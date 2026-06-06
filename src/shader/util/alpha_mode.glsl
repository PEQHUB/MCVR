#ifndef ALPHA_MODE_GLSL
#define ALPHA_MODE_GLSL

const uint ALPHA_MODE_OPAQUE = 0u;
const uint ALPHA_MODE_CUTOUT = 1u;
const uint ALPHA_MODE_TRANSPARENT = 2u;

// OptiFine texture.properties defaults alpha.cutout to 16 on an 8-bit alpha channel.
const float CUTOUT_ALPHA_THRESHOLD = 16.0 / 255.0;

float resolveSurfaceAlpha(float alpha, uint alphaMode) {
    alpha = clamp(alpha, 0.0, 1.0);

    if (alphaMode == ALPHA_MODE_OPAQUE) { return 1.0; }

    if (alphaMode == ALPHA_MODE_CUTOUT) {
        return alpha >= CUTOUT_ALPHA_THRESHOLD ? 1.0 : 0.0;
    }

    return alpha;
}

#endif
