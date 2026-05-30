#ifndef ALPHA_MODE_GLSL
#define ALPHA_MODE_GLSL

const uint ALPHA_MODE_OPAQUE = 0u;
const uint ALPHA_MODE_CUTOUT = 1u;
const uint ALPHA_MODE_TRANSPARENT = 2u;

const float CUTOUT_ALPHA_THRESHOLD = 0.5;

float resolveSurfaceAlpha(float alpha, uint alphaMode) {
    alpha = clamp(alpha, 0.0, 1.0);

    if (alphaMode == ALPHA_MODE_OPAQUE) { return 1.0; }

    if (alphaMode == ALPHA_MODE_CUTOUT) {
        return alpha >= CUTOUT_ALPHA_THRESHOLD ? 1.0 : 0.0;
    }

    return alpha;
}

#endif
