#ifndef ADV_RUNTIME_PRIMARY_SECONDARY_ARRAYS_GLSL
#define ADV_RUNTIME_PRIMARY_SECONDARY_ARRAYS_GLSL

ivec3 rtPrimarySecondaryArrayCoord(ivec2 pixel, int layer) {
    return ivec3(pixel, layer);
}

#ifdef ADV_RUNTIME_PRIMARY_SECONDARY_READONLY
#    define ADV_RUNTIME_PRIMARY_SECONDARY_QUAL readonly
#else
#    define ADV_RUNTIME_PRIMARY_SECONDARY_QUAL
#endif

layout(set = 5, binding = 3, rgba32f) uniform ADV_RUNTIME_PRIMARY_SECONDARY_QUAL image2DArray primarySurfaceCacheImage;
layout(set = 5, binding = 4, rgba16f) uniform ADV_RUNTIME_PRIMARY_SECONDARY_QUAL image2DArray primaryRayStateImage;
layout(set = 5, binding = 5, rgba32f) uniform ADV_RUNTIME_PRIMARY_SECONDARY_QUAL image2DArray secondarySurfaceCacheImage;
layout(set = 5, binding = 6, rgba32f) uniform ADV_RUNTIME_PRIMARY_SECONDARY_QUAL image2DArray secondaryRayStateImage;

vec4 loadPrimarySurfaceCacheLayer(ivec2 pixel, int layer) {
    return imageLoad(primarySurfaceCacheImage, rtPrimarySecondaryArrayCoord(pixel, layer));
}

vec4 loadPrimaryRayStateLayer(ivec2 pixel, int layer) {
    return imageLoad(primaryRayStateImage, rtPrimarySecondaryArrayCoord(pixel, layer));
}

vec4 loadSecondarySurfaceCacheLayer(ivec2 pixel, int layer) {
    return imageLoad(secondarySurfaceCacheImage, rtPrimarySecondaryArrayCoord(pixel, layer));
}

vec4 loadSecondaryRayStateLayer(ivec2 pixel, int layer) {
    return imageLoad(secondaryRayStateImage, rtPrimarySecondaryArrayCoord(pixel, layer));
}

#ifndef ADV_RUNTIME_PRIMARY_SECONDARY_READONLY
void storePrimarySurfaceCacheLayer(ivec2 pixel, int layer, vec4 value) {
    imageStore(primarySurfaceCacheImage, rtPrimarySecondaryArrayCoord(pixel, layer), value);
}

void storePrimaryRayStateLayer(ivec2 pixel, int layer, vec4 value) {
    imageStore(primaryRayStateImage, rtPrimarySecondaryArrayCoord(pixel, layer), value);
}

void storeSecondarySurfaceCacheLayer(ivec2 pixel, int layer, vec4 value) {
    imageStore(secondarySurfaceCacheImage, rtPrimarySecondaryArrayCoord(pixel, layer), value);
}

void storeSecondaryRayStateLayer(ivec2 pixel, int layer, vec4 value) {
    imageStore(secondaryRayStateImage, rtPrimarySecondaryArrayCoord(pixel, layer), value);
}
#endif

#undef ADV_RUNTIME_PRIMARY_SECONDARY_QUAL

#endif
