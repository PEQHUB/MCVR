#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_nonuniform_qualifier : require

#define ADV_ALLOW_VOLUMETRIC_CLOUD_MISS 0
#include "common/world_miss_eval.glsl"
