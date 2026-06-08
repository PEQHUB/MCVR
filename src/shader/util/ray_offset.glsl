#ifndef RAY_OFFSET_GLSL
#define RAY_OFFSET_GLSL

// NVIDIA Ray Tracing Gems, Chapter 6:
// "A Fast and Robust Method for Avoiding Self-Intersection"
// by Carsten Wächter & Nikolaus Binder
//
// Offsets ray origin using float bit manipulation to find the minimum
// displacement that places the origin on the correct side of the surface.
// Use with tMin = 0 for shadow/bounce rays to avoid skipping nearby geometry
// at block corners.
vec3 offset_ray(vec3 p, vec3 n) {
    const float origin      = 1.0 / 32.0;
    const float float_scale = 1.0 / 65536.0;
    const float int_scale   = 256.0;

    ivec3 of_i = ivec3(int_scale * n.x, int_scale * n.y, int_scale * n.z);

    vec3 p_i = vec3(
        intBitsToFloat(floatBitsToInt(p.x) + ((p.x < 0) ? -of_i.x : of_i.x)),
        intBitsToFloat(floatBitsToInt(p.y) + ((p.y < 0) ? -of_i.y : of_i.y)),
        intBitsToFloat(floatBitsToInt(p.z) + ((p.z < 0) ? -of_i.z : of_i.z))
    );

    return vec3(
        abs(p.x) < origin ? p.x + float_scale * n.x : p_i.x,
        abs(p.y) < origin ? p.y + float_scale * n.y : p_i.y,
        abs(p.z) < origin ? p.z + float_scale * n.z : p_i.z
    );
}

#endif
