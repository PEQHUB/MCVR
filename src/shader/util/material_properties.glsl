// material_properties.glsl — Unified material class SSBO via Buffer Device Address
//
// 256 contiguous MaterialClassEntry entries (128 bytes each, 32 KB total).
// Indexed by materialType - 1 (vertex emissiveBlockType bits 8-15).
// Accessed via BDA from push constant — avoids descriptor lookup overhead.
// Follows the same pattern as SHARC (BDA in push constant).
//
// Requires: shared.hpp included before this file (for MaterialClassMapping struct).
// Requires: GL_EXT_buffer_reference2 + GL_EXT_shader_explicit_arithmetic_types_int64

#ifndef MATERIAL_PROPERTIES_GLSL
#define MATERIAL_PROPERTIES_GLSL

// BDA buffer reference for material class mapping
layout(std430, buffer_reference, buffer_reference_align = 16) readonly buffer MaterialClassBufferRef {
    MaterialClassMapping materialClassMapping;
};

// Legacy descriptor binding kept for compatibility (SHARC update pass, etc.)
layout(set = 1, binding = 11) readonly buffer MaterialClassBuffer {
    MaterialClassMapping materialClassMappingDescriptor;
};

#endif // MATERIAL_PROPERTIES_GLSL
