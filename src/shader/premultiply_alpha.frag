#version 460

// Premultiplies overlay alpha for DWM composition.
// Input: R8G8B8A8_SRGB overlay (sampler auto-linearizes).
// Output: R8G8B8A8_SRGB render target (auto sRGB-encodes on write).
// Result: sRGB-encoded premultiplied alpha -- exactly what DWM expects
// from DXGI_FORMAT_R8G8B8A8_UNORM with DXGI_ALPHA_MODE_PREMULTIPLIED.

layout(set = 0, binding = 0) uniform sampler2D overlayUI;

layout(location = 0) in vec2 fragTexCoord;
layout(location = 0) out vec4 fragColor;

void main() {
    vec4 ui = texture(overlayUI, fragTexCoord); // linearized by SRGB sampler
    fragColor = vec4(ui.rgb * ui.a, ui.a);      // premultiply in linear space
}
