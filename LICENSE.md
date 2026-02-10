This project is primarily licensed under the GNU General Public License v3.0.

However, the following files / directories are licensed under the Apache License v2.0:

* NRD wrapper is modified based on https://github.com/nvpro-samples/vk_denoise_nrd/tree/main
  * src/core/render/modules/world/nrd/nrd_wrapper.hpp
  * src/core/render/modules/world/nrd/nrd_wrapper.cpp
* DLSS wrapper is modified based on https://github.com/nvpro-samples/vk_denoise_dlssrr/tree/main
  * src/core/render/modules/world/dlss/dlss_wrapper.hpp
  * src/core/render/modules/world/dlss/dlss_wrapper.cpp
* Random functions in shaders
  * src/shader/util/random.glsl

The following files / directories are licensed under the MIT Licence:

* FSR Setup from AMD's FidelityFX-SDK
  * src/core/render/modules/world/fsr_upscaler/fsr_setup.hpp
  * src/core/render/modules/world/fsr_upscaler/fsr_setup.cpp
* Disney BSDF is modified based on https://github.com/knightcrawler25/GLSL-PathTracer/tree/master
  * src/shader/util/disney.glsl
