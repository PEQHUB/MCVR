// Compile miniz implementation as C++ in this translation unit.
// Only inflate (decompression) is needed for Anvil region files.
#define MINIZ_NO_STDIO
#define MINIZ_NO_ARCHIVE_APIS
#define MINIZ_NO_ZLIB_COMPATIBLE_NAMES

extern "C" {
#include "miniz.c"
#include "miniz_tinfl.c"
#include "miniz_tdef.c"
}
