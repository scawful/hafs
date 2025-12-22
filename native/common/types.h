#ifndef HAFS_TYPES_H_
#define HAFS_TYPES_H_

#include <cstdint>

namespace hafs {

// =============================================================================
// Common Types for Native Module
// =============================================================================

// Quantization parameters for int8 embeddings
struct QuantizationParams {
  float scale;
  int32_t zero_point;
};

// Half-precision float type alias
// Note: ARM NEON has native float16 support via float16_t
#ifdef HAFS_ARM64
using float16_t = __fp16;
#else
using float16_t = uint16_t;  // Storage only on non-ARM
#endif

}  // namespace hafs

#endif  // HAFS_TYPES_H_
