#ifndef HAFS_INT8_OPS_H_
#define HAFS_INT8_OPS_H_

#include <cstddef>
#include <cstdint>

#ifdef HAFS_ARM64
#include <arm_neon.h>
#endif

namespace hafs {

// =============================================================================
// ARM NEON Int8 SIMD Operations
// =============================================================================

#ifdef HAFS_ARM64

// NEON dot product for int8 vectors using vdotq_s32 (ARMv8.2+)
// Processes 16 int8 values at a time
inline int32_t DotProductInt8Neon(const int8_t* a, const int8_t* b, size_t n) {
  int32x4_t sum = vdupq_n_s32(0);

  size_t i = 0;

#if defined(__ARM_FEATURE_DOTPROD)
  // ARMv8.2+ with dot product extension
  for (; i + 16 <= n; i += 16) {
    int8x16_t va = vld1q_s8(a + i);
    int8x16_t vb = vld1q_s8(b + i);
    sum = vdotq_s32(sum, va, vb);
  }
#else
  // Fallback for older ARM (expand to int16, then int32)
  for (; i + 8 <= n; i += 8) {
    int8x8_t va = vld1_s8(a + i);
    int8x8_t vb = vld1_s8(b + i);
    int16x8_t prod = vmull_s8(va, vb);
    sum = vpadalq_s16(sum, prod);
  }
#endif

  // Horizontal sum
  int32_t result = vaddvq_s32(sum);

  // Handle remaining elements
  for (; i < n; ++i) {
    result += static_cast<int32_t>(a[i]) * static_cast<int32_t>(b[i]);
  }

  return result;
}

// NEON squared norm for int8 vector
inline int32_t SquaredNormInt8Neon(const int8_t* a, size_t n) {
  int32x4_t sum = vdupq_n_s32(0);

  size_t i = 0;

#if defined(__ARM_FEATURE_DOTPROD)
  for (; i + 16 <= n; i += 16) {
    int8x16_t va = vld1q_s8(a + i);
    sum = vdotq_s32(sum, va, va);
  }
#else
  for (; i + 8 <= n; i += 8) {
    int8x8_t va = vld1_s8(a + i);
    int16x8_t prod = vmull_s8(va, va);
    sum = vpadalq_s16(sum, prod);
  }
#endif

  int32_t result = vaddvq_s32(sum);

  for (; i < n; ++i) {
    result += static_cast<int32_t>(a[i]) * static_cast<int32_t>(a[i]);
  }

  return result;
}

#else  // Scalar fallback

inline int32_t DotProductInt8Neon(const int8_t* a, const int8_t* b, size_t n) {
  int32_t sum = 0;
  for (size_t i = 0; i < n; ++i) {
    sum += static_cast<int32_t>(a[i]) * static_cast<int32_t>(b[i]);
  }
  return sum;
}

inline int32_t SquaredNormInt8Neon(const int8_t* a, size_t n) {
  int32_t sum = 0;
  for (size_t i = 0; i < n; ++i) {
    sum += static_cast<int32_t>(a[i]) * static_cast<int32_t>(a[i]);
  }
  return sum;
}

#endif  // HAFS_ARM64

}  // namespace hafs

#endif  // HAFS_INT8_OPS_H_
