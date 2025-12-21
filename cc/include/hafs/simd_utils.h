#ifndef HAFS_SIMD_UTILS_H_
#define HAFS_SIMD_UTILS_H_

#include <cmath>
#include <cstddef>

#ifdef HAFS_ARM64
#include <arm_neon.h>
#endif

namespace hafs {

// =============================================================================
// ARM NEON SIMD Optimized Operations
// =============================================================================

#ifdef HAFS_ARM64

// NEON-optimized dot product for ARM64.
// Processes 4 floats at a time using fused multiply-add.
inline float DotProductNeon(const float* a, const float* b, size_t n) {
  float32x4_t sum = vdupq_n_f32(0.0f);

  // Process 4 elements at a time
  size_t i = 0;
  for (; i + 4 <= n; i += 4) {
    float32x4_t va = vld1q_f32(a + i);
    float32x4_t vb = vld1q_f32(b + i);
    sum = vmlaq_f32(sum, va, vb);  // fused multiply-add
  }

  // Horizontal sum of the 4 lanes
  float result = vaddvq_f32(sum);

  // Handle remaining elements
  for (; i < n; ++i) {
    result += a[i] * b[i];
  }

  return result;
}

// NEON-optimized squared L2 norm.
inline float SquaredNormNeon(const float* a, size_t n) {
  float32x4_t sum = vdupq_n_f32(0.0f);

  size_t i = 0;
  for (; i + 4 <= n; i += 4) {
    float32x4_t va = vld1q_f32(a + i);
    sum = vmlaq_f32(sum, va, va);
  }

  float result = vaddvq_f32(sum);

  for (; i < n; ++i) {
    result += a[i] * a[i];
  }

  return result;
}

// NEON-optimized L2 distance squared.
inline float L2DistanceSquaredNeon(const float* a, const float* b, size_t n) {
  float32x4_t sum = vdupq_n_f32(0.0f);

  size_t i = 0;
  for (; i + 4 <= n; i += 4) {
    float32x4_t va = vld1q_f32(a + i);
    float32x4_t vb = vld1q_f32(b + i);
    float32x4_t diff = vsubq_f32(va, vb);
    sum = vmlaq_f32(sum, diff, diff);
  }

  float result = vaddvq_f32(sum);

  for (; i < n; ++i) {
    float diff = a[i] - b[i];
    result += diff * diff;
  }

  return result;
}

// NEON-optimized vector normalization in place.
inline void NormalizeVectorNeon(float* vec, size_t n) {
  float norm = std::sqrt(SquaredNormNeon(vec, n));
  if (norm < 1e-8f) return;  // Avoid division by zero

  float inv_norm = 1.0f / norm;
  float32x4_t scale = vdupq_n_f32(inv_norm);

  size_t i = 0;
  for (; i + 4 <= n; i += 4) {
    float32x4_t v = vld1q_f32(vec + i);
    v = vmulq_f32(v, scale);
    vst1q_f32(vec + i, v);
  }

  for (; i < n; ++i) {
    vec[i] *= inv_norm;
  }
}

#else  // Scalar fallback for non-ARM64

inline float DotProductNeon(const float* a, const float* b, size_t n) {
  float sum = 0.0f;
  for (size_t i = 0; i < n; ++i) {
    sum += a[i] * b[i];
  }
  return sum;
}

inline float SquaredNormNeon(const float* a, size_t n) {
  float sum = 0.0f;
  for (size_t i = 0; i < n; ++i) {
    sum += a[i] * a[i];
  }
  return sum;
}

inline float L2DistanceSquaredNeon(const float* a, const float* b, size_t n) {
  float sum = 0.0f;
  for (size_t i = 0; i < n; ++i) {
    float diff = a[i] - b[i];
    sum += diff * diff;
  }
  return sum;
}

inline void NormalizeVectorNeon(float* vec, size_t n) {
  float norm = std::sqrt(SquaredNormNeon(vec, n));
  if (norm < 1e-8f) return;
  float inv_norm = 1.0f / norm;
  for (size_t i = 0; i < n; ++i) {
    vec[i] *= inv_norm;
  }
}

#endif  // HAFS_ARM64

}  // namespace hafs

#endif  // HAFS_SIMD_UTILS_H_
