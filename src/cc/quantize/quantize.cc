#include "quantize.h"
#include "int8_ops.h"

#include <algorithm>
#include <cmath>
#include <limits>

#ifdef HAFS_ARM64
#include <arm_neon.h>
#endif

namespace hafs {

// =============================================================================
// Float32 <-> Int8 Quantization (Symmetric)
// =============================================================================

Int8QuantParams QuantizeF32ToInt8(const float* input, int8_t* output, size_t n) {
  if (n == 0) {
    return {1.0f, 0};
  }

  // Find max absolute value
  float max_abs = 0.0f;

#ifdef HAFS_ARM64
  float32x4_t max_vec = vdupq_n_f32(0.0f);
  size_t i = 0;
  for (; i + 4 <= n; i += 4) {
    float32x4_t v = vld1q_f32(input + i);
    max_vec = vmaxq_f32(max_vec, vabsq_f32(v));
  }
  float max_arr[4];
  vst1q_f32(max_arr, max_vec);
  max_abs = std::max({max_arr[0], max_arr[1], max_arr[2], max_arr[3]});
  for (; i < n; ++i) {
    max_abs = std::max(max_abs, std::abs(input[i]));
  }
#else
  for (size_t i = 0; i < n; ++i) {
    max_abs = std::max(max_abs, std::abs(input[i]));
  }
#endif

  // Compute scale (symmetric: -127 to 127)
  float scale = (max_abs > 1e-8f) ? (max_abs / 127.0f) : 1.0f;
  float inv_scale = 1.0f / scale;

  // Quantize
#ifdef HAFS_ARM64
  float32x4_t scale_vec = vdupq_n_f32(inv_scale);
  i = 0;
  for (; i + 4 <= n; i += 4) {
    float32x4_t v = vld1q_f32(input + i);
    float32x4_t scaled = vmulq_f32(v, scale_vec);

    // Round and clamp
    int32x4_t rounded = vcvtnq_s32_f32(scaled);
    int16x4_t narrowed = vqmovn_s32(rounded);
    int8x8_t clamped = vqmovn_s16(vcombine_s16(narrowed, narrowed));

    // Store first 4 elements
    vst1_lane_s8(output + i, clamped, 0);
    vst1_lane_s8(output + i + 1, clamped, 1);
    vst1_lane_s8(output + i + 2, clamped, 2);
    vst1_lane_s8(output + i + 3, clamped, 3);
  }
  for (; i < n; ++i) {
    float scaled = input[i] * inv_scale;
    int32_t rounded = static_cast<int32_t>(std::round(scaled));
    output[i] = static_cast<int8_t>(std::clamp(rounded, -127, 127));
  }
#else
  for (size_t i = 0; i < n; ++i) {
    float scaled = input[i] * inv_scale;
    int32_t rounded = static_cast<int32_t>(std::round(scaled));
    output[i] = static_cast<int8_t>(std::clamp(rounded, -127, 127));
  }
#endif

  return {scale, 0};
}

void DequantizeInt8ToF32(const int8_t* input, float* output, size_t n,
                          const Int8QuantParams& params) {
  float scale = params.scale;

#ifdef HAFS_ARM64
  float32x4_t scale_vec = vdupq_n_f32(scale);
  size_t i = 0;
  for (; i + 4 <= n; i += 4) {
    // Load 4 int8 values and convert to float
    int8x8_t vals = vld1_s8(input + i);
    int16x8_t vals16 = vmovl_s8(vals);
    int32x4_t vals32 = vmovl_s16(vget_low_s16(vals16));
    float32x4_t valsf = vcvtq_f32_s32(vals32);
    float32x4_t result = vmulq_f32(valsf, scale_vec);
    vst1q_f32(output + i, result);
  }
  for (; i < n; ++i) {
    output[i] = static_cast<float>(input[i]) * scale;
  }
#else
  for (size_t i = 0; i < n; ++i) {
    output[i] = static_cast<float>(input[i]) * scale;
  }
#endif
}

// =============================================================================
// Float32 <-> Float16 Conversion
// =============================================================================

void QuantizeF32ToF16(const float* input, uint16_t* output, size_t n) {
#ifdef HAFS_ARM64
  size_t i = 0;
  for (; i + 4 <= n; i += 4) {
    float32x4_t v = vld1q_f32(input + i);
    float16x4_t h = vcvt_f16_f32(v);
    vst1_f16(reinterpret_cast<__fp16*>(output + i), h);
  }
  for (; i < n; ++i) {
    // Single conversion
    float32x4_t v = vdupq_n_f32(input[i]);
    float16x4_t h = vcvt_f16_f32(v);
    output[i] = vget_lane_u16(vreinterpret_u16_f16(h), 0);
  }
#else
  // Simple IEEE 754 half-precision conversion (software fallback)
  for (size_t i = 0; i < n; ++i) {
    uint32_t bits;
    std::memcpy(&bits, &input[i], sizeof(float));

    uint32_t sign = (bits >> 31) & 0x1;
    int32_t exp = ((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t frac = bits & 0x7FFFFF;

    uint16_t result;
    if (exp <= 0) {
      result = static_cast<uint16_t>(sign << 15);  // Zero/denormal
    } else if (exp >= 31) {
      result = static_cast<uint16_t>((sign << 15) | 0x7C00);  // Inf
    } else {
      result = static_cast<uint16_t>((sign << 15) | (exp << 10) | (frac >> 13));
    }
    output[i] = result;
  }
#endif
}

void DequantizeF16ToF32(const uint16_t* input, float* output, size_t n) {
#ifdef HAFS_ARM64
  size_t i = 0;
  for (; i + 4 <= n; i += 4) {
    float16x4_t h = vld1_f16(reinterpret_cast<const __fp16*>(input + i));
    float32x4_t v = vcvt_f32_f16(h);
    vst1q_f32(output + i, v);
  }
  for (; i < n; ++i) {
    float16x4_t h = vreinterpret_f16_u16(vdup_n_u16(input[i]));
    float32x4_t v = vcvt_f32_f16(h);
    output[i] = vgetq_lane_f32(vcombine_f32(vget_low_f32(v), vget_low_f32(v)), 0);
  }
#else
  // Software fallback
  for (size_t i = 0; i < n; ++i) {
    uint16_t bits = input[i];
    uint32_t sign = (bits >> 15) & 0x1;
    uint32_t exp = (bits >> 10) & 0x1F;
    uint32_t frac = bits & 0x3FF;

    float result;
    if (exp == 0) {
      result = (sign ? -1.0f : 1.0f) * (frac / 1024.0f) * std::pow(2.0f, -14);
    } else if (exp == 31) {
      result = (frac == 0) ? (sign ? -INFINITY : INFINITY) : NAN;
    } else {
      uint32_t f32_bits = (sign << 31) | ((exp - 15 + 127) << 23) | (frac << 13);
      std::memcpy(&result, &f32_bits, sizeof(float));
    }
    output[i] = result;
  }
#endif
}

// =============================================================================
// Quantized Dot Products
// =============================================================================

int32_t DotProductInt8(const int8_t* a, const int8_t* b, size_t n) {
  return DotProductInt8Neon(a, b, n);
}

float CosineSimilarityInt8(const int8_t* a, const int8_t* b, size_t n,
                           const Int8QuantParams& params_a,
                           const Int8QuantParams& params_b) {
  int32_t dot = DotProductInt8(a, b, n);
  int32_t norm_a_sq = SquaredNormInt8Neon(a, n);
  int32_t norm_b_sq = SquaredNormInt8Neon(b, n);

  if (norm_a_sq == 0 || norm_b_sq == 0) {
    return 0.0f;
  }

  // Convert to float and compute cosine
  float fdot = static_cast<float>(dot) * params_a.scale * params_b.scale;
  float fnorm_a = std::sqrt(static_cast<float>(norm_a_sq)) * params_a.scale;
  float fnorm_b = std::sqrt(static_cast<float>(norm_b_sq)) * params_b.scale;

  return fdot / (fnorm_a * fnorm_b);
}

// =============================================================================
// Python Bindings
// =============================================================================

std::tuple<py::array_t<int8_t>, float, int32_t> PyQuantizeF32ToInt8(
    py::array_t<float> input) {
  auto buf = input.request();
  size_t n = buf.size;

  auto output = py::array_t<int8_t>(n);
  auto out_buf = output.request();

  Int8QuantParams params = QuantizeF32ToInt8(
      static_cast<float*>(buf.ptr),
      static_cast<int8_t*>(out_buf.ptr),
      n);

  return std::make_tuple(output, params.scale, params.zero_point);
}

py::array_t<float> PyDequantizeInt8ToF32(py::array_t<int8_t> input,
                                          float scale, int32_t zero_point) {
  auto buf = input.request();
  size_t n = buf.size;

  auto output = py::array_t<float>(n);
  auto out_buf = output.request();

  Int8QuantParams params{scale, zero_point};
  DequantizeInt8ToF32(
      static_cast<int8_t*>(buf.ptr),
      static_cast<float*>(out_buf.ptr),
      n, params);

  return output;
}

py::array_t<uint16_t> PyQuantizeF32ToF16(py::array_t<float> input) {
  auto buf = input.request();
  size_t n = buf.size;

  auto output = py::array_t<uint16_t>(n);
  auto out_buf = output.request();

  QuantizeF32ToF16(
      static_cast<float*>(buf.ptr),
      static_cast<uint16_t*>(out_buf.ptr),
      n);

  return output;
}

py::array_t<float> PyDequantizeF16ToF32(py::array_t<uint16_t> input) {
  auto buf = input.request();
  size_t n = buf.size;

  auto output = py::array_t<float>(n);
  auto out_buf = output.request();

  DequantizeF16ToF32(
      static_cast<uint16_t*>(buf.ptr),
      static_cast<float*>(out_buf.ptr),
      n);

  return output;
}

float PyCosineSimilarityInt8(py::array_t<int8_t> a, py::array_t<int8_t> b,
                              float scale_a, float scale_b) {
  auto buf_a = a.request();
  auto buf_b = b.request();

  if (buf_a.size != buf_b.size) {
    throw std::runtime_error("Arrays must have the same size");
  }

  Int8QuantParams params_a{scale_a, 0};
  Int8QuantParams params_b{scale_b, 0};

  return CosineSimilarityInt8(
      static_cast<int8_t*>(buf_a.ptr),
      static_cast<int8_t*>(buf_b.ptr),
      buf_a.size, params_a, params_b);
}

void RegisterQuantizeBindings(py::module& m) {
  m.def("quantize_f32_to_int8", &PyQuantizeF32ToInt8, py::arg("input"),
        "Quantize float32 array to int8 (symmetric). Returns (data, scale, zero_point)");

  m.def("dequantize_int8_to_f32", &PyDequantizeInt8ToF32,
        py::arg("input"), py::arg("scale"), py::arg("zero_point"),
        "Dequantize int8 array back to float32");

  m.def("quantize_f32_to_f16", &PyQuantizeF32ToF16, py::arg("input"),
        "Convert float32 to float16 (stored as uint16)");

  m.def("dequantize_f16_to_f32", &PyDequantizeF16ToF32, py::arg("input"),
        "Convert float16 (uint16) back to float32");

  m.def("cosine_similarity_int8", &PyCosineSimilarityInt8,
        py::arg("a"), py::arg("b"), py::arg("scale_a"), py::arg("scale_b"),
        "Compute cosine similarity between two int8 quantized vectors");
}

}  // namespace hafs
