#ifndef HAFS_QUANTIZE_H_
#define HAFS_QUANTIZE_H_

#include <cstddef>
#include <cstdint>
#include <tuple>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace hafs {

// =============================================================================
// Quantization Types
// =============================================================================

// Parameters for symmetric int8 quantization
struct Int8QuantParams {
  float scale;      // scale factor: float_val = int8_val * scale
  int32_t zero_point;  // always 0 for symmetric
};

// =============================================================================
// Float32 <-> Int8 Quantization (Symmetric)
// =============================================================================

// Quantize float32 array to int8 (symmetric: centered at 0)
// Returns: quantization parameters (scale, zero_point=0)
Int8QuantParams QuantizeF32ToInt8(const float* input, int8_t* output, size_t n);

// Dequantize int8 array back to float32
void DequantizeInt8ToF32(const int8_t* input, float* output, size_t n,
                          const Int8QuantParams& params);

// =============================================================================
// Float32 <-> Float16 Conversion
// =============================================================================

// Convert float32 to float16 (stored as uint16)
void QuantizeF32ToF16(const float* input, uint16_t* output, size_t n);

// Convert float16 back to float32
void DequantizeF16ToF32(const uint16_t* input, float* output, size_t n);

// =============================================================================
// Quantized Dot Products
// =============================================================================

// Int8 dot product with ARM NEON sdot (or scalar fallback)
// Returns: sum of (a[i] * b[i]) as int32
int32_t DotProductInt8(const int8_t* a, const int8_t* b, size_t n);

// Quantized cosine similarity using int8 representations
// Both vectors should be quantized with same scale for accurate results
float CosineSimilarityInt8(const int8_t* a, const int8_t* b, size_t n,
                           const Int8QuantParams& params_a,
                           const Int8QuantParams& params_b);

// =============================================================================
// Python Bindings
// =============================================================================

// Quantize numpy array to int8, returns (quantized_data, scale, zero_point)
std::tuple<py::array_t<int8_t>, float, int32_t> PyQuantizeF32ToInt8(
    py::array_t<float> input);

// Dequantize int8 array to float32
py::array_t<float> PyDequantizeInt8ToF32(py::array_t<int8_t> input,
                                          float scale, int32_t zero_point);

// Quantize to float16
py::array_t<uint16_t> PyQuantizeF32ToF16(py::array_t<float> input);

// Dequantize from float16
py::array_t<float> PyDequantizeF16ToF32(py::array_t<uint16_t> input);

// Int8 cosine similarity
float PyCosineSimilarityInt8(py::array_t<int8_t> a, py::array_t<int8_t> b,
                              float scale_a, float scale_b);

// Register quantization bindings
void RegisterQuantizeBindings(py::module& m);

}  // namespace hafs

#endif  // HAFS_QUANTIZE_H_
