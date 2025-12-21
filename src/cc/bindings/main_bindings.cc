#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../similarity/similarity.h"
#include "../index/hnsw_index.h"
#include "../quantize/quantize.h"
#include "../io/json_loader.h"
#include "../stream/streaming_index.h"

namespace py = pybind11;

PYBIND11_MODULE(_native, m) {
  m.doc() = "HAFS native operations with M1/ARM64 SIMD optimization";

  // Register similarity operations
  hafs::RegisterSimilarityBindings(m);

  // Register HNSW index
  hafs::RegisterHNSWBindings(m);

  // Register quantization operations
  hafs::RegisterQuantizeBindings(m);

  // Register IO operations
  hafs::RegisterIOBindings(m);

  // Register streaming index
  hafs::RegisterStreamingBindings(m);

  // Version info
  m.attr("__version__") = "0.2.0";

#ifdef HAFS_ARM64
  m.attr("__simd__") = "ARM NEON";
#else
  m.attr("__simd__") = "Scalar";
#endif

#ifdef HAFS_ARM64
  m.attr("__blas__") = "NEON SIMD";
#else
  m.attr("__blas__") = "Scalar";
#endif

  // Feature flags
#ifdef HAFS_HAS_HNSW
  m.attr("__has_hnsw__") = true;
#else
  m.attr("__has_hnsw__") = false;
#endif

  m.attr("__has_quantize__") = true;   // Phase 2
  // __has_simdjson__ set by RegisterIOBindings
  // __has_streaming__ set by RegisterStreamingBindings
  m.attr("__has_gemma__") = false;     // Phase 5
}
