cmake_minimum_required(VERSION 3.15)
project(hafs_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(FetchContent)

# M1/ARM64 optimizations with dotprod for int8 operations
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
  add_compile_options(-march=armv8-a+simd+dotprod -O3 -ffast-math)
  add_compile_definitions(HAFS_ARM64=1)
endif()

# =============================================================================
# Dependencies
# =============================================================================

option(HAFS_FETCH_PYBIND11 "Fetch pybind11 automatically if missing" ON)

# Prefer system pybind11, but fetch if missing
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND AND HAFS_FETCH_PYBIND11)
  message(STATUS "pybind11 not found - fetching with FetchContent")
  FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.11.1
  )
  FetchContent_MakeAvailable(pybind11)
  set(pybind11_FOUND TRUE)
endif()

if(NOT pybind11_FOUND)
  message(FATAL_ERROR
    "pybind11 not found. Install it or set HAFS_FETCH_PYBIND11=ON.")
endif()

# Fetch hnswlib (header-only library)
FetchContent_Declare(
  hnswlib
  GIT_REPOSITORY https://github.com/nmslib/hnswlib.git
  GIT_TAG v0.8.0
)
FetchContent_MakeAvailable(hnswlib)
add_compile_definitions(HAFS_HAS_HNSW=1)
message(STATUS "hnswlib fetched - HNSW index enabled")

# simdjson for fast JSON loading (optional)
find_package(simdjson QUIET)
if(simdjson_FOUND)
  add_compile_definitions(HAFS_HAS_SIMDJSON=1)
  message(STATUS "simdjson found - fast JSON loading enabled")
else()
  message(STATUS "simdjson not found - using built-in JSON parser")
endif()

# =============================================================================
# Build Python Extension Module
# =============================================================================

set(SIMILARITY_SOURCES
  similarity/similarity.cc
  similarity/batch_ops.cc
)

set(INDEX_SOURCES
  index/hnsw_index.cc
)

set(QUANTIZE_SOURCES
  quantize/quantize.cc
)

set(IO_SOURCES
  io/json_loader.cc
)

set(STREAM_SOURCES
  stream/streaming_index.cc
)

pybind11_add_module(_native
  ${SIMILARITY_SOURCES}
  ${INDEX_SOURCES}
  ${QUANTIZE_SOURCES}
  ${IO_SOURCES}
  ${STREAM_SOURCES}
  bindings/main_bindings.cc
)

target_include_directories(_native PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/common
  ${hnswlib_SOURCE_DIR}
)

# Link optional dependencies
if(simdjson_FOUND)
  target_link_libraries(_native PRIVATE simdjson::simdjson)
endif()

# Install to core package
install(TARGETS _native
  LIBRARY DESTINATION hafs/core
  RUNTIME DESTINATION hafs/core
)
