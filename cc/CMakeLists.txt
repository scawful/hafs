cmake_minimum_required(VERSION 3.15)
project(hafs_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# M1/ARM64 optimizations
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
  add_compile_options(-march=armv8-a+simd -O3 -ffast-math)
  add_compile_definitions(HAFS_ARM64=1)
endif()

# pybind11
find_package(pybind11 CONFIG REQUIRED)

# Source files
set(SIMILARITY_SOURCES
  src/similarity.cc
  src/batch_ops.cc
  src/bindings.cc
)

# Build the Python extension module
pybind11_add_module(_similarity ${SIMILARITY_SOURCES})

target_include_directories(_similarity PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Install to hafs.core package
install(TARGETS _similarity
  LIBRARY DESTINATION hafs/core
  RUNTIME DESTINATION hafs/core
)
