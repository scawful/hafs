# Legacy CMakeLists.txt - Delegates to new structure
#
# The C++ codebase has been restructured:
#   - GUI application: apps/studio/
#   - Native modules:  native/
#
# This file is kept for backward compatibility.

cmake_minimum_required(VERSION 3.15)

message(STATUS "src/cc/CMakeLists.txt is deprecated. Use:")
message(STATUS "  -DHAFS_BUILD_STUDIO=ON for GUI")
message(STATUS "  -DHAFS_BUILD_NATIVE=ON for Python bindings")

# Delegate to new locations
if(NOT TARGET _native)
  add_subdirectory(${CMAKE_SOURCE_DIR}/native native_build)
endif()
