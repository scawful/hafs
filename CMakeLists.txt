cmake_minimum_required(VERSION 3.15)
project(hafs LANGUAGES CXX)

# Options
option(HAFS_BUILD_STUDIO "Build hafs_studio GUI application" OFF)
option(HAFS_BUILD_NATIVE "Build native Python extension modules" ON)
option(HAFS_BUILD_TESTS "Build C++ unit tests" OFF)

# Build native Python bindings
if(HAFS_BUILD_NATIVE)
  add_subdirectory(native)
endif()

# Build hafs_studio GUI application
if(HAFS_BUILD_STUDIO)
  add_subdirectory(apps/studio)
endif()

# Legacy compatibility: delegate to src/cc if it exists and has content
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/cc/CMakeLists.txt")
  # Only include if not already handled by new structure
  if(NOT HAFS_BUILD_NATIVE AND NOT HAFS_BUILD_STUDIO)
    add_subdirectory(src/cc)
  endif()
endif()
