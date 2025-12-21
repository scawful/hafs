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

# Option to build only the viz app (no Python bindings)
option(HAFS_BUILD_VIZ_ONLY "Build only the visualization app (skip Python bindings)" OFF)
option(HAFS_BUILD_VIZ "Build ImGui visualization application" OFF)
option(HAFS_FETCH_PYBIND11 "Fetch pybind11 automatically if missing" ON)
option(HAFS_FETCH_GLFW "Fetch GLFW automatically if missing" ON)

# Prefer system pybind11, but fetch if missing for local CMake builds.
if(NOT HAFS_BUILD_VIZ_ONLY)
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
endif()

# Auto-enable VIZ_ONLY if VIZ is requested but we don't have pybind11
if(HAFS_BUILD_VIZ AND NOT HAFS_BUILD_VIZ_ONLY AND NOT pybind11_FOUND)
  message(STATUS "pybind11 not found - enabling VIZ_ONLY mode")
  set(HAFS_BUILD_VIZ_ONLY ON)
endif()

# Only require pybind11 for Python bindings (not viz-only builds)
if(NOT HAFS_BUILD_VIZ_ONLY)
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
endif()

# simdjson for fast JSON loading (optional for both)
find_package(simdjson QUIET)
if(simdjson_FOUND)
  add_compile_definitions(HAFS_HAS_SIMDJSON=1)
  message(STATUS "simdjson found - fast JSON loading enabled")
else()
  message(STATUS "simdjson not found - using built-in JSON parser")
endif()

# =============================================================================
# Build Unified Python Extension Module (skip for viz-only builds)
# =============================================================================

if(NOT HAFS_BUILD_VIZ_ONLY)
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

  # Install to hafs.core package
  install(TARGETS _native
    LIBRARY DESTINATION hafs/core
    RUNTIME DESTINATION hafs/core
  )
endif()

# =============================================================================
# Optional: ImGui Visualization Application
# =============================================================================

if(HAFS_BUILD_VIZ)
  message(STATUS "Building visualization application with ImGui/ImPlot")

  # Fetch Dear ImGui
  FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG docking
  )
  FetchContent_MakeAvailable(imgui)

  # Fetch ImPlot
  FetchContent_Declare(
    implot
    GIT_REPOSITORY https://github.com/epezent/implot.git
    GIT_TAG master
  )
  FetchContent_MakeAvailable(implot)

  # Find dependencies
  find_package(glfw3 3.3 QUIET)
  if(NOT glfw3_FOUND AND HAFS_FETCH_GLFW)
    message(STATUS "glfw3 not found - fetching with FetchContent")
    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
      glfw
      GIT_REPOSITORY https://github.com/glfw/glfw.git
      GIT_TAG 3.4
    )
    FetchContent_MakeAvailable(glfw)
    set(glfw3_FOUND TRUE)
  endif()
  if(NOT glfw3_FOUND)
    message(FATAL_ERROR "glfw3 not found. Install it or set HAFS_FETCH_GLFW=ON.")
  endif()
  find_package(OpenGL REQUIRED)

  # Collect ImGui sources
  set(IMGUI_SOURCES
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/imgui_demo.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
  )

  # Collect ImPlot sources
  set(IMPLOT_SOURCES
    ${implot_SOURCE_DIR}/implot.cpp
    ${implot_SOURCE_DIR}/implot_items.cpp
    ${implot_SOURCE_DIR}/implot_demo.cpp
  )

  # Visualization application
  add_executable(hafs_viz
    viz/main.cc
    viz/app.cc
    viz/data_loader.cc
    viz/widgets/text_editor.cc
    ${IMGUI_SOURCES}
    ${IMPLOT_SOURCES}
  )

  target_include_directories(hafs_viz PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/viz
    ${imgui_SOURCE_DIR}
    ${imgui_SOURCE_DIR}/backends
    ${implot_SOURCE_DIR}
  )

  target_compile_definitions(hafs_viz PRIVATE
    IMGUI_IMPL_OPENGL_LOADER_GLAD=0
  )

  if(TARGET OpenGL::OpenGL)
    set(HAFS_OPENGL_TARGET OpenGL::OpenGL)
  else()
    set(HAFS_OPENGL_TARGET OpenGL::GL)
  endif()

  target_link_libraries(hafs_viz PRIVATE
    glfw
    ${HAFS_OPENGL_TARGET}
  )

  # macOS specific
  if(APPLE)
    target_link_libraries(hafs_viz PRIVATE
      "-framework Cocoa"
      "-framework IOKit"
      "-framework CoreVideo"
    )
  endif()

  # Install
  install(TARGETS hafs_viz RUNTIME DESTINATION bin)

  message(STATUS "Visualization app will be built as 'hafs_viz'")
endif()
