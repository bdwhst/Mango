cmake_minimum_required(VERSION 3.24)
project(mango LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(MANGO_WITH_TORCH "Build the LibTorch neural-network evaluator" OFF)
option(MANGO_BUILD_TESTS "Build mango_tests" ON)

# ---------------------------------------------------------------------------
# Compiler flags
add_library(mango_flags INTERFACE)
if(MSVC)
  target_compile_options(mango_flags INTERFACE /W4 /permissive- /utf-8 /EHsc /Zc:__cplusplus /bigobj)
  target_compile_definitions(mango_flags INTERFACE _CRT_SECURE_NO_WARNINGS NOMINMAX)
else()
  target_compile_options(mango_flags INTERFACE -Wall -Wextra -Wpedantic)
endif()

# ---------------------------------------------------------------------------
# LibTorch: located through the venv's torch package unless CMAKE_PREFIX_PATH is given.
if(MANGO_WITH_TORCH)
  if(NOT CMAKE_PREFIX_PATH)
    if(WIN32)
      set(_venv_py "${CMAKE_SOURCE_DIR}/.venv/Scripts/python.exe")
    else()
      set(_venv_py "${CMAKE_SOURCE_DIR}/.venv/bin/python")
    endif()
    if(EXISTS "${_venv_py}")
      execute_process(
        COMMAND "${_venv_py}" -c "import torch, sys; sys.stdout.write(torch.utils.cmake_prefix_path)"
        OUTPUT_VARIABLE _torch_prefix
        RESULT_VARIABLE _torch_rc)
      if(_torch_rc EQUAL 0)
        list(APPEND CMAKE_PREFIX_PATH "${_torch_prefix}")
        message(STATUS "LibTorch from venv: ${_torch_prefix}")
      endif()
    endif()
  endif()
  find_package(Torch REQUIRED)
endif()

# Copy LibTorch DLLs next to executables on Windows.
function(mango_copy_torch_dlls target)
  if(MANGO_WITH_TORCH AND MSVC)
    file(GLOB _torch_dlls "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET ${target} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${_torch_dlls} $<TARGET_FILE_DIR:${target}>)
  endif()
endfunction()

# ---------------------------------------------------------------------------
# Core library: board, rules, features, symmetry, sgf, config. No LibTorch.
add_library(mango_core STATIC
  cpp/core/board.cpp
  cpp/core/features.cpp
  cpp/core/symmetry.cpp
  cpp/core/sgf.cpp
  cpp/core/config.cpp)
target_include_directories(mango_core PUBLIC cpp cpp/third_party)
target_link_libraries(mango_core PUBLIC mango_flags)

# GTP front-end library + executable.
add_library(mango_gtp_lib STATIC cpp/gtp/gtp.cpp)
target_link_libraries(mango_gtp_lib PUBLIC mango_core)

add_executable(mango_gtp cpp/gtp/gtp_main.cpp)
target_link_libraries(mango_gtp PRIVATE mango_gtp_lib)

# ---------------------------------------------------------------------------
# Tests
if(MANGO_BUILD_TESTS)
  enable_testing()
  add_executable(mango_tests
    cpp/tests/test_main.cpp
    cpp/tests/ref_board.cpp
    cpp/tests/test_board.cpp
    cpp/tests/test_history.cpp
    cpp/tests/test_symmetry.cpp
    cpp/tests/test_sgf_config.cpp
    cpp/tests/test_gtp.cpp)
  target_link_libraries(mango_tests PRIVATE mango_core mango_gtp_lib)
  add_test(NAME mango_tests COMMAND mango_tests)
endif()
