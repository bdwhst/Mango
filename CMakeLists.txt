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

# Neural-network layer: metadata always; the LibTorch evaluator only with MANGO_WITH_TORCH.
if(MANGO_WITH_TORCH)
  add_library(mango_nn STATIC cpp/nn/model_meta.cpp cpp/nn/torch_evaluator.cpp)
  target_link_libraries(mango_nn PUBLIC mango_core "${TORCH_LIBRARIES}")
  target_compile_definitions(mango_nn PUBLIC MANGO_WITH_TORCH=1)
  if(MSVC)
    # LibTorch headers instantiate STL templates that trip C4267/C4244 under /W4.
    target_compile_options(mango_nn PRIVATE /wd4267 /wd4244 /wd4251 /wd4275 /wd4702)
  endif()
else()
  add_library(mango_nn STATIC cpp/nn/model_meta.cpp)
  target_link_libraries(mango_nn PUBLIC mango_core)
endif()

# Search (PUCT). Depends only on the evaluator interface, not on LibTorch.
add_library(mango_search STATIC cpp/search/mcts.cpp)
target_link_libraries(mango_search PUBLIC mango_core mango_nn)

# GTP front-end library + executable.
add_library(mango_gtp_lib STATIC cpp/gtp/gtp.cpp cpp/gtp/gtp_options.cpp cpp/gtp/mcts_player.cpp)
target_link_libraries(mango_gtp_lib PUBLIC mango_core mango_search)

add_executable(mango_gtp cpp/gtp/gtp_main.cpp)
target_link_libraries(mango_gtp PRIVATE mango_gtp_lib mango_nn)
mango_copy_torch_dlls(mango_gtp)

# Self-play: chunk format + game runner (evaluator-agnostic) + batched / multi-threaded
# drivers (eval_queue: the request queue between search threads and the evaluation thread).
find_package(Threads REQUIRED)
add_library(mango_selfplay_lib STATIC cpp/selfplay/chunk.cpp cpp/selfplay/game_runner.cpp cpp/selfplay/batch_runner.cpp
  cpp/selfplay/eval_queue.cpp)
target_link_libraries(mango_selfplay_lib PUBLIC mango_core mango_search Threads::Threads)

add_executable(mango_selfplay cpp/selfplay/selfplay_main.cpp)
target_link_libraries(mango_selfplay PRIVATE mango_selfplay_lib mango_nn)
mango_copy_torch_dlls(mango_selfplay)

# Match driver: colour-swapped pairs, bootstrap interval + executable.
add_library(mango_match_lib STATIC cpp/match/match.cpp)
target_link_libraries(mango_match_lib PUBLIC mango_core mango_search mango_selfplay_lib)

add_executable(mango_match cpp/match/match_main.cpp)
target_link_libraries(mango_match PRIVATE mango_match_lib mango_nn)
mango_copy_torch_dlls(mango_match)

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
    cpp/tests/test_gtp.cpp
    cpp/tests/test_mcts.cpp
    cpp/tests/test_mcts_player.cpp
    cpp/tests/test_chunk.cpp
    cpp/tests/test_selfplay.cpp
    cpp/tests/test_match.cpp)
  if(MANGO_WITH_TORCH)
    target_sources(mango_tests PRIVATE cpp/tests/test_torch_eval.cpp)
  endif()
  target_compile_definitions(mango_tests PRIVATE MANGO_FIXTURE_DIR="${CMAKE_SOURCE_DIR}/cpp/tests/fixtures")
  target_link_libraries(mango_tests PRIVATE mango_core mango_nn mango_search mango_gtp_lib mango_selfplay_lib mango_match_lib)
  add_test(NAME mango_tests COMMAND mango_tests)
  mango_copy_torch_dlls(mango_tests)
endif()
