# Copyright 2022 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.16)

# INTERPROCEDURAL_OPTIMIZATION is enforced when enabled.
set(CMAKE_POLICY_DEFAULT_CMP0069 NEW)
# Default to GLVND if available.
set(CMAKE_POLICY_DEFAULT_CMP0072 NEW)

# This line has to appear before 'PROJECT' in order to be able to disable incremental linking
set(MSVC_INCREMENTAL_DEFAULT ON)

project(
  mujoco_mpc
  VERSION 0.0.1
  DESCRIPTION "MuJoCo predictive control"
  HOMEPAGE_URL "https://github.com/deepmind/mujoco_mpc"
)

enable_language(C)
enable_language(CXX)
if(APPLE)
  enable_language(OBJC)
  enable_language(OBJCXX)
endif()

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
include(MpcOptions)
include(MujocoMacOS)

option(MJPC_BUILD_TESTS "Build tests for MJPC" ON)

include(FindOrFetch)

find_package(Python)
find_package(Threads REQUIRED)

set(MUJOCO_BUILD_EXAMPLES OFF)
set(MUJOCO_BUILD_SIMULATE ON)
set(MUJOCO_BUILD_TESTS OFF)
set(MUJOCO_TEST_PYTHON_UTIL OFF)

findorfetch(
  USE_SYSTEM_PACKAGE
  OFF
  PACKAGE_NAME
  mujoco
  LIBRARY_NAME
  mujoco
  GIT_REPO
  https://github.com/deepmind/mujoco.git
  GIT_TAG
  main
  TARGETS
  mujoco::mujoco
  mujoco::platform_ui_adapter
  EXCLUDE_FROM_ALL
)

set(MUJOCO_DEP_VERSION_abseil
    c8a2f92586fe9b4e1aff049108f5db8064924d8e # LTS 20230125.1
    CACHE STRING "Version of `abseil` to be fetched."
)

set(MUJOCO_DEP_VERSION_glfw3
    7482de6071d21db77a7236155da44c172a7f6c9e # 3.3.8
    CACHE STRING "Version of `glfw` to be fetched."
)

set(MJPC_DEP_VERSION_lodepng
    b4ed2cd7ecf61d29076169b49199371456d4f90b
    CACHE STRING "Version of `lodepng` to be fetched."
)

set(BUILD_SHARED_LIBS_OLD ${BUILD_SHARED_LIBS})
set(BUILD_SHARED_LIBS
    OFF
    CACHE INTERNAL "Build SHARED libraries"
)

findorfetch(
  USE_SYSTEM_PACKAGE
  OFF
  PACKAGE_NAME
  GTest
  LIBRARY_NAME
  googletest
  GIT_REPO
  https://github.com/google/googletest.git
  GIT_TAG
  ${MUJOCO_DEP_VERSION_gtest}
  TARGETS
  gtest
  gmock
  gtest_main
  EXCLUDE_FROM_ALL
)

set(ABSL_PROPAGATE_CXX_STD ON)
set(ABSL_BUILD_TESTING OFF)
findorfetch(
  USE_SYSTEM_PACKAGE
  OFF
  PACKAGE_NAME
  absl
  LIBRARY_NAME
  abseil-cpp
  GIT_REPO
  https://github.com/abseil/abseil-cpp.git
  GIT_TAG
  ${MUJOCO_DEP_VERSION_abseil}
  TARGETS
  absl::core_headers
  EXCLUDE_FROM_ALL
)

set(GLFW_BUILD_EXAMPLES OFF)
set(GLFW_BUILD_TESTS OFF)
set(GLFW_BUILD_DOCS OFF)
set(GLFW_INSTALL OFF)

findorfetch(
  USE_SYSTEM_PACKAGE
  MJPC_USE_SYSTEM_GLFW
  PACKAGE_NAME
  glfw
  LIBRARY_NAME
  glfw
  GIT_REPO
  https://github.com/glfw/glfw.git
  GIT_TAG
  ${MUJOCO_DEP_VERSION_glfw3}
  TARGETS
  glfw
  EXCLUDE_FROM_ALL
)

set(BUILD_SHARED_LIBS
    ${BUILD_SHARED_LIBS_OLD}
    CACHE BOOL "Build SHARED libraries" FORCE
)
unset(BUILD_SHARED_LIBS_OLD)

FetchContent_Declare(
  menagerie
  GIT_REPOSITORY https://github.com/deepmind/mujoco_menagerie.git
  GIT_TAG main
)

FetchContent_GetProperties(menagerie)
if(NOT menagerie_POPULATED)
  FetchContent_Populate(menagerie)
endif()

add_custom_target(
  copy_menagerie_resources ALL
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          ${menagerie_SOURCE_DIR}/shadow_hand/assets
          ${CMAKE_CURRENT_BINARY_DIR}/tasks/hand/assets
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          ${menagerie_SOURCE_DIR}/franka_emika_panda/assets
          ${CMAKE_CURRENT_BINARY_DIR}/tasks/panda/assets
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          ${menagerie_SOURCE_DIR}/unitree_a1/assets
          ${CMAKE_CURRENT_BINARY_DIR}/tasks/quadruped/assets
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          ${menagerie_SOURCE_DIR}/franka_emika_panda
          ${CMAKE_CURRENT_BINARY_DIR}/tasks/manipulation
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          ${menagerie_SOURCE_DIR}/robotiq_2f85
          ${CMAKE_CURRENT_BINARY_DIR}/tasks/manipulation
  COMMAND ${Python_EXECUTABLE}
          ${CMAKE_CURRENT_BINARY_DIR}/tasks/manipulation/merge_panda_robotiq.py
          ${CMAKE_CURRENT_BINARY_DIR}/tasks/manipulation/panda_robotiq.xml
  COMMENT "Copying menagerie assets into binary directory")

if(NOT TARGET lodepng)
  FetchContent_Declare(
    lodepng
    GIT_REPOSITORY https://github.com/lvandeve/lodepng.git
    GIT_TAG ${MJPC_DEP_VERSION_lodepng}
  )

  FetchContent_GetProperties(lodepng)
  if(NOT lodepng_POPULATED)
    FetchContent_Populate(lodepng)
    # This is not a CMake project.
    set(LODEPNG_SRCS ${lodepng_SOURCE_DIR}/lodepng.cpp)
    set(LODEPNG_HEADERS ${lodepng_SOURCE_DIR}/lodepng.h)
    add_library(lodepng STATIC ${LODEPNG_HEADERS} ${LODEPNG_SRCS})
    target_compile_options(lodepng PRIVATE ${MUJOCO_MACOS_COMPILE_OPTIONS})
    target_link_options(lodepng PRIVATE ${MUJOCO_MACOS_LINK_OPTIONS})
    target_include_directories(lodepng PUBLIC ${lodepng_SOURCE_DIR})
  endif()
endif()

set(MJPC_COMPILE_OPTIONS "${AVX_COMPILE_OPTIONS}" "${EXTRA_COMPILE_OPTIONS}")
set(MJPC_LINK_OPTIONS "${EXTRA_LINK_OPTIONS}")

add_library(threadpool STATIC)
target_sources(
  threadpool
  PUBLIC threadpool.h
  PRIVATE threadpool.cc
)
target_link_libraries(
  threadpool
  absl::base
)
target_include_directories(threadpool PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)

add_library(
  libmjpc STATIC
  states/state.cc
  states/state.h
  states/include.cc
  states/include.h
  agent.cc
  agent.h
  trajectory.cc
  trajectory.h
  utilities.cc
  utilities.h
  tasks/tasks.cc
  tasks/tasks.h
  tasks/acrobot/acrobot.cc
  tasks/acrobot/acrobot.h
  tasks/cartpole/cartpole.cc
  tasks/cartpole/cartpole.h
  tasks/hand/hand.cc
  tasks/hand/hand.h
  tasks/humanoid/stand/stand.cc
  tasks/humanoid/stand/stand.h
  tasks/humanoid/tracking/tracking.cc
  tasks/humanoid/tracking/tracking.h
  tasks/humanoid/walk/walk.cc
  tasks/humanoid/walk/walk.h
  tasks/manipulation/common.cc
  tasks/manipulation/common.h
  tasks/manipulation/manipulation.cc
  tasks/manipulation/manipulation.h
  tasks/panda/panda.cc
  tasks/panda/panda.h
  tasks/particle/particle.cc
  tasks/particle/particle.h
  tasks/quadrotor/quadrotor.cc
  tasks/quadrotor/quadrotor.h
  tasks/quadruped/quadruped.cc
  tasks/quadruped/quadruped.h
  tasks/swimmer/swimmer.cc
  tasks/swimmer/swimmer.h
  tasks/walker/walker.cc
  tasks/walker/walker.h
  planners/planner.cc
  planners/planner.h
  planners/policy.h
  planners/include.cc
  planners/include.h
  planners/cost_derivatives.cc
  planners/cost_derivatives.h
  planners/linear_solve.cc
  planners/linear_solve.h
  planners/model_derivatives.cc
  planners/model_derivatives.h
  planners/sampling/planner.cc
  planners/sampling/planner.h
  planners/sampling/policy.cc
  planners/sampling/policy.h
  planners/gradient/gradient.cc
  planners/gradient/gradient.h
  planners/gradient/planner.cc
  planners/gradient/planner.h
  planners/gradient/policy.cc
  planners/gradient/policy.h
  planners/gradient/settings.h
  planners/gradient/spline_mapping.cc
  planners/gradient/spline_mapping.h
  planners/ilqg/backward_pass.cc
  planners/ilqg/backward_pass.h
  planners/ilqg/boxqp.h
  planners/ilqg/planner.cc
  planners/ilqg/planner.h
  planners/ilqg/policy.cc
  planners/ilqg/policy.h
  planners/ilqg/settings.h
  planners/ilqs/planner.cc
  planners/ilqs/planner.h
  norm.cc
  norm.h
  task.cc
  task.h
)
set_target_properties(libmjpc PROPERTIES OUTPUT_NAME mjpc)
target_compile_options(libmjpc PUBLIC ${MJPC_COMPILE_OPTIONS})
target_link_libraries(
  libmjpc
  absl::flat_hash_map
  absl::random_random
  mujoco::mujoco
  threadpool
  Threads::Threads
)
target_include_directories(libmjpc PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)

add_executable(
  mjpc
  app.cc
  app.h
  main.cc
  simulate.cc
  simulate.h
  $<TARGET_OBJECTS:mujoco::platform_ui_adapter>
)
target_link_libraries(
  mjpc
  absl::flags
  absl::flags_parse
  absl::random_random
  absl::strings
  glfw
  libmjpc
  lodepng
  mujoco::mujoco
  mujoco::platform_ui_adapter
  threadpool
  Threads::Threads
)
target_include_directories(mjpc PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_compile_options(mjpc PUBLIC ${MJPC_COMPILE_OPTIONS})
target_link_options(mjpc PRIVATE ${MJPC_LINK_OPTIONS})
target_compile_definitions(mjpc PRIVATE MJSIMULATE_STATIC)
if(APPLE)
  target_sources(mjpc PRIVATE macos_gui.mm)
  target_link_libraries(mjpc "-framework Cocoa")
endif()

add_subdirectory(tasks)

if(BUILD_TESTING AND MJPC_BUILD_TESTS)
  enable_testing()
  add_subdirectory(test)
endif()
