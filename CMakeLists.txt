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
  VERSION 0.1.0
  DESCRIPTION "MuJoCo predictive control"
  HOMEPAGE_URL "https://github.com/google-deepmind/mujoco_mpc"
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
option(MJPC_GRPC_BUILD_TESTS "Build tests for gRPC" ON)
option(MJPC_BUILD_GRPC_SERVICE "Build MJPC gRPC service." OFF)
option(PYMJPC_BUILD_TESTS "Build tests for Python bindings" ON)
option(ABSL_ENABLE_INSTALL "Enable installing ABSL libraries" ON)

include(FindOrFetch)

find_package(Python)
find_package(Threads REQUIRED)

set(MUJOCO_BUILD_EXAMPLES OFF)
set(MUJOCO_BUILD_SIMULATE ON)
set(MUJOCO_BUILD_TESTS OFF)
set(MUJOCO_TEST_PYTHON_UTIL OFF)

set(MUJOCO_MPC_MUJOCO_GIT_TAG
    088079eff0450e32b98ee743141780ed68307506
    CACHE STRING "Git revision for MuJoCo."
)

set(MUJOCO_MPC_MENAGERIE_GIT_TAG
    d82891f8b6c576f9f6a9df308f3a1155ebedbc40
    CACHE STRING "Git revision for MuJoCo Menagerie."
)

set(MUJOCO_MPC_DM_CONTROL_GIT_TAG
    774f46182140106e22725914aad3c6299ed91edd
    CACHE STRING "Git revision for dm_control."
)

findorfetch(
  USE_SYSTEM_PACKAGE
  OFF
  PACKAGE_NAME
  mujoco
  LIBRARY_NAME
  mujoco
  GIT_REPO
  https://github.com/google-deepmind/mujoco.git
  GIT_TAG
  ${MUJOCO_MPC_MUJOCO_GIT_TAG}
  TARGETS
  mujoco::mujoco
  mujoco::platform_ui_adapter
  EXCLUDE_FROM_ALL
)

# TODO(nimrod): Update to the latest version of abseil, or use the one defined
# by MuJoCo, once grpc fix their build issues.
set(MUJOCO_DEP_VERSION_abseil
    fb3621f4f897824c0dbe0615fa94543df6192f30 # LTS 20230802.1
    CACHE STRING "Version of `abseil` to be fetched."
    FORCE
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
# ABSL_ENABLE_INSTALL is needed for
# https://github.com/protocolbuffers/protobuf/issues/12185#issuecomment-1594685860
set(ABSL_ENABLE_INSTALL ON)
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
  GIT_REPOSITORY https://github.com/google-deepmind/mujoco_menagerie.git
  GIT_TAG ${MUJOCO_MPC_MENAGERIE_GIT_TAG}
)

FetchContent_GetProperties(menagerie)
if(NOT menagerie_POPULATED)
  FetchContent_Populate(menagerie)
endif()

FetchContent_Declare(
  dm_control
  GIT_REPOSITORY https://github.com/google-deepmind/dm_control.git
  GIT_TAG ${MUJOCO_MPC_DM_CONTROL_GIT_TAG}
)

FetchContent_GetProperties(dm_control)
if(NOT dm_control_POPULATED)
  FetchContent_Populate(dm_control)
endif()

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

add_subdirectory(mjpc)


set(CONFIG_PACKAGE_LOCATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
    VERSION 0.1.0
    COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(
    cmake/${PROJECT_NAME}Config.cmake.in "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
    INSTALL_DESTINATION ${CONFIG_PACKAGE_LOCATION}
)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
    DESTINATION ${CONFIG_PACKAGE_LOCATION}
)

file(GLOB_RECURSE HEADER_FILES "${CMAKE_SOURCE_DIR}/mjpc/*.h")
foreach (HEADER_FILE IN ITEMS ${HEADER_FILES})
  get_filename_component(ABSOLUTE_PATH ${HEADER_FILE} DIRECTORY)
  file(RELATIVE_PATH REL_PATH ${CMAKE_SOURCE_DIR} ${ABSOLUTE_PATH})
  install(FILES ${HEADER_FILE} DESTINATION include/${REL_PATH})
endforeach ()

install(TARGETS libmjpc mujoco threadpool glfw lodepng platform_ui_adapter  EXPORT libmjpcTargets)

install(
    EXPORT libmjpcTargets
    DESTINATION ${CONFIG_PACKAGE_LOCATION}
    NAMESPACE mujoco_mpc::
    FILE "${PROJECT_NAME}Targets.cmake"
)


#install(EXPORT abslTargets
#    FILE absl-targets.cmake
#    DESTINATION ${CONFIG_PACKAGE_LOCATION})


install(EXPORT libmjpcTargets
    FILE libmjpc-targets.cmake
    DESTINATION ${CONFIG_PACKAGE_LOCATION})


#platform_ui_adapter base mujoco check any_invocable flat_hash_map log random_random
#base_internal config core_headers dynamic_annotations log_severity spinlock_wait type_traits raw_logging_internal span
#throw_delegate algorithm errno_saver atomic_hook  random_seed_sequences random_internal_randen_engine
#random_internal_pool_urbg random_internal_pcg_engine random_internal_nonsecure_base random_distributions log_internal_log_impl
#memory meta log_internal_strip log_internal_message log_internal_conditions strings random_internal_wide_multiply
#random_internal_uniform_helper random_internal_traits  random_internal_iostream_state_saver random_internal_fastmath
#random_internal_fast_uniform_bits random_internal_distribution_caller random_internal_generate_real random_internal_seed_material
#random_internal_salted_seed_seq inlined_vector int128
#random_seed_gen_exception random_internal_randen endian random_internal_randen_slow random_internal_randen_hwaes
#random_internal_platform inlined_vector_internal optional bits fast_type_id utility bad_optional_access
#compressed_tuple random_internal_randen_hwaes_impl strings_internal string_view log_internal_voidify

#export(EXPORT libmjpcTargets
#    FILE ${CMAKE_CURRENT_BINARY_DIR}/fast_forward_kinematics-targets.cmake)
#install(
#    EXPORT libmjpcTargets
#    FILE fast_forward_kinematics-targets.cmake
#    DESTINATION "${CMAKE_INSTALL_DATADIR}/fast_forward_kinematics/cmake")

