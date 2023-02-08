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

set(MUJOCO_TEST_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/../..)
set(MUJOCO_TEST_WORKING_DIR ${CMAKE_CURRENT_SOURCE_DIR})

include(GoogleTest)

macro(test name)
  add_executable(${name} ${name}.cc)
  target_link_libraries(
    ${name}
    gtest_main
    libmjpc
    mujoco
  )
  target_include_directories(${name} PRIVATE ${MUJOCO_TEST_INCLUDE})
  set_target_properties(${name} PROPERTIES BUILD_RPATH ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
  # gtest_discover_tests is recommended over gtest_add_tests, but has some issues in Windows.
  gtest_add_tests(
    TARGET ${name}
    SOURCES ${name}.cc
    WORKING_DIRECTORY ${MUJOCO_TEST_WORKING_DIR}
    TEST_LIST testList
  )
  if(WIN32)
    set_tests_properties(
      ${testList} PROPERTIES ENVIRONMENT "PATH=$<TARGET_FILE_DIR:mujoco>;$ENV{PATH}"
    )
  endif()
endmacro()

# testdata path
add_library(load STATIC load.h load.cc)
target_include_directories(load PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../..)
target_compile_definitions(load PUBLIC MJSTATIC)
target_link_libraries(load PUBLIC mujoco::mujoco absl::core_headers absl::strings)
target_include_directories(load PRIVATE ${mujoco_SOURCE_DIR}/include gmock)

# finite difference
add_library(finite_difference STATIC finite_difference.h finite_difference.cc)
target_include_directories(finite_difference PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../..)
target_compile_definitions(finite_difference PUBLIC MJSTATIC)
target_link_libraries(finite_difference PUBLIC mujoco::mujoco)
target_include_directories(finite_difference PRIVATE ${mujoco_SOURCE_DIR}/include gmock)

# LQR
add_library(lqr STATIC lqr.h lqr.cc)
target_include_directories(lqr PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../..)
target_compile_definitions(lqr PUBLIC MJSTATIC)
target_link_libraries(lqr PUBLIC mujoco::mujoco)
target_include_directories(lqr PRIVATE ${mujoco_SOURCE_DIR}/include gmock)

enable_testing()

add_subdirectory(sampling_planner)
add_subdirectory(gradient_planner)
add_subdirectory(ilqg_planner)
add_subdirectory(tasks)
add_subdirectory(state)
add_subdirectory(agent)
