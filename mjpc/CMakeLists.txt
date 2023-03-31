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

add_library(threadpool STATIC)
target_sources(
  threadpool
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/threadpool.h
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/threadpool.cc
)
target_link_libraries(
  threadpool
  absl::base
)
target_include_directories(threadpool PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)

add_library(
  libmjpc STATIC
  ${CMAKE_CURRENT_SOURCE_DIR}/states/state.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/states/state.h
  ${CMAKE_CURRENT_SOURCE_DIR}/states/include.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/states/include.h
  ${CMAKE_CURRENT_SOURCE_DIR}/agent.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/agent.h
  ${CMAKE_CURRENT_SOURCE_DIR}/trajectory.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/trajectory.h
  ${CMAKE_CURRENT_SOURCE_DIR}/utilities.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/utilities.h
  ${CMAKE_CURRENT_SOURCE_DIR}/tasks/tasks.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/tasks/tasks.h
  ${CMAKE_CURRENT_SOURCE_DIR}/tasks/acrobot/acrobot.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/tasks/acrobot/acrobot.h
  ${CMAKE_CURRENT_SOURCE_DIR}/tasks/cartpole/cartpole.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/tasks/cartpole/cartpole.h
  ${CMAKE_CURRENT_SOURCE_DIR}/tasks/hand/hand.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/tasks/hand/hand.h
  ${CMAKE_CURRENT_SOURCE_DIR}/tasks/humanoid/stand/stand.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/tasks/humanoid/stand/stand.h
  ${CMAKE_CURRENT_SOURCE_DIR}/tasks/humanoid/tracking/tracking.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/tasks/humanoid/tracking/tracking.h
  ${CMAKE_CURRENT_SOURCE_DIR}/tasks/humanoid/walk/walk.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/tasks/humanoid/walk/walk.h
  ${CMAKE_CURRENT_SOURCE_DIR}/tasks/manipulation/common.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/tasks/manipulation/common.h
  ${CMAKE_CURRENT_SOURCE_DIR}/tasks/manipulation/manipulation.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/tasks/manipulation/manipulation.h
  ${CMAKE_CURRENT_SOURCE_DIR}/tasks/panda/panda.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/tasks/panda/panda.h
  ${CMAKE_CURRENT_SOURCE_DIR}/tasks/particle/particle.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/tasks/particle/particle.h
  ${CMAKE_CURRENT_SOURCE_DIR}/tasks/quadrotor/quadrotor.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/tasks/quadrotor/quadrotor.h
  ${CMAKE_CURRENT_SOURCE_DIR}/tasks/quadruped/quadruped.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/tasks/quadruped/quadruped.h
  ${CMAKE_CURRENT_SOURCE_DIR}/tasks/swimmer/swimmer.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/tasks/swimmer/swimmer.h
  ${CMAKE_CURRENT_SOURCE_DIR}/tasks/walker/walker.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/tasks/walker/walker.h
  ${CMAKE_CURRENT_SOURCE_DIR}/planners/planner.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/planners/planner.h
  ${CMAKE_CURRENT_SOURCE_DIR}/planners/policy.h
  ${CMAKE_CURRENT_SOURCE_DIR}/planners/include.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/planners/include.h
  ${CMAKE_CURRENT_SOURCE_DIR}/planners/cost_derivatives.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/planners/cost_derivatives.h
  ${CMAKE_CURRENT_SOURCE_DIR}/planners/linear_solve.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/planners/linear_solve.h
  ${CMAKE_CURRENT_SOURCE_DIR}/planners/model_derivatives.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/planners/model_derivatives.h
  ${CMAKE_CURRENT_SOURCE_DIR}/planners/sampling/planner.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/planners/sampling/planner.h
  ${CMAKE_CURRENT_SOURCE_DIR}/planners/sampling/policy.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/planners/sampling/policy.h
  ${CMAKE_CURRENT_SOURCE_DIR}/planners/gradient/gradient.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/planners/gradient/gradient.h
  ${CMAKE_CURRENT_SOURCE_DIR}/planners/gradient/planner.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/planners/gradient/planner.h
  ${CMAKE_CURRENT_SOURCE_DIR}/planners/gradient/policy.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/planners/gradient/policy.h
  ${CMAKE_CURRENT_SOURCE_DIR}/planners/gradient/settings.h
  ${CMAKE_CURRENT_SOURCE_DIR}/planners/gradient/spline_mapping.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/planners/gradient/spline_mapping.h
  ${CMAKE_CURRENT_SOURCE_DIR}/planners/ilqg/backward_pass.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/planners/ilqg/backward_pass.h
  ${CMAKE_CURRENT_SOURCE_DIR}/planners/ilqg/boxqp.h
  ${CMAKE_CURRENT_SOURCE_DIR}/planners/ilqg/planner.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/planners/ilqg/planner.h
  ${CMAKE_CURRENT_SOURCE_DIR}/planners/ilqg/policy.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/planners/ilqg/policy.h
  ${CMAKE_CURRENT_SOURCE_DIR}/planners/ilqg/settings.h
  ${CMAKE_CURRENT_SOURCE_DIR}/planners/ilqs/planner.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/planners/ilqs/planner.h
  ${CMAKE_CURRENT_SOURCE_DIR}/norm.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/norm.h
  ${CMAKE_CURRENT_SOURCE_DIR}/task.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/task.h
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

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(FindOrFetch)
FetchContent_MakeAvailable($<TARGET_OBJECTS:mujoco::mujoco> $<TARGET_OBJECTS:mujoco::platform_ui_adapter>)

add_executable(
  mjpc
  ${CMAKE_CURRENT_SOURCE_DIR}/app.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/app.h
  ${CMAKE_CURRENT_SOURCE_DIR}/main.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/simulate.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/simulate.h
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
