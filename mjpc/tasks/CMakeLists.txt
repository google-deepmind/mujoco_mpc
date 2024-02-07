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

# Copy the model files to the binary directory to make them available to the
# built binary.

add_custom_target(
  copy_menagerie_resources ALL
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          ${menagerie_SOURCE_DIR}/shadow_hand/assets
          ${CMAKE_CURRENT_BINARY_DIR}/hand/assets
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          ${menagerie_SOURCE_DIR}/franka_emika_panda/assets
          ${CMAKE_CURRENT_BINARY_DIR}/panda/assets
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          ${menagerie_SOURCE_DIR}/unitree_a1/assets
          ${CMAKE_CURRENT_BINARY_DIR}/quadruped/assets
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          ${menagerie_SOURCE_DIR}/franka_emika_panda
          ${CMAKE_CURRENT_BINARY_DIR}/manipulation
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          ${menagerie_SOURCE_DIR}/robotiq_2f85
          ${CMAKE_CURRENT_BINARY_DIR}/manipulation
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          ${menagerie_SOURCE_DIR}/skydio_x2/assets
          ${CMAKE_CURRENT_BINARY_DIR}/quadrotor/assets
  ## Cube solve task
  # copy cube model from MuJoCo
  COMMAND ${CMAKE_COMMAND} -E copy
          ${mujoco_SOURCE_DIR}/model/cube/cube_3x3x3.xml
          ${CMAKE_CURRENT_BINARY_DIR}/cube/cube_3x3x3.xml
  # copy cube assets from MuJoCo
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          ${mujoco_SOURCE_DIR}/model/cube/assets
          ${CMAKE_CURRENT_BINARY_DIR}/cube/assets
  # modified cube model for task
  COMMAND patch -o ${CMAKE_CURRENT_BINARY_DIR}/cube/cube_3x3x3_modified.xml
          ${CMAKE_CURRENT_BINARY_DIR}/cube/cube_3x3x3.xml
          <${CMAKE_CURRENT_SOURCE_DIR}/cube/cube_3x3x3.xml.patch
  # modified cube model to transition model for scramble mode
  COMMAND patch -o ${CMAKE_CURRENT_BINARY_DIR}/cube/transition_model.xml
          ${CMAKE_CURRENT_BINARY_DIR}/cube/cube_3x3x3.xml
          <${CMAKE_CURRENT_SOURCE_DIR}/cube/transition_model.xml.patch
  # copy hand model from Menagerie
  COMMAND ${CMAKE_COMMAND} -E copy
          ${menagerie_SOURCE_DIR}/shadow_hand/right_hand.xml
          ${CMAKE_CURRENT_BINARY_DIR}/cube/right_hand.xml
  # copy hand assets from Menagerie
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          ${menagerie_SOURCE_DIR}/shadow_hand/assets
          ${CMAKE_CURRENT_BINARY_DIR}/cube/assets

  # ALOHA
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          ${menagerie_SOURCE_DIR}/aloha/assets
          ${CMAKE_CURRENT_BINARY_DIR}/bimanual/assets
  COMMAND ${CMAKE_COMMAND} -E copy
          ${menagerie_SOURCE_DIR}/aloha/aloha.xml
          ${CMAKE_CURRENT_BINARY_DIR}/bimanual/aloha.xml
  COMMAND ${CMAKE_COMMAND} -E copy
          ${menagerie_SOURCE_DIR}/aloha/integrated_cartesian_actuators.xml
          ${CMAKE_CURRENT_BINARY_DIR}/bimanual/integrated_cartesian_actuators.xml
  COMMAND patch -o ${CMAKE_CURRENT_BINARY_DIR}/bimanual/aloha_cartesian.xml
          ${CMAKE_CURRENT_BINARY_DIR}/bimanual/aloha.xml
          <${CMAKE_CURRENT_SOURCE_DIR}/bimanual/aloha.patch

  COMMAND ${CMAKE_COMMAND} -E copy
          ${menagerie_SOURCE_DIR}/robotis_op3/op3.xml
          ${CMAKE_CURRENT_BINARY_DIR}/op3/op3.xml
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          ${menagerie_SOURCE_DIR}/robotis_op3/assets
          ${CMAKE_CURRENT_BINARY_DIR}/op3/assets
  COMMAND patch -o ${CMAKE_CURRENT_BINARY_DIR}/op3/op3_modified.xml
          ${CMAKE_CURRENT_BINARY_DIR}/op3/op3.xml
          <${CMAKE_CURRENT_SOURCE_DIR}/op3/op3.xml.patch

  COMMAND ${Python_EXECUTABLE}
          ${CMAKE_CURRENT_BINARY_DIR}/manipulation/merge_panda_robotiq.py
          ${CMAKE_CURRENT_BINARY_DIR}/manipulation/panda_robotiq.xml
  COMMENT "Copying menagerie assets into binary directory")

add_custom_target(copy_resources ALL
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Copying tasks into binary directory")

add_dependencies(copy_menagerie_resources copy_resources)

add_dependencies(libmjpc copy_menagerie_resources)
