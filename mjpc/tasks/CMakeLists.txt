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
  copy_model_resources ALL
  ## dm_control models
  # acrobot
  COMMAND ${CMAKE_COMMAND} -E copy
          ${dm_control_SOURCE_DIR}/dm_control/suite/acrobot.xml
          ${CMAKE_CURRENT_BINARY_DIR}/acrobot/acrobot.xml
  COMMAND patch -o ${CMAKE_CURRENT_BINARY_DIR}/acrobot/acrobot_modified.xml
          ${CMAKE_CURRENT_BINARY_DIR}/acrobot/acrobot.xml
          <${CMAKE_CURRENT_SOURCE_DIR}/acrobot/acrobot.xml.patch
  # cartpole
  COMMAND ${CMAKE_COMMAND} -E copy
          ${dm_control_SOURCE_DIR}/dm_control/suite/cartpole.xml
          ${CMAKE_CURRENT_BINARY_DIR}/cartpole/cartpole.xml
  COMMAND patch -o ${CMAKE_CURRENT_BINARY_DIR}/cartpole/cartpole_modified.xml
          ${CMAKE_CURRENT_BINARY_DIR}/cartpole/cartpole.xml
          <${CMAKE_CURRENT_SOURCE_DIR}/cartpole/cartpole.xml.patch
  # humanoid
  COMMAND ${CMAKE_COMMAND} -E copy
          ${dm_control_SOURCE_DIR}/dm_control/suite/humanoid.xml
          ${CMAKE_CURRENT_BINARY_DIR}/humanoid/humanoid.xml
  COMMAND patch -o ${CMAKE_CURRENT_BINARY_DIR}/humanoid/humanoid_modified.xml
          ${CMAKE_CURRENT_BINARY_DIR}/humanoid/humanoid.xml
          <${CMAKE_CURRENT_SOURCE_DIR}/humanoid/humanoid.xml.patch
  # particle
  COMMAND ${CMAKE_COMMAND} -E copy
          ${dm_control_SOURCE_DIR}/dm_control/suite/point_mass.xml
          ${CMAKE_CURRENT_BINARY_DIR}/particle/particle.xml
  COMMAND patch -o ${CMAKE_CURRENT_BINARY_DIR}/particle/particle_modified.xml
          ${CMAKE_CURRENT_BINARY_DIR}/particle/particle.xml
          <${CMAKE_CURRENT_SOURCE_DIR}/particle/particle.xml.patch
  # swimmer
  COMMAND ${CMAKE_COMMAND} -E copy
          ${dm_control_SOURCE_DIR}/dm_control/suite/swimmer.xml
          ${CMAKE_CURRENT_BINARY_DIR}/swimmer/swimmer.xml
  COMMAND patch -o ${CMAKE_CURRENT_BINARY_DIR}/swimmer/swimmer_modified.xml
          ${CMAKE_CURRENT_BINARY_DIR}/swimmer/swimmer.xml
          <${CMAKE_CURRENT_SOURCE_DIR}/swimmer/swimmer.xml.patch
  # walker
  COMMAND ${CMAKE_COMMAND} -E copy
          ${dm_control_SOURCE_DIR}/dm_control/suite/walker.xml
          ${CMAKE_CURRENT_BINARY_DIR}/walker/walker.xml
  COMMAND patch -o ${CMAKE_CURRENT_BINARY_DIR}/walker/walker_modified.xml
          ${CMAKE_CURRENT_BINARY_DIR}/walker/walker.xml
          <${CMAKE_CURRENT_SOURCE_DIR}/walker/walker.xml.patch

  ## Menagerie models
  COMMAND ${CMAKE_COMMAND} -E copy
          ${menagerie_SOURCE_DIR}/shadow_hand/right_hand.xml
          ${CMAKE_CURRENT_BINARY_DIR}/shadow_reorient/right_hand.xml
  COMMAND ${CMAKE_COMMAND} -E copy
          ${menagerie_SOURCE_DIR}/wonik_allegro/right_hand.xml
          ${CMAKE_CURRENT_BINARY_DIR}/allegro/right_hand.xml
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          ${menagerie_SOURCE_DIR}/shadow_hand/assets
          ${CMAKE_CURRENT_BINARY_DIR}/shadow_reorient/assets

  COMMAND ${CMAKE_COMMAND} -E copy_directory
          ${menagerie_SOURCE_DIR}/wonik_allegro/assets
          ${CMAKE_CURRENT_BINARY_DIR}/allegro/assets
  COMMAND patch -o ${CMAKE_CURRENT_BINARY_DIR}/allegro/right_hand_modified.xml
          ${CMAKE_CURRENT_BINARY_DIR}/allegro/right_hand.xml
          <${CMAKE_CURRENT_SOURCE_DIR}/allegro/right_hand.xml.patch

  COMMAND ${CMAKE_COMMAND} -E copy
          ${menagerie_SOURCE_DIR}/franka_emika_panda/panda.xml
          ${CMAKE_CURRENT_BINARY_DIR}/panda/panda.xml
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          ${menagerie_SOURCE_DIR}/franka_emika_panda/assets
          ${CMAKE_CURRENT_BINARY_DIR}/panda/assets
  COMMAND patch -o ${CMAKE_CURRENT_BINARY_DIR}/panda/panda_modified.xml
          ${CMAKE_CURRENT_BINARY_DIR}/panda/panda.xml
          <${CMAKE_CURRENT_SOURCE_DIR}/panda/panda.xml.patch

  COMMAND ${CMAKE_COMMAND} -E copy
          ${menagerie_SOURCE_DIR}/unitree_a1/a1.xml
          ${CMAKE_CURRENT_BINARY_DIR}/quadruped/a1.xml
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          ${menagerie_SOURCE_DIR}/unitree_a1/assets
          ${CMAKE_CURRENT_BINARY_DIR}/quadruped/assets
  COMMAND patch -o ${CMAKE_CURRENT_BINARY_DIR}/quadruped/a1_modified.xml
          ${CMAKE_CURRENT_BINARY_DIR}/quadruped/a1.xml
          <${CMAKE_CURRENT_SOURCE_DIR}/quadruped/a1.xml.patch

  COMMAND ${CMAKE_COMMAND} -E copy_directory
          ${menagerie_SOURCE_DIR}/franka_emika_panda
          ${CMAKE_CURRENT_BINARY_DIR}/manipulation
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          ${menagerie_SOURCE_DIR}/robotiq_2f85
          ${CMAKE_CURRENT_BINARY_DIR}/manipulation

  COMMAND ${CMAKE_COMMAND} -E copy
          ${menagerie_SOURCE_DIR}/skydio_x2/x2.xml
          ${CMAKE_CURRENT_BINARY_DIR}/quadrotor/quadrotor.xml
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          ${menagerie_SOURCE_DIR}/skydio_x2/assets
          ${CMAKE_CURRENT_BINARY_DIR}/quadrotor/assets
  COMMAND patch -o ${CMAKE_CURRENT_BINARY_DIR}/quadrotor/quadrotor_modified.xml
          ${CMAKE_CURRENT_BINARY_DIR}/quadrotor/quadrotor.xml
          <${CMAKE_CURRENT_SOURCE_DIR}/quadrotor/quadrotor.xml.patch

  ## Cube reorientation
  # patch cube from common assets
  COMMAND patch -o ${CMAKE_CURRENT_BINARY_DIR}/common_assets/cube_modified_shadow_reorient.xml
          ${CMAKE_CURRENT_BINARY_DIR}/common_assets/reorientation_cube.xml
          <${CMAKE_CURRENT_SOURCE_DIR}/common_assets/cube.xml.patch

  ## Cube solve task
  # copy cube model from MuJoCo
  COMMAND ${CMAKE_COMMAND} -E copy
          ${mujoco_SOURCE_DIR}/model/cube/cube_3x3x3.xml
          ${CMAKE_CURRENT_BINARY_DIR}/rubik/cube_3x3x3.xml
  # copy cube assets from MuJoCo
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          ${mujoco_SOURCE_DIR}/model/cube/assets
          ${CMAKE_CURRENT_BINARY_DIR}/rubik/assets
  # modified cube model for task
  COMMAND patch -o ${CMAKE_CURRENT_BINARY_DIR}/rubik/cube_3x3x3_modified.xml
          ${CMAKE_CURRENT_BINARY_DIR}/rubik/cube_3x3x3.xml
          <${CMAKE_CURRENT_SOURCE_DIR}/rubik/cube_3x3x3.xml.patch
  # modified cube model to transition model for scramble mode
  COMMAND patch -o ${CMAKE_CURRENT_BINARY_DIR}/rubik/transition_model.xml
          ${CMAKE_CURRENT_BINARY_DIR}/rubik/cube_3x3x3.xml
          <${CMAKE_CURRENT_SOURCE_DIR}/rubik/transition_model.xml.patch
  # copy hand model from Menagerie
  COMMAND ${CMAKE_COMMAND} -E copy
          ${menagerie_SOURCE_DIR}/shadow_hand/right_hand.xml
          ${CMAKE_CURRENT_BINARY_DIR}/rubik/right_hand.xml
  # copy hand assets from Menagerie
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          ${menagerie_SOURCE_DIR}/shadow_hand/assets
          ${CMAKE_CURRENT_BINARY_DIR}/rubik/assets

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

  # MJX ALOHA
  COMMAND ${CMAKE_COMMAND} -E copy
          ${menagerie_SOURCE_DIR}/aloha/mjx_aloha.patch
          ${CMAKE_CURRENT_BINARY_DIR}/bimanual/mjx_aloha.patch
  COMMAND patch -o ${CMAKE_CURRENT_BINARY_DIR}/bimanual/mjx_aloha.xml
          ${CMAKE_CURRENT_BINARY_DIR}/bimanual/aloha.xml
          <${CMAKE_CURRENT_BINARY_DIR}/bimanual/mjx_aloha.patch
  COMMAND ${CMAKE_COMMAND} -E copy
          ${menagerie_SOURCE_DIR}/aloha/scene.xml
          ${CMAKE_CURRENT_BINARY_DIR}/bimanual/scene.xml
  COMMAND ${CMAKE_COMMAND} -E copy
          ${menagerie_SOURCE_DIR}/aloha/mjx_scene.patch
          ${CMAKE_CURRENT_BINARY_DIR}/bimanual/mjx_scene.patch
  COMMAND patch -o ${CMAKE_CURRENT_BINARY_DIR}/bimanual/mjx_scene.xml
          ${CMAKE_CURRENT_BINARY_DIR}/bimanual/scene.xml
          <${CMAKE_CURRENT_BINARY_DIR}/bimanual/mjx_scene.patch

  COMMAND ${CMAKE_COMMAND} -E copy
          ${menagerie_SOURCE_DIR}/aloha/mjx_integrated_cartesian_actuators.patch
          ${CMAKE_CURRENT_BINARY_DIR}/bimanual/mjx_integrated_cartesian_actuators.patch
  COMMAND patch -o ${CMAKE_CURRENT_BINARY_DIR}/bimanual/mjx_integrated_cartesian_actuators.xml
          ${CMAKE_CURRENT_BINARY_DIR}/bimanual/integrated_cartesian_actuators.xml
          <${CMAKE_CURRENT_BINARY_DIR}/bimanual/mjx_integrated_cartesian_actuators.patch

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
  COMMENT "Copying Menagerie and dm_control assets into binary directory")

add_custom_target(copy_resources ALL
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Copying tasks into binary directory")

add_dependencies(copy_model_resources copy_resources)

add_dependencies(libmjpc copy_model_resources)
