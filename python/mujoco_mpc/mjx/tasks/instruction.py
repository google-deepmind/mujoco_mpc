# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from typing import Any, List, Tuple

from flax import struct
import jax
from jax import numpy as jnp
from mujoco import mjx
from mujoco.mjx._src import math


@struct.dataclass
class ObjectInstruction:
  position: jax.Array     # 3D desired position of the object
  orientation: jax.Array  # 4D desired quaternion of the object
  radii: jax.Array        # used for enforcing orientation penalty
  speed: float
  linear_weights: jax.Array
  body_index: int
  dof_index: int
  reference_index: int   # body index, 0 for ground


@struct.dataclass
class Instruction:
  left_target: jax.Array   # body index of reach target
  right_target: jax.Array  # body index of reach target
  object_instructions: List[ObjectInstruction]


def instruction_cost(
    m: mjx.Model, d: mjx.Data, instruction: Instruction
) -> Tuple[jax.Array, Any]:

  # reach
  left_gripper_site_index = 3
  right_gripper_site_index = 6

  left_gripper_pos = d.site_xpos[..., left_gripper_site_index, :]
  right_gripper_pos = d.site_xpos[..., right_gripper_site_index, :]
  reach_l = left_gripper_pos - d.xpos[..., instruction.left_target, :]
  reach_r = right_gripper_pos - d.xpos[..., instruction.right_target, :]

  residuals = [reach_l, reach_r]
  weights = [
      jnp.where(instruction.left_target > 0, 1, 0),
      jnp.where(instruction.right_target > 0, 1, 0),
  ]
  norm_p = [0.005, 0.005]

  def spur_pos_vel(obj_pos, obj_rot, obj_lin_vel, obj_ang_vel, local_spur):
    spur_delta = math.rotate(local_spur, obj_rot)
    spur_pos = obj_pos + spur_delta
    spur_vel = obj_lin_vel + jnp.cross(spur_delta, obj_ang_vel)
    return spur_pos, spur_vel

  def pos_vel_error(
      desired: ObjectInstruction,
      local_spur: jax.Array,
      obj_spur_pos: jax.Array,
      obj_spur_vel: jax.Array,
  ):
    ref_pos = d.xpos[..., desired.reference_index, :]
    ref_quat = d.xquat[..., desired.reference_index, :]
    center_pos = ref_pos + math.rotate(desired.position, ref_quat)
    desired_spur_pos = center_pos + math.rotate(local_spur, desired.orientation)
    offset = desired_spur_pos - obj_spur_pos
    dist = jnp.linalg.norm(offset)
    direction = offset / dist
    scaling = jnp.tanh(dist*10)  # at a distance of 5cm, stop moving
    desired_vel = direction * desired.speed * scaling
    return offset, desired.linear_weights * (desired_vel - obj_spur_vel)

  for obj_instruction in instruction.object_instructions:
    object_body_index = obj_instruction.body_index
    object_pos = d.xpos[..., object_body_index, :]
    object_rot = d.xquat[..., object_body_index, :]
    dof_index = obj_instruction.dof_index
    object_lin_vel = jax.lax.dynamic_slice_in_dim(d.qvel, dof_index, 3, axis=-1)
    object_ang_vel = jax.lax.dynamic_slice_in_dim(d.qvel, dof_index+3, 3, axis=-1)
    spurs = [
        jnp.array([obj_instruction.radii[0], 0, 0]),
        jnp.array([-obj_instruction.radii[0], 0, 0]),
        jnp.array([0, obj_instruction.radii[1], 0]),
        jnp.array([0, -obj_instruction.radii[1], 0]),
        jnp.array([0, 0, obj_instruction.radii[2]]),
        jnp.array([0, 0, -obj_instruction.radii[2]]),
    ]
    for local_spur in spurs:
      spur_pos, spur_vel = spur_pos_vel(
          object_pos, object_rot, object_lin_vel, object_ang_vel, local_spur
      )
      pos_err, vel_err = pos_vel_error(
          obj_instruction, local_spur, spur_pos, spur_vel
      )
      residuals.append(pos_err)
      weights.append(0.016)
      norm_p.append(0.005)
      residuals.append(vel_err)
      weights.append(0.16)
      norm_p.append(0.1)

  # NormType::kL2: y = sqrt(x*x' + p^2) - p
  terms = []
  for t, w, p in zip(residuals, weights, norm_p):
    terms.append(w * (jnp.sqrt(jnp.sum(t**2, axis=-1) + p**2) - p))
  terms = jnp.array(terms)
  cost = jnp.sum(terms, axis=-1)
  return cost, (terms, residuals)
