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

from typing import Any, Callable, List, Tuple

from etils import epath
# internal import
from flax import struct
import jax
from jax import numpy as jnp
import mujoco
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


def make_instruction(m: mjx.Model, d: mjx.Data) -> Instruction:
  box_instruction = ObjectInstruction(
      body_index=m.nbody - 1,
      reference_index=0,
      dof_index=m.nv - 6,
      position=jnp.array([-0.3, -0.2, 0.3]),
      orientation=jnp.array([0.5, 0.5, 0.5, 0.5]),
      speed=0.3,
      linear_weights=jnp.array([1, 1, 1]),
      radii=jnp.array([0.05, 0.05, 0.05]),
  )
  return Instruction(
      left_target=jnp.where(d.time > 3, m.nbody - 1, 0),
      right_target=jnp.where(d.time < 6, m.nbody - 1, 0),
      object_instructions=[box_instruction],
  )


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
    dof_index = obj_instruction.dof_index
    object_pos = d.xpos[..., object_body_index, :]
    object_rot = d.xquat[..., object_body_index, :]
    object_lin_vel = d.qvel[..., dof_index:dof_index+3]
    object_ang_vel = d.qvel[..., dof_index+3:dof_index+6]
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


def get_models_and_cost_fn() -> tuple[
    mujoco.MjModel,
    mujoco.MjModel,
    Callable[[mjx.Model, mjx.Data, Instruction], Tuple[jax.Array, Any]],
    Callable[[mjx.Model, mjx.Data], Instruction],
]:
  """Returns a planning model, a sim model, and a cost function."""
  path = epath.Path(
      'build/mjpc/tasks/bimanual/'
  )
  model_file_name = 'mjx_scene.xml'
  xml = (path / model_file_name).read_text()
  assets = {}
  for f in path.glob('*.xml'):
    if f.name == model_file_name:
      continue
    assets[f.name] = f.read_bytes()
  for f in (path / 'assets').glob('*'):
    assets[f.name] = f.read_bytes()
  sim_model = mujoco.MjModel.from_xml_string(xml, assets)
  plan_model = mujoco.MjModel.from_xml_string(xml, assets)
  plan_model.opt.timestep = 0.01
  return sim_model, plan_model, instruction_cost, make_instruction
