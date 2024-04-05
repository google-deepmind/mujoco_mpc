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

from typing import Callable, List

from etils import epath
# internal import
from flax import struct
import jax
from jax import numpy as jnp
import mujoco
from mujoco import mjx


@struct.dataclass
class ObjectInstruction:
  position: jax.Array     # 3D desired position of the object
  orientation: jax.Array  # 4D desired quaternion of the object
  linear_speed: float
  angular_speed: float
  linear_weights: jax.Array
  angular_weights: jax.Array
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
      position=jnp.array([-0.4, -0.2, 0.3]),
      orientation=jnp.array([1, 0, 0, 0]),
      linear_speed=0.3,
      angular_speed=0,
      linear_weights=jnp.array([1, 1, 1]),
      angular_weights=jnp.array([0, 0, 0]),
  )
  return Instruction(
      left_target=jnp.where(d.time > 3, m.nbody - 1, 0),
      right_target=jnp.where(d.time < 5, m.nbody - 1, 0),
      object_instructions=[box_instruction],
  )


def instruction_cost(
    m: mjx.Model, d: mjx.Data, instruction: Instruction
) -> jax.Array:
  def linear_errors(des: ObjectInstruction):
    offset = des.position - d.xpos[..., des.body_index, :]
    dist = jnp.linalg.norm(offset)
    direction = offset / dist
    scaling = jnp.tanh(dist*10)  # at a distance of 5cm, stop moving
    des_velocity = direction * des.linear_speed * scaling
    obj_velocity = d.qvel[..., des.dof_index:des.dof_index+3]
    return offset, des.linear_weights * (des_velocity - obj_velocity)

  pos_err, vel_err = linear_errors(instruction.object_instructions[0])

  # reach
  left_gripper_site_index = 3
  right_gripper_site_index = 6

  left_gripper_pos = d.site_xpos[..., left_gripper_site_index, :]
  right_gripper_pos = d.site_xpos[..., right_gripper_site_index, :]
  reach_l = left_gripper_pos - d.xpos[..., instruction.left_target, :]
  reach_r = right_gripper_pos - d.xpos[..., instruction.right_target, :]

  residuals = [reach_l, reach_r, pos_err, vel_err]
  weights = [
      jnp.where(instruction.left_target > 0, 1, 0),
      jnp.where(instruction.right_target > 0, 1, 0),
      0.1,
      1,
  ]
  norm_p = [0.005, 0.005, 0.005, 0.1]

  # NormType::kL2: y = sqrt(x*x' + p^2) - p
  terms = []
  for t, w, p in zip(residuals, weights, norm_p):
    terms.append(w * jnp.sqrt(jnp.sum(t**2, axis=-1) + p**2) - p)
  costs = jnp.sum(jnp.array(terms), axis=-1)

  return costs


def get_models_and_cost_fn() -> tuple[
    mujoco.MjModel,
    mujoco.MjModel,
    Callable[[mjx.Model, mjx.Data, Instruction], jax.Array],
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
