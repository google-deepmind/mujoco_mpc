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

from typing import Any, Callable, Tuple

from etils import epath
# internal import
import jax
from jax import numpy as jnp
import mujoco
from mujoco import mjx
from mujoco_mpc.mjx.tasks import instruction


def make_instruction(
    m: mjx.Model, d: mjx.Data
) -> Tuple[instruction.Instruction, jnp.ndarray]:
  align_instruction = instruction.ObjectInstruction(
      body_index=m.nbody - 1,
      reference_index=0,
      dof_index=m.nv - 6,
      position=jnp.array([0, 0, 0.15]),
      orientation=jnp.array([1, 0, 0, 0]),
      speed=0.3,
      linear_weights=jnp.array([1, 1, 1]),
      radii=jnp.array([0.15, 0.15, 0.15]),
  )
  align = instruction.Instruction(
      left_target=jnp.where(d.time > 0, 0, 0),
      right_target=jnp.where(d.time < 6, m.nbody - 1, 0),
      object_instructions=[align_instruction],
  )
  insert_instruction = instruction.ObjectInstruction(
      body_index=m.nbody - 1,
      reference_index=0,
      dof_index=m.nv - 6,
      position=jnp.array([0.0, 0.0, 0.03]),
      orientation=jnp.array([1, 0, 0, 0]),
      speed=0.3,
      linear_weights=jnp.array([1, 1, 1]),
      radii=jnp.array([0.07, 0.07, 0.07]),
  )
  insert = instruction.Instruction(
      left_target=jnp.where(d.time > 0, 0, 0),
      right_target=jnp.where(d.time < 6, m.nbody - 1, 0),
      object_instructions=[insert_instruction],
  )
  _, (terms, _) = instruction.instruction_cost(m, d, align)
  align_pos_err = jnp.sum(terms[..., 2:13:2], axis=-1)
  userdata = jnp.array(
      [jnp.where((d.userdata[0] > 0) | (align_pos_err < 0.007), 1.0, 0.0)]
  )
  return jax.lax.cond(userdata[0] > 0, lambda: insert, lambda: align), userdata


def get_models_and_cost_fn() -> tuple[
    mujoco.MjModel,
    mujoco.MjModel,
    Callable[
        [mjx.Model, mjx.Data, instruction.Instruction], Tuple[jax.Array, Any]
    ],
    Callable[[mjx.Model, mjx.Data], Tuple[instruction.Instruction, jax.Array]],
]:
  """Returns a planning model, a sim model, and a cost function."""
  path = epath.Path(
      'build/mjpc/tasks/bimanual/'
  )
  model_file_name = 'mjx_insert_cube.xml'
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
  return sim_model, plan_model, instruction.instruction_cost, make_instruction
