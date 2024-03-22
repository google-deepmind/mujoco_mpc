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

from typing import Callable

from etils import epath
# internal import
import jax
from jax import numpy as jnp
import mujoco
from mujoco import mjx


def bring_to_target(m: mjx.Model, d: mjx.Data) -> jax.Array:
  """Returns cost for bimanual bring to target task."""
  # reach
  left_gripper_site_index = 3
  right_gripper_site_index = 6
  box_body_index = m.nbody - 1
  left_gripper_pos = d.site_xpos[..., left_gripper_site_index, :]
  right_gripper_pos = d.site_xpos[..., right_gripper_site_index, :]
  box_pos = d.xpos[..., box_body_index, :]

  reach_l = left_gripper_pos - box_pos
  reach_r = right_gripper_pos - box_pos

  target = jnp.array([-0.4, -0.2, 0.3])
  bring = box_pos - target

  residuals = [reach_l, reach_r, bring]
  weights = [0.1, 0.1, 1]
  norm_p = [0.005, 0.005, 0.003]

  # NormType::kL2: y = sqrt(x*x' + p^2) - p
  terms = []
  for t, w, p in zip(residuals, weights, norm_p):
    terms.append(w * jnp.sqrt(jnp.sum(t**2, axis=-1) + p**2) - p)
  costs = jnp.sum(jnp.array(terms), axis=-1)

  return costs


def get_models_and_cost_fn() -> tuple[
    mujoco.MjModel,
    mujoco.MjModel,
    Callable[[mjx.Model, mjx.Data], jax.Array],
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
  plan_model.opt.timestep = 0.01  # incidentally, already the case
  return sim_model, plan_model, bring_to_target
