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

"""Predictive sampling for MPC."""

import time
from typing import Callable, Tuple

import jax
from jax import numpy as jnp
import mujoco
from mujoco import mjx
from mujoco.mjx._src import dataclasses

CostFn = Callable[[mjx.Model, mjx.Data], jax.Array]


class Planner(dataclasses.PyTreeNode):
  """Predictive sampling planner.

  Attributes:
    model: MuJoCo model
    cost: function returning per-timestep cost
    noise_scale: standard deviation of zero-mean Gaussian
    horizon: planning duration (steps)
    nspline: number of spline points to explore
    nsample: number of action sequence candidates sampled
    interp: type of action interpolation
  """
  model: mjx.Model
  cost: CostFn
  noise_scale: jax.Array
  horizon: int
  nspline: int
  nsample: int
  interp: str


def _rollout(p: Planner, d: mjx.Data, policy: jax.Array) -> jax.Array:
  """Expand the policy into actions and roll out dynamics and cost."""
  actions = get_actions(p, policy)

  def step(d, action):
    d = d.replace(ctrl=action)
    cost = p.cost(p.model, d)
    d = mjx.step(p.model, d)
    return d, cost

  _, costs = jax.lax.scan(step, d, actions, length=p.horizon)

  return jnp.sum(costs)


def get_actions(p: Planner, policy: jax.Array) -> jax.Array:
  """Gets actions over a planning duration from a policy."""
  if p.interp == 'zero':
    indices = [i * p.nspline // p.horizon for i in range(p.horizon)]
    actions = policy[jnp.array(indices)]
  elif p.interp == 'linear':
    locs = jnp.array([i * p.nspline / p.horizon for i in range(p.horizon)])
    idx = locs.astype(int)
    actions = jax.vmap(jnp.multiply)(policy[idx], 1 - locs + idx)
    actions += jax.vmap(jnp.multiply)(policy[idx + 1], locs - idx)
  else:
    raise ValueError(f'unimplemented interp: {p.interp}')

  return actions


def improve_policy(
    p: Planner, d: mjx.Data, policy: jax.Array, rng: jax.Array
) -> Tuple[jax.Array, jax.Array]:
  """Improves policy."""
  limit = p.model.actuator_ctrlrange

  # create noisy policies, with nominal policy at index 0
  noise = jax.random.normal(rng, (p.nsample, p.nspline, p.model.nu))
  noise = noise * p.noise_scale * (limit[:, 1] - limit[:, 0])
  policies = jnp.concatenate((policy[None], policy + noise))
  # clamp actions to ctrlrange
  policies = jnp.clip(policies, limit[:, 0], limit[:, 1])

  # perform nsample + 1 parallel rollouts
  costs = jax.vmap(_rollout, in_axes=(None, None, 0))(p, d, policies)
  costs = jnp.nan_to_num(costs, nan=jnp.inf)
  best_id = jnp.argmin(costs)

  return policies[best_id], costs[best_id]


def resample(p: Planner, policy: jax.Array, steps_per_plan: int) -> jax.Array:
  """Resample policy to new advanced time."""
  if p.horizon % p.nspline != 0:
    raise ValueError("horizon must be divisible by nspline")
  splinesteps = p.horizon // p.nspline
  if splinesteps % steps_per_plan != 0:
    raise ValueError(
        f'splinesteps ({splinesteps}) must be divisible by steps_per_plan'
        f' ({steps_per_plan})'
    )
  roll = splinesteps // steps_per_plan
  policy = jnp.roll(policy, -roll, axis=0)
  policy = policy.at[-roll:].set(policy[-roll - 1])

  return policy


def set_state(d_out, d_in):
  return d_out.replace(
      time=d_in.time, qpos=d_in.qpos, qvel=d_in.qvel, act=d_in.act,
      ctrl=d_in.ctrl)


def receding_horizon_optimization(
    p: Planner,
    plan_model_cpu,
    sim_model_cpu,
    nsteps,
    steps_per_plan,
    frame_skip,
):
  d = mujoco.MjData(plan_model_cpu)
  d = mjx.put_data(plan_model_cpu, d)
  m = mjx.put_model(plan_model_cpu)
  p = p.replace(model=m)
  jitted_cost = jax.jit(p.cost)

  policy = jnp.zeros((p.nspline, m.nu))
  rng = jax.random.key(0)
  improve_fn = (
      jax.jit(improve_policy)
      .lower(p, d, policy, rng)
      .compile()
  )
  step_fn = jax.jit(mjx.step).lower(m, d).compile()

  trajectory, costs = [], []
  plan_time = 0
  sim_data = mujoco.MjData(sim_model_cpu)
  mujoco.mj_resetDataKeyframe(sim_model_cpu, sim_data, 0)
  # without kinematics, the first cost is off:
  mujoco.mj_forward(sim_model_cpu, sim_data)
  sim_data = mjx.put_data(sim_model_cpu, sim_data)
  sim_model = mjx.put_model(sim_model_cpu)
  actions = get_actions(p, policy)

  for step in range(nsteps):
    if step % steps_per_plan == 0:
      # resample policy to new advanced time
      print('re-planning')
      policy = resample(p, policy, steps_per_plan)
      beg = time.perf_counter()
      d = set_state(d, sim_data)
      policy, _ = improve_fn(p, d, policy, jax.random.key(step))
      plan_time += time.perf_counter() - beg
      actions = get_actions(p, policy)

    sim_data = sim_data.replace(ctrl=actions[0])
    cost = jitted_cost(sim_model, sim_data)
    sim_data = step_fn(sim_model, sim_data)
    costs.append(cost)
    print(f'step: {step}')
    print(f'cost: {cost}')
    if step % frame_skip == 0:
      trajectory.append(jax.device_get(sim_data.qpos))

  return trajectory, costs, plan_time
