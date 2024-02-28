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


@jax.jit
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
    raise ValueError(f'unimplemented interpolation method: {p.interp}')

  return actions
v_get_actions = jax.vmap(get_actions, in_axes=[None, 0])


def improve_policy(
    p: Planner, d: mjx.Data, policy: jax.Array, rng: jax.Array
) -> Tuple[jax.Array, jax.Array]:
  """Improves policy."""

  # create noisy policies, with nominal policy at index 0
  noise = (
      jax.random.normal(rng, (p.nsample, p.nspline, p.model.nu)) * p.noise_scale
  )
  policies = jnp.concatenate((policy[None], policy + noise))
  # clamp actions to ctrlrange
  limit = p.model.actuator_ctrlrange
  policies = jnp.clip(policies, limit[:, 0], limit[:, 1])
  # perform nsample + 1 parallel rollouts
  costs = jax.vmap(_rollout, in_axes=(None, None, 0))(p, d, policies)
  costs = jnp.nan_to_num(costs, nan=jnp.inf)
  winners = jnp.argmin(costs)

  return policies[winners], winners


def resample(p: Planner, policy: jax.Array, steps_per_plan: int) -> jax.Array:
  """Resample policy to new advanced time."""
  if p.interp == 'zero':
    return policy  # assuming steps_per_plan < splinesteps
  elif p.interp == 'linear':
    actions = v_get_actions(p, policy)
    roll = steps_per_plan
    actions = jnp.roll(actions, -roll, axis=1)
    actions = actions.at[:, -roll:, :].set(actions[:, [-1], :])
    idx = jnp.floor(jnp.linspace(0, p.horizon, p.nspline)).astype(int)
    return actions[:, idx, :]

  return policy


def set_state(d, state):
  return d.replace(
      time=state.time, qpos=state.qpos, qvel=state.qvel, act=state.act,
      ctrl=state.ctrl)
set_states = jax.vmap(set_state, in_axes=[0, 0])


def receding_horizon_control(
    p: Planner,
    init_policy,
    rng,
    plan_model_cpu,
    sim_model_cpu,
    nsteps,
    nplans,
    steps_per_plan,
):
  """Receding horizon optimization, all nplans start from same keyframe."""
  plan_data = mujoco.MjData(plan_model_cpu)
  plan_data = mjx.put_data(plan_model_cpu, plan_data)
  m = mjx.put_model(plan_model_cpu)
  p = p.replace(model=m)

  sim_data = mujoco.MjData(sim_model_cpu)
  mujoco.mj_resetDataKeyframe(sim_model_cpu, sim_data, 0)
  # without kinematics, the first cost is off:
  mujoco.mj_forward(sim_model_cpu, sim_data)
  sim_data = mjx.put_data(sim_model_cpu, sim_data)
  sim_model = mjx.put_model(sim_model_cpu)

  multi_actions = v_get_actions(p, init_policy)
  first_actions = multi_actions[:, 0, :]  # just the first actions
  # duplicate data for each plan
  def set_action(data, action):
    return data.replace(ctrl=action)

  duplicate_data = jax.vmap(set_action, in_axes=[None, 0], out_axes=0)
  sim_datas = duplicate_data(sim_data, first_actions)
  plan_datas = duplicate_data(plan_data, first_actions)

  def step_and_cost(data, action):
    data = data.replace(ctrl=action)
    cost = p.cost(sim_model, data)
    data = mjx.step(sim_model, data)
    return data, cost

  multi_step = jax.vmap(step_and_cost, in_axes=[0, 0])

  def step_cost_traj(datas, actions):
    data, cost = multi_step(datas, actions)
    return data, (cost, data.qpos)

  improve_fn = jax.vmap(improve_policy, in_axes=(None, 0, 0, 0))

  def plan_and_step(carry, rng):
    sim_datas, policy = carry
    policy = resample(p, policy, steps_per_plan)
    policy, _ = improve_fn(
        p,
        set_states(plan_datas, sim_datas),
        policy,
        jax.random.split(rng, nplans),
    )
    multi_actions = v_get_actions(p, policy)
    sim_datas, (cost, traj) = jax.lax.scan(
        step_cost_traj,
        sim_datas,
        jnp.transpose(multi_actions[:, :steps_per_plan, :], (1, 0, 2)),
    )
    return (sim_datas, policy), (cost, traj)

  (_, final_policy), (costs, trajs) = jax.lax.scan(
      plan_and_step,
      (sim_datas, init_policy),
      jax.random.split(rng, nsteps // steps_per_plan),
  )
  return (
      trajs.reshape(nsteps, nplans, sim_model.nq).transpose(1, 0, 2),
      costs.reshape(nsteps, nplans).transpose(),
      final_policy,
  )
