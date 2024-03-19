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

from brax.base import State
import jax
from jax import numpy as jnp
from mujoco import mjx
from mujoco.mjx._src import dataclasses

CostFn = Callable[[mjx.Model, mjx.Data], jax.Array]


class Planner(dataclasses.PyTreeNode):
  """Predictive sampling planner.

  Attributes:
    model: MJX model
    cost: function returning per-timestep cost
    noise_scale: standard deviation of zero-mean Gaussian
    horizon: planning duration (steps)
    nspline: number of spline points to explore
    nsample: number of action sequence candidates sampled
    interp: type of action interpolation
  """
  model: mjx.Model
  cost: CostFn
  noise_scale: float
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
    raise ValueError(f'unimplemented interpolation method: {p.interp}')

  return actions


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
    actions = get_actions(p, policy)
    roll = steps_per_plan
    actions = jnp.roll(actions, -roll, axis=-2)
    actions = actions.at[..., -roll:, :].set(actions[..., [-1], :])
    idx = jnp.floor(jnp.linspace(0, p.horizon, p.nspline)).astype(int)
    return actions[..., idx, :]
  return policy


def set_state(d, state):
  return d.replace(
      time=state.time, qpos=state.qpos, qvel=state.qvel, act=state.act,
      ctrl=state.ctrl)

def mpc_rollout(
    nsteps,
    steps_per_plan,
    p: Planner,
    init_policy,
    rng,
    sim_model,
    sim_data,
):
  """Receding horizon optimization starting from sim_data's state."""
  plan_data = mjx.make_data(p.model)

  def plan_and_step(carry, rng):
    sim_data, policy = carry
    policy = resample(p, policy, steps_per_plan)
    policy, _ = improve_policy(
        p,
        set_state(plan_data, sim_data),
        policy,
        rng,
    )
    def step(d, action):
      d = d.replace(ctrl=action)
      cost = p.cost(sim_model, d)
      d = mjx.step(sim_model, d)
      return d, (
          cost,
          State(q=d.qpos, qd=d.qvel, x=None, xd=None, contact=None),
      )
    actions = get_actions(p, policy)
    sim_data, (cost, traj) = jax.lax.scan(
        step,
        sim_data,
        actions[:steps_per_plan, :],
    )
    return (sim_data, policy), (cost, traj)

  (sim_data, final_policy), (costs, trajs) = jax.lax.scan(
      plan_and_step,
      (sim_data, init_policy),
      jax.random.split(rng, nsteps // steps_per_plan),
  )
  return sim_data, final_policy, costs.flatten(), trajs
