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

from __future__ import annotations

import bisect
from typing import Callable, Tuple

import mujoco
import numpy as np


# predictive sampling
# https://arxiv.org/abs/2212.00541


class Policy:
  """Policy class for predictive sampling."""

  def __init__(
      self,
      naction: int,
      horizon: float,
      splinestep: float,
      interp: str = "zero",
      limits: np.ndarray | None = None):
    """Initialize policy class.

    Args:
        naction: number of actions
        horizon: planning horizon (seconds)
        splinestep: interval length between spline points
        interp (optional): type of action interpolation. Defaults to "zero".
        limits (optional): lower and upper bounds on actions. Defaults to None.
    """
    self._naction = naction
    self._splinestep = splinestep
    self._horizon = horizon
    self._nspline = int(horizon / splinestep) + 1
    self._parameters = np.zeros((self._naction, self._nspline), dtype=float)
    self._times = np.array(
        [t * self._splinestep for t in range(self._nspline)], dtype=float
    )
    self._interp = interp
    self._limits = limits

  def _find_interval(
      self, sequence: np.ndarray, value: float
  ) -> Tuple[int, int]:
    """Find neighboring indices in sequence containing value.

    Args:
        sequence: array of values
        value: value to find in interval

    Returns:
        lower and upper indices in sequence containing value
    """
    # bisection search to get interval
    upper = bisect.bisect_right(sequence, value)
    lower = upper - 1

    # length of sequence
    L = len(sequence)

    # return feasible interval
    if lower < 0:
      return (0, 0)
    if lower > L - 1:
      return (L - 1, L - 1)
    return (max(lower, 0), min(upper, L - 1))

  def _slope(
      self, times: np.ndarray, params: np.ndarray, value: float
  ) -> np.ndarray:
    """Compute interpolated slope vector at value.

    Args:
        times: sequence of time markers
        params: sequence of vectors
        value: input where to compute slope

    Returns:
        interpolated slope vector
    """
    # bounds
    bounds = self._find_interval(times, value)

    times_length = len(times)

    # lower out of bounds
    if bounds[0] == 0 and bounds[1] == 0:
      if times_length > 2:
        return (params[:, bounds[1] + 1] - params[:, bounds[1]]) / (
            times[bounds[1] + 1] - times[bounds[1]]
        )
      return np.zeros(params.shape[0])

    # upper out of bounds
    if bounds[0] == times_length - 1 and bounds[1] == times_length - 1:
      if times_length > 2:
        return (params[:, bounds[0]] - params[:, bounds[0] - 1]) / (
            times[bounds[0]] - times[bounds[0] - 1]
        )
      return np.zeros(params.shape[0])

    # lower boundary
    if bounds[0] == 0:
      return (params[:, bounds[1]] - params[:, bounds[0]]) / (
          times[bounds[1]] - times[bounds[0]]
      )

    # internal interval
    return 0.5 * (params[:, bounds[1]] - params[:, bounds[0]]) / (
        times[bounds[1]] - times[bounds[0]]
    ) + 0.5 * (params[:, bounds[0]] - params[:, bounds[0] - 1]) / (
        times[bounds[0]] - times[bounds[0] - 1]
    )

  def action(self, time: float) -> np.ndarray:
    """Return action from policy at time.

    Args:
        time: time value to evaluate plan for action

    Returns:
        interpolated action at time
    """
    # find interval containing time
    bounds = self._find_interval(self._times, time)

    # boundary case
    if bounds[0] == bounds[1]:
      return self.clamp(self._parameters[:, bounds[0]])

    # normalized time
    t = (time - self._times[bounds[0]]) / (
        self._times[bounds[1]] - self._times[bounds[0]]
    )

    if self._interp == "cubic":
      # spline coefficients
      c0 = 2.0 * t * t * t - 3.0 * t * t + 1.0
      c1 = (t * t * t - 2.0 * t * t + t) * (
          self._times[bounds[1]] - self._times[bounds[0]]
      )
      c2 = -2.0 * t * t * t + 3 * t * t
      c3 = (t * t * t - t * t) * (
          self._times[bounds[1]] - self._times[bounds[0]]
      )

      # slopes
      m0 = self._slope(self._times, self._parameters, self._times[bounds[0]])
      m1 = self._slope(self._times, self._parameters, self._times[bounds[1]])

      # interpolation
      return self.clamp(
          c0 * self._parameters[:, bounds[0]]
          + c1 * m0
          + c2 * self._parameters[:, bounds[1]]
          + c3 * m1
      )
    elif self._interp == "linear":
      return self.clamp(
          (1.0 - t) * self._parameters[:, bounds[0]]
          + t * self._parameters[:, bounds[1]]
      )
    else:  # self._interp == "zero"
      return self.clamp(self._parameters[:, bounds[0]])

  def resample(self, time: float):
    """Resample plan starting from time.

    Args:
        time: time value to start updated plan
    """
    # new times and parameters
    times = np.array(
        [i * self._splinestep + time for i in range(self._nspline)], dtype=float
    )
    parameters = np.vstack([self.action(t) for t in times]).T

    # update
    self._times = times
    self._parameters = parameters

  def add_noise(self, scale: float):
    """Add zero-mean Gaussian noise to plan.

    Args:
        scale: standard deviation of zero-mean Gaussian noise
    """
    # clamp within limits
    self._parameters = self.clamp(
        self._parameters
        + np.random.normal(scale=scale, size=(self._naction, self._nspline))
    )

  def noisy_copy(self, scale: float) -> Policy:
    """Return a copy of plan with added noise.

    Args:
        scale: standard deviation of zero-mean Gaussian noise

    Returns:
        copy of policy object with noisy plan
    """
    # create new policy object
    policy = Policy(self._naction, self._horizon, self._splinestep)

    # copy policy parameters into new object
    policy._parameters = np.copy(self._parameters)

    # get noisy parameters
    policy.add_noise(scale)

    return policy

  def clamp(self, action: np.ndarray) -> np.ndarray:
    """Return input clamped between limits.

    Args:
        action: input vector

    Returns:
        clamped input vector
    """
    # clamp within limits
    if self._limits is not None:
      return np.minimum(
          np.maximum(self._limits[:, 0], action), self._limits[:, 1]
      )
    return action


def rollout(
    qpos: np.ndarray,
    qvel: np.ndarray,
    act: np.ndarray,
    time: float,
    mocap_pos: np.ndarray,
    mocap_quat: np.ndarray,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    reward: Callable,
    policy: Policy,
    horizon: float,
) -> float:
  """Return total return by rollout out plan with forward dynamics.

  Args:
      qpos: initial configuration
      qvel: initial velocity
      act: initial activation
      time: current time
      mocap_pos: motion-capture body positions
      mocap_quat: motion-capture body orientations
      model: MuJoCo model
      data: MuJoCo data
      reward: function returning per-timestep reward value
      policy: plan for computing action at given time
      horizon: planning duration (seconds)

  Returns:
      total return (normalized by number of planning steps)
  """
  # number of steps
  steps = int(horizon / model.opt.timestep)

  # reset data
  mujoco.mj_resetData(model, data)

  # initialize state
  data.qpos = qpos
  data.qvel = qvel
  data.act = act
  data.time = time
  data.mocap_pos = mocap_pos
  data.mocap_quat = mocap_quat

  # initialize reward
  total_reward = 0.0

  # simulate
  for _ in range(steps):
    # get action from policy
    data.ctrl = policy.action(data.time)

    # evaluate current reward
    total_reward += reward(model, data)

    # step dynamics
    mujoco.mj_step(model, data)

  # terminal reward
  data.ctrl = np.zeros(model.nu)
  total_reward += reward(model, data)

  return total_reward / (steps + 1)


class Planner:
  """Predictive sampling planner class."""

  def __init__(
      self,
      model: mujoco.MjModel,
      reward: Callable,
      horizon: float,
      splinestep: float,
      planstep: float,
      nsample: int,
      noise_scale: float,
      nimprove: int,
      interp: str = "zero",
      limits: bool = True,
  ):
    """Initialize planner.

    Args:
        model: MuJoCo model
        reward: function returning per-timestep reward value
        horizon: planning duration (seconds)
        splinestep: interval length between spline points
        planstep: interval length between forward dynamics steps
        nsample: number of noisy plans to evaluate
        noise_scale: standard deviation of zero-mean Gaussian
        nimprove: number of iterations to improve plan for fixed initial
          state
        interp: type of action interpolation. Defaults to
          "zero".
        limits: lower and upper bounds on action. Defaults to
          True.
    """
    self._model = model.__copy__()
    self._model.opt.timestep = planstep
    self._data = mujoco.MjData(self._model)
    self._reward = reward
    self._horizon = horizon
    self.policy = Policy(
        model.nu,
        self._horizon,
        splinestep,
        interp=interp,
        limits=model.actuator_ctrlrange if limits else None,
    )
    self._nsample = nsample
    self._noise_scale = noise_scale
    self._nimprove = nimprove

  def action_from_policy(self, time: float) -> np.ndarray:
    """Return action at time from policy.

    Args:
        time: time to evaluate plan for action

    Returns:
        action interpolation at time
    """
    return self.policy.action(time)

  def improve_policy(
      self,
      qpos: np.ndarray,
      qvel: np.ndarray,
      act: np.ndarray,
      time: float,
      mocap_pos: np.ndarray,
      mocap_quat: np.ndarray,
  ):
    """Iteratively improve plan via searching noisy plans.

    Args:
        qpos: initial configuration
        qvel: initial velocity
        act: initial activation
        time: current time
        mocap_pos: motion-capture body position
        mocap_quat: motion-capture body orientation
    """
    # resample
    self.policy.resample(time)

    for _ in range(self._nimprove):
      # evaluate nominal policy
      reward_nominal = rollout(
          qpos,
          qvel,
          act,
          time,
          mocap_pos,
          mocap_quat,
          self._model,
          self._data,
          self._reward,
          self.policy,
          self._horizon,
      )

      # evaluate noisy policies
      policies = []
      rewards = []

      for _ in range(self._nsample):
        # noisy policy
        noisy_policy = self.policy.noisy_copy(self._noise_scale)
        noisy_reward = rollout(
            qpos,
            qvel,
            act,
            time,
            mocap_pos,
            mocap_quat,
            self._model,
            self._data,
            self._reward,
            noisy_policy,
            self._horizon,
        )

        # collect result
        policies.append(noisy_policy)
        rewards.append(noisy_reward)

      # find best policy
      idx = np.argmax(rewards)

      # return new policy
      if rewards[idx] > reward_nominal:
        self.policy = policies[idx]
