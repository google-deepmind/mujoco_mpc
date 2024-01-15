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
import mujoco
import numpy as np

# predictive sampling
# https://arxiv.org/abs/2212.00541


# policy class for predictive sampling
class Policy:
  # initialize policy
  def __init__(
      self,
      naction: int,
      horizon: float,
      splinestep: float,
      interp: str = "zero",
      limits: np.array = None,
  ):
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

  # find interval containing value
  def _find_interval(self, sequence: np.array, value: float) -> [int, int]:
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

  # compute slope at value
  def _slope(self, input: np.array, output: np.array, value: float) -> np.array:
    # bounds
    bounds = self._find_interval(input, value)

    # length of inputs
    L = len(input)

    # lower out of bounds
    if bounds[0] == 0 and bounds[1] == 0:
      if L > 2:
        return (output[:, bounds[1] + 1] - output[:, bounds[1]]) / (
            input[bounds[1] + 1] - input[bounds[1]]
        )
      return np.zeros(output.shape[0])

    # upper out of bounds
    if bounds[0] == L - 1 and bounds[1] == L - 1:
      if L > 2:
        return (output[:, bounds[0]] - output[:, bounds[0] - 1]) / (
            input[bounds[0]] - input[bounds[0] - 1]
        )
      return np.zeros(output.shape[0])

    # lower boundary
    if bounds[0] == 0:
      return (output[:, bounds[1]] - output[:, bounds[0]]) / (
          input[bounds[1]] - input[bounds[0]]
      )

    # internal interval
    return 0.5 * (output[:, bounds[1]] - output[:, bounds[0]]) / (
        input[bounds[1]] - input[bounds[0]]
    ) + 0.5 * (output[:, bounds[0]] - output[:, bounds[0] - 1]) / (
        input[bounds[0]] - input[bounds[0] - 1]
    )

  # get action from policy
  def action(self, time: float) -> np.array:
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

  # resample policy plan from current time
  def resample(self, time: float):
    # new times and parameters
    times = np.array(
        [i * self._splinestep + time for i in range(self._nspline)], dtype=float
    )
    parameters = np.vstack([self.action(t) for t in times]).T

    # update
    self._times = times
    self._parameters = parameters

  # add zero-mean Gaussian noise to policy parameters
  def add_noise(self, scale: float):
    # clamp within limits
    self._parameters = self.clamp(
        self._parameters
        + np.random.normal(scale=scale, size=(self._naction, self._nspline))
    )

  # return a copy of the policy with noisy parameters
  def noisy_copy(self, scale: float) -> Policy:
    # create new policy object
    policy = Policy(self._naction, self._horizon, self._splinestep)

    # copy policy parameters into new object
    policy._parameters = np.copy(self._parameters)

    # get noisy parameters
    policy.add_noise(scale)

    return policy

  # clamp action with limits
  def clamp(self, action: np.array) -> np.array:
    # clamp within limits
    if self._limits is not None:
      return np.minimum(
          np.maximum(self._limits[:, 0], action), self._limits[:, 1]
      )
    return action


# rollout
def rollout(
    qpos: np.array,
    qvel: np.array,
    act: np.array,
    time: float,
    mocap_pos: np.array,
    mocap_quat: np.array,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    reward: function,
    policy: Policy,
    horizon: float,
) -> float:
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


# predictive sampling planner class
class Planner:
  # initialize planner
  def __init__(
      self,
      model: mujoco.MjModel,
      reward: function,
      horizon: float,
      splinestep: float,
      planstep: float,
      nsample: int,
      noise_scale: float,
      nimprove: int,
      interp: str = "zero",
      limits: bool = True,
  ):
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

  # action from policy
  def action_from_policy(self, time: float) -> np.array:
    return self.policy.action(time)

  # improve policy
  def improve_policy(
      self,
      qpos: np.array,
      qvel: np.array,
      act: np.array,
      time: float,
      mocap_pos: np.array,
      mocap_quat: np.array,
  ):
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
