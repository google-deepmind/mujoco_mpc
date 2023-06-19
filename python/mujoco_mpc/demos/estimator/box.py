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
"""Estimate a box falling on a table from noisy simulated data."""

import os

import matplotlib.pyplot as plt
import mediapy as media
import numpy as np

import pathlib

import mujoco
from mujoco_mpc.grpc import estimator as estimator_lib

# %matplotlib inline
np.set_printoptions(precision=5, suppress=True, linewidth=100)
# %%
model_path = (
    pathlib.Path(os.path.abspath('')).parent.parent.parent
    / 'mujoco_mpc/mjpc/test/testdata/estimator/box/task1.xml'
)
model = mujoco.MjModel.from_xml_path(str(model_path))
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model)
# %%
configuration_length = 32
estimator = estimator_lib.Estimator(
    model=model,
    configuration_length=configuration_length
)
# %%
T = 256

# trajectories
qpos = np.zeros((model.nq, T))
qvel = np.zeros((model.nv, T))
qacc = np.zeros((model.nv, T))
ctrl = np.zeros((model.nu, T))
qfrc = np.zeros((model.nv, T))
sensor = np.zeros((model.nsensordata, T))
time = np.zeros(T)

# rollout
mujoco.mj_resetData(model, data)

data.qpos[2] = 1

# linear velocity
data.qvel[0] = 0.1
data.qvel[1] = -0.25
data.qvel[2] = 0.0

# angular velocity
data.qvel[3] = 0.5
data.qvel[4] = -0.75
data.qvel[5] = 0.625

# frames
frames = []
FPS = 1.0 / model.opt.timestep

# simulate
for t in range(T):

  # forward computes instantaneous qacc
  mujoco.mj_forward(model, data)

  # cache
  qpos[:, t] = data.qpos
  qvel[:, t] = data.qvel
  qacc[:, t] = data.qacc
  ctrl[:, t] = data.ctrl
  qfrc[:, t] = data.qfrc_actuator
  sensor[:, t] = data.sensordata
  time[t] = data.time

  # Euler
  mujoco.mj_Euler(model, data)

  # Render and save frames.
  renderer.update_scene(data, 0)
  pixels = renderer.render()
  frames.append(pixels)

# final cache
qpos[:, T - 1] = data.qpos
qvel[:, T - 1] = data.qvel
time[T - 1] = data.time

mujoco.mj_forward(model, data)
sensor[:, T - 1] = data.sensordata

# Display video.
media.show_video(frames, fps=FPS)
# %%
# plot position
fig = plt.figure()

# sensor
plt.plot(time, qpos[0, :], label='px', color='blue')
plt.plot(time, qpos[1, :], label='py', color='red')
plt.plot(time, qpos[2, :], label='pz', color='green')

plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Position')
