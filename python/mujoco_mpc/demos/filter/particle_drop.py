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

import matplotlib.pyplot as plt
import mediapy as media
import mujoco
from mujoco_mpc import filter as filter_lib
import numpy as np

# %%
xml = """
<mujoco model="Particle1D">
  <visual>
    <headlight ambient=".4 .4 .4" diffuse=".8 .8 .8" specular="0.1 0.1 0.1"/>
    <map znear=".01"/>
    <quality shadowsize="2048"/>
  </visual>

  <custom>
    <!-- filters: ground truth (0), EKF (1), UKF (2), batch (3) -->
    <numeric name="estimator" data="3"/>
    <!-- 3 <= batch_size <= 128 -->
    <numeric name="batch_configuration_length" data="32"/>
    <numeric name="unscented_alpha" data="1.0e-3"/>
    <numeric name="unscented_beta" data="2.0"/>
  </custom>

  <default>
    <geom solimp="0 0.95 0.001"/>
  </default>

  <asset>
    <texture name="skybox" type="skybox" builtin="gradient" rgb1="0 0 0" rgb2="0 0 0"
             width="800" height="800" mark="random" markrgb="0 0 0"/>
  </asset>

  <asset>
      <texture name="grid" type="2d" builtin="checker" rgb1=".1 .2 .3" rgb2=".2 .3 .4" width="300" height="300" mark="edge" markrgb=".2 .3 .4"/>
      <material name="grid" texture="grid" texrepeat="1 1" texuniform="true" reflectance=".2"/>
      <material name="self" rgba=".7 .5 .3 1"/>
      <material name="self_default" rgba=".7 .5 .3 1"/>
      <material name="self_highlight" rgba="0 .5 .3 1"/>
      <material name="effector" rgba=".7 .4 .2 1"/>
      <material name="effector_default" rgba=".7 .4 .2 1"/>
      <material name="effector_highlight" rgba="0 .5 .3 1"/>
      <material name="decoration" rgba=".3 .5 .7 1"/>
      <material name="eye" rgba="0 .2 1 1"/>
      <material name="target" rgba="0 1 0 0.5"/>
      <material name="target_default" rgba=".6 .3 .3 1"/>
      <material name="target_highlight" rgba=".6 .3 .3 .4"/>
      <material name="site" rgba=".5 .5 .5 .3"/>
  </asset>

  <option timestep="0.005" />

  <worldbody>
    <light name="light" pos="0 0 1"/>
    <camera name="fixed" pos="0 0 .75" quat="1 0 0 0"/>
    <geom name="ground" type="plane" pos="0 0 0" size=".3 .3 .1" material="grid" />

    <body name="pointmass" pos="0 0 0.25">
      <joint name="root_z" type="slide"  pos="0 0 0" axis="0 0 1" />
      <geom name="pointmass" type="sphere" size=".01" material="self" mass="1.0"/>
      <site name="center" type="sphere" size="0.01" rgba="1 0 0 0" pos="0 0 0"/>
    </body>
  </worldbody>

  <actuator>
    <motor name="z_motor" joint="root_z"/>
  </actuator>

  <sensor>
    <jointpos name="pos_z" joint="root_z"/>
    <accelerometer name="linacc" site="center"/>
  </sensor>
</mujoco>
"""
# %%
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model)

# %%
# rollout
T = 100

# trajectories
qpos = np.zeros((model.nq, T))
qvel = np.zeros((model.nv, T))
qacc = np.zeros((model.nv, T))
ctrl = np.zeros((model.nu, T))
qfrc = np.zeros((model.nv, T))
sensor = np.zeros((model.nsensordata, T))
time = np.zeros(T)

# state
nx = model.nq + model.nv + model.na
state_estimate = np.zeros((nx, T))

# rollout
mujoco.mj_resetData(model, data)

# frames
frames = []
FPS = 1.0 / model.opt.timestep

## set up filter
filter = filter_lib.Filter(model=model)

# initialize state
state_initial = np.array([data.qpos[0], data.qvel[0]])
# state_initial += np.array([0.01, 0.01]) # perturb
filter.state(state=state_initial)

# initialize covariance
covariance_initial = np.array([[1.0e-4, 0.0], [0.0, 1.0e-4]])
filter.covariance(covariance=covariance_initial)

# initialize noise
noise_process = np.array([1.0e-3, 1.0e-3])
noise_sensor = np.array([1.0e-2, 1.0e-2, 1.0e-2, 1.0e-2])
filter.noise(process=noise_process, sensor=noise_sensor)

# noisy sensor
noisy_sensor = np.zeros((model.nsensordata, T))

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
  noisy_sensor[:, t] = sensor[:, t] + 1.0 * np.random.normal(
      scale=1.0e-3, size=model.nsensordata
  )
  time[t] = data.time

  # cache filter estimate
  state = filter.state()["state"]
  state_estimate[:, t] = state

  # filter (measurement update)
  filter.update(ctrl=ctrl[:, t], sensor=noisy_sensor[:, t])

  # Euler
  mujoco.mj_Euler(model, data)

  # Render and save frames.
  renderer.update_scene(data)
  pixels = renderer.render()
  frames.append(pixels)

# Display video.
media.show_video(frames, fps=FPS)

# %%
# height plot
fig = plt.figure()

plt.plot(time, qpos[0, :], label="simulation", color="blue")
plt.plot(time, noisy_sensor[0, :], label="noisy sensor", color="magenta")
plt.plot(time, state_estimate[0, :], label="estimate", color="orange")

plt.legend()
plt.title("Filter")
plt.xlabel("Time (s)")
plt.ylabel("Height (m)")

# %%
# velocity plot
fig = plt.figure()

plt.plot(time, qvel[0, :], label="simulation", color="blue")
plt.plot(time, state_estimate[1, :], label="estimate", color="orange")

plt.legend()
plt.title("Filter")
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m / s)")
