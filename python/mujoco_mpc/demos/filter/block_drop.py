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

# %matplotlib inline
# %%
xml = """
<mujoco>
  <option timestep="0.005"/>
  <default>
    <geom solimp="0 0.95 0.001"/>
  </default>

  <visual>
    <headlight ambient=".4 .4 .4" diffuse=".8 .8 .8" specular="0.1 0.1 0.1"/>
    <map znear=".01"/>
    <quality shadowsize="2048"/>
    <global offheight="1024" offwidth="1024"/>
  </visual>

  <asset>
    <texture name="skybox" type="skybox" builtin="gradient" rgb1="0 0 0" rgb2="0 0 0"
             width="800" height="800" mark="random" markrgb="0 0 0"/>
  </asset>

  <custom>
    <!-- filters: ground truth (0), EKF (1), UKF (2), batch (3) -->
    <numeric name="estimator" data="1" />
    <!-- 3 <= batch_size <= 128 -->
    <numeric name="batch_configuration_length" data="12" />
    <numeric name="unscented_alpha" data="1.0e-3"/>
    <numeric name="unscented_beta" data="2.0"/>
  </custom>

  <asset>
      <texture name="grid" type="2d" builtin="checker" rgb1=".1 .2 .3" rgb2=".2 .3 .4" width="300" height="300" mark="edge" markrgb=".2 .3 .4"/>
      <material name="grid" texture="grid" texrepeat="1 1" texuniform="true" reflectance=".2"/>
      <material name="self" rgba=".7 .5 .3 1"/>
  </asset>

  <worldbody>
    <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
    <camera pos="-0.079 -0.587 0.400" xyaxes="0.951 -0.310 0.000 0.133 0.410 0.902"/>
    <geom name="ground" type="plane" pos="0 0 0" size=".3 .3 .1" material="grid"/>
    <geom size=".07" pos="-.03 0.03 0"/>
    <body name="m1" mocap="true" pos="0.1 0.1 0.1">
      <geom type="sphere" size="0.0075" contype="0" conaffinity="0" rgba="1 0 0 0.5"/>
    </body>
    <body name="m2" mocap="true" pos="0.1 0.1 0.1">
      <geom type="sphere" size="0.0075" contype="0" conaffinity="0" rgba="1 0 0 0.5"/>
    </body>
    <body name="m3" mocap="true" pos="0.1 0.1 0.1">
      <geom type="sphere" size="0.0075" contype="0" conaffinity="0" rgba="1 0 0 0.5"/>
    </body>
    <body name="m4" mocap="true" pos="0.1 0.1 0.1">
      <geom type="sphere" size="0.0075" contype="0" conaffinity="0" rgba="1 0 0 0.5"/>
    </body>
    <body name="m5" mocap="true" pos="0.1 0.1 0.1">
      <geom type="sphere" size="0.0075" contype="0" conaffinity="0" rgba="1 0 0 0.5"/>
    </body>
    <body name="m6" mocap="true" pos="0.1 0.1 0.1">
      <geom type="sphere" size="0.0075" contype="0" conaffinity="0" rgba="1 0 0 0.5"/>
    </body>
    <body name="m7" mocap="true" pos="0.1 0.1 0.1">
      <geom type="sphere" size="0.0075" contype="0" conaffinity="0" rgba="1 0 0 0.5"/>
    </body>
    <body name="m8" mocap="true" pos="0.1 0.1 0.1">
      <geom type="sphere" size="0.0075" contype="0" conaffinity="0" rgba="1 0 0 0.5"/>
    </body>
    <body name="root" pos="0 0 0.25">
      <joint type="free"/>
      <geom type="box" size=".05 .05 .05" material="self" mass="1.0"/>
      <site name="corner1" type="sphere" size="0.05" rgba="1 0 0 0" pos=".05 .05 .05"/>
      <site name="corner2" type="sphere" size="0.05" rgba="1 0 0 0" pos="-.05 .05 .05"/>
      <site name="corner3" type="sphere" size="0.05" rgba="1 0 0 0" pos=".05 -.05 .05"/>
      <site name="corner4" type="sphere" size="0.05" rgba="1 0 0 0" pos=".05 .05 -.05"/>
      <site name="corner5" type="sphere" size="0.05" rgba="1 0 0 0" pos="-.05 -.05 .05"/>
      <site name="corner6" type="sphere" size="0.05" rgba="1 0 0 0" pos=".05 -.05 -.05"/>
      <site name="corner7" type="sphere" size="0.05" rgba="1 0 0 0" pos="-.05 .05 -.05"/>
      <site name="corner8" type="sphere" size="0.05" rgba="1 0 0 0" pos="-.05 -.05 -.05"/>
    </body>
  </worldbody>

  <sensor>
    <!-- corner positions -->
    <framepos name="corner_position1" objtype="site" objname="corner1"/>
    <framepos name="corner_position2" objtype="site" objname="corner2"/>
    <framepos name="corner_position3" objtype="site" objname="corner3"/>
    <framepos name="corner_position4" objtype="site" objname="corner4"/>
    <framepos name="corner_position5" objtype="site" objname="corner5"/>
    <framepos name="corner_position6" objtype="site" objname="corner6"/>
    <framepos name="corner_position7" objtype="site" objname="corner7"/>
    <framepos name="corner_position8" objtype="site" objname="corner8"/>
  </sensor>
</mujoco>
"""
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model)
filter = filter_lib.Filter(model=model)
# %%
# rollout
T = 200

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
FPS = 50.0
time_between_frames = 1.0 / FPS
physics_steps_between_frames = time_between_frames / model.opt.timestep

# initialize state
state_initial = np.concatenate((data.qpos[:], data.qvel[:]))

# noisy sensor
noisy_sensor = np.zeros((model.nsensordata, T))

# simulate
for t in range(T):
  # step, but do not integrate yet
  mujoco.mj_forward(model, data)

  # cache
  qpos[:, t] = data.qpos
  qvel[:, t] = data.qvel
  qacc[:, t] = data.qacc
  ctrl[:, t] = data.ctrl
  qfrc[:, t] = data.qfrc_actuator
  sensor[:, t] = data.sensordata
  noisy_sensor[:, t] = sensor[:, t] + np.random.normal(
      scale=1.0e-2, size=model.nsensordata
  )
  time[t] = data.time

  # integrate one timestep forward
  mujoco.mj_Euler(model, data)

  if t % physics_steps_between_frames == 0:
    renderer.update_scene(data)
    pixels = renderer.render()
    frames.append(pixels)

# display video
media.show_video(frames, fps=FPS)
# %%
# filter

# state_initial += np.array([0.025, -0.01]) # corrupt initialization
filter.state(state=state_initial)

# initialize covariance
covariance_initial = 1e-4 * np.eye(model.nv + model.nv + model.na)
filter.covariance(covariance=covariance_initial)

# initialize noise
noise_process = 1e-4 * np.ones((model.nv + model.nv + model.na))
noise_sensor = 1e-4 * np.ones((model.nsensordata))
filter.noise(process=noise_process, sensor=noise_sensor)

for t in range(T):
  state = filter.state()
  state_estimate[:, t] = state["state"]

  # filter (measurement update)
  filter.update(ctrl=ctrl[:, t], sensor=noisy_sensor[:, t])

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
plt.plot(time, state_estimate[model.nq, :], label="estimate", color="orange")

plt.legend()
plt.title("Filter")
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m / s)")
