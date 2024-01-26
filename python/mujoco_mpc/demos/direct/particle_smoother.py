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
from mujoco_mpc import direct as direct_lib
import numpy as np

# %%
# 1D Particle Model
xml = """
<mujoco model="Particle1D">
  <visual>
    <headlight ambient=".4 .4 .4" diffuse=".8 .8 .8" specular="0.1 0.1 0.1"/>
    <map znear=".01"/>
    <quality shadowsize="2048"/>
  </visual>

  <asset>
    <texture name="skybox" type="skybox" builtin="gradient" rgb1="0 0 0" rgb2="0 0 0"
             width="800" height="800" mark="random" markrgb="0 0 0"/>
  </asset>

  <asset>
      <texture name="grid" type="2d" builtin="checker" rgb1=".1 .2 .3" rgb2=".2 .3 .4" width="300" height="300" mark="edge" markrgb=".2 .3 .4"/>
      <material name="grid" texture="grid" texrepeat="1 1" texuniform="true" reflectance=".2"/>
      <material name="self" rgba=".7 .5 .3 1"/>
  </asset>

  <option timestep="0.001" />

  <worldbody>
    <light name="light" pos="0 0 1"/>
    <geom name="ground" type="plane" pos="0 0 0" size=".3 .3 .1" material="grid" />

    <body name="pointmass" pos="0 0 0">
      <joint name="root_z" type="slide" damping="0" pos="0 0 0" axis="0 0 1" />
      <geom name="pointmass" type="sphere" size=".01" material="self" mass="1.0"/>
    </body>
  </worldbody>

  <actuator>
    <motor name="z_motor" joint="root_z"/>
  </actuator>

  <sensor>
    <jointpos name="joint_z" joint="root_z" />
  </sensor>
</mujoco>"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model)
# %%
## rollout
np.random.seed(0)

# simulation horizon
T = 1000

# trajectories
qpos = np.zeros((model.nq, T))
qvel = np.zeros((model.nv, T))
qacc = np.zeros((model.nv, T))
ctrl = np.zeros((model.nu, T))
qfrc = np.zeros((model.nv, T))
sensor = np.zeros((model.nsensordata, T))
noisy_sensor = np.zeros((model.nsensordata, T))
time = np.zeros(T)

# set initial state
mujoco.mj_resetData(model, data)
data.qpos[0] = 0.025

# frames
frames = []
FPS = 1.0 / model.opt.timestep

# simulate
for t in range(T):
  # set ctrl
  data.ctrl = np.zeros(model.nu)

  # forward dynamics
  mujoco.mj_forward(model, data)

  # cache
  qpos[:, t] = data.qpos
  qvel[:, t] = data.qvel
  qacc[:, t] = data.qacc
  ctrl[:, t] = data.ctrl
  qfrc[:, t] = data.qfrc_actuator
  sensor[:, t] = data.sensordata
  time[t] = data.time

  # noisy sensors
  noisy_sensor[:, t] = sensor[:, t] + np.random.normal(
      scale=1.0e-3, size=model.nsensordata
  )

  # intergrate with Euler
  mujoco.mj_Euler(model, data)

  # render and save frames
  # renderer.update_scene(data)
  # pixels = renderer.render()
  # frames.append(pixels)

# display video.
# SLOWDOWN = 0.5
# media.show_video(frames, fps=SLOWDOWN * FPS)
# %%
# plot position
fig = plt.figure()

# position (sensor)
plt.plot(time, noisy_sensor[0, :], label="sensor", ls="--", color="cyan")

# position (simulation)
plt.plot(time, qpos[0, :], label="simulation", color="black")

plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("Position")

# plot velocity
fig = plt.figure()

# velocity (simulation)
plt.plot(time, qvel[0, :], label="simulation", color="black")

plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("Velocity")
# %%
# optimizer model
model_optimizer = mujoco.MjModel.from_xml_string(xml)

# direct optimizer
configuration_length = T
optimizer = direct_lib.Direct(
    model=model_optimizer, configuration_length=configuration_length
)
# %%
# configuration initialization
qinit = np.zeros((model.nq, configuration_length))

# set data in optimizer
for t in range(configuration_length):
  # constant initialization
  qinit[:, t] = qpos[:, 0] + np.random.normal(scale=1.0e-3, size=model.nq)

  # set data
  optimizer.data(
      t,
      configuration=qinit[:, t],
      sensor_measurement=noisy_sensor[:, t],
      force_measurement=qfrc[:, t],
      time=np.array([time[t]]),
  )
# %%
# set noise std
optimizer.noise(
    process=np.full(model.nv, 1.0), sensor=np.full(model.nsensor, 5.0e-1)
)

# set settings
optimizer.settings(
    sensor_flag=True,
    force_flag=True,
    max_smoother_iterations=100,
    max_search_iterations=1000,
    regularization_initial=1.0e-12,
    gradient_tolerance=1.0e-6,
    search_direction_tolerance=1.0e-6,
    cost_tolerance=1.0e-6,
)

# optimize
optimizer.optimize()

# costs
optimizer.print_cost()

# status
optimizer.print_status()
# %%
# get optimizer solution
q_est = np.zeros((model_optimizer.nq, configuration_length))
v_est = np.zeros((model_optimizer.nv, configuration_length))
s_est = np.zeros((model_optimizer.nsensordata, configuration_length))
t_est = np.zeros(configuration_length)
for t in range(configuration_length):
  data_ = optimizer.data(t)
  q_est[:, t] = data_["configuration"]
  v_est[:, t] = data_["velocity"]
  s_est[:, t] = data_["sensor_prediction"]
  t_est[t] = data_["time"]

# plot position
fig = plt.figure()

# position
plt.plot(time, qpos[0, :], label="simulation", color="black")
plt.plot(time, qinit[0, :], label="initialization", color="orange")
plt.plot(t_est, q_est[0, :], label="optimized", ls="--", color="magenta")

plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("Position")

# plot velocity
fig = plt.figure()

# velocity (simulation)
plt.plot(time, qvel[0, :], label="simulation", color="black")

# velocity (optimizer)
plt.plot(t_est[1:], v_est[0, 1:], label="optimized", ls="--", color="magenta")

plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("Velocity")
