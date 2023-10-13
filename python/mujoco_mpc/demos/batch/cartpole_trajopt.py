# Copyright 2022 DeepMind Technologies Limited
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
import numpy as np

# set current directory to mjpc/python/mujoco_mpc
from mujoco_mpc import direct as direct_lib

# %matplotlib inline

# cart-pole model
xml = """
<mujoco model="Cartpole">

  <visual>
    <headlight ambient=".4 .4 .4" diffuse=".8 .8 .8" specular="0.1 0.1 0.1"/>
    <map znear=".01"/>
    <quality shadowsize="2048"/>
    <global elevation="-15" offwidth="1920" offheight="1080"/>
  </visual>

  <asset>
    <texture name="blue_grid" type="2d" builtin="checker" rgb1=".02 .14 .44" rgb2=".27 .55 1" width="300" height="300" mark="edge" markrgb="1 1 1"/>
    <material name="blue_grid" texture="blue_grid" texrepeat="1 1" texuniform="true" reflectance=".2"/>

    <texture name="grey_grid" type="2d" builtin="checker" rgb1=".26 .26 .26" rgb2=".6 .6 .6" width="300" height="300" mark="edge" markrgb="1 1 1"/>
    <material name="grey_grid" texture="grey_grid" texrepeat="1 1" texuniform="true" reflectance=".2"/>
    <texture name="skybox" type="skybox" builtin="gradient" rgb1=".66 .79 1" rgb2=".9 .91 .93" width="800" height="800"/>

    <material name="self" rgba=".7 .5 .3 1"/>
    <material name="self_default" rgba=".7 .5 .3 1"/>
    <material name="self_highlight" rgba="0 .5 .3 1"/>
    <material name="effector" rgba=".7 .4 .2 1"/>
    <material name="effector_default" rgba=".7 .4 .2 1"/>
    <material name="effector_highlight" rgba="0 .5 .3 1"/>
    <material name="decoration" rgba=".2 .6 .3 1"/>
    <material name="eye" rgba="0 .2 1 1"/>
    <material name="target" rgba=".6 .3 .3 1"/>
    <material name="target_default" rgba=".6 .3 .3 1"/>
    <material name="target_highlight" rgba=".6 .3 .3 .4"/>
    <material name="site" rgba=".5 .5 .5 .3"/>
  </asset>

  <option timestep="0.01">
    <flag contact="disable"/>
  </option>

  <default>
    <default class="pole">
      <joint type="hinge" axis="0 1 0"  damping="2e-6"/>
      <geom type="capsule" fromto="0 0 0 0 0 1" size="0.045" material="self" mass=".1"/>
    </default>
  </default>

  <worldbody>
    <light name="light" pos="0 0 6"/>
    <camera name="fixed" pos="0 -4 1" zaxis="0 -1 0"/>
    <camera name="lookatcart" mode="targetbody" target="cart" pos="0 -2 2"/>
    <geom name="floor" pos="0 0 -.05" size="4 4 .2" type="plane" material="blue_grid"/>
    <geom name="rail1" type="capsule" pos="0  .07 1" zaxis="1 0 0" size="0.02 2" material="decoration" />
    <geom name="rail2" type="capsule" pos="0 -.07 1" zaxis="1 0 0" size="0.02 2" material="decoration" />
    <body name="cart" pos="0 0 1">
      <joint name="slider" type="slide" limited="true" axis="1 0 0" range="-1.8 1.8" solreflimit=".08 1" solimplimit="0 0.95 0.001" damping="1.0e-4"/>
      <geom name="cart" type="box" size="0.2 0.15 0.1" material="self"  mass="1"/>
      <body name="pole_1" childclass="pole">
        <joint name="hinge_1" damping="1.0e-4"/>
        <geom name="pole_1"/>
        <site name="tip" pos="0 0 1"/>
      </body>
    </body>
  </worldbody>

  <actuator>
    <motor name="slide" joint="slider" gear="10" ctrllimited="true" ctrlrange="-1 1" />
  </actuator>

  <sensor>
    <jointpos name="slider" joint="slider"/>
    <jointpos name="hinge_1" joint="hinge_1"/>
  </sensor>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, height=768, width=1366)

# linear interpolation initialization
T = 500
q0 = np.array([0.0, np.pi])
qT = np.array([0.0, 0.0])

# compute linear interpolation
qinterp = np.zeros((model.nq, T))
for t in range(T):
  # slope
  slope = (qT - q0) / T

  # interpolation
  qinterp[:, t] = q0 + t * slope


# time
time = [t * model.opt.timestep for t in range(T)]

# plot position
fig = plt.figure()

# arm position
plt.plot(time, qinterp[0, :], label="q0 interpolation", color="orange")
plt.plot(time, qinterp[1, :], label="q1 interpolation", color="cyan")

plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("Configuration")

# optimizer model
model_optimizer = mujoco.MjModel.from_xml_string(xml)

# direct optimizer
configuration_length = T
optimizer = direct_lib.Direct(
    model=model_optimizer,
    configuration_length=configuration_length,
)

# set data
for t in range(configuration_length):
  # unpack
  qt = np.zeros(model.nq)
  st = np.zeros(model.nsensordata)
  mt = np.zeros(model.nsensor)
  ft = np.zeros(model.nv)
  tt = np.array([t * model.opt.timestep])

  # set initial state
  if t == 0 or t == 1:
    qt = q0
    st = q0
    mt = np.array([1, 1], dtype=int)

  # set goal
  elif t >= configuration_length - 2:
    qt = qT
    st = qT
    mt = np.array([1, 1], dtype=int)

  # initialize qpos
  else:
    qt = qinterp[:, t]
    mt = np.array([0, 0], dtype=int)

  # set data
  data_ = optimizer.data(
      t,
      configuration=qt,
      sensor_measurement=st,
      sensor_mask=mt,
      force_measurement=ft,
      time=tt,
  )

# set std^2
optimizer.noise(process=np.array([1.0e-2, 1.0e-8]), sensor=np.array([1.0, 1.0]))

# set settings
optimizer.settings(
    sensor_flag=True,
    force_flag=True,
    max_smoother_iterations=1000,
    max_search_iterations=1000,
    regularization_initial=1.0e-12,
    gradient_tolerance=1.0e-6,
    search_direction_tolerance=1.0e-6,
    cost_tolerance=1.0e-6,
    first_step_position_sensors=True,
    last_step_position_sensors=True,
    last_step_velocity_sensors=True,
)

# optimize
optimizer.optimize()

# costs
optimizer.print_cost()

# status
optimizer.print_status()

# get optimized trajectories
q_est = np.zeros((model_optimizer.nq, configuration_length))
v_est = np.zeros((model_optimizer.nv, configuration_length))
s_est = np.zeros((model_optimizer.nsensordata, configuration_length))
f_est = np.zeros((model_optimizer.nv, configuration_length))
t_est = np.zeros(configuration_length)
for t in range(configuration_length):
  data_ = optimizer.data(t)
  q_est[:, t] = data_["configuration"]
  v_est[:, t] = data_["velocity"]
  s_est[:, t] = data_["sensor_prediction"]
  f_est[:, t] = data_["force_prediction"]
  t_est[t] = data_["time"]

# plot position
fig = plt.figure()

plt.plot(
    time, qinterp[0, :], label="q0 (interpolation)", ls="--", color="orange"
)
plt.plot(time, qinterp[1, :], label="q1 (interpolation)", ls="--", color="cyan")

plt.plot(
    t_est - model.opt.timestep,
    q_est[0, :],
    label="q0 (optimized)",
    color="orange",
)
plt.plot(
    t_est - model.opt.timestep,
    q_est[1, :],
    label="q1 (optimized)",
    color="cyan",
)

plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("Configuration")

# plot velocity
fig = plt.figure()

# velocity
plt.plot(
    t_est[1:] - model.opt.timestep,
    v_est[0, 1:],
    label="v0 (optimized)",
    color="orange",
)
plt.plot(
    t_est[1:] - model.opt.timestep,
    v_est[1, 1:],
    label="v1 (optimized)",
    color="cyan",
)

# plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("Velocity")

# frames optimized
frames_opt = []

# simulate
for t in range(configuration_length - 1):
  # set configuration
  data.qpos = q_est[:, t]
  data.qvel = v_est[:, t]

  mujoco.mj_forward(model, data)

  # render and save frames
  renderer.update_scene(data)
  pixels = renderer.render()
  frames_opt.append(pixels)

# display video
media.show_video(frames_opt, fps=1.0 / model.opt.timestep, loop=False)

# forces
fig = plt.figure()

plt.plot(t_est[1:-1], f_est[0, 1:-1], color="orange", label="slider")
plt.plot(
    t_est[1:-1], f_est[1, 1:-1], color="cyan", label="hinge (magic force)"
)  # should be ~0

plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("Forces")

# qfrc verification with forward dynamics
mujoco.mj_resetData(model, data)
data.qpos = q_est[:, 1]
data.qvel = v_est[:, 1]

Qpos = np.zeros((model.nq, T))

for t in range(1, T - 1):
  data.qfrc_applied = f_est[:, t]
  mujoco.mj_step(model, data)
  Qpos[:, t] = data.qpos

# plot position
fig = plt.figure()

plt.plot(
    time[1:],
    Qpos[0, 1:],
    label="q0 (forward simulation)",
    ls="--",
    color="orange",
)
plt.plot(
    time[1:],
    Qpos[1, 1:],
    label="q1 (forward simulation)",
    ls="--",
    color="cyan",
)

plt.plot(
    t_est - model.opt.timestep,
    q_est[0, :],
    label="q0 (optimized)",
    color="orange",
)
plt.plot(
    t_est - model.opt.timestep,
    q_est[1, :],
    label="q1 (optimized)",
    color="cyan",
)

plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("Configuration")
