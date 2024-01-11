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

import mujoco
import numpy as np
import matplotlib.pyplot as plt
import mediapy as media

# set current directory to mjpc/python/mujoco_mpc
from mujoco_mpc import direct as direct_lib
# %%
# 2D Particle Model
xml = """
<mujoco model="Particle">
  <visual>
    <headlight ambient=".4 .4 .4" diffuse=".8 .8 .8" specular="0.1 0.1 0.1"/>
    <map znear=".01"/>
    <quality shadowsize="2048"/>
    <global elevation="-15"/>
  </visual>

  <asset>
    <texture name="blue_grid" type="2d" builtin="checker" rgb1=".02 .14 .44" rgb2=".27 .55 1" width="300" height="300" mark="edge" markrgb="1 1 1"/>
    <material name="blue_grid" texture="blue_grid" texrepeat="1 1" texuniform="true" reflectance=".2"/>
    <texture name="skybox" type="skybox" builtin="gradient" rgb1=".66 .79 1" rgb2=".9 .91 .93" width="800" height="800"/>
    <material name="self" rgba=".7 .5 .3 1"/>
    <material name="decoration" rgba=".2 .6 .3 1"/>
  </asset>

  <option timestep="0.01"></option>

  <default>
    <joint type="hinge" axis="0 0 1" limited="true" range="-.29 .29" damping="1"/>
    <motor gear=".1" ctrlrange="-1 1" ctrllimited="true"/>
  </default>

  <worldbody>
    <light name="light" pos="0 0 1"/>
    <camera name="fixed" pos="0 0 .75" quat="1 0 0 0"/>
    <geom name="ground" type="plane" pos="0 0 0" size=".3 .3 .1" material="blue_grid"/>
    <geom name="wall_x" type="plane" pos="-.3 0 .02" zaxis="1 0 0"  size=".02 .3 .02" material="decoration"/>
    <geom name="wall_y" type="plane" pos="0 -.3 .02" zaxis="0 1 0"  size=".3 .02 .02" material="decoration"/>
    <geom name="wall_neg_x" type="plane" pos=".3 0 .02" zaxis="-1 0 0"  size=".02 .3 .02" material="decoration"/>
    <geom name="wall_neg_y" type="plane" pos="0 .3 .02" zaxis="0 -1 0"  size=".3 .02 .02" material="decoration"/>

    <body name="pointmass" pos="0 0 .01">
      <camera name="cam0" pos="0 -0.3 0.3" xyaxes="1 0 0 0 0.7 0.7"/>
      <joint name="root_x" type="slide"  pos="0 0 0" axis="1 0 0" />
      <joint name="root_y" type="slide"  pos="0 0 0" axis="0 1 0" />
      <geom name="pointmass" type="sphere" size=".01" material="self" mass=".3"/>
      <site name="tip" pos="0 0 0" size="0.01"/>
    </body>
  </worldbody>

  <actuator>
    <motor name="x_motor" joint="root_x" gear="1" ctrllimited="true" ctrlrange="-1 1"/>
    <motor name="y_motor" joint="root_y" gear="1" ctrllimited="true" ctrlrange="-1 1"/>
  </actuator>

  <sensor>
    <jointpos name="x" joint="root_x" />
    <jointpos name="y" joint="root_y" />
  </sensor>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model)
# %%
# initialization
T = 100
q0 = np.array([-0.25, -0.25])
qM = np.array([-0.25, 0.25])
qN = np.array([0.25, -0.25])
qT = np.array([0.25, 0.25])

# compute linear interpolation
qinterp = np.zeros((model.nq, T))
for t in range(T):
  # slope
  slope = (qT - q0) / T

  # interpolation
  qinterp[:, t] = q0 + t * slope

# time
time = [t * model.opt.timestep for t in range(T)]
# %%
# plot position
fig = plt.figure()

# arm position
plt.plot(qinterp[0, :], qinterp[1, :], label="interpolation", color="black")
plt.plot(q0[0], q0[1], color="magenta", label="waypoint", marker="o")
plt.plot(qM[0], qM[1], color="magenta", marker="o")
plt.plot(qN[0], qN[1], color="magenta", marker="o")
plt.plot(qT[0], qT[1], color="magenta", marker="o")

plt.legend()
plt.xlabel("X")
plt.ylabel("Y")
# %%
# optimizer model
model_optimizer = mujoco.MjModel.from_xml_string(xml)

# direct optimizer
configuration_length = T + 2
optimizer = direct_lib.Direct(
    model=model_optimizer,
    configuration_length=configuration_length,
)
# %%
# set data
for t in range(configuration_length):
  # unpack
  qt = np.zeros(model.nq)
  st = np.zeros(model.nsensordata)
  mt = np.zeros(model.nsensor)
  ft = np.zeros(model.nv)
  ct = np.zeros(model.nu)
  tt = np.array([t * model.opt.timestep])

  # set initial state
  if t == 0 or t == 1:
    qt = q0
    st = q0
    mt = np.array([1, 1])

  # set goal
  elif t >= configuration_length - 2:
    qt = qT
    st = qT
    mt = np.array([1, 1])

  # set waypoint
  elif t == 25:
    st = qM
    mt = np.array([1, 1])

  # set waypoint
  elif t == 75:
    st = qN
    mt = np.array([1, 1])

  # initialize qpos
  else:
    qt = qinterp[:, t - 1]
    mt = np.array([0, 0])

  # set data
  data_ = optimizer.data(
      t,
      configuration=qt,
      ctrl=ct,
      sensor_measurement=st,
      sensor_mask=mt,
      force_measurement=ft,
      time=tt,
  )
# %%
# set std
optimizer.noise(process=np.array([1000.0, 1000.0]), sensor=np.array([1.0, 1.0]))

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
# %%
# get estimated trajectories
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
# %%
# plot position
fig = plt.figure()

plt.plot(qinterp[0, :], qinterp[1, :], label="interpolation", color="black")
plt.plot(q_est[0, :], q_est[1, :], label="direct trajopt", color="orange")
plt.plot(q0[0], q0[1], color="magenta", label="waypoint", marker="o")
plt.plot(qM[0], qM[1], color="magenta", marker="o")
plt.plot(qN[0], qN[1], color="magenta", marker="o")
plt.plot(qT[0], qT[1], color="magenta", marker="o")

plt.legend()
plt.xlabel("X")
plt.ylabel("Y")

# plot velocity
fig = plt.figure()

# velocity
plt.plot(t_est[1:] - model.opt.timestep, v_est[0, 1:], label="v0", color="cyan")
plt.plot(
    t_est[1:] - model.opt.timestep, v_est[1, 1:], label="v1", color="orange"
)

plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("Velocity")
# %%
# frames optimized
frames_opt = []

# simulate
for t in range(configuration_length - 1):
  # get solution from optimizer
  data_ = optimizer.data(t)

  # set configuration
  data.qpos = q_est[:, t]
  data.qvel = v_est[:, t]

  mujoco.mj_forward(model, data)

  # render and save frames
  renderer.update_scene(data)
  pixels = renderer.render()
  frames_opt.append(pixels)

# display video
# media.show_video(frames_opt, fps=1.0 / model.opt.timestep, loop=False)
