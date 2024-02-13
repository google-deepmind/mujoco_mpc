# %%
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
import mujoco
import direct_optimizer
import numpy as np
# %%
## Example

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

# %%
# initialization
T = 400
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
# create optimizer
optimizer = direct_optimizer.DirectOptimizer(model, T)

# settings
optimizer.max_iterations = 10
optimizer.max_search_iterations = 10

# force weight
fw = 5.0e2

# set data
for t in range(T):
  # set initial state
  if t == 0 or t == 1:
    optimizer.pinned[t] = True
    optimizer.qpos[:, t] = q0
    if t == 1:
      optimizer.weights_force[:, t] = fw

  # set goal
  elif t >= T - 2:
    optimizer.pinned[t] = True
    optimizer.qpos[:, t] = qT
    if t == T - 2:
      optimizer.weights_force[:, t] = fw

  # set waypoint
  elif t == 100:
    optimizer.pinned[t] = False
    optimizer.qpos[:, t] = qM
    optimizer.sensor_target[: model.nq, t] = qM
    optimizer.weights_force[:, t] = fw
    optimizer.weights_sensor[:, t] = 1.0

  # set waypoint
  elif t == 300:
    optimizer.pinned[t] = False
    optimizer.qpos[:, t] = qN
    optimizer.sensor_target[: model.nq, t] = qN
    optimizer.weights_force[:, t] = fw
    optimizer.weights_sensor[:, t] = 1.0

  # initialize qpos
  else:
    optimizer.pinned[t] = False
    optimizer.qpos[:, t] = qinterp[:, t]
    optimizer.weights_force[:, t] = fw
    optimizer.weights_sensor[:, t] = 0.0

# optimize
optimizer.optimize()

# status
optimizer.status()

# %%
# plot position
fig = plt.figure()

plt.plot(qinterp[0, :], qinterp[1, :], label="interpolation", color="black")
plt.plot(
    optimizer.qpos[0, :],
    optimizer.qpos[1, :],
    label="direct trajopt",
    color="orange",
)
plt.plot(q0[0], q0[1], color="magenta", label="waypoint", marker="o")
plt.plot(qM[0], qM[1], color="magenta", marker="o")
plt.plot(qN[0], qN[1], color="magenta", marker="o")
plt.plot(qT[0], qT[1], color="magenta", marker="o")

plt.legend()
plt.xlabel("X")
plt.ylabel("Y")

# %%
# recover ctrl
mujoco.mj_forward(model, data)
ctrl = np.vstack(
    [
        np.linalg.pinv(data.actuator_moment.T) @ optimizer.force[:, t]
        for t in range(T)
    ]
)

# plot ctrl
fig = plt.figure()
times = [t * model.opt.timestep for t in range(T)]

plt.step(times, ctrl[:, 0], label="action 0", color="orange")
plt.step(times, ctrl[:, 1], label="action 1", color="magenta")
plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("ctrl")
