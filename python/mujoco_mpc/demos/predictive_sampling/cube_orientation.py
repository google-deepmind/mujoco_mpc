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

import mediapy as media
import mujoco
import numpy as np
import pathlib

import predictive_sampling
# %%
# path to hand task

model_path = (
    pathlib.Path(__file__).parent.parent.parent
    / "../../build/mjpc/tasks/shadow_reorient/task.xml"
)
# create simulation model + data
model = mujoco.MjModel.from_xml_path(str(model_path))
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model)


# %%
# reward

def reward(model: mujoco.MjModel, data: mujoco.MjData) -> float:
  # cube position - palm position (L22 norm)
  pos_error = (
      data.sensor("cube_position").data - data.sensor("palm_position").data
  )
  p = 0.02
  q = 2.0
  c = np.dot(pos_error, pos_error)
  a = c ** (0.5 * q) + p**q
  s = a ** (1 / q)
  r0 = -(s - p)

  # cube orientation - goal orientation
  goal_orientation = data.sensor("cube_goal_orientation").data
  cube_orientation = data.sensor("cube_orientation").data
  subquat = np.zeros(3)
  mujoco.mju_subQuat(subquat, goal_orientation, cube_orientation)
  r1 = -0.5 * np.dot(subquat, subquat)

  # cube linear velocity
  linvel = data.sensor("cube_linear_velocity").data
  r2 = -0.5 * np.dot(linvel, linvel)

  # actuator
  effort = data.actuator_force
  r3 = -0.5 * np.dot(effort, effort)

  # grasp
  graspdiff = data.qpos[7:] - model.key_qpos[0][7:]
  r4 = -0.5 * np.dot(graspdiff, graspdiff)

  # joint velocity
  jntvel = data.qvel[6:]
  r5 = -0.5 * np.dot(jntvel, jntvel)

  return 20.0 * r0 + 5.0 * r1 + 10.0 * r2 + 0.1 * r3 + 2.5 * r4 + 1.0e-4 * r5


# %%
# planner
horizon = 0.25
splinestep = 0.05
planstep = 0.01
nimprove = 10
nsample = 10
noise_scale = 0.1
interp = "zero"
planner = predictive_sampling.Planner(
    model,
    reward,
    horizon,
    splinestep,
    planstep,
    nsample,
    noise_scale,
    nimprove,
    interp=interp,
)
# %%
# simulate
mujoco.mj_resetData(model, data)
steps = 500

# history
qpos = [data.qpos]
qvel = [data.qvel]
act = [data.act]
ctrl = []
rewards = []

# frames
frames = []
FPS = 1.0 / model.opt.timestep

# verbose
VERBOSE = False

for _ in range(steps):
  ## predictive sampling

  # improve policy
  planner.improve_policy(
      data.qpos, data.qvel, data.act, data.time, data.mocap_pos, data.mocap_quat
  )

  # get action from policy
  data.ctrl = planner.action_from_policy(data.time)
  # data.ctrl = np.random.normal(scale=0.1, size=model.nu)

  # reward
  rewards.append(reward(model, data))

  if VERBOSE:
    print("time  : ", data.time)
    print(" qpos  : ", data.qpos)
    print(" qvel  : ", data.qvel)
    print(" act   : ", data.act)
    print(" action: ", data.ctrl)
    print(" reward: ", rewards[-1])

  # step
  mujoco.mj_step(model, data)

  # history
  qpos.append(data.qpos)
  qvel.append(data.qvel)
  act.append(data.act)
  ctrl.append(ctrl)

  # render and save frames
  renderer.update_scene(data)
  pixels = renderer.render()
  frames.append(pixels)
# %%
media.show_video(frames, fps=FPS)
