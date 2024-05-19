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

import jax
import matplotlib.pyplot as plt
import mediapy
import mujoco
from mujoco_mpc.mjx import predictive_sampling
from mujoco_mpc.mjx.tasks.bimanual import handover
import numpy as np
# %%
costs_to_compare = {}
for it in [0.3, 0.5, 0.8]:
  nsteps = 300
  steps_per_plan = 10
  frame_skip = 5  # how many steps between each rendered frame
  batch_size = 8192
  nsamples = 512
  nplans = batch_size // nsamples
  print(f'nplans: {nplans}')

  sim_model_cpu, plan_model_cpu, cost_fn = handover.get_models_and_cost_fn()
  p = predictive_sampling.Planner(
      model=plan_model_cpu,  # dummy
      cost=cost_fn,
      noise_scale=it,
      horizon=128,
      nspline=4,
      nsample=nsamples - 1,
      interp='zero',
  )

  policy = np.tile(sim_model_cpu.key_ctrl[0, :], (nplans, p.nspline, 1))
  trajectories, costs, _ = jax.jit(
      predictive_sampling.receding_horizon_control
  )(
      p,
      jax.device_put(policy),
      jax.random.key(0),
      plan_model_cpu,
      sim_model_cpu,
      nsteps,
      nplans,
      steps_per_plan,
  )

  plt.figure()
  plt.xlim([0, nsteps * sim_model_cpu.opt.timestep])
  plt.ylim([0, max(costs.flatten())])
  plt.xlabel('time')
  plt.ylabel('cost')
  x_time = [i * sim_model_cpu.opt.timestep for i in range(nsteps)]
  for i in range(nplans):
    plt.plot(x_time, costs[i, :], alpha=0.1)
  avg = np.mean(costs, axis=0)
  plt.plot(x_time, avg, linewidth=2.0)
  var = np.var(costs, axis=0)
  plt.errorbar(
      x_time,
      avg,
      yerr=var,
      fmt='none',
      ecolor='b',
      elinewidth=1,
      alpha=0.2,
      capsize=0,
  )

  plt.show()
  costs_to_compare[it] = costs
# %%
plt.figure()
plt.xlim([0, nsteps * sim_model_cpu.opt.timestep])
plt.ylim([0, max(costs.flatten())])
plt.xlabel('time')
plt.ylabel('cost')
x_time = [i * sim_model_cpu.opt.timestep for i in range(nsteps)]
for val, costs in costs_to_compare.items():
  avg = np.mean(costs, axis=0)
  plt.plot(x_time, avg, label=str(val))
  var = np.var(costs, axis=0)
  plt.errorbar(
      x_time, avg, yerr=var, fmt='none', elinewidth=1, alpha=0.2, capsize=0
  )

plt.legend()
plt.show()
# %%
frames = []
renderer = mujoco.Renderer(sim_model_cpu)
d = mujoco.MjData(sim_model_cpu)
trajectory = trajectories[0, ...]
for qpos in trajectory:
  d.qpos = qpos
  mujoco.mj_forward(sim_model_cpu, d)
  renderer.update_scene(d)
  frames.append(renderer.render())
# %%
mediapy.show_video(frames, fps=1/sim_model_cpu.opt.timestep/frame_skip)
