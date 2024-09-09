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
from mujoco import mjx
from mujoco_mpc.mjx import predictive_sampling
from mujoco_mpc.mjx.tasks.bimanual import handover
import numpy as np

# %%
sim_model_cpu, plan_model_cpu, cost_fn = handover.get_models_and_cost_fn()
# %%
p = predictive_sampling.Planner(
    model=mjx.put_model(plan_model_cpu),
    cost=cost_fn,
    noise_scale=0.5,
    horizon=128,
    nspline=4,
    nsample=8192 - 1,
    interp='zero',
)

sim_data = mujoco.MjData(sim_model_cpu)
mujoco.mj_resetDataKeyframe(sim_model_cpu, sim_data, 0)
mujoco.mj_forward(sim_model_cpu, sim_data)
sim_data = mjx.put_data(sim_model_cpu, sim_data)
policy = np.tile(sim_model_cpu.key_ctrl[0, :], (p.nspline, 1))

_, _, costs, traj = jax.jit(
    predictive_sampling.mpc_rollout, static_argnums=[0, 1]
)(
    500,
    10,
    p,
    jax.device_put(policy),
    jax.random.key(0),
    mjx.put_model(sim_model_cpu),
    sim_data,
)
plt.figure()
plt.xlim([0, costs.size * sim_model_cpu.opt.timestep])
plt.ylim([0, 1])
plt.xlabel('time')
plt.ylabel('cost')
x_time = [i * sim_model_cpu.opt.timestep for i in range(costs.size)]
plt.plot(x_time, costs)

plt.legend()
plt.show()
# %%
frame_skip = 5
frames = []
renderer = mujoco.Renderer(sim_model_cpu)
d = mujoco.MjData(sim_model_cpu)
qs = trajectories.q[0, ...].reshape(-1, sim_model_cpu.nq)[0:-1:frame_skip, :]
for qpos in qs:
  d.qpos = qpos
  mujoco.mj_forward(sim_model_cpu, d)
  renderer.update_scene(d)
  frames.append(renderer.render())
  mediapy.show_video(frames, fps=1/sim_model_cpu.opt.timestep/frame_skip)
