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

import matplotlib.pyplot as plt
import mediapy
import mujoco
from mujoco_mpc.mjx import predictive_sampling
from mujoco_mpc.mjx.tasks.bimanual import handover
# %%
nsteps = 500
steps_per_plan = 4
frame_skip = 5  # how many steps between each rendered frame


sim_model, plan_model, cost_fn = handover.get_models_and_cost_fn()
p = predictive_sampling.Planner(
    model=plan_model,
    cost=cost_fn,
    noise_scale=0.3,
    horizon=128,
    nspline=4,
    nsample=128 - 1,
    interp='zero',
)

trajectory, costs, plan_time = (
    predictive_sampling.receding_horizon_optimization(
        p,
        plan_model,
        sim_model,
        nsteps,
        steps_per_plan,
        frame_skip,
    )
)
# %%
plt.xlim([0, nsteps * sim_model.opt.timestep])
plt.ylim([0, max(costs)])
plt.xlabel('time')
plt.ylabel('cost')
plt.plot([i * sim_model.opt.timestep for i in range(nsteps)], costs)
plt.show()

sim_time = nsteps * sim_model.opt.timestep
plan_steps = nsteps // steps_per_plan
real_factor = sim_time / plan_time
print(f'Total wall time ({plan_steps} planning steps): {plan_time} s'
      f' ({real_factor:.2f}x realtime)')
# %%
frames = []
renderer = mujoco.Renderer(sim_model)
d = mujoco.MjData(sim_model)

for qpos in trajectory:
  d.qpos = qpos
  mujoco.mj_forward(sim_model, d)
  renderer.update_scene(d)
  frames.append(renderer.render())
# %%
mediapy.show_video(frames, fps=1/sim_model.opt.timestep/frame_skip)
# %%
