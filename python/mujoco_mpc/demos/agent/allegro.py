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
import mujoco
import numpy as np
import os
import pathlib
import timeit

from mujoco_mpc import agent as agent_lib

# Set up model
# Assumes this is run from mujoco_mpc/python/mujoco_mpc
model_path = (
        pathlib.Path(os.path.abspath("")).parent.parent.parent
        / "mujoco_mpc/mjpc/tasks/allegro_cube/task.xml"
        )
model = mujoco.MjModel.from_xml_path(str(model_path))
data = mujoco.MjData(model)

# Set number of rollouts for predictive sampling
num_samples = 100
model.numeric("sampling_trajectories").data[0] = num_samples

# Set an initial position with non-trivial contact
q0 = np.array([1.,  0.,  0.,  0.,  0.26926841,
               -0.00202754,  0.02846826,  0.70730151,  0.68283427,  0.12776131,
               -0.13091594, -0.17055464,  0.46287199,  0.75434123,  0.67566514,
               0.01532882,  0.06954104,  1.00649882,  0.51173668,  0.07650671,
               1.07602678,  0.67203893,  0.4372858,  0.94586334,  0.50472875,
               0.21540332,  0.63196814])
data.qpos[:] = q0
mujoco.mj_step(model, data)

# Create the planning agent
agent = agent_lib.Agent(task_id="AllegroCube", model=model)
    
# set the planner state
agent.set_state(
        time=data.time,
        qpos=data.qpos,
        qvel=data.qvel,
        act=data.act,
        mocap_pos=data.mocap_pos,
        mocap_quat=data.mocap_quat,
        userdata=data.userdata,
        )

# Visualize this initial position
print("Previewing initial position")
renderer = mujoco.Renderer(model)
renderer.update_scene(data)
pixels = renderer.render()
plt.imshow(pixels)
plt.show()

# Profile the planner
print("Running profiler")
def test_fn():
    """Little test function for timeit."""
    agent.planner_step()
    u = agent.get_action()
    return u

N = 1000
total_time = timeit.timeit(test_fn, number=N)
print(f"Average planning time: {total_time / N} seconds")
