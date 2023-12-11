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


def display_current_state(model, data):
    """Display the current state of the simulation with matplotlib."""
    renderer = mujoco.Renderer(model)
    renderer.update_scene(data)
    pixels = renderer.render()
    plt.imshow(pixels)
    plt.show()


def get_iteration_time(num_rollouts, timeit_iterations=1000, display=False):
    """Get the average time per iteration of the planner.

    Args:
        num_rollouts: Number of rollouts to use for predictive sampling.
        timeit_iterations: Number of iterations to run timeit for.
        display: Whether to show the simulated state with matplotlib.
    """
    # Set up model
    # Assumes this is run from mujoco_mpc/python/mujoco_mpc
    model_path = (
            pathlib.Path(os.path.abspath("")).parent.parent.parent
            / "mujoco_mpc/mjpc/tasks/allegro_cube/task.xml"
            )
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    # Set number of rollouts for predictive sampling
    model.numeric("sampling_trajectories").data[0] = num_rollouts

    # Set an initial position with non-trivial contact
    q0 = np.array([1.,  0.,  0.,  0.,  0.26926841,
                -0.00202754,  0.02846826,  0.70730151,  0.68283427,  0.12776131,
                -0.13091594, -0.17055464,  0.46287199,  0.75434123,  0.67566514,
                0.01532882,  0.06954104,  1.00649882,  0.51173668,  0.07650671,
                1.07602678,  0.67203893,  0.4372858,  0.94586334,  0.50472875,
                0.21540332,  0.63196814])
    data.qpos[:] = q0
    mujoco.mj_step(model, data)

    # Display the initial state
    if display:
        display_current_state(model, data)

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
    
    # Profile the planner
    def test_fn():
        """Little test function for timeit."""
        agent.planner_step()
        u = agent.get_action()
        return u
    total_time = timeit.timeit(test_fn, number=timeit_iterations)

    return total_time / timeit_iterations

if __name__=="__main__":
    # Get iteration times for different numbers of rollouts
    num_rollouts = [5*i for i in range(1,12)]
    iteration_times = []

    for num_rollout in num_rollouts:
        print(f"Number of rollouts: {num_rollout}")
        iter_time = get_iteration_time(num_rollout, display=False)

        print(f"Time per iteration: {iter_time}s")
        iteration_times.append(iter_time)
        print("")

    # Plot the results
    plt.plot(num_rollouts, iteration_times, 'o')
    plt.xlabel("Number of rollouts")
    plt.ylabel("Time per iteration (s)")

    plt.show()