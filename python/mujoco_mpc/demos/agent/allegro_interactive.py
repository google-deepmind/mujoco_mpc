import mujoco
import mujoco.viewer
import numpy as np
import time

from mujoco_mpc import agent as agent_lib

"""
Run an interactive simulation of the Allegro hand with a cube.

This script is intended to be run from the root of the mujoco_mpc repository.
"""

# Set up the model
model_path = "mjpc/tasks/allegro_cube/task.xml"
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# Set the number of rollouts for predictive sampling
model.numeric("sampling_trajectories").data[0] = 10

print(model.opt.timestep)

# Create the planning agent
agent = agent_lib.Agent(task_id="AllegroCube", model=model)

# Launch the simulation
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()

        # Compute the action from the agent
        agent.set_state(
            time=data.time,
            qpos=data.qpos,
            qvel=data.qvel,
            act=data.act,
            mocap_pos=data.mocap_pos,
            mocap_quat=data.mocap_quat,
            userdata=data.userdata,
        )
        agent.planner_step()
        data.ctrl = agent.get_action()

        # Step the simulation
        mujoco.mj_step(model, data)

        # Sync data from the viewer
        viewer.sync()

        # Try to run in roughly realtime
        elapsed = time.time() - step_start
        if elapsed < model.opt.timestep:
            time.sleep(model.opt.timestep - elapsed)
