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

import contextlib

from absl.testing import absltest
import mujoco
from mujoco_mpc import agent as agent_lib
import numpy as np

import pathlib


def get_observation(model, data):
  del model
  return np.concatenate([data.qpos, data.qvel])


def environment_step(model, data, action):
  data.ctrl[:] = action
  mujoco.mj_step(model, data)
  return get_observation(model, data)


def environment_reset(model, data):
  mujoco.mj_resetData(model, data)
  return get_observation(model, data)


class AgentTest(absltest.TestCase):

  def test_step_env_with_planner(self):
    model_path = (
        pathlib.Path(__file__).parent.parent.parent
        / "mjpc"
        / "tasks"
        / "particle"
        / "task_timevarying.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    agent = agent_lib.Agent(task_id="Particle", model=model)

    with contextlib.closing(agent):
      actions = []
      observations = [environment_reset(model, data)]

      num_steps = 10
      for _ in range(num_steps):
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
        actions.append(agent.get_action())
        observations.append(environment_step(model, data, actions[-1]))

    observations = np.array(observations)
    actions = np.array(actions)

    self.assertFalse((observations == 0).all())
    self.assertFalse((actions == 0).all())

  def test_stepping_on_agent_side(self):
    """Test an alternative way of stepping the physics, on the agent side."""
    model_path = (
        pathlib.Path(__file__).parent.parent.parent
        / "mjpc"
        / "tasks"
        / "cartpole"
        / "task.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    agent = agent_lib.Agent(task_id="Cartpole", model=model)
    with contextlib.closing(agent):
      agent.set_task_parameter("Goal", -1.0)

      num_steps = 10
      observations = []
      for _ in range(num_steps):
        agent.planner_step()
        agent.step()
        state = agent.get_state()
        data.time = state.time
        data.qpos = state.qpos
        data.qvel = state.qvel
        data.act = state.act
        data.mocap_pos = np.array(state.mocap_pos).reshape(data.mocap_pos.shape)
        data.mocap_quat = np.array(state.mocap_quat).reshape(
            data.mocap_quat.shape
        )
        data.userdata = np.array(state.userdata).reshape(data.userdata.shape)
        observations.append(get_observation(model, data))

      self.assertNotEqual(agent.get_state().time, 0)

    observations = np.array(observations)
    self.assertFalse((observations == 0).all())

  def test_set_cost_weights(self):
    model_path = (
        pathlib.Path(__file__).parent.parent.parent
        / "mjpc"
        / "tasks"
        / "cartpole"
        / "task.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))
    agent = agent_lib.Agent(task_id="Cartpole", model=model)

    # by default, planner would produce a non-zero action
    with contextlib.closing(agent):
      agent.set_task_parameter("Goal", -1.0)
      agent.planner_step()
      action = agent.get_action()
      self.assertFalse(np.allclose(action, 0))

      # setting all costs to 0 apart from control should end up with a zero
      # action
      agent.reset()
      agent.set_task_parameter("Goal", -1.0)
      agent.set_cost_weights(
          {"Vertical": 0, "Velocity": 0, "Centered": 0, "Control": 1}
      )
      agent.planner_step()
      action = agent.get_action()
    np.testing.assert_allclose(action, 0)

  def test_set_state_with_lists(self):
    model_path = (
        pathlib.Path(__file__).parent.parent.parent
        / "mjpc"
        / "tasks"
        / "particle"
        / "task_timevarying.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    agent = agent_lib.Agent(task_id="Particle", model=model)

    with contextlib.closing(agent):
      actions = []
      observations = [environment_reset(model, data)]

      agent.set_state(
          time=data.time,
          qpos=list(data.qpos),
          qvel=list(data.qvel),
          act=list(data.act),
          mocap_pos=list(data.mocap_pos.flatten()),
          mocap_quat=list(data.mocap_quat.flatten()),
          userdata=list(data.userdata),
      )
      agent.planner_step()


if __name__ == "__main__":
  absltest.main()
