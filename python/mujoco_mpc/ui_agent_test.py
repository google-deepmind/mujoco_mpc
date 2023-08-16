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

"""A test for agent.py that brings up a UI. Can only be run locally."""

import time

from absl.testing import absltest
import grpc
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


class UiAgentTest(absltest.TestCase):

  def test_stepping_on_agent_side(self):
    """Test an alternative way of stepping the physics, on the agent side."""
    model_path = (
        pathlib.Path(__file__).parent.parent.parent / "mjpc/tasks/cartpole/task.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    with self.get_agent(task_id="Cartpole", model=model) as agent:
      agent.set_task_parameter("Goal", -1.0)

      num_steps = 10
      observations = []
      for _ in range(num_steps):
        agent.planner_step()
        time.sleep(0.1)
        agent.step()
        state = agent.get_state()
        data.time = state.time
        data.qpos = state.qpos
        data.qvel = state.qvel
        data.act = state.act
        data.mocap_pos = np.array(state.mocap_pos).reshape(data.mocap_pos.shape)
        data.mocap_quat = np.array(state.mocap_quat).reshape(data.mocap_quat.shape)
        data.userdata = np.array(state.userdata).reshape(data.userdata.shape)
        observations.append(get_observation(model, data))

      self.assertNotEqual(agent.get_state().time, 0)

    observations = np.array(observations)
    self.assertFalse((observations == 0).all())

  def test_set_cost_weights(self):
    model_path = (
        pathlib.Path(__file__).parent.parent.parent / "mjpc/tasks/cartpole/task.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))

    # by default, planner would produce a non-zero action
    with self.get_agent(task_id="Cartpole", model=model) as agent:
      agent.set_task_parameter("Goal", -1.0)
      agent.planner_step()
      # wait so a planning cycle definitely finishes
      # TODO(nimrod): make sure planner_step waits for a planning step
      time.sleep(0.5)
      action = agent.get_action()
      self.assertFalse(np.allclose(action, 0))

      # setting all costs to 0 apart from control should end up with a zero
      # action
      agent.reset()
      agent.set_task_parameter("Goal", -1.0)
      agent.set_cost_weights(
          {"Vertical": 0, "Velocity": 0, "Centered": 0, "Control": 1}
      )
      # wait so a planning cycle definitely finishes
      # TODO(nimrod): make sure planner_step waits for a planning step
      time.sleep(0.5)
      action = agent.get_action()
    np.testing.assert_allclose(0, action, rtol=1, atol=1e-7)

  def test_get_cost_weights(self):
    model_path = (
        pathlib.Path(__file__).parent.parent.parent / "mjpc/tasks/cartpole/task.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))

    # by default, planner would produce a non-zero action
    with self.get_agent(task_id="Cartpole", model=model) as agent:
      agent.set_task_parameter("Goal", -1.0)
      agent.planner_step()
      cost = agent.get_total_cost()
      self.assertNotEqual(cost, 0)

      agent.reset()
      agent.set_task_parameter("Goal", -1.0)
      agent.set_cost_weights(
          {"Vertical": 1, "Velocity": 0, "Centered": 1, "Control": 0}
      )
      for _ in range(10):
        agent.planner_step()
        agent.step()
      agent.set_task_parameter("Goal", 1.0)
      agent.set_cost_weights(
          {"Vertical": 1, "Velocity": 1, "Centered": 1, "Control": 1}
      )
      agent.set_state(qpos=[0, 0.5], qvel=[1, 1])
      terms_dict = agent.get_cost_term_values()
      terms = list(terms_dict.values())
      self.assertFalse(np.any(np.isclose(terms, 0, rtol=0, atol=1e-6)))

  def test_set_state_with_lists(self):
    model_path = (
        pathlib.Path(__file__).parent.parent.parent
        / "mjpc/tasks/particle/task_timevarying.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    with self.get_agent(task_id="Particle", model=model) as agent:
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

  def test_set_get_mode(self):
    model_path = (
        pathlib.Path(__file__).parent.parent.parent / "mjpc/tasks/cartpole/task.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))
    with self.get_agent(task_id="Cartpole", model=model) as agent:
      agent.set_mode("default_mode")
      self.assertEqual(agent.get_mode(), "default_mode")

  @absltest.skip("asset import issue")
  def test_get_set_mode(self):
    model_path = (
        pathlib.Path(__file__).parent.parent.parent
        / "mjpc/tasks/quadruped/task_flat.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))
    with self.get_agent(task_id="Quadruped Flat", model=model) as agent:
      agent.set_mode("Walk")
      self.assertEqual(agent.get_mode(), "Walk")

  @absltest.skip("asset import issue")
  def test_set_mode_error(self):
    model_path = (
        pathlib.Path(__file__).parent.parent.parent
        / "mjpc/tasks/quadruped/task_flat.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))
    with self.get_agent(task_id="Quadruped Flat", model=model) as agent:
      self.assertRaises(grpc.RpcError, lambda: agent.set_mode("Run"))

  def get_agent(self, **kwargs) -> agent_lib.Agent:
    return agent_lib.Agent(
        server_binary_path=pathlib.Path(agent_lib.__file__).parent
        / "mjpc"
        / "ui_agent_server",
        **kwargs
    )


if __name__ == "__main__":
  absltest.main()
