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

import os
from absl.testing import absltest
from absl.testing import parameterized
import grpc
import mujoco
from mujoco_mpc import agent as agent_lib
import numpy as np

import pathlib
from mujoco_mpc.proto import agent_pb2


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


class AgentTest(parameterized.TestCase):

  def test_set_task_parameters(self):
    model_path = (
        pathlib.Path(__file__).parent.parent.parent
        / "build/mjpc/tasks/cartpole/task.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))
    with agent_lib.Agent(task_id="Cartpole", model=model) as agent:
      agent.set_task_parameters({"Goal": 13})
      self.assertEqual(agent.get_task_parameters()["Goal"], 13)

  def test_set_subprocess_working_dir(self):
    model_path = (
        pathlib.Path(__file__).parent.parent.parent
        / "build/mjpc/tasks/cartpole/task.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))

    cwd = "$$$INVALID_PATH$$$"
    with self.assertRaises(FileNotFoundError):
      agent_lib.Agent(
          task_id="Cartpole", model=model, subprocess_kwargs={"cwd": cwd}
      )

    cwd = os.getcwd()
    with agent_lib.Agent(
        task_id="Cartpole", model=model, subprocess_kwargs={"cwd": cwd}
    ) as agent:
      agent.set_task_parameters({"Goal": 13})
      self.assertEqual(agent.get_task_parameters()["Goal"], 13)

  def test_step_env_with_planner(self):
    model_path = (
        pathlib.Path(__file__).parent.parent.parent
        / "build/mjpc/tasks/particle/task_timevarying.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    with agent_lib.Agent(task_id="Particle", model=model) as agent:
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

  def test_env_initialized_to_home_keyframe(self):
    model_path = (
        pathlib.Path(__file__).parent.parent.parent
        / "build/mjpc/tasks/quadruped/task_flat.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))

    with agent_lib.Agent(task_id="Quadruped Flat", model=model) as agent:
      state = agent.get_state()
      # Validate that the first three components of the initial qpos are defined
      # by the home keyframe.
      home_qpos = np.array([0, 0, 0.26])
      np.testing.assert_almost_equal(state.qpos[:home_qpos.shape[0]], home_qpos)

  @parameterized.parameters({"nominal": False}, {"nominal": True})
  def test_action_averaging_doesnt_change_state(self, nominal):
    # when calling get_action with action averaging, the Agent needs to roll
    # out physics, but the API should be implemented not to mutate the state
    model_path = (
        pathlib.Path(__file__).parent.parent.parent
        / "build/mjpc/tasks/cartpole/task.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    control_timestep = model.opt.timestep * 5

    with agent_lib.Agent(task_id="Cartpole", model=model) as agent:
      agent.set_task_parameters({"Goal": 13})
      agent.reset()
      environment_reset(model, data)
      agent.set_state(
          time=data.time,
          qpos=data.qpos,
          qvel=data.qvel,
          act=data.act,
          mocap_pos=data.mocap_pos,
          mocap_quat=data.mocap_quat,
          userdata=data.userdata,
      )
      agent.get_action(
          averaging_duration=control_timestep, nominal_action=nominal
      )
      state_after = agent.get_state()
      self.assertEqual(data.time, state_after.time)
      np.testing.assert_allclose(data.qpos, state_after.qpos)
      np.testing.assert_allclose(data.qvel, state_after.qvel)
      np.testing.assert_allclose(data.act, state_after.act)
      np.testing.assert_allclose(data.userdata, state_after.userdata)

  def test_action_averaging_improves_control(self):
    # try controlling the cartpole task at 1/10th frequency with action
    # repeats, and with action averaging.
    # expect action averaging to be a bit better
    model_path = (
        pathlib.Path(__file__).parent.parent.parent
        / "build/mjpc/tasks/cartpole/task.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    repeats = 10
    control_timestep = model.opt.timestep * repeats

    def get_action_simple(agent):
      return agent.get_action()

    def get_action_averaging(agent):
      return agent.get_action(averaging_duration=control_timestep)

    def run_episode(agent, get_action):
      agent.set_task_parameters({"Goal": 13})
      agent.reset()
      environment_reset(model, data)
      num_steps = 10
      total_cost = 0.0
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
        action = get_action(agent)
        for _ in range(repeats):
          environment_step(model, data, action)
          total_cost += agent.get_total_cost()
      return total_cost

    with agent_lib.Agent(task_id="Cartpole", model=model) as agent:
      averaging_cost = run_episode(agent, get_action_averaging)
      repeat_cost = run_episode(agent, get_action_simple)

    self.assertLess(averaging_cost, repeat_cost)
    # averaging actions should be better but not amazingly so.
    self.assertLess(
        np.abs(averaging_cost - repeat_cost) / repeat_cost,
        0.1,
        "Difference between costs is too large.",
    )

  def test_stepping_on_agent_side(self):
    """Test an alternative way of stepping the physics, on the agent side."""
    model_path = (
        pathlib.Path(__file__).parent.parent.parent
        / "build/mjpc/tasks/cartpole/task.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    with agent_lib.Agent(task_id="Cartpole", model=model) as agent:
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
        / "build/mjpc/tasks/cartpole/task.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))

    # by default, planner would produce a non-zero action
    with agent_lib.Agent(task_id="Cartpole", model=model) as agent:
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

  def test_get_cost_weights(self):
    model_path = (
        pathlib.Path(__file__).parent.parent.parent
        / "build/mjpc/tasks/cartpole/task.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))

    # by default, planner would produce a non-zero action
    with agent_lib.Agent(task_id="Cartpole", model=model) as agent:
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
      self.assertEqual(
          agent.get_cost_weights(),
          {"Vertical": 1, "Velocity": 1, "Centered": 1, "Control": 1},
      )
      agent.set_state(qpos=[0, 0.5], qvel=[1, 1])
      terms_dict = agent.get_cost_term_values()
      terms = list(terms_dict.values())
      self.assertFalse(np.any(np.isclose(terms, 0, rtol=0, atol=1e-4)))

      residuals_dict = agent.get_residuals()
      residuals = list(residuals_dict.values())
      self.assertFalse(np.any(np.isclose(residuals, 0, rtol=0, atol=1e-4)))

  def test_set_state_with_lists(self):
    model_path = (
        pathlib.Path(__file__).parent.parent.parent
        / "build/mjpc/tasks/particle/task_timevarying.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    with agent_lib.Agent(task_id="Particle", model=model) as agent:
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

  def test_get_set_default_mode(self):
    model_path = (
        pathlib.Path(__file__).parent.parent.parent
        / "build/mjpc/tasks/cartpole/task.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))
    with agent_lib.Agent(task_id="Cartpole", model=model) as agent:
      agent.set_mode("default_mode")
      self.assertEqual(agent.get_mode(), "default_mode")

  @absltest.skip("asset import issue")
  def test_get_set_mode(self):
    model_path = (
        pathlib.Path(__file__).parent.parent.parent
        / "build/mjpc/tasks/quadruped/task_flat.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))
    with agent_lib.Agent(task_id="Quadruped Flat", model=model) as agent:
      agent.set_mode("Walk")
      self.assertEqual(agent.get_mode(), "Walk")

  @absltest.skip("asset import issue")
  def test_get_all_modes(self):
    model_path = (
        pathlib.Path(__file__).parent.parent.parent
        / "build/mjpc/tasks/quadruped/task_flat.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))
    with agent_lib.Agent(task_id="Quadruped Flat", model=model) as agent:
      self.assertEqual(
          tuple(agent.get_all_modes()),
          ("Quadruped", "Biped", "Walk", "Scramble", "Flip"),
      )

  @absltest.skip("asset import issue")
  def test_set_mode_error(self):
    model_path = (
        pathlib.Path(__file__).parent.parent.parent
        / "build/mjpc/tasks/quadruped/task_flat.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))
    with agent_lib.Agent(task_id="Quadruped Flat", model=model) as agent:
      self.assertRaises(grpc.RpcError, lambda: agent.set_mode("Run"))

  def test_set_task_parameters_from_another_agent(self):
    model_path = (
        pathlib.Path(__file__).parent.parent.parent
        / "build/mjpc/tasks/cartpole/task.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))
    with agent_lib.Agent(task_id="Cartpole", model=model) as agent:
      agent.set_task_parameters({"Goal": 13})
      self.assertEqual(agent.get_task_parameters()["Goal"], 13)

      with agent_lib.Agent(
          task_id="Cartpole",
          run_init=False,
          connect_to=f"localhost:{agent.port}",
      ) as serverless_agent:
        serverless_agent.set_task_parameters({"Goal": 14})

      self.assertEqual(agent.get_task_parameters()["Goal"], 14)

  def test_best_trajectory(self):
    model_path = (
        pathlib.Path(__file__).parent.parent.parent
        / "build/mjpc/tasks/particle/task_timevarying.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    agent = agent_lib.Agent(task_id="Particle", model=model)
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
    best_traj = agent.best_trajectory()

    self.assertEqual(best_traj["states"].shape, (51, 4))
    self.assertEqual(best_traj["actions"].shape, (50, 2))
    self.assertEqual(best_traj["times"].shape, (51,))

  def test_set_mocap(self):
    model_path = (
        pathlib.Path(__file__).parent.parent.parent
        / "build/mjpc/tasks/particle/task_timevarying.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))
    with agent_lib.Agent(task_id="ParticleFixed", model=model) as agent:
      pose = agent_pb2.Pose(pos=[13, 14, 15], quat=[1, 1, 1, 1])
      agent.set_mocap({"goal": pose})
      final_state = agent.get_state()
      self.assertEqual(final_state.mocap_pos, pose.pos)
      self.assertEqual(
          final_state.mocap_quat,
          [0.5, 0.5, 0.5, 0.5],
          "quaternions should be normalized",
      )


if __name__ == "__main__":
  absltest.main()
