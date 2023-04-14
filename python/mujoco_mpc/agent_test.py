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

  def test_step_with_planner(self):
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

    actions = []
    observations = [environment_reset(model, data)]

    agent.set_task_parameter("Goal", -1.0)

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


if __name__ == "__main__":
  absltest.main()
