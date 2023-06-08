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
import grpc
import mujoco
from mujoco_mpc import agent as agent_lib
import numpy as np

import pathlib


class EstimatorTest(absltest.TestCase):

  def test_initialization(self):
    model_path = (
        pathlib.Path(__file__).parent.parent.parent
        / "mjpc/test/testdata/estimator/particle/task.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    configuration_length = 5
    estimator = agent_lib.Estimator(model, configuration_length)

    # self.assertFalse((observations == 0).all())

if __name__ == "__main__":
  absltest.main()
