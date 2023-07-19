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
from mujoco_mpc import kalman as kalman_lib
import numpy as np

import pathlib


class KALMANTest(absltest.TestCase):

  def test_settings(self):
    # load model
    model_path = (
        pathlib.Path(__file__).parent.parent.parent
        / "mjpc/test/testdata/estimator/particle/task1D.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))

    # initialize
    kalman = kalman_lib.KALMAN(model=model)

    # settings
    epsilon = 2.0
    flg_centered = True
    auto_timestep = True
    settings = kalman.settings(
        epsilon=epsilon, flg_centered=flg_centered, auto_timestep=auto_timestep
    )

    # test
    self.assertLess(np.abs(settings["epsilon"] - epsilon), 1.0e-5)
    self.assertTrue(settings["flg_centered"] == flg_centered)
    self.assertTrue(settings["auto_timestep"] == auto_timestep)

  def test_updates(self):
    # load model
    model_path = (
        pathlib.Path(__file__).parent.parent.parent
        / "mjpc/test/testdata/estimator/particle/task1D.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))

    # initialize
    kalman = kalman_lib.KALMAN(model=model)

    # state
    state = np.random.normal(scale=1.0, size=(model.nq + model.nv))
    state_response = kalman.state(state=state)

    # test state
    self.assertLess(np.linalg.norm(state_response - state), 1.0e-5)

    # covariance
    nvelocity = 2 * model.nv
    F = np.random.normal(scale=1.0, size=(nvelocity**2)).reshape(nvelocity, nvelocity)
    covariance = F.T @ F
    covariance_response = kalman.covariance(covariance=covariance)

    # test covariance
    self.assertLess(np.linalg.norm((covariance_response - covariance).ravel()), 1.0e-5)
    self.assertTrue(covariance_response.shape == (nvelocity, nvelocity))

    # noise
    process = np.random.normal(scale=1.0e-3, size=nvelocity)
    sensor = np.random.normal(scale=1.0e-3, size=model.nsensordata)
    noise = kalman.noise(process=process, sensor=sensor)

    # test noise
    self.assertLess(np.linalg.norm(noise["process"] - process), 1.0e-5)
    self.assertLess(np.linalg.norm(noise["sensor"] - sensor), 1.0e-5)

    # measurement update
    ctrl = np.random.normal(scale=1.0, size=model.nu)
    sensor = np.random.normal(scale=1.0, size=model.nsensordata)
    kalman.update_measurement(ctrl=ctrl, sensor=sensor)

    # # prediction update
    kalman.update_prediction()


    # timers
    timer = kalman.timers()

    self.assertTrue(timer["measurement"] > 0.0)
    self.assertTrue(timer["prediction"] > 0.0)

if __name__ == "__main__":
  absltest.main()
