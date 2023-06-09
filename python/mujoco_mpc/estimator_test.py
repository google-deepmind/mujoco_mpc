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

  def test_get_set(self):
    # load model
    model_path = (
        pathlib.Path(__file__).parent.parent.parent
        / "mjpc/test/testdata/estimator/particle/task.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))

    # initialize
    configuration_length = 5
    estimator = agent_lib.Estimator(model=model, configuration_length=configuration_length)

    # time index
    index = 0

    ## configuration

    # set 
    configuration = np.random.rand(model.nq)
    estimator.set_configuration(configuration, index)

    # get configuration
    out_configuration = estimator.get_configuration(index)

    # test that input and output match
    self.assertTrue(np.linalg.norm(configuration - out_configuration) < 1.0e-3)

    ## velocity 

    # set 
    velocity = np.random.rand(model.nv)
    estimator.set_velocity(velocity, index)

    # get velocity
    out_velocity = estimator.get_velocity(index)

    # test that input and output match
    self.assertTrue(np.linalg.norm(velocity - out_velocity) < 1.0e-3)

    ## acceleration 

    # set 
    acceleration = np.random.rand(model.nv)
    estimator.set_acceleration(acceleration, index)

    # get acceleration
    out_acceleration = estimator.get_acceleration(index)

    # test that input and output match
    self.assertTrue(np.linalg.norm(acceleration - out_acceleration) < 1.0e-3)

    ## action 

    # set 
    action = np.random.rand(model.nu)
    estimator.set_action(action, index)

    # get action
    out_action = estimator.get_action(index)

    # test that input and output match
    self.assertTrue(np.linalg.norm(action - out_action) < 1.0e-3)

    ## time 

    # set 
    time = np.random.rand(1)
    estimator.set_time(time, index)

    # get time
    out_time = estimator.get_time(index)

    # test that input and output match
    self.assertTrue(np.linalg.norm(time - out_time) < 1.0e-3)

    ## configuration prior 

    # set 
    configuration_prior = np.random.rand(model.nq)
    estimator.set_configuration_prior(configuration_prior, index)

    # get configuration prior
    out_configuration_prior = estimator.get_configuration_prior(index)

    # test that input and output match
    self.assertTrue(np.linalg.norm(configuration_prior - out_configuration_prior) < 1.0e-3)

    ## sensor measurement 

    # set 
    sensor_measurement = np.random.rand(model.nsensordata)
    estimator.set_sensor_measurement(sensor_measurement, index)

    # get sensor measurement
    out_sensor_measurement = estimator.get_sensor_measurement(index)

    # test that input and output match
    self.assertTrue(np.linalg.norm(sensor_measurement - out_sensor_measurement) < 1.0e-3)

    ## sensor prediction 

    # set 
    sensor_prediction = np.random.rand(model.nsensordata)
    estimator.set_sensor_prediction(sensor_prediction, index)

    # get sensor prediction
    out_sensor_prediction = estimator.get_sensor_prediction(index)

    # test that input and output match
    self.assertTrue(np.linalg.norm(sensor_prediction - out_sensor_prediction) < 1.0e-3)

    ## force measurement 

    # set 
    force_measurement = np.random.rand(model.nv)
    estimator.set_force_measurement(force_measurement, index)

    # get force_measurement
    out_force_measurement = estimator.get_force_measurement(index)

    # test that input and output match
    self.assertTrue(np.linalg.norm(force_measurement - out_force_measurement) < 1.0e-3)

    ## force prediction 

    # set 
    force_prediction = np.random.rand(model.nv)
    estimator.set_force_prediction(force_prediction, index)

    # get force_prediction
    out_force_prediction = estimator.get_force_prediction(index)

    # test that input and output match
    self.assertTrue(np.linalg.norm(force_prediction - out_force_prediction) < 1.0e-3)

if __name__ == "__main__":
  absltest.main()
