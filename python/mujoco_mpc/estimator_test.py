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

  def test_get_set_data(self):
    # load model
    model_path = (
        pathlib.Path(__file__).parent.parent.parent
        / "mjpc/test/testdata/estimator/particle/task.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))

    # initialize
    configuration_length = 5
    estimator = agent_lib.Estimator(
        model=model, configuration_length=configuration_length
    )

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
    self.assertTrue(
        np.linalg.norm(configuration_prior - out_configuration_prior) < 1.0e-3
    )

    ## sensor measurement

    # set
    sensor_measurement = np.random.rand(model.nsensordata)
    estimator.set_sensor_measurement(sensor_measurement, index)

    # get sensor measurement
    out_sensor_measurement = estimator.get_sensor_measurement(index)

    # test that input and output match
    self.assertTrue(
        np.linalg.norm(sensor_measurement - out_sensor_measurement) < 1.0e-3
    )

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

  def test_settings(self):
    # load model
    model_path = (
        pathlib.Path(__file__).parent.parent.parent
        / "mjpc/test/testdata/estimator/particle/task.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))

    # initialize
    configuration_length = 5
    estimator = agent_lib.Estimator(
        model=model, configuration_length=configuration_length
    )

    # initial configuration length
    self.assertTrue(configuration_length == estimator.get_configuration_length())

    # get/set configuration length
    in_configuration_length = 7
    estimator.set_configuration_length(in_configuration_length)
    out_configuration_length = estimator.get_configuration_length()

    self.assertTrue(in_configuration_length == out_configuration_length)

    # get/set search type
    in_search_type = 1
    estimator.set_search_type(in_search_type)
    out_search_type = estimator.get_search_type()

    self.assertTrue(in_search_type == out_search_type)

    # get/set prior flag
    in_prior_flag = False
    estimator.set_prior_flag(in_prior_flag)
    out_prior_flag = estimator.get_prior_flag()

    self.assertTrue(in_prior_flag == out_prior_flag)

    # get/set sensor flag
    in_sensor_flag = False
    estimator.set_sensor_flag(in_sensor_flag)
    out_sensor_flag = estimator.get_sensor_flag()

    self.assertTrue(in_sensor_flag == out_sensor_flag)

    # get/set force flag
    in_force_flag = False
    estimator.set_force_flag(in_force_flag)
    out_force_flag = estimator.get_force_flag()

    self.assertTrue(in_force_flag == out_force_flag)

    # get/set smoother iterations
    in_iterations = 25
    estimator.set_smoother_iterations(in_iterations)
    out_iterations = estimator.get_smoother_iterations()

    self.assertTrue(in_iterations == out_iterations)

  def test_costs(self):
    # load model
    model_path = (
        pathlib.Path(__file__).parent.parent.parent
        / "mjpc/test/testdata/estimator/particle/task.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))

    # initialize
    configuration_length = 5
    estimator = agent_lib.Estimator(
        model=model, configuration_length=configuration_length
    )

    # TODO(taylor): better tests

    # cost
    cost = estimator.get_cost()

    self.assertTrue(np.abs(cost - 0.0) < 1.0e-5)

    # cost prior
    cost_prior = estimator.get_cost_prior()

    self.assertTrue(np.abs(cost_prior - 0.0) < 1.0e-5)

    # cost sensor
    cost_sensor = estimator.get_cost_sensor()

    self.assertTrue(np.abs(cost_sensor - 0.0) < 1.0e-5)

    # cost force
    cost_force = estimator.get_cost_force()

    self.assertTrue(np.abs(cost_force - 0.0) < 1.0e-5)

    # cost initial
    cost_initial = estimator.get_cost_initial()

    self.assertTrue(np.abs(cost_initial - 0.0) < 1.0e-5)

  def test_weights(self):
    # load model
    model_path = (
        pathlib.Path(__file__).parent.parent.parent
        / "mjpc/test/testdata/estimator/particle/task.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))

    # initialize
    configuration_length = 5
    estimator = agent_lib.Estimator(
        model=model, configuration_length=configuration_length
    )

    ## prior
    in_prior_weight = 2.5
    estimator.set_prior_weight(in_prior_weight)
    out_prior_weight = estimator.get_prior_weight()

    self.assertTrue(np.abs(in_prior_weight - out_prior_weight) < 1.0e-5)

    ## sensor
    in_sensor_weight = np.random.rand(model.nsensordata)
    estimator.set_sensor_weight(in_sensor_weight)
    out_sensor_weight = estimator.get_sensor_weight()

    self.assertTrue(np.linalg.norm(in_sensor_weight - out_sensor_weight) < 1.0e-5)

    ## force
    in_force_weight = np.random.rand(model.nv)
    estimator.set_force_weight(in_force_weight)
    out_force_weight = estimator.get_force_weight()

    self.assertTrue(np.linalg.norm(in_force_weight - out_force_weight) < 1.0e-5)

  def test_shift(self):
    # load model
    model_path = (
        pathlib.Path(__file__).parent.parent.parent
        / "mjpc/test/testdata/estimator/particle/task.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))

    # initialize
    configuration_length = 5
    estimator = agent_lib.Estimator(
        model=model, configuration_length=configuration_length
    )

    # no shift
    head = estimator.shift_trajectory(0)
    self.assertTrue(head == 0)

    # shift
    shift = 1
    head = estimator.shift_trajectory(shift)
    self.assertTrue(head == 1)

    shift = 2
    head = estimator.shift_trajectory(shift)
    self.assertTrue(head == 3)

  def test_shift(self):
    # load model
    model_path = (
        pathlib.Path(__file__).parent.parent.parent
        / "mjpc/test/testdata/estimator/particle/task.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))

    # initialize
    configuration_length = 5
    estimator = agent_lib.Estimator(
        model=model, configuration_length=configuration_length
    )

    # set
    index = 1
    configuration = np.random.rand(model.nq)
    estimator.set_configuration(configuration, index)
    sensor_measurement = np.random.rand(model.nsensordata)
    estimator.set_sensor_measurement(sensor_measurement, index)

    # check that elements are set
    self.assertTrue(np.linalg.norm(estimator.get_configuration(index)) > 0.0)
    self.assertTrue(np.linalg.norm(estimator.get_sensor_measurement(index)) > 0.0)

    # reset
    estimator.reset()

    # check that elements are reset to zero
    self.assertTrue(np.linalg.norm(estimator.get_configuration(index)) < 1.0e-5)
    self.assertTrue(np.linalg.norm(estimator.get_sensor_measurement(index)) < 1.0e-5)

  def test_optimize(self):
    # load model
    model_path = (
        pathlib.Path(__file__).parent.parent.parent
        / "mjpc/test/testdata/estimator/particle/task.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))

    # initialize
    configuration_length = 5
    estimator = agent_lib.Estimator(
        model=model, configuration_length=configuration_length
    )

    # TODO(taylor): setup

    # optimize
    # estimator.optimize()

  def test_status(self):
    # load model
    model_path = (
        pathlib.Path(__file__).parent.parent.parent
        / "mjpc/test/testdata/estimator/particle/task.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))

    # initialize
    configuration_length = 5
    estimator = agent_lib.Estimator(
        model=model, configuration_length=configuration_length
    )

    # TODO(taylor): setup

    # search iterations
    search_iterations = estimator.search_iterations()
    self.assertTrue(search_iterations == 0)

    # smoother iterations
    smoother_iterations = estimator.smoother_iterations()
    self.assertTrue(smoother_iterations == 0)

    # step size
    step_size = estimator.step_size()
    self.assertTrue(np.abs(step_size - 1.0) < 1.0e-5)

    # regularization
    regularization = estimator.regularization()
    self.assertTrue(np.abs(regularization - 1.0e-5) < 1.0e-6)

    # gradient norm
    gradient_norm = estimator.gradient_norm()
    self.assertTrue(np.abs(gradient_norm) < 1.0e-3)


if __name__ == "__main__":
  absltest.main()
