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
from mujoco_mpc import estimator as estimator_lib
import numpy as np

import pathlib


class EstimatorTest(absltest.TestCase):

  def test_data(self):
    # load model
    model_path = (
        pathlib.Path(__file__).parent.parent.parent
        / "mjpc/test/testdata/estimator/particle/task.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))

    # initialize
    configuration_length = 5
    estimator = estimator_lib.Estimator(
        model=model, configuration_length=configuration_length
    )

    # time index
    index = 0

    ## configuration

    # set
    configuration = np.random.rand(model.nq)
    data = estimator.data(index, configuration=configuration)

    # test that input and output match
    self.assertTrue(np.linalg.norm(configuration - data["configuration"]) < 1.0e-3)

    ## velocity

    # set
    velocity = np.random.rand(model.nv)
    data = estimator.data(index, velocity=velocity)

    # test that input and output match
    self.assertTrue(np.linalg.norm(velocity - data["velocity"]) < 1.0e-3)

    ## acceleration

    # set
    acceleration = np.random.rand(model.nv)
    data = estimator.data(index, acceleration=acceleration)

    # test that input and output match
    self.assertTrue(np.linalg.norm(acceleration - data["acceleration"]) < 1.0e-3)

    ## time

    # set
    time = np.random.rand(1)
    data = estimator.data(index, time=time)

    # test that input and output match
    self.assertTrue(np.linalg.norm(time - data["time"]) < 1.0e-3)

    ## configuration prior

    # set
    configuration_prior = np.random.rand(model.nq)
    data = estimator.data(index, configuration_prior=configuration_prior)

    # test that input and output match
    self.assertTrue(
        np.linalg.norm(configuration_prior - data["configuration_prior"]) < 1.0e-3
    )

    ## sensor measurement

    # set
    sensor_measurement = np.random.rand(model.nsensordata)
    data = estimator.data(index, sensor_measurement=sensor_measurement)

    # test that input and output match
    self.assertTrue(
        np.linalg.norm(sensor_measurement - data["sensor_measurement"]) < 1.0e-3
    )

    ## sensor prediction

    # set
    sensor_prediction = np.random.rand(model.nsensordata)
    data = estimator.data(index, sensor_prediction=sensor_prediction)

    # test that input and output match
    self.assertTrue(
        np.linalg.norm(sensor_prediction - data["sensor_prediction"]) < 1.0e-3
    )

    ## force measurement

    # set
    force_measurement = np.random.rand(model.nv)
    data = estimator.data(index, force_measurement=force_measurement)

    # test that input and output match
    self.assertTrue(
        np.linalg.norm(force_measurement - data["force_measurement"]) < 1.0e-3
    )

    ## force prediction

    # set
    force_prediction = np.random.rand(model.nv)
    data = estimator.data(index, force_prediction=force_prediction)

    # test that input and output match
    self.assertTrue(
        np.linalg.norm(force_prediction - data["force_prediction"]) < 1.0e-3
    )

  def test_settings(self):
    # load model
    model_path = (
        pathlib.Path(__file__).parent.parent.parent
        / "mjpc/test/testdata/estimator/particle/task.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))

    # initialize
    configuration_length = 5
    estimator = estimator_lib.Estimator(
        model=model, configuration_length=configuration_length
    )

    # initial configuration length
    settings = estimator.settings()
    self.assertTrue(configuration_length == settings["configuration_length"])

    # get/set configuration length
    in_configuration_length = 7
    settings = estimator.settings(configuration_length=in_configuration_length)
    self.assertTrue(in_configuration_length == settings["configuration_length"])

    # get/set search type
    in_search_type = 1
    settings = estimator.settings(search_type=in_search_type)

    self.assertTrue(in_search_type == settings["search_type"])

    # get/set prior flag
    in_prior_flag = False
    settings = estimator.settings(prior_flag=in_prior_flag)
    self.assertTrue(in_prior_flag == settings["prior_flag"])

    # get/set sensor flag
    in_sensor_flag = False
    settings = estimator.settings(sensor_flag=in_sensor_flag)
    self.assertTrue(in_sensor_flag == settings["sensor_flag"])

    # get/set force flag
    in_force_flag = False
    settings = estimator.settings(force_flag=in_force_flag)
    self.assertTrue(in_force_flag == settings["force_flag"])

    # get/set smoother iterations
    in_iterations = 25
    settings = estimator.settings(smoother_iterations=in_iterations)
    self.assertTrue(in_iterations == settings["smoother_iterations"])

    # get/set skip prior weight update
    in_skip = True
    settings = estimator.settings(skip_prior_weight_update=in_skip)
    self.assertTrue(in_skip == settings["skip_prior_weight_update"])

  def test_costs(self):
    # load model
    model_path = (
        pathlib.Path(__file__).parent.parent.parent
        / "mjpc/test/testdata/estimator/particle/task.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))

    # initialize
    configuration_length = 5
    estimator = estimator_lib.Estimator(
        model=model, configuration_length=configuration_length
    )

    # TODO(taylor): better tests

    # cost
    cost = estimator.cost()

    self.assertTrue(np.abs(cost["total"] - 0.0) < 1.0e-5)

    # cost prior
    self.assertTrue(np.abs(cost["prior"] - 0.0) < 1.0e-5)

    # cost sensor
    self.assertTrue(np.abs(cost["sensor"] - 0.0) < 1.0e-5)

    # cost force
    self.assertTrue(np.abs(cost["force"] - 0.0) < 1.0e-5)

    # cost initial
    self.assertTrue(np.abs(cost["initial"] - 0.0) < 1.0e-5)

  def test_weights(self):
    # load model
    model_path = (
        pathlib.Path(__file__).parent.parent.parent
        / "mjpc/test/testdata/estimator/particle/task.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))

    # initialize
    configuration_length = 5
    estimator = estimator_lib.Estimator(
        model=model, configuration_length=configuration_length
    )

    ## prior
    in_prior_weight = 2.5
    weight = estimator.weight(prior=in_prior_weight)
    self.assertTrue(np.abs(in_prior_weight - weight["prior"]) < 1.0e-5)

    ## sensor
    in_sensor_weight = np.random.rand(model.nsensordata)
    weight = estimator.weight(sensor=in_sensor_weight)
    self.assertTrue(np.linalg.norm(in_sensor_weight - weight["sensor"]) < 1.0e-5)

    ## force
    in_force_weight = np.random.rand(model.nv)
    weight = estimator.weight(force=in_force_weight)
    self.assertTrue(np.linalg.norm(in_force_weight - weight["force"]) < 1.0e-5)

  def test_shift(self):
    # load model
    model_path = (
        pathlib.Path(__file__).parent.parent.parent
        / "mjpc/test/testdata/estimator/particle/task.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))

    # initialize
    configuration_length = 5
    estimator = estimator_lib.Estimator(
        model=model, configuration_length=configuration_length
    )

    # no shift
    head = estimator.shift(0)
    self.assertTrue(head == 0)

    # shift
    shift = 1
    head = estimator.shift(shift)
    self.assertTrue(head == 1)

    shift = 2
    head = estimator.shift(shift)
    self.assertTrue(head == 3)

  def test_reset(self):
    # load model
    model_path = (
        pathlib.Path(__file__).parent.parent.parent
        / "mjpc/test/testdata/estimator/particle/task.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))

    # initialize
    configuration_length = 5
    estimator = estimator_lib.Estimator(
        model=model, configuration_length=configuration_length
    )

    # set
    index = 1
    configuration = np.random.rand(model.nq)
    sensor_measurement = np.random.rand(model.nsensordata)
    data = estimator.data(
        index, configuration=configuration, sensor_measurement=sensor_measurement
    )

    # check that elements are set
    self.assertTrue(np.linalg.norm(data["configuration"]) > 0.0)
    self.assertTrue(np.linalg.norm(data["sensor_measurement"]) > 0.0)

    # reset
    estimator.reset()

    # get data
    data = estimator.data(index)

    # check that elements are reset to zero
    self.assertTrue(np.linalg.norm(data["configuration"]) < 1.0e-5)
    self.assertTrue(np.linalg.norm(data["sensor_measurement"]) < 1.0e-5)

  def test_optimize(self):
    # load model
    model_path = (
        pathlib.Path(__file__).parent.parent.parent
        / "mjpc/test/testdata/estimator/particle/task.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))

    # initialize
    configuration_length = 5
    estimator = estimator_lib.Estimator(
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
    estimator = estimator_lib.Estimator(
        model=model, configuration_length=configuration_length
    )

    # TODO(taylor): setup
    status = estimator.status()

    # search iterations
    self.assertTrue(status["search_iterations"] == 0)

    # smoother iterations
    self.assertTrue(status["smoother_iterations"] == 0)

    # step size
    self.assertTrue(np.abs(status["step_size"] - 1.0) < 1.0e-5)

    # regularization
    self.assertTrue(np.abs(status["regularization"] - 1.0e-5) < 1.0e-6)

    # gradient norm
    self.assertTrue(np.abs(status["gradient_norm"]) < 1.0e-3)

  def test_cost_hessian(self):
    # load model
    model_path = (
        pathlib.Path(__file__).parent.parent.parent
        / "mjpc/test/testdata/estimator/particle/task.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))

    # initialize
    configuration_length = 5
    estimator = estimator_lib.Estimator(
        model=model, configuration_length=configuration_length
    )

    # Hessian
    hessian = estimator.cost_hessian()

    # test dimension
    dim = configuration_length * model.nv
    self.assertTrue(hessian.shape == (dim, dim))

  def test_prior_matrix(self):
    # load model
    model_path = (
        pathlib.Path(__file__).parent.parent.parent
        / "mjpc/test/testdata/estimator/particle/task.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))

    # initialize
    configuration_length = 5
    estimator = estimator_lib.Estimator(
        model=model, configuration_length=configuration_length
    )

    # dimension
    dim = configuration_length * model.nv

    # get uninitialized (zero) matrix
    prior0 = estimator.prior_matrix()

    # test
    self.assertTrue(prior0.shape == (dim, dim))
    self.assertTrue(not prior0.any())

    # random
    in_prior = np.random.rand(dim, dim)
    out_prior = estimator.prior_matrix(prior=in_prior)

    # test
    self.assertTrue(np.linalg.norm(in_prior - out_prior) < 1.0e-4)


if __name__ == "__main__":
  absltest.main()
