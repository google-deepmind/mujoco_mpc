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
from mujoco_mpc import batch_estimator as batch_estimator_lib
import numpy as np

import pathlib


class BatchEstimatorTest(absltest.TestCase):

  def test_initialized(self):
    # load model
    model_path = (
        pathlib.Path(__file__).parent.parent.parent
        / "mjpc/test/testdata/estimator/quadruped/task.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))

    # initialize
    configuration_length = 5
    batch_estimator = batch_estimator_lib.BatchEstimator(
        model=model, configuration_length=configuration_length, send_as="mjb"
    )

    batch_estimator.data(0)["configuration"]

  def test_data(self):
    # load model
    model_path = (
        pathlib.Path(__file__).parent.parent.parent
        / "mjpc/test/testdata/estimator/particle/task.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))

    # initialize
    configuration_length = 5
    batch_estimator = batch_estimator_lib.BatchEstimator(
        model=model, configuration_length=configuration_length
    )

    # time index
    index = 0

    ## configuration

    # set
    configuration = np.random.rand(model.nq)
    data = batch_estimator.data(index, configuration=configuration)

    # test that input and output match
    self.assertTrue(np.linalg.norm(configuration - data["configuration"]) < 1.0e-5)

    ## velocity

    # set
    velocity = np.random.rand(model.nv)
    data = batch_estimator.data(index, velocity=velocity)

    # test that input and output match
    self.assertLess(np.linalg.norm(velocity - data["velocity"]), 1.0e-5)

    ## acceleration

    # set
    acceleration = np.random.rand(model.nv)
    data = batch_estimator.data(index, acceleration=acceleration)

    # test that input and output match
    self.assertLess(np.linalg.norm(acceleration - data["acceleration"]), 1.0e-5)

    ## time

    # set
    time = np.random.rand(1)
    data = batch_estimator.data(index, time=time)

    # test that input and output match
    self.assertLess(np.linalg.norm(time - data["time"]), 1.0e-5)

    ## ctrl

    # set
    ctrl = np.random.rand(model.nu)
    data = batch_estimator.data(index, ctrl=ctrl)

    # test that input and output match
    self.assertLess(np.linalg.norm(ctrl - data["ctrl"]), 1.0e-5)

    ## configuration prev

    # set
    configuration_previous = np.random.rand(model.nq)
    data = batch_estimator.data(index, configuration_previous=configuration_previous)

    # test that input and output match
    self.assertLess(
        np.linalg.norm(configuration_previous - data["configuration_previous"]), 1.0e-5
    )

    ## sensor measurement

    # set
    sensor_measurement = np.random.rand(model.nsensordata)
    data = batch_estimator.data(index, sensor_measurement=sensor_measurement)

    # test that input and output match
    self.assertTrue(
        np.linalg.norm(sensor_measurement - data["sensor_measurement"]) < 1.0e-5
    )

    ## sensor prediction

    # set
    sensor_prediction = np.random.rand(model.nsensordata)
    data = batch_estimator.data(index, sensor_prediction=sensor_prediction)

    # test that input and output match
    self.assertTrue(
        np.linalg.norm(sensor_prediction - data["sensor_prediction"]) < 1.0e-5
    )

    ## sensor mask 

    # set
    sensor_mask = np.array([1, 0, 1, 0])
    data = batch_estimator.data(index, sensor_mask=sensor_mask)

    # test that input and output match
    self.assertTrue(
      np.linalg.norm(sensor_mask - data["sensor_mask"]) < 1.0e-5
    )

    ## force measurement

    # set
    force_measurement = np.random.rand(model.nv)
    data = batch_estimator.data(index, force_measurement=force_measurement)

    # test that input and output match
    self.assertTrue(
        np.linalg.norm(force_measurement - data["force_measurement"]) < 1.0e-5
    )

    ## force prediction

    # set
    force_prediction = np.random.rand(model.nv)
    data = batch_estimator.data(index, force_prediction=force_prediction)

    # test that input and output match
    self.assertTrue(
        np.linalg.norm(force_prediction - data["force_prediction"]) < 1.0e-5
    )

  def test_settings(self):
    # load model
    model_path = (
        pathlib.Path(__file__).parent.parent.parent
        / "mjpc/test/testdata/estimator/particle/task.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))

    # initialize
    configuration_length = 15
    batch_estimator = batch_estimator_lib.BatchEstimator(
        model=model, configuration_length=configuration_length
    )

    # initial configuration length
    settings = batch_estimator.settings()
    self.assertTrue(configuration_length == settings["configuration_length"])

    # get/set configuration length
    in_configuration_length = 7
    settings = batch_estimator.settings(configuration_length=in_configuration_length)
    self.assertTrue(in_configuration_length == settings["configuration_length"])

    # get/set prior flag
    in_prior_flag = False
    settings = batch_estimator.settings(prior_flag=in_prior_flag)
    self.assertTrue(in_prior_flag == settings["prior_flag"])

    # get/set sensor flag
    in_sensor_flag = False
    settings = batch_estimator.settings(sensor_flag=in_sensor_flag)
    self.assertTrue(in_sensor_flag == settings["sensor_flag"])

    # get/set force flag
    in_force_flag = False
    settings = batch_estimator.settings(force_flag=in_force_flag)
    self.assertTrue(in_force_flag == settings["force_flag"])

    # get/set search iterations
    in_search_iterations = 25
    settings = batch_estimator.settings(max_search_iterations=in_search_iterations)
    self.assertTrue(in_search_iterations == settings["max_search_iterations"])

    # get/set smoother iterations
    in_smoother_iterations = 25
    settings = batch_estimator.settings(max_smoother_iterations=in_smoother_iterations)
    self.assertTrue(in_smoother_iterations == settings["max_smoother_iterations"])

    # get/set gradient tolerance 
    gradient_tolerance = 1.23456
    settings = batch_estimator.settings(gradient_tolerance=gradient_tolerance)
    self.assertTrue(np.abs(gradient_tolerance - settings["gradient_tolerance"]) < 1.0e-6)

    # get/set verbose iteration
    verbose_iteration = True 
    settings = batch_estimator.settings(verbose_iteration=verbose_iteration)
    self.assertTrue(verbose_iteration == settings["verbose_iteration"])

    # get/set verbose optimize
    verbose_optimize = True 
    settings = batch_estimator.settings(verbose_optimize=verbose_optimize)
    self.assertTrue(verbose_optimize == settings["verbose_optimize"])

    # get/set verbose cost
    verbose_cost = True 
    settings = batch_estimator.settings(verbose_cost=verbose_cost)
    self.assertTrue(verbose_cost == settings["verbose_cost"])

    # get/set verbose prior
    verbose_prior = True 
    settings = batch_estimator.settings(verbose_prior=verbose_prior)
    self.assertTrue(verbose_prior == settings["verbose_prior"])

    # get/set band prior
    band_prior = False 
    settings = batch_estimator.settings(band_prior=band_prior)
    self.assertTrue(band_prior == settings["band_prior"])

    # get/set search type
    in_search_type = 0
    settings = batch_estimator.settings(search_type=in_search_type)
    self.assertTrue(in_search_type == settings["search_type"])

    # get/set step scaling 
    in_step_scaling = 2.5 
    settings = batch_estimator.settings(step_scaling=in_step_scaling)
    self.assertTrue(np.abs(in_step_scaling - settings["step_scaling"]) < 1.0e-4)

    # get/set regularization initial 
    in_regularization_initial = 3.0e1
    settings = batch_estimator.settings(regularization_initial=in_regularization_initial)
    self.assertTrue(np.abs(in_regularization_initial - settings["regularization_initial"]) < 1.0e-4)

    # get/set regularization scaling 
    in_regularization_scaling = 7.1
    settings = batch_estimator.settings(regularization_scaling=in_regularization_scaling)
    self.assertTrue(np.abs(in_regularization_scaling - settings["regularization_scaling"]) < 1.0e-4)

    # get/set band copy 
    in_band_copy = False 
    settings = batch_estimator.settings(band_copy=in_band_copy) 
    self.assertTrue(in_band_copy == settings["band_copy"])

    # get/set search direction tolerance
    search_direction_tolerance = 3.3 
    settings = batch_estimator.settings(search_direction_tolerance=search_direction_tolerance)
    self.assertTrue(np.abs(search_direction_tolerance - settings["search_direction_tolerance"]) < 1.0e-5)
   
    # get/set cost tolerance 
    cost_tolerance = 1.0e-3 
    settings = batch_estimator.settings(cost_tolerance=cost_tolerance)
    self.assertTrue(np.abs(cost_tolerance - settings["cost_tolerance"]) < 1.0e-5)

    # get/set assemble prior Jacobian 
    assemble_prior_jacobian = True
    settings = batch_estimator.settings(assemble_prior_jacobian=assemble_prior_jacobian)
    self.assertTrue(assemble_prior_jacobian == settings["assemble_prior_jacobian"])

    # get/set assemble sensor Jacobian 
    assemble_sensor_jacobian = True
    settings = batch_estimator.settings(assemble_sensor_jacobian=assemble_sensor_jacobian)
    self.assertTrue(assemble_sensor_jacobian == settings["assemble_sensor_jacobian"])

    # get/set assemble force Jacobian 
    assemble_force_jacobian = True
    settings = batch_estimator.settings(assemble_force_jacobian=assemble_force_jacobian)
    self.assertTrue(assemble_force_jacobian == settings["assemble_force_jacobian"])

    # get/set assemble sensor norm Hessian
    assemble_sensor_norm_hessian = True
    settings = batch_estimator.settings(assemble_sensor_norm_hessian=assemble_sensor_norm_hessian)
    self.assertTrue(assemble_sensor_norm_hessian == settings["assemble_sensor_norm_hessian"])

    # get/set assemble force norm Hessian
    assemble_force_norm_hessian = True
    settings = batch_estimator.settings(assemble_force_norm_hessian=assemble_force_norm_hessian)
    self.assertTrue(assemble_force_norm_hessian == settings["assemble_force_norm_hessian"])

  def test_costs(self):
    # load model
    model_path = (
        pathlib.Path(__file__).parent.parent.parent
        / "mjpc/test/testdata/estimator/particle/task.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))

    # initialize
    configuration_length = 5
    batch_estimator = batch_estimator_lib.BatchEstimator(
        model=model, configuration_length=configuration_length
    )

    # TODO(taylor): better tests

    # cost
    cost = batch_estimator.cost()

    self.assertTrue(np.abs(cost["total"] - 0.0) < 1.0e-5)

    # cost prior
    self.assertTrue(np.abs(cost["prior"] - 0.0) < 1.0e-5)

    # cost sensor
    self.assertTrue(np.abs(cost["sensor"] - 0.0) < 1.0e-5)

    # cost force
    self.assertTrue(np.abs(cost["force"] - 0.0) < 1.0e-5)

    # cost initial
    self.assertTrue(np.abs(cost["initial"] - 0.0) < 1.0e-5)

    #TODO(taylor): derivatives 

    #TODO(taylor): internals

  def test_weights(self):
    # load model
    model_path = (
        pathlib.Path(__file__).parent.parent.parent
        / "mjpc/test/testdata/estimator/particle/task.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))

    # initialize
    configuration_length = 5
    batch_estimator = batch_estimator_lib.BatchEstimator(
        model=model, configuration_length=configuration_length
    )

    ## prior
    in_prior_weight = 2.5
    weight = batch_estimator.weight(prior=in_prior_weight)
    self.assertTrue(np.abs(in_prior_weight - weight["prior"]) < 1.0e-5)

    ## sensor
    in_sensor_weight = np.random.rand(model.nsensor)
    weight = batch_estimator.weight(sensor=in_sensor_weight)
    self.assertTrue(np.linalg.norm(in_sensor_weight - weight["sensor"]) < 1.0e-5)

    ## force
    in_force_weight = np.random.rand(model.nv)
    weight = batch_estimator.weight(force=in_force_weight)
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
    batch_estimator = batch_estimator_lib.BatchEstimator(
        model=model, configuration_length=configuration_length
    )

    # no shift
    head = batch_estimator.shift(0)
    self.assertTrue(head == 0)

    # shift
    shift = 1
    head = batch_estimator.shift(shift)
    self.assertTrue(head == 1)

    shift = 2
    head = batch_estimator.shift(shift)
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
    batch_estimator = batch_estimator_lib.BatchEstimator(
        model=model, configuration_length=configuration_length
    )

    # set
    index = 1
    configuration = np.random.rand(model.nq)
    sensor_measurement = np.random.rand(model.nsensordata)
    data = batch_estimator.data(
        index, configuration=configuration, sensor_measurement=sensor_measurement
    )

    # check that elements are set
    self.assertTrue(np.linalg.norm(data["configuration"]) > 0.0)
    self.assertTrue(np.linalg.norm(data["sensor_measurement"]) > 0.0)

    # reset
    batch_estimator.reset()

    # get data
    data = batch_estimator.data(index)

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
    batch_estimator = batch_estimator_lib.BatchEstimator(
        model=model, configuration_length=configuration_length
    )

    # TODO(taylor): setup

    # optimize
    # batch_estimator.optimize()

  def test_status(self):
    # load model
    model_path = (
        pathlib.Path(__file__).parent.parent.parent
        / "mjpc/test/testdata/estimator/particle/task.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))

    # initialize
    configuration_length = 5
    batch_estimator = batch_estimator_lib.BatchEstimator(
        model=model, configuration_length=configuration_length
    )

    # TODO(taylor): setup
    status = batch_estimator.status()

    # search iterations
    self.assertTrue(status["search_iterations"] == 0)

    # smoother iterations
    self.assertTrue(status["smoother_iterations"] == 0)

    # step size
    self.assertTrue(np.abs(status["step_size"] - 1.0) < 1.0e-5)

    # regularization
    self.assertTrue(np.abs(status["regularization"] - batch_estimator.settings()["regularization_initial"]) < 1.0e-6)

    # gradient norm
    self.assertTrue(np.abs(status["gradient_norm"]) < 1.0e-5)

    # search direction norm 
    self.assertTrue(np.abs(status["search_direction_norm"]) < 1.0e-5)

    # solve status 
    self.assertTrue(status["solve_status"] == 0)

    # cost difference 
    self.assertTrue(np.abs(status["cost_difference"]) < 1.0e-5)

  def test_prior_matrix(self):
    # load model
    model_path = (
        pathlib.Path(__file__).parent.parent.parent
        / "mjpc/test/testdata/estimator/particle/task.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))

    # initialize
    configuration_length = 5
    batch_estimator = batch_estimator_lib.BatchEstimator(
        model=model, configuration_length=configuration_length
    )

    # dimension
    dim = configuration_length * model.nv

    # get uninitialized (zero) matrix
    prior0 = batch_estimator.prior_matrix()

    # test
    self.assertTrue(prior0.shape == (dim, dim))
    self.assertTrue(not prior0.any())

    # random
    in_prior = np.random.rand(dim, dim)
    out_prior = batch_estimator.prior_matrix(prior=in_prior)

    # test
    self.assertTrue(np.linalg.norm(in_prior - out_prior) < 1.0e-4)

  def test_norm(self):
    # load model
    model_path = (
        pathlib.Path(__file__).parent.parent.parent
        / "mjpc/test/testdata/estimator/particle/task.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))

    # initialize
    configuration_length = 5
    batch_estimator = batch_estimator_lib.BatchEstimator(
        model=model, configuration_length=configuration_length
    )

    # get norm data
    data = batch_estimator.norm()

    # test norm types
    self.assertTrue((data["sensor_type"] == np.zeros(model.nsensor)).all())

    # test norm paramters
    self.assertTrue(not data["sensor_parameters"].any())

    # set norm data
    sensor_type = np.array([1, 2, 3, 4])
    sensor_parameters = np.random.rand(3 * model.nsensor)
    data = batch_estimator.norm(
        sensor_type=sensor_type,
        sensor_parameters=sensor_parameters,
    )

    # test
    self.assertTrue((sensor_type == data["sensor_type"]).all())
    self.assertLess(
        np.linalg.norm(sensor_parameters - data["sensor_parameters"]), 1.0e-5
    )

  # TODO(taylor): test initialize_data, update_data() 

if __name__ == "__main__":
  absltest.main()
