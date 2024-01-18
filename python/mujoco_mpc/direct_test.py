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
from mujoco_mpc import direct as direct_lib
import numpy as np

import pathlib


class DirectTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._direct = None

  def tearDown(self):
    if self._direct is not None:
      self._direct.close()
    super().tearDown()

  def test_data(self):
    # load model
    model_path = (
        pathlib.Path(__file__).parent.parent.parent
        / "mjpc/test/testdata/estimator/particle/task.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))

    # initialize
    configuration_length = 5
    self._direct = direct_lib.Direct(
        model=model, configuration_length=configuration_length
    )

    # time index
    index = 0

    ## configuration

    # set
    configuration = np.random.rand(model.nq)
    data = self._direct.data(index, configuration=configuration)

    # test that input and output match
    self.assertLess(
        np.linalg.norm(configuration - data["configuration"]), 1.0e-5
    )

    ## velocity

    # set
    velocity = np.random.rand(model.nv)
    data = self._direct.data(index, velocity=velocity)

    # test that input and output match
    self.assertLess(np.linalg.norm(velocity - data["velocity"]), 1.0e-5)

    ## acceleration

    # set
    acceleration = np.random.rand(model.nv)
    data = self._direct.data(index, acceleration=acceleration)

    # test that input and output match
    self.assertLess(np.linalg.norm(acceleration - data["acceleration"]), 1.0e-5)

    ## time

    # set
    time = np.random.rand(1)
    data = self._direct.data(index, time=time)

    # test that input and output match
    self.assertLess(np.linalg.norm(time - data["time"]), 1.0e-5)

    ## configuration prev

    # set
    configuration_previous = np.random.rand(model.nq)
    data = self._direct.data(
        index, configuration_previous=configuration_previous
    )

    # test that input and output match
    self.assertLess(
        np.linalg.norm(configuration_previous - data["configuration_previous"]),
        1.0e-5,
    )

    ## sensor measurement

    # set
    sensor_measurement = np.random.rand(model.nsensordata)
    data = self._direct.data(index, sensor_measurement=sensor_measurement)

    # test that input and output match
    self.assertLess(
        np.linalg.norm(sensor_measurement - data["sensor_measurement"]),
        1.0e-5,
    )

    ## sensor prediction

    # set
    sensor_prediction = np.random.rand(model.nsensordata)
    data = self._direct.data(index, sensor_prediction=sensor_prediction)

    # test that input and output match
    self.assertLess(
        np.linalg.norm(sensor_prediction - data["sensor_prediction"]),
        1.0e-5,
    )

    ## sensor mask

    # set
    sensor_mask = np.array([1, 0, 1, 0, 0], dtype=int)
    data = self._direct.data(index, sensor_mask=sensor_mask)

    # test that input and output match
    self.assertLess(np.linalg.norm(sensor_mask - data["sensor_mask"]), 1.0e-5)

    ## force measurement

    # set
    force_measurement = np.random.rand(model.nv)
    data = self._direct.data(index, force_measurement=force_measurement)

    # test that input and output match
    self.assertLess(
        np.linalg.norm(force_measurement - data["force_measurement"]),
        1.0e-5,
    )

    ## force prediction

    # set
    force_prediction = np.random.rand(model.nv)
    data = self._direct.data(index, force_prediction=force_prediction)

    # test that input and output match
    self.assertLess(
        np.linalg.norm(force_prediction - data["force_prediction"]), 1.0e-5
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
    self._direct = direct_lib.Direct(
        model=model, configuration_length=configuration_length
    )

    # initial configuration length
    settings = self._direct.settings()
    self.assertTrue(configuration_length == settings["configuration_length"])

    # get/set configuration length
    in_configuration_length = 7
    settings = self._direct.settings(
        configuration_length=in_configuration_length
    )
    self.assertTrue(in_configuration_length == settings["configuration_length"])

    # get/set sensor flag
    in_sensor_flag = False
    settings = self._direct.settings(sensor_flag=in_sensor_flag)
    self.assertTrue(in_sensor_flag == settings["sensor_flag"])

    # get/set force flag
    in_force_flag = False
    settings = self._direct.settings(force_flag=in_force_flag)
    self.assertTrue(in_force_flag == settings["force_flag"])

    # get/set search iterations
    in_search_iterations = 25
    settings = self._direct.settings(max_search_iterations=in_search_iterations)
    self.assertTrue(in_search_iterations == settings["max_search_iterations"])

    # get/set smoother iterations
    in_smoother_iterations = 25
    settings = self._direct.settings(
        max_smoother_iterations=in_smoother_iterations
    )
    self.assertTrue(
        in_smoother_iterations == settings["max_smoother_iterations"]
    )

    # get/set gradient tolerance
    gradient_tolerance = 1.23456
    settings = self._direct.settings(gradient_tolerance=gradient_tolerance)
    self.assertLess(
        np.abs(gradient_tolerance - settings["gradient_tolerance"]), 1.0e-6
    )

    # get/set verbose iteration
    verbose_iteration = True
    settings = self._direct.settings(verbose_iteration=verbose_iteration)
    self.assertTrue(verbose_iteration == settings["verbose_iteration"])

    # get/set verbose optimize
    verbose_optimize = True
    settings = self._direct.settings(verbose_optimize=verbose_optimize)
    self.assertTrue(verbose_optimize == settings["verbose_optimize"])

    # get/set verbose cost
    verbose_cost = True
    settings = self._direct.settings(verbose_cost=verbose_cost)
    self.assertTrue(verbose_cost == settings["verbose_cost"])

    # get/set search type
    in_search_type = 0
    settings = self._direct.settings(search_type=in_search_type)
    self.assertTrue(in_search_type == settings["search_type"])

    # get/set step scaling
    in_step_scaling = 2.5
    settings = self._direct.settings(step_scaling=in_step_scaling)
    self.assertLess(np.abs(in_step_scaling - settings["step_scaling"]), 1.0e-4)

    # get/set regularization initial
    in_regularization_initial = 3.0e1
    settings = self._direct.settings(
        regularization_initial=in_regularization_initial
    )
    self.assertLess(
        np.abs(in_regularization_initial - settings["regularization_initial"]),
        1.0e-4,
    )

    # get/set regularization scaling
    in_regularization_scaling = 7.1
    settings = self._direct.settings(
        regularization_scaling=in_regularization_scaling
    )
    self.assertLess(
        np.abs(in_regularization_scaling - settings["regularization_scaling"]),
        1.0e-4,
    )

    # get/set search direction tolerance
    search_direction_tolerance = 3.3
    settings = self._direct.settings(
        search_direction_tolerance=search_direction_tolerance
    )
    self.assertLess(
        np.abs(
            search_direction_tolerance - settings["search_direction_tolerance"]
        ),
        1.0e-5,
    )

    # get/set cost tolerance
    cost_tolerance = 1.0e-3
    settings = self._direct.settings(cost_tolerance=cost_tolerance)
    self.assertLess(np.abs(cost_tolerance - settings["cost_tolerance"]), 1.0e-5)

    # get/set assemble sensor Jacobian
    assemble_sensor_jacobian = True
    settings = self._direct.settings(
        assemble_sensor_jacobian=assemble_sensor_jacobian
    )
    self.assertTrue(
        assemble_sensor_jacobian == settings["assemble_sensor_jacobian"]
    )

    # get/set assemble force Jacobian
    assemble_force_jacobian = True
    settings = self._direct.settings(
        assemble_force_jacobian=assemble_force_jacobian
    )
    self.assertTrue(
        assemble_force_jacobian == settings["assemble_force_jacobian"]
    )

    # get/set assemble sensor norm Hessian
    assemble_sensor_norm_hessian = True
    settings = self._direct.settings(
        assemble_sensor_norm_hessian=assemble_sensor_norm_hessian
    )
    self.assertTrue(
        assemble_sensor_norm_hessian == settings["assemble_sensor_norm_hessian"]
    )

    # get/set assemble force norm Hessian
    assemble_force_norm_hessian = True
    settings = self._direct.settings(
        assemble_force_norm_hessian=assemble_force_norm_hessian
    )
    self.assertTrue(
        assemble_force_norm_hessian == settings["assemble_force_norm_hessian"]
    )

    # get/set first step position sensors
    first_step_position_sensors = True
    settings = self._direct.settings(
        first_step_position_sensors=first_step_position_sensors
    )
    self.assertTrue(
        first_step_position_sensors == settings["first_step_position_sensors"]
    )

    # get/set last step position sensors
    last_step_position_sensors = True
    settings = self._direct.settings(
        last_step_position_sensors=last_step_position_sensors
    )
    self.assertTrue(
        last_step_position_sensors == settings["last_step_position_sensors"]
    )

    # get/set last step velocity sensors
    last_step_velocity_sensors = True
    settings = self._direct.settings(
        last_step_velocity_sensors=last_step_velocity_sensors
    )
    self.assertTrue(
        last_step_velocity_sensors == settings["last_step_velocity_sensors"]
    )

  def test_costs(self):
    # load model
    model_path = (
        pathlib.Path(__file__).parent.parent.parent
        / "mjpc/test/testdata/estimator/particle/task.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))

    # initialize
    configuration_length = 5
    self._direct = direct_lib.Direct(
        model=model, configuration_length=configuration_length
    )

    # cost
    cost = self._direct.cost(derivatives=True, internals=True)

    self.assertLess(np.abs(cost["total"] - 0.001), 1.0e-4)

    # cost sensor
    self.assertLess(np.abs(cost["sensor"] - 0.001), 1.0e-4)

    # cost force
    self.assertLess(np.abs(cost["force"] - 0.0), 1.0e-5)

    # cost initial
    self.assertLess(np.abs(cost["initial"] - 0.0), 1.0e-5)

    # derivatives
    nvar = model.nv * configuration_length
    nsensor = model.nsensordata * (configuration_length - 1)
    nforce = model.nv * (configuration_length - 2)

    self.assertEqual(nvar, cost["nvar"])
    self.assertEqual(nsensor, cost["nsensor"])
    self.assertEqual(nforce, cost["nforce"])

    self.assertEqual(cost["gradient"].size, nvar)
    self.assertEqual(cost["hessian"].shape, (nvar, nvar))

    self.assertEqual(cost["residual_sensor"].size, nsensor)
    self.assertEqual(cost["residual_force"].size, nforce)

    self.assertEqual(cost["jacobian_sensor"].shape, (nsensor, nvar))
    self.assertEqual(cost["jacobian_force"].shape, (nforce, nvar))

    self.assertEqual(cost["norm_gradient_sensor"].size, nsensor)
    self.assertEqual(cost["norm_gradient_force"].size, nforce)

    self.assertEqual(cost["norm_hessian_sensor"].shape, (nsensor, nsensor))
    self.assertEqual(cost["norm_hessian_force"].shape, (nforce, nforce))

  def test_noise(self):
    # load model
    model_path = (
        pathlib.Path(__file__).parent.parent.parent
        / "mjpc/test/testdata/estimator/particle/task.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))

    # initialize
    configuration_length = 5
    self._direct = direct_lib.Direct(
        model=model, configuration_length=configuration_length
    )

    ## process
    in_process = np.random.rand(model.nv)
    noise = self._direct.noise(process=in_process)
    self.assertLess(np.linalg.norm(in_process - noise["process"]), 1.0e-5)

    ## sensor
    in_sensor = np.random.rand(model.nsensor)
    noise = self._direct.noise(sensor=in_sensor)
    self.assertLess(np.linalg.norm(in_sensor - noise["sensor"]), 1.0e-5)

  def test_reset(self):
    # load model
    model_path = (
        pathlib.Path(__file__).parent.parent.parent
        / "mjpc/test/testdata/estimator/particle/task.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))

    # initialize
    configuration_length = 5
    self._direct = direct_lib.Direct(
        model=model, configuration_length=configuration_length
    )

    # set
    index = 1
    configuration = np.random.rand(model.nq)
    sensor_measurement = np.random.rand(model.nsensordata)
    data = self._direct.data(
        index,
        configuration=configuration,
        sensor_measurement=sensor_measurement,
    )

    # check that elements are set
    self.assertLess(0, np.linalg.norm(data["configuration"]))
    self.assertLess(0, np.linalg.norm(data["sensor_measurement"]))

    # reset
    self._direct.reset()

    # get data
    data = self._direct.data(index)

    # check that elements are reset to zero
    self.assertLess(np.linalg.norm(data["configuration"]), 1.0e-5)
    self.assertLess(np.linalg.norm(data["sensor_measurement"]), 1.0e-5)

  def test_optimize(self):
    # load model
    model_path = (
        pathlib.Path(__file__).parent.parent.parent
        / "mjpc/test/testdata/estimator/particle/task.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))

    # initialize
    configuration_length = 5
    self._direct = direct_lib.Direct(
        model=model, configuration_length=configuration_length
    )

    # TODO(taylor): setup

    # optimize
    self._direct.optimize()

  def test_status(self):
    # load model
    model_path = (
        pathlib.Path(__file__).parent.parent.parent
        / "mjpc/test/testdata/estimator/particle/task.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))

    # initialize
    configuration_length = 5
    self._direct = direct_lib.Direct(
        model=model, configuration_length=configuration_length
    )

    # TODO(taylor): setup
    status = self._direct.status()

    # search iterations
    self.assertEqual(status["search_iterations"], 0)

    # smoother iterations
    self.assertEqual(status["smoother_iterations"], 0)

    # step size
    self.assertLess(np.abs(status["step_size"] - 1.0), 1.0e-5)

    # # regularization
    # self.assertTrue(
    #     np.abs(
    #         status["regularization"]
    #         - self._direct.settings()["regularization_initial"]
    #     ),
    #     1.0e-6,
    # )

    # gradient norm
    self.assertLess(np.abs(status["gradient_norm"]), 1.0e-5)

    # search direction norm
    self.assertLess(np.abs(status["search_direction_norm"]), 1.0e-5)

    # solve status
    self.assertTrue(status["solve_status"] == 0)

    # cost difference
    self.assertLess(np.abs(status["cost_difference"]), 1.0e-5)

  def test_sensor_info(self):
    # load model
    model_path = (
        pathlib.Path(__file__).parent.parent.parent
        / "mjpc/test/testdata/estimator/particle/task.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))

    # initialize
    configuration_length = 5
    self._direct = direct_lib.Direct(
        model=model, configuration_length=configuration_length
    )

    # get sensor info
    info = self._direct.sensor_info()

    # test
    self.assertEqual(info["start_index"], 0)
    self.assertEqual(info["num_measurements"], 5)
    self.assertEqual(info["dim_measurements"], 7)

  def test_parameters(self):
    # load model
    model_path = (
        pathlib.Path(__file__).parent.parent.parent
        / "mjpc/test/testdata/estimator/particle/task1D_framepos.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))

    # initialize
    configuration_length = 3
    self._direct = direct_lib.Direct(
        model=model, configuration_length=configuration_length
    )

    # random parameters
    parameters = np.random.normal(size=6, scale=1.0e-1)

    # set / get data
    data = self._direct.data(0, parameters=parameters)

    # test
    self.assertLess(np.linalg.norm(data["parameters"] - parameters), 1.0e-5)

    # noise
    noise = np.random.normal(size=6, scale=1.0)
    data = self._direct.noise(parameter=noise)

    # test
    self.assertLess(np.linalg.norm(data["parameter"] - noise), 1.0e-5)


if __name__ == "__main__":
  absltest.main()
