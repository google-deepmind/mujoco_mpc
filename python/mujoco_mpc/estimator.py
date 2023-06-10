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

"""Python interface for the to interface with Estimator."""

import atexit
import pathlib
import subprocess
import tempfile
from typing import Literal, Optional

import grpc
import mujoco
from numpy import typing as npt

from utilities import find_free_port

# INTERNAL IMPORT
from mujoco_mpc.proto import estimator_pb2
from mujoco_mpc.proto import estimator_pb2_grpc


class Estimator:
  """`Estimator` class to interface with MuJoCo MPC estimator.

  Attributes:
    port:
    channel:
    stub:
    server_process:
  """

  def __init__(
      self,
      model: Optional[mujoco.MjModel] = None,
      configuration_length: int = 3,
      server_binary_path: Optional[str] = None,
  ):

    if server_binary_path is None:
      binary_name = "estimator_server"
      server_binary_path = pathlib.Path(__file__).parent / "mjpc" / binary_name
    self.port = find_free_port()
    self.server_process = subprocess.Popen(
        [str(server_binary_path), f"--mjpc_port={self.port}"]
    )
    atexit.register(self.server_process.kill)

    credentials = grpc.local_channel_credentials(grpc.LocalConnectionType.LOCAL_TCP)
    self.channel = grpc.secure_channel(f"localhost:{self.port}", credentials)
    grpc.channel_ready_future(self.channel).result(timeout=10)
    self.stub = estimator_pb2_grpc.EstimatorStub(self.channel)
    self.init(
        model,
        configuration_length,
        send_as="xml",
    )

  def close(self):
    self.channel.close()
    self.server_process.kill()
    self.server_process.wait()

  def init(
      self,
      model: Optional[mujoco.MjModel] = None,
      configuration_length: int = 3,
      send_as: Literal["xml"] = "xml",
  ):
    """Initialize the estimator for estimation horizon `configuration_length`.

    Args:
      model: optional `MjModel` instance, which, if provided, will be used as
        the underlying model for planning. If not provided, the default MJPC
        task xml will be used.
      configuration_length: estimation horizon.
      send_as: The serialization format for sending the model over gRPC; "xml".
    """

    def model_to_xml(model: mujoco.MjModel) -> str:
      tmp = tempfile.NamedTemporaryFile()
      mujoco.mj_saveLastXML(tmp.name, model)
      with pathlib.Path(tmp.name).open("rt") as f:
        xml_string = f.read()
      return xml_string

    if model is not None:
      model_message = estimator_pb2.MjModel(xml=model_to_xml(model))
    else:
      print("Failed to find xml.")
      model_message = None

    init_request = estimator_pb2.InitRequest(
        model=model_message,
        configuration_length=configuration_length,
    )
    self.stub.Init(init_request)

  def set_configuration(self, configuration: npt.ArrayLike, index: int):
    request = estimator_pb2.SetDataRequest(
        configuration=configuration, index=index
    )
    self.stub.SetData(request)

  def get_configuration(self, index: int) -> npt.ArrayLike:
    request = estimator_pb2.GetDataRequest(index=index, configuration=True)
    return self.stub.GetData(request).configuration

  def set_velocity(self, velocity: npt.ArrayLike, index: int):
    request = estimator_pb2.SetDataRequest(velocity=velocity, index=index)
    self.stub.SetData(request)

  def get_velocity(self, index: int) -> npt.ArrayLike:
    request = estimator_pb2.GetDataRequest(index=index, velocity=True)
    return self.stub.GetData(request).velocity

  def set_acceleration(self, acceleration: npt.ArrayLike, index: int):
    request = estimator_pb2.SetDataRequest(acceleration=acceleration, index=index)
    self.stub.SetData(request)

  def get_acceleration(self, index: int) -> npt.ArrayLike:
    request = estimator_pb2.GetDataRequest(index=index, acceleration=True)
    return self.stub.GetData(request).acceleration

  def set_action(self, action: npt.ArrayLike, index: int):
    request = estimator_pb2.SetDataRequest(action=action, index=index)
    self.stub.SetData(request)

  def get_action(self, index: int) -> npt.ArrayLike:
    request = estimator_pb2.GetDataRequest(index=index, action=True)
    return self.stub.GetData(request).action

  def set_time(self, time: npt.ArrayLike, index: int):
    request = estimator_pb2.SetDataRequest(time=time, index=index)
    self.stub.SetData(request)

  def get_time(self, index: int) -> npt.ArrayLike:
    request = estimator_pb2.GetDataRequest(index=index, time=True)
    return self.stub.GetData(request).time

  def set_configuration_prior(self, configuration_prior: npt.ArrayLike, index: int):
    request = estimator_pb2.SetDataRequest(
        configuration_prior=configuration_prior, index=index
    )
    self.stub.SetData(request)

  def get_configuration_prior(self, index: int) -> npt.ArrayLike:
    request = estimator_pb2.GetDataRequest(index=index, configuration_prior=True)
    return self.stub.GetData(request).configuration_prior

  def set_sensor_measurement(self, sensor_measurement: npt.ArrayLike, index: int):
    request = estimator_pb2.SetDataRequest(
        sensor_measurement=sensor_measurement, index=index
    )
    self.stub.SetData(request)

  def get_sensor_measurement(self, index: int) -> npt.ArrayLike:
    request = estimator_pb2.GetDataRequest(index=index, sensor_measurement=True)
    return self.stub.GetData(request).sensor_measurement

  def set_sensor_prediction(self, sensor_prediction: npt.ArrayLike, index: int):
    request = estimator_pb2.SetDataRequest(
        sensor_prediction=sensor_prediction, index=index
    )
    self.stub.SetData(request)

  def get_sensor_prediction(self, index: int) -> npt.ArrayLike:
    request = estimator_pb2.GetDataRequest(index=index, sensor_prediction=True)
    return self.stub.GetData(request).sensor_prediction

  def set_force_measurement(self, force_measurement: npt.ArrayLike, index: int):
    request = estimator_pb2.SetDataRequest(
        force_measurement=force_measurement, index=index
    )
    self.stub.SetData(request)

  def get_force_measurement(self, index: int) -> npt.ArrayLike:
    request = estimator_pb2.GetDataRequest(index=index, force_measurement=True)
    return self.stub.GetData(request).force_measurement

  def set_force_prediction(self, force_prediction: npt.ArrayLike, index: int):
    request = estimator_pb2.SetDataRequest(
        force_prediction=force_prediction, index=index
    )
    self.stub.SetData(request)

  def get_force_prediction(self, index: int) -> npt.ArrayLike:
    request = estimator_pb2.GetDataRequest(index=index, force_prediction=True)
    return self.stub.GetData(request).force_prediction

  def set_configuration_length(self, configuration_length: int):
    request = estimator_pb2.SetSettingsRequest(
        configuration_length=configuration_length
    )
    self.stub.SetSettings(request)

  def get_configuration_length(self) -> int:
    request = estimator_pb2.GetSettingsRequest(configuration_length=True)
    return self.stub.GetSettings(request).configuration_length

  def set_search_type(self, search_type: int):
    request = estimator_pb2.SetSettingsRequest(search_type=search_type)
    self.stub.SetSettings(request)

  def get_search_type(self) -> int:
    request = estimator_pb2.GetSettingsRequest(search_type=True)
    return self.stub.GetSettings(request).search_type

  def set_prior_flag(self, flag: bool):
    request = estimator_pb2.SetSettingsRequest(prior_flag=flag)
    self.stub.SetSettings(request)

  def get_prior_flag(self) -> bool:
    request = estimator_pb2.GetSettingsRequest(prior_flag=True)
    return self.stub.GetSettings(request).prior_flag

  def set_sensor_flag(self, flag: bool):
    request = estimator_pb2.SetSettingsRequest(sensor_flag=flag)
    self.stub.SetSettings(request)

  def get_sensor_flag(self) -> bool:
    request = estimator_pb2.GetSettingsRequest(sensor_flag=True)
    return self.stub.GetSettings(request).sensor_flag

  def set_force_flag(self, flag: bool):
    request = estimator_pb2.SetSettingsRequest(force_flag=flag)
    self.stub.SetSettings(request)

  def get_force_flag(self) -> bool:
    request = estimator_pb2.GetSettingsRequest(force_flag=True)
    return self.stub.GetSettings(request).force_flag

  def set_smoother_iterations(self, iterations: int):
    request = estimator_pb2.SetSettingsRequest(smoother_iterations=iterations)
    self.stub.SetSettings(request)

  def get_smoother_iterations(self) -> int:
    request = estimator_pb2.GetSettingsRequest(smoother_iterations=True)
    return self.stub.GetSettings(request).smoother_iterations

  def get_cost(self) -> float:
    request = estimator_pb2.GetCostsRequest(cost=True)
    return self.stub.GetCosts(request).cost

  def get_cost_prior(self) -> float:
    request = estimator_pb2.GetCostsRequest(prior=True)
    return self.stub.GetCosts(request).prior

  def get_cost_sensor(self) -> float:
    request = estimator_pb2.GetCostsRequest(sensor=True)
    return self.stub.GetCosts(request).sensor

  def get_cost_force(self) -> float:
    request = estimator_pb2.GetCostsRequest(force=True)
    return self.stub.GetCosts(request).force

  def get_cost_initial(self) -> float:
    request = estimator_pb2.GetCostsRequest(initial=True)
    return self.stub.GetCosts(request).initial

  def set_prior_weight(self, weight: float):
    request = estimator_pb2.SetWeightsRequest(prior=weight)
    self.stub.SetWeights(request)

  def get_prior_weight(self) -> float:
    request = estimator_pb2.GetWeightsRequest(prior=True)
    return self.stub.GetWeights(request).prior

  def set_sensor_weight(self, weight: npt.ArrayLike):
    request = estimator_pb2.SetWeightsRequest(sensor=weight)
    self.stub.SetWeights(request)

  def get_sensor_weight(self) -> npt.ArrayLike:
    request = estimator_pb2.GetWeightsRequest(sensor=True)
    return self.stub.GetWeights(request).sensor

  def set_force_weight(self, weight: npt.ArrayLike):
    request = estimator_pb2.SetWeightsRequest(force=weight)
    self.stub.SetWeights(request)

  def get_force_weight(self) -> npt.ArrayLike:
    request = estimator_pb2.GetWeightsRequest(force=True)
    return self.stub.GetWeights(request).force

  def shift_trajectory(self, shift: int) -> int:
    request = estimator_pb2.ShiftTrajectoriesRequest(shift=shift)
    return self.stub.ShiftTrajectories(request).head

  def reset(self):
    request = estimator_pb2.ResetRequest()
    self.stub.Reset(request)

  def optimize(self):
    request = estimator_pb2.OptimizeRequest()
    self.stub.Optimize(request)

  def search_iterations(self) -> int:
    request = estimator_pb2.GetStatusRequest(search_iterations=True)
    return self.stub.GetStatus(request).search_iterations

  def smoother_iterations(self) -> int:
    request = estimator_pb2.GetStatusRequest(smoother_iterations=True)
    return self.stub.GetStatus(request).smoother_iterations

  def step_size(self) -> float:
    request = estimator_pb2.GetStatusRequest(step_size=True)
    return self.stub.GetStatus(request).step_size

  def regularization(self) -> float:
    request = estimator_pb2.GetStatusRequest(regularization=True)
    return self.stub.GetStatus(request).regularization

  def gradient_norm(self) -> float:
    request = estimator_pb2.GetStatusRequest(gradient_norm=True)
    return self.stub.GetStatus(request).gradient_norm
