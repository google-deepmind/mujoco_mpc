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

"""Python interface for interface with Estimator."""

import atexit
import pathlib
import socket
import subprocess
import tempfile
from typing import Literal, Optional

import grpc
import mujoco
import numpy as np
from numpy import typing as npt

# INTERNAL IMPORT
from mujoco_mpc.proto import estimator_pb2
from mujoco_mpc.proto import estimator_pb2_grpc


def find_free_port() -> int:
  """Find an available TCP port on the system.

    This function creates a temporary socket, binds it to an available port
    chosen by the operating system, and returns the chosen port number.

  Returns:
      int: An available TCP port number.
  """
  with socket.socket(family=socket.AF_INET6) as s:
    s.bind(("", 0))
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    return s.getsockname()[1]


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

  def data(
      self,
      index: int,
      configuration: Optional[npt.ArrayLike] = [],
      velocity: Optional[npt.ArrayLike] = [],
      acceleration: Optional[npt.ArrayLike] = [],
      action: Optional[npt.ArrayLike] = [],
      time: Optional[npt.ArrayLike] = [],
      configuration_prior: Optional[npt.ArrayLike] = [],
      sensor_measurement: Optional[npt.ArrayLike] = [],
      sensor_prediction: Optional[npt.ArrayLike] = [],
      force_measurement: Optional[npt.ArrayLike] = [],
      force_prediction: Optional[npt.ArrayLike] = [],
  ) -> dict[str, np.ndarray]:
    inputs = estimator_pb2.Data(
        configuration=configuration,
        velocity=velocity,
        acceleration=acceleration,
        time=time,
        configuration_prior=configuration_prior,
        sensor_measurement=sensor_measurement,
        sensor_prediction=sensor_prediction,
        force_measurement=force_measurement,
        force_prediction=force_prediction,
    )
    request = estimator_pb2.DataRequest(data=inputs, index=index)
    data = self.stub.Data(request).data

    return {
        "configuration": np.array(data.configuration),
        "velocity": np.array(data.velocity),
        "acceleration": np.array(data.acceleration),
        "time": np.array(data.time),
        "configuration_prior": np.array(data.configuration_prior),
        "sensor_measurement": np.array(data.sensor_measurement),
        "sensor_prediction": np.array(data.sensor_prediction),
        "force_measurement": np.array(data.force_measurement),
        "force_prediction": np.array(data.force_prediction),
    }

  def settings(
      self,
      configuration_length: Optional[int] = None,
      search_type: Optional[int] = None,
      prior_flag: Optional[bool] = None,
      sensor_flag: Optional[bool] = None,
      force_flag: Optional[bool] = None,
      smoother_iterations: Optional[int] = None,
      skip_prior_weight_update: Optional[bool] = None,
  ) -> dict[str, int | bool]:
    inputs = estimator_pb2.Settings(
        configuration_length=configuration_length,
        search_type=search_type,
        prior_flag=prior_flag,
        sensor_flag=sensor_flag,
        force_flag=force_flag,
        smoother_iterations=smoother_iterations,
        skip_prior_weight_update=skip_prior_weight_update,
    )
    request = estimator_pb2.SettingsRequest(
        settings=inputs,
    )
    settings = self.stub.Settings(request).settings

    return {
        "configuration_length": settings.configuration_length,
        "search_type": settings.search_type,
        "prior_flag": settings.prior_flag,
        "sensor_flag": settings.sensor_flag,
        "force_flag": settings.force_flag,
        "smoother_iterations": settings.smoother_iterations,
        "skip_prior_weight_update": settings.skip_prior_weight_update,
    }

  def weight(
      self,
      prior: Optional[float] = None,
      sensor: Optional[npt.ArrayLike] = [],
      force: Optional[npt.ArrayLike] = [],
  ) -> dict[str, float | np.ndarray]:
    inputs = estimator_pb2.Weight(
        prior=prior,
        sensor=sensor,
        force=force,
    )
    request = estimator_pb2.WeightsRequest(weight=inputs)
    weight = self.stub.Weights(request).weight

    return {
        "prior": weight.prior,
        "sensor": np.array(weight.sensor),
        "force": np.array(weight.force),
    }

  def cost(self) -> dict[str, float]:
    request = estimator_pb2.CostRequest()
    cost = self.stub.Cost(request).cost
    return {
        "total": cost.total,
        "prior": cost.prior,
        "sensor": cost.sensor,
        "force": cost.force,
        "initial": cost.initial,
    }

  def status(self) -> dict[str, int]:
    request = estimator_pb2.StatusRequest()
    status = self.stub.Status(request).status
    return {
        "search_iterations": status.search_iterations,
        "smoother_iterations": status.smoother_iterations,
        "step_size": status.step_size,
        "regularization": status.regularization,
        "gradient_norm": status.gradient_norm,
    }

  def shift(self, shift: int) -> int:
    request = estimator_pb2.ShiftRequest(shift=shift)
    return self.stub.Shift(request).head

  def reset(self):
    request = estimator_pb2.ResetRequest()
    self.stub.Reset(request)

  def optimize(self):
    request = estimator_pb2.OptimizeRequest()
    self.stub.Optimize(request)

  def cost_hessian(self) -> np.ndarray:
    request = estimator_pb2.CostHessianRequest()
    response = self.stub.CostHessian(request)
    hessian = np.array(response.hessian).reshape(response.dimension, response.dimension)
    return hessian

  def prior_matrix(self, prior: Optional[npt.ArrayLike] = None) -> np.ndarray:
    request = estimator_pb2.PriorMatrixRequest(
        prior=prior.flatten() if prior is not None else None
    )
    response = self.stub.PriorMatrix(request)
    mat = np.array(response.prior).reshape(response.dimension, response.dimension)
    return mat
