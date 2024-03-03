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

"""Python interface for interface with Filter."""

import atexit
import os
import pathlib
import socket
import subprocess
import sys
import tempfile
from typing import Literal, Optional

import grpc
import mujoco
import numpy as np
from numpy import typing as npt

# INTERNAL IMPORT
from mujoco_mpc.proto import filter_pb2
from mujoco_mpc.proto import filter_pb2_grpc


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


class Filter:
  """`Filter` class to interface with MuJoCo MPC filter.

  Attributes:
    port:
    channel:
    stub:
    server_process:
  """

  def __init__(
      self,
      model: mujoco.MjModel,
      server_binary_path: Optional[str] = None,
      send_as: Literal["mjb", "xml"] = "xml",
      colab_logging: bool = True,
  ):
    # server
    if server_binary_path is None:
      binary_name = "filter_server"
      server_binary_path = pathlib.Path(__file__).parent / "mjpc" / binary_name
    self._colab_logging = colab_logging
    self.port = find_free_port()
    self.server_process = subprocess.Popen(
        [str(server_binary_path), f"--mjpc_port={self.port}"],
        stdout=subprocess.PIPE if colab_logging else None,
    )
    os.set_blocking(self.server_process.stdout.fileno(), False)
    atexit.register(self.server_process.kill)

    credentials = grpc.local_channel_credentials(grpc.LocalConnectionType.LOCAL_TCP)
    self.channel = grpc.secure_channel(f"localhost:{self.port}", credentials)
    grpc.channel_ready_future(self.channel).result(timeout=10)
    self.stub = filter_pb2_grpc.StateEstimationStub(self.channel)

    # initialize
    self.init(
        model,
        send_as=send_as,
    )

  def close(self):
    self.channel.close()
    self.server_process.kill()
    self.server_process.wait()

  def init(
      self,
      model: mujoco.MjModel,
      send_as: Literal["mjb", "xml"] = "xml",
  ):
    """
    Args:
      model: optional `MjModel` instance, which, if provided, will be used as
        the underlying model for planning. If not provided, the default MJPC
        task xml will be used.
      configuration_length: estimation horizon.
      send_as: The serialization format for sending the model over gRPC; "xml".
    """

    # setup model
    def model_to_mjb(model: mujoco.MjModel) -> bytes:
      buffer_size = mujoco.mj_sizeModel(model)
      buffer = np.empty(shape=buffer_size, dtype=np.uint8)
      mujoco.mj_saveModel(model, None, buffer)
      return buffer.tobytes()

    def model_to_xml(model: mujoco.MjModel) -> str:
      tmp = tempfile.NamedTemporaryFile()
      mujoco.mj_saveLastXML(tmp.name, model)
      with pathlib.Path(tmp.name).open("rt") as f:
        xml_string = f.read()
      return xml_string

    if model is not None:
      if send_as == "mjb":
        model_message = filter_pb2.MjModel(mjb=model_to_mjb(model))
      else:
        model_message = filter_pb2.MjModel(xml=model_to_xml(model))
    else:
      model_message = None

    # initialize request
    init_request = filter_pb2.InitRequest(
        model=model_message,
    )

    # initialize response
    self._wait(self.stub.Init.future(init_request))

  def available_filters(self):
    return {
        "ground truth",
        "extended Kalman filter",
        "unscented Kalman filter",
        "batch filter",
    }

  def reset(self):
    # reset request
    request = filter_pb2.ResetRequest()

    # reset response
    self._wait(self.stub.Reset.future(request))

  def update(
      self,
      ctrl: Optional[npt.ArrayLike] = [],
      sensor: Optional[npt.ArrayLike] = [],
      mode: Optional[int] = 0,
  ):
    # request
    request = filter_pb2.UpdateRequest(
        ctrl=ctrl,
        sensor=sensor,
        mode=mode,
    )

    # response
    self._wait(self.stub.Update.future(request))

  def state(
      self, state: Optional[npt.ArrayLike] = [], time: Optional[float] = None
  ) -> dict[str | float, np.ndarray]:
    # input
    input = filter_pb2.State(state=state, time=time)

    # request
    request = filter_pb2.StateRequest(
        state=input,
    )

    # response
    response = self._wait(self.stub.State.future(request))

    # return state
    return {
        "state": np.array(response.state.state),
        "time": response.state.time,
        "qfrc": np.array(response.state.qfrc),
    }

  def covariance(
      self, covariance: Optional[npt.ArrayLike] = None
  ) -> np.ndarray:
    # input
    inputs = filter_pb2.Covariance(
        covariance=covariance.flatten() if covariance is not None else None,
    )

    # request
    request = filter_pb2.CovarianceRequest(
        covariance=inputs,
    )

    # response
    response = self._wait(self.stub.Covariance.future(request)).covariance

    # return covariance
    return np.array(response.covariance).reshape(
        response.dimension, response.dimension
    )

  def noise(
      self,
      process: Optional[npt.ArrayLike] = [],
      sensor: Optional[npt.ArrayLike] = [],
  ) -> dict[str, np.ndarray]:
    # inputs
    inputs = filter_pb2.Noise(
        process=process,
        sensor=sensor,
    )

    # request
    request = filter_pb2.NoiseRequest(
        noise=inputs,
    )

    # response
    response = self._wait(self.stub.Noise.future(request)).noise

    # return noise
    return {
        "process": np.array(response.process),
        "sensor": np.array(response.sensor),
    }

  def _wait(self, future):
    """Waits for the future to complete, while printing out subprocess stdout."""
    if self._colab_logging:
      while True:
        line = self.server_process.stdout.readline()
        if line:
            sys.stdout.write(line.decode("utf-8"))
        if future.done():
            break
    return future.result()
