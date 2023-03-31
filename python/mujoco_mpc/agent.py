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

"""Python interface for the to interface with MuJoCo MPC agents."""

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

import pathlib
from mujoco_mpc import agent_pb2
from mujoco_mpc import agent_pb2_grpc


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


class Agent:
  """`Agent` class to interface with MuJoCo MPC agents.

  Attributes:
    task_id:
    model:
    port:
    channel:
    stub:
    server_process:
  """

  def __init__(
      self, task_id: str, model: Optional[mujoco.MjModel] = None
  ) -> None:
    self.task_id = task_id
    self.model = model

    binary_name = "agent_service"
    server_binary_path = pathlib.Path(__file__).parent / "mjpc" / binary_name
    self.port = find_free_port()
    self.server_process = subprocess.Popen(
        [str(server_binary_path), f"--port={self.port}"]
    )
    atexit.register(self.server_process.terminate)

    credentials = grpc.local_channel_credentials(grpc.LocalConnectionType.LOCAL_TCP)
    self.channel = grpc.secure_channel(f"localhost:{self.port}", credentials)
    grpc.channel_ready_future(self.channel).result(timeout=10)
    self.stub = agent_pb2_grpc.AgentStub(self.channel)
    self.init(task_id, model)

  def init(
      self,
      task_id: str,
      model: Optional[mujoco.MjModel] = None,
      send_as: Literal["mjb", "xml"] = "mjb",
  ):
    """Initialize the agent for task `task_id`.

    Args:
      task_id: the identifier for the MuJoCo MPC task, for example "Cartpole" or
        "Humanoid Track".
      model: optional `MjModel` instance, which, if provided, will be used as
        the underlying model for planning. If not provided, the default MJPC
        task xml will be used.
      send_as: The serialization format to send the model over gRPC as. Either
        "mjb" or "xml".
    """

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
        model_message = agent_pb2.MjModel(mjb=model_to_mjb(model))
      else:
        model_message = agent_pb2.MjModel(xml=model_to_xml(model))
    else:
      model_message = None

    init_request = agent_pb2.InitRequest(task_id=task_id, model=model_message)
    self.stub.Init(init_request)

  def set_state(
      self,
      time: Optional[float] = None,
      qpos: Optional[npt.ArrayLike] = None,
      qvel: Optional[npt.ArrayLike] = None,
      act: Optional[npt.ArrayLike] = None,
      mocap_pos: Optional[npt.ArrayLike] = None,
      mocap_quat: Optional[npt.ArrayLike] = None,
      userdata: Optional[npt.ArrayLike] = None,
  ):
    """Set `Agent`'s MuJoCo `data` state.

    Args:
      time: `data.time`, i.e. the simulation time.
      qpos: `data.qpos`.
      qvel: `data.qvel`.
      act: `data.act`.
      mocap_pos: `data.mocap_pos`.
      mocap_quat: `data.mocap_quat`.
      userdata: `data.userdata`.
    """
    state = agent_pb2.State(
        time=time if time is not None else None,
        qpos=qpos if qpos is not None else [],
        qvel=qvel if qvel is not None else [],
        act=act if act is not None else [],
        mocap_pos=mocap_pos if mocap_pos is not None else [],
        mocap_quat=mocap_quat if mocap_quat is not None else [],
        userdata=userdata if userdata is not None else [],
    )

    set_state_request = agent_pb2.SetStateRequest(state=state)
    self.stub.SetState(set_state_request)

  def get_action(self, time: Optional[float] = None) -> np.ndarray:
    """Return latest `action` from the `Agent`'s planner.

    Args:
      time: `data.time`, i.e. the simulation time.

    Returns:
      action: `Agent`'s planner's latest action.
    """
    get_action_request = agent_pb2.GetActionRequest(time=time)
    get_action_response = self.stub.GetAction(get_action_request)
    return np.array(get_action_response.action)

  def planner_step(self):
    """Send a planner request."""
    planner_step_request = agent_pb2.PlannerStepRequest()
    self.stub.PlannerStep(planner_step_request)

  def reset(self):
    """Reset the `Agent`'s data, settings, planner, and states."""
    reset_request = agent_pb2.ResetRequest()
    self.stub.Reset(reset_request)

  def set_task_parameter(self, name: str, value: float):
    """Set the `Agent`'s task parameters.

    Args:
      name: the name to identify the parameter.
      value: value to to set the parameter to.
    """
    set_task_parameter_request = agent_pb2.SetTaskParameterRequest(
        name=name, value=value
    )
    self.stub.SetTaskParameter(set_task_parameter_request)
