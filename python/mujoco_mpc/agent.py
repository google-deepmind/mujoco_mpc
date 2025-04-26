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
import contextlib
import pathlib
import re
import socket
import subprocess
import tempfile
import time
import threading
from typing import Any, Callable, Dict, List, Literal, Mapping, Optional, Sequence, Tuple, Union

import grpc
import mujoco
from mujoco_mpc import mjpc_parameters
import numpy as np
from numpy import typing as npt

# INTERNAL IMPORT
from mujoco_mpc.proto import agent_pb2
from mujoco_mpc.proto import agent_pb2_grpc


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


def parse_port(server_addr: str) -> int:
  """Parse a port from a string.

  Args:
    server_addr: server address

  Returns:
    int: the port
  """
  match = re.search(r":(\d+)", server_addr)
  if match is None:
    raise ValueError(f"Port not specified in server address: '{server_addr}'")
  return int(match.group(1))


class Agent(contextlib.AbstractContextManager):
  """`Agent` class to interface with MuJoCo MPC agents.

  Attributes:
    task_id:
    model:
    port:
    channel:
    stub:
    server_process:
    server_addr:
  """

  def __init__(
      self,
      task_id: str,
      model: Optional[mujoco.MjModel] = None,
      server_binary_path: Optional[str] = None,
      extra_flags: Sequence[str] = (),
      real_time_speed: float = 1.0,
      subprocess_kwargs: Optional[Mapping[str, Any]] = None,
      connect_to: Optional[str] = None,
      run_init: bool = True,
      auto_reconnect: bool = False,
      reconnect_attempts: int = 3,
      reconnect_delay: float = 2.0,
  ):
    self.task_id = task_id
    self.model = model
    self.port = (
        find_free_port() if connect_to is None else parse_port(connect_to)
    )
    self.auto_reconnect = auto_reconnect
    self.reconnect_attempts = reconnect_attempts
    self.reconnect_delay = reconnect_delay
    self._server_binary_path = server_binary_path
    self._extra_flags = extra_flags
    self._subprocess_kwargs = subprocess_kwargs or {}
    self._real_time_speed = real_time_speed
    self._is_connected = False
    self._last_plan_time = 0.0
    self._planning_metrics = {"avg_step_time": 0.0, "last_plan_duration": 0.0}
    self._callback_registry = {}

    if server_binary_path is None:
      binary_name = "agent_server"
      self._server_binary_path = str(pathlib.Path(__file__).parent / "mjpc" / binary_name)

    self.server_process = None
    if connect_to is None:
      self._start_server()
    else:
      self.server_addr = connect_to

    self._establish_connection()

    if run_init:
      self.init(
          task_id,
          model,
          send_as="mjb",
          real_time_speed=real_time_speed,
      )

  def _start_server(self):
    self.server_process = subprocess.Popen(
        [str(self._server_binary_path), f"--mjpc_port={self.port}"]
        + list(self._extra_flags),
        **self._subprocess_kwargs,
    )
    atexit.register(self.server_process.kill)
    self.server_addr = f"localhost:{self.port}"

  def _establish_connection(self):
    credentials = grpc.local_channel_credentials(grpc.LocalConnectionType.LOCAL_TCP)
    self.channel = grpc.secure_channel(self.server_addr, credentials)
    try:
      grpc.channel_ready_future(self.channel).result(timeout=30)
      self.stub = agent_pb2_grpc.AgentStub(self.channel)
      self._is_connected = True
    except grpc.FutureTimeoutError:
      self._is_connected = False
      raise ConnectionError(f"Failed to connect to server at {self.server_addr}")

  def _try_reconnect(self):
    if not self.auto_reconnect:
      return False
      
    for attempt in range(self.reconnect_attempts):
      self.channel.close()
      if self.server_process and not self.server_process.poll():
        self.server_process.kill()
        self.server_process.wait()
        
      time.sleep(self.reconnect_delay)
      
      try:
        self._start_server()
        self._establish_connection()
        return True
      except (ConnectionError, grpc.RpcError):
        continue
    
    return False

  def __exit__(self, exc_type, exc_value, traceback):
    self.close()

  def close(self):
    self._is_connected = False
    self.channel.close()

    if self.server_process is not None:
      self.server_process.kill()
      self.server_process.wait()

  def init(
      self,
      task_id: str,
      model: Optional[mujoco.MjModel] = None,
      send_as: Literal["mjb", "xml"] = "xml",
      real_time_speed: float = 1.0,
  ):
    """Initialize the agent for task `task_id`.

    Args:
      task_id: the identifier for the MuJoCo MPC task, for example "Cartpole" or
        "Humanoid Track".
      model: optional `MjModel` instance, which, if provided, will be used as
        the underlying model for planning. If not provided, the default MJPC
        task xml will be used.
      send_as: The serialization format for sending the model over gRPC. Either
        "mjb" or "xml".
      real_time_speed: ratio of running speed to wall clock, from 0 to 1. Only
        affects async (UI) binaries, and not ones where planning is
        synchronous.
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

    init_request = agent_pb2.InitRequest(
        task_id=task_id, model=model_message, real_time_speed=real_time_speed
    )
    
    try:
      self.stub.Init(init_request)
    except grpc.RpcError as e:
      if self._try_reconnect():
        self.stub.Init(init_request)
      else:
        raise e

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
    # if mocap_pos is an ndarray rather than a list, flatten it
    if hasattr(mocap_pos, "flatten"):
      mocap_pos = mocap_pos.flatten()
    if hasattr(mocap_quat, "flatten"):
      mocap_quat = mocap_quat.flatten()

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
    
    try:
      self.stub.SetState(set_state_request)
    except grpc.RpcError as e:
      if self._try_reconnect():
        self.stub.SetState(set_state_request)
      else:
        raise e

  def get_state(self) -> agent_pb2.State:
    try:
      return self.stub.GetState(agent_pb2.GetStateRequest()).state
    except grpc.RpcError as e:
      if self._try_reconnect():
        return self.stub.GetState(agent_pb2.GetStateRequest()).state
      else:
        raise e

  def get_action(
      self,
      time: Optional[float] = None,
      averaging_duration: float = 0,
      nominal_action: bool = False,
  ) -> np.ndarray:
    """Return latest `action` from the `Agent`'s planner.

    Args:
      time: `data.time`, i.e. the simulation time.
      averaging_duration: the duration over which actions should be averaged
        (e.g. the control timestep).
      nominal_action: if True, don't apply feedback terms in the policy

    Returns:
      action: `Agent`'s planner's latest action.
    """
    get_action_request = agent_pb2.GetActionRequest(
        time=time,
        averaging_duration=averaging_duration,
        nominal_action=nominal_action,
    )
    
    try:
      get_action_response = self.stub.GetAction(get_action_request)
      return np.array(get_action_response.action)
    except grpc.RpcError as e:
      if self._try_reconnect():
        get_action_response = self.stub.GetAction(get_action_request)
        return np.array(get_action_response.action)
      else:
        raise e

  def get_total_cost(self) -> float:
    try:
      terms = self.stub.GetCostValuesAndWeights(
          agent_pb2.GetCostValuesAndWeightsRequest()
      )
      total_cost = 0
      for _, value_weight in terms.values_weights.items():
        total_cost += value_weight.weight * value_weight.value
      return total_cost
    except grpc.RpcError as e:
      if self._try_reconnect():
        terms = self.stub.GetCostValuesAndWeights(
            agent_pb2.GetCostValuesAndWeightsRequest()
        )
        total_cost = 0
        for _, value_weight in terms.values_weights.items():
          total_cost += value_weight.weight * value_weight.value
        return total_cost
      else:
        raise e

  def get_cost_term_values(self) -> dict[str, float]:
    try:
      terms = self.stub.GetCostValuesAndWeights(
          agent_pb2.GetCostValuesAndWeightsRequest()
      )
      return {
          name: value_weight.value
          for name, value_weight in terms.values_weights.items()
      }
    except grpc.RpcError as e:
      if self._try_reconnect():
        terms = self.stub.GetCostValuesAndWeights(
            agent_pb2.GetCostValuesAndWeightsRequest()
        )
        return {
            name: value_weight.value
            for name, value_weight in terms.values_weights.items()
        }
      else:
        raise e

  def get_residuals(self) -> dict[str, Sequence[float]]:
    try:
      residuals = self.stub.GetResiduals(agent_pb2.GetResidualsRequest())
      return {name: residual.values
              for name, residual in residuals.values.items()}
    except grpc.RpcError as e:
      if self._try_reconnect():
        residuals = self.stub.GetResiduals(agent_pb2.GetResidualsRequest())
        return {name: residual.values
                for name, residual in residuals.values.items()}
      else:
        raise e

  def planner_step(self):
    """Send a planner request."""
    start_time = time.time()
    try:
      planner_step_request = agent_pb2.PlannerStepRequest()
      self.stub.PlannerStep(planner_step_request)
      end_time = time.time()
      self._planning_metrics["last_plan_duration"] = end_time - start_time
      
      # Update average step time with exponential moving average
      alpha = 0.2  # Smoothing factor
      self._planning_metrics["avg_step_time"] = (
          alpha * self._planning_metrics["last_plan_duration"] + 
          (1 - alpha) * self._planning_metrics["avg_step_time"]
      )
      self._last_plan_time = end_time
      
      # Run any registered callbacks
      if "after_plan" in self._callback_registry:
        for callback in self._callback_registry["after_plan"]:
          callback(self)
    except grpc.RpcError as e:
      if self._try_reconnect():
        planner_step_request = agent_pb2.PlannerStepRequest()
        self.stub.PlannerStep(planner_step_request)
      else:
        raise e

  def step(self):
    """Step the physics on the agent side."""
    try:
      self.stub.Step(agent_pb2.StepRequest())
    except grpc.RpcError as e:
      if self._try_reconnect():
        self.stub.Step(agent_pb2.StepRequest())
      else:
        raise e

  def reset(self):
    """Reset the `Agent`'s data, settings, planner, and states."""
    try:
      reset_request = agent_pb2.ResetRequest()
      self.stub.Reset(reset_request)
    except grpc.RpcError as e:
      if self._try_reconnect():
        reset_request = agent_pb2.ResetRequest()
        self.stub.Reset(reset_request)
      else:
        raise e

  def set_task_parameter(self, name: str, value: float):
    """Set the `Agent`'s task parameters.

    Args:
      name: the name to identify the parameter.
      value: value to to set the parameter to.
    """
    self.set_task_parameters({name: value})

  def set_task_parameters(self, parameters: dict[str, float | str]):
    """Sets the `Agent`'s task parameters.

    Args:
      parameters: a map from parameter name to value. string values will be
          treated as "selection" values, i.e. parameters with names that start
          with "residual_select_" in the XML.
    """
    request = agent_pb2.SetTaskParametersRequest()
    for name, value in parameters.items():
      if isinstance(value, str):
        request.parameters[name].selection = value
      else:
        request.parameters[name].numeric = value
    
    try:
      self.stub.SetTaskParameters(request)
    except grpc.RpcError as e:
      if self._try_reconnect():
        self.stub.SetTaskParameters(request)
      else:
        raise e

  def get_task_parameters(self) -> dict[str, float | str]:
    """Returns the agent's task parameters."""
    try:
      response = self.stub.GetTaskParameters(agent_pb2.GetTaskParametersRequest())
      result = {}
      for name, value in response.parameters.items():
        if value.selection:
          result[name] = value.selection
        else:
          result[name] = value.numeric
      return result
    except grpc.RpcError as e:
      if self._try_reconnect():
        response = self.stub.GetTaskParameters(agent_pb2.GetTaskParametersRequest())
        result = {}
        for name, value in response.parameters.items():
          if value.selection:
            result[name] = value.selection
          else:
            result[name] = value.numeric
        return result
      else:
        raise e

  def set_cost_weights(
      self, weights: dict[str, float], reset_to_defaults: bool = False
  ):
    """Sets the agent's cost weights by name.

    Args:
      weights: a map for cost term name to weight value
      reset_to_defaults: if true, cost weights will be reset before applying the
        map
    """
    request = agent_pb2.SetCostWeightsRequest(
        cost_weights=weights, reset_to_defaults=reset_to_defaults
    )
    
    try:
      self.stub.SetCostWeights(request)
    except grpc.RpcError as e:
      if self._try_reconnect():
        self.stub.SetCostWeights(request)
      else:
        raise e

  def get_cost_weights(self) -> dict[str, float]:
    """Returns the agent's cost weights."""
    try:
      terms = self.stub.GetCostValuesAndWeights(
          agent_pb2.GetCostValuesAndWeightsRequest()
      )
      return {
          name: value_weight.weight
          for name, value_weight in terms.values_weights.items()
      }
    except grpc.RpcError as e:
      if self._try_reconnect():
        terms = self.stub.GetCostValuesAndWeights(
            agent_pb2.GetCostValuesAndWeightsRequest()
        )
        return {
            name: value_weight.weight
            for name, value_weight in terms.values_weights.items()
        }
      else:
        raise e

  def get_mode(self) -> str:
    try:
      return self.stub.GetMode(agent_pb2.GetModeRequest()).mode
    except grpc.RpcError as e:
      if self._try_reconnect():
        return self.stub.GetMode(agent_pb2.GetModeRequest()).mode
      else:
        raise e

  def set_mode(self, mode: str):
    try:
      request = agent_pb2.SetModeRequest(mode=mode)
      self.stub.SetMode(request)
    except grpc.RpcError as e:
      if self._try_reconnect():
        request = agent_pb2.SetModeRequest(mode=mode)
        self.stub.SetMode(request)
      else:
        raise e

  def get_all_modes(self) -> Sequence[str]:
    try:
      return self.stub.GetAllModes(agent_pb2.GetAllModesRequest()).mode_names
    except grpc.RpcError as e:
      if self._try_reconnect():
        return self.stub.GetAllModes(agent_pb2.GetAllModesRequest()).mode_names
      else:
        raise e

  def set_parameters(self, parameters: mjpc_parameters.MjpcParameters):
    # TODO(nimrod): Add a single RPC that does this
    if parameters.mode is not None:
      self.set_mode(parameters.mode)
    if parameters.task_parameters:
      self.set_task_parameters(parameters.task_parameters)
    if parameters.cost_weights:
      self.set_cost_weights(parameters.cost_weights)

  def best_trajectory(self):
    try:
      request = agent_pb2.GetBestTrajectoryRequest()
      response = self.stub.GetBestTrajectory(request)
      if self.model is None:
        raise ValueError("model is None")
      else:
        return {
            "states": np.array(response.states).reshape(
                response.steps,
                self.model.nq + self.model.nv + self.model.na,
            ),
            "actions": np.array(response.actions).reshape(
                response.steps - 1, self.model.nu
            ),
            "times": np.array(response.times),
        }
    except grpc.RpcError as e:
      if self._try_reconnect():
        request = agent_pb2.GetBestTrajectoryRequest()
        response = self.stub.GetBestTrajectory(request)
        if self.model is None:
          raise ValueError("model is None")
        else:
          return {
              "states": np.array(response.states).reshape(
                  response.steps,
                  self.model.nq + self.model.nv + self.model.na,
              ),
              "actions": np.array(response.actions).reshape(
                  response.steps - 1, self.model.nu
              ),
              "times": np.array(response.times),
          }
      else:
        raise e

  def set_mocap(self, mocap_map: Mapping[str, mjpc_parameters.Pose]):
    try:
      request = agent_pb2.SetAnythingRequest()
      for key, value in mocap_map.items():
        if value.pos is not None:
          request.mocap[key].pos.extend(value.pos)
        if value.quat is not None:
          request.mocap[key].quat.extend(value.quat)
      self.stub.SetAnything(request)
    except grpc.RpcError as e:
      if self._try_reconnect():
        request = agent_pb2.SetAnythingRequest()
        for key, value in mocap_map.items():
          if value.pos is not None:
            request.mocap[key].pos.extend(value.pos)
          if value.quat is not None:
            request.mocap[key].quat.extend(value.quat)
        self.stub.SetAnything(request)
      else:
        raise e
        
  def get_planning_metrics(self) -> Dict[str, float]:
    """Return metrics about the planning process."""
    return self._planning_metrics.copy()
    
  def register_callback(self, event_type: str, callback: Callable[["Agent"], None]):
    """Register a callback function to be called on specific events.
    
    Args:
      event_type: The type of event to trigger the callback. Currently supported:
                  "after_plan" - called after each planner_step() completes
      callback: A function that takes this Agent instance as its only argument
    """
    if event_type not in self._callback_registry:
      self._callback_registry[event_type] = []
    self._callback_registry[event_type].append(callback)
    
  def unregister_callback(self, event_type: str, callback: Callable[["Agent"], None]):
    """Remove a previously registered callback function.
    
    Args:
      event_type: The event type the callback was registered for
      callback: The callback function to remove
    """
    if event_type in self._callback_registry:
      if callback in self._callback_registry[event_type]:
        self._callback_registry[event_type].remove(callback)
        
  def run_planning_loop(self, iterations: int, async_mode: bool = False):
    """Run the planner for a specified number of iterations.
    
    Args:
      iterations: Number of planning steps to run
      async_mode: If True, run planning in a background thread
    
    Returns:
      If async_mode is False, returns a dict of planning metrics
      If async_mode is True, returns a thread object
    """
    def _planning_loop():
      for _ in range(iterations):
        self.planner_step()
        
    if async_mode:
      thread = threading.Thread(target=_planning_loop)
      thread.start()
      return thread
    else:
      _planning_loop()
      return self.get_planning_metrics()
      
  def save_model(self, filepath: str):
    """Save the current model to a file.
    
    Args:
      filepath: Path where to save the model
    """
    if self.model is None:
      raise ValueError("No model available to save")
      
    mujoco.mj_saveModel(self.model, filepath, None)
    
  def is_connected(self) -> bool:
    """Check if the connection to the server is active.
    
    Returns:
      True if connected, False otherwise
    """
    return self._is_connected
