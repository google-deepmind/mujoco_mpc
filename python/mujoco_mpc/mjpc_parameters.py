"""Dataclass for MJPC task parameters and cost weights."""

import dataclasses
from typing import Optional, Union

import numpy as np


@dataclasses.dataclass(frozen=True)
class Pose:
  pos: np.ndarray | None  # 3D vector
  quat: np.ndarray | None  # Unit quaternion


@dataclasses.dataclass(frozen=True)
class MjpcParameters:
  """Dataclass to store and set task settings."""
  mode: Optional[str] = None
  task_parameters: dict[str, Union[str, float]] = dataclasses.field(
      default_factory=dict
  )
  cost_weights: dict[str, float] = dataclasses.field(default_factory=dict)
  # A map from body name to pose
  mocap: dict[str, Pose] = dataclasses.field(default_factory=dict)
