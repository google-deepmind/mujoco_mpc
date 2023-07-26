"""Dataclass for MJPC task parameters and cost weights."""

import dataclasses
from typing import Optional, Union


@dataclasses.dataclass(frozen=True)
class MjpcParameters:
  """Dataclass to store and set task mode, task parameters and cost weights."""
  mode: Optional[str] = None
  task_parameters: dict[str, Union[str, float]] = dataclasses.field(
      default_factory=dict
  )
  cost_weights: dict[str, float] = dataclasses.field(default_factory=dict)
