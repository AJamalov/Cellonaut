"""Explicit run stages and the runtime facts consumed by run summaries."""

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import TypedDict


class PipelineStage(str, Enum):
    PREPARING = "preparing"
    INITIALIZING = "initializing"
    RUNNING = "running"
    STAGING = "staging"
    LOADING = "loading"
    MASKS = "masks"
    SEGMENTATION = "segmentation"
    MEASUREMENT = "measurement"
    EXPORT = "export"
    SCANNING = "scanning"
    CHECKING = "checking"
    SUCCESS = "success"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    ERROR = "error"


class StageUpdate(TypedDict):
    stage: str
    text: str


def stage_update(stage: PipelineStage, text: str) -> StageUpdate:
    """Keep the existing process queue payload primitive and serializable."""
    return {"stage": stage.value, "text": text}


@dataclass
class PipelineRuntime:
    """One execution's stage callback and observed successful inference devices."""

    stage_func: Callable[[StageUpdate], None] | None = None
    cellpose_inference_devices: set[str] = field(default_factory=set)

    def stage(self, stage: PipelineStage, text: str) -> None:
        if self.stage_func is not None:
            self.stage_func(stage_update(stage, text))
