"""Shared application exceptions."""

from enum import Enum


class PipelineCancelled(Exception):
    """Raised when a user requests cooperative pipeline cancellation."""


class SetupErrorCode(str, Enum):
    OVERLAY_MASK = "overlay_mask"
    WHOLE_CELL_OVERLAY = "whole_cell_overlay"
    CELL_SOURCE = "cell_source"
    PER_CELL_MASK = "per_cell_mask"
    MEASURED_CHANNEL = "measured_channel"
    OVERLAY_BASE = "overlay_base"
    NO_SAMPLES = "no_samples"
    CLASSIFIER_MISSING = "classifier_missing"
    INPUT_MISSING = "input_missing"
    FIJI_MISSING = "fiji_missing"
    ND2_INPUT = "nd2_input"
    ND2_CHANNELS = "nd2_channels"
    ND2_OUTPUT = "nd2_output"
    ND2_BACKEND = "nd2_backend"


class SetupError(ValueError):
    """Application-owned validation failure with wording-independent guidance."""

    def __init__(self, code: SetupErrorCode, message: str):
        super().__init__(message)
        self.code = code

    def __reduce__(self):
        return type(self), (self.code, str(self))


class SetupFileNotFoundError(SetupError, FileNotFoundError):
    """Keep the established FileNotFoundError contract for missing paths."""


class SetupImportError(SetupError, ImportError):
    """Keep the established ImportError contract for missing optional backends."""
