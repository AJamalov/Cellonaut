"""Shared pipeline cancellation checks."""

from __future__ import annotations

from collections.abc import Callable

from cellonaut.exceptions import PipelineCancelled


# One checkpoint implementation gives every backend the same
# terminal outcome instead of allowing cancellation to be counted as failure.
def check_cancel(
    should_cancel: Callable[[], bool] | None,
    *,
    message: str = "Pipeline cancelled by user.",
) -> None:
    if should_cancel is not None and should_cancel():
        raise PipelineCancelled(message)
