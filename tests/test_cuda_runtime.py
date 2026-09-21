from __future__ import annotations

from cellonaut.release_checks.cuda_runtime import validate_versions


def test_cuda_release_versions_match():
    assert validate_versions("2.13.0+cu126", "0.28.0+cu126", "12.6") is None


def test_cuda_release_versions_report_mismatch():
    problem = validate_versions("2.13.0+cpu", "0.28.0+cpu", None)

    assert problem is not None
    assert "Expected CUDA 12.6" in problem
    assert "torch=2.13.0+cpu" in problem
