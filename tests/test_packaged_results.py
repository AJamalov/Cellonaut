from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from cellonaut.release_checks.packaged_results import validate_packaged_result


def _write_result(path: Path, **overrides) -> Path:
    result = {"app_version": "1.0.0", "failures": []}
    result.update(overrides)
    path.write_text(json.dumps(result), encoding="utf-8")
    return path


def test_packaged_result_accepts_clean_smoke_check(tmp_path: Path):
    result = _write_result(tmp_path / "smoke.json")

    assert validate_packaged_result(result, "1.0.0") == []


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"app_version": "0.9.0"}, "Expected app version"),
        ({"failures": None}, "no valid failures list"),
        ({"failures": ["Fiji failed"]}, "Fiji failed"),
    ],
)
def test_packaged_result_rejects_bad_results(tmp_path: Path, overrides, expected):
    result = _write_result(tmp_path / "bad.json", **overrides)

    assert expected in " ".join(validate_packaged_result(result, "1.0.0"))


@pytest.mark.parametrize("status", ["passed", "no_compatible_gpu"])
def test_windows_cuda_result_accepts_checked_status(tmp_path: Path, status):
    result = _write_result(tmp_path / "runtime.json", cuda_inference=status)

    assert validate_packaged_result(result, "1.0.0", require_cuda_check=True) == []


@pytest.mark.parametrize("status", [None, "not_applicable", "failed", []])
def test_windows_cuda_result_rejects_missing_or_skipped_check(tmp_path: Path, status):
    result = _write_result(tmp_path / "runtime.json", cuda_inference=status)

    problems = validate_packaged_result(result, "1.0.0", require_cuda_check=True)

    assert any("Windows CUDA check did not run successfully" in problem for problem in problems)


def test_packaged_result_cli_fails_under_optimized_python(tmp_path: Path):
    result = _write_result(tmp_path / "runtime.json", cuda_inference="not_applicable")

    completed = subprocess.run(
        [
            sys.executable,
            "-O",
            "-m",
            "cellonaut.release_checks.packaged_results",
            "--input",
            str(result),
            "--app-version",
            "1.0.0",
            "--require-cuda-check",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 1
    assert "Windows CUDA check did not run successfully" in completed.stdout
