from __future__ import annotations

from pathlib import Path

import pytest

from cellonaut.release_checks import python_runtime


def test_release_python_version_is_read_and_validated(tmp_path: Path):
    (tmp_path / ".python-version").write_text("3.12.10\n", encoding="utf-8")

    assert python_runtime.expected_release_python(tmp_path) == "3.12.10"
    assert python_runtime.validate_release_python(tmp_path, actual="3.12.10") == "3.12.10"


def test_release_python_rejects_wrong_or_invalid_versions(tmp_path: Path):
    version_file = tmp_path / ".python-version"
    version_file.write_text("3.12.10\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="require Python 3.12.10"):
        python_runtime.validate_release_python(tmp_path, actual="3.12.9")

    version_file.write_text("three.twelve\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Invalid release Python version"):
        python_runtime.expected_release_python(tmp_path)


def test_release_python_cli_reports_validation_failure(tmp_path: Path, capsys, monkeypatch):
    (tmp_path / ".python-version").write_text("3.12.10\n", encoding="utf-8")
    monkeypatch.setattr(python_runtime.platform, "python_version", lambda: "3.12.9")

    assert python_runtime.main(["--project-root", str(tmp_path)]) == 1
    assert "Official release builds require Python 3.12.10" in capsys.readouterr().err
