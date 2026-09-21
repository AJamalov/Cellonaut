from __future__ import annotations

import sys
from pathlib import Path

import pytest

from cellonaut.gui import runtime


def test_venv_relaunch_preserves_spaced_script_path(monkeypatch):
    project_root = Path(runtime.__file__).resolve().parents[2]
    venv_python = project_root / ".venv" / "Scripts" / "python.exe"
    script_path = project_root / "cellonaut" / "__main__.py"
    popen_calls = []

    monkeypatch.setattr(sys, "executable", str(project_root / "base python.exe"))
    monkeypatch.setattr(sys, "argv", [str(script_path)])
    monkeypatch.setattr(runtime, "_project_venv_python", lambda _root: venv_python)
    monkeypatch.setattr(
        runtime.subprocess,
        "Popen",
        lambda args, **kwargs: popen_calls.append((args, kwargs)),
    )

    with pytest.raises(SystemExit) as exc_info:
        runtime.relaunch_with_project_venv()

    assert exc_info.value.code == 0
    assert popen_calls[0][0] == [str(venv_python), str(script_path)]
    assert popen_calls[0][1]["cwd"] == str(project_root)
    if sys.platform.startswith("win"):
        assert popen_calls[0][1]["creationflags"] == runtime.subprocess.CREATE_NO_WINDOW  # pyright: ignore[reportAttributeAccessIssue]
        assert popen_calls[0][1]["startupinfo"].dwFlags & runtime.subprocess.STARTF_USESHOWWINDOW  # pyright: ignore[reportAttributeAccessIssue]


def test_project_venv_python_detects_windows_layout(tmp_path):
    venv_python = tmp_path / ".venv" / "Scripts" / "python.exe"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("", encoding="utf-8")

    assert runtime._project_venv_python(tmp_path) == venv_python


def test_rotate_crash_log_keeps_one_bounded_previous_log(tmp_path):
    crash_log = tmp_path / "crash.log"
    previous_log = tmp_path / "crash.previous.log"
    crash_log.write_text("new diagnostic details", encoding="utf-8")
    previous_log.write_text("older details", encoding="utf-8")

    rotated = runtime.rotate_crash_log(crash_log, max_bytes=4)

    assert rotated == previous_log
    assert not crash_log.exists()
    assert previous_log.read_text(encoding="utf-8") == "new diagnostic details"


def test_rotate_crash_log_leaves_small_log_in_place(tmp_path):
    crash_log = tmp_path / "crash.log"
    crash_log.write_text("ok", encoding="utf-8")

    assert runtime.rotate_crash_log(crash_log, max_bytes=4) is None
    assert crash_log.read_text(encoding="utf-8") == "ok"
