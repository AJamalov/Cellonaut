from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

from cellonaut.version import format_app_version


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "command",
    [
        [sys.executable, "-m", "cellonaut", "--version"],
        [sys.executable, str(PROJECT_ROOT / "cellonaut" / "__main__.py"), "--version"],
    ],
)
def test_application_entry_points_report_version(command: list[str]):
    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == format_app_version()


def test_windows_app_identity_sets_taskbar_app_id(monkeypatch):
    from cellonaut import app as cellonaut_app
    from cellonaut.version import APP_NAME, __version__

    calls = []
    fake_ctypes = SimpleNamespace(
        windll=SimpleNamespace(
            shell32=SimpleNamespace(
                SetCurrentProcessExplicitAppUserModelID=lambda app_id: calls.append(app_id)
            )
        )
    )
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setitem(sys.modules, "ctypes", fake_ctypes)

    cellonaut_app.configure_windows_app_identity()

    assert calls == [f"Cellonaut.{APP_NAME}.{__version__}"]


def test_direct_entry_point_loads_third_party_packaging():
    entry_point = PROJECT_ROOT / "cellonaut" / "__main__.py"
    script = (
        "import runpy, sys; "
        f"runpy.run_path({str(entry_point)!r}, run_name='entry_point_test'); "
        "import packaging; "
        "print(packaging.__file__)"
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    normalized = result.stdout.strip().lower().replace("\\", "/")
    assert "/site-packages/packaging/__init__.py" in normalized
