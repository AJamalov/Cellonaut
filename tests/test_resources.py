from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from cellonaut.resources import APP_ICON_FILE, PACKAGE_DATA_ROOT, get_bundled_fiji_path, get_resource_path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_source_resource_paths_find_package_assets():
    assert get_resource_path(APP_ICON_FILE).name == "icon.ico"
    assert get_resource_path(APP_ICON_FILE).exists()


def test_packaged_resource_paths_find_bundled_assets():
    assert (PACKAGE_DATA_ROOT / APP_ICON_FILE).exists()
    assert (PACKAGE_DATA_ROOT / "presets" / "Default.json").exists()


def user_config_env(tmp_path):
    env = os.environ.copy()
    env["APPDATA"] = str(tmp_path)
    return env


def expected_user_app_dir(tmp_path):
    return tmp_path / "Cellonaut"


def test_resource_import_does_not_create_user_app_dir(tmp_path):
    env = user_config_env(tmp_path)

    result = subprocess.run(
        [sys.executable, "-c", "import cellonaut.resources"],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert not expected_user_app_dir(tmp_path).exists()


def test_ensure_user_app_dirs_creates_user_app_dirs(tmp_path):
    env = user_config_env(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from cellonaut.resources import ensure_user_app_dirs; ensure_user_app_dirs()",
        ],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    user_app_dir = expected_user_app_dir(tmp_path)
    assert user_app_dir.is_dir()
    assert (user_app_dir / "presets").is_dir()


def test_platform_user_config_dir_uses_windows_appdata(tmp_path, monkeypatch):
    import cellonaut.resources as resources

    monkeypatch.setenv("APPDATA", str(tmp_path))

    assert resources.platform_user_config_dir() == tmp_path / "Cellonaut"


def test_resource_base_dirs_prefers_pyinstaller_meipass(tmp_path, monkeypatch):
    import cellonaut.resources as resources

    bundle_root = tmp_path / "bundle"
    bundle_root.mkdir()
    monkeypatch.setattr(resources.sys, "_MEIPASS", str(bundle_root), raising=False)
    monkeypatch.setattr(resources.sys, "frozen", True, raising=False)
    monkeypatch.setattr(resources.sys, "executable", str(tmp_path / "Cellonaut"))

    assert resources.resource_base_dirs()[0] == bundle_root


def test_get_resource_path_finds_bundled_file_from_meipass(tmp_path, monkeypatch):
    import cellonaut.resources as resources

    bundle_root = tmp_path / "bundle"
    asset = bundle_root / "assets" / "icon.ico"
    asset.parent.mkdir(parents=True)
    asset.write_text("icon", encoding="utf-8")
    monkeypatch.setattr(resources.sys, "_MEIPASS", str(bundle_root), raising=False)

    assert resources.get_resource_path("assets/icon.ico") == asset


def test_bundled_fiji_path_uses_offline_meipass_asset(tmp_path, monkeypatch):
    import cellonaut.resources as resources

    bundle_root = tmp_path / "bundle"
    fiji = bundle_root / "offline" / "Fiji.app"
    fiji.mkdir(parents=True)
    monkeypatch.setattr(resources.sys, "_MEIPASS", str(bundle_root), raising=False)

    assert get_bundled_fiji_path() == fiji
