from __future__ import annotations

import json
from pathlib import Path

import pytest

from cellonaut.release_checks import dependency_lock


def test_validate_lock_requires_versions_and_sha256_for_release_packages(tmp_path: Path):
    packages = []
    for name in sorted(dependency_lock.REQUIRED_PACKAGES):
        packages.append(
            "\n".join(
                (
                    "[[packages]]",
                    f'name = "{name}"',
                    'version = "1.0"',
                    "[[packages.wheels]]",
                    f'name = "{name}-1.0-py3-none-any.whl"',
                    f'url = "https://example.invalid/{name}.whl"',
                    f'hashes = {{sha256 = "{"a" * 64}"}}',
                )
            )
        )
    lock = tmp_path / "pylock.test.toml"
    lock.write_text('lock-version = "1.0"\ncreated-by = "pip"\n\n' + "\n\n".join(packages) + "\n", encoding="utf-8")

    summary = dependency_lock.validate_lock(lock)

    assert summary["packages"] == len(dependency_lock.REQUIRED_PACKAGES)


def test_validate_lock_rejects_non_hex_sha256(tmp_path: Path):
    packages = []
    for name in sorted(dependency_lock.REQUIRED_PACKAGES):
        digest = "z" * 64 if name == "cellpose" else "a" * 64
        packages.append(
            "\n".join(
                (
                    "[[packages]]",
                    f'name = "{name}"',
                    'version = "1.0"',
                    "[[packages.wheels]]",
                    f'name = "{name}-1.0-py3-none-any.whl"',
                    f'url = "https://example.invalid/{name}.whl"',
                    f'hashes = {{sha256 = "{digest}"}}',
                )
            )
        )
    lock = tmp_path / "pylock.invalid.toml"
    lock.write_text('lock-version = "1.0"\ncreated-by = "pip"\n\n' + "\n\n".join(packages) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="without SHA-256"):
        dependency_lock.validate_lock(lock)


@pytest.mark.parametrize("profile_name", tuple(dependency_lock.LOCK_PROFILES))
def test_generate_lock_uses_native_profile_cutoff_and_writes_metadata(tmp_path: Path, monkeypatch, profile_name):
    profile = dependency_lock.LOCK_PROFILES[profile_name]
    monkeypatch.setattr(dependency_lock.platform, "system", lambda: profile.system)
    monkeypatch.setattr(dependency_lock.platform, "machine", lambda: profile.machine)
    monkeypatch.setattr(dependency_lock, "validate_lock", lambda _path: {"packages": 42, "names": ["torch"]})
    commands = []

    def fake_run(command, **kwargs):
        commands.append((command, kwargs))
        output = Path(command[command.index("--output") + 1])
        output.write_text('lock-version = "1.0"\ncreated-by = "pip"\npackages = []\n', encoding="utf-8")

    monkeypatch.setattr(dependency_lock.subprocess, "run", fake_run)
    output = dependency_lock.generate_lock(profile_name, tmp_path, tmp_path, Path("python"))

    command, kwargs = commands[0]
    assert command[:3] == ["python", "-m", "cellonaut.release_checks.pip_lock"]
    assert "--only-deps" in command
    assert f".[{profile.extra}]" in command
    assert (dependency_lock.LOCK_CUTOFF in command) == (not profile.extra_index)
    if profile.extra_index:
        assert command[-2:] == ["--extra-index-url", profile.extra_index]
    assert kwargs == {"cwd": tmp_path, "check": True}
    metadata = json.loads(output.with_suffix(".metadata.json").read_text(encoding="utf-8"))
    assert metadata["profile"] == profile_name
    assert metadata["cutoff"] == (None if profile.extra_index else dependency_lock.LOCK_CUTOFF)
    assert metadata["packages"] == 42


def test_generate_lock_rejects_the_wrong_operating_system(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(dependency_lock.platform, "system", lambda: "Linux")
    monkeypatch.setattr(dependency_lock.platform, "machine", lambda: "x86_64")

    with pytest.raises(RuntimeError, match="require Windows AMD64"):
        dependency_lock.generate_lock("windows-standard", tmp_path, tmp_path, Path("python"))
