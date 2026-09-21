"""Generate and validate native, hash-locked release dependency sets."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
import tomllib


PIP_VERSION = "26.2.1"
LOCK_CUTOFF = "2026-09-01T00:00:00Z"
REQUIRED_PACKAGES = {
    "cellpose",
    "pip",
    "pip-audit",
    "pyinstaller",
    "pyside6",
    "pytest",
    "setuptools",
    "torch",
    "torchvision",
    "wheel",
}


@dataclass(frozen=True)
class LockProfile:
    system: str
    machine: str
    extra: str
    extra_index: str = ""


LOCK_PROFILES = {
    "windows-standard": LockProfile("Windows", "AMD64", "release"),
    "windows-cuda126": LockProfile(
        "Windows", "AMD64", "release,windows-cuda126", "https://download.pytorch.org/whl/cu126"
    ),
}


def lock_path(output_dir: Path, profile_name: str) -> Path:
    return output_dir / f"pylock.{profile_name}.toml"


def _validate_host(profile_name: str) -> LockProfile:
    profile = LOCK_PROFILES[profile_name]
    current_system = platform.system()
    current_machine = platform.machine()
    if current_system != profile.system or current_machine.lower() != profile.machine.lower():
        raise RuntimeError(
            f"{profile_name} locks require {profile.system} {profile.machine}; "
            f"this host is {current_system} {current_machine}."
        )
    return profile


def validate_lock(path: Path) -> dict[str, object]:
    """Reject incomplete PEP 751 locks or package artifacts without SHA-256 hashes."""
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    if data.get("lock-version") != "1.0" or not data.get("created-by"):
        raise RuntimeError(f"Invalid PEP 751 lock metadata: {path}")
    packages = data.get("packages")
    if not isinstance(packages, list) or not packages:
        raise RuntimeError(f"Dependency lock contains no packages: {path}")
    names: set[str] = set()
    for package in packages:
        if not isinstance(package, dict) or not package.get("name") or not package.get("version"):
            raise RuntimeError(f"Dependency lock contains an unversioned package: {path}")
        names.add(str(package["name"]).lower().replace("_", "-"))
        artifacts = [package.get("archive"), package.get("sdist"), *(package.get("wheels") or [])]
        artifacts = [artifact for artifact in artifacts if isinstance(artifact, dict)]
        if not artifacts or any(
            re.fullmatch(r"[0-9a-fA-F]{64}", str(artifact.get("hashes", {}).get("sha256", ""))) is None
            for artifact in artifacts
        ):
            raise RuntimeError(f"Dependency lock has an artifact without SHA-256: {package['name']}")
    missing = sorted(REQUIRED_PACKAGES - names)
    if missing:
        raise RuntimeError(f"Dependency lock is missing required packages: {', '.join(missing)}")
    return {"packages": len(packages), "names": sorted(names)}


def generate_lock(profile_name: str, output_dir: Path, project_root: Path, python_exe: Path) -> Path:
    """Resolve one native release profile and record every selected artifact hash."""
    profile = _validate_host(profile_name)
    project_root = project_root.resolve()
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output = lock_path(output_dir, profile_name)
    command = [
        str(python_exe),
        "-m",
        "cellonaut.release_checks.pip_lock",
        "--only-deps",
        f".[{profile.extra}]",
        "--output",
        str(output),
        "--no-build-isolation",
        "--no-input",
        "--progress-bar",
        "off",
        "--prefer-binary",
    ]
    if not profile.extra_index:
        command.extend(["--uploaded-prior-to", LOCK_CUTOFF])
    # PyTorch's indexes omit upload dates; hash-lock those profiles without a date filter.
    if profile.extra_index:
        command.extend(["--extra-index-url", profile.extra_index])
    subprocess.run(command, cwd=project_root, check=True)
    summary = validate_lock(output)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    metadata = {
        "profile": profile_name,
        "python": platform.python_version(),
        "system": platform.system(),
        "machine": platform.machine(),
        "cutoff": None if profile.extra_index else LOCK_CUTOFF,
        "cutoff_reason": "Index lacks upload-time metadata" if profile.extra_index else "",
        "pip": PIP_VERSION,
        "lock_sha256": digest,
        **summary,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=tuple(LOCK_PROFILES))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    args = parser.parse_args(argv)
    output = generate_lock(args.profile, args.output_dir, args.project_root, args.python)
    print(f"Dependency lock: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
