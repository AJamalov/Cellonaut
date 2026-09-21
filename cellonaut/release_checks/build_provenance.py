"""Record and verify the source revision used for a packaged release."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cellonaut.version import __version__


def git_state(project_root: Path) -> tuple[str, bool]:
    """Return the current Git commit and whether tracked/untracked files changed."""
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if not revision:
        raise RuntimeError("Git did not report a source revision.")
    return revision, bool(status.strip())


def write_build_info(output: Path, project_root: Path, profile: str, *, allow_dirty: bool = False) -> dict[str, Any]:
    """Write build provenance, refusing an unreproducible dirty release by default."""
    revision, dirty = git_state(project_root)
    if dirty and not allow_dirty:
        raise RuntimeError(
            "The source tree has uncommitted changes. Commit them before a release build, "
            "or use the explicit development-build override."
        )
    info = {
        "app_version": __version__,
        "build_profile": profile,
        "source_revision": revision,
        "source_dirty": dirty,
        "python_version": platform.python_version(),
        "built_utc": datetime.now(timezone.utc).isoformat(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return info


def verify_build_info(input_path: Path, project_root: Path) -> dict[str, Any]:
    """Ensure a packaged app matches the current clean checkout and version."""
    try:
        info = json.loads(input_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read valid build provenance from {input_path}: {exc}") from exc

    revision, dirty = git_state(project_root)
    problems = []
    if info.get("app_version") != __version__:
        problems.append(f"built version {info.get('app_version')!r} does not match current version {__version__!r}")
    if info.get("source_revision") != revision:
        problems.append("the packaged app was built from a different Git revision")
    if info.get("source_dirty") is not False:
        problems.append("the packaged app was built from uncommitted source")
    if info.get("python_version") != platform.python_version():
        problems.append("the packaged app was built with a different Python version")
    if dirty:
        problems.append("the current source tree has uncommitted changes")
    if problems:
        raise RuntimeError("Stale or non-release build: " + "; ".join(problems) + ".")
    return info


def main(argv: list[str] | None = None) -> int:
    """Write or verify build provenance from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    write_parser = subparsers.add_parser("write", help="Write BUILD_INFO.json.")
    write_parser.add_argument("--output", type=Path, required=True)
    write_parser.add_argument("--project-root", type=Path, default=Path.cwd())
    write_parser.add_argument("--profile", required=True)
    write_parser.add_argument("--allow-dirty", action="store_true")

    verify_parser = subparsers.add_parser("verify", help="Verify BUILD_INFO.json against this checkout.")
    verify_parser.add_argument("--input", type=Path, required=True)
    verify_parser.add_argument("--project-root", type=Path, default=Path.cwd())

    args = parser.parse_args(argv)
    try:
        if args.command == "write":
            write_build_info(args.output, args.project_root, args.profile, allow_dirty=args.allow_dirty)
        else:
            verify_build_info(args.input, args.project_root)
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
