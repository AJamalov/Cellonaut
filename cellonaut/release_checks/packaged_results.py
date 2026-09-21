"""Validate the JSON written by a packaged Cellonaut release check."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def validate_packaged_result(
    result_path: Path,
    expected_version: str,
    *,
    require_cuda_check: bool = False,
) -> list[str]:
    """Return release-check problems rather than relying on Python assertions."""
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return [f"Could not read packaged check result: {exc}"]

    if not isinstance(result, dict):
        return ["Packaged check result must be a JSON object."]

    problems = []
    if result.get("app_version") != expected_version:
        problems.append(f"Expected app version {expected_version}, found {result.get('app_version')!r}.")

    failures = result.get("failures")
    if not isinstance(failures, list):
        problems.append("Packaged check result has no valid failures list.")
    elif failures:
        problems.append(f"Packaged check reported failures: {failures!r}")

    cuda_status = result.get("cuda_inference")
    if require_cuda_check and (not isinstance(cuda_status, str) or cuda_status not in {"passed", "no_compatible_gpu"}):
        problems.append(
            "Windows CUDA check did not run successfully "
            f"(status: {cuda_status!r})."
        )
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--app-version", required=True)
    parser.add_argument("--require-cuda-check", action="store_true")
    args = parser.parse_args(argv)

    problems = validate_packaged_result(args.input, args.app_version, require_cuda_check=args.require_cuda_check)
    for problem in problems:
        print(f"ERROR: {problem}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
