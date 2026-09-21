"""Create a standard SHA-256 sidecar file for a release artifact."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def _sha256(artifact: Path) -> str:
    digest = hashlib.sha256()
    with artifact.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_sha256(artifact: Path, output: Path | None = None) -> Path:
    """Hash *artifact* and write ``<digest>  <filename>`` beside it."""
    destination = output or artifact.with_name(artifact.name + ".sha256")
    with destination.open("w", encoding="ascii", newline="\n") as target:
        target.write(f"{_sha256(artifact)}  {artifact.name}\n")
    return destination


def write_sha256_manifest(artifacts: list[Path], output: Path) -> Path:
    """Write one deterministic checksum manifest for a multipart release."""
    files = sorted((Path(path) for path in artifacts), key=lambda path: path.name.casefold())
    if not files:
        raise ValueError("At least one release artifact is required.")
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Release artifacts not found: {', '.join(missing)}")
    with output.open("w", encoding="ascii", newline="\n") as target:
        target.writelines(f"{_sha256(path)}  {path.name}\n" for path in files)
    return output


def main(argv: list[str] | None = None) -> int:
    """Write a SHA-256 sidecar or multipart release manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", type=Path)
    input_group.add_argument("--input-pattern", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.input_pattern is not None:
        if args.output is None:
            parser.error("--output is required with --input-pattern")
        pattern = args.input_pattern
        output_resolved = args.output.resolve()
        artifacts = [
            path
            for path in pattern.parent.glob(pattern.name)
            if path.is_file() and path.resolve() != output_resolved
        ]
        output = write_sha256_manifest(artifacts, args.output)
    else:
        assert args.input is not None
        output = write_sha256(args.input, args.output)
    print(f"SHA-256: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
