"""Work around pip 26.2.1's metadata-only locking of unhashed artifacts."""

from __future__ import annotations

import sys
from typing import Any, cast


def main(argv: list[str] | None = None) -> int:
    import pip
    from pip._internal.cli.main import main as pip_main
    from pip._internal.operations.prepare import RequirementPreparer

    from cellonaut.release_checks.dependency_lock import PIP_VERSION

    if pip.__version__ != PIP_VERSION:
        raise RuntimeError(f"Release locking requires pip {PIP_VERSION}; found {pip.__version__}.")

    original = RequirementPreparer._fetch_metadata_only

    def fetch_metadata(self, req):
        # Without an artifact hash, pip must read the archive rather than just its sidecar.
        if req.link is not None and not req.link.has_hash:
            return None
        return original(self, req)

    preparer_type = cast(Any, RequirementPreparer)
    preparer_type._fetch_metadata_only = fetch_metadata
    try:
        return pip_main(["lock", *(sys.argv[1:] if argv is None else argv)])
    finally:
        preparer_type._fetch_metadata_only = original


if __name__ == "__main__":
    raise SystemExit(main())
