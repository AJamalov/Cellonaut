"""Thin launcher used by PyInstaller builds."""

from __future__ import annotations

from cellonaut.app import main


if __name__ == "__main__":
    raise SystemExit(main())
