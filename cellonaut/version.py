"""Authoritative Cellonaut application version."""

APP_NAME = "Cellonaut"
__version__ = "1.0.1"


# Centralize user-facing version text so the CLI and GUI cannot drift apart.
def format_app_version() -> str:
    return f"{APP_NAME} {__version__}"
