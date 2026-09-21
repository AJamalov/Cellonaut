"""Central presentation tokens for the Cellonaut desktop interface."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SpacingTokens:
    xs: int = 4
    sm: int = 8
    md: int = 12


@dataclass(frozen=True)
class ControlTokens:
    radius: int = 5


@dataclass(frozen=True)
class TypographyTokens:
    caption_pt: float = 8.5
    body_pt: float = 9.0
    label_pt: float = 9.5
    workspace_title_pt: float = 10.0
    section_title_pt: float = 13.0
    brand_pt: float = 13.0
    medium_weight: int = 500
    semibold_weight: int = 600
    bold_weight: int = 700


@dataclass(frozen=True)
class WorkspaceTokens:
    sidebar_width: int = 120
    header_height: int = 60
    pipeline_row_height: int = 72
    preview_tool_strip_width: int = 64
    preview_tool_button_size: int = 36
    preview_bottom_bar_height: int = 42
    inspector_min_width: int = 190
    inspector_max_width: int = 220
    status_bar_height: int = 28


SPACING = SpacingTokens()
CONTROLS = ControlTokens()
TYPOGRAPHY = TypographyTokens()
WORKSPACE = WorkspaceTokens()
