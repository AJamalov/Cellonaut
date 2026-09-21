"""Shared color choices for preview layers and generated mask overlays."""

from __future__ import annotations


PREVIEW_COLOR_PALETTE: tuple[tuple[str, str], ...] = (
    ("White", "#FFFFFF"),
    ("Gray", "#B0B0B0"),
    ("Magenta", "#FF00FF"),
    ("Blue", "#0000FF"),
    ("Cyan", "#00FFFF"),
    ("Green", "#00FF00"),
    ("Yellow", "#FFFF00"),
    ("Orange", "#FFA500"),
    ("Red", "#FF0000"),
    ("Pink", "#FFC0CB"),
    ("Purple", "#8000FF"),
)

DEFAULT_PREVIEW_LAYER_COLORS: tuple[str, ...] = (
    "#FFFFFF",
    "#FF00FF",
    "#0000FF",
    "#00FF00",
    "#FF0000",
    "#00FFFF",
    "#FFFF00",
    "#FFA500",
    "#FFC0CB",
    "#8000FF",
    "#B0B0B0",
)

ADJUSTED_MASK_COLOR_PRIORITIES: tuple[str, ...] = (
    "#FF0000",
    "#8000FF",
    "#00FFFF",
    "#FFFF00",
    "#FFA500",
    "#FFC0CB",
    "#00FF00",
    "#0000FF",
    "#FFFFFF",
    "#B0B0B0",
)

NEUTRAL_PREVIEW_COLORS = frozenset({"", "#FFFFFF", "#B0B0B0"})
DEFAULT_PREVIEW_COLOR = "#FFFFFF"
