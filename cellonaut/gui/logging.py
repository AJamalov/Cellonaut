"""Buffered, theme-aware output for the live GUI Log tab."""

from __future__ import annotations

import html
import sys

from PySide6.QtWidgets import QApplication

from cellonaut.gui.mixin import GuiMixin

MAX_LOG_BLOCKS = 10_000

LOG_LINE_STYLES: dict[str, tuple[str, int]] = {
    "default": ("#d4d4d4", 400),
    "error": ("#ff6b6b", 700),
    "warning": ("#ffd166", 700),
    "preview": ("#7bdff2", 600),
    "nd2": ("#b2f7ef", 600),
    "qc": ("#cdb4db", 600),
    "weka": ("#f4a261", 600),
    "cellpose": ("#ffafcc", 600),
    "export": ("#a7c957", 600),
    "success": ("#80ed99", 700),
    "progress": ("#90caf9", 600),
}

LIGHT_LOG_LINE_STYLES: dict[str, tuple[str, int]] = {
    "default": ("#243244", 400),
    "error": ("#b42318", 700),
    "warning": ("#8a570d", 700),
    "preview": ("#096b83", 600),
    "nd2": ("#087267", 600),
    "qc": ("#6d3a8c", 600),
    "weka": ("#9a4f0b", 600),
    "cellpose": ("#a62f68", 600),
    "export": ("#4d7418", 600),
    "success": ("#237a38", 700),
    "progress": ("#1769b0", 600),
}


def current_log_line_styles() -> dict[str, tuple[str, int]]:
    """Choose readable log colors for the active light or dark palette."""
    app = QApplication.instance()
    theme_name = str(app.property("cellonautTheme") or "") if isinstance(app, QApplication) else ""
    return LIGHT_LOG_LINE_STYLES if theme_name == "light_blue" else LOG_LINE_STYLES


# Route every live message through one formatter so severity colors, buffering,
# and native worker fragments remain consistent across all operations.
class CellonautGuiLoggingMixin(GuiMixin):
    """Render bounded, severity-aware pipeline logs in the GUI."""

    # Check severity before subsystem tags so a failed Weka, Cellpose, preview,
    # or ND2 operation is always shown as an error rather than routine progress.
    def log_line_kind(self, text: str) -> str:
        upper = str(text).upper()
        if (
            "[ERROR]" in upper
            or " ERROR]" in upper
            or upper.startswith("ERROR:")
            or "TRACEBACK" in upper
            or "EXCEPTION" in upper
        ):
            return "error"
        if "[WARN" in upper or "WARNING" in upper or "COULD NOT" in upper:
            return "warning"
        if "FAILED" in upper or "FAILURE" in upper:
            return "error"
        if "[PREVIEW]" in upper:
            return "preview"
        if "[ND2" in upper:
            return "nd2"
        if "[QC" in upper:
            return "qc"
        if "[WEKA" in upper or "CLASSIFIER" in upper:
            return "weka"
        if "[CELLPOSE" in upper or "CELL SEGMENT" in upper:
            return "cellpose"
        if any(token in upper for token in ("[EXPORT", "SAVED", "WROTE")):
            return "export"
        if any(token in upper for token in ("FINISHED", "DONE", "SUCCESS")):
            return "success"
        if any(
            token in upper
            for token in (
                "RUNNING",
                "PREPARING",
                "LOADING",
                "[RUN]",
                "[SCAN]",
                "[AUTO]",
                "[SAMPLE ",
                "[TARGET]",
            )
        ):
            return "progress"
        return "default"

    # Use inline colors because QTextDocument class rules can be overridden by
    # the application stylesheet on real platform widgets even when isolated
    # QTextEdit tests appear correct.
    def colorize_log_line(self, text: str) -> str:
        kind = self.log_line_kind(text)
        color, weight = current_log_line_styles()[kind]
        escaped = html.escape(str(text), quote=False)
        return (
            f'<span class="log-{kind}" style="color:{color}; font-weight:{weight}; '
            f'font-family:Consolas, monospace; white-space:pre-wrap;">{escaped}</span>'
        )

    # Keep document rules as a fallback for text restored from rich HTML while
    # live messages carry inline colors that cannot be lost to Qt style precedence.
    def refresh_log_theme(self) -> None:
        if not hasattr(self, "log_box"):
            return
        rules = []
        for kind, (color, weight) in current_log_line_styles().items():
            rules.append(
                f".log-{kind} {{ color: {color}; font-weight: {weight}; "
                "font-family: Consolas, monospace; white-space: pre-wrap; }}"
            )
        self.log_box.document().setDefaultStyleSheet("\n".join(rules))
        self.log_box.viewport().update()

    # Limit stored log lines to keep memory use manageable during long runs.
    def configure_log_box(self) -> None:
        self.log_box.document().setMaximumBlockCount(MAX_LOG_BLOCKS)
        self.refresh_log_theme()

    # Respect a user's position when they scroll up to inspect an earlier event;
    # only follow new output when the view was already at the bottom.
    def append_colored_log_lines(self, lines: list[str]) -> None:
        if not lines:
            return

        scrollbar = self.log_box.verticalScrollBar()
        was_at_bottom = scrollbar.value() >= scrollbar.maximum() - 2
        rendered = "<br>".join(self.colorize_log_line(line) for line in lines)
        self.log_box.append(rendered)

        if was_at_bottom:
            scrollbar.setValue(scrollbar.maximum())

    # Buffer messages so the log does not redraw for every update.
    def log(self, text: str) -> None:
        self._log_buffer.append(str(text))

    # Display the current batch, leaving new messages queued.
    # Write to stderr if display fails.
    def flush_log_buffer(self) -> None:
        if not self._log_buffer or not hasattr(self, "log_box"):
            return

        lines = self._log_buffer[:]
        self._log_buffer.clear()

        try:
            self.append_colored_log_lines(lines)
        except Exception as exc:
            sys.stderr.write(f"[GUI LOG ERROR] Could not display buffered messages: {exc}\n")
            sys.stderr.write("\n".join(lines) + "\n")
