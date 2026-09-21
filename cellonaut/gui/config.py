"""GUI-to-pipeline configuration and machine-local startup settings."""

from __future__ import annotations

from pathlib import Path
from PySide6.QtWidgets import QApplication
from cellonaut.config.adapter import build_pipeline_config_from_gui_state, parse_cell_qc_limits_text
from cellonaut.config.preset_store import (
    private_path_tail as private_path_tail,
    privacy_safe_preset as privacy_safe_preset,
)
from cellonaut.config.state import CellonautGuiState
from cellonaut.gui.presets import CellonautGuiPresetMixin
from cellonaut.resources import (
    SETTINGS_FILE,
    get_bundled_fiji_path,
    get_internal_fiji_path,
)
from cellonaut.gui.theme import apply_theme


class CellonautGuiConfigMixin(CellonautGuiPresetMixin):
    """Translate GUI state into pipeline configuration and startup preferences."""

    # Keep one parser implementation in the configuration adapter while
    # exposing it through the mixin API used by dialogs.
    def parse_cell_qc_limits_text(self, text: str) -> dict:
        return parse_cell_qc_limits_text(text)

    def collect_gui_state(self) -> CellonautGuiState:
        """Read a detached snapshot; callers commit pending edits explicitly."""
        return self.configuration_state.snapshot()

    def collect_config(self):
        """Commit once, then build execution data independent of live widgets."""
        self.commit_gui_edits(require_ready_input=True)
        return build_pipeline_config_from_gui_state(self.collect_gui_state())

    # Last-session state contains machine preferences only; dataset choices
    # belong to named presets and should not reopen automatically.
    def get_startup_state_dict(self):
        recent_images = []
        seen_paths: set[str] = set()
        for value in getattr(self, "_recent_preview_images", []):
            path = str(value or "").strip()
            if not path or path in seen_paths or not Path(path).is_file():
                continue
            seen_paths.add(path)
            recent_images.append(path)
            if len(recent_images) == 10:
                break

        return {
            "appearance_mode": self.appearance_mode.get(),
            "recent_preview_images": recent_images,
        }

    # Apply only machine-local choices here so loading last-session state cannot
    # disturb the workflow that will be restored from the Default preset.
    def apply_startup_state_dict(self, settings: dict):
        if "appearance_mode" in settings:
            self.appearance_mode.set(settings["appearance_mode"])
            app = QApplication.instance()
            if isinstance(app, QApplication):
                apply_theme(app, self.appearance_mode.get())

        if hasattr(self, "fiji_app"):
            bundled_fiji = get_bundled_fiji_path()
            internal_fiji = get_internal_fiji_path()
            saved_fiji = str(settings.get("fiji_app_path", "") or "").strip()
            if bundled_fiji is not None:
                # Official packages always use their tested offline runtime;
                # stale settings must never redirect them to another Fiji.
                self.fiji_app.set(str(bundled_fiji))
            elif internal_fiji is not None:
                self.fiji_app.set(str(internal_fiji))
            elif saved_fiji and Path(saved_fiji).is_dir():
                self.fiji_app.set(saved_fiji)
            else:
                self.fiji_app.set(saved_fiji)

        recent_images = settings.get("recent_preview_images", [])
        restored: list[str] = []
        seen_paths: set[str] = set()
        if isinstance(recent_images, list):
            for value in recent_images:
                if not isinstance(value, str):
                    continue
                path = value.strip()
                if not path or path in seen_paths or not Path(path).is_file():
                    continue
                seen_paths.add(path)
                restored.append(path)
                if len(restored) == 10:
                    break
        self._recent_preview_images = restored
        if hasattr(self, "preview_recent_images_button"):
            self.preview_recent_images_button.setEnabled(bool(restored))

    # Last-session settings are optional convenience data; corruption should be
    # logged and ignored rather than preventing the main window from opening.
    def load_last_settings_if_available(self):
        path = Path(SETTINGS_FILE)
        if not path.exists():
            return

        try:
            settings = self._read_json_file(path)
            self.apply_startup_state_dict(settings)
            self.log(f"[SETTINGS] Loaded last settings: {path}")
        except Exception as e:
            self.log(f"[SETTINGS][WARN] Could not load last settings: {e}")

    # Persist only startup preferences here so closing one dataset does not make
    # it the implicit input for the next application session.
    def save_last_settings(self):
        try:
            self._write_json_file(Path(SETTINGS_FILE), self.get_startup_state_dict())
        except Exception as e:
            self.log(f"Could not save last settings: {e}")
