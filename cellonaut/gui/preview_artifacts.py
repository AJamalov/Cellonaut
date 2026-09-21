"""Preview artifact discovery and navigation controls."""

from __future__ import annotations

import re
from pathlib import Path

from cellonaut.gui.mixin import GuiMixin
from cellonaut.results.artifacts import ArtifactResolver, ArtifactMetadataError, artifact_root
from cellonaut.results.layout import build_results_layout, results_root

PREVIEW_ARTIFACT_VIEW_LABELS = {
    "overlay": "Overlay",
    "png": "PNG",
    "montage": "Montage",
    "probability": "Weka probability map",
    "mask_image": "Weka threshold mask",
    "binary_mask": "Final binary mask",
    "snapshot": "Snapshot",
}

PREVIEW_ARTIFACT_SELECTOR_LABELS = {
    "overlay": "TIFF overlays",
    "png": "PNG previews",
    "montage": "Processing montages",
    "probability": "Weka probability maps",
    "mask_image": "Weka threshold masks",
    "binary_mask": "Final binary masks",
    "snapshot": "Snapshots",
}

PREVIEW_OVERLAY_SUFFIXES = (
    "_combined_overlay",
    "_cellpose_outline_stack",
    "_cellpose_outline",
    "_qc_overlay",
    "_qc",
    "_flat_overlay",
    "_overlay",
    "_binary",
)


# Sample selection is separate from artifact selection so moving between
# overlays or montages never resets the currently inspected sample.
class CellonautGuiPreviewArtifactsMixin(GuiMixin):
    # Prefer the result folder tied to the visible file, but include recent
    # previews so navigation still works immediately after a preview run.
    def preview_artifact_results_roots(self) -> list[Path]:
        candidates: list[Path] = []

        output_widget = getattr(self, "output_dir", None)
        output_text = output_widget.get().strip() if output_widget is not None else ""
        output_root = Path(output_text) if output_text else None

        stored = self.preview_state.artifact_results_root
        if stored:
            stored_root = Path(stored)
            if output_root is None or output_root == stored_root or output_root in stored_root.parents:
                candidates.append(stored_root)

        current_file = Path(self.preview_state.file_path or "")
        if current_file.name:
            root = artifact_root(current_file)
            if root is not None and (output_root is None or output_root == root or output_root in root.parents):
                candidates.append(root)

        if output_root is not None:
            candidates.append(results_root(output_root))
            if output_root.exists() and output_root.is_dir():
                preview_children = [child for child in output_root.glob("preview_*") if child.is_dir()]
                preview_children.sort(
                    key=lambda child: child.stat().st_mtime,
                    reverse=True,
                )
                candidates.extend(results_root(child) for child in preview_children[:3])

        unique: list[Path] = []
        seen: set[str] = set()
        for candidate in candidates:
            key = str(candidate.resolve()) if candidate.exists() else str(candidate)
            if key in seen:
                continue
            seen.add(key)
            unique.append(candidate)
        return unique

    # Category patterns are explicit because similarly named TIFFs have very
    # different meanings and should not appear in the same selector view.
    def preview_artifact_mode_paths(self, root: Path, mode: str) -> list[Path]:
        layout = build_results_layout(Path(root), create_root=False)
        searches_by_mode = {
            "overlay": [
                (layout["mask_overlays"], "**/*_combined_overlay.tif"),
                (layout["cell_segmentation_outlines"], "**/*_cellpose_outline_stack.tif"),
                (layout["mask_overlays"], "**/*_overlay.tif"),
            ],
            "png": [
                (layout["qc_overlay_pngs"], "**/*_qc.png"),
                (layout["qc_overlay_pngs"], "**/*_qc_overlay.png"),
                (layout["qc_overlay_pngs"], "**/*_flat_overlay.png"),
                (layout["cell_segmentation_qc_pngs"], "**/*_qc.png"),
                (layout["cell_segmentation_qc_pngs"], "**/*_qc_overlay.png"),
            ],
            "probability": [
                (layout["weka_probability_maps"], "**/*_prob_*.tif"),
            ],
            "mask_image": [
                (layout["weka_threshold_masks"], "**/*_thr_*.tif"),
            ],
            "binary_mask": [
                (layout["final_binary_masks"], "**/*_binary.tif"),
            ],
            "montage": [
                (layout["processing_montages"], "**/*_montage.png"),
            ],
            "snapshot": [
                (layout["snapshots"], "**/*.png"),
                (layout["snapshots"], "**/*.jpg"),
                (layout["snapshots"], "**/*.jpeg"),
                (layout["snapshots"], "**/*.bmp"),
                (layout["snapshots"], "**/*.tif"),
                (layout["snapshots"], "**/*.tiff"),
            ],
        }
        searches = searches_by_mode.get(mode)
        if not searches:
            return []

        files: list[Path] = []
        seen: set[str] = set()
        for search_root, pattern in searches:
            if not search_root.exists():
                continue
            for candidate in sorted(search_root.glob(pattern)):
                if not candidate.exists() or not candidate.is_file():
                    continue
                key = str(candidate)
                if key in seen:
                    continue
                seen.add(key)
                files.append(candidate)
        return files

    # Include configured display names when separating sample IDs from channel
    # suffixes, while retaining common microscopy labels for reopened results.
    def preview_artifact_tail_names(self) -> list[str]:
        names: list[str] = []
        for image_def in list(getattr(self, "image_definitions", []) or []):
            for key in ("name", "folder", "folder_name", "label"):
                value = str(
                    image_def.get(key, "") if isinstance(image_def, dict) else getattr(image_def, key, "") or ""
                ).strip()
                if value:
                    names.append(value)
        names.extend(["GFP", "RFP", "DAPI", "mCherry", "Signal", "Base", "Cells", "Cellpose"])

        safe_names: list[str] = []
        seen: set[str] = set()
        for name in names:
            safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")
            for candidate in (name, safe):
                candidate = str(candidate or "").strip()
                if not candidate:
                    continue
                key = candidate.lower()
                if key in seen:
                    continue
                seen.add(key)
                safe_names.append(candidate)
        return sorted(safe_names, key=len, reverse=True)

    # Derive one stable sample key before grouping; artifact arrows then stay
    # within the selected sample instead of jumping back to the first sample.
    def preview_artifact_sample_key(self, root: Path, path: Path, mode: str) -> str:
        if mode == "snapshot":
            return "Snapshots"
        root = Path(root)
        path = Path(path)
        try:
            relative = path.relative_to(root)
            parts = relative.parts
        except ValueError:
            parts = path.parts

        if mode == "montage" and "Processing Montages" in parts:
            index = parts.index("Processing Montages")
            if len(parts) > index + 2:
                return "/".join(parts[index + 1 : -1])

        parent = path.parent
        flat_artifact_folders = {
            "PNG",
            "TIFF Overlays",
            "TIFF Outlines",
            "ProbabilityMaps",
            "MaskImages",
            "BinaryMasks",
        }
        if parent != root and parent.name not in flat_artifact_folders:
            return str(parent.relative_to(root)) if root in parent.parents else parent.name

        stem = path.stem
        for suffix in PREVIEW_OVERLAY_SUFFIXES:
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        stem = re.sub(r"_prob_[^_]+$", "", stem, flags=re.IGNORECASE)
        stem = re.sub(r"_class\d+_thr_[^_]+$", "", stem, flags=re.IGNORECASE)
        for tail_name in self.preview_artifact_tail_names():
            if stem.lower().endswith(f"_{tail_name}".lower()):
                stem = stem[: -(len(tail_name) + 1)]
                break
        stem = re.sub(r"_(?:image|channel|base|signal|mask|roi)?\d+$", "", stem, flags=re.IGNORECASE)
        return stem or path.stem

    # Stop at the first result root containing the requested category so files
    # from separate runs are never mixed into one sample sequence.
    def collect_preview_artifact_entries(self, mode: str) -> list[dict]:
        entries: list[dict] = []
        for root in self.preview_artifact_results_roots():
            if not root.exists() or not root.is_dir():
                continue
            if mode != "snapshot":
                try:
                    resolver = ArtifactResolver.load(root)
                    if resolver is not None:
                        kinds = {
                            "overlay": {"combined_overlay", "cell_outline_stack"},
                            "png": {"overlay_png", "cell_png"},
                            "probability": {"probability"},
                            "mask_image": {"threshold"},
                            "binary_mask": {"binary_mask"},
                            "montage": {"montage"},
                        }.get(mode, set())
                        grouped_metadata: dict[str, list[str]] = {}
                        for record in resolver.records:
                            if record["kind"] in kinds:
                                grouped_metadata.setdefault(record["sample"], []).append(str(resolver.path(record)))
                        if grouped_metadata:
                            return [
                                {"sample": sample, "path": paths[0], "files": paths, "root": str(root)}
                                for sample, paths in sorted(grouped_metadata.items())
                            ]
                        continue
                except ArtifactMetadataError as exc:
                    log = getattr(self, "log", None)
                    if callable(log):
                        log(f"[PREVIEW][WARN] {exc}")
                    return []
            # Legacy folders and user snapshots retain filename navigation.
            files = self.preview_artifact_mode_paths(root, mode)
            if not files:
                continue
            grouped: dict[str, list[Path]] = {}
            for path in files:
                sample_key = self.preview_artifact_sample_key(root, path, mode)
                grouped.setdefault(sample_key, []).append(path)
            for sample_key in sorted(grouped, key=str.lower):
                sample_files = sorted(grouped[sample_key], key=lambda item: str(item).lower())
                entries.append(
                    {
                        "sample": sample_key,
                        "path": str(sample_files[0]),
                        "files": [str(item) for item in sample_files],
                        "root": str(root),
                    }
                )
            break
        return entries

    def refresh_snapshot_artifact_availability(self):
        selector = getattr(self, "preview_artifact_selector", None)
        if selector is None:
            return
        available = bool(self.collect_preview_artifact_entries("snapshot"))
        for index in range(selector.count()):
            if selector.itemData(index) != "snapshot":
                continue
            item = selector.model().item(index)
            if item is not None:
                item.setEnabled(available)
            break

    # Block signals while synchronizing old buttons and the selector to prevent
    # a programmatic selection from reopening the same artifact recursively.
    def set_preview_artifact_buttons_checked(self, mode: str | None):
        selector = getattr(self, "preview_artifact_selector", None)
        if selector is not None:
            selector.blockSignals(True)
            selector.setCurrentIndex(-1)
            for index in range(selector.count()):
                if selector.itemData(index) == mode:
                    selector.setCurrentIndex(index)
                    break
            selector.blockSignals(False)

        for attr, value in (
            ("preview_overlay_button", "overlay"),
            ("preview_montage_button", "montage"),
        ):
            button = getattr(self, attr, None)
            if button is not None:
                button.blockSignals(True)
                button.setChecked(mode == value)
                button.blockSignals(False)

    # Use short singular labels in counters even though selector entries are
    # plural category names.
    def current_preview_artifact_view_label(self) -> str:
        mode = str(self.preview_state.artifact_mode or "")
        return PREVIEW_ARTIFACT_VIEW_LABELS.get(mode, "Preview")

    # Show only the filename in the limited header space and retain the complete
    # path in the tooltip for disambiguating repeated sample names.
    def set_preview_file_title(self, file_path: str | Path | None):
        label = getattr(self, "preview_file_title_label", None)
        if label is None:
            return
        path_text = str(file_path or "").strip()
        if not path_text:
            label.setText("No preview loaded. Run Preview One Sample or open an image from the Files tab.")
            return
        path = Path(path_text)
        label.setText(path.name)
        if hasattr(label, "setDetailToolTip"):
            label.setDetailToolTip(str(path))
        else:
            label.setToolTip(str(path))

    def clear_preview_artifact_navigation(self) -> None:
        """Discard selections tied to a dataset that is no longer displayed."""
        self.preview_state.artifact_entries = []
        self.preview_state.artifact_mode = ""
        self.preview_state.artifact_index = -1
        self.preview_state.artifact_file_index = -1
        self.preview_state.artifact_results_root = None
        self.set_preview_artifact_buttons_checked(None)
        self.update_preview_artifact_nav_controls()

    # Recompute labels and enabled states together so controls cannot advertise
    # movement beyond the currently selected sample or artifact list.
    def update_preview_artifact_nav_controls(self):
        entries = list(self.preview_state.artifact_entries or [])
        index = int(self.preview_state.artifact_index)
        total = len(entries)
        view_label = self.current_preview_artifact_view_label()
        if total <= 0 or not 0 <= index < total:
            sample_label = "Sample: - / -"
            sample_prev_enabled = False
            sample_next_enabled = False
            artifact_label = f"{view_label}: - / -"
            artifact_prev_enabled = False
            artifact_next_enabled = False
        else:
            sample_label = f"Sample: {index + 1} / {total}"
            sample_prev_enabled = index > 0
            sample_next_enabled = index < total - 1
            files = list(entries[index].get("files", []) or [])
            file_total = len(files)
            file_index = int(self.preview_state.artifact_file_index)
            if file_total <= 0 or not 0 <= file_index < file_total:
                artifact_label = f"{view_label}: - / -"
                artifact_prev_enabled = False
                artifact_next_enabled = False
            else:
                artifact_label = f"{view_label}: {file_index + 1} / {file_total}"
                artifact_prev_enabled = file_index > 0
                artifact_next_enabled = file_index < file_total - 1

        if hasattr(self, "preview_sample_nav_label"):
            self.preview_sample_nav_label.setText(sample_label)
        if hasattr(self, "preview_prev_sample_button"):
            self.preview_prev_sample_button.setEnabled(sample_prev_enabled)
        if hasattr(self, "preview_next_sample_button"):
            self.preview_next_sample_button.setEnabled(sample_next_enabled)
        if hasattr(self, "preview_artifact_nav_label"):
            self.preview_artifact_nav_label.setText(artifact_label)
        if hasattr(self, "preview_prev_artifact_button"):
            self.preview_prev_artifact_button.setEnabled(artifact_prev_enabled)
        if hasattr(self, "preview_next_artifact_button"):
            self.preview_next_artifact_button.setEnabled(artifact_next_enabled)

    # Keep sample and file selections within range when the result list changes.
    def open_preview_artifact_entry(self, index: int, file_index: int = 0):
        entries = list(self.preview_state.artifact_entries or [])
        if not entries:
            self.preview_state.artifact_index = -1
            self.preview_state.artifact_file_index = -1
            self.update_preview_artifact_nav_controls()
            return

        index = max(0, min(int(index), len(entries) - 1))
        files = list(entries[index].get("files", []) or [])
        self.preview_state.artifact_index = index
        self.preview_state.artifact_file_index = max(0, min(int(file_index), len(files) - 1)) if files else -1
        self.open_current_preview_artifact_file()

    # Route all artifact opens through preview_file so TIFF, PNG, and ND2 state
    # cleanup remains identical to opening a file from the Files tab.
    def open_current_preview_artifact_file(self):
        entries = list(self.preview_state.artifact_entries or [])
        index = int(self.preview_state.artifact_index)
        if not entries or not 0 <= index < len(entries):
            self.update_preview_artifact_nav_controls()
            return

        entry = entries[index]
        files = list(entry.get("files", []) or [])
        file_index = int(self.preview_state.artifact_file_index)
        if not files:
            self.update_preview_artifact_nav_controls()
            return
        file_index = max(0, min(file_index, len(files) - 1))
        self.preview_state.artifact_file_index = file_index

        path = str(files[file_index])
        self.preview_file(path)
        mode = str(self.preview_state.artifact_mode or "")
        if mode == "snapshot" and hasattr(self, "fit_preview_image"):
            self.fit_preview_image()
        mode_label = PREVIEW_ARTIFACT_VIEW_LABELS.get(mode, "Preview")
        sample = str(entry.get("sample") or Path(path).stem)
        file_count = len(files)
        extra = f"\nView: {mode_label}\nSample: {sample}"
        if file_count > 1:
            extra += f"\n{mode_label}: {file_index + 1} / {file_count}"
        if hasattr(self, "preview_info_label"):
            current = self.preview_info_label.text()
            self.preview_info_label.setText(f"{current}{extra}")
        self.update_preview_artifact_nav_controls()

    # Keep the current file if it belongs to the new category; otherwise open the first file.
    def select_preview_artifact_view(self, mode: str):
        mode = str(mode or "").strip().lower()
        if mode not in PREVIEW_ARTIFACT_VIEW_LABELS:
            return

        entries = self.collect_preview_artifact_entries(mode)
        self.preview_state.artifact_mode = mode
        self.preview_state.artifact_entries = entries
        self.preview_state.artifact_index = 0 if entries else -1
        self.preview_state.artifact_file_index = 0 if entries else -1
        self.set_preview_artifact_buttons_checked(mode)

        if not entries:
            mode_label = PREVIEW_ARTIFACT_VIEW_LABELS.get(mode, "Preview")
            if hasattr(self, "preview_info_label"):
                self.preview_info_label.setText(
                    f"No {mode_label.lower()} files were found for the selected output folder.\n"
                    "Run the pipeline or choose an output folder containing Results."
                )
            self.update_preview_artifact_nav_controls()
            return

        current_path = str(self.preview_state.file_path or "")
        current_key = self.preview_artifact_path_key(current_path) if current_path else ""
        selected_file_index = 0
        for entry_index, entry in enumerate(entries):
            files = [str(path) for path in entry.get("files", [])]
            for file_index, file_path in enumerate(files):
                if current_key and self.preview_artifact_path_key(file_path) == current_key:
                    self.preview_state.artifact_index = entry_index
                    selected_file_index = file_index
                    break
            else:
                continue
            break
        self.open_preview_artifact_entry(self.preview_state.artifact_index, selected_file_index)

    # Resolve and case-fold paths so Windows aliases and case differences do not
    # break synchronization with the file already displayed.
    def preview_artifact_path_key(self, path: str | Path) -> str:
        candidate = Path(path)
        try:
            candidate = candidate.resolve()
        except OSError:
            pass
        return str(candidate).casefold()

    # Synchronize before arrow movement because files opened from the Files tab
    # bypass the artifact selector's stored indices.
    def sync_preview_artifact_selection_to_current_file(self) -> bool:
        current_path = str(self.preview_state.file_path or "").strip()
        entries = list(self.preview_state.artifact_entries or [])
        if not current_path or not entries:
            return False

        current_key = self.preview_artifact_path_key(current_path)
        for entry_index, entry in enumerate(entries):
            files = list(entry.get("files", []) or [])
            for file_index, file_path in enumerate(files):
                if self.preview_artifact_path_key(file_path) == current_key:
                    self.preview_state.artifact_index = entry_index
                    self.preview_state.artifact_file_index = file_index
                    return True
        return False

    # Reset the artifact index when changing samples because each sample can
    # contain a different number of montages or channel-specific overlays.
    def step_preview_artifact_sample(self, delta: int):
        mode = str(self.preview_state.artifact_mode or "")
        if mode not in PREVIEW_ARTIFACT_VIEW_LABELS:
            mode = "overlay"
            self.preview_state.artifact_mode = mode
            self.preview_state.artifact_entries = self.collect_preview_artifact_entries(mode)
            self.preview_state.artifact_index = 0 if self.preview_state.artifact_entries else -1
            self.preview_state.artifact_file_index = 0 if self.preview_state.artifact_entries else -1
            self.set_preview_artifact_buttons_checked(mode)

        self.sync_preview_artifact_selection_to_current_file()
        entries = list(self.preview_state.artifact_entries or [])
        if not entries:
            self.update_preview_artifact_nav_controls()
            return

        current = int(self.preview_state.artifact_index)
        self.open_preview_artifact_entry(current + int(delta), 0)

    # Change only the file index here; keeping the sample index untouched is the
    # key distinction between the two navigation controls.
    def step_preview_artifact_file(self, delta: int):
        self.sync_preview_artifact_selection_to_current_file()
        entries = list(self.preview_state.artifact_entries or [])
        index = int(self.preview_state.artifact_index)
        if not entries or not 0 <= index < len(entries):
            self.update_preview_artifact_nav_controls()
            return

        files = list(entries[index].get("files", []) or [])
        if not files:
            self.update_preview_artifact_nav_controls()
            return

        current = int(self.preview_state.artifact_file_index)
        self.preview_state.artifact_file_index = max(0, min(current + int(delta), len(files) - 1))
        self.open_current_preview_artifact_file()
