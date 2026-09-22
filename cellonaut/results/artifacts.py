"""Explicit, run-local artifact relationships; old folders have no artifact section.

Writers publish files first, then atomically update the existing run manifest.
Runs own distinct output directories and write serially. Paths are relative to
that directory's Results root; display labels are presentation, not pair keys.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from cellonaut.io.writers import write_text


class ArtifactMetadataError(ValueError):
    """Metadata exists but cannot safely identify the requested artifact."""


def artifact_root(path: Path) -> Path | None:
    candidates = (path, *path.parents)
    for parent in candidates:
        if (parent / "Logs" / "RunSummary.json").is_file():
            return parent
    if (path / "Results" / "Logs" / "RunSummary.json").is_file():
        return path / "Results"
    return next((parent for parent in candidates if parent.name.lower() == "results"), None)


def read_manifest(root: Path) -> dict[str, Any]:
    path = root / "Logs" / "RunSummary.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("expected an object")
        return data
    except (OSError, UnicodeError, ValueError) as exc:
        raise ArtifactMetadataError(f"Could not read artifact manifest {path}: {exc}") from exc


def _relative_path(value: Any) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ArtifactMetadataError(f"Invalid relative artifact path: {value!r}")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or PureWindowsPath(value).drive
        or ".." in path.parts
        or value == "."
        or path.as_posix() != value
    ):
        raise ArtifactMetadataError(f"Invalid relative artifact path: {value!r}")
    return value


def _artifact_records(manifest: dict[str, Any]) -> list[dict[str, Any]] | None:
    if "artifacts" not in manifest:
        return None  # Legacy folder: do not mutate it on read.
    section = manifest["artifacts"]
    if not isinstance(section, dict) or section.get("version") != 1 or not isinstance(section.get("files"), list):
        raise ArtifactMetadataError("Invalid or unsupported artifact metadata.")
    seen: set[str] = set()
    for item in section["files"]:
        if not isinstance(item, dict) or any(
            not isinstance(item.get(key), str) or not item[key] for key in ("path", "sample", "target", "kind")
        ):
            raise ArtifactMetadataError("Incomplete artifact metadata record.")
        name = _relative_path(item["path"])
        if name.casefold() in seen:
            raise ArtifactMetadataError(f"Duplicate artifact path: {name}")
        seen.add(name.casefold())
        if "sidecar" in item:
            _relative_path(item["sidecar"])
        for key in ("label", "mask", "mask_label", "classifier", "cell_mask"):
            if key in item and not isinstance(item[key], str):
                raise ArtifactMetadataError(f"Invalid artifact {key}: {item[key]!r}")
        required = {
            "combined_overlay": ("sidecar",),
            "cell_signal": ("mask", "mask_label"),
            "threshold": ("mask", "classifier"),
        }.get(item["kind"], ())
        if any(not item.get(key) for key in required):
            raise ArtifactMetadataError(f"Incomplete {item['kind']} relationship metadata.")
    return section["files"]


@dataclass
class ArtifactResolver:
    """Resolve one run's recorded sample/target relationships, never filename guesses."""

    root: Path
    records: list[dict[str, Any]]

    @classmethod
    def load(cls, path: Path) -> ArtifactResolver | None:
        """Load metadata from a result root or a path below it.

        None permits legacy discovery only when no artifact inventory exists.
        Malformed metadata raises ArtifactMetadataError; an empty inventory is
        still authoritative. Referenced files are checked when resolved.
        """
        root = artifact_root(Path(path))
        if root is None:
            return None
        records = _artifact_records(read_manifest(root))
        return cls(root, records) if records is not None else None

    def path(self, record: dict[str, Any], field: str = "path") -> Path:
        """Require an existing file inside this result root, including after symlink resolution."""
        path = self.root / _relative_path(record.get(field))
        if not path.resolve().is_relative_to(self.root.resolve()) or not path.is_file():
            raise ArtifactMetadataError(f"Referenced artifact is missing or outside the result folder: {path}")
        return path

    def record(self, path: Path) -> dict[str, Any]:
        absolute = Path(path).resolve()
        for item in self.records:
            if (self.root / item["path"]).resolve() == absolute:
                return item
        raise ArtifactMetadataError(f"Artifact is not recorded in this run: {path}")

    def matching(
        self, *, sample: str | None = None, target: str | None = None, kind: str | None = None
    ) -> list[dict[str, Any]]:
        return [
            item
            for item in self.records
            if all(
                value is None or item[key] == value
                for key, value in (("sample", sample), ("target", target), ("kind", kind))
            )
        ]

    def related(self, record: dict[str, Any], kind: str) -> dict[str, Any]:
        related_target = (
            str(record.get("cell_mask", "") or record["target"])
            if kind in {"cell_labels", "cell_outline"}
            else record["target"]
        )
        matches = self.matching(sample=record["sample"], target=related_target, kind=kind)
        cell_mask = str(record.get("cell_mask", "") or "")
        if cell_mask and kind not in {"cell_labels", "cell_outline"}:
            exact = [item for item in matches if str(item.get("cell_mask", "") or "") == cell_mask]
            matches = exact or [item for item in matches if not str(item.get("cell_mask", "") or "")]
        if len(matches) != 1:
            raise ArtifactMetadataError(
                f"Expected one {kind} for {record['sample']} / {related_target}; found {len(matches)}."
            )
        self.path(matches[0])
        return matches[0]

    def signal(
        self, record: dict[str, Any], mask_label: str, image_defs: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        exact_matches = []
        legacy_matches = []
        record_cell_mask = str(record.get("cell_mask", "") or "")
        for signal in self.matching(sample=record["sample"], target=record["target"], kind="cell_signal"):
            definition = image_definition({"target": signal["mask"], "label": signal["mask_label"]}, image_defs)
            current_label = definition.get("name") if definition is not None else signal["mask_label"]
            if current_label == mask_label:
                signal_cell_mask = str(signal.get("cell_mask", "") or "")
                if signal_cell_mask == record_cell_mask:
                    exact_matches.append(signal)
                elif not signal_cell_mask:
                    legacy_matches.append(signal)
        matches = exact_matches or legacy_matches
        if len(matches) > 1:
            raise ArtifactMetadataError(f"Multiple signal tables recorded for mask {mask_label}.")
        return matches[0] if matches else None

    def sidecar(self, record: dict[str, Any]) -> dict[str, Any]:
        if "sidecar" not in record:
            return {}
        path = self.path(record, "sidecar")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("expected an object")
            return data
        except (OSError, UnicodeError, ValueError) as exc:
            raise ArtifactMetadataError(f"Could not read artifact sidecar {path}: {exc}") from exc


def initialize_artifact_manifest(root: Path) -> None:
    """Mark a new execution as metadata-backed even before its first artifact."""
    manifest = read_manifest(root)
    if "artifacts" not in manifest:
        manifest["artifacts"] = {"version": 1, "files": []}
        write_text(root / "Logs" / "RunSummary.json", json.dumps(manifest, indent=2))
    else:
        ArtifactResolver.load(root)  # Refuse to write into corrupt metadata.


def record_artifact(
    path: Path,
    *,
    sample: str,
    target: str,
    kind: str,
    label: str = "",
    sidecar: Path | None = None,
    mask: str = "",
    mask_label: str = "",
    classifier: str = "",
    cell_mask: str = "",
) -> None:
    """Atomically register an already-written file using run-local sample/target keys.

    A repeated path may replace its record only with the same sample/target/kind.
    Standalone exports outside a recognized Results root are not registered.
    Publication assumes serial writers within a run; it is not a multiwriter lock.
    """
    root = artifact_root(Path(path))
    if root is None:
        return
    if not path.is_file():
        raise ArtifactMetadataError(f"Cannot record an unwritten artifact: {path}")
    manifest = read_manifest(root)
    files = list(_artifact_records(manifest) or [])
    relative = Path(path).relative_to(root).as_posix()
    item = {"path": relative, "sample": sample, "target": target, "kind": kind, "label": label}
    if classifier:
        item["classifier"] = classifier
    if mask:
        item["mask"] = mask
    if mask_label:
        item["mask_label"] = mask_label
    if cell_mask:
        item["cell_mask"] = cell_mask
    if sidecar is not None:
        item["sidecar"] = sidecar.relative_to(root).as_posix()
    for existing in files:
        if existing["path"].casefold() == relative.casefold() and any(
            existing[key] != item[key] for key in ("sample", "target", "kind")
        ):
            raise ArtifactMetadataError(f"Conflicting artifact identity: {relative}")
    files = [existing for existing in files if existing["path"].casefold() != relative.casefold()]
    files.append(item)
    manifest["artifacts"] = {"version": 1, "files": sorted(files, key=lambda entry: entry["path"])}
    _artifact_records(manifest)
    write_text(root / "Logs" / "RunSummary.json", json.dumps(manifest, indent=2))


def image_definition(record: dict[str, Any], image_defs: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Match an explicit key, then a saved label, then unchanged GUI slot order.

    GUI definitions currently derive imageN keys from ordering; these are not
    globally persistent IDs. Reordering and renaming together needs a new run.
    """
    for image in image_defs:
        if image.get("key") == record["target"]:
            return image
    for image in image_defs:
        if not image.get("key") and image.get("name") == record.get("label"):
            return image
    for index, image in enumerate(image_defs, 1):
        if record["target"] == f"image{index}" and not image.get("key"):
            return image
    return None


def current_layer_labels(sidecar: dict[str, Any], image_defs: list[dict[str, Any]]) -> list[str]:
    """Translate existing layer keys at the current-settings boundary only."""
    names = list(sidecar.get("layer_labels", []))
    for index, key in enumerate(sidecar.get("layer_keys", [])):
        if index < len(names) and key:
            definition = image_definition({"target": key, "label": names[index]}, image_defs)
            if definition is not None:
                names[index] = str(definition.get("name") or names[index])
    return names


def saved_cell_labels(root: Path, sample: str, target: str, label: str, relative_parent: Path = Path()) -> Path:
    """Reuse and setup validation share the same exact label association."""
    resolver = ArtifactResolver.load(root)
    if resolver is not None:
        if not target:
            matches = [
                record
                for record in resolver.matching(sample=sample, kind="cell_labels")
                if record.get("label") == label
            ]
            if len(matches) != 1:
                raise ArtifactMetadataError(f"No unique saved label image for {sample} / {label}.")
            return resolver.path(matches[0])
        return resolver.path(resolver.related({"sample": sample, "target": target}, "cell_labels"))
    return root / "Cells" / "TIFF Labels" / relative_parent / f"{sample}_{label}_01_cellpose_labels.tif"


def load_artifact_sidecar(preview_path: Path) -> dict[str, Any]:
    resolver = ArtifactResolver.load(preview_path)
    if resolver is not None:
        try:
            record = resolver.record(preview_path)
        except ArtifactMetadataError:
            # User-created snapshots/imports can live under Results too. Their
            # embedded display metadata is sufficient; never infer result links.
            return {}
        return resolver.sidecar(record)
    # Legacy sidecar layouts (never searched for metadata-backed results).
    sidecar_candidates = [
        preview_path.with_suffix(".json"),
        preview_path.parent.parent / "JSON" / f"{preview_path.stem}.json",
    ]
    for parent in preview_path.parents:
        if parent.name == "TIFF Overlays":
            sidecar_candidates.append(parent.parent / "JSON" / preview_path.relative_to(parent).with_suffix(".json"))
            break
    sidecar = next((path for path in sidecar_candidates if path.exists()), None)
    if sidecar is None:
        return {}
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


# Use the same table filename handling for exact and fallback lookup.
