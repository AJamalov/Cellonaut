"""Shared Cellpose segmentation and export helpers."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
import re
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from skimage.measure import find_contours, regionprops, regionprops_table

from cellonaut.runtime import PipelineRuntime
from cellonaut.cell_segmentation.runtime import configure_local_cellpose_models, require_cellpose_model_runtime
from cellonaut.config.defaults import (
    DEFAULT_CELL_MIN_SIZE,
    DEFAULT_CELLPOSE_CUSTOM_MODEL_PATH,
    DEFAULT_CELLPOSE_MODEL_TYPE,
    DEFAULT_CELL_REMOVE_BORDER,
    DEFAULT_CELLPROB_THRESHOLD,
    DEFAULT_FLOW_THRESHOLD,
)
from cellonaut.io.writers import save_pil_image, write_dataframe_csv
from cellonaut.pipeline.cancellation import check_cancel
from cellonaut.system.gpu_detection import describe_cellpose_backend


def normalize_cell_segmentation_export_table(df: pd.DataFrame) -> pd.DataFrame:
    """Translate internal cell-table labels into the public export vocabulary."""

    if df is None:
        return pd.DataFrame()
    out = df.copy()
    out = out.rename(columns={col: re.sub(r"Mean(?!GrayValue)", "MeanGrayValue", str(col)) for col in out.columns})
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].map(lambda value: "AVERAGE" if str(value) == "MEAN" else value)
    return out


@dataclass
class CellSegmentationConfig:
    """Cellpose inference and post-processing settings independent of GUI state."""

    model_type: str = DEFAULT_CELLPOSE_MODEL_TYPE
    custom_model_path: str = DEFAULT_CELLPOSE_CUSTOM_MODEL_PATH
    diameter: Optional[float] = None
    min_size: int = int(DEFAULT_CELL_MIN_SIZE)
    use_gpu: bool = True
    gpu_requested: Optional[bool] = None

    cellprob_threshold: float = float(DEFAULT_CELLPROB_THRESHOLD)
    flow_threshold: float = float(DEFAULT_FLOW_THRESHOLD)

    save_rois_csv: bool = True
    show_labels_in_qc: bool = True
    remove_border: bool = DEFAULT_CELL_REMOVE_BORDER
    save_qc_overlay: bool = True


_CELLPOSE_MODEL_CACHE_LIMIT = 2
_CELLPOSE_MODEL_CACHE: OrderedDict[tuple[Any, ...], Any] = OrderedDict()
_CELLPOSE_MODEL_CACHE_LOCK = Lock()


# Include model path, modification time, and size in the cache key to detect file changes.
def _cellpose_model_cache_key(cfg: CellSegmentationConfig) -> tuple[Any, ...]:
    custom_model = str(cfg.custom_model_path or "").strip()
    if custom_model:
        path = Path(custom_model).expanduser()
        try:
            path = path.resolve(strict=False)
        except OSError:
            path = path.absolute()
        try:
            stat = path.stat()
            fingerprint: tuple[int | None, int | None] = (stat.st_mtime_ns, stat.st_size)
        except OSError:
            fingerprint = (None, None)
        return ("custom", str(path), *fingerprint, bool(cfg.use_gpu))

    model_type = str(cfg.model_type or DEFAULT_CELLPOSE_MODEL_TYPE).strip() or DEFAULT_CELLPOSE_MODEL_TYPE
    return ("built_in", model_type, bool(cfg.use_gpu))


def clear_cellpose_model_cache() -> None:
    """Release cached Cellpose models and their potentially large Torch state."""

    with _CELLPOSE_MODEL_CACHE_LOCK:
        _CELLPOSE_MODEL_CACHE.clear()


def normalize_for_cellpose(img: np.ndarray) -> np.ndarray:
    """Normalize integer or floating-point image data for Cellpose inference."""

    img = np.asarray(img)

    if np.issubdtype(img.dtype, np.integer):
        info = np.iinfo(img.dtype)
        if info.max > 0:
            img = img.astype(np.float32) / float(info.max)
        else:
            img = img.astype(np.float32)
    else:
        img = img.astype(np.float32)
        maxv = np.nanmax(img)
        if maxv > 0:
            img = img / maxv

    return np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)


def normalize_to_uint8_display(img: np.ndarray) -> np.ndarray:
    """Create an 8-bit RGB display copy without modifying measurement data."""

    img = np.asarray(img)

    if img.ndim == 2:
        arr = img.astype(np.float32)
        lo = np.nanpercentile(arr, 1)
        hi = np.nanpercentile(arr, 99)
        if hi <= lo:
            hi = lo + 1e-6
        arr = np.clip((arr - lo) / (hi - lo), 0, 1)
        u8 = np.multiply(arr, np.float32(255.0)).astype(np.uint8)
        return np.stack([u8, u8, u8], axis=-1)

    if img.ndim == 3 and img.shape[-1] in (3, 4):
        arr = img[..., :3].astype(np.float32)
        lo = np.nanpercentile(arr, 1)
        hi = np.nanpercentile(arr, 99)
        if hi <= lo:
            hi = lo + 1e-6
        arr = np.clip((arr - lo) / (hi - lo), 0, 1)
        return np.multiply(arr, np.float32(255.0)).astype(np.uint8)

    raise ValueError(f"Unsupported image shape for display normalization: {img.shape}")


def get_cellpose_model(cfg: CellSegmentationConfig):
    """Load or reuse the Cellpose model matching the supplied configuration."""

    # Set the local model path before importing Cellpose; it reads the path during import.
    configure_local_cellpose_models()
    try:
        from cellpose import models
    except Exception as e:
        raise RuntimeError("Cellpose could not be imported. " f"Original error: {e}") from e

    key = _cellpose_model_cache_key(cfg)
    with _CELLPOSE_MODEL_CACHE_LOCK:
        cached = _CELLPOSE_MODEL_CACHE.get(key)
        if cached is not None:
            _CELLPOSE_MODEL_CACHE.move_to_end(key)
            return cached

        custom_model = str(cfg.custom_model_path or "").strip()
        model_type = (
            str(cfg.model_type or DEFAULT_CELLPOSE_MODEL_TYPE).strip() or DEFAULT_CELLPOSE_MODEL_TYPE
        )
        if custom_model:
            model = models.CellposeModel(
                gpu=cfg.use_gpu,
                pretrained_model=custom_model,
            )
        else:
            require_cellpose_model_runtime(model_type)
            model = models.CellposeModel(
                gpu=cfg.use_gpu,
                pretrained_model=model_type,
            )

        _CELLPOSE_MODEL_CACHE[key] = model
        _CELLPOSE_MODEL_CACHE.move_to_end(key)
        while len(_CELLPOSE_MODEL_CACHE) > _CELLPOSE_MODEL_CACHE_LIMIT:
            _CELLPOSE_MODEL_CACHE.popitem(last=False)
        return model


def run_cellpose_segmentation(
    img: np.ndarray,
    cfg: CellSegmentationConfig,
    log_func: Optional[Callable[[str], None]] = None,
    runtime: PipelineRuntime | None = None,
) -> np.ndarray:
    """Run Cellpose inference and return its integer label image."""

    model = get_cellpose_model(cfg)
    img_norm = normalize_for_cellpose(img)

    if log_func:
        import torch

        backend = describe_cellpose_backend(
            torch_module=torch,
            acceleration_allowed=cfg.use_gpu,
            acceleration_requested=cfg.gpu_requested,
        )
        log_func(f"{backend.label}: {backend.detail}")

    if img_norm.ndim != 2 and not (img_norm.ndim == 3 and img_norm.shape[-1] == 3):
        raise ValueError(f"Unsupported image shape for Cellpose: {img_norm.shape}")

    masks, _, _ = model.eval(
        img_norm,
        diameter=cfg.diameter,
        min_size=cfg.min_size,
        cellprob_threshold=cfg.cellprob_threshold,
        flow_threshold=cfg.flow_threshold,
    )
    device_type = getattr(getattr(model, "device", None), "type", None)
    if isinstance(device_type, str):
        if runtime is not None:
            runtime.cellpose_inference_devices.add(device_type.upper())
        if log_func:
            log_func(f"Cellpose inference device: {device_type.upper()}")
    return masks.astype(np.int32)


def relabel_sequential(label_img: np.ndarray) -> np.ndarray:
    """Compact positive labels into the consecutive IDs expected by exports."""

    label_img = np.asarray(label_img, dtype=np.int32)
    out = np.zeros_like(label_img, dtype=np.int32)

    labels = np.unique(label_img)
    labels = labels[labels != 0]

    for new_id, old_id in enumerate(labels, start=1):
        out[label_img == old_id] = new_id

    return out


def filter_labels_by_size(label_img: np.ndarray, min_size: int) -> np.ndarray:
    """Remove labels smaller than the configured post-inference area threshold."""

    label_img = np.asarray(label_img, dtype=np.int32)

    if label_img.max() == 0:
        return np.zeros_like(label_img, dtype=np.int32)

    props = regionprops_table(label_img, properties=("label", "area"))
    props_df = pd.DataFrame(props)

    keep = set(int(x) for x in props_df.loc[props_df["area"] >= min_size, "label"].tolist())

    if not keep:
        return np.zeros_like(label_img, dtype=np.int32)

    out = np.zeros_like(label_img, dtype=np.int32)
    for old_label in keep:
        out[label_img == old_label] = old_label

    return relabel_sequential(out)


def remove_border_labels_fn(label_img: np.ndarray) -> np.ndarray:
    """Remove complete labels touching an image edge as partial cells."""

    label_img = np.asarray(label_img, dtype=np.int32)

    if label_img.max() == 0:
        return np.zeros_like(label_img, dtype=np.int32)

    border_labels = set()
    border_labels.update(np.unique(label_img[0, :]))
    border_labels.update(np.unique(label_img[-1, :]))
    border_labels.update(np.unique(label_img[:, 0]))
    border_labels.update(np.unique(label_img[:, -1]))
    border_labels.discard(0)

    out = label_img.copy()
    for lab in border_labels:
        out[out == lab] = 0

    return relabel_sequential(out)


def make_per_cell_table(label_img: np.ndarray, intensity_img: np.ndarray) -> pd.DataFrame:
    """Measure a 2-D label image against matching intensity pixels.

    Zero is background; positive IDs become CellID. Area is in pixels squared
    and perimeter in pixels, without spatial calibration. RawIntDen and IntDen
    both equal area times mean intensity. Circularity is 4*pi*area/perimeter**2
    (NaN for zero perimeter). A trailing 1/3/4-component intensity axis uses
    component zero. Inputs are not modified; no cells yields a schema-only table.
    """

    label_img = np.asarray(label_img, dtype=np.int32)
    intensity_img = np.asarray(intensity_img)

    if label_img.shape[:2] != intensity_img.shape[:2]:
        raise ValueError(
            f"Label image shape {label_img.shape[:2]} does not match intensity image shape {intensity_img.shape[:2]}"
        )

    if label_img.max() == 0:
        return pd.DataFrame(columns=pd.Index(["CellID", "Area", "Perimeter", "Solidity", "Circularity", "Mean", "Min", "Max", "Median", "RawIntDen", "IntDen"]))

    if intensity_img.ndim == 3:
        if intensity_img.shape[-1] not in (1, 3, 4):
            raise ValueError(f"Unsupported intensity image shape for measurement: {intensity_img.shape}")
        intensity_img = intensity_img[..., 0]

    props = regionprops_table(
        label_img,
        intensity_image=intensity_img,
        properties=(
            "label",
            "area",
            "perimeter",
            "solidity",
            "intensity_mean",
            "intensity_min",
            "intensity_max",
        ),
    )
    df = pd.DataFrame(props)

    df["RawIntDen"] = df["area"] * df["intensity_mean"]
    df["IntDen"] = df["RawIntDen"]
    medians = {
        int(item.label): float(np.median(item.intensity_image[item.image]))
        for item in regionprops(label_img, intensity_image=intensity_img)
    }
    df["Median"] = pd.Series(df["label"], index=df.index).map(medians)
    perimeter = np.asarray(pd.to_numeric(df["perimeter"], errors="coerce"), dtype=float)
    area = np.asarray(pd.to_numeric(df["area"], errors="coerce"), dtype=float)
    circularity = np.full(perimeter.shape, np.nan, dtype=float)
    np.divide(
        4.0 * np.pi * area,
        np.square(perimeter),
        out=circularity,
        where=perimeter > 0,
    )
    df["Circularity"] = circularity

    return df.rename(
        columns={
            "label": "CellID",
            "area": "Area",
            "perimeter": "Perimeter",
            "solidity": "Solidity",
            "intensity_mean": "Mean",
            "intensity_min": "Min",
            "intensity_max": "Max",
        }
    )


# Centralize empty-region handling so raw and corrected intensity paths use identical rules.
def _masked_intensity_stats(intensity_img: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    pixels = intensity_img[mask]
    if pixels.size == 0:
        return np.nan, 0.0
    return float(pixels.mean()), float(pixels.sum())


def _masked_intensity_distribution(intensity_img: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    pixels = intensity_img[mask]
    if pixels.size == 0:
        return {
            "mean": np.nan,
            "intden": 0.0,
            "std_dev": np.nan,
            "min": np.nan,
            "max": np.nan,
            "median": np.nan,
        }
    return {
        "mean": float(pixels.mean()),
        "intden": float(pixels.sum()),
        "std_dev": float(pixels.std(ddof=1)) if pixels.size > 1 else np.nan,
        "min": float(pixels.min()),
        "max": float(pixels.max()),
        "median": float(np.median(pixels)),
    }


def make_per_cell_organelle_signal_tables(
    cell_label_img: np.ndarray,
    organelle_mask: np.ndarray,
    intensity_img: np.ndarray,
    organelle_prefix: str = "Organelle",
    corrected_intensity_img: Optional[np.ndarray] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (signal, geometry) tables for matching YX labels, mask and intensity.

    Positive mask pixels define the region inside each cell; its complement
    inside that cell is RestOfCell. The caller supplies raw and optional
    corrected intensities; this function performs no background subtraction.
    Nonfinite intensities become zero. Empty regions have zero area/sum and
    NaN distribution statistics; StdDev uses the sample estimate (ddof=1).

    Geometry uses zero-based pixel coordinates (Y=row, X=column) with exclusive
    bounding-box maxima. Inputs are not modified; trailing component axes use
    component zero. The mask name supplies column prefixes, not biological rules.
    """

    cell_label_img = np.asarray(cell_label_img, dtype=np.int32)
    organelle_mask = np.asarray(organelle_mask)
    intensity_img = np.asarray(intensity_img)

    if cell_label_img.shape[:2] != organelle_mask.shape[:2]:
        raise ValueError(
            f"Cell label image shape {cell_label_img.shape[:2]} does not match organelle mask shape {organelle_mask.shape[:2]}"
        )
    if cell_label_img.shape[:2] != intensity_img.shape[:2]:
        raise ValueError(
            f"Cell label image shape {cell_label_img.shape[:2]} does not match intensity image shape {intensity_img.shape[:2]}"
        )

    if organelle_mask.ndim == 3:
        organelle_mask = organelle_mask[..., 0]
    if intensity_img.ndim == 3:
        intensity_img = intensity_img[..., 0]

    organelle_mask = organelle_mask > 0
    raw_intensity_img = np.nan_to_num(
        intensity_img.astype(np.float64),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    corrected_img = None
    if corrected_intensity_img is not None:
        corrected_img = np.asarray(corrected_intensity_img)
        if cell_label_img.shape[:2] != corrected_img.shape[:2]:
            raise ValueError(
                f"Cell label image shape {cell_label_img.shape[:2]} does not match corrected intensity image shape {corrected_img.shape[:2]}"
            )
        if corrected_img.ndim == 3:
            corrected_img = corrected_img[..., 0]
        corrected_img = np.nan_to_num(
            corrected_img.astype(np.float64),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    biology_rows: List[Dict[str, Any]] = []
    geometry_rows: List[Dict[str, Any]] = []

    cell_ids = np.unique(cell_label_img)
    cell_ids = cell_ids[cell_ids != 0]

    for cell_id in cell_ids:
        cell_mask = cell_label_img == cell_id
        cell_area = int(cell_mask.sum())
        if cell_area == 0:
            continue
        organelle_in_cell = cell_mask & organelle_mask
        organelle_area = int(organelle_in_cell.sum())

        rest_of_cell_mask = cell_mask & (~organelle_mask)
        rest_area = int(rest_of_cell_mask.sum())

        raw_cell_mean, raw_cell_intden = _masked_intensity_stats(raw_intensity_img, cell_mask)
        organelle_stats = _masked_intensity_distribution(raw_intensity_img, organelle_in_cell)
        raw_rest_mean, raw_rest_intden = _masked_intensity_stats(raw_intensity_img, rest_of_cell_mask)

        ys, xs = np.where(cell_mask)
        centroid_y = float(ys.mean())
        centroid_x = float(xs.mean())
        bbox_min_row = int(ys.min())
        bbox_min_col = int(xs.min())
        bbox_max_row = int(ys.max()) + 1
        bbox_max_col = int(xs.max()) + 1

        row = {
            "CellID": int(cell_id),
            "CellArea": cell_area,
            "CellMean": raw_cell_mean,
            "CellIntDen": raw_cell_intden,
            f"{organelle_prefix}Area_InCell": organelle_area,
            f"{organelle_prefix}Mean_InCell": organelle_stats["mean"],
            f"{organelle_prefix}StdDev_InCell": organelle_stats["std_dev"],
            f"{organelle_prefix}Min_InCell": organelle_stats["min"],
            f"{organelle_prefix}Max_InCell": organelle_stats["max"],
            f"{organelle_prefix}Median_InCell": organelle_stats["median"],
            f"{organelle_prefix}IntDen_InCell": organelle_stats["intden"],
            "RestOfCellArea": rest_area,
            "RestOfCellMean": raw_rest_mean,
            "RestOfCellIntDen": raw_rest_intden,
        }
        if corrected_img is not None:
            corrected_cell_mean, corrected_cell_intden = _masked_intensity_stats(corrected_img, cell_mask)
            corrected_organelle_stats = _masked_intensity_distribution(corrected_img, organelle_in_cell)
            corrected_rest_mean, corrected_rest_intden = _masked_intensity_stats(corrected_img, rest_of_cell_mask)

            row.update(
                {
                    "CellCorrectedMean": corrected_cell_mean,
                    "CellCorrectedIntDen": corrected_cell_intden,
                    f"{organelle_prefix}CorrectedMean_InCell": corrected_organelle_stats["mean"],
                    f"{organelle_prefix}CorrectedStdDev_InCell": corrected_organelle_stats["std_dev"],
                    f"{organelle_prefix}CorrectedMin_InCell": corrected_organelle_stats["min"],
                    f"{organelle_prefix}CorrectedMax_InCell": corrected_organelle_stats["max"],
                    f"{organelle_prefix}CorrectedMedian_InCell": corrected_organelle_stats["median"],
                    f"{organelle_prefix}CorrectedIntDen_InCell": corrected_organelle_stats["intden"],
                    "RestOfCellCorrectedMean": corrected_rest_mean,
                    "RestOfCellCorrectedIntDen": corrected_rest_intden,
                }
            )

        biology_rows.append(row)
        geometry_rows.append(
            {
                "CellID": int(cell_id),
                "Centroid_Y": centroid_y,
                "Centroid_X": centroid_x,
                "BBox_Min_Row": bbox_min_row,
                "BBox_Min_Col": bbox_min_col,
                "BBox_Max_Row": bbox_max_row,
                "BBox_Max_Col": bbox_max_col,
            }
        )

    biology_df = pd.DataFrame(biology_rows)
    geometry_df = pd.DataFrame(geometry_rows)
    if not biology_rows:
        columns = ["CellID", "CellArea", "CellMean", "CellIntDen", "RestOfCellArea", "RestOfCellMean", "RestOfCellIntDen"]
        columns.extend(f"{organelle_prefix}{metric}_InCell" for metric in ("Area", "Mean", "StdDev", "Min", "Max", "Median", "IntDen"))
        if corrected_img is not None:
            columns.extend(["CellCorrectedMean", "CellCorrectedIntDen", "RestOfCellCorrectedMean", "RestOfCellCorrectedIntDen"])
            columns.extend(f"{organelle_prefix}Corrected{metric}_InCell" for metric in ("Mean", "StdDev", "Min", "Max", "Median", "IntDen"))
        biology_df = pd.DataFrame(columns=pd.Index(columns))
        geometry_df = pd.DataFrame(columns=pd.Index(["CellID", "Centroid_Y", "Centroid_X", "BBox_Min_Row", "BBox_Min_Col", "BBox_Max_Row", "BBox_Max_Col"]))
    return biology_df, geometry_df


def export_per_cell_organelle_signal_tables(
    cell_label_img: np.ndarray,
    organelle_mask: np.ndarray,
    intensity_img: np.ndarray,
    out_biology_csv: Optional[Path],
    out_geometry_csv: Optional[Path] = None,
    organelle_prefix: str = "Organelle",
    corrected_intensity_img: Optional[np.ndarray] = None,
    measurement_options: Optional[Dict[str, bool]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Write per-cell organelle and geometry tables in their public CSV forms."""

    biology_df, geometry_df = make_per_cell_organelle_signal_tables(
        cell_label_img=cell_label_img,
        organelle_mask=organelle_mask,
        intensity_img=intensity_img,
        organelle_prefix=organelle_prefix,
        corrected_intensity_img=corrected_intensity_img,
    )

    if measurement_options is not None:
        optional_columns = {
            "cell_area": ["CellArea"],
            "cell_mean": ["CellMean", "CellCorrectedMean"],
            "cell_raw_intden": ["CellIntDen", "CellCorrectedIntDen"],
            "positive_area_in_cell": [f"{organelle_prefix}Area_InCell"],
            "mean_in_positive_area": [
                f"{organelle_prefix}Mean_InCell",
                f"{organelle_prefix}CorrectedMean_InCell",
            ],
            "std_dev_in_positive_area": [
                f"{organelle_prefix}StdDev_InCell",
                f"{organelle_prefix}CorrectedStdDev_InCell",
            ],
            "min_max_in_positive_area": [
                f"{organelle_prefix}Min_InCell",
                f"{organelle_prefix}Max_InCell",
                f"{organelle_prefix}CorrectedMin_InCell",
                f"{organelle_prefix}CorrectedMax_InCell",
            ],
            "median_in_positive_area": [
                f"{organelle_prefix}Median_InCell",
                f"{organelle_prefix}CorrectedMedian_InCell",
            ],
            "raw_intden_in_cell": [
                f"{organelle_prefix}IntDen_InCell",
                f"{organelle_prefix}CorrectedIntDen_InCell",
            ],
        }
        omitted = [
            column
            for key, columns in optional_columns.items()
            if not measurement_options.get(key, False)
            for column in columns
        ]
        biology_df = biology_df.drop(columns=omitted, errors="ignore")

    mean_only_cols = [
        "CellMean",
        "RestOfCellMean",
        f"{organelle_prefix}Mean_InCell",
        f"{organelle_prefix}StdDev_InCell",
        f"{organelle_prefix}Min_InCell",
        f"{organelle_prefix}Max_InCell",
        f"{organelle_prefix}Median_InCell",
        "CellCorrectedMean",
        "RestOfCellCorrectedMean",
        f"{organelle_prefix}CorrectedMean_InCell",
        f"{organelle_prefix}CorrectedStdDev_InCell",
        f"{organelle_prefix}CorrectedMin_InCell",
        f"{organelle_prefix}CorrectedMax_InCell",
        f"{organelle_prefix}CorrectedMedian_InCell",
        "Centroid_Y",
        "Centroid_X",
        "BBox_Min_Row",
        "BBox_Min_Col",
        "BBox_Max_Row",
        "BBox_Max_Col",
    ]

    if not biology_df.empty and not geometry_df.empty and "CellID" in biology_df.columns:
        biology_df = biology_df.merge(geometry_df, on="CellID", how="left")

    biology_export_df = append_sum_and_mean_summary_rows(
        biology_df,
        label_col="CellID",
        sum_label="SUM",
        mean_label="MEAN",
        mean_only_cols=mean_only_cols,
    )
    biology_export_df = reorder_summary_rows_first(
        biology_export_df,
        label_col="CellID",
        summary_labels=["SUM", "MEAN"],
    )

    if out_biology_csv is not None:
        write_dataframe_csv(normalize_cell_segmentation_export_table(biology_export_df), out_biology_csv, index=False)

    if out_geometry_csv is not None:
        write_dataframe_csv(normalize_cell_segmentation_export_table(geometry_df), out_geometry_csv, index=False)

    return biology_df, geometry_df


def append_sum_and_mean_summary_rows(
    df: pd.DataFrame,
    label_col: str = "CellID",
    sum_label: str = "SUM",
    mean_label: str = "MEAN",
    mean_only_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Append spreadsheet-oriented sum and mean rows to an export copy."""

    if df is None:
        return pd.DataFrame()

    mean_only_set = set(mean_only_cols or [])

    if df.empty:
        out = df.copy()
        if label_col in out.columns:
            empty_sum_row: Dict[str, Any] = {col: "" for col in out.columns}
            empty_mean_row: Dict[str, Any] = {col: "" for col in out.columns}
            empty_sum_row[label_col] = sum_label
            empty_mean_row[label_col] = mean_label
            out = pd.concat([out, pd.DataFrame([empty_sum_row, empty_mean_row])], ignore_index=True)
        return out

    out = df.copy()

    sum_row: Dict[str, Any] = {}
    mean_row: Dict[str, Any] = {}

    for col in out.columns:
        if col == label_col:
            sum_row[col] = sum_label
            mean_row[col] = mean_label
            continue

        numeric = cast(pd.Series, pd.to_numeric(pd.Series(out[col], index=out.index), errors="coerce"))
        if numeric.notna().any():
            if col in mean_only_set:
                sum_row[col] = ""
            else:
                sum_row[col] = float(numeric.sum())
            mean_row[col] = float(numeric.mean())
        else:
            sum_row[col] = ""
            mean_row[col] = ""

    return pd.concat([out, pd.DataFrame([sum_row, mean_row])], ignore_index=True)


def reorder_summary_rows_first(
    df: pd.DataFrame,
    label_col: str = "CellID",
    summary_labels: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Move export summary rows before the real per-cell records."""

    if df is None or df.empty or label_col not in df.columns:
        return df

    summary_labels = summary_labels or ["SUM", "MEAN"]

    work = df.copy()
    work["_sort_key"] = 2

    for idx, label in enumerate(summary_labels):
        work.loc[work[label_col].astype(str) == label, "_sort_key"] = idx

    work = work.sort_values(by=["_sort_key", label_col], kind="stable").drop(columns=["_sort_key"])
    return work.reset_index(drop=True)


def save_roi_outline_csv(label_img: np.ndarray, out_csv: Path) -> None:
    """Export every cell boundary as portable contour coordinates."""

    rows: List[Dict[str, Any]] = []

    if label_img.max() > 0:
        for lab in np.unique(label_img):
            if lab == 0:
                continue
            binary = (label_img == lab).astype(np.uint8)
            contours = find_contours(binary, level=0.5)
            for contour_idx, contour in enumerate(contours):
                for pt_idx, pt in enumerate(contour):
                    rows.append(
                        {
                            "CellID": int(lab),
                            "ContourIndex": contour_idx,
                            "PointIndex": pt_idx,
                            "Y": float(pt[0]),
                            "X": float(pt[1]),
                        }
                    )

    write_dataframe_csv(pd.DataFrame(rows), out_csv, index=False)


def build_qc_overlay(base_img: np.ndarray, label_img: np.ndarray, show_labels: bool = True) -> Image.Image:
    """Render cell outlines and optional IDs on a non-mutating display copy."""

    rgb = normalize_to_uint8_display(base_img)

    pil_img = Image.fromarray(rgb).convert("RGBA")
    overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for lab in np.unique(label_img):
        if lab == 0:
            continue

        binary = (label_img == lab).astype(np.uint8)
        contours = find_contours(binary, level=0.5)

        for contour in contours:
            pts = [(float(p[1]), float(p[0])) for p in contour]
            if len(pts) >= 2:
                draw.line(pts, width=2, fill=(255, 0, 0, 255))

        if show_labels:
            ys, xs = np.where(binary > 0)
            if ys.size > 0:
                x = float(xs.mean())
                y = float(ys.mean())
                draw.text((x, y), str(int(lab)), fill=(255, 255, 0, 255))

    return Image.alpha_composite(pil_img, overlay).convert("RGB")


def run_cell_segmentation_on_image(
    img: np.ndarray,
    cfg: CellSegmentationConfig,
    log_func: Optional[Callable[[str], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
    runtime: PipelineRuntime | None = None,
) -> np.ndarray:
    """Run inference and configured cleanup, checking cancellation between stages."""

    check_cancel(should_cancel)
    masks = run_cellpose_segmentation(img, cfg, log_func=log_func, runtime=runtime)

    check_cancel(should_cancel)
    masks = filter_labels_by_size(masks, cfg.min_size)

    if cfg.remove_border:
        check_cancel(should_cancel)
        masks = remove_border_labels_fn(masks)

    return masks


# Receive the two destinations separately because recursive input folders may be
# mirrored beneath each result category; inferring siblings from either path
# would put one export in the wrong branch.
def export_cell_segmentation_diagnostics(
    label_img: np.ndarray,
    base_img: np.ndarray,
    qc_png_dir: Path,
    table_dir: Path,
    base_name: str,
    cfg: CellSegmentationConfig,
) -> Dict[str, Any]:
    """Write contour and QC artifacts to their independently mirrored destinations."""

    results: Dict[str, Any] = {}

    if cfg.save_rois_csv:
        roi_path = table_dir / f"{base_name}_ROI_outlines.csv"
        save_roi_outline_csv(label_img, roi_path)
        results["roi_outlines_csv"] = str(roi_path)

    if cfg.save_qc_overlay:
        qc_img = build_qc_overlay(base_img, label_img, show_labels=cfg.show_labels_in_qc)
        qc_path = qc_png_dir / f"{base_name}_04_qc_overlay.png"
        save_pil_image(qc_img, qc_path)
        results["qc_overlay"] = str(qc_path)

    return results
