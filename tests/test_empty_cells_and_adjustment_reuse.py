"""Regression coverage for final-mask reuse and empty-cell sample totals."""
from types import SimpleNamespace as NS
from typing import Any

import numpy as np
import pandas as pd
import pytest

from cellonaut.masks import cellpose as cp
from cellonaut.measurement.table_formatting import tidy_measurements_table, measurement_summary_table


@pytest.mark.parametrize('adjustments', [{'dx': 1}, {'grow_px': 1}])
def test_reused_final_labels_do_not_accumulate_adjustments(tmp_path, monkeypatch, adjustments):
    dirs = cp.build_common_results_export_dirs(tmp_path)
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    cfg = NS(output_dir=tmp_path, reuse_existing_masks=False, cell_mask_adjustments=adjustments)
    labels = np.zeros((9, 9), dtype=np.int32)
    labels[4, 4] = 1
    monkeypatch.setattr(cp, 'build_cell_segmentation_cfg_from_pipeline', lambda _: NS())
    monkeypatch.setattr(cp, 'run_cell_segmentation_on_image', lambda *_args, **_kwargs: labels)
    args: dict[str, Any] = dict(cfg=cfg, cell_source_img=np.ones((9, 9)), cell_source_label='BF',
                source_def=NS(label='GFP'), export_dirs=dirs,
                result_id='sample', id_label='sample', log_func=lambda _: None, should_cancel=None)
    first = cp._load_or_generate_cell_labels(**args)[0]
    assert not np.array_equal(first, labels)
    cfg.reuse_existing_masks = True
    for _ in range(2):
        reused = cp._load_or_generate_cell_labels(**args)[0]
        np.testing.assert_array_equal(reused, first)


def test_empty_cell_sample_exports_and_contributes_zero_totals(tmp_path, monkeypatch):
    dirs = cp.build_common_results_export_dirs(tmp_path)
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(cp, '_load_or_generate_cell_labels', lambda **_: (
        np.zeros((4, 4), dtype=np.int32), np.ones((4, 4)),
        NS(save_rois_csv=False, save_qc_overlay=False)))
    row = {'Label': 'empty'}
    options = dict.fromkeys(cp.WHOLE_CELL_MEASUREMENT_COLUMNS, True)
    cp.run_cell_segmentation_for_sample(
        cfg=NS(do_cell_segmentation=True, cell_segmentation_source='bf', images=[NS(key='bf', label='BF')], measurement_options=options),
        image_map={'bf': np.ones((4, 4))}, roi_map={}, roi_defs=[], row=row,
        source_def=NS(label='GFP'), source_measure_img=np.ones((4, 4)),
        measurement_source_img=np.ones((4, 4)), has_source_background_subtraction=False,
        export_dirs=dirs, result_id='empty', id_label='empty',
        log_func=lambda _: None)
    key = 'GFP_measured_with_BF_cellpose_mask_PerCell_TotalCellArea'
    assert row[key] == 0
    assert row['GFP_CellCount'] == 0
    assert 'GFP_measured_with_BF_cellpose_mask_PerCell_MeanOfCellMeans' not in row
    saved = pd.read_csv(dirs['cell_segmentation_tables'] / 'empty_GFP_cell_measurements.csv')
    assert saved.empty and 'CellID' in saved.columns
    summary = measurement_summary_table(tidy_measurements_table(
        pd.DataFrame([row, {'Label': 'positive', key: 100}]), sample_column='Label'))
    area = summary.loc[summary['Measurement'] == key].iloc[0]
    assert area['N'] == 2 and area['Mean'] == 50


def test_empty_mask_signal_totals_keep_schema_and_selection(tmp_path):
    from cellonaut.cell_segmentation.core import export_per_cell_organelle_signal_tables
    from cellonaut.measurement.math import summarize_per_cell_table

    output = tmp_path / 'signal.csv'
    table, _ = export_per_cell_organelle_signal_tables(
        cell_label_img=np.zeros((4, 4), dtype=np.int32), organelle_mask=np.ones((4, 4)),
        intensity_img=np.ones((4, 4)), out_biology_csv=output, organelle_prefix='Mask',
        measurement_options={'positive_area_in_cell': True, 'raw_intden_in_cell': True})
    summary = summarize_per_cell_table(table, 'Mask', 'GFP', 'BF')
    assert summary['GFP_measured_with_Mask_mask_PerCell_TotalMaskArea'] == 0
    assert summary['GFP_measured_with_Mask_mask_PerCell_SumIntDen'] == 0
    assert 'GFP_measured_with_BF_cellpose_mask_PerCell_TotalCellArea' not in summary
    assert 'CellID' in pd.read_csv(output).columns
