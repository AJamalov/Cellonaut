from __future__ import annotations

import pandas as pd

from cellonaut.config.defaults import INPUT_STRUCTURE_GROUPED_BY_PROTEIN
from cellonaut.measurement.table_summaries import insert_group_header_rows, move_summary_rows_to_front


def test_group_headers_keep_first_group_order_and_place_summaries_consistently():
    table = pd.DataFrame(
        {
            "Label": [
                "ProteinB_cell1",
                "ProteinA_cell1",
                "ProteinB_cell2",
                "ProteinA_AVERAGE",
                "ProteinB_AVERAGE",
                "SUMMARY_MEAN",
            ],
            "Signal_Mean": [8.0, 2.0, 12.0, 2.0, 10.0, 6.0],
        }
    )

    result = insert_group_header_rows(table, INPUT_STRUCTURE_GROUPED_BY_PROTEIN)

    assert result["Label"].tolist() == [
        "ProteinB",
        "ProteinB_AVERAGE",
        "ProteinB_cell1",
        "ProteinB_cell2",
        "ProteinA",
        "ProteinA_AVERAGE",
        "ProteinA_cell1",
        "SUMMARY_MEAN",
    ]
    assert result.loc[result["Label"].isin(["ProteinA", "ProteinB"]), "Signal_Mean"].tolist() == ["", ""]


def test_group_headers_accept_non_string_labels_as_unknown_group():
    table = pd.DataFrame({"Label": [101, "SUMMARY_MEAN"], "Signal_Mean": [4.0, 4.0]})

    result = insert_group_header_rows(table, INPUT_STRUCTURE_GROUPED_BY_PROTEIN)

    assert result["Label"].tolist() == ["UNKNOWN", 101, "SUMMARY_MEAN"]


def test_group_headers_leave_non_grouped_or_incomplete_tables_unchanged():
    table = pd.DataFrame({"Label": ["Sample_1"], "Value": [1]})
    missing_label = pd.DataFrame({"Value": [1]})

    assert insert_group_header_rows(table, "flat") is table
    assert insert_group_header_rows(missing_label, INPUT_STRUCTURE_GROUPED_BY_PROTEIN) is missing_label


def test_summary_rows_move_to_front_in_requested_order_and_keep_sample_order():
    table = pd.DataFrame(
        {
            "Label": ["Sample_B", "SUMMARY_MEAN", "Sample_A", "ProteinA_AVERAGE"],
            "Value": [2, 3, 1, 4],
        }
    )

    result = move_summary_rows_to_front(table, summary_first_labels=["ProteinA_AVERAGE", "SUMMARY_MEAN"])

    assert result["Label"].tolist() == ["ProteinA_AVERAGE", "SUMMARY_MEAN", "Sample_B", "Sample_A"]


def test_summary_sort_preserves_user_columns_that_resemble_internal_sort_keys():
    table = pd.DataFrame(
        {
            "Label": ["Sample", "SUMMARY_MEAN"],
            "_sort_key": ["sample metadata", "summary metadata"],
            "_cellonaut_summary_sort": [9, 8],
        }
    )

    result = move_summary_rows_to_front(table)

    assert result.columns.tolist() == table.columns.tolist()
    assert result["_sort_key"].tolist() == ["summary metadata", "sample metadata"]
    assert result["_cellonaut_summary_sort"].tolist() == [8, 9]
