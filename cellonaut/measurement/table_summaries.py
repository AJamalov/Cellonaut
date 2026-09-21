"""Summary and grouping rows for measurement exports."""

from __future__ import annotations

from collections.abc import Hashable
from typing import Any, Optional, cast

import pandas as pd

from cellonaut.config.defaults import INPUT_STRUCTURE_GROUPED_BY_PROTEIN
from cellonaut.results.labels import GROUP_AVERAGE_SUFFIX, SUMMARY_MEAN_LABEL, is_summary_label


# Match case-insensitively because sample filenames and user-entered tags may differ in case.
def row_matches_exclusion_tag(row_label: Optional[str], exclusion_tag: str) -> bool:
    if row_label is None or exclusion_tag is None:
        return False
    tag = str(exclusion_tag).strip()
    if not tag:
        return False
    return tag.lower() in str(row_label).lower()


# Insert visual group labels only in readable grouped exports, never in calculation tables.
def insert_group_header_rows(df: pd.DataFrame, input_structure: str) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if input_structure != INPUT_STRUCTURE_GROUPED_BY_PROTEIN or "Label" not in df.columns:
        return df

    out_rows = []
    numeric_cols = [col for col in df.columns if col != "Label"]
    work = df.copy()
    sample_rows = work[~work["Label"].fillna("").apply(is_summary_label)].copy()

    grouped: dict[str, list[pd.Series]] = {}
    for _, row in sample_rows.iterrows():
        label = str(row["Label"])
        protein = label.split("_", 1)[0] if "_" in label else "UNKNOWN"
        grouped.setdefault(protein, []).append(row)

    seen = []
    for label in sample_rows["Label"].tolist():
        label = str(label)
        protein = label.split("_", 1)[0] if "_" in label else "UNKNOWN"
        if protein not in seen:
            seen.append(protein)

    for protein in seen:
        header_row = {"Label": protein}
        for col in numeric_cols:
            header_row[col] = ""
        out_rows.append(header_row)

        avg_match = work[work["Label"] == f"{protein}{GROUP_AVERAGE_SUFFIX}"]
        if not avg_match.empty:
            out_rows.append(avg_match.iloc[0].to_dict())

        for row in grouped.get(protein, []):
            out_rows.append(row.to_dict())

    summary_match = work[work["Label"] == SUMMARY_MEAN_LABEL]
    if not summary_match.empty:
        out_rows.append(summary_match.iloc[0].to_dict())

    return pd.DataFrame(out_rows, columns=df.columns)


# Calculate the overall mean from sample rows while optionally excluding tagged controls.
def add_summary_mean_row(
    df: pd.DataFrame,
    label: str,
    exclude_matching_exclusion_tag: bool,
    exclusion_tag: str,
) -> pd.DataFrame:
    if df.empty:
        return df

    numeric_cols = [col for col in df.columns if col != "Label"]
    work = df.copy()
    if exclude_matching_exclusion_tag and str(exclusion_tag or "").strip():
        work = work[
            ~work["Label"].fillna("").apply(lambda row_label: row_matches_exclusion_tag(row_label, exclusion_tag))
        ]

    summary: dict[str, Any] = {"Label": label}
    for col in numeric_cols:
        values = pd.Series(pd.to_numeric(work[col], errors="coerce"), index=work.index)
        summary[str(col)] = values.mean()

    return pd.concat([df, pd.DataFrame([summary])], ignore_index=True)


# Group by the filename prefix used by the protein-folder input layout.
def add_per_folder_average_rows(
    df: pd.DataFrame,
    input_structure: str,
    exclusion_tag: str,
) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if input_structure != INPUT_STRUCTURE_GROUPED_BY_PROTEIN or "Label" not in df.columns:
        return df

    work = df.copy()
    labels = pd.Series(work["Label"], index=work.index).fillna("")
    sample_rows = cast(
        pd.DataFrame,
        work.loc[~labels.apply(is_summary_label)].copy(),
    )
    if sample_rows.empty:
        return work

    if str(exclusion_tag or "").strip():
        sample_labels = pd.Series(
            sample_rows["Label"],
            index=sample_rows.index,
        ).fillna("")
        sample_rows = cast(
            pd.DataFrame,
            sample_rows.loc[
                ~sample_labels.apply(lambda row_label: row_matches_exclusion_tag(row_label, exclusion_tag))
            ].copy(),
        )
    if sample_rows.empty:
        return work

    numeric_cols = [col for col in work.columns if col != "Label"]
    grouped: dict[str, list[dict[Hashable, Any]]] = {}
    for _, row in sample_rows.iterrows():
        label = str(row["Label"])
        protein = label.split("_", 1)[0] if "_" in label else "UNKNOWN"
        grouped.setdefault(protein, []).append(row.to_dict())

    avg_rows = []
    for protein, rows in grouped.items():
        group_df = pd.DataFrame(rows)
        avg_row: dict[str, Any] = {"Label": f"{protein}{GROUP_AVERAGE_SUFFIX}"}
        for col in numeric_cols:
            values = pd.Series(
                pd.to_numeric(group_df[col], errors="coerce"),
                index=group_df.index,
            )
            avg_row[str(col)] = values.mean()
        avg_rows.append(avg_row)

    if avg_rows:
        work = pd.concat([work, pd.DataFrame(avg_rows)], ignore_index=True)
    return work


# Put high-level summaries first for spreadsheet readers while preserving sample order.
def move_summary_rows_to_front(
    df: pd.DataFrame,
    label_col: str = "Label",
    summary_first_labels: Optional[list[str]] = None,
) -> pd.DataFrame:
    if df is None or df.empty or label_col not in df.columns:
        return df

    summary_first_labels = summary_first_labels or [SUMMARY_MEAN_LABEL]
    work = df.copy()
    sort_column = "_cellonaut_summary_sort"
    while sort_column in work.columns:
        sort_column = f"_{sort_column}"
    work[sort_column] = len(summary_first_labels)
    for index, label in enumerate(summary_first_labels):
        work.loc[work[label_col].astype(str) == label, sort_column] = index

    work = work.sort_values(by=[sort_column], kind="stable").drop(columns=[sort_column])
    return work.reset_index(drop=True)
