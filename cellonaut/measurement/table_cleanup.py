"""Shared cleanup helpers for generated measurement tables."""

from __future__ import annotations

import re
from typing import Optional

import pandas as pd


_DERIVED_RATIO_COLUMN = re.compile(
    r"(?:Circularity|Solidity|Roundness|AspectRatio|AreaFraction(?:_InCell)?|"
    r"PositiveAreaFractionInCell|FractionOfCell(?:Corrected)?IntDen|"
    r"(?:Corrected)?IntDenPerCellArea|(?:Corrected)?MeanRatio|"
    r"(?:Corrected)?IntDenRatio)$"
)


def drop_derived_ratio_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Keep ratio metrics available to filters without writing them as measurements."""
    return df.drop(columns=[column for column in df.columns if _DERIVED_RATIO_COLUMN.search(str(column))])


# Remove empty export noise while retaining identifier columns required to understand the table.
def drop_empty_rows_and_columns(df: pd.DataFrame, *, protected_columns: Optional[list[str]] = None) -> pd.DataFrame:
    if df is None or df.empty:
        return df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()
    protected = set(protected_columns or [])
    work = df.copy()
    work = work.dropna(axis=0, how="all")
    drop_cols = []
    for col in work.columns:
        if str(col) in protected:
            continue
        values = pd.Series(work[col], index=work.index)
        if values.isna().all():
            drop_cols.append(col)
            continue
        text_values = values.map(lambda value: str(value).strip())
        if ((text_values == "") | (text_values.map(str.lower) == "nan")).all():
            drop_cols.append(col)
    if drop_cols:
        work = work.drop(columns=drop_cols)
    return work
