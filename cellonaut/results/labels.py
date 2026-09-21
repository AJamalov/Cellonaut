"""Reserved labels that distinguish generated summaries from real samples."""

from __future__ import annotations


SUMMARY_MEAN_LABEL = "SUMMARY_MEAN"
GROUP_AVERAGE_SUFFIX = "_AVERAGE"


# Identify group-average rows by their reserved suffix.
def is_group_average_label(label: str | None) -> bool:
    return label is not None and str(label).endswith(GROUP_AVERAGE_SUFFIX)


# Summary markers must be excluded from later calculations or repeated export
# passes would count already-aggregated values as if they were samples.
def is_summary_label(label: str | None) -> bool:
    return label is not None and (str(label) == SUMMARY_MEAN_LABEL or is_group_average_label(label))
