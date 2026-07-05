"""Quality-control helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd


@dataclass(frozen=True)
class QCReport:
    """Summary of a deterministic quality-control operation."""

    input_rows: int
    output_rows: int
    removed_rows: int
    criteria: Mapping[str, str]

    def as_dict(self) -> dict[str, object]:
        return {
            "input_rows": self.input_rows,
            "output_rows": self.output_rows,
            "removed_rows": self.removed_rows,
            "criteria": dict(self.criteria),
        }


def filter_physical_bounds(
    df: pd.DataFrame,
    bounds: Mapping[str, tuple[float | None, float | None]],
) -> tuple[pd.DataFrame, QCReport]:
    """Filter rows using explicit physical bounds.

    Parameters
    ----------
    df:
        Input table.
    bounds:
        Mapping from column name to ``(lower, upper)``. Use ``None`` for an
        open bound. Bounds are inclusive.
    """

    mask = pd.Series(True, index=df.index)
    criteria: dict[str, str] = {}
    for column, (lower, upper) in bounds.items():
        if column not in df.columns:
            raise ValueError(f"Cannot filter missing column: {column}")
        column_mask = df[column].notna()
        text_parts: list[str] = ["not missing"]
        if lower is not None:
            column_mask &= df[column] >= lower
            text_parts.append(f">= {lower:g}")
        if upper is not None:
            column_mask &= df[column] <= upper
            text_parts.append(f"<= {upper:g}")
        mask &= column_mask
        criteria[column] = " and ".join(text_parts)

    filtered = df.loc[mask].copy().reset_index(drop=True)
    report = QCReport(
        input_rows=len(df),
        output_rows=len(filtered),
        removed_rows=len(df) - len(filtered),
        criteria=criteria,
    )
    return filtered, report


def remove_duplicate_coordinates(
    df: pd.DataFrame,
    coordinate_columns: tuple[str, str] = ("x_um", "y_um"),
    keep: str = "first",
) -> tuple[pd.DataFrame, QCReport]:
    """Remove duplicate coordinate rows."""

    for column in coordinate_columns:
        if column not in df.columns:
            raise ValueError(f"Cannot check duplicates without column: {column}")
    duplicate_mask = df.duplicated(list(coordinate_columns), keep=keep)
    filtered = df.loc[~duplicate_mask].copy().reset_index(drop=True)
    report = QCReport(
        input_rows=len(df),
        output_rows=len(filtered),
        removed_rows=int(duplicate_mask.sum()),
        criteria={
            "duplicates": (
                f"drop duplicate rows by {coordinate_columns[0]},{coordinate_columns[1]} "
                f"keeping {keep}"
            )
        },
    )
    return filtered, report
