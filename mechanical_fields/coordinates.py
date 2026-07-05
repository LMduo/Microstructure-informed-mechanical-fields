"""Coordinate and grid utilities for indentation maps."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GridInfo:
    """Description of a regular 2D grid."""

    x_values: np.ndarray
    y_values: np.ndarray
    dx: float
    dy: float

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.y_values), len(self.x_values))


def _unique_sorted(values: pd.Series) -> np.ndarray:
    return np.array(sorted(pd.Series(values).dropna().unique()), dtype=float)


def _spacing(values: np.ndarray, axis_name: str, tolerance: float) -> float:
    if len(values) < 2:
        raise ValueError(f"Need at least two unique {axis_name} coordinates.")
    diffs = np.diff(values)
    spacing = float(np.median(diffs))
    if not np.allclose(diffs, spacing, atol=tolerance, rtol=0):
        raise ValueError(
            f"{axis_name} coordinates are not regular within tolerance {tolerance:g}."
        )
    return spacing


def infer_regular_grid(
    df: pd.DataFrame,
    x_col: str = "x_um",
    y_col: str = "y_um",
    tolerance: float = 1e-9,
) -> GridInfo:
    """Infer regular grid coordinates and spacing."""

    for column in (x_col, y_col):
        if column not in df.columns:
            raise ValueError(f"Missing coordinate column: {column}")
    x_values = _unique_sorted(df[x_col])
    y_values = _unique_sorted(df[y_col])
    return GridInfo(
        x_values=x_values,
        y_values=y_values,
        dx=_spacing(x_values, x_col, tolerance),
        dy=_spacing(y_values, y_col, tolerance),
    )


def assign_grid_indices(
    df: pd.DataFrame,
    x_col: str = "x_um",
    y_col: str = "y_um",
    tolerance: float = 1e-9,
) -> pd.DataFrame:
    """Add zero-based integer ``i`` and ``j`` grid indices."""

    grid = infer_regular_grid(df, x_col=x_col, y_col=y_col, tolerance=tolerance)
    x_lookup = {value: index for index, value in enumerate(grid.x_values)}
    y_lookup = {value: index for index, value in enumerate(grid.y_values)}
    out = df.copy()
    out["i"] = out[x_col].map(x_lookup).astype(int)
    out["j"] = out[y_col].map(y_lookup).astype(int)
    return out


def split_quadrants(
    df: pd.DataFrame,
    x_col: str = "x_um",
    y_col: str = "y_um",
) -> dict[str, pd.DataFrame]:
    """Split a coordinate table into lower-left/right and upper-left/right quadrants."""

    for column in (x_col, y_col):
        if column not in df.columns:
            raise ValueError(f"Missing coordinate column: {column}")
    x_mid = (float(df[x_col].min()) + float(df[x_col].max())) / 2.0
    y_mid = (float(df[y_col].min()) + float(df[y_col].max())) / 2.0
    return {
        "lower_left": df[(df[x_col] <= x_mid) & (df[y_col] <= y_mid)].copy(),
        "lower_right": df[(df[x_col] > x_mid) & (df[y_col] <= y_mid)].copy(),
        "upper_left": df[(df[x_col] <= x_mid) & (df[y_col] > y_mid)].copy(),
        "upper_right": df[(df[x_col] > x_mid) & (df[y_col] > y_mid)].copy(),
    }
