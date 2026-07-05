"""Voxel material-table generation for finite element mapping."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .coordinates import assign_grid_indices


VOXEL_COLUMNS = ["element_id", "x_um", "y_um", "E_GPa", "nu"]


def grid_to_voxel_material_table(
    df: pd.DataFrame,
    modulus_col: str = "E_GPa",
    nu: float | str | None = None,
) -> pd.DataFrame:
    """Convert a coordinate modulus grid to an ordered FEM material table."""

    if nu is None:
        if "nu" in df.columns:
            nu = "nu"
        else:
            raise ValueError("nu must be supplied explicitly or provided as a column.")
    required = ["x_um", "y_um", modulus_col]
    if isinstance(nu, str):
        required.append(nu)
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for voxel mapping: {missing}")
    mapped = assign_grid_indices(df[required].copy()).sort_values(["j", "i"]).reset_index(drop=True)
    out = pd.DataFrame(
        {
            "element_id": range(1, len(mapped) + 1),
            "x_um": mapped["x_um"].astype(float),
            "y_um": mapped["y_um"].astype(float),
            "E_GPa": mapped[modulus_col].astype(float),
        }
    )
    if isinstance(nu, str):
        out["nu"] = mapped[nu].astype(float).to_numpy()
    else:
        out["nu"] = float(nu)
    validate_voxel_material_table(out)
    return out[VOXEL_COLUMNS]


def validate_voxel_material_table(df: pd.DataFrame) -> None:
    """Validate a voxel material table."""

    missing = [column for column in VOXEL_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Voxel material table is missing columns: {missing}")
    expected_ids = list(range(1, len(df) + 1))
    if df["element_id"].astype(int).tolist() != expected_ids:
        raise ValueError("element_id must be consecutive and one-based.")
    if (df["E_GPa"] <= 0).any():
        raise ValueError("E_GPa must be positive for all voxels.")
    if ((df["nu"] <= 0) | (df["nu"] >= 0.5)).any():
        raise ValueError("nu must be between 0 and 0.5 for linear elasticity.")


def export_voxel_material_table(df: pd.DataFrame, path: str | Path) -> None:
    """Validate and write a voxel material table."""

    validate_voxel_material_table(df)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
