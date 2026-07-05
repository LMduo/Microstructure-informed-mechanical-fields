"""CSV readers and schema checks for released data tables."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def require_columns(df: pd.DataFrame, columns: Iterable[str], table_name: str = "table") -> None:
    """Raise a clear error if required columns are missing."""

    required = list(columns)
    missing = [column for column in required if column not in df.columns]
    if missing:
        available = ", ".join(map(str, df.columns))
        raise ValueError(
            f"{table_name} is missing required columns {missing}. "
            f"Available columns: {available}"
        )


def load_csv(path: str | Path, required_columns: Iterable[str] | None = None) -> pd.DataFrame:
    """Load a CSV file and check required columns when requested."""

    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file does not exist: {csv_path}")
    df = pd.read_csv(csv_path)
    if required_columns is not None:
        require_columns(df, required_columns, table_name=csv_path.name)
    return df


def load_modulus_hardness(path: str | Path) -> pd.DataFrame:
    """Load modulus-hardness pairs exported from the indentation workflow."""

    columns = ["measurement_id", "modulus_GPa", "hardness_GPa"]
    df = load_csv(path, columns)
    return df[columns].copy()


def load_indentation_grid(path: str | Path) -> pd.DataFrame:
    """Load a spatial modulus grid with coordinates in micrometres."""

    df = load_csv(path, ["x_um", "y_um", "E_GPa"])
    columns = [column for column in ["point_id", "grid_id", "x_um", "y_um", "E_GPa", "nu"] if column in df.columns]
    return df[columns].copy()


def load_voxel_material_table(path: str | Path) -> pd.DataFrame:
    """Load a voxel material table for FEM assignment."""

    columns = ["element_id", "x_um", "y_um", "E_GPa", "nu"]
    df = load_csv(path, columns)
    return df[columns].copy()
