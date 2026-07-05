"""DBSCAN-based high-modulus core identification."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN


@dataclass(frozen=True)
class DBSCANResult:
    """Cluster labels and summary for high-modulus points."""

    labeled_points: pd.DataFrame
    cluster_summary: pd.DataFrame
    threshold_GPa: float
    n_candidates: int
    n_clusters: int
    n_noise: int


def identify_high_modulus_cores(
    df: pd.DataFrame,
    modulus_col: str = "E_GPa",
    coordinate_cols: tuple[str, str] = ("x_um", "y_um"),
    percentile: float = 90.0,
    eps_um: float = 15.0,
    min_samples: int = 3,
) -> DBSCANResult:
    """Identify spatial clusters among the highest-modulus points."""

    required = [modulus_col, *coordinate_cols]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required DBSCAN columns: {missing}")
    if not 0 < percentile < 100:
        raise ValueError("percentile must lie between 0 and 100.")
    if eps_um <= 0:
        raise ValueError("eps_um must be positive.")
    if min_samples < 1:
        raise ValueError("min_samples must be at least 1.")

    clean = df.loc[:, required].dropna().copy()
    if clean.empty:
        raise ValueError("No finite rows are available for DBSCAN.")
    threshold = float(np.percentile(clean[modulus_col], percentile))
    candidates = clean.loc[clean[modulus_col] >= threshold].copy().reset_index(drop=True)
    if candidates.empty:
        raise ValueError("No high-modulus candidates were found.")

    labels = DBSCAN(eps=eps_um, min_samples=min_samples).fit_predict(
        candidates.loc[:, list(coordinate_cols)].to_numpy(dtype=float)
    )
    candidates["cluster_id"] = labels
    n_clusters = int(len(set(labels)) - (1 if -1 in labels else 0))
    n_noise = int(np.sum(labels == -1))
    x_col, y_col = coordinate_cols
    clustered = candidates[candidates["cluster_id"] >= 0]
    if clustered.empty:
        cluster_summary = pd.DataFrame(
            columns=[
                "cluster_id",
                "n_points",
                f"{x_col}_centroid",
                f"{y_col}_centroid",
                f"{modulus_col}_mean",
                f"{modulus_col}_max",
            ]
        )
    else:
        cluster_summary = (
            clustered.groupby("cluster_id", as_index=False)
            .agg(
                n_points=(modulus_col, "size"),
                **{
                    f"{x_col}_centroid": (x_col, "mean"),
                    f"{y_col}_centroid": (y_col, "mean"),
                    f"{modulus_col}_mean": (modulus_col, "mean"),
                    f"{modulus_col}_max": (modulus_col, "max"),
                },
            )
            .sort_values(["n_points", f"{modulus_col}_mean"], ascending=[False, False])
            .reset_index(drop=True)
        )
    return DBSCANResult(
        labeled_points=candidates,
        cluster_summary=cluster_summary,
        threshold_GPa=threshold,
        n_candidates=len(candidates),
        n_clusters=n_clusters,
        n_noise=n_noise,
    )
