"""Linear-elastic FEniCSx voxel material mapping.

The public FEniCSx Python interface is provided by the ``dolfinx`` module.
Run this example in a FEniCSx/dolfinx Python environment.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from mpi4py import MPI
from petsc4py import PETSc
import dolfinx
from dolfinx import fem, mesh
from dolfinx.fem.petsc import LinearProblem
import ufl


@dataclass(frozen=True)
class FenicsxResult:
    """Small set of scalar diagnostics from the linear-elastic solve."""

    applied_strain: float
    average_sigma_xx_GPa: float
    effective_modulus_GPa: float
    voigt_average_GPa: float


def load_voxel_table(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = ["element_id", "x_um", "y_um", "E_GPa", "nu"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Voxel table is missing columns: {missing}")
    df = df[required].copy()
    if df.duplicated(["x_um", "y_um"]).any():
        raise ValueError("Voxel table contains duplicate x_um,y_um coordinates.")
    if (df["E_GPa"] <= 0).any():
        raise ValueError("E_GPa must be positive.")
    if ((df["nu"] <= 0) | (df["nu"] >= 0.5)).any():
        raise ValueError("nu must be between 0 and 0.5.")
    return df.sort_values(["y_um", "x_um"]).reset_index(drop=True)


def voigt_average_modulus(df: pd.DataFrame) -> float:
    """Return the arithmetic average modulus as a reference value."""

    return float(df["E_GPa"].mean())


def regular_axis(values: pd.Series, name: str) -> tuple[np.ndarray, float]:
    axis = np.array(sorted(values.unique()), dtype=float)
    if len(axis) < 2:
        raise ValueError(f"Need at least two unique {name} coordinates.")
    spacing = float(np.median(np.diff(axis)))
    if spacing <= 0:
        raise ValueError(f"{name} spacing must be positive.")
    if not np.allclose(np.diff(axis), spacing, rtol=0.0, atol=1e-9):
        raise ValueError(f"{name} coordinates must form a regular grid.")
    return axis, spacing


def lame_parameters(E: np.ndarray, nu: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray]:
    if mode == "plane_stress":
        lambda_ = E * nu / (1.0 - nu**2)
    elif mode == "plane_strain":
        lambda_ = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    else:
        raise ValueError("mode must be 'plane_stress' or 'plane_strain'.")
    mu = E / (2.0 * (1.0 + nu))
    return lambda_, mu


def assign_cellwise_materials(df: pd.DataFrame, domain, E_func, nu_func) -> None:
    """Map the nearest voxel values to DG0 cells."""

    x_axis, dx = regular_axis(df["x_um"], "x_um")
    y_axis, dy = regular_axis(df["y_um"], "y_um")
    E_grid = df.pivot(index="y_um", columns="x_um", values="E_GPa").sort_index().sort_index(axis=1)
    nu_grid = df.pivot(index="y_um", columns="x_um", values="nu").sort_index().sort_index(axis=1)

    tdim = domain.topology.dim
    num_cells = domain.topology.index_map(tdim).size_local
    cell_centers = dolfinx.mesh.compute_midpoints(domain, tdim, np.arange(num_cells))
    ix = np.clip(np.floor(cell_centers[:, 0] / dx).astype(int), 0, len(x_axis) - 1)
    iy = np.clip(np.floor(cell_centers[:, 1] / dy).astype(int), 0, len(y_axis) - 1)

    E_func.x.array[:num_cells] = E_grid.to_numpy()[iy, ix]
    nu_func.x.array[:num_cells] = nu_grid.to_numpy()[iy, ix]
    E_func.x.scatter_forward()
    nu_func.x.scatter_forward()


def solve_with_fenicsx(
    df: pd.DataFrame,
    mode: str = "plane_strain",
    nx: int = 100,
    ny: int = 100,
    strain: float = 0.001,
) -> FenicsxResult:
    """Solve a small heterogeneous linear-elasticity problem with FEniCSx."""

    if nx <= 0 or ny <= 0:
        raise ValueError("nx and ny must be positive.")
    if strain <= 0:
        raise ValueError("strain must be positive.")

    x_axis, dx = regular_axis(df["x_um"], "x_um")
    y_axis, dy = regular_axis(df["y_um"], "y_um")
    width = float((x_axis.max() - x_axis.min()) + dx)
    height = float((y_axis.max() - y_axis.min()) + dy)

    domain = mesh.create_rectangle(
        MPI.COMM_WORLD,
        [np.array([0.0, 0.0]), np.array([width, height])],
        [nx, ny],
        cell_type=mesh.CellType.quadrilateral,
    )
    V = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim,)))
    Q = fem.functionspace(domain, ("DG", 0))

    E_func = fem.Function(Q, name="E_GPa")
    nu_func = fem.Function(Q, name="nu")
    assign_cellwise_materials(df, domain, E_func, nu_func)

    lambda_values, mu_values = lame_parameters(E_func.x.array, nu_func.x.array, mode)
    lambda_func = fem.Function(Q, name="lambda")
    mu_func = fem.Function(Q, name="mu")
    lambda_func.x.array[:] = lambda_values
    mu_func.x.array[:] = mu_values

    def eps(u):
        return ufl.sym(ufl.grad(u))

    def sigma(u):
        return 2.0 * mu_func * eps(u) + lambda_func * ufl.tr(eps(u)) * ufl.Identity(2)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(sigma(u), eps(v)) * ufl.dx
    zero = fem.Constant(domain, PETSc.ScalarType((0.0, 0.0)))
    L = ufl.dot(zero, v) * ufl.dx

    left_facets = mesh.locate_entities_boundary(domain, 1, lambda x: np.isclose(x[0], 0.0))
    right_facets = mesh.locate_entities_boundary(domain, 1, lambda x: np.isclose(x[0], width))
    left_dofs = fem.locate_dofs_topological(V, 1, left_facets)
    right_dofs_x = fem.locate_dofs_topological(V.sub(0), 1, right_facets)
    bc_left = fem.dirichletbc(np.array((0.0, 0.0), dtype=PETSc.ScalarType), left_dofs, V)
    bc_right = fem.dirichletbc(PETSc.ScalarType(-strain * width), right_dofs_x, V.sub(0))

    problem = LinearProblem(
        a,
        L,
        bcs=[bc_left, bc_right],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    )
    displacement = problem.solve()
    displacement.name = "u"

    area_local = fem.assemble_scalar(fem.form(1.0 * ufl.dx(domain)))
    sigma_xx_local = fem.assemble_scalar(fem.form(sigma(displacement)[0, 0] * ufl.dx))
    area = domain.comm.allreduce(area_local, op=MPI.SUM)
    sigma_xx_integral = domain.comm.allreduce(sigma_xx_local, op=MPI.SUM)
    average_sigma_xx = float(sigma_xx_integral / area)
    effective_modulus = abs(average_sigma_xx / strain)

    return FenicsxResult(
        applied_strain=float(strain),
        average_sigma_xx_GPa=average_sigma_xx,
        effective_modulus_GPa=float(effective_modulus),
        voigt_average_GPa=voigt_average_modulus(df),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--mode", choices=["plane_stress", "plane_strain"], default="plane_strain")
    parser.add_argument("--nx", type=int, default=100)
    parser.add_argument("--ny", type=int, default=100)
    parser.add_argument("--strain", type=float, default=0.001)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_voxel_table(args.csv)
    print(f"Loaded {len(df)} voxel material rows.")
    print(f"Voigt-average modulus: {voigt_average_modulus(df):.6g} GPa")
    result = solve_with_fenicsx(
        df,
        mode=args.mode,
        nx=args.nx,
        ny=args.ny,
        strain=args.strain,
    )
    print(f"Applied strain: {result.applied_strain:.6g}")
    print(f"Average sigma_xx: {result.average_sigma_xx_GPa:.6g} GPa")
    print(f"Effective modulus estimate: {result.effective_modulus_GPa:.6g} GPa")


if __name__ == "__main__":
    main()
