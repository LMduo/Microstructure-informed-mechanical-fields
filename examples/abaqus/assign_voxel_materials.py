"""Assign voxel-wise elastic materials in Abaqus from a CSV table."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class MaterialRow:
    element_id: int
    E_GPa: float
    nu: float


@dataclass(frozen=True)
class MaterialDefinition:
    name: str
    E_GPa: float
    nu: float


@dataclass(frozen=True)
class AssignmentPlan:
    materials: dict[str, MaterialDefinition]
    element_to_material: dict[int, str]


@dataclass(frozen=True)
class ManuscriptAbaqusSettings:
    """Elastic voxel-model settings reported in the manuscript."""

    element_type: str = "CPE4"
    kinematics: str = "plane_strain"
    loading: str = "displacement-controlled uniaxial compression"
    normal_contact: str = "hard"
    tangential_formulation: str = "penalty"
    friction_coefficient: float = 0.2


def read_material_rows(path: str | Path, nu_override: float | None = None) -> list[MaterialRow]:
    csv_path = Path(path)
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"element_id", "E_GPa", "nu"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{csv_path} is missing required columns: {sorted(missing)}")
        rows = []
        for raw in reader:
            element_id = int(float(raw["element_id"]))
            E_GPa = float(raw["E_GPa"])
            nu = float(nu_override) if nu_override is not None else float(raw["nu"])
            if element_id <= 0:
                raise ValueError(f"Invalid element_id {element_id}; labels must be positive.")
            if E_GPa <= 0:
                raise ValueError(f"Invalid E_GPa {E_GPa} for element {element_id}.")
            if not 0 < nu < 0.5:
                raise ValueError(f"Invalid nu {nu} for element {element_id}.")
            rows.append(MaterialRow(element_id=element_id, E_GPa=E_GPa, nu=nu))
    if not rows:
        raise ValueError(f"No material rows were read from {csv_path}.")
    return rows


def _material_name(prefix: str, E_GPa: float, nu: float) -> str:
    safe_E = f"{E_GPa:.6g}".replace(".", "p").replace("-", "m")
    safe_nu = f"{nu:.4g}".replace(".", "p")
    return f"{prefix}_E{safe_E}_nu{safe_nu}"


def build_assignment_plan(
    rows: Iterable[MaterialRow],
    strategy: str = "unique",
    bin_width_gpa: float = 0.5,
) -> AssignmentPlan:
    """Build material definitions and element-to-material assignments."""

    rows = list(rows)
    if strategy not in {"unique", "binned"}:
        raise ValueError("strategy must be 'unique' or 'binned'.")
    if bin_width_gpa <= 0:
        raise ValueError("bin_width_gpa must be positive.")

    materials: dict[str, MaterialDefinition] = {}
    element_to_material: dict[int, str] = {}
    for row in rows:
        if strategy == "unique":
            E_key = row.E_GPa
            name = f"voxel_{row.element_id:06d}"
        else:
            E_key = round(row.E_GPa / bin_width_gpa) * bin_width_gpa
            name = _material_name("bin", E_key, row.nu)
        if name not in materials:
            materials[name] = MaterialDefinition(name=name, E_GPa=float(E_key), nu=row.nu)
        element_to_material[row.element_id] = name
    return AssignmentPlan(materials=materials, element_to_material=element_to_material)


def validate_element_labels(csv_labels: Iterable[int], abaqus_labels: Iterable[int]) -> None:
    csv_set = set(csv_labels)
    abaqus_set = set(abaqus_labels)
    missing_in_part = sorted(csv_set.difference(abaqus_set))
    missing_in_csv = sorted(abaqus_set.difference(csv_set))
    if missing_in_part or missing_in_csv:
        message = []
        if missing_in_part:
            message.append(f"CSV labels not found in part: {missing_in_part[:10]}")
        if missing_in_csv:
            message.append(f"Part labels missing from CSV: {missing_in_csv[:10]}")
        raise ValueError("; ".join(message))


def validate_element_type(part, expected_element_type: str = "CPE4") -> None:
    """Validate the Abaqus element type when the API exposes it."""

    mismatched: list[tuple[int, str]] = []
    for element in part.elements:
        observed = getattr(element, "type", None)
        if observed is None:
            continue
        observed_text = str(observed)
        if expected_element_type not in observed_text:
            mismatched.append((element.label, observed_text))
    if mismatched:
        preview = ", ".join(f"{label}:{etype}" for label, etype in mismatched[:10])
        raise ValueError(
            f"Part contains elements that do not match {expected_element_type}: {preview}"
        )


def apply_plan_to_abaqus(
    plan: AssignmentPlan,
    model_name: str,
    part_name: str,
    elastic_modulus_scale: float = 1000.0,
    instance_name: str | None = None,
    expected_element_type: str = "CPE4",
    check_element_type: bool = True,
) -> None:
    """Apply an assignment plan inside Abaqus/CAE."""

    from abaqus import mdb  # type: ignore
    import regionToolset  # type: ignore

    model = mdb.models[model_name]
    part = model.parts[part_name]
    if instance_name is not None and instance_name not in model.rootAssembly.instances:
        raise ValueError(f"Assembly instance was not found: {instance_name}")

    part_labels = [element.label for element in part.elements]
    validate_element_labels(plan.element_to_material.keys(), part_labels)
    if check_element_type:
        validate_element_type(part, expected_element_type=expected_element_type)

    for material in plan.materials.values():
        if material.name not in model.materials:
            mat = model.Material(name=material.name)
            mat.Elastic(table=((material.E_GPa * elastic_modulus_scale, material.nu),))
        section_name = f"{material.name}_section"
        if section_name not in model.sections:
            model.HomogeneousSolidSection(name=section_name, material=material.name, thickness=None)

    for element_id, material_name in plan.element_to_material.items():
        section_name = f"{material_name}_section"
        element_sequence = part.elements.sequenceFromLabels((element_id,))
        region = regionToolset.Region(elements=element_sequence)
        part.SectionAssignment(region=region, sectionName=section_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="Voxel material CSV.")
    parser.add_argument("--model", default="Model-1", help="Abaqus model name.")
    parser.add_argument("--part", default="Part-1", help="Abaqus part name.")
    parser.add_argument("--instance", default=None, help="Optional assembly instance name to validate.")
    parser.add_argument("--strategy", choices=["unique", "binned"], default="unique")
    parser.add_argument("--bin-width-gpa", type=float, default=0.5)
    parser.add_argument("--nu", type=float, default=None, help="Override CSV Poisson's ratio.")
    parser.add_argument("--elastic-modulus-scale", type=float, default=1000.0)
    parser.add_argument("--expected-element-type", default="CPE4")
    parser.add_argument("--skip-element-type-check", action="store_true")
    parser.add_argument("--friction-coefficient", type=float, default=0.2)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_material_rows(args.csv, nu_override=args.nu)
    plan = build_assignment_plan(rows, strategy=args.strategy, bin_width_gpa=args.bin_width_gpa)
    manuscript_settings = ManuscriptAbaqusSettings(
        element_type=args.expected_element_type,
        friction_coefficient=args.friction_coefficient,
    )
    print(f"Read {len(rows)} voxel rows.")
    print(f"Prepared {len(plan.materials)} material definitions using '{args.strategy}' strategy.")
    print(f"Manuscript Abaqus settings: {manuscript_settings}")
    if args.dry_run:
        first_items = list(plan.element_to_material.items())[:5]
        print(f"First assignments: {first_items}")
        return
    apply_plan_to_abaqus(
        plan,
        model_name=args.model,
        part_name=args.part,
        elastic_modulus_scale=args.elastic_modulus_scale,
        instance_name=args.instance,
        expected_element_type=args.expected_element_type,
        check_element_type=not args.skip_element_type_check,
    )
    print("Abaqus material assignment completed.")


if __name__ == "__main__":
    main()
