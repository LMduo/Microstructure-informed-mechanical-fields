# FEniCSx Linear Elasticity

FEniCSx is imported through its Python module name, `dolfinx`.

Input CSV:

```text
element_id,x_um,y_um,E_GPa,nu
```

Run in a FEniCSx/dolfinx environment:

```bash
python examples/fenicsx_linear_elasticity/linear_elastic_voxel_demo.py --csv path/to/voxel_material_table.csv --nx 100 --ny 100
```

This example is limited to heterogeneous linear elastic voxel mapping. It does
not include phase-field fracture or cohesive-zone modeling.
