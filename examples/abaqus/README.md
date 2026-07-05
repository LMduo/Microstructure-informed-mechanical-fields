# Abaqus Material Assignment

Input CSV:

```text
element_id,x_um,y_um,E_GPa,nu
```

Dry run:

```bash
python examples/abaqus/assign_voxel_materials.py --csv path/to/voxel_material_table.csv --dry-run
```

In Abaqus/CAE:

```bash
abaqus cae noGUI=examples/abaqus/assign_voxel_materials.py -- --csv path/to/voxel_material_table.csv --model Model-1 --part Part-1 --expected-element-type CPE4
```

Default manuscript settings: CPE4 plane strain elements, displacement-controlled
compression, hard normal contact, and tangential penalty friction coefficient
0.2.
