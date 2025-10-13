# sim2d_3 — Minimal 2D LBM with Geometry and Instantaneous KE Output

- Loads a geometry file specifying per-cell types and Bouzidi thetas.
- Supports toggling Bouzidi near-wall interpolation on/off.
- Uses parabolic inflow profile.
- Produces a single output value: the integral of the instantaneous kinetic energy over the third quarter of the domain (x in [0.5W, 0.75W], across the interior in y).

## CLI

```
build/sim_2D/sim2d_3 <resolution> <geometry_file> [--no-bouzidi] [--type1-bouzidi on|off|auto]
```

- `resolution`: integer scaling of the default domain.
- `geometry_file`: path or basename under `sim_2D/ellipses` (e.g., `8.txt`).
- `--no-bouzidi`: disable Bouzidi interpolation (treat type-1 as fluid).
- `--type1-bouzidi`: controls mapping of geometry type `1` cells:
  - `on`  → map to `GEO_FLUID_NEAR_WALL` (Bouzidi near-wall)
  - `off` → map to `GEO_FLUID`
  - `auto` → uses the build default (on)

## Output

At the end of the simulation, the solver writes one ASCII number to:

```
sim_2D/values/value_<geometry_filename>
```

The value is the integral of `0.5*(u^2+v^2)` in SI units (m^2/s^2 integrated over area, i.e., multiplied by cell area) over x ∈ [0.5W, 0.75W] and interior y ∈ [1, Ny-2], restricted to fluid cells.

## Using run_lbm_simulation.py

`run_lbm_simulation.py` can be pointed at `sim2d_3` directly:

CLI:

```
python run_lbm_simulation.py 32.txt 8 --binary build/sim_2D/sim2d_3 --wait
```

- You can also pass a path relative to the repo root (e.g. `sim_2D/sim2d_3`); the script will try `build/` prefixed path automatically.

Python API:

```
run_lbm_simulation.prepare_submission(
    geometry="32.txt",
    resolution=8,
    solver_binary=Path("build/sim_2D/sim2d_3"),
)
```

No changes are needed to result collection because `sim2d_3` writes the value to the same `sim_2D/values/value_<geometry>` path the script reads.
