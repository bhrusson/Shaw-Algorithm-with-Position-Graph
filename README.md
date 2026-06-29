# BQSKit Shuttling Algorithms

This branch is a publishable code snapshot for the QCCD shuttling mapping work,
including the SHAW implementation and cached SABRE experiments.

## Layout

- `bqskit/shuttling/qccd/`: PGS-native SHAW/QCCD machine models, mapping passes,
  scheduling helpers, and benchmark QASM inputs.
- `bqskit/mapping/shaw.py`: standalone SHAW mapping algorithm entry point.
- `bqskit/superconducting/`: cached SABRE mapping cores for CouplingGraph and PositionGraph,
  benchmark drivers, Qiskit comparison helpers, and small smoke checks.

Generated logs, profiles, compiled result dumps, paper-output directories, IDE
metadata, and notebook scratch files have been removed from this branch.

## Install

```powershell
python -m pip install -e .
```

The required runtime dependencies are `bqskit`, `numpy`, and `rustworkx`.

## Quick Checks

Import the main modules:

```powershell
python -c "import bqskit.mapping.shaw; import bqskit.superconducting.mapping.cached_sabre; import bqskit.superconducting.mapping.sabre_pgs"
```

Run the local SABRE/PGS executability smoke check:

```powershell
python -m bqskit.superconducting.testCasesPGS.multiqudit_can_exe_smoke --quiet
```

Run a small SABRE benchmark:

```powershell
python -m bqskit.superconducting.testCasesPGS.benchmark_sabre_pgs QAOA_wsq_8_compiled --architecture grid-3x3 --algorithm sabre
```

Run a small SHAW/PGS grid case:

```powershell
python -m bqskit.shuttling.qccd.run_grid_pgs_shaw QAOA_wsq_8_compiled --grid-cols 3 --grid-rows 3 --trap-capacity 3
```
