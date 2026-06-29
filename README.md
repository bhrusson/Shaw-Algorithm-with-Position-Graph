# BQSKit Shuttling Algorithms

This branch is a publishable code snapshot for the QCCD shuttling mapping work,
including the SHAW implementation and cached SABRE experiments.

The verified QCCD command-line entry point in this snapshot is `run_qccd.py`.
It currently exposes the SHAW/PGS workflow. SHAPER/PAM support is not wired into
the public runner on this branch yet.

## Layout

- `bqskit/shuttling/qccd/`: PGS-native SHAW/QCCD machine models, mapping passes,
  scheduling helpers, and benchmark QASM inputs.
- `bqskit/mapping/shaw.py`: standalone SHAW mapping algorithm entry point.
- `bqskit/superconducting/`: cached SABRE mapping cores for CouplingGraph and PositionGraph,
  benchmark drivers, Qiskit comparison helpers, and small smoke checks.
- `examples/qccd/`: a minimal runnable QCCD SHAW/PGS example.

Generated logs, profiles, compiled result dumps, paper-output directories, IDE
metadata, and notebook scratch files have been removed from this branch.

## Install

This repository uses `setup.py` for package metadata and editable installs.

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
python -m bqskit.superconducting.testCasesPGS.benchmark_sabre_pgs QAOA_wsq_8_compiled --architecture grid-3x3
```

Run a small QCCD SHAW/PGS grid case:

```powershell
python run_qccd.py QAOA_wsq_8_compiled --algorithm shaw --grid-cols 3 --grid-rows 3 --trap-capacity 3 --num-layout-passes 1 --summary-only
```

The `run_qccd.py` CLI accepts benchmark stems, `.qasm` filenames, or explicit
paths. Use `--benchmark-dir` or `BQSKIT_SHUTTLING_BENCHMARK_DIR` to point it at
another benchmark directory.

Run the example script:

```powershell
python examples/qccd/quickstart.py
```

Save compiled QASM and pickle output under `outputs/qccd/`:

```powershell
python run_qccd.py QAOA_wsq_8_compiled --algorithm shaw --grid-cols 3 --grid-rows 3 --trap-capacity 3 --num-layout-passes 1 --summary-only --save-results
```

## License

This project is released under the MIT License. See `LICENSE`.
