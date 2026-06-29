# QCCD Examples

This folder contains small examples that exercise the public QCCD runner.

Run the quickstart from the repository root:

```powershell
python examples/qccd/quickstart.py
```

It runs the verified SHAW/PGS workflow on `QAOA_wsq_8_compiled` with a 3x3 grid
machine and prints the same summary format used by sweep scripts.

Equivalent direct command:

```powershell
python run_qccd.py QAOA_wsq_8_compiled --algorithm shaw --grid-cols 3 --grid-rows 3 --trap-capacity 3 --num-layout-passes 1 --summary-only
```
