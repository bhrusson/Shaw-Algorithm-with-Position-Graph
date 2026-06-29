# Benchmark Circuits

This directory contains the QASM inputs used by the QCCD shuttling and SABRE
benchmark drivers.

The Python package resolves these files through
`bqskit.shuttling.qccd.benchmark_paths`. To use a different benchmark directory,
set `BQSKIT_SHUTTLING_BENCHMARK_DIR` to an absolute or relative path.
