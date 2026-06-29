Benchmark circuit resolver used for SABRE experiments.

The superconducting SABRE benchmark entry points import this package to find benchmark
QASM circuits. The canonical circuits live in
`bqskit/shuttling/qccd/benchmark_circuits`; this package can hold local
overrides when needed, but duplicated QASM mirrors are intentionally not kept
in the repository.
