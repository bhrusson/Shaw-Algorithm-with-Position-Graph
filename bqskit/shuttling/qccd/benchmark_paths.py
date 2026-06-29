from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path


BENCHMARK_DIR_ENV_VAR = 'BQSKIT_SHUTTLING_BENCHMARK_DIR'
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BENCHMARK_CIRCUITS_DIR = REPO_ROOT / 'benchmark_circuits'


def benchmark_circuits_dir() -> Path:
    """Return the directory containing benchmark QASM circuits."""
    override = os.environ.get(BENCHMARK_DIR_ENV_VAR)
    if override:
        return Path(override).expanduser().resolve()
    return DEFAULT_BENCHMARK_CIRCUITS_DIR


def benchmark_filename(input_filename: str) -> str:
    """Normalize a benchmark stem or filename to a .qasm filename."""
    return input_filename if input_filename.endswith('.qasm') else f'{input_filename}.qasm'


def resolve_benchmark_circuit_path(input_filename: str) -> Path:
    """Resolve a benchmark QASM path from the top-level benchmark directory."""
    root = benchmark_circuits_dir()
    path = root / benchmark_filename(input_filename)
    if path.exists():
        return path

    raise FileNotFoundError(
        f'Could not find benchmark circuit {path.name} in {root}. '
        f'Set {BENCHMARK_DIR_ENV_VAR} to use another benchmark directory.',
    )


def iter_benchmark_circuit_paths(pattern: str = '*.qasm') -> Iterator[Path]:
    """Yield benchmark circuits from the top-level benchmark directory."""
    yield from sorted(benchmark_circuits_dir().glob(pattern))
