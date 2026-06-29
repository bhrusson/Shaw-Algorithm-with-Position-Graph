from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path


LOCAL_BENCHMARK_CIRCUITS_DIR = Path(__file__).resolve().parent
UPSTREAM_BENCHMARK_CIRCUITS_DIR = (
    LOCAL_BENCHMARK_CIRCUITS_DIR.parent.parent
    / 'shuttling'
    / 'qccd'
    / 'benchmark_circuits'
)


def resolve_benchmark_circuit_path(input_filename: str) -> Path:
    filename = f'{input_filename}.qasm'
    local_path = LOCAL_BENCHMARK_CIRCUITS_DIR / filename
    if local_path.exists():
        return local_path

    upstream_path = UPSTREAM_BENCHMARK_CIRCUITS_DIR / filename
    if upstream_path.exists():
        return upstream_path

    raise FileNotFoundError(
        f'Could not find benchmark circuit {filename} in '
        f'{LOCAL_BENCHMARK_CIRCUITS_DIR} or {UPSTREAM_BENCHMARK_CIRCUITS_DIR}.',
    )


def iter_benchmark_circuit_paths(pattern: str = '*.qasm') -> Iterator[Path]:
    """Yield benchmark circuits, preferring local overrides when present."""
    seen: set[str] = set()
    for root in (LOCAL_BENCHMARK_CIRCUITS_DIR, UPSTREAM_BENCHMARK_CIRCUITS_DIR):
        for path in sorted(root.glob(pattern)):
            if path.name in seen:
                continue
            seen.add(path.name)
            yield path
