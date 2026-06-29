from __future__ import annotations

import argparse
from pathlib import Path

from bqskit.shuttling.qccd.benchmark_paths import benchmark_circuits_dir


def build_qasm(num_qudits: int, window: int, repetitions: int) -> str:
    if num_qudits < 2:
        raise ValueError('num_qudits must be at least 2.')
    if window < 1:
        raise ValueError('window must be at least 1.')
    if repetitions < 1:
        raise ValueError('repetitions must be at least 1.')

    lines: list[str] = [
        'OPENQASM 2.0;',
        'include "qelib1.inc";',
        f'qreg q[{num_qudits}];',
    ]

    # This is a QFT-style CX benchmark intended for grid experiments.
    # Each target interacts with a fixed forward window of controls, and
    # each pair is repeated to emulate a "compiled" two-qudit-heavy trace.
    for target in range(num_qudits - 1, 0, -1):
        control_start = max(0, target - window)
        for control in range(control_start, target):
            for _ in range(repetitions):
                lines.append(f'cx q[{control}], q[{target}];')

    return '\n'.join(lines) + '\n'


def default_output_path(num_qudits: int) -> Path:
    return benchmark_circuits_dir() / f'QFT_wsq_{num_qudits}_compiled_grid.qasm'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Generate a QFT-style CX benchmark for grid experiments.',
    )
    parser.add_argument('--num-qudits', type=int, required=True)
    parser.add_argument('--window', type=int, default=12)
    parser.add_argument('--repetitions', type=int, default=2)
    parser.add_argument('--output', type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output or default_output_path(args.num_qudits)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        build_qasm(args.num_qudits, args.window, args.repetitions),
        encoding='utf-8',
    )
    print(f'Wrote {output_path}')


if __name__ == '__main__':
    main()
