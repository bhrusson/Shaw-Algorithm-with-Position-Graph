from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Sequence

from bqskit.ir.gates.constant.ccx import CCXGate
from bqskit.ir.gates.parameterized.unitary import VariableUnitaryGate
from bqskit.ir.operation import Operation

from bqskit.superconducting.mapping.sabre_pgs import (
    GeneralizedSabreAlgorithmPGS,
)
from bqskit.superconducting.position.graph import EdgeCapability
from bqskit.superconducting.position.graph import EdgeLabel
from bqskit.superconducting.position.graph import PositionCapability
from bqskit.superconducting.position.graph import PositionGraph
from bqskit.superconducting.position.graph import PositionLabel
from bqskit.superconducting.position.state import PositionGraphState


NUM_QUDITS = 8


@dataclass(frozen=True)
class CanExeCase:
    name: str
    location: tuple[int, ...]
    expected: bool


def build_line_position_graph(num_qudits: int = NUM_QUDITS) -> PositionGraph:
    """Build a PGS graph with explicit EXECUTE-capable line edges."""
    pos_label = PositionLabel(
        capability=PositionCapability.STARTING | PositionCapability.EXECUTE,
        weights={
            PositionCapability.STARTING: 0.0,
            PositionCapability.EXECUTE: 1.0,
        },
    )
    edge_capability = (
        EdgeCapability.MOVE
        | EdgeCapability.SWAP
        | EdgeCapability.EXECUTE
    )
    edge_label = EdgeLabel(
        capability=edge_capability,
        weights={
            EdgeCapability.MOVE: 1.0,
            EdgeCapability.SWAP: 1.0,
            EdgeCapability.EXECUTE: 1.0,
        },
    )

    edge_labels: dict[tuple[int, int], EdgeLabel] = {}
    for i in range(num_qudits - 1):
        edge_labels[(i, i + 1)] = edge_label
        edge_labels[(i + 1, i)] = edge_label

    return PositionGraph([pos_label for _ in range(num_qudits)], edge_labels)


def build_identity_pgs(position_graph: PositionGraph) -> PositionGraphState:
    """Create a PGS state whose logical-to-position mapping is identity."""
    pgs = PositionGraphState(position_graph, radices=(2,) * NUM_QUDITS)
    for logical in range(NUM_QUDITS):
        pgs.set_qudit_position(logical, logical)
    return pgs


def make_gate(num_qudits: int):
    """Return a real multi-qudit gate for each arity under test."""
    if num_qudits == 3:
        return CCXGate()

    return VariableUnitaryGate(num_qudits)


def make_operation(location: Sequence[int]) -> Operation:
    gate = make_gate(len(location))
    params = [0.0] * gate.num_params
    return Operation(gate, tuple(location), params)


def test_cases() -> list[CanExeCase]:
    return [
        CanExeCase(
            name='3q-connected-line',
            location=(0, 1, 2),
            expected=True,
        ),
        CanExeCase(
            name='3q-disconnected-induced-subgraph',
            location=(0, 2, 4),
            expected=False,
        ),
        CanExeCase(
            name='4q-connected-line',
            location=(2, 3, 4, 5),
            expected=True,
        ),
        CanExeCase(
            name='4q-disconnected-induced-subgraph',
            location=(0, 1, 3, 4),
            expected=False,
        ),
        CanExeCase(
            name='5q-connected-line',
            location=(3, 4, 5, 6, 7),
            expected=True,
        ),
        CanExeCase(
            name='5q-disconnected-induced-subgraph',
            location=(0, 1, 2, 5, 6),
            expected=False,
        ),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Smoke-test PGS SABRE _can_exe behavior on 3-, 4-, '
            'and 5-qudit gates over an 8-position line graph.'
        ),
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Only print failures and the final summary.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    position_graph = build_line_position_graph()
    pgs = build_identity_pgs(position_graph)

    pgs_algo = GeneralizedSabreAlgorithmPGS()

    failures: list[str] = []
    for case in test_cases():
        op = make_operation(case.location)
        pgs_can_exe = pgs_algo._can_exe(op, pgs)
        ok = pgs_can_exe == case.expected

        if not args.quiet or not ok:
            print(
                f'{case.name}: location={case.location}, '
                f'expected={case.expected}, '
                f'pgs={pgs_can_exe}, '
                f'{"PASS" if ok else "FAIL"}'
            )

        if not ok:
            failures.append(case.name)

    if failures:
        failure_list = ', '.join(failures)
        raise SystemExit(f'FAILED multi-qudit can_exe smoke cases: {failure_list}')

    print(
        'PASS: PGS _can_exe matches expected behavior for 3-, 4-, '
        'and 5-qudit gates on an 8-position line graph.'
    )


if __name__ == '__main__':
    main()
