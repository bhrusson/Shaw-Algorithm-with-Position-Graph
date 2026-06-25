from __future__ import annotations

from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CNOTGate
from bqskit.ir.gates import HGate
from bqskit.qis.graph import CouplingGraph

from bqskit.superconducting.position.graph import (
    EdgeCapability,
    EdgeLabel,
    PositionCapability,
    PositionGraph,
    PositionLabel,
)


GRID4_ROWS = 4
GRID4_COLS = 4
GRID4_NUM_QUDITS = GRID4_ROWS * GRID4_COLS


def grid4_idx(row: int, col: int) -> int:
    return row * GRID4_COLS + col


def build_4x4_grid_edges() -> list[tuple[int, int]]:
    edges: list[tuple[int, int]] = []
    for row in range(GRID4_ROWS):
        for col in range(GRID4_COLS):
            node = grid4_idx(row, col)

            if col < GRID4_COLS - 1:
                edges.append((node, grid4_idx(row, col + 1)))

            if row < GRID4_ROWS - 1:
                edges.append((node, grid4_idx(row + 1, col)))

    return edges


def build_4x4_position_graph() -> PositionGraph:
    default_pos_label = PositionLabel(
        capability=(
            PositionCapability.EXECUTE
            | PositionCapability.MEASURE
            | PositionCapability.STARTING
        ),
        weights={
            PositionCapability.EXECUTE: 1.0,
            PositionCapability.MEASURE: 1.0,
            PositionCapability.STARTING: 1.0,
        },
    )

    default_edge_label = EdgeLabel(
        capability=(
            EdgeCapability.MOVE
            | EdgeCapability.SWAP
            | EdgeCapability.EXECUTE
        ),
        weights={
            EdgeCapability.MOVE: 1.0,
            EdgeCapability.SWAP: 1.0,
            EdgeCapability.EXECUTE: 1.0,
        },
    )

    pos_labels = [default_pos_label for _ in range(GRID4_NUM_QUDITS)]
    edge_labels: dict[tuple[int, int], EdgeLabel] = {}

    for u, v in build_4x4_grid_edges():
        edge_labels[(u, v)] = default_edge_label
        edge_labels[(v, u)] = default_edge_label

    return PositionGraph(pos_labels=pos_labels, edge_labels=edge_labels)


def build_4x4_coupling_graph() -> CouplingGraph:
    return CouplingGraph(build_4x4_grid_edges())


def _snake_row(row: int) -> list[int]:
    cols = list(range(GRID4_COLS))
    if row % 2 == 1:
        cols.reverse()
    return [grid4_idx(row, col) for col in cols]


def _row_band_pairs(row_a: int, row_b: int, offset: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for col in range(GRID4_COLS):
        target_col = (col + offset) % GRID4_COLS
        pairs.append((grid4_idx(row_a, col), grid4_idx(row_b, target_col)))
    return pairs


def _column_band_pairs(col_a: int, col_b: int, offset: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for row in range(GRID4_ROWS):
        target_row = (row + offset) % GRID4_ROWS
        pairs.append((grid4_idx(row, col_a), grid4_idx(target_row, col_b)))
    return pairs


def build_4x4_challenge_circuit(rounds: int = 2) -> Circuit:
    if rounds < 1:
        raise ValueError('rounds must be positive.')

    circ = Circuit(GRID4_NUM_QUDITS)

    for qudit in range(GRID4_NUM_QUDITS):
        circ.append_gate(HGate(), (qudit,))

    for round_index in range(rounds):
        row_offset = 1 + (round_index % 2)
        col_offset = 1 + (round_index % 3)

        for row in range(GRID4_ROWS):
            snake_nodes = _snake_row(row)
            rotated = snake_nodes[row_offset:] + snake_nodes[:row_offset]
            if row % 2 == 1:
                rotated = list(reversed(rotated))

            for control, target in zip(snake_nodes, rotated):
                if control != target:
                    circ.append_gate(CNOTGate(), (control, target))

        for row in range(0, GRID4_ROWS - 1, 2):
            for control, target in _row_band_pairs(row, row + 1, row_offset):
                if control != target:
                    circ.append_gate(CNOTGate(), (control, target))

        far_row = round_index % GRID4_ROWS
        opposite_row = (far_row + 2) % GRID4_ROWS
        for control, target in _row_band_pairs(far_row, opposite_row, col_offset):
            if control != target:
                circ.append_gate(CNOTGate(), (control, target))

        for col in range(0, GRID4_COLS - 1, 2):
            for control, target in _column_band_pairs(col, col + 1, col_offset):
                if control != target:
                    circ.append_gate(CNOTGate(), (control, target))

        far_col = (round_index * 2) % GRID4_COLS
        opposite_col = (far_col + 2) % GRID4_COLS
        for control, target in _column_band_pairs(far_col, opposite_col, row_offset):
            if control != target:
                circ.append_gate(CNOTGate(), (control, target))

        for step in range(GRID4_ROWS):
            src = grid4_idx(step, step)
            dst = grid4_idx(GRID4_ROWS - 1 - step, (step + 2 + round_index) % GRID4_COLS)
            if src != dst:
                circ.append_gate(CNOTGate(), (src, dst))

        if round_index != rounds - 1:
            for qudit in range(round_index % 2, GRID4_NUM_QUDITS, 2):
                circ.append_gate(HGate(), (qudit,))

    return circ
