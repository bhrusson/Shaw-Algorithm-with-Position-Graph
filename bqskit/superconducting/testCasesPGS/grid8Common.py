from __future__ import annotations

from bqskit.superconducting.position.graph import EdgeCapability
from bqskit.superconducting.position.graph import EdgeLabel
from bqskit.superconducting.position.graph import PositionCapability
from bqskit.superconducting.position.graph import PositionGraph
from bqskit.superconducting.position.graph import PositionLabel


GRID8_ROWS = 8
GRID8_COLS = 8
GRID8_NUM_QUDITS = GRID8_ROWS * GRID8_COLS


def grid8_idx(row: int, col: int) -> int:
    return row * GRID8_COLS + col


def build_8x8_grid_edges() -> list[tuple[int, int]]:
    edges: list[tuple[int, int]] = []
    for row in range(GRID8_ROWS):
        for col in range(GRID8_COLS):
            node = grid8_idx(row, col)

            if col < GRID8_COLS - 1:
                edges.append((node, grid8_idx(row, col + 1)))

            if row < GRID8_ROWS - 1:
                edges.append((node, grid8_idx(row + 1, col)))

    return edges


def build_8x8_position_graph() -> PositionGraph:
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

    pos_labels = [default_pos_label for _ in range(GRID8_NUM_QUDITS)]
    edge_labels: dict[tuple[int, int], EdgeLabel] = {}

    for u, v in build_8x8_grid_edges():
        edge_labels[(u, v)] = default_edge_label
        edge_labels[(v, u)] = default_edge_label

    return PositionGraph(pos_labels=pos_labels, edge_labels=edge_labels)
