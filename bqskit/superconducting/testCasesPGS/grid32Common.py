from __future__ import annotations

from bqskit.qis.graph import CouplingGraph

from bqskit.superconducting.position.graph import EdgeCapability
from bqskit.superconducting.position.graph import EdgeLabel
from bqskit.superconducting.position.graph import PositionCapability
from bqskit.superconducting.position.graph import PositionGraph
from bqskit.superconducting.position.graph import PositionLabel


GRID32_ROWS = 32
GRID32_COLS = 32
GRID32_NUM_QUDITS = GRID32_ROWS * GRID32_COLS


def grid32_idx(row: int, col: int) -> int:
    return row * GRID32_COLS + col


def build_32x32_grid_edges() -> list[tuple[int, int]]:
    edges: list[tuple[int, int]] = []
    for row in range(GRID32_ROWS):
        for col in range(GRID32_COLS):
            node = grid32_idx(row, col)

            if col < GRID32_COLS - 1:
                edges.append((node, grid32_idx(row, col + 1)))

            if row < GRID32_ROWS - 1:
                edges.append((node, grid32_idx(row + 1, col)))

    return edges


def build_32x32_position_graph() -> PositionGraph:
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

    pos_labels = [default_pos_label for _ in range(GRID32_NUM_QUDITS)]
    edge_labels: dict[tuple[int, int], EdgeLabel] = {}

    for u, v in build_32x32_grid_edges():
        edge_labels[(u, v)] = default_edge_label
        edge_labels[(v, u)] = default_edge_label

    return PositionGraph(pos_labels=pos_labels, edge_labels=edge_labels)


def build_32x32_coupling_graph() -> CouplingGraph:
    return CouplingGraph(build_32x32_grid_edges())
