from __future__ import annotations

import re

from bqskit.qis.graph import CouplingGraph

from bqskit_local.position.graph import EdgeCapability
from bqskit_local.position.graph import EdgeLabel
from bqskit_local.position.graph import PositionCapability
from bqskit_local.position.graph import PositionGraph
from bqskit_local.position.graph import PositionLabel


GRID_ARCHITECTURE_RE = re.compile(r'^grid-?(?P<rows>\d+)x(?P<cols>\d+)$')


def parse_square_grid_architecture(architecture: str) -> tuple[int, int] | None:
    architecture_key = architecture.strip().lower()
    aliases = {
        'grid8x8': (8, 8),
        'grid-8x8': (8, 8),
        'grid-16x16': (16, 16),
        'grid-32x32': (32, 32),
    }
    if architecture_key in aliases:
        return aliases[architecture_key]

    match = GRID_ARCHITECTURE_RE.match(architecture_key)
    if match is None:
        return None

    rows = int(match.group('rows'))
    cols = int(match.group('cols'))
    if rows < 1 or cols < 1:
        raise ValueError(f'Grid dimensions must be positive: {architecture}.')
    return rows, cols


def square_grid_num_qudits(rows: int, cols: int) -> int:
    return rows * cols


def square_grid_idx(row: int, col: int, cols: int) -> int:
    return row * cols + col


def build_square_grid_edges(rows: int, cols: int) -> list[tuple[int, int]]:
    edges: list[tuple[int, int]] = []
    for row in range(rows):
        for col in range(cols):
            node = square_grid_idx(row, col, cols)

            if col < cols - 1:
                edges.append((node, square_grid_idx(row, col + 1, cols)))

            if row < rows - 1:
                edges.append((node, square_grid_idx(row + 1, col, cols)))

    return edges


def build_square_grid_position_graph(rows: int, cols: int) -> PositionGraph:
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

    pos_labels = [default_pos_label for _ in range(square_grid_num_qudits(rows, cols))]
    edge_labels: dict[tuple[int, int], EdgeLabel] = {}

    for u, v in build_square_grid_edges(rows, cols):
        edge_labels[(u, v)] = default_edge_label
        edge_labels[(v, u)] = default_edge_label

    return PositionGraph(pos_labels=pos_labels, edge_labels=edge_labels)


def build_square_grid_coupling_graph(rows: int, cols: int) -> CouplingGraph:
    return CouplingGraph(build_square_grid_edges(rows, cols))


def format_square_grid_architecture(rows: int, cols: int) -> str:
    return f'{rows}x{cols} grid subset'
