from __future__ import annotations

import logging

from bqskit.compiler import CompilationTask, Compiler
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CNOTGate

from bqskit_local.layout.sabrePassPGS import GeneralizedSabreLayoutPassPGS
from bqskit_local.mapping.setPGSPass import SetPGSPass
from bqskit_local.position.graph import (
    EdgeCapability,
    EdgeLabel,
    PositionCapability,
    PositionGraph,
    PositionLabel,
)
from bqskit_local.position.state import PositionGraphState
from bqskit_local.routing.sabreRoutingPGS import GeneralizedSabreRoutingPassPGS

_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


rows = 4
cols = 4
n = rows * cols


def idx(r: int, c: int) -> int:
    return r * cols + c


circ = Circuit(n)

# A compact nonlocal pattern that is easier to inspect by eye than all-pairs.
gate_pairs = [
    (0, 3),
    (0, 5),
    (0, 10),
    (0, 12),
    (0, 15),
    (15, 3),
    (15, 5),
    (15, 10),
    (15, 12),
    (1, 14),
    (6, 9),
]

for control, target in gate_pairs:
    circ.append_gate(CNOTGate(), (control, target))

print("Number of qudits:", circ.num_qudits)
print("Number of CNOTs:", circ.num_operations)

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

pos_labels = [default_pos_label for _ in range(n)]
edge_labels: dict[tuple[int, int], EdgeLabel] = {}

for r in range(rows):
    for c in range(cols):
        u = idx(r, c)

        if c < cols - 1:
            v = idx(r, c + 1)
            edge_labels[(u, v)] = default_edge_label
            edge_labels[(v, u)] = default_edge_label

        if r < rows - 1:
            v = idx(r + 1, c)
            edge_labels[(u, v)] = default_edge_label
            edge_labels[(v, u)] = default_edge_label

pg = PositionGraph(pos_labels=pos_labels, edge_labels=edge_labels)
template_pgs = PositionGraphState(pg, radices=[2] * n)

print("Number of positions:", len(pg.position_labels))
print("Number of directed edges:", len(pg.edge_labels))
print("Move neighbors of 0:", pg.get_swap_neighbors(0))
print("Distance 0 -> 15:", pg.distance(0, 15))

passes = [
    SetPGSPass(template_pgs, placement=list(range(n))),
    GeneralizedSabreLayoutPassPGS(template_pgs, total_passes=3),
    GeneralizedSabreRoutingPassPGS(template_pgs, decay_delta=0.5),
]
print("passes", str(passes))

compiler = Compiler()
task = CompilationTask(circ, passes)
data = task.data

_logger.info("Passes: %s", passes)
_logger.info("Driver data before compile: initial_mapping=%s", data.get("initial_mapping"))
_logger.info("Driver data before compile: final_mapping=%s", data.get("final_mapping"))
_logger.info("Driver data before compile: placement=%s", data.get("placement"))

compiled = compiler.compile(circ, passes, data=data)

_logger.info("Driver data after compile: initial_mapping=%s", data.get("initial_mapping"))
_logger.info("Driver data after compile: final_mapping=%s", data.get("final_mapping"))
_logger.info("Driver data after compile: placement=%s", data.get("placement"))

print("Original circuit:")
for i, op in enumerate(circ):
    print(f"{i}: {op}")

print("\nCompiled circuit:")
for i, op in enumerate(compiled):
    print(f"{i}: {op}")
