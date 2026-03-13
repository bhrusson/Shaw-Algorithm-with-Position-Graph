from bqskit.compiler import Compiler
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import HGate, CNOTGate
from bqskit.compiler.passdata import PassData


from bqskit_local.compiler.passDataPGS import PassDataPGS
from bqskit_local.mapping.setPGSPass import SetPGSPass
from bqskit_local.layout.sabrePassPGS import GeneralizedSabreLayoutPassPGS
from bqskit_local.routing.sabreRoutingPGS import GeneralizedSabreRoutingPassPGS

from bqskit_local.position.state import PositionGraphState
from bqskit_local.position.testingMethods import *

#resulted in 4434 opperations

# Build circuit
# Build the circuit
n = 64
circ = Circuit(n)

# One CNOT for every unordered pair (i, j) with i < j
for control in range(n):
    for target in range(control + 1, n):
        circ.append_gate(CNOTGate(), (control, target))
        #2016 gates

print("Number of qudits:", circ.num_qudits)
print("Number of CNOTs:", circ.num_operations)


rows = 8
cols = 8


def idx(r: int, c: int) -> int:
    return r * cols + c


# Same label for every position
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

pos_labels = [default_pos_label for _ in range(rows * cols)]


# Same label for every nearest-neighbor edge
# Add EXECUTE here if adjacent positions can directly do 2-qudit gates
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

edge_labels = {}

for r in range(rows):
    for c in range(cols):
        u = idx(r, c)

        # Right neighbor
        if c < cols - 1:
            v = idx(r, c + 1)
            edge_labels[(u, v)] = default_edge_label
            edge_labels[(v, u)] = default_edge_label

        # Down neighbor
        if r < rows - 1:
            v = idx(r + 1, c)
            edge_labels[(u, v)] = default_edge_label
            edge_labels[(v, u)] = default_edge_label


pg = PositionGraph(pos_labels=pos_labels, edge_labels=edge_labels)

print("Number of positions:", len(pg.position_labels))
print("Number of directed edges:", len(pg.edge_labels))
print("Move neighbors of 0:", pg.get_swap_neighbors(0))
print("Distance 0 -> 63:", pg.distance(0, 63))

pgs = PositionGraphState(pg, radices=[2] * 64)
data = PassData(circ)

passes = [
    SetPGSPass(pgs, placement=list(range(64))),
    #GeneralizedSabreLayoutPassPGS(total_passes=3),
    GeneralizedSabreRoutingPassPGS(decay_delta=0.5),
]


import asyncio

compiled = circ.copy()

for p in passes:
    asyncio.run(p.run(compiled, data))

print("Initial mapping:", data["initial_mapping"])
print("Final mapping:", data["final_mapping"])
print("PGS mapping:", data["pgs"].logical_to_position)

print("\nCompiled circuit:")
for i, op in enumerate(compiled):
    print(f"{i}: {op}")