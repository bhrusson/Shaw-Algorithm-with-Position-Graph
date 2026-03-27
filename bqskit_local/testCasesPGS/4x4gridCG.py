from __future__ import annotations

import logging

from bqskit.compiler import CompilationTask, Compiler, MachineModel
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CNOTGate, HGate
from bqskit.passes import GeneralizedSabreLayoutPass, GeneralizedSabreRoutingPass, SetModelPass
from bqskit.qis.graph import CouplingGraph

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

edges: list[tuple[int, int]] = []
for r in range(rows):
    for c in range(cols):
        node = idx(r, c)

        if c < cols - 1:
            edges.append((node, idx(r, c + 1)))

        if r < rows - 1:
            edges.append((node, idx(r + 1, c)))

cg = CouplingGraph(edges)
model = MachineModel(
    num_qudits=n,
    coupling_graph=cg,
    gate_set={CNOTGate(), HGate()},
)

passes = [
    SetModelPass(model),
    GeneralizedSabreLayoutPass(total_passes=3),
    GeneralizedSabreRoutingPass(decay_delta=0.5),
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
