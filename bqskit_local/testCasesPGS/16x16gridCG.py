from __future__ import annotations

import logging
from time import perf_counter

from bqskit.compiler import CompilationTask, Compiler, MachineModel
from bqskit.ir.gates import CNOTGate, HGate
from bqskit.passes import GeneralizedSabreLayoutPass, GeneralizedSabreRoutingPass, SetModelPass

from bqskit_local.testCasesPGS.grid16Common import (
    GRID16_NUM_QUDITS,
    build_16x16_challenge_circuit,
    build_16x16_coupling_graph,
    build_16x16_grid_edges,
)

_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


ROUNDS = 2


circ = build_16x16_challenge_circuit(rounds=ROUNDS)
cg = build_16x16_coupling_graph()
model = MachineModel(
    num_qudits=GRID16_NUM_QUDITS,
    coupling_graph=cg,
    gate_set={CNOTGate(), HGate()},
)

print("Architecture: 16x16 grid CouplingGraph")
print("Challenge rounds:", ROUNDS)
print("Number of qudits:", circ.num_qudits)
print("Number of operations:", circ.num_operations)
print("Number of undirected couplings:", len(build_16x16_grid_edges()))

passes = [
    SetModelPass(model),
    GeneralizedSabreLayoutPass(total_passes=3),
    GeneralizedSabreRoutingPass(decay_delta=0.5),
]
print("passes", str(passes))

compiler = Compiler()
task = CompilationTask(circ, passes)
data = task.data

_logger.info("Driver data before compile: initial_mapping=%s", data.get("initial_mapping"))
_logger.info("Driver data before compile: final_mapping=%s", data.get("final_mapping"))
_logger.info("Driver data before compile: placement=%s", data.get("placement"))

start_time = perf_counter()
compiled = compiler.compile(circ, passes, data=data)
elapsed_time = perf_counter() - start_time

_logger.info("Driver data after compile: initial_mapping=%s", data.get("initial_mapping"))
_logger.info("Driver data after compile: final_mapping=%s", data.get("final_mapping"))
_logger.info("Driver data after compile: placement=%s", data.get("placement"))

print("Compilation runtime (s):", f"{elapsed_time:.3f}")
print("Original operation count:", circ.num_operations)
print("Compiled operation count:", compiled.num_operations)
