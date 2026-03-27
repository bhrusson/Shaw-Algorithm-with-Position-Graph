from __future__ import annotations

import logging

from bqskit.compiler import CompilationTask, Compiler, MachineModel
from bqskit.ir.gates import CNOTGate, HGate
from bqskit.passes import GeneralizedSabreLayoutPass, GeneralizedSabreRoutingPass, SetModelPass
from bqskit.qis.graph import CouplingGraph

from bqskit_local.testCasesPGS.ibmEagleCommon import (
    IBM_EAGLE_COUPLING_MAP,
    IBM_EAGLE_NUM_QUDITS,
    IBM_EAGLE_UNDIRECTED_COUPLING_MAP,
    build_eagle_test_circuit,
)

_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


circ = build_eagle_test_circuit()
cg = CouplingGraph(IBM_EAGLE_UNDIRECTED_COUPLING_MAP, IBM_EAGLE_NUM_QUDITS)
model = MachineModel(
    num_qudits=IBM_EAGLE_NUM_QUDITS,
    coupling_graph=cg,
    gate_set={CNOTGate(), HGate()},
)

print("Architecture: IBM Eagle / Washington")
print("Number of qudits:", circ.num_qudits)
print("Number of CNOTs:", circ.num_operations)
print("Number of undirected couplings:", len(IBM_EAGLE_UNDIRECTED_COUPLING_MAP))

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
