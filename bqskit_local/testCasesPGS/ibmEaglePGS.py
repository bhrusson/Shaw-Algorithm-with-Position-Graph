from __future__ import annotations

import logging

from bqskit.compiler import CompilationTask, Compiler
from bqskit.ir.gates import CNOTGate

from bqskit_local.layout.sabrePassPGS import GeneralizedSabreLayoutPassPGS
from bqskit_local.mapping.setPGSPass import SetPGSPass
from bqskit_local.position.state import PositionGraphState
from bqskit_local.routing.sabreRoutingPGS import GeneralizedSabreRoutingPassPGS
from bqskit_local.testCasesPGS.ibmEagleCommon import (
    IBM_EAGLE_COUPLING_MAP,
    IBM_EAGLE_NUM_QUDITS,
    build_eagle_position_graph,
    build_eagle_test_circuit,
)

_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


circ = build_eagle_test_circuit()
pg = build_eagle_position_graph()
template_pgs = PositionGraphState(pg, radices=[2] * IBM_EAGLE_NUM_QUDITS)

print("Architecture: IBM Eagle / Washington PositionGraph")
print("Number of qudits:", circ.num_qudits)
print("Number of CNOTs:", circ.num_operations)
print("Number of directed couplings:", len(IBM_EAGLE_COUPLING_MAP))
print("Move neighbors of 0:", pg.get_swap_neighbors(0))
print("Distance 0 -> 126:", pg.distance(0, 126))

passes = [
    SetPGSPass(template_pgs, placement=list(range(IBM_EAGLE_NUM_QUDITS))),
    GeneralizedSabreLayoutPassPGS(template_pgs, total_passes=3),
    GeneralizedSabreRoutingPassPGS(template_pgs, decay_delta=0.5),
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
