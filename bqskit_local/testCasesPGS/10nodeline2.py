from __future__ import annotations

import logging

from bqskit.compiler import Compiler, CompilationTask
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import HGate, CNOTGate

from bqskit_local.mapping.setPGSPass import SetPGSPass
from bqskit_local.layout.sabrePassPGS import GeneralizedSabreLayoutPassPGS
from bqskit_local.routing.sabreRoutingPGS import GeneralizedSabreRoutingPassPGS

from bqskit_local.position.state import PositionGraphState
from bqskit_local.position.testingMethods import make_line_graph


_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


# Build the circuit exactly like the CG script
circ = Circuit(10)
for i in range(10):
    circ.append_gate(HGate(), [i])
    if i != 0:
        circ.append_gate(CNOTGate(), [0, i])


pg = make_line_graph(10)
template_pgs = PositionGraphState(pg, radices=[2] * 10)

passes = [
    SetPGSPass(template_pgs, placement=list(range(10))),
    GeneralizedSabreLayoutPassPGS(template_pgs, total_passes=1),
    GeneralizedSabreRoutingPassPGS(template_pgs, decay_delta=0.5),
]
print("passes", str(passes))


# Create compiler + compilation task like the CG script
compiler = Compiler()
task = CompilationTask(circ, passes)
data = task.data

_logger.info("Passes: %s", passes)
_logger.info("Driver data before compile: keys=%s", list(data.keys()))
_logger.info("Driver data before compile: initial_mapping=%s", data.get("initial_mapping"))
_logger.info("Driver data before compile: final_mapping=%s", data.get("final_mapping"))
_logger.info("Driver data before compile: placement=%s", getattr(data, "placement", None))

compiled = compiler.compile(circ, passes, data=data)

_logger.info("Driver data after compile: keys=%s", list(data.keys()))
_logger.info("Driver data after compile: initial_mapping=%s", data.get("initial_mapping"))
_logger.info("Driver data after compile: final_mapping=%s", data.get("final_mapping"))


print("\nOriginal circuit:")
for i, op in enumerate(circ):
    print(f"{i}: {op}")

print("\nCompiled circuit:")
for i, op in enumerate(compiled):
    print(f"{i}: {op}")