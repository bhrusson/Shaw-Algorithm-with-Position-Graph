from bqskit.compiler import Compiler
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import HGate, CNOTGate
from bqskit.compiler.passdata import PassData

from bqskit_local.mapping.setPGSPass import SetPGSPass
from bqskit_local.layout.sabrePassPGS import GeneralizedSabreLayoutPassPGS
from bqskit_local.routing.sabreRoutingPGS import GeneralizedSabreRoutingPassPGS

from bqskit_local.position.state import PositionGraphState
from bqskit_local.position.testingMethods import *


# Build circuit
circ = Circuit(10)

for i in range(10):
    circ.append_gate(HGate(), [i])

for i in range(1, 10):
    circ.append_gate(CNOTGate(), [0, i])


# Build position graph
pg = make_line_graph(10)
pgs = PositionGraphState(pg, radices=[2] * 10)
data = PassData(circ)

passes = [
    SetPGSPass(pgs, placement=list(range(10))),
    GeneralizedSabreLayoutPassPGS(pgs, total_passes=1),
    GeneralizedSabreRoutingPassPGS(pgs, decay_delta=0.5),
]


import asyncio

compiled = circ.copy()

for p in passes:
    asyncio.run(p.run(compiled, data))

print("Initial mapping:", data["initial_mapping"])
print("Final mapping:", data["final_mapping"])

print("\nCompiled circuit:")
for i, op in enumerate(compiled):
    print(f"{i}: {op}")
