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


# Build circuit
num_qudits = 100
circ = Circuit(num_qudits)

for i in range(num_qudits):
    circ.append_gate(HGate(), [i])

for i in range(1, num_qudits):
    circ.append_gate(CNOTGate(), [0, i])


# Build position graph
pg = make_line_graph(num_qudits)
pgs = PositionGraphState(pg, radices=[2] * num_qudits)
data = PassData(circ)

passes = [
    SetPGSPass(pgs, placement=list(range(num_qudits))),
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