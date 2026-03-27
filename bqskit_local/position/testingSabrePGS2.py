import copy
import logging

from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import HGate, CNOTGate

from bqskit_local.position.graph import (
    PositionGraph,
    PositionLabel,
    EdgeLabel,
    PositionCapability,
    EdgeCapability,
)
from bqskit_local.position.state import PositionGraphState
from bqskit_local.mapping.sabre_pgs import GeneralizedSabreAlgorithmPGS
from bqskit_local.position.testingMethods import *



logging.basicConfig(level=logging.DEBUG)
_logger = logging.getLogger(__name__)


# Build the circuit
circ = Circuit(10)
for i in range(10):
    circ.append_gate(HGate(), [i])
    if i !=0:
        circ.append_gate(CNOTGate(), [0,i])



# 2. Build a simple PositionGraph roughly analogous to a 5-line machine
#pg = make_10_node_ring_position_graph()
pg = make_line_graph(10)

# 3. Build state and choose an initial placement
pgs = PositionGraphState(pg, radices=[2, 2, 2, 2, 2, 2, 2, 2, 2, 2])

# trivial initial placement: logical q -> same position
for q in range(10):
    pgs.set_qudit_position(q, q)

print("Initial logical -> position:")
print(pgs.logical_to_position)

print("\nInitial executable logical pairs:")
#print(list(pgs.to_coupling_graph()))


# 4. Run your algorithm directly on a copy
alg = GeneralizedSabreAlgorithmPGS(decay_delta=0.5)

compiled = circ.copy()
alg.forward_pass(compiled, pgs, modify_circuit=True)


# 5. Inspect the result
print("\nFinal logical -> position:")
print(pgs.logical_to_position)

print("\nCompiled circuit:")
for i, op in enumerate(compiled):
    print(f"{i}: {op}")