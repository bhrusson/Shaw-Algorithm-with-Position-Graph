import copy
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import HGate, CNOTGate
from bqskit.qis.graph import CouplingGraph
from bqskit_local.position.state import PositionGraphState
from bqskit_local.mapping.sabre_pgs import GeneralizedSabreAlgorithmPGS
from bqskit_local.position.testingMethods import *

# Build the circuit
circ = Circuit(8)
for i in range(8):
    circ.append_gate(HGate(), [i])
    if i == i % 3:
        circ.append_gate(CNOTGate(), [0, 5])
    else:
        circ.append_gate(CNOTGate(), [i, i % 3])
    
    if i % 2 == i % 7:
        circ.append_gate(CNOTGate(), [7, 0])
    else:
        circ.append_gate(CNOTGate(), [i % 2, i % 7])
    circ.append_gate(HGate(), [(i % 3 * 121) % 8])

# Make a deep copy for reference
original_circ = copy.deepcopy(circ)


pg = make_32_node_sparse_graph()
radices = [2,2,2,2,2,2,2,2,2,2,2,2]

# Create PositionGraphState
pgs = PositionGraphState(pg, radices )
for i in range (len(radices)):
    pgs.set_qudit_position(i,i)

# Create the Sabre-PGS algorithm
sabre = GeneralizedSabreAlgorithmPGS(
    decay_delta=0.5,
    decay_reset_interval=5,
    extended_set_size=20,
    extended_set_weight=0.5
)

# Forward pass (modify the circuit)
sabre.forward_pass(circ, pgs, modify_circuit=True)

# Backward pass (optional, modifies circuit further)
sabre.backward_pass(circ, pgs)

# Print the original circuit
print("Original circuit:")
for i, op in enumerate(original_circ):
    print(f"{i}: {op}")

# Print the compiled/mapped circuit
print("\nCompiled (mapped) circuit:")
for i, op in enumerate(circ):
    print(f"{i}: {op}")
