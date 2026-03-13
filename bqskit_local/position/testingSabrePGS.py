import copy
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import HGate, CNOTGate
#from bqskit.qis.graph import CouplingGraph
#from bqskit.passes import SetModelPass
#from bqskit.compiler import Compiler, MachineModel

#from bqskit_local.compiler.passDataPGS import PassDataPGS
from bqskit.passes import * #SetModelPass, GeneralizedSabreLayoutPass, GeneralizedSabreRoutingPass

from bqskit_local.compiler.compilerPGS import  CompilerPGS
from bqskit_local.compiler.quickPartitionerPGS import  QuickPartitionerPGS

from bqskit_local.compiler.unfoldPGS import  UnfoldPassPGS
from bqskit_local.mapping.applyPlacementPGS import  ApplyPlacementPGS
from bqskit_local.mapping.setPGSPass import SetPGSPass
from bqskit_local.position.state import PositionGraphState
from bqskit_local.mapping.sabre_pgs import GeneralizedSabreAlgorithmPGS

from bqskit_local.routing.sabreRoutingPGS import GeneralizedSabreRoutingPassPGS
from bqskit_local.layout.sabrePassPGS import GeneralizedSabreLayoutPassPGS
from bqskit_local.position.testingMethods import *

# Build the circuit
# Build the circuit
circ = Circuit(5)
for i in range(5):
    circ.append_gate(HGate(), [i])
circ.append_gate(CNOTGate(), [0,1])
circ.append_gate(CNOTGate(), [0,2])
circ.append_gate(CNOTGate(), [0,3])
circ.append_gate(CNOTGate(), [0,4])

# Make a deep copy for reference
original_circ = copy.deepcopy(circ)


#pg = make_all_connected(4)
pg = make_line_graph(5)
radices = [2] * 5

print("len pg.poslbl: ",len(pg.position_labels))

# Create PositionGraphState
pgs = PositionGraphState(pg, radices)

for i in range (len(radices)):
    print("i is" + str(i))
    pgs.set_qudit_position(i,i)
print("logical to pos")
print(pgs.logical_to_position)
print("pos to logical")
print(pgs.position_to_logical)


# Define the compilation passes
passes = [
    UnfoldPassPGS(),
    SetPGSPass(pgs),
    #QuickPartitionerPGS(2),
    #ApplyPlacementPGS(),
    GeneralizedSabreLayoutPassPGS(total_passes=1),
    GeneralizedSabreRoutingPassPGS(decay_delta=0.5),
    ApplyPlacementPGS(),
    #UnfoldPassPGS()
]
# Create the Sabre-PGS algorithm


"""
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
"""


compiler = CompilerPGS()

# Compile the circuit with passes
compiled = compiler.compile(circ, passes)

print("Original circuit:")
for i, op in enumerate(circ):
    print(f"{i}: {op}")

print("\nCompiled circuit:")
for i, op in enumerate(compiled):
    print(f"{i}: {op}")