
from bqskit.compiler import Compiler, MachineModel, CompilationTask
from bqskit.passes import * #SetModelPass, GeneralizedSabreLayoutPass, GeneralizedSabreRoutingPass
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import HGate, CNOTGate
from bqskit.compiler.passdata import PassData
#from bqskit_local.mapping.sabre_pgs import GeneralizedSabreAlgorithmPGS
from bqskit.qis.graph import CouplingGraph
import logging

_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

# Build the circuit
circ = Circuit(100)
for i in range(100):
    circ.append_gate(HGate(), [i])
    if i !=0:
        circ.append_gate(CNOTGate(), [0,i])

cg = CouplingGraph.linear(100)

model = MachineModel(
    num_qudits=100,
    coupling_graph=cg,
    gate_set={CNOTGate(), HGate()}
)

print("hello,test",model.coupling_graph)
# Define the compilation passes


passes = [
    UnfoldPass(),
    SetModelPass(model),
    QuickPartitioner(2),
    ApplyPlacement(),
    #GeneralizedSabreLayoutPass(total_passes=3),
    GeneralizedSabreRoutingPass(decay_delta=0.5),
    ApplyPlacement(),
    UnfoldPass()
]
print("passes",str(passes))
# Create the compiler
compiler = Compiler()
task = CompilationTask(circ, passes)
data = task.data

print("Available keys in pass data:")
print(list(data.keys()))

print("Initial mapping (logical -> physical):")
print(data["initial_mapping"])

print("\nFinal mapping (logical -> physical):")
print(data["final_mapping"])

print("\ndata[machine_model].coupling_graph:")
print(data["machine_model"].coupling_graph)


# Compile the circuit with passes
compiled = compiler.compile(circ, passes, data=task.data)



print("Original circuit:")
for i, op in enumerate(circ):
    print(f"{i}: {op}")

print("\nCompiled circuit:")
for i, op in enumerate(compiled):
    print(f"{i}: {op}")
