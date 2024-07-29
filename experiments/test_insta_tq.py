import numpy as np
from bqskit.ir import Circuit
from bqskit.ir.gates import *

param = [np.pi/3]
circuit = Circuit(4)
circuit.append_gate(RZZGate(), (1, 2))
target_unitary = circuit.get_unitary()

tmp_circuit = Circuit(4)
for i in range(4):
    tmp_circuit.append_gate(U3Gate(), i)
tmp_circuit.append_gate(RZZGate(), (0, 1))
for i in range(4):
    tmp_circuit.append_gate(U3Gate(), i)
tmp_circuit.append_gate(CNOTGate(), (2, 3))
for i in range(4):
    tmp_circuit.append_gate(U3Gate(), i)


new_circuit = tmp_circuit.instantiate(target_unitary, 'minimization', multistarts=10)
print(new_circuit.to('qasm'))
print("Distance: ", new_circuit.get_unitary().get_distance_from(target_unitary))