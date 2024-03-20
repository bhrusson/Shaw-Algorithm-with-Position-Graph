from circuit_generator import circuit_generate
# from bqskit.shuttling.util import check_executable_circuit
from bqskit import Circuit
from bqskit.qis import UnitaryMatrix
from bqskit.ir.gates import PermutationGate
from bqskit.shuttling.util import get_duration_from_circ
from pytket.phir.qtm_machine import QtmMachine

qtm_machine = QtmMachine.H1_1

# target_circuit = circuit_generate("QFT", 3, 3, False)
# print("QASM: ")
# print(target_circuit.to("qasm"))
# target_circuit.save("experiments/results/input_QFT.qasm")

num_qudits = 3
circuit_type = "qv"
print(circuit_type)
target_circuit = Circuit(num_qudits).from_file(f"experiments/results/experiment_circuits/input_circuits/{circuit_type}"
                                               ".qasm")

circ = Circuit(num_qudits).from_file(f"experiments/results/experiment_circuits/output_circuits/{circuit_type}.qasm")
print(circ.gate_counts)
# print("QASM: ")
# print(circ.to("qasm"))
circ.append_gate(PermutationGate(num_qudits,  [0, 1, 2]), [0, 1, 2])

print("Distance from final unitary to the wanted unitary: ", circ.get_unitary().get_distance_from(
                                                            target_circuit.get_unitary(), 1))

# qc = QuantumVolume(num_qudits, depth=num_qudits, seed=0)
# qasm2.dump(qc, "experiments/results/input_circuits/qv.qasm")