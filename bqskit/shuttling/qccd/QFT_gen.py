from qiskit import QuantumCircuit
from qiskit import circuit
from qiskit import qasm2
from bqskit import Circuit
from bqskit.ext import qiskit_to_bqskit
def build_qft_circuit(n: int) -> QuantumCircuit:
    circ = circuit.library.QFT(n, do_swaps=False)
    circ._build()
    return circ
n = 45
qc = build_qft_circuit(n)
# qasm2.dump(qc, f"bqskit/shuttling/qccd/benchmark_circuits/qiskit_QFT_{n}.qasm")
circ = qiskit_to_bqskit(qc)
circ.save(f"bqskit/shuttling/qccd/benchmark_circuits/bqskit_QFT_{n}.qasm")
