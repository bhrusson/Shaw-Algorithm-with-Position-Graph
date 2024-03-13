from bqskit.ir import Circuit
from numpy import pi
from bqskit.ir.gates import TGate, CNOTGate, TdgGate, HGate, SwapGate, U1Gate, ControlledGate
from bqskit.ext import qiskit_to_bqskit
from qiskit.circuit.library import QuantumVolume


def circuit_generate(circuit_type: str = "Toffoli", num_qubits: int = 3,
                     depth: int = 1, to_unitary: bool = False):
    circuit = Circuit(num_qubits)
    if circuit_type == "Toffoli":
        circuit.append_gate(HGate(), 2)
        circuit.append_gate(CNOTGate(), [1, 2])
        circuit.append_gate(TdgGate(), 2)
        circuit.append_gate(CNOTGate(), [0, 2])
        circuit.append_gate(TGate(), 2)
        circuit.append_gate(CNOTGate(), [1, 2])
        circuit.append_gate(TdgGate(), 2)
        circuit.append_gate(CNOTGate(), [0, 2])
        circuit.append_gate(TGate(), 1)
        circuit.append_gate(TGate(), 2)
        circuit.append_gate(CNOTGate(), [0, 1])
        circuit.append_gate(TdgGate(), 0)
        circuit.append_gate(TGate(), 1)
        circuit.append_gate(CNOTGate(), [0, 1])
        circuit.append_gate(HGate(), 2)
    elif circuit_type == "Fredkin":
        circuit.append_gate(CNOTGate(), [2, 1])
        circuit.append_gate(CNOTGate(), [0, 1])
        circuit.append_gate(HGate(), 2)
        circuit.append_gate(TGate(), 0)
        circuit.append_gate(TdgGate(), 1)
        circuit.append_gate(TGate(), 2)
        circuit.append_gate(CNOTGate(), [2, 1])
        circuit.append_gate(CNOTGate(), [0, 2])
        circuit.append_gate(TGate(), 1)
        circuit.append_gate(CNOTGate(), [0, 1])
        circuit.append_gate(TdgGate(), 2)
        circuit.append_gate(TdgGate(), 1)
        circuit.append_gate(CNOTGate(), [0, 2])
        circuit.append_gate(CNOTGate(), [2, 1])
        circuit.append_gate(TGate(), 1)
        circuit.append_gate(HGate(), 2)
        circuit.append_gate(CNOTGate(), [2, 1])
    elif circuit_type == "QFT":
        circuit.append_gate(HGate(), 0)
        circuit.append_gate(ControlledGate(U1Gate(pi / 2), 1), [1, 0])
        circuit.append_gate(ControlledGate(U1Gate(pi / 4), 1), [2, 0])
        circuit.append_gate(HGate(), 1)
        circuit.append_gate(ControlledGate(U1Gate(pi / 2), 1), [2, 1])
        circuit.append_gate(HGate(), 2)
        circuit.append_gate(SwapGate(), [0, 2])
    elif circuit_type == "QuantumVolume":
        qc = QuantumVolume(num_qubits, depth)
        tmp_circuit = qiskit_to_bqskit(qc)
        circuit.append_circuit(tmp_circuit, range(num_qubits))
    if to_unitary:
        return circuit.get_unitary()
    return circuit
