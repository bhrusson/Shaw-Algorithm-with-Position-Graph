import networkx as nx
import numpy as np
from bqskit.ir import Circuit
from bqskit.passes import UnfoldPass
from bqskit.compiler import Compiler
from bqskit.ir.gates import TGate, CNOTGate, TdgGate, HGate, RZZGate, RXGate, CPGate, SwapGate
from bqskit.ext import qiskit_to_bqskit
from qiskit import QuantumCircuit
from qiskit.circuit.library import QuantumVolume, GroverOperator, PhaseEstimation


def circuit_generate(circuit_type: str = "Toffoli", num_qubits: int = 3,
                     depth: int = 1, to_unitary: bool = False,
                     QAOA_graph: nx.Graph = None, seed: int = None):
    circuit = Circuit(num_qubits)
    rng = np.random.default_rng(seed=seed)
    workflow = [UnfoldPass()]
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
        circuit.append_gate(TGate(), 0)
        circuit.append_gate(TdgGate(), 1)
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
        circuit = Circuit(num_qubits)
        approximation_degree = 0
        for j in reversed(range(num_qubits)):
            circuit.append_gate(HGate(), j)
            num_entanglements = max(0, j - max(0, approximation_degree - (num_qubits - j - 1)))
            for k in reversed(range(j - num_entanglements, j)):
                # Use negative exponents so that the angle safely underflows to zero, rather than
                # using a temporary variable that overflows to infinity in the worst case.
                lam = np.pi * (2.0 ** (k - j))
                circuit.append_gate(CPGate(), [j, k], [lam])
        for i in range(num_qubits // 2):
            circuit.append_gate(SwapGate(), [i, num_qubits - i - 1])

    elif circuit_type == "QuantumVolume":
        qc = QuantumVolume(num_qubits, depth, seed=seed)
        circuit = qiskit_to_bqskit(qc)
        with Compiler() as compiler:
            output_circuit = compiler.compile(circuit, workflow)
        circuit.become(output_circuit)

    elif circuit_type == "QAOA":
        if QAOA_graph is None:
            graph = nx.erdos_renyi_graph(n=num_qubits, p=0.2, seed=seed)
        else:
            graph = QAOA_graph
        qc = Circuit(num_qubits)
        param = rng.uniform(size=(depth, 2))
        for qubit in range(num_qubits):
            qc.append_gate(HGate(), qubit)
        for i in range(depth):
            for edge in graph.edges:
                qc.append_gate(RZZGate(), edge, [float(param[i][0])])
            for j in range(num_qubits):
                qc.append_gate(RXGate(), j, [float(param[i][0])])
        circuit = qc

    elif circuit_type == "Grover":
        oracle = QuantumCircuit(num_qubits)
        oracle.cz(0, 1)
        qc = GroverOperator(oracle).decompose(reps=3)
        circuit = qiskit_to_bqskit(qc)
        with Compiler() as compiler:
            output_circuit = compiler.compile(circuit, workflow)
        circuit.become(output_circuit)

    elif circuit_type == "PhaseEstimator":
        unitary = QuantumCircuit(3)
        unitary.ccx(0, 1, 2)
        unitary.cx(1, 2)
        unitary.ccx(0, 1, 2)
        qc = PhaseEstimation(num_qubits - 3, unitary).decompose()
        circuit = qiskit_to_bqskit(qc)
        with Compiler() as compiler:
            output_circuit = compiler.compile(circuit, workflow)
        circuit.become(output_circuit)

    if to_unitary:
        return circuit.get_unitary()
    return circuit


if __name__ == "__main__":
    num_qubits = 8
    seed = 11
    circuit_type = "PhaseEstimator"
    cir = circuit_generate(circuit_type=circuit_type,
                           num_qubits=num_qubits,
                           depth=3,
                           to_unitary=False,
                           seed=seed)
    cir.save(f"experiments/results/experiment_circuits/input_circuits/{circuit_type}_{num_qubits}.qasm")
