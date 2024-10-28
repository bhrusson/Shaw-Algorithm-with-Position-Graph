from bqskit.ir import Circuit
from bqskit.ir.gates import CNOTGate, XGate
from bqskit.shuttling.qccd.QCCD_physical_components import QCCD_physical_machine


def create_testing_physical_machine() -> QCCD_physical_machine:
    num_traps = 4
    num_junctions = 2
    max_traps_size = [3] * num_traps
    executable = [True, True, True, True]
    measurable = [False, True, False, True]
    physical_machine = QCCD_physical_machine(num_traps=num_traps,
                                             num_junctions=num_junctions,
                                             max_traps_size=max_traps_size,
                                             executable_traps=executable,
                                             measurable_traps=measurable)
    print("Creating a QCCD machine ...")
    # print("Adding segments ...")
    physical_machine.add_segment(left=physical_machine.trap_list[0],
                                 right=physical_machine.junction_list[0])
    physical_machine.add_segment(left=physical_machine.trap_list[1],
                                 right=physical_machine.junction_list[0])
    physical_machine.add_segment(left=physical_machine.junction_list[0],
                                 right=physical_machine.junction_list[1])
    physical_machine.add_segment(left=physical_machine.trap_list[2],
                                 right=physical_machine.junction_list[1])
    physical_machine.add_segment(left=physical_machine.trap_list[3],
                                 right=physical_machine.junction_list[1])
    physical_machine.print_physical_machine()
    print("Finished creating the QCCD physical machine.")
    return physical_machine


def create_simple_circuit_1(num_qubits: int) -> Circuit:
    circuit = Circuit(num_qubits)
    for qubit in range(num_qubits):
        circuit.append_gate(XGate(), qubit)
    for qubit in range(num_qubits - 1):
        circuit.append_gate(CNOTGate(), (qubit, qubit + 1))
    for qubit in range(num_qubits - 1, 0, -1):
        circuit.append_gate(CNOTGate(), (qubit, qubit - 1))
    return circuit


def create_simple_circuit_2(num_qubits: int) -> Circuit:
    circuit = Circuit(num_qubits)
    for qubit in range(num_qubits):
        circuit.append_gate(XGate(), qubit)
    for qubit in range(num_qubits - 1):
        circuit.append_gate(CNOTGate(), (qubit, qubit + 1))
    for qubit in range(num_qubits - 1):
        circuit.append_gate(CNOTGate(), (qubit, qubit + 1))
    return circuit
