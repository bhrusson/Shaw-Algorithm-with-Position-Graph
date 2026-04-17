import math
import os
from bqskit.ir import Circuit
from bqskit.ir.gates import CNOTGate, XGate
from bqskit.shuttling.qccd.QCCD_physical_components import QCCD_physical_machine


def _machine_logging_enabled() -> bool:
    return os.environ.get('BQSKIT_QCCD_PRINT_MACHINE', '1').lower() not in (
        '0',
        'false',
        'no',
    )

def create_grid_physical_machine(
        num_cols: int,
        num_rows: int,
        trap_capacity : int,
        initial_ions: list[int] | None = None) -> QCCD_physical_machine:
    if num_cols <= 0 or num_rows <= 0:
        raise ValueError("num_cols and num_rows must be greater than zero")
    num_traps = (num_rows + 1) * (num_cols + 1)
    num_junctions = num_rows * (num_cols + 1)
    max_traps_size = [trap_capacity] * num_traps
    executable = [True] * num_traps
    measurable = [True] * num_traps
    physical_machine = QCCD_physical_machine(num_traps=num_traps,
                                             num_junctions=num_junctions,
                                             max_traps_size=max_traps_size,
                                             initial_ions=initial_ions,
                                             executable_traps=executable,
                                             measurable_traps=measurable)
    if _machine_logging_enabled():
        print("Creating a QCCD machine ...")
    for row_idx in range(num_rows + 1):
        for col_idx in range(num_cols + 1):
            if row_idx != num_rows:
                physical_machine.add_segment(left=physical_machine.trap_list[row_idx * (num_cols + 1) + col_idx],
                                             right=physical_machine.junction_list[row_idx * (num_cols + 1) + col_idx])
            if row_idx != 0:
                physical_machine.add_segment(left=physical_machine.junction_list[(row_idx - 1) * (num_cols + 1) + col_idx],
                                             right=physical_machine.trap_list[row_idx * (num_cols + 1) + col_idx])

    for row_idx in range(num_rows):
        for col_idx in range(num_cols):
            physical_machine.add_segment(left=physical_machine.junction_list[row_idx * (num_cols + 1) + col_idx],
                                         right=physical_machine.junction_list[row_idx * (num_cols + 1) + col_idx + 1])

    if _machine_logging_enabled():
        physical_machine.print_physical_machine()
        print("Finished creating the QCCD physical machine.")
    return physical_machine

def create_testing_physical_machine(
        type: str,
        trap_capacity: int,
        num_traps: int = None) -> QCCD_physical_machine:
    if type == "H":
        num_traps = 4
        num_junctions = 2
        max_traps_size = [trap_capacity] * num_traps
        executable = [True, True, True, True]
        measurable = [False, True, False, True]
        physical_machine = QCCD_physical_machine(num_traps=num_traps,
                                                 num_junctions=num_junctions,
                                                 max_traps_size=max_traps_size,
                                                 executable_traps=executable,
                                                 measurable_traps=measurable)
        if _machine_logging_enabled():
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
        if _machine_logging_enabled():
            physical_machine.print_physical_machine()
    elif type == "H2":
        num_traps = 12
        max_traps_size = [trap_capacity] * 10
        max_traps_size.append(20)
        max_traps_size.append(20)
        num_junctions = 0
        executable = [True, True, True, True, True, True, True, True, True, True, False, False]
        measurable = [True, True, True, True, True, True, True, True, True, True, False, False]
        physical_machine = QCCD_physical_machine(num_traps=num_traps,
                                                 num_junctions=num_junctions,
                                                 max_traps_size=max_traps_size,
                                                 executable_traps=executable,
                                                 measurable_traps=measurable)
        if _machine_logging_enabled():
            print("Creating a QCCD machine ...")
        """ Bottom 4 trap """
        physical_machine.add_segment(left=physical_machine.trap_list[0],
                                     right=physical_machine.trap_list[1])
        physical_machine.add_segment(left=physical_machine.trap_list[1],
                                     right=physical_machine.trap_list[2])
        physical_machine.add_segment(left=physical_machine.trap_list[2],
                                     right=physical_machine.trap_list[3])
        physical_machine.add_segment(left=physical_machine.trap_list[3],
                                     right=physical_machine.trap_list[4])
        """ Upper 4 trap """
        physical_machine.add_segment(left=physical_machine.trap_list[9],
                                     right=physical_machine.trap_list[8])
        physical_machine.add_segment(left=physical_machine.trap_list[8],
                                     right=physical_machine.trap_list[7])
        physical_machine.add_segment(left=physical_machine.trap_list[7],
                                     right=physical_machine.trap_list[6])
        physical_machine.add_segment(left=physical_machine.trap_list[6],
                                     right=physical_machine.trap_list[5])
        """Left trap"""
        physical_machine.add_segment(left=physical_machine.trap_list[10],
                                     right=physical_machine.trap_list[0])
        physical_machine.add_segment(left=physical_machine.trap_list[5],
                                     right=physical_machine.trap_list[10])
        """Right trap"""
        physical_machine.add_segment(left=physical_machine.trap_list[4],
                                     right=physical_machine.trap_list[11])
        physical_machine.add_segment(left=physical_machine.trap_list[11],
                                     right=physical_machine.trap_list[9])
        if _machine_logging_enabled():
            physical_machine.print_physical_machine()
    elif type == "Helios":
        num_traps = 9
        max_traps_size = [trap_capacity] * 8
        max_traps_size.append(10)
        num_junctions = 1
        executable = [True, True, True, True, True, True, True, True, False]
        measurable = [True, True, True, True, True, True, True, True, False]
        physical_machine = QCCD_physical_machine(num_traps=num_traps,
                                                 num_junctions=num_junctions,
                                                 max_traps_size=max_traps_size,
                                                 executable_traps=executable,
                                                 measurable_traps=measurable)
        if _machine_logging_enabled():
            print("Creating a QCCD machine ...")
        """ Bottom 4 trap """
        physical_machine.add_segment(left=physical_machine.trap_list[0],
                                     right=physical_machine.trap_list[1])
        physical_machine.add_segment(left=physical_machine.trap_list[1],
                                     right=physical_machine.trap_list[2])
        physical_machine.add_segment(left=physical_machine.trap_list[2],
                                     right=physical_machine.trap_list[3])
        """ Upper 4 trap """
        physical_machine.add_segment(left=physical_machine.trap_list[4],
                                     right=physical_machine.trap_list[5])
        physical_machine.add_segment(left=physical_machine.trap_list[5],
                                     right=physical_machine.trap_list[6])
        physical_machine.add_segment(left=physical_machine.trap_list[6],
                                     right=physical_machine.trap_list[7])
        """Junction to traps"""
        physical_machine.add_segment(left=physical_machine.junction_list[0],
                                     right=physical_machine.trap_list[0])
        physical_machine.add_segment(left=physical_machine.junction_list[0],
                                     right=physical_machine.trap_list[4])
        """Junction to storage"""
        physical_machine.add_segment(left=physical_machine.trap_list[8],
                                     right=physical_machine.junction_list[0])
        physical_machine.add_segment(left=physical_machine.junction_list[0],
                                     right=physical_machine.trap_list[8])
        if _machine_logging_enabled():
            physical_machine.print_physical_machine()
    elif type == "Enchilada":
        num_traps = 9
        num_junctions = 6
        max_traps_size = [trap_capacity] * num_traps
        for i in range(num_traps):
            if i % 2 == 1:
                max_traps_size[i] = math.ceil(max_traps_size[i] / 2)
            else:
                continue
        executable = [True, False, True, False, True, False, True, False, True]
        measurable = [True, False, True, False, True, False, True, False, True]
        physical_machine = QCCD_physical_machine(num_traps=num_traps,
                                                 num_junctions=num_junctions,
                                                 max_traps_size=max_traps_size,
                                                 executable_traps=executable,
                                                 measurable_traps=measurable)
        if _machine_logging_enabled():
            print("Creating a QCCD machine ...")
        # print("Adding segments ...")
        physical_machine.add_segment(left=physical_machine.trap_list[0],
                                     right=physical_machine.junction_list[0])
        physical_machine.add_segment(left=physical_machine.junction_list[0],
                                     right=physical_machine.trap_list[1])
        physical_machine.add_segment(left=physical_machine.junction_list[0],
                                     right=physical_machine.junction_list[2])
        physical_machine.add_segment(left=physical_machine.trap_list[2],
                                     right=physical_machine.junction_list[1])
        physical_machine.add_segment(left=physical_machine.junction_list[1],
                                     right=physical_machine.junction_list[2])
        physical_machine.add_segment(left=physical_machine.junction_list[1],
                                     right=physical_machine.trap_list[3])
        physical_machine.add_segment(left=physical_machine.junction_list[2],
                                     right=physical_machine.trap_list[4])
        physical_machine.add_segment(left=physical_machine.trap_list[4],
                                     right=physical_machine.junction_list[3])
        physical_machine.add_segment(left=physical_machine.junction_list[3],
                                     right=physical_machine.junction_list[4])
        physical_machine.add_segment(left=physical_machine.junction_list[4],
                                     right=physical_machine.trap_list[5])
        physical_machine.add_segment(left=physical_machine.junction_list[3],
                                     right=physical_machine.junction_list[5])
        physical_machine.add_segment(left=physical_machine.junction_list[5],
                                     right=physical_machine.trap_list[7])
        physical_machine.add_segment(left=physical_machine.junction_list[4],
                                     right=physical_machine.trap_list[6])
        physical_machine.add_segment(left=physical_machine.junction_list[5],
                                     right=physical_machine.trap_list[8])
        if _machine_logging_enabled():
            physical_machine.print_physical_machine()
    elif type == "one_trap":
        num_traps = 1
        num_junctions = 0
        max_traps_size = [trap_capacity] * num_traps
        executable = [True]
        measurable = [True]
        physical_machine = QCCD_physical_machine(num_traps=num_traps,
                                                 num_junctions=num_junctions,
                                                 max_traps_size=max_traps_size,
                                                 executable_traps=executable,
                                                 measurable_traps=measurable)
        if _machine_logging_enabled():
            print("Creating a QCCD machine ...")
        # print("Adding segments ...")
        if _machine_logging_enabled():
            physical_machine.print_physical_machine()
    elif type == "linear":
        num_junctions = 0
        max_traps_size = [trap_capacity] * num_traps
        executable = [True] * num_traps
        measurable = [True] * num_traps
        physical_machine = QCCD_physical_machine(num_traps=num_traps,
                                                 num_junctions=num_junctions,
                                                 max_traps_size=max_traps_size,
                                                 executable_traps=executable,
                                                 measurable_traps=measurable)
        print("Creating a QCCD machine ...")
        for i in range(num_traps-1):
            physical_machine.add_segment(left=physical_machine.trap_list[i],
                                         right=physical_machine.trap_list[i+1])
    elif type == "G2x3":
        num_traps = 6
        num_junctions = 3
        max_traps_size = [trap_capacity] * num_traps
        executable = [True] * num_traps
        measurable = [True] * num_traps
        physical_machine = QCCD_physical_machine(num_traps=num_traps,
                                                 num_junctions=num_junctions,
                                                 max_traps_size=max_traps_size,
                                                 executable_traps=executable,
                                                 measurable_traps=measurable)
        print("Creating a QCCD machine ...")
        # print("Adding segments ...")
        physical_machine.add_segment(left=physical_machine.trap_list[0],
                                     right=physical_machine.junction_list[0])
        physical_machine.add_segment(left=physical_machine.trap_list[5],
                                     right=physical_machine.junction_list[0])
        physical_machine.add_segment(left=physical_machine.trap_list[1],
                                     right=physical_machine.junction_list[1])
        physical_machine.add_segment(left=physical_machine.trap_list[4],
                                     right=physical_machine.junction_list[1])
        physical_machine.add_segment(left=physical_machine.trap_list[2],
                                     right=physical_machine.junction_list[2])
        physical_machine.add_segment(left=physical_machine.trap_list[3],
                                     right=physical_machine.junction_list[2])
        physical_machine.add_segment(left=physical_machine.junction_list[0],
                                     right=physical_machine.junction_list[1])
        physical_machine.add_segment(left=physical_machine.junction_list[1],
                                     right=physical_machine.junction_list[2])
        physical_machine.print_physical_machine()
    else:
        raise ValueError("The type is not specified...")
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

if __name__ == '__main__':
    machine = create_grid_physical_machine(2, 2, 3)
