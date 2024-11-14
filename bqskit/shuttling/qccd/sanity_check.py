from bqskit import Circuit
from bqskit.ir.gates import *
from bqskit.shuttling.qccd.QCCD_machine import QCCDMachineModel
from bqskit.shuttling.qccd.QCCD_util import create_testing_physical_machine


def check_circuit_unitary(circuit: Circuit,
                          initial_mapping: list[int],
                          final_mapping: list[int]) -> Circuit:
    unitary = circuit.get_unitary()
    initial_mapping_unitary = PermutationGate(circuit.num_qudits, initial_mapping).get_unitary()
    final_mapping_unitary = PermutationGate(circuit.num_qudits, final_mapping).get_unitary()
    return final_mapping_unitary @ unitary @ initial_mapping_unitary


def check_if_circuit_executable(circuit: Circuit,
                                machine: QCCDMachineModel,
                                initial_ion_assignment: dict,
                                initial_mapping: list[int]):
    print("Initial ion assignment:", initial_ion_assignment)
    permuted_ion_assignment = {}
    for p in initial_ion_assignment.keys():
        permuted_ion_assignment[p] = initial_ion_assignment[initial_mapping.index(p)]
    print("Permuted ion assignment:", permuted_ion_assignment)
    num_cycles = circuit.num_cycles
    pi = list(range(circuit.num_qudits))
    executable_flag = True
    for i in range(num_cycles):
        circ_ops = circuit[i]
        for op in circ_ops:
            print("Checking operation: ", op)
            if op.gate != SwapGate():
                if machine.gate_is_executable(op, pi, permuted_ion_assignment):
                    continue
                else:
                    print(f"Gate {op} is not executable with the current ion assignment")
                    executable_flag = False
            else:
                move = op.location
                l1 = list(permuted_ion_assignment.keys())[list(permuted_ion_assignment.values()).index(move[0])] \
                    if move[0] in list(permuted_ion_assignment.values()) else None
                l2 = list(permuted_ion_assignment.keys())[list(permuted_ion_assignment.values()).index(move[1])] \
                    if move[1] in list(permuted_ion_assignment.values()) else None
                if l1 is None and l2 is None:
                    raise RuntimeError(f'The move {move} is not a valid move as there is no ion in these assignment.')
                if l1 is None:
                    permuted_ion_assignment[l2] = move[0]  # Move ion to the adjacent available space
                elif l2 is None:
                    permuted_ion_assignment[l1] = move[1]  # Move ion to the adjacent available space
                else:
                    permuted_ion_assignment[l1], permuted_ion_assignment[l2] = move[1], move[0]  # Inner trap swap
                print(f"Ion assignment after move {move}: {permuted_ion_assignment}")
    return executable_flag


if __name__ == '__main__':
    initial_mapping = [2, 4, 0, 3, 1]
    final_mapping = [3, 0, 4, 2, 1]
    data_circuit = Circuit.from_file("data/input_qasms/Grover_5.qasm")
    test_circuit = Circuit.from_file('bqskit/shuttling/qccd/result/Grover_5.qasm')
    physical_model = create_testing_physical_machine()
    timing_data = {'sq_timings': 30e-6,
                   'tq_timings': 40e-6,
                   'segment': 5e-6,
                   'inner_swap': 42e-6,
                   'split': 80e-6,
                   'merge': 80e-6,
                   'junction_Y': 100e-6,
                   'junction_X': 120e-6}
    machine_model = QCCDMachineModel(physical_graph=physical_model,
                                     multi_qudit_gate_type='FM',
                                     timing_data=timing_data)
    ion_assignment = {0: 0, 1: 1, 2: 2,
                      3: 6, 4: 10}
    machine_model.update_wrt_perm(initial_placement=list(range(machine_model.num_qudits)),
                                  permutation=initial_mapping)
    print("Coupling graph after permutation: ", machine_model.position_graph)
    print(check_if_circuit_executable(circuit=test_circuit,
                                      machine=machine_model,
                                      initial_ion_assignment=ion_assignment,
                                      initial_mapping=initial_mapping))
