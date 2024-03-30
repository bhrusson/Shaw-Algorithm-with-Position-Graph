from bqskit import Circuit
import numpy as np
from bqskit.ir.gates import RZGate, U1qGate, RZZGate, SwapGate
from pytket.phir.qtm_machine import QtmMachine
from bqskit.shuttling.util import get_gate_time
from bqskit.shuttling.ShuttlingShift import ShuttlingShiftGate
qtm_machine = QtmMachine.H1_1

num_qudits = 9
circuit_type = "adder9"
print(circuit_type)
target_circuit = Circuit(num_qudits).from_file(f"experiments/results/experiment_circuits/output_circuits/{circuit_type}"
                                               ".qasm")


# def get_duration_from_circ(circuit: Circuit, qtm_machine: QtmMachine) -> [float, int]:
#     circ_depth = circuit.num_cycles
#     total_duration = 0.0
#     count_shift_gate = 0
#     for ix in range(circ_depth):
#         layer = circuit[ix]
#         op = layer[0]
#         if op.gate == ShuttlingShiftGate(circuit.num_qudits):
#             count_shift_gate += 1
#             gate_duration = 0
#         elif op.gate == RZGate():
#             gate_duration = 0
#         else:
#             gate_duration = get_gate_time(op.gate, qtm_machine)
#         total_duration += gate_duration
#     return total_duration, count_shift_gate


# def scheduling_circuit(circuit: Circuit):
#     new_circuit = Circuit(circuit.num_qudits)
#     cycle_index = 0
#     need_shift_flg = False
#     while len(list(circuit.front)) != 0:
#         print("Front: ", circuit.front)
#         ops = circuit.get_operations(list(circuit.front))
#         rz_layer = []
#         u1qpi_layer = [[], []]  # odd and even
#         u1qpi2_layer = [[], []]  # odd and even
#         rzz_layer = [[], []]  # unshift and shift
#         swap_layer = [[], []]  # unshift and shift
#         for op in ops:
#             if op.gate == RZGate():
#                 rz_layer.append(op)
#             elif op.gate == U1qGate():
#                 if op.params[0] == np.pi:
#                     if op.location[0] % 2 == 0:
#                         u1qpi_layer[0].append(op)
#                     else:
#                         u1qpi_layer[1].append(op)
#                 elif op.params[0] == np.pi / 2:
#                     if op.location[0] % 2 == 0:
#                         u1qpi2_layer[0].append(op)
#                     else:
#                         u1qpi2_layer[1].append(op)
#                 else:
#                     raise ValueError(f"Invalid param for U1qGate operation, theta = {op.params[0]} is not "
#                                      f"supported")
#             elif op.gate == RZZGate():
#                 if op.location[0] % 2 == 0 and op.location[0] < op.location[1]:
#                     rzz_layer[0].append(op)
#                 elif op.location[1] % 2 == 0 and op.location[1] < op.location[0]:
#                     rzz_layer[0].append(op)
#                 else:
#                     rzz_layer[1].append(op)
#
#             elif op.gate == SwapGate():
#                 if op.location[0] % 2 == 0 and op.location[0] < op.location[1]:
#                     swap_layer[0].append(op)
#                 elif op.location[1] % 2 == 0 and op.location[1] < op.location[0]:
#                     swap_layer[0].append(op)
#                 else:
#                     swap_layer[1].append(op)
#             else:
#                 raise ValueError(f"Invalid gate type after synthesis, gate {op.gate} is not supported")
#
#         for point in list(circuit.front):
#             circuit.pop(point)
#
#         # TODO: put the layer into the correct cycle
#         if rz_layer != []:
#             new_circuit._append_cycle()
#             for op in rz_layer:
#                 new_circuit.insert(cycle_index, op)
#             cycle_index += 1
#
#         if u1qpi2_layer[0] != []:
#             new_circuit._append_cycle()
#             for even_pi2_op in u1qpi2_layer[0]:
#                 new_circuit.insert(cycle_index, even_pi2_op)
#             cycle_index += 1
#
#         if u1qpi2_layer[1] != []:
#             new_circuit._append_cycle()
#             for odd_pi2_op in u1qpi2_layer[1]:
#                 new_circuit.insert(cycle_index, odd_pi2_op)
#             cycle_index += 1
#
#         if u1qpi_layer[0] != []:
#             new_circuit._append_cycle()
#             for even_pi_op in u1qpi_layer[0]:
#                 new_circuit.insert(cycle_index, even_pi_op)
#             cycle_index += 1
#
#         if u1qpi_layer[1] != []:
#             new_circuit._append_cycle()
#             for odd_pi_op in u1qpi_layer[1]:
#                 new_circuit.insert(cycle_index, odd_pi_op)
#             cycle_index += 1
#
#         if rzz_layer[0] != []:
#             new_circuit._append_cycle()
#             if not need_shift_flg:
#                 for rzz_unshift_op in rzz_layer[0]:
#                     new_circuit.insert(cycle_index, rzz_unshift_op)
#             else:
#                 new_circuit.append_gate(ShuttlingShiftGate(num_qudits=circuit.num_qudits),
#                                         location=range(circuit.num_qudits))
#                 cycle_index += 1
#                 for rzz_unshift_op in rzz_layer[0]:
#                     new_circuit.insert(cycle_index, rzz_unshift_op)
#                 need_shift_flg = False
#             cycle_index += 1
#
#
#         if rzz_layer[1] != []:
#             new_circuit._append_cycle()
#             if need_shift_flg:
#                 for rzz_shift_op in rzz_layer[1]:
#                     new_circuit.insert(cycle_index, rzz_shift_op)
#             else:
#                 new_circuit.append_gate(ShuttlingShiftGate(num_qudits=circuit.num_qudits),
#                                         location=range(circuit.num_qudits))
#                 cycle_index += 1
#                 for rzz_shift_op in rzz_layer[1]:
#                     new_circuit.insert(cycle_index, rzz_shift_op)
#                 need_shift_flg = True
#             cycle_index += 1
#
#         if swap_layer[0] != []:
#             new_circuit._append_cycle()
#             for swap_unshift_op in swap_layer[0]:
#                 new_circuit.insert(cycle_index, swap_unshift_op)
#             cycle_index += 1
#
#         if swap_layer[1] != []:
#             new_circuit._append_cycle()
#             for swap_shift_op in swap_layer[1]:
#                 new_circuit.insert(cycle_index, swap_shift_op)
#             cycle_index += 1
#     return new_circuit


new_circuit = scheduling_circuit(target_circuit)
num_depth = new_circuit.num_cycles
print("Number of cycles: ", num_depth)
for i in range(num_depth):
    print("Layer ", i)
    print(new_circuit[i])

# duration, num_shifts = get_duration_from_circ(new_circuit, qtm_machine.H1_1)
# print(f"Duration:{duration} with {num_shifts} shift gates")
