from __future__ import annotations

import numpy as np
from bqskit import Circuit
from bqskit.ir import Operation
from bqskit.compiler import PassData
from bqskit.compiler.basepass import BasePass
from bqskit.ir.gates import RZGate, U1qPiGate, U1qPi2Gate, RZZGate, SwapGate, FrozenParameterGate
from bqskit.shuttling.ShuttlingShift import ShuttlingShiftGate


class OddEvenSchedulingPass(BasePass):
    def __init__(self) -> None:
        """
        Initializes the OddEvenSchedulingPass
        """

    async def run(self, circuit: Circuit, data: PassData) -> None:
        new_circuit = Circuit(circuit.num_qudits + 1)
        cycle_index = 0
        need_shift_flg = False
        while len(list(circuit.front)) != 0:
            # print("Front: ", circuit.front)
            ops = circuit.get_operations(list(circuit.front))
            rz_layer = []
            u1qpi_layer = [[], []]  # odd and even
            u1qpi2_layer = [[], []]  # odd and even
            rzz_layer = [[], []]  # unshift and shift
            swap_layer = [[], []]  # unshift and shift
            for op in ops:
                if op.gate == RZGate():
                    if not need_shift_flg:
                        rz_layer.append(op)
                    else:
                        new_op = Operation(gate=op.gate, location=[op.location[0] + 1], params=op.params)
                        rz_layer.append(new_op)

                elif op.gate == U1qPiGate:
                    if not need_shift_flg:
                        if op.location[0] % 2 == 0:
                            u1qpi_layer[0].append(op)
                        else:
                            u1qpi_layer[1].append(op)
                    else:
                        new_op = Operation(gate=op.gate, location=[op.location[0] + 1], params=op.params)
                        if new_op.location[0] % 2 == 0:
                            u1qpi_layer[0].append(new_op)
                        else:
                            u1qpi_layer[1].append(new_op)

                elif op.gate == U1qPi2Gate:
                    if not need_shift_flg:
                        if op.location[0] % 2 == 0:
                            u1qpi2_layer[0].append(op)
                        else:
                            u1qpi2_layer[1].append(op)
                    else:
                        new_op = Operation(gate=op.gate, location=[op.location[0] + 1], params=op.params)
                        if new_op.location[0] % 2 == 0:
                            u1qpi2_layer[0].append(new_op)
                        else:
                            u1qpi2_layer[1].append(new_op)

                elif op.gate == RZZGate():
                    if op.location[0] % 2 == 0 and op.location[0] < op.location[1]:
                        rzz_layer[0].append(op)
                    elif op.location[1] % 2 == 0 and op.location[1] < op.location[0]:
                        rzz_layer[0].append(op)
                    else:
                        rzz_layer[1].append(op)

                elif op.gate == SwapGate():
                    if not need_shift_flg:
                        if op.location[0] % 2 == 0 and op.location[0] < op.location[1]:
                            swap_layer[0].append(op)
                        elif op.location[1] % 2 == 0 and op.location[1] < op.location[0]:
                            swap_layer[0].append(op)
                        else:
                            swap_layer[1].append(op)
                    if need_shift_flg:
                        new_op = Operation(gate=op.gate, location=[op.location[0] + 1, op.location[1] + 1],
                                           params=op.params)
                        if new_op.location[0] % 2 == 0 and new_op.location[0] < new_op.location[1]:
                            swap_layer[0].append(new_op)
                        elif new_op.location[1] % 2 == 0 and new_op.location[1] < new_op.location[0]:
                            swap_layer[0].append(new_op)
                        else:
                            swap_layer[1].append(new_op)
                else:
                    raise ValueError(f"Invalid gate type after synthesis, gate {op.gate} is not supported")

            for point in list(circuit.front):
                circuit.pop(point)
            if rz_layer != []:
                new_circuit._append_cycle()
                for op in rz_layer:
                    new_circuit.insert(cycle_index, op)
                cycle_index += 1

            if u1qpi2_layer[0] != []:
                new_circuit._append_cycle()
                for even_pi2_op in u1qpi2_layer[0]:
                    new_circuit.insert(cycle_index, even_pi2_op)
                cycle_index += 1

            if u1qpi2_layer[1] != []:
                new_circuit._append_cycle()
                for odd_pi2_op in u1qpi2_layer[1]:
                    new_circuit.insert(cycle_index, odd_pi2_op)
                cycle_index += 1

            if u1qpi_layer[0] != []:
                new_circuit._append_cycle()
                for even_pi_op in u1qpi_layer[0]:
                    new_circuit.insert(cycle_index, even_pi_op)
                cycle_index += 1

            if u1qpi_layer[1] != []:
                new_circuit._append_cycle()
                for odd_pi_op in u1qpi_layer[1]:
                    new_circuit.insert(cycle_index, odd_pi_op)
                cycle_index += 1

            if rzz_layer[0] != []:
                new_circuit._append_cycle()
                if not need_shift_flg:
                    for rzz_unshift_op in rzz_layer[0]:
                        new_circuit.insert(cycle_index, rzz_unshift_op)
                else:
                    new_circuit.append_gate(ShuttlingShiftGate(num_qudits=circuit.num_qudits),
                                            location=range(circuit.num_qudits))
                    cycle_index += 1
                    for rzz_unshift_op in rzz_layer[0]:
                        new_circuit.insert(cycle_index, rzz_unshift_op)
                    need_shift_flg = False
                cycle_index += 1

            if rzz_layer[1] != []:
                new_circuit._append_cycle()
                if need_shift_flg:
                    for rzz_shift_op in rzz_layer[1]:
                        new_rzz_shift = Operation(gate=rzz_shift_op.gate,
                                                  location=[rzz_shift_op.location[0] + 1,
                                                            rzz_shift_op.location[1] + 1],
                                                  params=rzz_shift_op.params)
                        new_circuit.insert(cycle_index, new_rzz_shift)
                else:
                    new_circuit.append_gate(ShuttlingShiftGate(num_qudits=circuit.num_qudits),
                                            location=range(circuit.num_qudits))
                    cycle_index += 1
                    for rzz_shift_op in rzz_layer[1]:
                        new_rzz_shift = Operation(gate=rzz_shift_op.gate,
                                                  location=[rzz_shift_op.location[0] + 1,
                                                            rzz_shift_op.location[1] + 1],
                                                  params=rzz_shift_op.params)
                        new_circuit.insert(cycle_index, new_rzz_shift)
                    need_shift_flg = True
                cycle_index += 1

            if swap_layer[0] != []:
                new_circuit._append_cycle()
                for swap_unshift_op in swap_layer[0]:
                    new_circuit.insert(cycle_index, swap_unshift_op)
                cycle_index += 1

            if swap_layer[1] != []:
                new_circuit._append_cycle()
                for swap_shift_op in swap_layer[1]:
                    new_circuit.insert(cycle_index, swap_shift_op)
                cycle_index += 1

        circuit.become(new_circuit)
        return None
