"""This module implements the SHAW algorithm."""

from __future__ import annotations
import math
import copy
import logging
import itertools
from typing import Iterator
from typing import Sequence
from itertools import permutations, combinations
import numpy as np

from bqskit.ir.circuit import Circuit
from bqskit.ir.operation import Operation
from bqskit.ir.point import CircuitPoint
#from ..old.QCCD_machine import QCCDMachineModel
#from bqskit.shuttling.qccd.QCCD_machine import QCCDMachineModel
from bqskit.ir.gates.barrier import BarrierPlaceholder
from bqskit_local.position.graph import *

_logger = logging.getLogger(__name__)

class sabre_position_graph:

    
    def __init__(
        self,
        position_graph: PositionGraph,
        decay_delta: float = 0.001,
        decay_reset_interval: int = 5,
        decay_reset_on_gate: bool = True,
        extended_set_size: int = 5,
        extended_set_weight: float = 0.5,
        congestion_rate: float = 0.6,
    ) -> None:        
        self.position_graph = position_graph
        self.decay_delta = decay_delta
        self.decay_reset_interval = decay_reset_interval
        self.decay_reset_on_gate = decay_reset_on_gate
        self.extended_set_size = extended_set_size
        self.extended_set_weight = extended_set_weight
        self.congestion_rate = congestion_rate

    
    def gate_is_executable(self, op, pi, ion_assignment):
        """Check if gate operands are physically adjacent."""
        phys_positions = [ion_assignment[q] for q in op.location]
        # For 2-qubit gates
        if len(phys_positions) == 2:
            return self.position_graph.are_neighbors(*phys_positions, capability='EXECUTE')
        # For 3+ qubit gates, may extend based on EXECUTE cluster capability
        return self.position_graph.qubits_are_coupled(phys_positions)
    
    def check_valid_assignment(self, ion_assignment: dict[int, int]) -> bool:
        """Ensure unique assignment and valid positions."""
        vals = list(ion_assignment.values())
        if len(vals) != len(set(vals)):
            return False
        return all(0 <= v < self.position_graph.num_qudits for v in vals)
    
    def forward_pass(
            self,
            circuit: Circuit,
            pi: list[int],
            ion_assignment: dict,
            modify_circuit: bool
    ) -> None:
        """
        Apply a forward pass of the Sabre algorithm to `pi`.

        Args:
            circuit (Circuit): The circuit to pass over.

            pi (list[int]): The input logical-to-physical mapping. This
                            maps logical qudits to physical qudits. So, `pi[l] == p`
                            implies logical qudit `l` is sitting on physical qudit `p`.

            ion_assignment (dict): The input logical-to-position mapping. This
                maps logical qudits to the physical position of position graph.
                So, `pi[l] == p` implies logical qudit `l` is sitting on physical
                position `p` of the position graph.

            modify_circuit (bool): Whether to modify the circuit as the
                pass is applied or not. (Default: False)
        """
        # Preprocessing
        # print("The position graph: ", self.qccd_machine.position_graph)
        # if not self.qccd_machine.check_valid_assignment(ion_assignment):
        #     raise ValueError("The ion assignment is not valid."
        #                      " There is either repetition in the assignment
        #                      or the ions are not initially inside traps.")
        D = self.position_graph.all_pairs_dijkstra_shortest_paths()
        F = circuit.front
        decay = [1.0 for _ in range(self.position_graph.num_qudits)]
        repeated_path = False
        tmp_F = []
        self.iter_count = 0
        initial_extended_set_size = self.extended_set_size
        prev_executed_counts: dict[CircuitPoint, int] = {n: 0 for n in F}
        leading_moves: list[tuple[int, int]] = []
        _logger.debug(f'Starting forward sabre pass with ion assignment: {ion_assignment}.')
        print(f"Starting forward sabre pass with ion assignment: {ion_assignment}.")

        if modify_circuit:
            instructions_list = []
            mapped_circuit = Circuit(circuit.num_qudits, circuit.radixes)

        if not all(r == circuit.radixes[0] for r in circuit.radixes):
            raise RuntimeError('Cannot currently map to hybrid-level systems.')
        longest_path = np.max([len(path) for path in self.position_graph.all_pairs_dijkstra_shortest_paths()])
        # Main Loop
        executed_flag = False
        heuristic_move = True
        while len(F) > 0:
            if len(leading_moves) > 2 and leading_moves[-1] == leading_moves[-2] and not executed_flag:
                print("There is repetition..... !!!!!")
                repeated_path = True
            if self.iter_count > math.ceil(longest_path / 4):
                print(f"Try bruteforce due to multiple steps ({self.iter_count}) to solve one gate")
                brute_force_moves = self._brute_force_congestion(circuit[list(F)[0]], D, pi, ion_assignment)
                if modify_circuit:
                    for move in brute_force_moves:
                        # mapped_circuit.append_gate(SwapGate(), move)
                        instructions_list.append(
                            [f"Move {move}", f"{ion_assignment}", f"cost: {D[move[0]][move[1]]} seconds"])
                        mapped_circuit.append_gate(BarrierPlaceholder(circuit.num_qudits),
                                                   list(range(circuit.num_qudits)))
                leading_moves += brute_force_moves
            print("Current ion mapping: ", ion_assignment)
            execute_list = [n for n in F if self.position_graph.gate_is_executable(circuit[n], pi, ion_assignment)]
            # Execute the gates and update F
            if len(execute_list) > 0:
                executed_flag = True
                # Rest penalty from reptition
                self.extended_set_size = initial_extended_set_size
                # Add the temporary F to current F
                if tmp_F:
                    F += tmp_F
                    tmp_F = []
                F = set(F)
                for n in execute_list:
                    F.remove(n)
                    prev_executed_counts.pop(n)
                    _logger.debug(f'Executing gate at point {n}.')
                    print(f'Executing gate at point {n}.')
                    if modify_circuit:
                        op = circuit[n]
                        physical_location = [pi[q] for q in op.location]
                        mapped_circuit.append_gate(op.gate, op.location)
                        instructions_list.append([f"Execute at {physical_location}", f"{ion_assignment}"])
                        mapped_circuit.append_gate(BarrierPlaceholder(circuit.num_qudits),
                                                   list(range(circuit.num_qudits)))
                    for successor in circuit.next(n):
                        if successor not in prev_executed_counts:
                            prev_executed_counts[successor] = 1
                        else:
                            prev_executed_counts[successor] += 1
                        num_prev_executed = prev_executed_counts[successor]
                        total_num_prev = len(circuit.prev(successor))
                        if num_prev_executed == total_num_prev:
                            F.add(successor)
                # Reset decay if necessary
                if self.decay_reset_on_gate:
                    self.iter_count = 0
                    for i in range(circuit.num_qudits):
                        decay[i] = 1.0
                continue  # Restart main loop if we executed at least one gate

            executed_flag = False
            # Pick and apply a swap
            if repeated_path:
                # If there is repetition, first take into account only one gate in F
                # Then change the extended set size to 0
                repeated_path = False
                if len(F) == 1 and self.extended_set_size != 0:
                    self.extended_set_size = 0
                elif len(F) == 1 and self.extended_set_size == 0:
                    # Retrieve executable gates giving the current ion assignment `pi`
                    if self.iter_count > 2:
                        print("Try bruteforce due to repeated pattern...")
                        brute_force_moves = self._brute_force_congestion(circuit[list(F)[0]], D, pi, ion_assignment)
                        if modify_circuit:
                            for move in brute_force_moves:
                                # mapped_circuit.append_gate(SwapGate(), move)
                                instructions_list.append(
                                    [f"Move {move}", f"{ion_assignment}", f"cost: {D[move[0]][move[1]]} seconds"])
                                mapped_circuit.append_gate(BarrierPlaceholder(circuit.num_qudits),
                                                           list(range(circuit.num_qudits)))
                        leading_moves += brute_force_moves
                        continue
                else:
                    tmp_F = list(F)[1:]
                    F = [list(F)[0]]
                    print(f"Front is modified to {F}.")
            E = self._calc_extended_set(circuit, F)
            print(f"Extended set: {[circuit[n] for n in E]}")
            best_move = self._get_best_move(circuit, F, E, D, pi, ion_assignment, decay, heuristic_move)
            if best_move is None:
                brute_force_moves = self._brute_force_congestion(circuit[list(F)[0]], D, pi, ion_assignment)
                if modify_circuit:
                    for move in brute_force_moves:
                        instructions_list.append(
                            [f"Move {move}", f"{ion_assignment}", f"cost: {D[move[0]][move[1]]} seconds"])
                        mapped_circuit.append_gate(BarrierPlaceholder(circuit.num_qudits),
                                                   list(range(circuit.num_qudits)))
                leading_moves += brute_force_moves
                continue
            print(f"Best move: {best_move}")
            self._apply_move(best_move, ion_assignment)
            leading_moves.append(best_move)

            if modify_circuit:
                # mapped_circuit.append_gate(SwapGate(), best_move)
                instructions_list.append(
                    [f"Move {best_move} ", f"{ion_assignment}", f"cost: {D[best_move[0]][best_move[1]]} seconds"])
                mapped_circuit.append_gate(BarrierPlaceholder(circuit.num_qudits), list(range(circuit.num_qudits)))

            self.iter_count += 1
        if modify_circuit:
            circuit.become(mapped_circuit)
            return instructions_list