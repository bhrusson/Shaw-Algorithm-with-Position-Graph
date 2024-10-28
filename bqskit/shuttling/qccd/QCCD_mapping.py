"""This module implements the GeneralizedSabreAlgorithm class."""
from __future__ import annotations

import copy
import logging
from typing import Iterator
from typing import Sequence
from itertools import permutations, combinations
import numpy as np

from bqskit.ir.circuit import Circuit
from bqskit.ir.gates.barrier import BarrierPlaceholder
from bqskit.ir.gates.circuitgate import CircuitGate
from bqskit.ir.gates.constant.swap import SwapGate
from bqskit.ir.operation import Operation
from bqskit.ir.point import CircuitPoint
from bqskit.qis.graph import CouplingGraph
from bqskit.shuttling.qccd.QCCD_machine import QCCDMachineModel

_logger = logging.getLogger(__name__)


class QCCDMappingAlgorithm:
    """
    Implements methods for Sabre-based QCCD layout and routing algorithms using a
    modified heuristic to accommodate larger than 2-qudit gates.

    References:
        Gushu Li, Yufei Ding, and Yuan Xie. 2019. Tackling the Qubit
        Mapping Problem for NISQ-Era Quantum Devices. In Proceedings of
        the 24th ACM International Conference on Architectural
        Support for Programming Languages and Operating Systems
        (ASPLOS 2019). Association for Computing Machinery, New York, NY,
        USA, 1001-1014. https://doi.org/10.1145/3297858.3304023

        Casey Duckering, Jonathan M. Baker, Andrew Litteken, and Frederic
        T. Chong. 2021. Orchestrated trios: compiling for efficient
        communication in Quantum programs with 3-Qubit gates. In Proceedings
        of the 26th ACM International Conference on Architectural Support
        for Programming Languages and Operating Systems (ASPLOS 2021).
        Association for Computing Machinery, New York, NY, USA, 375-385.
        https://doi.org/10.1145/3445814.3446718

        J. Liu, E. Younis, M. Weiden, P. Hovland, J. Kubiatowicz and C. Iancu,
        "Tackling the Qubit Mapping Problem with Permutation-Aware Synthesis,"
        2023 IEEE International Conference on Quantum Computing and Engineering (QCE),
         Bellevue, WA, USA, 2023, pp. 745-756,  https://doi.org/10.1109/QCE57702.2023.00090.

    """

    def __init__(
            self,
            decay_delta: float = 0.001,
            decay_reset_interval: int = 5,
            decay_reset_on_gate: bool = True,
            extended_set_size: int = 20,
            extended_set_weight: float = 0.5,
            qccd_machine: QCCDMachineModel = None
    ) -> None:
        """
        Construct a GeneralizedSabreAlgorithm.

        Args:
            decay_delta (float): The amount to adjust the decay factor by
                each time a swap is applied. Set to zero to disable decay.
                (Default: 0.001)

            decay_reset_interval (int): The amount of swaps to apply before
                reseting the decay factors. (Default: 5)

            decay_reset_on_gate (bool): If true, reset decay factors when
                a logical gate is applied. (Default: True)

            extended_set_size (int): The size of the look-ahead or extended
                set. Set to zero to disable look ahead. (Default: 20)

            extended_set_weight (float): The weight on the extended set
                term when scoring potential swaps. (Default: 0.5)

            qccd_machine (QCCDMachineModel): Machine model of current QCCD hardware
        """
        if not isinstance(decay_delta, float):
            raise TypeError(
                'Expected float for decay_delta'
                f', got {type(decay_delta)}',
            )

        if not isinstance(decay_reset_interval, int):
            raise TypeError(
                'Expected int for decay_reset_interval'
                f', got {type(decay_reset_interval)}',
            )

        if not isinstance(decay_reset_on_gate, bool):
            raise TypeError(
                'Expected bool for decay_reset_on_gate'
                f', got {type(decay_reset_on_gate)}',
            )

        if not isinstance(extended_set_size, int):
            raise TypeError(
                'Expected int for extended_set_size'
                f', got {type(extended_set_size)}',
            )

        if not isinstance(extended_set_weight, float):
            raise TypeError(
                'Expected float for extended_set_weight'
                f', got {type(extended_set_weight)}',
            )

        if decay_reset_interval < 1:
            raise ValueError('Decay reset interval must be a positive integer.')

        if extended_set_size < 0:
            raise ValueError('Extended set size must be a nonnegative integer.')

        self.decay_delta = decay_delta
        self.decay_reset_interval = decay_reset_interval
        self.decay_reset_on_gate = decay_reset_on_gate
        self.extended_set_size = extended_set_size
        self.extended_set_weight = extended_set_weight
        self.qccd_machine = qccd_machine

    def forward_pass(
            self,
            circuit: Circuit,
            pi: list[int],
            ion_assignment: dict
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
        print("The position graph: ", self.qccd_machine.position_graph)
        # if not self.qccd_machine.check_valid_assignment(ion_assignment):
        #     raise ValueError("The ion assignment is not valid."
        #                      " There is either repetition in the assignment
        #                      or the ions are not initially inside traps.")
        D = self.qccd_machine.all_pair_travelling_time()
        F = circuit.front
        decay = [1.0 for _ in range(self.qccd_machine.position_graph.num_qudits)]
        repeated_path = False
        tmp_F = []
        self.iter_count = 0
        initial_extended_set_size = self.extended_set_size
        prev_executed_counts: dict[CircuitPoint, int] = {n: 0 for n in F}
        leading_moves: list[tuple[int, int]] = []
        _logger.debug(f'Starting forward sabre pass with ion assignment: {ion_assignment}.')
        print(f"Starting forward sabre pass with ion assignment: {ion_assignment}.")
        if not all(r == circuit.radixes[0] for r in circuit.radixes):
            raise RuntimeError('Cannot currently map to hybrid-level systems.')
        total_moving_time = 0.0
        total_moves = {'segment': 0,
                       'split_merge': 0,
                       'inner_swap': 0,
                       'junction_X': 0,
                       'junction_Y': 0}
        # Main Loop
        executed_flag = False
        while len(F) > 0:
            print("Front: ", [circuit[n] for n in F])
            if len(leading_moves) > 2 and leading_moves[-1] == leading_moves[-2] and not executed_flag:
                print("There is repetition..... !!!!!")
                repeated_path = True
            if self.iter_count > 15:
                print(f"Try bruteforce due to multiple steps ({self.iter_count}) to solve one gate")
                leading_moves += self._brute_force_congestion(circuit[list(F)[0]], D, pi, ion_assignment)
            print("Current ion mapping: ", ion_assignment)
            execute_list = [n for n in F if self.qccd_machine.gate_is_executable(circuit[n], pi, ion_assignment)]
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
                    if len(circuit[n].location) == 1:
                        total_moving_time += self.qccd_machine.timing_data['sq_timings']
                    elif len(circuit[n].location) == 2:
                        location = circuit[n].location
                        total_moving_time += self.qccd_machine.two_qudit_gate_time(p1=ion_assignment[pi[location[0]]],
                                                                                   p2=ion_assignment[pi[location[1]]])
                    print(f'Executing gate at point {n}.')
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
                        leading_moves += self._brute_force_congestion(circuit[list(F)[0]], D, pi, ion_assignment)
                        continue
                else:
                    tmp_F = list(F)[1:]
                    F = [list(F)[0]]
                    print(f"Front is modified to {F}.")
            E = self._calc_extended_set(circuit, F)
            print(f"Extended set: {[circuit[n] for n in E]}")
            best_move = self._get_best_move(circuit, F, E, D, pi, ion_assignment, decay)
            if best_move is None:
                leading_moves += self._brute_force_congestion(circuit[list(F)[0]], D, pi, ion_assignment)
                continue
            print(f"Best move: {best_move}")
            self._apply_move(best_move, ion_assignment)
            leading_moves.append(best_move)
            self.iter_count += 1

        print(f"All movement... Success:{True if self.iter_count == 0 else False}")
        for move in leading_moves:
            print(f"Move: {move} ({self.qccd_machine.segment_assignment[move]} which cost {D[move[0]][move[1]]}s)")
            if D[move[0]][move[1]] == self.qccd_machine.timing_data['segment']:
                total_moves['segment'] += 1
            elif D[move[0]][move[1]] == self.qccd_machine.timing_data['inner_swap']:
                total_moves['inner_swap'] += 1
            elif (D[move[0]][move[1]] == self.qccd_machine.timing_data['merge'] +
                  self.qccd_machine.timing_data['segment']):
                total_moves['split_merge'] += 1
            elif (D[move[0]][move[1]] == self.qccd_machine.timing_data['junction_Y'] +
                  self.qccd_machine.timing_data['segment']):
                total_moves['junction_Y'] += 1
            elif (D[move[0]][move[1]] == self.qccd_machine.timing_data['junction_X'] +
                  self.qccd_machine.timing_data['segment']):
                total_moves['junction_X'] += 1
            else:
                raise ValueError(f"The move {move} is not recognizable....")
            total_moving_time += D[move[0]][move[1]]
        print("Total moving time:", total_moving_time)
        print("Total moves: ", total_moves)

    def _brute_force_congestion(
            self,
            gate: Operation,
            D: list[list[float]],
            pi:list,
            ion_assignment: dict,
    ) -> list[tuple[int, int]]:
        """
            Logical function
        """
        gate_pos = []
        leading_moves = []
        for p in gate.location:
            gate_pos.append(ion_assignment[pi[p]])
        selected_trap_space = []
        selected_end_point = None
        relative_distance = np.inf
        # Select which trap to brute force in
        for trap in self.qccd_machine.physical_graph.executable_trap_list:
            all_trap_space = list(self.qccd_machine.physical_to_position[trap.id])
            relative_dis_to_trap = self._get_distance_from_position_to_trap(gate_pos,
                                                                            all_trap_space,
                                                                            D,
                                                                            ion_assignment)
            if relative_dis_to_trap < relative_distance:
                selected_trap_space = all_trap_space
                # ToDo: If there are more than two endpoints?
                selected_end_point = self.qccd_machine.trap_end_points[trap.id][0]
                relative_distance = relative_dis_to_trap
        print("Selected trap: ", selected_trap_space)
        # Select the order of moving position
        distance_to_trap_lst = []
        for pos in gate_pos:
            distance_to_trap = float(np.min([D[pos][trap_space] for trap_space in selected_trap_space]))
            distance_to_trap_lst.append(distance_to_trap)
        gate_pos = np.array(gate_pos)[np.argsort(distance_to_trap_lst)]
        print("Order of moving ions: ", gate_pos)
        # Select the trap space order
        trap_space_distance_to_end_point = []
        for trap_space in selected_trap_space:
            trap_space_distance_to_end_point.append(D[trap_space][selected_end_point])
        selected_trap_space = np.array(selected_trap_space)[np.argsort(trap_space_distance_to_end_point)]
        # Move the pos to the selected trap
        for pos_idx in range(len(gate_pos)):
            print(f"Trying to moving ion {gate_pos[pos_idx]}...")
            leading_moves += self._brute_force_move(
                int(gate_pos[pos_idx]),
                int(selected_trap_space[len(selected_trap_space) - 1 - pos_idx]), ion_assignment
            )
            print(f"Ion assignment after moving ion {gate_pos[pos_idx]}: {ion_assignment}")
            if (selected_end_point in ion_assignment.values() and
                    (pos_idx != len(gate_pos) - 1 and
                     list(ion_assignment.keys())[list(ion_assignment.values()).index(selected_end_point)]
                     not in gate.location)):
                end_point_neighbors = self.qccd_machine.position_graph.get_neighbors_of(selected_end_point)
                for neighbor in end_point_neighbors:
                    if neighbor in ion_assignment.values():
                        end_point_neighbors.remove(neighbor)
                self._apply_move((selected_end_point, end_point_neighbors[0]), ion_assignment)
                leading_moves.append(tuple(sorted((selected_end_point, end_point_neighbors[0]))))
                print(f"Perform move {(selected_end_point, end_point_neighbors[0])} to clear the endpoint")
        return leading_moves

    def _brute_force_move(
            self,
            position: int,
            trap_space: int,
            ion_assignment: dict
    ) -> list[tuple[int, int]]:
        """
            Physical function
        """
        leading_moves = []
        shortest_path_pos1 = self.qccd_machine.position_graph.get_shortest_path_tree(position)
        path = shortest_path_pos1[trap_space]
        ion_status = self.qccd_machine.position_to_physical[position]
        for idx_point in range(len(path) - 1):
            possible_move = (path[idx_point], path[idx_point + 1])
            if path[idx_point + 1] not in ion_assignment.values():
                self._apply_move(possible_move, ion_assignment)
                leading_moves.append(tuple(sorted(possible_move)))
                if ion_status == 'segment' and self.qccd_machine.position_to_physical[path[idx_point + 1]] == 'trap':
                    ion_status = 'trap'
                elif ion_status == 'trap' and self.qccd_machine.position_to_physical[path[idx_point + 1]] == 'segment':
                    ion_status = 'segment'
                print(
                    f"Perform move {(possible_move, ion_assignment)} as there is no ion in the neighbor, "
                    f"ion status: {ion_status}")
            elif ion_status == 'trap' and self.qccd_machine.position_to_physical[path[idx_point + 1]] == 'trap':
                self._apply_move(possible_move, ion_assignment)
                leading_moves.append(tuple(sorted(possible_move)))
                print(f"Perform move {possible_move} with inner-swap, ion status: {ion_status}")
            else:
                ion_pos = path[idx_point]
                blockage = path[idx_point + 1]
                print(f"There is blockage at {blockage}, try to resolve it...")
                leading_moves += self._resolve_congestion(ion_pos, path, blockage, ion_assignment)
                self._apply_move(possible_move, ion_assignment)
                leading_moves.append(tuple(sorted(possible_move)))
                # print(f"Ion assignment after resolving blockage: {ion_assignment}")
                print(f"Perform move {possible_move} after resolving blockage")
        return leading_moves

    def _resolve_congestion(
            self,
            target: int,
            path: list[int],
            blockage: int,
            ion_assignment: dict
    ) -> list[tuple[int, int]]:
        """
            Physical function
        """
        leading_moves = []
        blockage_neighbors = self.qccd_machine.position_graph.get_neighbors_of(blockage)
        blockage_neighbors.remove(target)
        for neighbor in blockage_neighbors:
            if neighbor in path:
                if len(blockage_neighbors) == 1:
                    continue
                else:
                    blockage_neighbors.remove(neighbor)
        potential_blockage = []
        for neighbor in blockage_neighbors:
            if neighbor in ion_assignment.keys():
                potential_blockage.append(neighbor)
                blockage_neighbors.remove(neighbor)
        # Todo: Instead of simply use the first element, can we do sth better?
        if blockage_neighbors:
            self._apply_move((blockage, blockage_neighbors[0]), ion_assignment)
            leading_moves.append(tuple(sorted((blockage, blockage_neighbors[0]))))
            print(f"Perform move {(blockage, blockage_neighbors[0])} to try resolving the blockage at {blockage}")
            return leading_moves
        else:
            leading_moves += self._resolve_congestion(blockage, path, potential_blockage[0], ion_assignment)

    def backward_pass(
        self,
        circuit: Circuit,
        pi: list[int],
        ion_assignment: dict
    ) -> None:
        """
        Apply a backward pass of the Sabre algorithm to `pi`.

        Args:
            circuit (Circuit): The circuit to pass over.

            pi (list[int]): The input logical-to-physical mapping. This
                            maps logical qudits to physical qudits. So, `pi[l] == p`
                            implies logical qudit `l` is sitting on physical qudit `p`.

            ion_assignment (dict): The input logical-to-position mapping. This
                maps logical qudits to the physical position of position graph.
                So, `pi[l] == p` implies logical qudit `l` is sitting on physical
                position `p` of the position graph.
        """
        # Preprocessing
        D = self.qccd_machine.all_pair_travelling_time()
        F = circuit.rear
        decay = [1.0 for _ in range(self.qccd_machine.position_graph.num_qudits)]
        repeated_path = False
        executed_flag = False
        tmp_F = []
        self.iter_count = 0
        initial_extended_set_size = self.extended_set_size
        leading_moves: list[tuple[int, int]] = []
        next_executed_counts: dict[CircuitPoint, int] = {n: 0 for n in F}
        _logger.debug(f'Starting backward sabre QCCD pass with ion assignment: {pi}.')

        # Main Loop
        while len(F) > 0:
            # Retrieve executable gates giving the current ion assignment: pi
            print("Front: ", [circuit[n] for n in F])
            if len(leading_moves) > 2 and leading_moves[-1] == leading_moves[-2] and not executed_flag:
                print("There is repetition..... !!!!!")
                repeated_path = True
            if self.iter_count > 15:
                print(f"Try bruteforce due to multiple steps ({self.iter_count}) to solve one gate")
                leading_moves += self._brute_force_congestion(circuit[list(F)[0]], D, ion_assignment)
            print("Current ion mapping: ", ion_assignment)
            execute_list = [n for n in F if self.qccd_machine.gate_is_executable(circuit[n], ion_assignment)]
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
                    next_executed_counts.pop(n)
                    _logger.debug(f'Executing gate at point {n}.')
                    for predessor in circuit.prev(n):
                        if predessor not in next_executed_counts:
                            next_executed_counts[predessor] = 1
                        else:
                            next_executed_counts[predessor] += 1
                        num_next_executed = next_executed_counts[predessor]
                        total_num_next = len(circuit.next(predessor))
                        if num_next_executed == total_num_next:
                            F.add(predessor)

                # Reset decay if necessary
                if self.decay_reset_on_gate:
                    iter_count = 0
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
                    # Retrieve executable gates giving the current mapping `pi`
                    if self.iter_count > 2:
                        print("Try bruteforce due to repeated pattern...")
                        leading_moves += self._brute_force_congestion(circuit[list(F)[0]], D, pi, ion_assignment)
                        continue
                else:
                    tmp_F = list(F)[1:]
                    F = [list(F)[0]]
                    print(f"Front is modified to {F}.")

            # Pick and apply a swap
            E = self._calc_extended_set(circuit, F)
            print(f"Extended set: {[circuit[n] for n in E]}")
            best_move = self._get_best_move(circuit, F, E, D, pi, ion_assignment, decay)
            if best_move is None:
                leading_moves += self._brute_force_congestion(circuit[list(F)[0]], D, pi, ion_assignment)
                continue
            print(f"Best move: {best_move}")
            self._apply_move(best_move, ion_assignment)
            self.iter_count += 1

    def _calc_extended_set(
            self,
            circuit: Circuit,
            F: set[CircuitPoint],
    ) -> set[CircuitPoint]:
        """Calculate the Extended Set for look-ahead capabilities."""
        extended_set: set[CircuitPoint] = set()
        frontier = list(copy.copy(F))
        while len(frontier) > 0 and len(extended_set) < self.extended_set_size:
            n = frontier.pop(0)
            extended_set.update(circuit.next(n))
            frontier.extend(circuit.next(n))
        return extended_set

    def _get_best_move(
            self,
            circuit: Circuit,
            F: set[CircuitPoint],
            E: set[CircuitPoint],
            D: list[list[float]],
            pi: list,
            ion_assignment: dict,
            decay: list[float],
    ) -> tuple[int, int]:
        """Return the best move given the current algorithm state and ion assignment. (Logical function)"""
        # Track best one
        best_score = np.inf
        best_move = None

        # Gather all considerable moves
        move_candidate_list = self._obtain_moves(circuit, pi, ion_assignment)
        print("All candidate move: ", move_candidate_list)
        list_of_best_score = []
        # Score them, tracking the best one
        for move in move_candidate_list:
            score = self._score_move(circuit, F, D, pi, ion_assignment, move, decay, E)
            if score < best_score:
                best_score = score
                best_move = move
                list_of_best_score = [move]
            elif score == best_score:
                list_of_best_score.append(move)
        if best_move is None:
            print("*** Unable to find best move. ***")
            return None
            # raise RuntimeError('Unable to find best move.')
        print(f"List of best move: {list_of_best_score}")
        if len(list_of_best_score) == 1:
            return best_move
        else:
            # ToDo: There is some case where we have to decide between moves with
            #  same score, we choose move with the most potential influence... (Done)
            move_relative_scores = []
            for move in list_of_best_score:
                move_relative_score = 0.0
                if D[move[0]][move[1]] == self.qccd_machine.timing_data['merge']:
                    move_relative_score = -self.qccd_machine.timing_data['merge']
                for n in F:
                    location = circuit[n].location
                    p1, p2 = ion_assignment[pi[location[0]]], ion_assignment[pi[location[1]]]
                    if p1 in move or p2 in move:
                        move_relative_score = 0.0
                    else:
                        move_relative_score += (np.min([D[p1][move[0]], D[p1][move[1]]]) +
                                                np.min([D[p2][move[0]], D[p2][move[1]]]))
                move_relative_scores.append(move_relative_score)
            return list_of_best_score[np.argmin(move_relative_scores)]

    def _obtain_moves(
            self,
            circuit: Circuit,
            pi: list,
            ion_assignment: dict,
    ) -> set[tuple[int, int]]:
        """Produce all possible realizable physical moves given the current QCCD hardware."""
        position_graph = self.qccd_machine.position_graph
        position_to_physical = self.qccd_machine.position_to_physical
        physical_qudit_positions = [ion_assignment[pi[i]] for i in range(circuit.num_qudits)]
        moves = set()
        for physical_qudit_position in physical_qudit_positions:
            neighbors = position_graph.get_neighbors_of(physical_qudit_position)
            for neighbor in neighbors:
                a = min(neighbor, physical_qudit_position)
                b = max(neighbor, physical_qudit_position)
                # Enforce condition to avoid two ions go into one segment
                if a in list(ion_assignment.values()) and b in list(ion_assignment.values()):
                    if position_to_physical[a] == 'segment' or position_to_physical[b] == 'segment':
                        continue
                moves.add((a, b))
        return moves

    def _score_move(
            self,
            circuit: Circuit,
            F: set[CircuitPoint],
            D: list[list[float]],
            pi: list,
            ion_assignment: dict,
            move: tuple[int, int],
            decay: list[float],
            E: set[CircuitPoint]
    ) -> float:
        """Score the candidate realizable physical moves given the current algorithm state and ion assignment."""
        # Apply potential move  which is physical
        # print("Initial ion assignement: ", pi)
        l1 = list(ion_assignment.keys())[list(ion_assignment.values()).index(move[0])] \
            if move[0] in list(ion_assignment.values()) else None
        l2 = list(ion_assignment.keys())[list(ion_assignment.values()).index(move[1])] \
            if move[1] in list(ion_assignment.values()) else None
        if l1 is None and l2 is None:
            raise RuntimeError(f'The move {move} is not a valid move as there is no ion in these assignment.')

        if l1 is None:
            ion_assignment[l2] = move[0]  # Move ion to the adjacent available space
        elif l2 is None:
            ion_assignment[l1] = move[1]  # Move ion to the adjacent available space
        else:
            ion_assignment[l1], ion_assignment[l2] = move[1], move[0]  # Inner trap swap
        # print("Ion assignment after moved: ", pi)
        # Calculate front set term
        front = 0.0
        for n in F:
            logical_qudits = circuit[n].location
            front += self._get_distance(logical_qudits, pi, ion_assignment, D)
        front /= len(F)

        # Calculate extended set term
        extend = 0.0
        if len(E) > 0:
            for n in E:
                extend += self._get_distance(circuit[n].location, pi, ion_assignment, D)
            extend /= len(E)
            extend *= self.extended_set_weight

        # Calculate decay factor
        # decay_factor = max(decay[move[0]], decay[move[1]])
        # Undo potential move
        if l1 is None:
            ion_assignment[l2] = move[1]  # Re-move ion to the adjacent available space
        elif l2 is None:
            ion_assignment[l1] = move[0]  # Re-move ion to the adjacent available space
        else:
            ion_assignment[l1], ion_assignment[l2] = move[0], move[1]  # Inner trap swap
        # print(f"Calculating score move {move} w.r.t pi: {pi} yields the front value of {front}"
        #       f" and extend value of {extend}")
        # print("-------------------------------------------------------------------------------")
        return front + extend

    # def _get_distance_from_position_to_trap(self,
    #                                         position: int,
    #                                         available_space: list[int],
    #                                         D: list[list[float]],
    #                                         pi: dict) -> float:
    #     distance = np.inf
    #     for space in available_space:
    #         # ToDo: Find a way to cooperate the penalty
    #         #  (The penalty is the cost to resolve not able to get to the trap)
    #         #  of not able to move to a trap to the  distance wise (when calculating the min) (DONE)
    #
    #         _, block_w = self.qccd_machine.path_is_blocked(position, space, pi)
    #         # print(f"Number of block in path {position, space}: {block_w}")
    #         space_distance = D[position][space]
    #         for block_position in block_w:
    #             if self.qccd_machine.position_to_physical[block_position] == 'segment':
    #                 space_distance += self.qccd_machine.timing_data['junction_Y']
    #             elif self.qccd_machine.position_to_physical[block_position] == 'trap':
    #                 min_to_endpoints = np.min([D[block_position][end_point] for end_point in
    #                                            self.qccd_machine.trap_end_points[
    #                                                self.qccd_machine.get_trap_id(block_position)]])
    #                 # print("Minimum distance to endpoints: ", min_to_endpoints)
    #                 space_distance += (self.qccd_machine.timing_data['split'] + min_to_endpoints)
    #         # print(f"Distance when considering space {space}: ", space_distance)
    #         distance = np.min([space_distance,
    #                            distance])
    #     return distance

    # def _get_distance_from_two_position_to_trap(self,
    #                                             positions: list[int],
    #                                             available_space: list[int],
    #                                             D: list[list[float]],
    #                                             pi: dict) -> (float, float):
    #     # ToDo: If two point refer to the same point on the same trap, this create
    #     #  local min situation and we need to modify this (2 point to 2 point on the same trap)
    #     distance = np.inf
    #     #print("Available space: ", available_space)
    #     for space in permutations(available_space, 2):
    #         #print("Considering space combination: {}".format(space))
    #         _, block_w_0 = self.qccd_machine.path_is_blocked(positions[0], space[0], pi)
    #         _, block_w_1 = self.qccd_machine.path_is_blocked(positions[1], space[1], pi)
    #         #print(f"Number of block in path {positions[0], space[0]}: {block_w_0}")
    #         #print(f"Number of block in path {positions[1], space[1]}: {block_w_1}")
    #         space_distance = D[positions[0]][space[0]] + D[positions[1]][space[1]]
    #         #print("Space distance: ", space_distance)
    #         for block_position in block_w_0:
    #             if self.qccd_machine.position_to_physical[block_position] == 'segment':
    #                 space_distance += self.qccd_machine.timing_data['junction_Y']
    #             elif self.qccd_machine.position_to_physical[block_position] == 'trap':
    #                 min_to_endpoints = np.min([D[block_position][end_point] for end_point in
    #                                            self.qccd_machine.trap_end_points[
    #                                                self.qccd_machine.get_trap_id(block_position)]])
    #                 space_distance += (self.qccd_machine.timing_data['split'] + min_to_endpoints)
    #         for block_position in block_w_1:
    #             if self.qccd_machine.position_to_physical[block_position] == 'segment':
    #                 space_distance += self.qccd_machine.timing_data['junction_Y']
    #             elif self.qccd_machine.position_to_physical[block_position] == 'trap':
    #                 min_to_endpoints = np.min([D[block_position][end_point] for end_point in
    #                                            self.qccd_machine.trap_end_points[
    #                                                self.qccd_machine.get_trap_id(block_position)]])
    #                 space_distance += (self.qccd_machine.timing_data['split'] + min_to_endpoints)
    #         distance = np.min([space_distance,
    #                            distance])
    #         # print(f"Distance when considering space {space}: ", space_distance)
    #         # print(f"Current minimum distance: ", distance)
    #     return distance / 2, distance / 2
    def _get_distance_from_position_to_trap(
            self,
            positions: list[int],
            available_space: list[int],
            D: list[list[float]],
            ion_assignment: dict
    ) -> float:
        """
            Get minimum distance from all the position of the gate to the trap...
        """
        distance = np.inf
        #print("Available space: ", available_space)
        for space in permutations(available_space, len(positions)):
            #print("Considering space combination: {}".format(space))
            blockage = [self.qccd_machine.path_is_blocked(positions[i], space[i], ion_assignment)[1]
                        for i in range(len(positions))]
            #print("Blockage: ", blockage)
            space_distance = np.sum([D[positions[i]][space[i]] for i in range(len(positions))])
            #print("Space distance: ", space_distance)
            for block_w in blockage:
                for block_position in block_w:
                    if self.qccd_machine.position_to_physical[block_position] == 'segment':
                        resolve_cost = self.qccd_machine.timing_data['junction_Y']
                    elif self.qccd_machine.position_to_physical[block_position] == 'trap':
                        min_to_endpoints = np.min([D[block_position][end_point] for end_point in
                                                   self.qccd_machine.trap_end_points[
                                                       self.qccd_machine.get_trap_id(block_position)]])
                        resolve_cost = (self.qccd_machine.timing_data['split'] + min_to_endpoints)
                    else:
                        raise ValueError("The block position is undefined as it sit on ",
                                         self.qccd_machine.position_to_physical[block_position])
                    space_distance += resolve_cost
            # print(f"Distance when considering space {space}: ", space_distance)
            # print(f"Current minimum distance: ", distance)
            # print("........")
            distance = np.min([space_distance, distance])
        return distance

    def _get_distance(
            self,
            logical_qudits: Sequence[int],
            pi: list,
            ion_assignment: dict,
            D: list[list[float]],
    ) -> float:
        """
            Calculate the expected cost w.r.t distance to connect logical qudits.
        """
        # Single qudit case
        if len(logical_qudits) == 1:
            p = [ion_assignment[pi[logical_qudits[0]]]]
            trap_p = self.qccd_machine.get_trap_id(p)
            if trap_p is not None:
                return 0.0
            else:
                distance_to_trap = np.inf
                for trap in self.qccd_machine.physical_graph.executable_trap_list:
                    _, available_space = self.qccd_machine.trap_is_fully_occupied(trap.id, ion_assignment)
                    distance_to_trap = np.min([self._get_distance_from_position_to_trap(p,
                                                                                        available_space,
                                                                                        D,
                                                                                        ion_assignment
                                                                                        ), distance_to_trap])
                return distance_to_trap
        # Multi-qudit case
        p_list = [ion_assignment[pi[logical_qudit]] for logical_qudit in logical_qudits]
        pairwise_distance = [D[p1][p2] for p1, p2 in combinations(p_list, 2)]
        distance = np.max(pairwise_distance)
        # Distance to nearest trap from p
        trap_p = [self.qccd_machine.get_trap_id(p) for p in p_list]
        if trap_p.count(None) == 0 and trap_p.count(trap_p[0]) == len(trap_p):
            total_F = 0.0
        else:
            total_F = np.inf
            for trap in self.qccd_machine.physical_graph.executable_trap_list:
                _, available_space = self.qccd_machine.trap_is_fully_occupied(trap.id, ion_assignment)
                considering_p = []
                for trap_p_index, p in zip(trap_p, p_list):
                    if trap_p_index == trap.id:
                        continue
                    else:
                        considering_p.append(p)
                if not considering_p:
                    considering_dist_to_F = 0.0
                else:
                    considering_dist_to_F = self._get_distance_from_position_to_trap(considering_p,
                                                                                     available_space,
                                                                                     D,
                                                                                     ion_assignment)
                total_F = np.min([total_F, considering_dist_to_F])
        # print(
        #     f"Physical distance w.r.t gate {logical_qudits} is {distance} "
        #     f"Total distance to nearest similar trap: {total_F}"
        # )
        return distance + total_F

    def _apply_move(
            self,
            move: tuple[int, int],
            ion_assignment: dict,
    ) -> None:
        """Apply the move to `pi` and update `decay`."""
        _logger.debug('applying move %s' % str(move))
        # Apply potential move
        l1 = list(ion_assignment.keys())[list(ion_assignment.values()).index(move[0])] \
            if move[0] in list(ion_assignment.values()) else None
        l2 = list(ion_assignment.keys())[list(ion_assignment.values()).index(move[1])] \
            if move[1] in list(ion_assignment.values()) else None
        if l1 is None and l2 is None:
            raise RuntimeError(f'The move {move} is not a valid move as there is no ion in these assignment.')
        if l1 is None:
            ion_assignment[l2] = move[0]  # Move ion to the adjacent available space
        elif l2 is None:
            ion_assignment[l1] = move[1]  # Move ion to the adjacent available space
        else:
            ion_assignment[l1], ion_assignment[l2] = move[1], move[0]  # Inner trap swap
        # decay[move[0]] += self.decay_delta
        # decay[move[1]] += self.decay_delta

    # def _uphill_swaps(
    #     self,
    #     logical_qudits: Sequence[int],
    #     cg: CouplingGraph,
    #     pi: list[int],
    #     D: list[list[int]],
    # ) -> Iterator[tuple[int, int]]:
    #     """Yield the swaps necessary to bring some of the qudits together."""
    #     center_qudit = min(
    #         logical_qudits,
    #         key=lambda q: sum(
    #             D[pi[q]][pi[p]]
    #             for p in logical_qudits
    #             if p != q
    #         ),
    #     )
    #
    #     for q in logical_qudits:
    #         if q == center_qudit:
    #             continue
    #
    #         # TODO: Do not need to calculate entire tree
    #         spt = cg.get_shortest_path_tree(pi[center_qudit])
    #         path = list(reversed(spt[pi[q]]))
    #
    #         _logger.debug(f'Moving {q} to {center_qudit} via {path}.')
    #
    #         for p1, p2 in zip(path, path[1:]):
    #             if pi[center_qudit] == p1 or pi[center_qudit] == p2:
    #                 continue
    #             yield (p1, p2)

    def _apply_perm(self, perm: Sequence[int], pi: list[int], ion_assignment: dict) -> None:
        """Apply the `perm` permutation to the current mapping `pi`."""
        _logger.debug('applying permutation %s' % str(perm))
        pi_c = {q: pi[perm[i]] for i, q in enumerate(sorted(perm))}
        ion_c = {q: ion_assignment[pi_c[i]] for i, q in enumerate(sorted(perm))}
        for q in perm:
            pi[q] = pi_c[q]
        for p, q in zip(sorted(pi), sorted(perm)):
            ion_assignment[p] = ion_c[q]


if __name__ == '__main__':
    from bqskit.shuttling.qccd.QCCD_util import create_testing_physical_machine
    from bqskit import Circuit

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
    # ion_assignment = {0: 0, 1: 1, 2: 2,
    #                   3: 6, 4: 10}
    # circuit = Circuit.from_file("data/input_qasms/Grover_5.qasm")
    # circuit = Circuit.from_file("data/input_qasms/PhaseEstimator_5.qasm")
    # ion_assignment = {0: 0, 1: 1, 2: 2,
    #                   3: 6, 4: 10, 5: 11,
    #                   6: 7, 7: 9}
    # circuit = Circuit.from_file("data/input_qasms/Grover_8.qasm")
    # circuit = Circuit.from_file("data/input_qasms/PhaseEstimator_8.qasm")
    pi = [i for i in range(9)]
    ion_assign = {0: 6, 1: 2, 2: 0,
                  3: 1, 4: 5, 5: 3,
                  6: 7, 7: 8, 8: 4}
    circuit = Circuit.from_file("data/input_qasms/adder9.qasm")
    # circuit = Circuit(5)
    # circuit.append_gate(CNOTGate(), (0, 1))
    # circuit.append_gate(CNOTGate(), (1, 2))

    mapping_algo = QCCDMappingAlgorithm(qccd_machine=machine_model,
                                        decay_delta=0.0,
                                        extended_set_size=5,
                                        extended_set_weight=0.5)
    mapping_algo.forward_pass(circuit, pi, ion_assign)
