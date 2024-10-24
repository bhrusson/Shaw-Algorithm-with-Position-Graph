"""This module implements the GeneralizedSabreAlgorithm class."""
from __future__ import annotations

import copy
import logging
from typing import Iterator
from typing import Sequence
from itertools import permutations
import numpy as np

from bqskit.ir.circuit import Circuit
from bqskit.ir.gates.barrier import BarrierPlaceholder
from bqskit.ir.gates.circuitgate import CircuitGate
from bqskit.ir.gates.constant.swap import SwapGate
from bqskit.ir.operation import Operation
from bqskit.ir.point import CircuitPoint
from bqskit.qis.graph import CouplingGraph
from bqskit.shuttling.qccd.QCCD_machine import QCCDMachineModel
from bqskit.shuttling.qccd.QCCD_physical_components import Trap

_logger = logging.getLogger(__name__)


class QCCDMappingAlgorithm():
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
            pi: dict,
            modify_circuit: bool = False,
    ) -> None:
        """
        Apply a forward pass of the Sabre algorithm to `pi`.

        Args:
            circuit (Circuit): The circuit to pass over.

            pi (list[int]): The input logical-to-position mapping. This
                maps logical qudits to the physical position of position graph.
                So, `pi[l] == p` implies logical qudit `l` is sitting on physical
                position `p` of the position graph.


            modify_circuit (bool): Whether to modify the circuit as the
                pass is applied or not. (Default: False)
        """
        # Preprocessing
        print("The position graph: ", self.qccd_machine.position_graph)
        # if not self.qccd_machine.check_valid_assignment(pi):
        #     raise ValueError("The ion assignment is not valid."
        #                      " There is either repetition in the assignment or the ions are not initially inside traps.")
        D = self.qccd_machine.all_pair_travelling_time()
        F = circuit.front
        repeated_path = False
        tmp_F = []
        decay = [1.0 for i in range(circuit.num_qudits)]
        iter_count = 0
        latest_move = ()
        prev_executed_counts: dict[CircuitPoint, int] = {n: 0 for n in F}
        leading_moves: list[tuple[int, int]] = []
        _logger.debug(f'Starting forward sabre pass with ion assignment pi: {pi}.')
        print(f"Starting forward sabre pass with ion assignment pi: {pi}.")
        if not all(r == circuit.radixes[0] for r in circuit.radixes):
            raise RuntimeError('Cannot currently map to hybrid-level systems.')

        # Main Loop
        while len(F) > 0:
            print("Front: ", [circuit[n] for n in F])
            # Retrieve executable gates giving the current mapping `pi`
            # if (len(leading_moves) > 4 and
            #         leading_moves[-1] == leading_moves[-3] == leading_moves[-5] and
            #         leading_moves[-2] == leading_moves[-4] == leading_moves[-6]):
            if len(leading_moves) > 3 and leading_moves[-1] == leading_moves[-2] and not executed_flag:
                print("There is repetition..... !!!!!")
                repeated_path = True
            print("Current ion mapping: ", pi)
            execute_list = [n for n in F if self.qccd_machine.gate_is_executable(circuit[n], pi)]
            # Execute the gates and update F
            if len(execute_list) > 0:
                executed_flag = True
                # leading_moves = []
                if tmp_F != []:
                    F += tmp_F
                    tmp_F = []
                F = set(F)
                for n in execute_list:
                    F.remove(n)
                    prev_executed_counts.pop(n)
                    _logger.debug(f'Executing gate at point {n}.')
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
                    iter_count = 0
                    for i in range(circuit.num_qudits):
                        decay[i] = 1.0

                continue  # Restart main loop if we executed at least one gate

            # # If execute list is empty, check for local-minima
            # elif len(leading_swaps) > 5 * position_graph.num_qudits:
            #     _logger.debug('Sabre stuck in local minima, backtracking...')
            #
            #     # Backtrack by removing leading swaps
            #     for swap in reversed(leading_swaps):
            #         self._apply_swap(swap, pi, decay)
            #         if modify_circuit:
            #             point = mapped_circuit._rear[swap[0]]
            #             mapped_circuit.pop(point)
            #     leading_swaps = []
            #
            #     # Override heuristic search to progress
            #     _logger.debug('Overriding sabre search...')
            #     all_logical_qudits = [circuit[n].location for n in F]
            #     qudits = min(
            #         all_logical_qudits,
            #         key=lambda qs: self._get_distance(qs, pi, D),
            #     )
            #     for swap in self._uphill_swaps(qudits, cg, pi, D):
            #         self._apply_swap(swap, pi, decay)
            #         if modify_circuit:
            #             mapped_circuit.append_gate(SwapGate(radix), swap)
            #     _logger.debug('Stopping override.')
            #     continue
            executed_flag = False
            # Pick and apply a swap
            if repeated_path:
                tmp_F = list(F)[1:]
                F = [list(F)[0]]
                repeated_path = False
                print(f"Front is modified to {F}.")
            E = self._calc_extended_set(circuit, F)
            print(f"Extended set: {[circuit[n] for n in E]}")
            best_move = self._get_best_move(circuit, F, E, D, pi, latest_move, decay)
            print(f"Best move: {best_move}")
            self._apply_move(best_move, pi, decay)
            leading_moves.append(best_move)
            latest_move = best_move

            # Update loop counter and reset decay if necessary
            iter_count += 1
            if iter_count % self.decay_reset_interval == 0:
                for i in range(circuit.num_qudits):
                    decay[i] = 1.0
        print(f"All movement contain {iter_count} steps...")
        total_moving_time = 0.0
        for move in leading_moves:
            print(f"Move: {move} ({self.qccd_machine.segment_assignment[move]} which cost {D[move[0]][move[1]]}s)")
            total_moving_time += D[move[0]][move[1]]
        print("Total moving time:", total_moving_time)

    # def backward_pass(
    #     self,
    #     circuit: Circuit,
    #     pi: list[int],
    #     cg: CouplingGraph,
    # ) -> None:
    #     """
    #     Apply a backward pass of the Sabre algorithm to `pi`.
    #
    #     Args:
    #         circuit (Circuit): The circuit to pass over.
    #
    #         pi (list[int]): The input logical-to-physical mapping. This
    #             maps logical qudits to physical qudits. So, `pi[l] == p`
    #             implies logical qudit `l` is sitting on physical qudit `p`.
    #
    #         cg (CouplingGraph): The connectivity of the hardware.
    #     """
    #     # Preprocessing
    #     D = cg.all_pairs_shortest_path()
    #     F = circuit.rear
    #     decay = [1.0 for i in range(circuit.num_qudits)]
    #     iter_count = 0
    #     leading_swaps: list[tuple[int, int]] = []
    #     next_executed_counts: dict[CircuitPoint, int] = {n: 0 for n in F}
    #     _logger.debug(f'Starting backward sabre pass with pi: {pi}.')
    #
    #     # Main Loop
    #     while len(F) > 0:
    #
    #         # Retrieve executable gates giving the current mapping: pi
    #         execute_list = [n for n in F if self._can_exe(circuit[n], pi, cg)]
    #
    #         # Execute the gates and update F
    #         if len(execute_list) > 0:
    #             leading_swaps = []
    #
    #             for n in execute_list:
    #                 F.remove(n)
    #                 next_executed_counts.pop(n)
    #                 _logger.debug(f'Executing gate at point {n}.')
    #
    #                 for predessor in circuit.prev(n):
    #                     if predessor not in next_executed_counts:
    #                         next_executed_counts[predessor] = 1
    #                     else:
    #                         next_executed_counts[predessor] += 1
    #                     num_next_executed = next_executed_counts[predessor]
    #                     total_num_next = len(circuit.next(predessor))
    #                     if num_next_executed == total_num_next:
    #                         F.add(predessor)
    #
    #             # Reset decay if necessary
    #             if self.decay_reset_on_gate:
    #                 iter_count = 0
    #                 for i in range(circuit.num_qudits):
    #                     decay[i] = 1.0
    #
    #             continue  # Restart main loop if we executed at least one gate
    #
    #         # If execute list is empty, check for local-minima
    #         elif len(leading_swaps) > 5 * cg.num_qudits:
    #             _logger.debug('Sabre stuck in local minima, backtracking...')
    #
    #             # Backtrack by removing leading swaps
    #             for swap in reversed(leading_swaps):
    #                 self._apply_swap(swap, pi, decay)
    #             leading_swaps = []
    #
    #             # Override heuristic search to progress
    #             _logger.debug('Overriding sabre search...')
    #             all_logical_qudits = [circuit[n].location for n in F]
    #             qudits = min(
    #                 all_logical_qudits,
    #                 key=lambda qs: self._get_distance(qs, pi, D),
    #             )
    #             for swap in self._uphill_swaps(qudits, cg, pi, D):
    #                 self._apply_swap(swap, pi, decay)
    #             _logger.debug('Stopping override.')
    #             continue
    #
    #         # Pick and apply a swap
    #         E = self._calc_extended_set(circuit, F)
    #         best_swap = self._get_best_swap(circuit, F, E, D, cg, pi, decay)
    #         self._apply_swap(best_swap, pi, decay)
    #         leading_swaps.append(best_swap)
    #
    #         # Update loop counter and reset decay if necessary
    #         iter_count += 1
    #         if iter_count % self.decay_reset_interval == 0:
    #             for i in range(circuit.num_qudits):
    #                 decay[i] = 1.0

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
            pi: dict,
            latest_move: tuple[int, int],
            decay: list[float],
    ) -> tuple[int, int]:
        """Return the best move given the current algorithm state and ion assignment."""
        # Track best one
        best_score = np.inf
        best_move = None

        # Gather all considerable moves
        move_candidate_list = self._obtain_moves(circuit, pi)
        # if latest_move != ():
        #     move_candidate_list.remove(latest_move)
        print("All candidate move: ", move_candidate_list)
        list_of_best_score = []
        # Score them, tracking the best one
        for move in move_candidate_list:
            score = self._score_move(circuit, F, D, pi, move, E)
            if score < best_score:
                best_score = score
                best_move = move
                list_of_best_score = [move]
            elif score == best_score:
                list_of_best_score.append(move)

        if best_move is None:
            raise RuntimeError('Unable to find best move.')
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
                    p1, p2 = pi[location[0]], pi[location[1]]
                    if p1 in move or p2 in move:
                        move_relative_score = 0.0
                    else:
                        move_relative_score += (np.min([D[p1][move[0]], D[p1][move[1]]]) +
                                                np.min([D[p2][move[0]], D[p2][move[1]]]))
                move_relative_scores.append(move_relative_score)
            # print(f"List of best move relative score: {move_relative_scores}")
            return list_of_best_score[np.argmin(move_relative_scores)]

    def _obtain_moves(
            self,
            circuit: Circuit,
            pi: dict,
    ) -> set[tuple[int, int]]:
        """Produce all possible realizable physical moves given the current QCCD hardware."""
        position_graph = self.qccd_machine.position_graph
        position_to_physical = self.qccd_machine.position_to_physical
        physical_qudit_positions = [pi[i] for i in range(circuit.num_qudits)]
        moves = set()
        # print("All positions: ", physical_qudit_positions)
        for physical_qudit_position in physical_qudit_positions:
            neighbors = position_graph.get_neighbors_of(physical_qudit_position)
            for neighbor in neighbors:
                a = min(neighbor, physical_qudit_position)
                b = max(neighbor, physical_qudit_position)
                # Enforce condition to avoid two ions go into one segment
                if a in list(pi.values()) and b in list(pi.values()):
                    if position_to_physical[a] == 'segment' or position_to_physical[b] == 'segment':
                        continue
                moves.add((a, b))
        return moves

    def _score_move(
            self,
            circuit: Circuit,
            F: set[CircuitPoint],
            D: list[list[float]],
            pi: dict,
            move: tuple[int, int],
            E: set[CircuitPoint],
    ) -> float:
        """Score the candidate realizable physical moves given the current algorithm state and ion assignment."""
        # Apply potential move
        # print("Initial ion assignement: ", pi)
        l1 = list(pi.keys())[list(pi.values()).index(move[0])] if move[0] in list(pi.values()) else None
        l2 = list(pi.keys())[list(pi.values()).index(move[1])] if move[1] in list(pi.values()) else None
        if l1 is None and l2 is None:
            raise RuntimeError(f'The move {move} is not a valid move as there is no ion in these assignment.')

        if l1 is None:
            pi[l2] = move[0]  # Move ion to the adjacent available space
        elif l2 is None:
            pi[l1] = move[1]  # Move ion to the adjacent available space
        else:
            pi[l1], pi[l2] = move[1], move[0]  # Inner trap swap
        # print("Ion assignment after moved: ", pi)
        # Calculate front set term
        front = 0.0
        for n in F:
            logical_qudits = circuit[n].location
            front += self._get_distance(logical_qudits, pi, D)
        front /= len(F)

        # Calculate extended set term
        extend = 0.0
        if len(E) > 0:
            for n in E:
                extend += self._get_distance(circuit[n].location, pi, D)
            extend /= len(E)
            extend *= self.extended_set_weight

        # Undo potential move
        if l1 is None:
            pi[l2] = move[1]  # Re-move ion to the adjacent available space
        elif l2 is None:
            pi[l1] = move[0]  # Re-move ion to the adjacent available space
        else:
            pi[l1], pi[l2] = move[0], move[1]  # Inner trap swap
        # print(f"Calculating score move {move} w.r.t pi: {pi} yields the front value of {front}"
        #       f" and extend value of {extend}")
        # print("-------------------------------------------------------------------------------")
        return front + extend

    def _get_distance_from_position_to_trap(self,
                                            position: int,
                                            available_space: list[int],
                                            D: list[list[float]],
                                            pi: dict) -> float:
        distance = np.inf
        for space in available_space:
            # ToDo: Find a way to cooperate the penalty
            #  (The penalty is the cost to resolve not able to get to the trap)
            #  of not able to move to a trap to the  distance wise (when calculating the min) (DONE)

            _, block_w = self.qccd_machine.path_is_blocked(position, space, pi)
            # print(f"Number of block in path {position, space}: {block_w}")
            space_distance = D[position][space]
            for block_position in block_w:
                if self.qccd_machine.position_to_physical[block_position] == 'segment':
                    space_distance += self.qccd_machine.timing_data['junction_Y']
                elif self.qccd_machine.position_to_physical[block_position] == 'trap':
                    min_to_endpoints = np.min([D[block_position][end_point] for end_point in
                                               self.qccd_machine.trap_end_points[
                                                   self.qccd_machine.get_trap_id(block_position)]])
                    # print("Minimum distance to endpoints: ", min_to_endpoints)
                    space_distance += (self.qccd_machine.timing_data['split'] + min_to_endpoints)
            # print(f"Distance when considering space {space}: ", space_distance)
            distance = np.min([space_distance,
                               distance])
        return distance

    def _get_distance_from_two_position_to_trap(self,
                                                positions: list[int],
                                                available_space: list[int],
                                                D: list[list[float]],
                                                pi: dict) -> (float, float):
        # ToDo: If two point refer to the same point on the same trap, this create
        #  local min situation and we need to modify this (2 point to 2 point on the same trap)
        distance = np.inf
        #print("Available space: ", available_space)
        for space in permutations(available_space, 2):
            #print("Considering space combination: {}".format(space))
            _, block_w_0 = self.qccd_machine.path_is_blocked(positions[0], space[0], pi)
            _, block_w_1 = self.qccd_machine.path_is_blocked(positions[1], space[1], pi)
            #print(f"Number of block in path {positions[0], space[0]}: {block_w_0}")
            #print(f"Number of block in path {positions[1], space[1]}: {block_w_1}")
            space_distance = np.min([D[positions[0]][space[0]] + D[positions[1]][space[1]],
                                     D[positions[0]][space[1]] + D[positions[1]][space[0]]])
            #print("Space distance: ", space_distance)
            for block_position in block_w_0:
                if self.qccd_machine.position_to_physical[block_position] == 'segment':
                    space_distance += self.qccd_machine.timing_data['junction_Y']
                elif self.qccd_machine.position_to_physical[block_position] == 'trap':
                    min_to_endpoints = np.min([D[block_position][end_point] for end_point in
                                               self.qccd_machine.trap_end_points[
                                                   self.qccd_machine.get_trap_id(block_position)]])
                    space_distance += (self.qccd_machine.timing_data['split'] + min_to_endpoints)
            for block_position in block_w_1:
                if self.qccd_machine.position_to_physical[block_position] == 'segment':
                    space_distance += self.qccd_machine.timing_data['junction_Y']
                elif self.qccd_machine.position_to_physical[block_position] == 'trap':
                    min_to_endpoints = np.min([D[block_position][end_point] for end_point in
                                               self.qccd_machine.trap_end_points[
                                                   self.qccd_machine.get_trap_id(block_position)]])
                    space_distance += (self.qccd_machine.timing_data['split'] + min_to_endpoints)
            distance = np.min([space_distance,
                               distance])
            # print(f"Distance when considering space {space}: ", space_distance)
            # print(f"Current minimum distance: ", distance)
        return distance / 2, distance / 2

    def _get_distance(
            self,
            logical_qudits: Sequence[int],
            pi: dict,
            D: list[list[int]],
    ) -> float:
        """Calculate the expected number of moves to connect logical qudits."""
        # Single qudit case
        if len(logical_qudits) == 1:
            p = pi[logical_qudits[0]]
            trap_p = self.qccd_machine.get_trap_id(p)
            if trap_p is not None:
                return 0.0
            else:
                distance_to_trap = np.inf
                for trap in self.qccd_machine.physical_graph.executable_trap_list:
                    _, available_space = self.qccd_machine.trap_is_fully_occupied(trap.id, pi)
                    distance_to_trap = np.min([self._get_distance_from_position_to_trap(p,
                                                                                        available_space,
                                                                                        D,
                                                                                        pi), distance_to_trap])
                return distance_to_trap

        # Two qudit case
        p1, p2 = pi[logical_qudits[0]], pi[logical_qudits[1]]
        if logical_qudits[0] == logical_qudits[1]:
            raise ValueError("The two logical qudits on one gate cannot be the same.")
        distance = D[p1][p2]
        # Sum distance to the nearest trap from p1 and p2
        trap_p1 = self.qccd_machine.get_trap_id(p1)
        trap_p2 = self.qccd_machine.get_trap_id(p2)
        # print(f"P1 is {p1} which is in trap {trap_p1} and P2 is {p2} which is in trap {trap_p2}")
        if trap_p1 is not None and trap_p2 is not None and trap_p1 == trap_p2:
            total_F = 0.0
        else:
            total_F = np.inf
            for trap in self.qccd_machine.physical_graph.executable_trap_list:
                _, available_space = self.qccd_machine.trap_is_fully_occupied(trap.id, pi)
                if trap_p1 == trap.id:
                    min_F_p1 = 0.0
                    min_F_p2 = self._get_distance_from_position_to_trap(p2,
                                                                        available_space,
                                                                        D,
                                                                        pi)
                elif trap_p2 == trap.id:
                    min_F_p1 = self._get_distance_from_position_to_trap(p1,
                                                                        available_space,
                                                                        D,
                                                                        pi)
                    min_F_p2 = 0.0
                else:
                    # ToDo: If two point refer to the same point on the same trap, this create
                    #  local min situation and we need to modify this (2 point to 2 point on the same trap)
                    min_F_p1, min_F_p2 = self._get_distance_from_two_position_to_trap([p1, p2],
                                                                                      available_space,
                                                                                      D,
                                                                                      pi)
                total_F = np.min([min_F_p1 + min_F_p2, total_F])
        # print(
        #     f"Physical distance w.r.t gate {logical_qudits} is {distance} "
        #     f"Total distance to nearest similar trap: {total_F}"
        # )
        return distance + total_F

    def _apply_move(
            self,
            move: tuple[int, int],
            pi: dict,
            decay: list[float],
    ) -> None:
        """Apply the move to `pi` and update `decay`."""
        _logger.debug('applying move %s' % str(move))
        # Apply potential move
        l1 = list(pi.keys())[list(pi.values()).index(move[0])] if move[0] in list(pi.values()) else None
        l2 = list(pi.keys())[list(pi.values()).index(move[1])] if move[1] in list(pi.values()) else None
        if l1 is None and l2 is None:
            raise RuntimeError(f'The move {move} is not a valid move as there is no ion in these assignment.')

        if l1 is None:
            pi[l2] = move[0]  # Move ion to the adjacent available space
        elif l2 is None:
            pi[l1] = move[1]  # Move ion to the adjacent available space
        else:
            pi[l1], pi[l2] = move[1], move[0]  # Inner trap swap

        # decay[swap[0]] += self.decay_delta
        # decay[swap[1]] += self.decay_delta

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

    def _apply_perm(self, perm: Sequence[int], pi: list[int]) -> None:
        """Apply the `perm` permutation to the current mapping `pi`."""
        _logger.debug('applying permutation %s' % str(perm))
        pi_c = {q: pi[perm[i]] for i, q in enumerate(sorted(perm))}
        for q in perm:
            pi[q] = pi_c[q]


if __name__ == '__main__':
    from bqskit.shuttling.qccd.QCCD_util import create_testing_physical_machine
    from bqskit import Circuit
    from bqskit.ir.gates import CNOTGate

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
                                     timing_data=timing_data)
    ion_assignment = {0: 0, 1: 1, 2: 2,
                      3: 6, 4: 10}
    #circuit = Circuit.from_file("data/input_qasms/Grover_5.qasm")
    # ion_assignment = {0: 0, 1: 1, 2: 2,
    #                   3: 3, 4: 4, 5: 5,
    #                   6: 6, 7: 8, 8: 9}
    # circuit = Circuit.from_file("data/input_qasms/adder9.qasm")
    # ion_assignment = {0: 0, 1: 1, 2: 2,
    #                   3: 6, 4: 10, 5: 11,
    #                   6: 7, 7: 9}
    # circuit = Circuit.from_file("data/input_qasms/Grover_8.qasm")
    # circuit = Circuit(5)
    # circuit.append_gate(CNOTGate(), (0, 1))
    # circuit.append_gate(CNOTGate(), (1, 2))

    mapping_algo = QCCDMappingAlgorithm(qccd_machine=machine_model,
                                        extended_set_size=5,
                                        extended_set_weight=0.01)
    mapping_algo.forward_pass(circuit, ion_assignment)
