"""This module implements the PAMRoutingPass class."""
from __future__ import annotations

import copy
import math
import itertools as it
import logging
from typing import Dict
from typing import Literal
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import TypedDict

import numpy as np

from bqskit.ir.circuit import Circuit
from bqskit.ir.gates.barrier import BarrierPlaceholder
from bqskit.ir.gates.constant.swap import SwapGate
from bqskit.ir.point import CircuitPoint
from bqskit.qis.graph import CouplingGraph
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix

from bqskit.shuttling.qccd.QCCD_mapping import QCCDMappingAlgorithm
from bqskit.shuttling.qccd.QCCD_machine import QCCDMachineModel

_logger = logging.getLogger(__name__)

PAMBlockPermData = Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], Circuit]
PAMBlockTAPermData = Dict[CouplingGraph, PAMBlockPermData]


class PAMBlockResultData(TypedDict):
    pre_perm: tuple[int, ...]
    post_perm: tuple[int, ...]
    original_utry: UnitaryMatrix


PAMBlockResultDict = Dict[CircuitPoint, PAMBlockResultData]


class PermutationAwareQCCDMappingAlgorithm(QCCDMappingAlgorithm):
    """
        Implements methods for Sabre-based permutation-aware QCCD layout and routing algorithms using a
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
            gate_count_weight: float = .1,
            decay_delta: float = 0.001,
            decay_reset_interval: int = 5,
            decay_reset_on_gate: bool = True,
            extended_set_size: int = 5,
            extended_set_weight: float = 0.5,
            qccd_machine: QCCDMachineModel = None,
            cogestion_segment_rate: float = 0.6
    ) -> None:
        """
        Construct a PermutationAwareMappingAlgorithm.

        Args:
            gate_count_weight (float): The weight on block gate count
                versus mapping score when selecting a permutation.

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
        """
        if not isinstance(gate_count_weight, float):
            bad_type = type(gate_count_weight)
            m = f'Expected float for gate_count_weight, got {bad_type}'
            raise TypeError(m)

        self.gate_count_weight = gate_count_weight

        super().__init__(
            decay_delta,
            decay_reset_interval,
            decay_reset_on_gate,
            extended_set_size,
            extended_set_weight,
            qccd_machine,
            cogestion_segment_rate
        )

    @overload  # type: ignore
    def forward_pass(
            self,
            circuit: Circuit,
            pi: list[int],
            ion_assignment: dict,
            cg: CouplingGraph,
            perm_data: dict[CircuitPoint, PAMBlockTAPermData],
            modify_circuit: Literal[False] = False,
    ) -> None:
        ...

    @overload
    def forward_pass(
            self,
            circuit: Circuit,
            pi: list[int],
            ion_assignment: dict,
            cg: CouplingGraph,
            perm_data: dict[CircuitPoint, PAMBlockTAPermData],
            modify_circuit: Literal[True],
    ) -> PAMBlockResultDict:
        ...

    def forward_pass(  # type: ignore
            self,
            circuit: Circuit,
            pi: list[int],
            ion_assignment: dict,
            cg: CouplingGraph,
            perm_data: dict[CircuitPoint, PAMBlockTAPermData],
            modify_circuit: bool = False,
    ) -> (PAMBlockResultDict | None, list, int):
        """
        Apply a forward pass of the PAM algorithm to `pi`.

        Args:
            circuit (Circuit): The circuit to pass over.

            pi (list[int]): The input logical-to-physical mapping. This
                maps logical qudits to physical qudits. So, `pi[l] == p`
                implies logical qudit `l` is sitting on physical qudit `p`.

            ion_assignment (dict): The input logical-to-position mapping. This
                maps logical qudits to the physical position of position graph.
                So, `pi[l] == p` implies logical qudit `l` is sitting on physical
                position `p` of the position graph.

            cg (CouplingGraph): The connectivity of the hardware.

            perm_data (dict[CircuitPoint, PAMBlockTAPermData]):
                Maps each permutation configuration for every block.

            modify_circuit (bool): Whether to modify the circuit as the
                pass is applied or not. (Default: False)
        """
        # Preprocessing
        D = self.qccd_machine.all_pair_travelling_time()
        F = circuit.front
        decay = [1.0 for _ in range(self.qccd_machine.position_graph.num_qudits)]
        repeated_path = False
        executed_flag = False
        tmp_F = []
        self.iter_count = 0
        initial_extended_set_size = self.extended_set_size
        prev_executed_counts: dict[CircuitPoint, int] = {n: 0 for n in F}
        leading_moves: list[tuple[int, int]] = []
        _logger.debug(f'Starting forward pam pass with ion assignment: {ion_assignment}.')
        longest_path = np.max([len(path) for path in self.qccd_machine.position_graph.all_pairs_shortest_path()])
        if modify_circuit:
            instructions_list = []
            mapped_circuit = Circuit(circuit.num_qudits, circuit.radixes)
            out_data: PAMBlockResultDict = {}
            runtime = 0.0

        # Main Loop
        while len(F) > 0:
            print("Front: ", [circuit[n] for n in F])
            if len(leading_moves) > 2 and leading_moves[-1] == leading_moves[-2] and not executed_flag:
                print("There is repetition..... !!!!!")
                repeated_path = True
            if self.iter_count > math.ceil(longest_path/2):
                _logger.debug(f"Try bruteforce due to multiple steps ({self.iter_count}) to solve one gate")
                print(f"Try bruteforce due to multiple steps ({self.iter_count}) to solve one gate")
                brute_force_moves = self._brute_force_congestion(circuit[list(F)[0]], D, pi, ion_assignment)
                if modify_circuit:
                    for move in brute_force_moves:
                        # mapped_circuit.append_gate(SwapGate(), move)
                        instructions_list.append(
                            [f"Move {move}", f"{ion_assignment}", f"cost: {D[move[0]][move[1]]} seconds"])
                        mapped_circuit.append_gate(BarrierPlaceholder(circuit.num_qudits),
                                                   list(range(circuit.num_qudits)))
                        runtime += D[move[0]][move[1]]
                leading_moves += brute_force_moves
            print("Current ion mapping: ", ion_assignment)
            execute_list = [n for n in F if self.qccd_machine.gate_is_executable(circuit[n], pi, ion_assignment)]

            # Execute the gates and update F
            if len(execute_list) > 0:
                executed_flag = True
                # Rest penalty from reptition
                self.extended_set_size = initial_extended_set_size
                # Add the temporary F to current F
                if tmp_F != []:
                    F += tmp_F
                    tmp_F = []
                F = set(F)
                for n in execute_list:
                    F.remove(n)
                    prev_executed_counts.pop(n)
                    _logger.debug(f'Executing gate at point {n}.')
                    for successor in circuit.next(n):
                        if successor not in prev_executed_counts:
                            prev_executed_counts[successor] = 1
                        else:
                            prev_executed_counts[successor] += 1
                        num_prev_executed = prev_executed_counts[successor]
                        total_num_prev = len(circuit.prev(successor))
                        if num_prev_executed == total_num_prev:
                            F.add(successor)

                # Permute the qubits on the just executed gates
                E = self._calc_extended_set(circuit, F)
                for n in execute_list:
                    op = circuit[n]

                    if isinstance(op.gate, BarrierPlaceholder):
                        if modify_circuit:
                            physical_location = [pi[q] for q in op.location]
                            mapped_circuit.append_gate(op.gate, op.location)
                            instructions_list.append([f"Execute at {physical_location}", f"{ion_assignment}"])
                            mapped_circuit.append_gate(BarrierPlaceholder(circuit.num_qudits),
                                                       list(range(circuit.num_qudits)))
                        continue

                    p1, circ, p2 = self._get_best_perm(
                        circuit,
                        perm_data[n],
                        cg,
                        F,
                        pi,
                        ion_assignment,
                        D,
                        E,
                        op.location,
                    )
                    self._apply_perm(p1, pi, ion_assignment)
                    if modify_circuit:
                        physical_location = [pi[q] for q in op.location]
                        cycle = mapped_circuit.append_circuit(
                            circ,
                            physical_location,
                            True,
                        )
                        mapped_circuit.append_gate(BarrierPlaceholder(circuit.num_qudits),
                                                   list(range(circuit.num_qudits)))
                        instructions_list.append([f"Execute at {physical_location}", f"{ion_assignment}"])
                        new_point = CircuitPoint(cycle, physical_location[0])
                        out_data[new_point] = {
                            'pre_perm': self._global_to_local_perm(p1),
                            'post_perm': self._global_to_local_perm(p2),
                            'original_utry': op.get_unitary(),
                        }
                    self._apply_perm(p2, pi, ion_assignment)
                    if modify_circuit:
                        _logger.debug(f"Ion assignment after applying perm: {ion_assignment}")
                        instructions_list[-1].append(f"{ion_assignment}")
                        # instructions_list.append(f"Ion assignment after applying perm: {ion_assignment}")

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
                        _logger.debug("Try bruteforce due to repeated pattern...")
                        print("Try bruteforce due to repeated pattern...")
                        brute_force_moves = self._brute_force_congestion(circuit[list(F)[0]], D, pi, ion_assignment)
                        if modify_circuit:
                            for move in brute_force_moves:
                                # mapped_circuit.append_gate(SwapGate(), move)
                                instructions_list.append(
                                    [f"Move {move}", f"{ion_assignment}", f"cost: {D[move[0]][move[1]]} seconds"])
                                mapped_circuit.append_gate(BarrierPlaceholder(circuit.num_qudits),
                                                           list(range(circuit.num_qudits)))
                                runtime += D[move[0]][move[1]]
                        leading_moves += brute_force_moves
                        continue
                else:
                    tmp_F = list(F)[1:]
                    F = [list(F)[0]]
                    print(f"Front is modified to {F}.")
            # Pick and apply a swap
            E = self._calc_extended_set(circuit, F)
            # print(f"Extended set: {[circuit[n] for n in E]}")
            best_move = self._get_best_move(circuit, F, E, D, pi, ion_assignment, decay)
            if best_move is None:
                _logger.debug("Try bruteforce due to no best move is found...")
                brute_force_moves = self._brute_force_congestion(circuit[list(F)[0]], D, pi, ion_assignment)
                if modify_circuit:
                    for move in brute_force_moves:
                        # mapped_circuit.append_gate(SwapGate(), move)
                        instructions_list.append(
                            [f"Move {move}", f"{ion_assignment}", f"cost: {D[move[0]][move[1]]} seconds"])
                        mapped_circuit.append_gate(BarrierPlaceholder(circuit.num_qudits),
                                                   list(range(circuit.num_qudits)))
                        runtime += D[move[0]][move[1]]
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
                runtime += D[best_move[0]][best_move[1]]

            # Update loop counter and reset decay if necessary
            self.iter_count += 1
        #_logger.debug(f"Leading moves: {leading_moves}")
        if modify_circuit:
            circuit.become(mapped_circuit)
            return out_data, instructions_list, runtime

    # def _apply_move(
    #         self,
    #         move: tuple[int, int],
    #         ion_assignment: dict,
    #         modify_circuit: bool = False,
    # ) -> None:
    #     """Overide Apply the move to `pi`"""
    #     _logger.debug('applying move %s' % str(move))
    #     # Apply potential move
    #     l1 = list(ion_assignment.keys())[list(ion_assignment.values()).index(move[0])] \
    #         if move[0] in list(ion_assignment.values()) else None
    #     l2 = list(ion_assignment.keys())[list(ion_assignment.values()).index(move[1])] \
    #         if move[1] in list(ion_assignment.values()) else None
    #     if l1 is None and l2 is None:
    #         raise RuntimeError(f'The move {move} is not a valid move as there is no ion in these assignment.')
    #     if l1 is None:
    #         ion_assignment[l2] = move[0]  # Move ion to the adjacent available space
    #     elif l2 is None:
    #         ion_assignment[l1] = move[1]  # Move ion to the adjacent available space
    #     else:
    #         ion_assignment[l1], ion_assignment[l2] = move[1], move[0]  # Inner trap swap
    #     # if modify_circuit:
    #     #     instructions_list.append(
    #     #         [f"Move {move} ", f"{ion_assignment}", f"cost: {D[move[0]][move[1]]} seconds"])
    #     _logger.debug('ion assignment after move %s' % str(ion_assignment))

    def _global_to_local_perm(self, gperm: Sequence[int]) -> tuple[int, ...]:
        """Return the local permutation from a global permutation."""
        global_to_local_map = {q: i for i, q in enumerate(sorted(gperm))}
        return tuple(global_to_local_map[i] for i in gperm)

    def _get_best_perm(
            self,
            circuit: Circuit,
            perm_data: PAMBlockTAPermData,
            cg: CouplingGraph,
            F: set[CircuitPoint],
            pi: list[int],
            ion_assignment: dict,
            D: list[list[float]],
            E: set[CircuitPoint],
            qudits: Sequence[int],
    ) -> tuple[tuple[int, ...], Circuit, tuple[int, ...]]:
        """Return the best permutation to apply before and after a gate."""
        _logger.debug(f'Ion assignment: {ion_assignment}')
        _logger.debug(f'Initial pi: {pi}')
        _logger.debug(f'Initial qudits: {qudits}')
        _logger.debug(f"Qudits location after pi: {[pi[i] for i in qudits]}")
        # Local permutations determine how a gate is permuted in it own space
        local_perms = list(it.permutations(range(len(qudits))))
        # Global perms capture local perms' effect on the full logical space
        global_perms = [
            tuple(qudits[i] for i in lperm)
            for lperm in local_perms
        ]
        # Inverted Permutations
        inv_local_perms = [
            tuple(lperm.index(i) for i in range(len(qudits)))
            for lperm in local_perms
        ]
        # Inverted Global Permutations
        inv_global_perms = [
            tuple(qudits[i] for i in ilperm)
            for ilperm in inv_local_perms
        ]
        # _logger.debug(f'Permutation data: {perm_data}')
        # Gather valid pre, circ, post triples
        pre_circ_post_triples = []
        perm_iter = zip(local_perms, inv_local_perms, inv_global_perms)
        for lperm, ilperm, gperm1 in perm_iter:
            _logger.debug(f'pi after permutation {ilperm}: {[pi[qudits[p]] for p in ilperm]}')
            physical_location = [ion_assignment[pi[qudits[p]]] for p in ilperm]
            _logger.debug(f'physical location: {physical_location}')
            local_graph = cg.get_subgraph(physical_location)
            _logger.debug(f'local_graph: {local_graph}')
            if (len(local_graph._edges) < 2) and len(physical_location) > 2:
                trap_ids = []
                for position in physical_location:
                    tmp_trap_id = self.qccd_machine.get_trap_id(position)
                    if tmp_trap_id is not None:
                        trap_ids.append(tmp_trap_id)
                trap_id = max(set(trap_ids), key=trap_ids.count)
                _logger.debug(f"Trap id: {trap_id}")
                physical_location = list(self.qccd_machine.physical_to_position[trap_id])[:3]
                local_graph = cg.get_subgraph(physical_location)
                _logger.debug(f"Updated local graph: {local_graph}")
                if len(local_graph._edges) < 2:
                    local_graph = CouplingGraph([(0, 1), (1, 2)], 3)
                    _logger.debug(f"Updated local graph: {local_graph}")
            if local_graph.get_qudit_degrees() == [0] * local_graph.num_qudits:
                _logger.debug(f"The coupling graph is empty")
                if len(physical_location) > 2:
                    _logger.debug(f"FullCoupling graph: {cg}, local graph: {local_graph} ")
                    raise ValueError("Where the corner case is bigger than 2 qubits")
                elif len(physical_location) == 2:
                    _logger.debug(f"Trap id of first location:{self.qccd_machine.get_trap_id(physical_location[0])}")
                    _logger.debug(f"Trap id of second location:{self.qccd_machine.get_trap_id(physical_location[1])}")
                    # if (self.qccd_machine.get_trap_id(physical_location[0]) ==
                    #         self.qccd_machine.get_trap_id(physical_location[1])):
                    local_graph = CouplingGraph([(0, 1)], 2)
            _logger.debug(f"FullCoupling graph: {cg}, local graph: {local_graph} ")
            if local_graph in perm_data:
                for perms, circ in perm_data[local_graph].items():
                    if lperm == perms[0]:
                        gperm2 = global_perms[local_perms.index(perms[1])]
                        pre_circ_post_triples.append((gperm1, circ, gperm2))
                        # _logger.debug(f"Pre_circ_triples: {gperm1}, {gperm2} ")

        if len(pre_circ_post_triples) == 0:
            raise RuntimeError(
                'Unable to find any valid permutated circuits.\n'
                'You must embed proper permutation aware synthesis results'
                ' first before running this pass.\n'
                'If you are already running an'
                ' EmbedAllPermutationsPass, try toggling topology selection.',
            )

        # For each permutation get the entangling gate count
        mq_gate_counts = []
        sq_gate_counts = []
        for _, circ, _ in pre_circ_post_triples:
            num_tq_gates = 0
            num_sq_gates = 0
            for gate, count in circ.gate_counts.items():
                if gate.num_qudits >= 2:
                    num_tq_gates += count
                else:
                    num_sq_gates += count
            mq_gate_counts.append(num_tq_gates)
            sq_gate_counts.append(num_sq_gates)

        # If no more gates after this one, then pick the shortest circuit
        if len(F) == 0:
            best_index = np.argmin(mq_gate_counts)
            return pre_circ_post_triples[best_index]

        # Calculate best scoring permutation
        best_triple = pre_circ_post_triples[0]
        best_perm = (best_triple[0], best_triple[2])
        mapping_score = self._score_perm(circuit, F, pi, ion_assignment, D, best_perm, E)
        gate_score = mq_gate_counts[0] * self.gate_count_weight / len(F)
        best_score = mapping_score + gate_score

        for i in range(1, len(pre_circ_post_triples)):
            gperm = (pre_circ_post_triples[i][0], pre_circ_post_triples[i][2])
            score = self._score_perm(circuit, F, pi, ion_assignment, D, gperm, E)
            score = mq_gate_counts[i] * self.gate_count_weight / len(F) + score
            if score < best_score:
                best_score = score
                best_perm = gperm
                best_triple = pre_circ_post_triples[i]
        return best_triple

    def _score_perm(
            self,
            circuit: Circuit,
            F: set[CircuitPoint],
            pi: list[int],
            ion_assignment: dict,
            D: list[list[float]],
            perm: tuple[Sequence[int], Sequence[int]],
            E: set[CircuitPoint],
    ) -> float:
        """Calculating the routing score after applying `perm`."""
        pi_bkp = pi.copy()
        ion_assignment_bkp = ion_assignment.copy()
        pi_c = {q: pi[perm[0][i]] for i, q in enumerate(sorted(perm[0]))}
        for q in perm[0]:
            pi[q] = pi_c[q]
        ion_assignment_tmp = ion_assignment.copy()
        for p in ion_assignment.keys():
            ion_assignment[p] = ion_assignment_tmp[pi.index(p)]

        pi_c = {q: pi[perm[1][i]] for i, q in enumerate(sorted(perm[1]))}
        for q in perm[1]:
            pi[q] = pi_c[q]
        ion_assignment_tmp = ion_assignment.copy()
        for p in ion_assignment.keys():
            ion_assignment[p] = ion_assignment_tmp[pi.index(p)]

        # Front Set Term
        front = 0.0
        for n in F:
            min_term = np.inf
            for q in circuit[n].location:
                term = 0.0
                for p in circuit[n].location:
                    if p == q:
                        continue
                    term += D[ion_assignment[q]][ion_assignment[p]]
                min_term = min(term, min_term)
            front += min_term
        front /= len(F)

        # Extended Set Term
        extend = 0.0
        if len(E) > 0:
            for n in E:
                min_term = np.inf
                for q in circuit[n].location:
                    term = 0.0
                    for p in circuit[n].location:
                        if p == q:
                            continue
                        term += D[ion_assignment[q]][ion_assignment[p]]
                    min_term = min(term, min_term)
                extend += min_term
            extend /= len(E)
            extend *= self.extended_set_weight

        pi[:] = pi_bkp[:]
        ion_assignment.update(ion_assignment_bkp)
        return front + extend
