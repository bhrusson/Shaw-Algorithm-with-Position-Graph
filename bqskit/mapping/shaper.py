"""This module implements the GeneralizedSabreAlgorithm class."""
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
from bqskit.shuttling.qccd.QCCD_machine import QCCDMachineModel
from bqskit.ir.gates.barrier import BarrierPlaceholder

_logger = logging.getLogger(__name__)

# TODO: Bao, organize shaper code here, do not repeat between shaw and shaper (initially)

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
        heuristic_move = True
        # Main Loop
        while len(F) > 0:
            #print("Front: ", [circuit[n] for n in F])
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
                    #print(f"Front is modified to {F}.")
            # Pick and apply a swap
            E = self._calc_extended_set(circuit, F)
            # print(f"Extended set: {[circuit[n] for n in E]}")
            best_move = self._get_best_move(circuit, F, E, D, pi, ion_assignment, decay, heuristic_move)
            if best_move is None:
                _logger.debug("Try bruteforce due to no best move is found...")
                brute_force_moves = self._brute_force_congestion(circuit[list(F)[0]], D, pi, ion_assignment)
                if modify_circuit:
                    for move in brute_force_moves:
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
            extended_set_size: int = 5,
            extended_set_weight: float = 0.5,
            qccd_machine: QCCDMachineModel = None,
            cogestion_rate: float = 0.6
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
        self.cogestion_segment_rate = cogestion_rate

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
        # print(f"Starting forward sabre pass with ion assignment: {ion_assignment}.")

        if modify_circuit:
            instructions_list = []
            mapped_circuit = Circuit(circuit.num_qudits, circuit.radixes)

        if not all(r == circuit.radixes[0] for r in circuit.radixes):
            raise RuntimeError('Cannot currently map to hybrid-level systems.')
        longest_path = np.max([len(path) for path in self.qccd_machine.position_graph.all_pairs_shortest_path()])
        # Main Loop
        executed_flag = False
        heuristic_move = True
        while len(F) > 0:
            if len(leading_moves) > 2 and leading_moves[-1] == leading_moves[-2] and not executed_flag:
                # print("There is repetition..... !!!!!")
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
            # print("Current ion mapping: ", ion_assignment)
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
                    # print(f"Front is modified to {F}.")
            E = self._calc_extended_set(circuit, F)
            # print(f"Extended set: {[circuit[n] for n in E]}")
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

    ########################Local minima resolution#############################
    def _brute_force_congestion(
            self,
            gate: Operation,
            D: list[list[float]],
            pi: list,
            ion_assignment: dict,
    ) -> list[tuple[int, int]]:
        """
            Logical function
        """
        gate_pos = []
        leading_moves = []
        for p in gate.location:
            gate_pos.append(ion_assignment[pi[p]])
        print(f"Trying to solve brute-force congestion at gate {gate_pos} with {ion_assignment}")
        #raise ValueError("Stopping for debug....")
        _logger.debug(f"Trying to solve brute-force congestion at gate {gate_pos} with {ion_assignment}")
        selected_trap_space = []
        selected_end_point = []
        relative_distance = np.inf
        selected_trap_id = None
        # Select which trap to brute force in
        for trap in self.qccd_machine.physical_graph.executable_trap_list:
            all_trap_space = list(self.qccd_machine.physical_to_position[trap.id])
            # endpoints_trap_space = self.qccd_machine.trap_end_points[trap.id]
            _, available_trap_space = self.qccd_machine.trap_is_fully_occupied(trap.id, ion_assignment)
            # Change to endpoints of trap space ... TODO
            relative_dis_to_trap = self._get_distance_from_position_to_trap(gate_pos,
                                                                            all_trap_space,
                                                                            D,
                                                                            ion_assignment)
            """
                Only need to calculate unoccupied spaces (the one near the endpoints) if exits or only the endpoints...  
            """
            num_available_trap_space = len(available_trap_space)
            relative_dis_to_trap -= num_available_trap_space * 100e-6
            # print(f"Considering trap: {trap.id} with distance {relative_dis_to_trap} and number of available space :{num_available_trap_space}")
            if relative_dis_to_trap < relative_distance:
                selected_trap_space = all_trap_space
                # ToDo: If there are more than two endpoints?
                # selected_end_point = self.qccd_machine.trap_end_points[trap.id]
                selected_trap_id = trap.id
                relative_distance = relative_dis_to_trap
        # print(f"Selected trap: {selected_trap_space}", )
        # Select the order of moving position
        distance_to_trap_lst = []
        for pos in gate_pos:
            tmp_distance_to_trap = [D[pos][trap_space] for trap_space in selected_trap_space]
            distance_to_trap = float(np.min(tmp_distance_to_trap))
            end_point = selected_trap_space[int(np.argmax(tmp_distance_to_trap))]
            if end_point not in self.qccd_machine.trap_end_points[selected_trap_id]:
                end_point = self.qccd_machine.trap_end_points[selected_trap_id][0]
            selected_end_point.append(end_point)
            distance_to_trap_lst.append(distance_to_trap)
        gate_pos = np.array(gate_pos)[np.argsort(distance_to_trap_lst)]
        selected_end_point = np.array(selected_end_point)[np.argsort(distance_to_trap_lst)]
        ion_order = [list(ion_assignment.keys())[list(ion_assignment.values()).index(i)] for i in gate_pos]
        _logger.debug(f"Selected end point: {selected_end_point}", )
        # print(f"Selected end point: {selected_end_point}")
        # print("Order of moving ions: ", ion_order)
        for ion_index in range(len(ion_order)):
            gate_pos[ion_index] = ion_assignment[ion_order[ion_index]]
        # print("Gate pos: ", gate_pos)
        # Select the trap space order
        trap_space_distance_to_end_point = []
        #for gate
        for trap_space in selected_trap_space:
            trap_space_distance_to_end_point.append(
                float(np.min([D[trap_space][end_point] for end_point in selected_end_point])))
        selected_trap_space = np.array(selected_trap_space)[np.argsort(trap_space_distance_to_end_point)]
        # print("Order selected traps: ", selected_trap_space)

        # Move the pos to the selected trap
        for pos_idx in range(len(gate_pos)):
            print(
                f"Trying to moving ion {gate_pos[pos_idx]}... to {int(selected_trap_space[len(selected_trap_space) - 1 - pos_idx])}")
            #_logger.debug(f"Trying to moving ion {gate_pos[pos_idx]}...")
            # if pos_idx != len(gate_pos) - 1:
            #     print(f"Endpoint: {selected_end_point[pos_idx+1]}")
            leading_moves += self._brute_force_move(
                int(gate_pos[pos_idx]),
                int(selected_trap_space[len(selected_trap_space) - 1 - pos_idx]), ion_assignment
            )
            print(f"Ion assignment after moving ion {gate_pos[pos_idx]}: {ion_assignment}")
            # _logger.debug(f"Ion assignment after moving ion {gate_pos[pos_idx]}: {ion_assignment}")
            # _logger.debug(f"Selected end point: {selected_end_point}")
            gate_pi = [pi[p] for p in gate.location]
            # _logger.debug(f"Gate pi: {gate_pi}")
            # if selected_end_point in ion_assignment.values():
            #     _logger.debug(f"Position of ion: {list(ion_assignment.keys())[list(ion_assignment.values()).index(selected_end_point)]}")
            """
                If too many ions are in the segment, move them back to trap. (Disable when trying H2 architecture)
            """
            # number_of_segment = len(self.qccd_machine.physical_to_position["segment_space"])
            # ion_at_segment = []
            # for ion in ion_assignment.keys():
            #     if ion_assignment[ion] in self.qccd_machine.physical_to_position["segment_space"]:
            #         ion_at_segment.append(ion)
            # if len(ion_at_segment) / number_of_segment >= self.cogestion_segment_rate:
            #     print("As there are many ions outside the traps, move them to the trap...")
            #     available_spaces = []
            #     for trap in self.qccd_machine.physical_graph.trap_list:
            #         _, available_space = self.qccd_machine.trap_is_fully_occupied(trap.id, ion_assignment)
            #         available_spaces += available_space
            #     while ion_at_segment:
            #         leading_moves += self._brute_force_move(
            #             int(ion_assignment[ion_at_segment[0]]),
            #             int(available_spaces[0]), ion_assignment
            #         )
            #         available_spaces = []
            #         for trap in self.qccd_machine.physical_graph.trap_list:
            #             _, available_space = self.qccd_machine.trap_is_fully_occupied(trap.id, ion_assignment)
            #             available_spaces += available_space
            #         ion_at_segment = []
            #         for ion in ion_assignment.keys():
            #             if ion_assignment[ion] in self.qccd_machine.physical_to_position["segment_space"]:
            #                 ion_at_segment.append(ion)
            #         # print("Ion at segment: ", ion_at_segment)
            #         # print("Available trap space: ", available_spaces)
            #         # print(f"Trying to solve brute-force congestion at gate {gate_pos} with {ion_assignment}")
            #     for ion_index in range(len(ion_order)):
            #         gate_pos[ion_index] = ion_assignment[ion_order[ion_index]]
            """
                Clearing the end-point of the selected trap
            """
            if pos_idx == len(gate_pos) - 1:
                continue
            elif (selected_end_point[pos_idx + 1] in ion_assignment.values() and
                  (pos_idx != len(gate_pos) - 1 and
                   list(ion_assignment.keys())[list(ion_assignment.values()).index(selected_end_point[pos_idx + 1])]
                   not in gate_pi)):
                print(f"Clearing endpoint {selected_end_point[pos_idx+1]}.........")
                end_point_neighbors = self.qccd_machine.position_graph.get_neighbors_of(selected_end_point[pos_idx + 1])
                # if any(position in end_point_neighbors for position in gate_pos):
                #     print(f"Not clearing endpoint as it affect the next gate position...")
                #     for ion_index in range(len(ion_order)):
                #         gate_pos[ion_index] = ion_assignment[ion_order[ion_index]]
                #     print(f"Gate position gate updated to {gate_pos}--------------c")
                #     continue
                occupied_neighbors = []
                for neighbor in end_point_neighbors:
                    if neighbor in list(ion_assignment.values()):
                        occupied_neighbors.append(neighbor)
                end_point_neighbors = [i for i in end_point_neighbors if i not in occupied_neighbors]
                if not end_point_neighbors:
                    potential_blockage = [i for i in occupied_neighbors if self.qccd_machine.get_trap_id(i) is None]
                    leading_moves += self._brute_force_move(
                        int(selected_end_point[pos_idx + 1]), int(potential_blockage[0]),
                        ion_assignment, clearing_ep=True
                    )
                else:
                    self._apply_move((selected_end_point[pos_idx + 1], end_point_neighbors[0]), ion_assignment)
                    leading_moves.append(tuple(sorted((selected_end_point[pos_idx + 1], end_point_neighbors[0]))))
                    # print(f"Perform move {(selected_end_point[pos_idx + 1], end_point_neighbors[0])} to clear "
                    #       f"the endpoint")
            for ion_index in range(len(ion_order)):
                gate_pos[ion_index] = ion_assignment[ion_order[ion_index]]
            print(f"Gate position gate updated to {gate_pos}--------------c")
        return leading_moves

    def _brute_force_move(
            self,
            position: int,
            trap_space: int,
            ion_assignment: dict,
            clearing_ep: bool = False
    ) -> list[tuple[int, int]]:
        """
            Physical function
        """
        # print(
        #     f"Trying to move position {position} to trap space {trap_space} with ion assignment {ion_assignment}")
        leading_moves = []
        shortest_path_pos1 = self.qccd_machine.position_graph.get_shortest_path_tree(position)
        path = shortest_path_pos1[trap_space]
        ion_status = self.qccd_machine.position_to_physical[position]
        for idx_point in range(len(path) - 1):
            possible_move = (path[idx_point], path[idx_point + 1])
            if path[idx_point + 1] not in ion_assignment.values():
                self._apply_move(possible_move, ion_assignment)
                leading_moves.append(tuple(sorted(possible_move)))
                # print(
                #     f"Perform move {(possible_move, ion_assignment)} as there is no ion in the neighbor, "
                #     f"ion status: {ion_status}")
            elif ion_status == 'trap' and self.qccd_machine.position_to_physical[path[idx_point + 1]] == 'trap':
                self._apply_move(possible_move, ion_assignment)
                leading_moves.append(tuple(sorted(possible_move)))
                # print(f"Perform move {possible_move} with inner-swap, ion status: {ion_status}")
            else:
                ion_pos = path[idx_point]
                blockage = path[idx_point + 1]
                # print(f"There is blockage at {blockage}, try to resolve it...")
                leading_moves += self._resolve_congestion(ion_pos, path, blockage, ion_assignment, ion_pos, blockage)
                self._apply_move(possible_move, ion_assignment)
                leading_moves.append(tuple(sorted(possible_move)))
                # print(f"Perform move {possible_move} after resolving blockage")
                # print(f"Ion assignment after resolving blockage: {ion_assignment}")
            if ion_status == 'segment' and self.qccd_machine.position_to_physical[path[idx_point + 1]] == 'trap':
                ion_status = 'trap'
            elif ion_status == 'trap' and self.qccd_machine.position_to_physical[path[idx_point + 1]] == 'segment':
                ion_status = 'segment'
        return leading_moves

    def _resolve_congestion(
            self,
            target: int,
            path: list[int],
            blockage: int,
            ion_assignment: dict,
            original_target: int,
            original_blockage: int,
            num_call: int = 0,
            clearing_ep: bool = False,
    ) -> list[tuple[int, int]]:
        """
            Physical function
        """
        if num_call > 100:
            raise ValueError("Too many repetitive call...")
        # print(
        #     f"Trying to resolve blockage {blockage} from the target {target} path with ion assignment {ion_assignment}")
        _logger.debug(
            f"Trying to resolve blockage {blockage} from the target {target} path with ion assignment {ion_assignment}")
        leading_moves = []
        # print("Path: {}".format(path))
        # print("Original target: ", original_target)
        # print("Original blockage: ", original_blockage)
        target_ion_index = list(ion_assignment.keys())[list(ion_assignment.values()).index(target)]
        blockage_neighbors = self.qccd_machine.position_graph.get_neighbors_of(blockage)
        # print("Initial blockage neighbors: ", blockage_neighbors)
        if original_target in blockage_neighbors:
            blockage_neighbors.remove(original_target)
        if original_blockage in blockage_neighbors:
            blockage_neighbors.remove(original_blockage)
        if target in blockage_neighbors:
            blockage_neighbors.remove(target)
        removed_blockage_neighbors = []
        #print("blockage neighbors: ", blockage_neighbors)
        for neighbor in blockage_neighbors:
            # print("Neighbor: {}".format(neighbor))
            if neighbor in path:
                removed_labeled = True
                if len(blockage_neighbors) == 1:
                    removed_labeled = False
                for neighbor_ext in self.qccd_machine.position_graph.get_neighbors_of(neighbor):
                    if neighbor_ext not in ion_assignment.values() and neighbor_ext not in path:
                        removed_labeled = False
                if removed_labeled:
                    removed_blockage_neighbors.append(neighbor)

        blockage_neighbors = [i for i in blockage_neighbors if i not in removed_blockage_neighbors]
        # print("Blockage_neighbors: ", blockage_neighbors)
        # _logger.debug(f"Blockage neighbors: {blockage_neighbors}")
        potential_blockage = []
        for neighbor in blockage_neighbors:
            if neighbor in ion_assignment.values():
                potential_blockage.append(neighbor)
        for neighbor in potential_blockage:
            blockage_neighbors.remove(neighbor)
        # _logger.debug(f"Potential blockage neighbors: {potential_blockage}")
        # print(f"Updated Blockage neighbors: {blockage_neighbors}")
        # print(f"Potential blockage neighbors: {potential_blockage}")
        # Todo: Instead of simply use the first element, can we do sth better? (DONE)
        if blockage_neighbors:
            congestion = np.array([self.congestion_rate(blockage_neighbor, target, blockage, ion_assignment, depth=self.qccd_machine.max_ion_capacity - 1 + num_call) for
                                         blockage_neighbor in blockage_neighbors])
            _logger.debug(f"Congestion: {congestion}")
            congestion_rates = congestion[:, 0]
            congestion_scores = congestion[:, 1]
            # print("Cogestion rates: ", congestion_rates)
            choosen_indices = list(np.where(congestion_rates == np.min(congestion_rates))) #int(np.argmin(cogestion_rates))
            # print("Choosen indices:", choosen_indices[0])
            # print("Len choosen indices: ", len(choosen_indices[0]))
            # print("Choosen indices types: ", len(choosen_indices[0]))
            if len(choosen_indices[0]) > 1:
                # print("Congestion scores: ", congestion_scores)
                choosen_idx = int(np.argmin(congestion_scores[choosen_indices[0]]))
            else:
                choosen_idx = int(choosen_indices[0][0])
            # print(f"Choose to resolve {blockage_neighbors[choosen_idx]}")
            self._apply_move((blockage, blockage_neighbors[choosen_idx]), ion_assignment)
            leading_moves.append(tuple(sorted((blockage, blockage_neighbors[choosen_idx]))))
            # print(f"Blockage: {blockage}, blockage neighbors: {blockage_neighbors[choosen_idx]}")
            # print(
            #     f"Perform move (1) {(blockage, blockage_neighbors[choosen_idx])} to try resolving the blockage at {blockage}")
            # print("Current ion assignment: ", ion_assignment)
            return leading_moves
        elif potential_blockage:
            congestion = np.array([self.congestion_rate(blockage_neighbor, target, blockage, ion_assignment, depth=self.qccd_machine.max_ion_capacity - 1 + num_call) for
                                        blockage_neighbor in potential_blockage])
            congestion_rates = congestion[:, 0]
            congestion_scores = congestion[:, 1]
            # print(f"Congestion rates: {congestion_rates}")
            if len(congestion_rates) == 0:
                congestion_rates = [1.0]
            choosen_indices = list(np.where(congestion_rates == np.min(congestion_rates))) #int(np.argmin(cogestion_rates))
            # print("Choosen indices:", choosen_indices[0])
            # print("Len choosen indices: ", len(choosen_indices[0]))
            if len(choosen_indices[0]) > 1:
                # print("Congestion score: ", congestion_scores)
                choosen_idx = int(np.argmin(congestion_scores[choosen_indices[0]]))
            else:
                choosen_idx = int(choosen_indices[0][0])

            if congestion_rates[choosen_idx] == 1.0 and clearing_ep is False:
                # print(f"As the best path leads to deadend, we choose to re-add the target to potential neighbor")
                if self.congestion_rate(target, target, blockage, ion_assignment, depth=self.qccd_machine.max_ion_capacity - 1 + num_call)[0] <= congestion_rates[choosen_idx]:
                    # Reverse move (treat target as blockage and vice versa)
                    # print(f"Blockage: {blockage}, target: {target}")
                    leading_moves += self._resolve_congestion(blockage, [], target, ion_assignment,
                                                                    original_target, original_blockage, num_call + 1)
                    self._apply_move((blockage, target), ion_assignment)
                    leading_moves.append(tuple(sorted((blockage, target))))
                    # print(
                    #     f"Perform move (2) {(blockage, target)} to try resolving the blockage at {blockage}")
                    # print("Current ion assignment: ", ion_assignment)
                    blockage = target
                    target = ion_assignment[target_ion_index]
                    # print(f"Blockage: {blockage}, target: {target}")
                    if (self.qccd_machine.get_trap_id(blockage) != self.qccd_machine.get_trap_id(target)
                            and self.qccd_machine.get_trap_id(blockage) is None):
                        leading_moves += self._resolve_congestion(target, [], blockage, ion_assignment,
                                                                  original_target, original_blockage, num_call+1)
                    self._apply_move((blockage, target), ion_assignment)
                    leading_moves.append(tuple(sorted((blockage, target))))
                    # print(
                    #     f"Perform move (2') {(blockage, target)} to try resolving the blockage at {blockage}")
                    # print("Current ion assignment: ", ion_assignment)
                else:
                    raise ValueError("This method does not resolve this case !!!")
            else:
                #print(f"Choose to resolve {potential_blockage[choosen_idx]}")
                leading_moves += self._resolve_congestion(blockage, path, potential_blockage[choosen_idx],
                                                          ion_assignment, original_target, original_blockage,
                                                          num_call + 1)
                self._apply_move((blockage, potential_blockage[choosen_idx]), ion_assignment)
                leading_moves.append(tuple(sorted((blockage, potential_blockage[choosen_idx]))))
                # print(f"Blockage: {blockage}, potential blockage: {potential_blockage[choosen_idx]}")
                # print(
                #     f"Perform move (3) {(blockage, potential_blockage[choosen_idx])} to try resolving the blockage "
                #     f"at {blockage} as we have moved the target ions.")
                # print("Current ion assignment: ", ion_assignment)
        else:
            # print("No blockage neighbors...")
            # print(f"As the best path leads to deadend, we choose to re-add the target to potential neighbor")
            # print(f"Blockage: {blockage}, target: {target}")
            leading_moves += self._resolve_congestion(blockage, [], target, ion_assignment,
                                                      original_target, original_blockage, num_call+1)
            self._apply_move((blockage, target), ion_assignment)
            leading_moves.append(tuple(sorted((blockage, target))))
            # print(
            #     f"Perform move (4) {(blockage, target)} to try resolving the blockage at {blockage}")
            # print("Current ion assignment: ", ion_assignment)
            blockage = target
            target = ion_assignment[target_ion_index]
            # print(f"Blockage: {blockage}, target: {target}")
            if (self.qccd_machine.get_trap_id(blockage) != self.qccd_machine.get_trap_id(target)
                    and self.qccd_machine.get_trap_id(blockage) is None):
                leading_moves += self._resolve_congestion(target, [], blockage, ion_assignment,
                                                          original_target, original_blockage, num_call+1)
            self._apply_move((blockage, target), ion_assignment)
            leading_moves.append(tuple(sorted((blockage, target))))
            # print(
            #     f"Perform move (4') {(blockage, target)} to try resolving the blockage at {blockage}")
            # print("Current ion assignment: ", ion_assignment)
        return leading_moves

    def congestion_rate(self,
                        position: int,
                        target: int,
                        blockage: int,
                        ion_assignment: dict,
                        depth: int = 2):
        # print(f"Position: {position}")
        # print(f"Blockage: {blockage}")
        depth_d_neighbors = []
        full_neighbors = []
        while len(depth_d_neighbors) < depth:
            if len(depth_d_neighbors) == 0:
                node_d_neighbors = self.qccd_machine.position_graph.get_neighbors_of(position)
                if blockage in node_d_neighbors:
                    node_d_neighbors.remove(blockage)
                full_neighbors += node_d_neighbors
            else:
                node_d_neighbors = []
                for pos in depth_d_neighbors[-1]:
                    for node in self.qccd_machine.position_graph.get_neighbors_of(pos):
                        if node not in full_neighbors and node != target and node != blockage and node != position:
                            node_d_neighbors.append(node)
                            full_neighbors.append(node)
            depth_d_neighbors.append(list(set(node_d_neighbors)))
        # print("Depth d neighbors:", depth_d_neighbors)
        num_neighbors = 0
        num_occupied_neighbors = 0
        if position in ion_assignment.values():
            congestion_score = 1.0
        else:
            congestion_score = 0.0
        layer_weight = 1
        for layer in depth_d_neighbors:
            for node in layer:
                if node in ion_assignment.values():
                    congestion_score += 1 * layer_weight
            layer_weight -= .1
        # print(f"Congestion score: {congestion_score}")
        neighbors = list(itertools.chain.from_iterable(depth_d_neighbors))
        neighbors = set(neighbors)
        # print(f"d_neighbors: {neighbors}")
        for neighbor in neighbors:
            num_neighbors += 1
            if neighbor in ion_assignment.values():
                num_occupied_neighbors += 1
        if num_neighbors == 0:
            return 1.0, np.inf
        return num_occupied_neighbors / num_neighbors, congestion_score

    ######################################################################################
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
        heuristic_move = True
        next_executed_counts: dict[CircuitPoint, int] = {n: 0 for n in F}
        _logger.debug(f'Starting backward sabre QCCD pass with ion assignment: {pi}.')
        longest_path = np.max([len(path) for path in self.qccd_machine.position_graph.all_pairs_shortest_path()])
        # Main Loop
        while len(F) > 0:
            # Retrieve executable gates giving the current ion assignment: pi
            # print("Front: ", [circuit[n] for n in F])
            if len(leading_moves) > 2 and leading_moves[-1] == leading_moves[-2] and not executed_flag:
                # print("There is repetition..... !!!!!")
                repeated_path = True
            if self.iter_count > math.ceil(longest_path/4):
                # print(f"Try bruteforce due to multiple steps ({self.iter_count}) to solve one gate")
                leading_moves += self._brute_force_congestion(circuit[list(F)[0]], D, pi, ion_assignment)
            # print("Current ion mapping: ", ion_assignment)
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
                        # print("Try bruteforce due to repeated pattern...")
                        leading_moves += self._brute_force_congestion(circuit[list(F)[0]], D, pi, ion_assignment)
                        continue
                else:
                    tmp_F = list(F)[1:]
                    F = [list(F)[0]]
                    #print(f"Front is modified to {F}.")

            # Pick and apply a swap
            E = self._calc_extended_set(circuit, F)
            #print(f"Extended set: {[circuit[n] for n in E]}")
            best_move = self._get_best_move(circuit, F, E, D, pi, ion_assignment, decay, heuristic_move)
            if best_move is None:
                leading_moves += self._brute_force_congestion(circuit[list(F)[0]], D, pi, ion_assignment)
                continue
            # print(f"Best move: {best_move}")
            self._apply_move(best_move, ion_assignment)
            leading_moves.append(best_move)
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
            heuristic_move: bool,
    ) -> tuple[int, int]:
        """Return the best move given the current algorithm state and ion assignment. (Logical function)"""
        # Track best one
        best_score = np.inf
        best_move = None

        # Gather all considerable moves
        if heuristic_move:
            move_candidate_list = self._obtain_heuristic_moves(circuit, F, pi, ion_assignment)
        else:
            move_candidate_list = self._obtain_moves(circuit, pi, ion_assignment)
        # print("All candidate move: ", move_candidate_list)
        list_of_best_move = []
        # Score them, tracking the best one
        # scores = Parallel(n_jobs=5)(delayed(self._score_move)(circuit, F, D, pi, ion_assignment, move, decay, E)
        #                             for move in move_candidate_list)
        # list_of_best_score = np.argwhere(scores == np.max(scores)).flatten().tolist()
        # list_of_best_moves = list(move_candidate_list)[list_of_best_score]
        for move in move_candidate_list:
            score = self._score_move(circuit, F, D, pi, ion_assignment, move, decay, E)
            if score < best_score:
                best_score = score
                best_move = move
                list_of_best_move = [move]
            elif score == best_score:
                list_of_best_move.append(move)
            # print(f"Score of move {move}: {score}")
        if best_move is None:
            # print("*** Unable to find best move. ***")
            return None
            # raise RuntimeError('Unable to find best move.')
        # print(f"List of best move: {list_of_best_move}")
        if len(list_of_best_move) == 1:
            return best_move
        else:
            # ToDo: There is some case where we have to decide between moves with
            #  same score, we choose move with the most potential influence... (Done)
            move_relative_scores = []
            for move in list_of_best_move:
                move_relative_score = 0.0
                if D[move[0]][move[1]] == self.qccd_machine.timing_data['merge']:
                    move_relative_score = -self.qccd_machine.timing_data['merge']
                for n in F:
                    location = circuit[n].location
                    p = [ion_assignment[pi[loc]] for loc in location]
                    if any([pos in move for pos in p]):
                        move_relative_score = 0.0
                    else:
                        move_relative_score += np.sum([np.min([D[pos][move[0]], D[pos][move[1]]]) for pos in p])
                move_relative_scores.append(move_relative_score)
            return list_of_best_move[np.argmin(move_relative_scores)]

    def _obtain_heuristic_moves(
            self,
            circuit: Circuit,
            F: set[CircuitPoint],
            pi: list,
            ion_assignment: dict,
    ) -> set[tuple[int, int]]:
        """Produce all possible realizable physical moves w.r.t frontier given the current QCCD hardware."""
        position_graph = self.qccd_machine.position_graph
        position_to_physical = self.qccd_machine.position_to_physical
        physical_qudit_positions = []
        for location in F:
            block = circuit[location]
            for qudit in block.location:
                physical_qudit_positions.append(ion_assignment[pi[qudit]])
        physical_qudit_positions = list(set(physical_qudit_positions))
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

    def _obtain_moves(
            self,
            circuit: Circuit,
            pi: list,
            ion_assignment: dict,
    ) -> set[tuple[int, int]]:
        """Produce all possible realizable physical moves given the current QCCD hardware."""
        position_graph = self.qccd_machine.position_graph
        position_to_physical = self.qccd_machine.position_to_physical
        physical_qudit_positions = [ion_assignment[pi[i]] for i in circuit.active_qudits]
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
    #     # local min situation and we need to modify this (2 point to 2 point on the same trap)
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
        """
            All position of trap are considered...
        """
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
        """
            Only consider the endpoint
        """
        # for space in available_space:
        #     blockage = [self.qccd_machine.path_is_blocked(positions[i], space, ion_assignment)[1]
        #                 for i in range(len(positions))]
        #     space_distance = np.sum([D[positions[i]][space] for i in range(len(positions))])
        #     for block_w in blockage:
        #         for block_position in block_w:
        #             if self.qccd_machine.position_to_physical[block_position] == 'segment':
        #                 resolve_cost = self.qccd_machine.timing_data['junction_Y']
        #             elif self.qccd_machine.position_to_physical[block_position] == 'trap':
        #                 min_to_endpoints = np.min([D[block_position][end_point] for end_point in
        #                                            self.qccd_machine.trap_end_points[
        #                                                self.qccd_machine.get_trap_id(block_position)]])
        #                 resolve_cost = (self.qccd_machine.timing_data['split'] + min_to_endpoints)
        #             else:
        #                 raise ValueError("The block position is undefined as it sit on ",
        #                                  self.qccd_machine.position_to_physical[block_position])
        #             space_distance += resolve_cost
        #     distance = np.min([space_distance, distance])
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
                    #endpoints_trap_space = self.qccd_machine.trap_end_points[trap.id]
                    # Change to endpoints of trap space ... TODO
                    distance_to_trap = np.min([self._get_distance_from_position_to_trap(p, available_space, D,
                                                                                        ion_assignment),
                                               distance_to_trap])
                return distance_to_trap
        # Multi-qudit case
        p_list = [ion_assignment[pi[logical_qudit]] for logical_qudit in logical_qudits]
        pairwise_distance = [D[p1][p2] for p1, p2 in combinations(p_list, 2)]
        distance = np.max(pairwise_distance)
        # Distance to nearest trap from pg
        trap_p = [self.qccd_machine.get_trap_id(p) for p in p_list]
        if trap_p.count(None) == 0 and trap_p.count(trap_p[0]) == len(trap_p):
            total_F = 0.0
        else:
            total_F = np.inf
            for trap in self.qccd_machine.physical_graph.executable_trap_list:
                #endpoints_trap_space = self.qccd_machine.trap_end_points[trap.id]
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
                    # Change to endpoints of trap space ... TODO
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
        _logger.debug('ion assignment after move %s' % str(ion_assignment))
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
        _logger.debug('initial pi %s' % str(pi))
        _logger.debug('initial ion_assignment %s' % str(ion_assignment))
        _logger.debug('applying permutation %s' % str(perm))
        pi_c = {q: pi[perm[i]] for i, q in enumerate(sorted(perm))}
        for q in perm:
            pi[q] = pi_c[q]
        ion_assignment_tmp = ion_assignment.copy()
        for p in ion_assignment.keys():
            ion_assignment[p] = ion_assignment_tmp[pi.index(p)]
        _logger.debug('ion assignment after permutation %s ' % str(ion_assignment))


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
    ion_assignment = {0: 0, 1: 1, 2: 2,
                      3: 6, 4: 10}
    circuit = Circuit.from_file("data/input_qasms/Grover_5.qasm")
    # circuit = Circuit.from_file("data/input_qasms/PhaseEstimator_5.qasm")
    # ion_assignment = {0: 0, 1: 1, 2: 2,
    #                   3: 6, 4: 10, 5: 11,
    #                   6: 7, 7: 9}
    # circuit = Circuit.from_file("data/input_qasms/Grover_8.qasm")
    # circuit = Circuit.from_file("data/input_qasms/PhaseEstimator_8.qasm")
    pi = [i for i in range(5)]
    # ion_assignment = {0: 6, 1: 2, 2: 0,
    #               3: 1, 4: 5, 5: 3,
    #               6: 7, 7: 8, 8: 4}
    # circuit = Circuit.from_file("data/input_qasms/adder9_trapsize3.qasm")
    # circuit = Circuit(5)
    # circuit.append_gate(CNOTGate(), (0, 1))
    # circuit.append_gate(CNOTGate(), (1, 2))

    mapping_algo = QCCDMappingAlgorithm(qccd_machine=machine_model,
                                        decay_delta=0.0,
                                        extended_set_size=5,
                                        extended_set_weight=0.5)
    mapping_algo.forward_pass(circuit, pi, ion_assignment, modify_circuit=True)
