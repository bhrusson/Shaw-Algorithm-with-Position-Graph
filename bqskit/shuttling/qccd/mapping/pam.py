"""This module implements the PAMRoutingPass class."""
from __future__ import annotations

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
        gate_count_weight: float = 0.1,
        decay_delta: float = 0.001,
        decay_reset_interval: int = 5,
        decay_reset_on_gate: bool = True,
        extended_set_size: int = 20,
        extended_set_weight: float = 0.5,
        qccd_machine: QCCDMachineModel = None
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
            qccd_machine
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
    ) -> PAMBlockResultDict | None:
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

        if modify_circuit:
            mapped_circuit = Circuit(circuit.num_qudits, circuit.radixes)
            out_data: PAMBlockResultDict = {}

        # Main Loop
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
                        new_point = CircuitPoint(cycle, physical_location[0])
                        out_data[new_point] = {
                            'pre_perm': self._global_to_local_perm(p1),
                            'post_perm': self._global_to_local_perm(p2),
                            'original_utry': op.get_unitary(),
                        }

                    self._apply_perm(p2, pi, ion_assignment)

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

            if modify_circuit:
                mapped_circuit.append_gate(SwapGate(), best_move)

            # Update loop counter and reset decay if necessary
            self.iter_count += 1

        if modify_circuit:
            circuit.become(mapped_circuit)
            return out_data

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

        # Gather valid pre, circ, post triples
        pre_circ_post_triples = []
        perm_iter = zip(local_perms, inv_local_perms, inv_global_perms)
        for lperm, ilperm, gperm1 in perm_iter:
            physical_location = [pi[qudits[p]] for p in ilperm]
            local_graph = cg.get_subgraph(physical_location)
            if local_graph in perm_data:
                for perms, circ in perm_data[local_graph].items():
                    if lperm == perms[0]:
                        gperm2 = global_perms[local_perms.index(perms[1])]
                        pre_circ_post_triples.append((gperm1, circ, gperm2))

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
        ion_c = {q: ion_assignment[pi_c[i]] for i, q in enumerate(sorted(perm[0]))}
        for q in perm[0]:
            pi[q] = pi_c[q]
        for p, q in zip(sorted(pi), sorted(perm[0])):
            ion_assignment[p] = ion_c[q]

        pi_c = {q: pi[perm[1][i]] for i, q in enumerate(sorted(perm[1]))}
        ion_c = {q: ion_assignment[pi_c[i]] for i, q in enumerate(sorted(perm[1]))}
        for q in perm[1]:
            pi[q] = pi_c[q]
        for p, q in zip(sorted(pi), sorted(perm[1])):
            ion_assignment[p] = ion_c[q]

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
        ion_assignment[:] = ion_assignment_bkp[:]
        return front + extend
