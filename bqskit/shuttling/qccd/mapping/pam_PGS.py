"""This module implements the permutation-aware PGS QCCD mapping algorithm."""
from __future__ import annotations

import copy
import itertools as it
import logging
import math
from typing import Dict
from typing import Literal
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import TypedDict

import numpy as np

from bqskit.ir.circuit import Circuit
from bqskit.ir.gates.barrier import BarrierPlaceholder
from bqskit.ir.point import CircuitPoint
from bqskit.qis.graph import CouplingGraph
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix

from bqskit.shuttling.qccd.QCCD_machine_PGS import QCCDMachineModel
from bqskit.shuttling.qccd.QCCD_mapping_PGS import QCCDMappingAlgorithm
from bqskit.shuttling.qccd.position_graph_state_PGS import PositionGraphState

_logger = logging.getLogger(__name__)

PAMBlockPermData = Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], Circuit]
PAMBlockTAPermData = Dict[CouplingGraph, PAMBlockPermData]


class PAMBlockResultData(TypedDict):
    pre_perm: tuple[int, ...]
    post_perm: tuple[int, ...]
    original_utry: UnitaryMatrix


PAMBlockResultDict = Dict[CircuitPoint, PAMBlockResultData]


class PermutationAwareQCCDMappingAlgorithmPGS(QCCDMappingAlgorithm):
    """Permutation-aware QCCD mapper using PositionGraphState as live state."""

    def __init__(
        self,
        gate_count_weight: float = .1,
        decay_delta: float = 0.001,
        decay_reset_interval: int = 5,
        decay_reset_on_gate: bool = True,
        extended_set_size: int = 5,
        extended_set_weight: float = 0.5,
        qccd_machine: QCCDMachineModel = None,
        cogestion_rate: float = 0.6,
    ) -> None:
        if not isinstance(gate_count_weight, float):
            raise TypeError(
                f'Expected float for gate_count_weight, got {type(gate_count_weight)}',
            )

        self.gate_count_weight = gate_count_weight
        super().__init__(
            decay_delta,
            decay_reset_interval,
            decay_reset_on_gate,
            extended_set_size,
            extended_set_weight,
            qccd_machine,
            cogestion_rate,
        )

    @overload
    def forward_pass(
        self,
        circuit: Circuit,
        pgs: PositionGraphState,
        cg: CouplingGraph,
        perm_data: dict[CircuitPoint, PAMBlockTAPermData],
        modify_circuit: Literal[False] = False,
    ) -> None:
        ...

    @overload
    def forward_pass(
        self,
        circuit: Circuit,
        pgs: PositionGraphState,
        cg: CouplingGraph,
        perm_data: dict[CircuitPoint, PAMBlockTAPermData],
        modify_circuit: Literal[True],
    ) -> tuple[PAMBlockResultDict, list, float]:
        ...

    def forward_pass(  # type: ignore[override]
        self,
        circuit: Circuit,
        pgs: PositionGraphState,
        cg: CouplingGraph,
        perm_data: dict[CircuitPoint, PAMBlockTAPermData],
        modify_circuit: bool = False,
    ) -> tuple[PAMBlockResultDict, list, float] | None:
        """Apply a forward pass of the PAM algorithm to the live PGS state."""
        D = self.qccd_machine.all_pair_travelling_time()
        F = circuit.front
        decay = [1.0 for _ in range(self.qccd_machine.num_positions)]
        repeated_path = False
        executed_flag = False
        tmp_F = []
        self.iter_count = 0
        initial_extended_set_size = self.extended_set_size
        prev_executed_counts: dict[CircuitPoint, int] = {n: 0 for n in F}
        leading_moves: list[tuple[int, int]] = []
        _logger.debug(
            'Starting forward PAM PGS pass with ion assignment: %s.',
            self._assignment_from_pgs(pgs),
        )
        longest_path = self.qccd_machine.num_positions
        if modify_circuit:
            instructions_list = []
            mapped_circuit = Circuit(
                self.qccd_machine.num_positions,
                [circuit.radixes[0]] * self.qccd_machine.num_positions,
            )
            barrier_qudits = list(range(self.qccd_machine.num_positions))
            out_data: PAMBlockResultDict = {}
            runtime = 0.0
        heuristic_move = True

        while len(F) > 0:
            if len(leading_moves) > 2 and leading_moves[-1] == leading_moves[-2] and not executed_flag:
                repeated_path = True
            if self.iter_count > math.ceil(longest_path / 2):
                log_pgs = self._clone_pgs(pgs, slot='pam_forward_log')
                brute_force_moves = self._brute_force_congestion(
                    circuit[self._sorted_points(F)[0]], D, pgs,
                )
                if modify_circuit:
                    for move in brute_force_moves:
                        if not self._append_logged_move_instruction(
                            instructions_list,
                            move,
                            log_pgs,
                            D,
                        ):
                            continue
                        mapped_circuit.append_gate(
                            BarrierPlaceholder(self.qccd_machine.num_positions),
                            barrier_qudits,
                        )
                        runtime += D[move[0]][move[1]]
                leading_moves += brute_force_moves

            execute_list = self._sorted_points([
                n for n in F if self.qccd_machine.gate_is_executable_pgs(circuit[n], pgs)
            ])

            if len(execute_list) > 0:
                executed_flag = True
                self.extended_set_size = initial_extended_set_size
                if tmp_F:
                    F += tmp_F
                    tmp_F = []
                F = set(F)
                for n in execute_list:
                    F.remove(n)
                    prev_executed_counts.pop(n)
                    for successor in circuit.next(n):
                        if successor not in prev_executed_counts:
                            prev_executed_counts[successor] = 1
                        else:
                            prev_executed_counts[successor] += 1
                        if prev_executed_counts[successor] == len(circuit.prev(successor)):
                            F.add(successor)

                E = self._calc_extended_set(circuit, F)
                for n in execute_list:
                    op = circuit[n]

                    if isinstance(op.gate, BarrierPlaceholder):
                        if modify_circuit:
                            physical_location = [self._position_of_qudit(q, pgs) for q in op.location]
                            mapped_circuit.append_gate(op.gate, op.location)
                            instructions_list.append(
                                [f"Execute at {physical_location}", f"{self._assignment_from_pgs(pgs)}"],
                            )
                            mapped_circuit.append_gate(
                                BarrierPlaceholder(self.qccd_machine.num_positions),
                                barrier_qudits,
                            )
                        continue

                    p1, circ, p2 = self._get_best_perm(
                        circuit,
                        perm_data[n],
                        cg,
                        F,
                        pgs,
                        D,
                        E,
                        op.location,
                    )
                    self._apply_perm_pgs(p1, pgs)
                    if modify_circuit:
                        physical_location = [self._position_of_qudit(q, pgs) for q in op.location]
                        cycle = mapped_circuit.append_circuit(circ, physical_location, True)
                        mapped_circuit.append_gate(
                            BarrierPlaceholder(self.qccd_machine.num_positions),
                            barrier_qudits,
                        )
                        instructions_list.append(
                            [f"Execute at {physical_location}", f"{self._assignment_from_pgs(pgs)}"],
                        )
                        new_point = CircuitPoint(cycle, physical_location[0])
                        out_data[new_point] = {
                            'pre_perm': self._global_to_local_perm(p1),
                            'post_perm': self._global_to_local_perm(p2),
                            'original_utry': op.get_unitary(),
                        }
                    self._apply_perm_pgs(p2, pgs)
                    if modify_circuit:
                        instructions_list[-1].append(f"{self._assignment_from_pgs(pgs)}")

                if self.decay_reset_on_gate:
                    self.iter_count = 0
                    for i in range(circuit.num_qudits):
                        decay[i] = 1.0
                continue

            executed_flag = False
            if repeated_path:
                repeated_path = False
                if len(F) == 1 and self.extended_set_size != 0:
                    self.extended_set_size = 0
                elif len(F) == 1 and self.extended_set_size == 0:
                    if self.iter_count > 2:
                        log_pgs = self._clone_pgs(pgs, slot='pam_forward_log')
                        brute_force_moves = self._brute_force_congestion(
                            circuit[self._sorted_points(F)[0]], D, pgs,
                        )
                        if modify_circuit:
                            for move in brute_force_moves:
                                if not self._append_logged_move_instruction(
                                    instructions_list,
                                    move,
                                    log_pgs,
                                    D,
                                ):
                                    continue
                                mapped_circuit.append_gate(
                                    BarrierPlaceholder(self.qccd_machine.num_positions),
                                    barrier_qudits,
                                )
                                runtime += D[move[0]][move[1]]
                        leading_moves += brute_force_moves
                        continue
                else:
                    ordered_F = self._sorted_points(F)
                    tmp_F = ordered_F[1:]
                    F = [ordered_F[0]]

            E = self._calc_extended_set(circuit, F)
            best_move = self._get_best_move(circuit, F, E, D, decay, heuristic_move, pgs)
            if best_move is None:
                log_pgs = self._clone_pgs(pgs, slot='pam_forward_log')
                brute_force_moves = self._brute_force_congestion(
                    circuit[self._sorted_points(F)[0]], D, pgs,
                )
                if modify_circuit:
                    for move in brute_force_moves:
                        if not self._append_logged_move_instruction(
                            instructions_list,
                            move,
                            log_pgs,
                            D,
                        ):
                            continue
                        mapped_circuit.append_gate(
                            BarrierPlaceholder(self.qccd_machine.num_positions),
                            barrier_qudits,
                        )
                        runtime += D[move[0]][move[1]]
                leading_moves += brute_force_moves
                continue

            log_pgs = self._clone_pgs(pgs, slot='pam_forward_log')
            self._apply_move(best_move, pgs=pgs)
            leading_moves.append(best_move)
            if modify_circuit:
                if self._append_logged_move_instruction(
                    instructions_list,
                    best_move,
                    log_pgs,
                    D,
                ):
                    mapped_circuit.append_gate(
                        BarrierPlaceholder(self.qccd_machine.num_positions),
                        barrier_qudits,
                    )
                    runtime += D[best_move[0]][best_move[1]]
            self.iter_count += 1

        if modify_circuit:
            circuit.become(mapped_circuit)
            return out_data, instructions_list, runtime
        return None

    def _global_to_local_perm(self, gperm: Sequence[int]) -> tuple[int, ...]:
        global_to_local_map = {q: i for i, q in enumerate(sorted(gperm))}
        return tuple(global_to_local_map[i] for i in gperm)

    def _apply_perm_pgs(self, perm: Sequence[int], pgs: PositionGraphState) -> None:
        """Apply a logical permutation directly to the live PGS state."""
        affected = sorted(int(q) for q in perm)
        old_positions = {q: self._position_of_qudit(q, pgs) for q in affected}

        for q in affected:
            pgs.position_to_logical[old_positions[q]] = -1

        for i, logical in enumerate(affected):
            new_position = old_positions[int(perm[i])]
            pgs.logical_to_position[logical] = new_position
            pgs.position_to_logical[new_position] = logical

    def _get_best_perm(
        self,
        circuit: Circuit,
        perm_data: PAMBlockTAPermData,
        cg: CouplingGraph,
        F: set[CircuitPoint],
        pgs: PositionGraphState,
        D: list[list[float]],
        E: set[CircuitPoint],
        qudits: Sequence[int],
    ) -> tuple[tuple[int, ...], Circuit, tuple[int, ...]]:
        """Return the best permutation to apply before and after a gate."""
        local_perms = list(it.permutations(range(len(qudits))))
        global_perms = [tuple(qudits[i] for i in lperm) for lperm in local_perms]
        inv_local_perms = [
            tuple(lperm.index(i) for i in range(len(qudits)))
            for lperm in local_perms
        ]
        inv_global_perms = [
            tuple(qudits[i] for i in ilperm)
            for ilperm in inv_local_perms
        ]

        assignment = self._assignment_from_pgs(pgs)
        pre_circ_post_triples = []
        perm_iter = zip(local_perms, inv_local_perms, inv_global_perms)
        for lperm, ilperm, gperm1 in perm_iter:
            physical_location = [assignment[qudits[p]] for p in ilperm]
            local_graph = cg.get_subgraph(physical_location)
            if (len(local_graph._edges) < 2) and len(physical_location) > 2:
                trap_ids = []
                for position in physical_location:
                    tmp_trap_id = self.qccd_machine.get_trap_id(position)
                    if tmp_trap_id is not None:
                        trap_ids.append(tmp_trap_id)
                trap_id = max(set(trap_ids), key=trap_ids.count)
                physical_location = list(self.qccd_machine.physical_to_position[trap_id])[:3]
                local_graph = cg.get_subgraph(physical_location)
                if len(local_graph._edges) < 2:
                    local_graph = CouplingGraph([(0, 1), (1, 2)], 3)
            if local_graph.get_qudit_degrees() == [0] * local_graph.num_qudits:
                if len(physical_location) > 2:
                    raise ValueError('Corner case larger than 2 qudits not handled.')
                elif len(physical_location) == 2:
                    local_graph = CouplingGraph([(0, 1)], 2)

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

        mq_gate_counts = []
        for _, circ, _ in pre_circ_post_triples:
            num_tq_gates = 0
            for gate, count in circ.gate_counts.items():
                if gate.num_qudits >= 2:
                    num_tq_gates += count
            mq_gate_counts.append(num_tq_gates)

        if len(F) == 0:
            return pre_circ_post_triples[int(np.argmin(mq_gate_counts))]

        best_triple = pre_circ_post_triples[0]
        best_perm = (best_triple[0], best_triple[2])
        mapping_score = self._score_perm(circuit, F, D, best_perm, E, pgs)
        gate_score = mq_gate_counts[0] * self.gate_count_weight / len(F)
        best_score = mapping_score + gate_score

        for i in range(1, len(pre_circ_post_triples)):
            gperm = (pre_circ_post_triples[i][0], pre_circ_post_triples[i][2])
            score = self._score_perm(circuit, F, D, gperm, E, pgs)
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
        D: list[list[float]],
        perm: tuple[Sequence[int], Sequence[int]],
        E: set[CircuitPoint],
        pgs: PositionGraphState,
    ) -> float:
        """Calculate the routing score after applying a pair of permutations."""
        snapshot = self._snapshot_logical_positions(
            pgs,
            list(perm[0]) + list(perm[1]),
        )
        try:
            self._apply_perm_pgs(perm[0], pgs)
            self._apply_perm_pgs(perm[1], pgs)

            front = 0.0
            for n in F:
                front += self._get_distance(circuit[n].location, D, pgs)
            front /= len(F)

            extend = 0.0
            if len(E) > 0:
                for n in E:
                    extend += self._get_distance(circuit[n].location, D, pgs)
                extend /= len(E)
                extend *= self.extended_set_weight

            return front + extend
        finally:
            self._restore_logical_positions(pgs, snapshot)
