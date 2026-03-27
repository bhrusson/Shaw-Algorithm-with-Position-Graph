"""This module implements the GeneralizedSabreAlgorithm class."""
from __future__ import annotations

import copy
import logging
import random
from typing import Iterator, Sequence, Tuple, Optional

#import numpy as np

from bqskit.ir.circuit import Circuit
from bqskit.ir.gates.barrier import BarrierPlaceholder
from bqskit.ir.gates.circuitgate import CircuitGate
from bqskit.ir.gates.constant.swap import SwapGate
from bqskit.ir.operation import Operation
from bqskit.ir.point import CircuitPoint
#from bqskit.qis.graph import CouplingGraph

from bqskit_local.position.graph import EdgeCapability
from bqskit_local.position.state import PositionGraphState



#logging.basicConfig(level=logging.DEBUG)
_logger = logging.getLogger(__name__)



class GeneralizedSabreAlgorithmPGS():
    """
    Implements methods for Sabre-based layout and routing algorithms using a
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
    """

    def __init__(
        self,
        decay_delta: float = 0.001,
        decay_reset_interval: int = 5,
        decay_reset_on_gate: bool = True,
        extended_set_size: int = 20,
        extended_set_weight: float = 0.5,
        
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

    def forward_pass(
        self,
        circuit: Circuit,        
        pgs: PositionGraphState,
        modify_circuit: bool = False,
    ) -> None:
        """
        Forward pass of the Sabre algorithm 
        """
        D = pgs.position_graph.move_cost_matrix
        F = set(circuit.front)
        decay = [1.0 for i in range(pgs.num_qudits)]
        iter_count = 0
        prev_executed_counts: dict[CircuitPoint, int] = {n: 0 for n in F}
        leading_swaps: list[tuple[int,int]] = []

        _logger.debug(f'Starting forward sabre pass with pgs: {pgs}.')

        radix = circuit.radixes[0]
        if modify_circuit:
            mapped_circuit = Circuit(circuit.num_qudits, circuit.radixes)

        while len(F) > 0:
            # Get executable gates
            execute_list = [n for n in F if self._can_exe(circuit[n], pgs)]

            if len(execute_list) > 0:
                # Execute gates
                leading_swaps = []
                for n in execute_list:
                    F.remove(n)
                    prev_executed_counts.pop(n)
                    _logger.debug(f'Executing gate at point {n}.')
                    
                    if modify_circuit:
                        op = circuit[n]

                        # physical positions
                        op_positions = [int(pgs.logical_to_position[q]) for q in op.location]

                        # logical qudits for debug
                        logicals = list(op.location)

                        # debug print
                        _logger.debug(
                            f"Executing {op.gate.name} "
                            f"logical{tuple(logicals)} -> physical{tuple(op_positions)}"
                        )

                        mapped_circuit.append_gate(op.gate, op_positions, op.params)

                    # Update successors
                    for successor in circuit.next(n):
                        if successor not in prev_executed_counts:
                            prev_executed_counts[successor] = 1
                        else:
                            prev_executed_counts[successor] += 1
                        num_prev_executed = prev_executed_counts[successor]
                        total_num_prev = len(circuit.prev(successor))
                        if num_prev_executed == total_num_prev:
                            F.add(successor)

                # Reset decay
                if self.decay_reset_on_gate:
                    iter_count = 0
                    for i in range(circuit.num_qudits):
                        decay[i] = 1.0

                continue

            elif len(leading_swaps) > 5 *pgs.num_qudits:
                _logger.debug('Sabre stuck in local minima, backtracking...')

                # Backtrack by removing leading swaps
                for swap in reversed(leading_swaps):
                    self._apply_swap(swap, pgs, decay)
                    if modify_circuit:
                        point = mapped_circuit._rear[swap[0]]
                        mapped_circuit.pop(point)
                leading_swaps = []

                # Override heuristic search to progress
                _logger.debug('Overriding sabre search...')
                all_logical_qudits = [circuit[n].location for n in F]
                qudits = min(
                    all_logical_qudits,
                    key=lambda qs: self._get_distance(qs, pgs, D),
                )
                for swap in self._uphill_swaps(qudits, pgs, D):
                    self._apply_swap(swap, pgs, decay)
                    if modify_circuit:
                        mapped_circuit.append_gate(SwapGate(radix), swap)
                _logger.debug('Stopping override.')
                continue

            # If no gates executable, pick a swap
            E = self._calc_extended_set(circuit, F)
            best_swap = self._get_best_swap(circuit, F, E, D, pgs, decay)

            # Apply swap
            self._apply_swap(best_swap, pgs, decay)
            leading_swaps.append(best_swap)

            if modify_circuit: 
                mapped_circuit.append_gate(SwapGate(radix), best_swap)


            # Update counters
            iter_count += 1
            if iter_count % self.decay_reset_interval == 0:
                for i in range(circuit.num_qudits):
                    decay[i] = 1.0

        if modify_circuit:
            circuit.become(mapped_circuit)

    def backward_pass(
        self,
        circuit: Circuit,
        pgs: PositionGraphState,
    ) -> None:
        D = pgs.position_graph.move_cost_matrix
        F = set(circuit.rear)
        decay = [1.0 for _ in range(pgs.num_qudits)]
        iter_count = 0
        leading_swaps: list[tuple[int, int]] = []
        next_executed_counts: dict[CircuitPoint, int] = {n: 0 for n in F}

        _logger.debug(f'Starting backward sabre pass with pgs: {pgs}.')

        while len(F) > 0:
            _logger.debug(f'Backward front before drain: {sorted(F)}')

            executed_any = False
            while True:
                n = next((pt for pt in F if self._can_exe(circuit[pt], pgs)), None)
                if n is None:
                    break

                executed_any = True
                leading_swaps = []
                F.remove(n)
                next_executed_counts.pop(n, None)

                _logger.debug(f'Executing backward gate at point {n}.')

                for predecessor in circuit.prev(n):
                    next_executed_counts[predecessor] = (
                        next_executed_counts.get(predecessor, 0) + 1
                    )

                    num_next_executed = next_executed_counts[predecessor]
                    successors = list(circuit.next(predecessor))
                    total_num_next = len(successors)

                    _logger.debug(
                        'Backward update pred=%s successors=%s executed=%d/%d',
                        predecessor,
                        successors,
                        num_next_executed,
                        total_num_next,
                    )

                    if num_next_executed == total_num_next:
                        F.add(predecessor)
                        _logger.debug(
                            f'Backward adding predecessor to front: {predecessor}'
                        )

            if executed_any:
                _logger.debug(f'Backward front after drain: {sorted(F)}')
                if self.decay_reset_on_gate:
                    iter_count = 0
                    for i in range(circuit.num_qudits):
                        decay[i] = 1.0
                continue

            if len(leading_swaps) > 5 * pgs.num_qudits:
                _logger.debug('Sabre stuck in local minima, backtracking...')
                for swap in reversed(leading_swaps):
                    self._apply_swap(swap, pgs, decay)
                leading_swaps = []

                _logger.debug('Overriding sabre search...')
                all_logical_qudits = [circuit[n].location for n in F]
                qudits = min(
                    all_logical_qudits,
                    key=lambda qs: self._get_distance(qs, pgs, D),
                )
                for swap in self._uphill_swaps(qudits, pgs, D):
                    self._apply_swap(swap, pgs, decay)
                _logger.debug('Stopping override.')
                continue

            E = self._calc_extended_set(circuit, F)
            best_swap = self._get_best_swap(circuit, F, E, D, pgs, decay)

            self._apply_swap(best_swap, pgs, decay)
            leading_swaps.append(best_swap)

            iter_count += 1
            if iter_count % self.decay_reset_interval == 0:
                for i in range(circuit.num_qudits):
                    decay[i] = 1.0

    def _can_exe(self, op: Operation, pgs: PositionGraphState) -> bool:
        """Return true if `op` is executable given the current mapping in `pgs`."""
        if isinstance(op.gate, BarrierPlaceholder):
            return True

        if isinstance(op.gate, CircuitGate):
            if all(g.num_qudits == 1 for g in op.gate._circuit.gate_set):
                return True

        if op.num_qudits == 1:
            return True

        physical_positions = [int(pgs.logical_to_position[i]) for i in op.location]

        # Guard against partially unplaced mappings.
        if any(p < 0 for p in physical_positions):
            return False

        # Match CouplingGraph behavior:
        # cg.get_subgraph(physical_qudits).is_fully_connected()
        #
        # For a 2-qudit gate, this reduces to checking the execute edge.
        # For larger gates, every pair must be connected.
        for i in range(len(physical_positions)):
            for j in range(i + 1, len(physical_positions)):
                p = physical_positions[i]
                q = physical_positions[j]
                if not pgs.position_graph.execute_graph.has_edge(p, q):
                    return False

        return True
            
        

    def _calc_extended_set(
        self,
        circuit: Circuit,
        F: set[CircuitPoint],
    ) -> set[CircuitPoint]:
        """Calculate the Extended Set for look-ahead capabilities."""
        extended_set: set[CircuitPoint] = set()

        # Use deterministic ordering for debugging/comparison.
        frontier = list(copy.copy(F))

        while len(frontier) > 0 and len(extended_set) < self.extended_set_size:
            n = frontier.pop(0)
            next_nodes = circuit.next(n)
            extended_set.update(next_nodes)
            frontier.extend(next_nodes)

        return extended_set

    def _get_best_swap(
        self,
        circuit: Circuit,
        F: set[CircuitPoint],
        E: set[CircuitPoint],
        D: list[list[float]],
        pgs: PositionGraphState,
        decay: list[float],
    ) -> tuple[int, int]:
        """Return the best swap given the current algorithm state."""
        best_score = float('inf')
        best_swap: tuple[int, int] | None = None

        swap_candidate_list = self._obtain_swaps(circuit, F, pgs)

        _logger.debug("Front layer F: %s", F)
        _logger.debug("Extended set E: %s", E if E is not None else None)
        _logger.debug("Candidate swaps: %s", swap_candidate_list)

        for swap in swap_candidate_list:
            score = self._score_swap(circuit, F, pgs, D, swap, decay, E)
            if score < best_score:
                best_score = score
                best_swap = swap
            l1, l2 = swap
            p1 = int(pgs.logical_to_position[l1])
            p2 = int(pgs.logical_to_position[l2])
            _logger.debug(
                "Swap candidate logical(%d,%d) physical(%d,%d) score=%f map=%s",
                l1, l2, p1, p2, score, pgs.logical_to_position.tolist(),
            )

        if best_swap is None:
            raise RuntimeError('Unable to find best swap.')

        return best_swap
    
    def _obtain_swaps(
        self,
        circuit: Circuit,
        F: set[CircuitPoint],
        pgs: PositionGraphState,
    ) -> set[tuple[int, int]]:
        """Produce all physical swaps with at least one qudit in F."""
        all_qudits: set[int] = set()
        for n in F:
            all_qudits.update(circuit[n].location)

        physical_positions = [int(pgs.logical_to_position[q]) for q in all_qudits]

        swaps: set[tuple[int, int]] = set()
        for pos in physical_positions:
            if pos < 0:
                continue
            for neighbor in pgs.position_graph.swap_neighbors[pos]:
                a = min(pos, neighbor)
                b = max(pos, neighbor)
                swaps.add((a, b))

        return swaps


    def _score_swap(
        self,
        circuit: Circuit,
        F: set[CircuitPoint],
        pgs: PositionGraphState,
        D: list[list[float]],
        swap: tuple[int, int],
        decay: list[float],
        E: set[CircuitPoint],
    ) -> float:
        """Score the candidate swap given the current algorithm state."""
        pos1, pos2 = swap

        # Find which logical qudits occupy the swapped physical positions.
        l1 = int(pgs.position_to_logical[pos1])
        l2 = int(pgs.position_to_logical[pos2])

        # Full permutation and both positions are occupied.
        temp_logical_to_pos = pgs.logical_to_position.copy()
        temp_pos_to_logical = pgs.position_to_logical.copy()

        # Apply potential swap.
        temp_pos_to_logical[pos1], temp_pos_to_logical[pos2] = (
            temp_pos_to_logical[pos2],
            temp_pos_to_logical[pos1],
        )

        if l1 != -1:
            temp_logical_to_pos[l1] = pos2
        if l2 != -1:
            temp_logical_to_pos[l2] = pos1

        # Calculate front set term: average of _get_distance over frontier.
        front = 0.0
        for n in F:
            logical_qudits = circuit[n].location

            # disallow meaningless swaps
            physical_qudits = [int(temp_logical_to_pos[i]) for i in logical_qudits]
            if pos1 in physical_qudits and pos2 in physical_qudits:
                return float('inf')

            front += self._get_distance_from_mapping(logical_qudits, temp_logical_to_pos, D)

        front /= len(F)

        # Calculate extended set term exactly like CG version:
        extend = 0.0
        if len(E) > 0:
            for n in E:
                extend += self._get_distance_from_mapping(
                    circuit[n].location,
                    temp_logical_to_pos,
                    D,
                )
            extend /= len(E)
            extend *= self.extended_set_weight

        # Match CG decay logic exactly.
        decay_factor = max(decay[pos1], decay[pos2])

        return decay_factor * (front + extend)
    
    def _apply_swap(
        self,
        swap: tuple[int, int],
        pgs: PositionGraphState,
        decay: list[float] | None = None,
    ) -> None:
        """Apply the swap to `pgs` and update `decay`."""
        pos1, pos2 = swap
        l1 = int(pgs.position_to_logical[pos1])
        l2 = int(pgs.position_to_logical[pos2])

        _logger.debug(f'Applying swap physical{swap} logical({l1},{l2})')

        pgs.position_to_logical[pos1], pgs.position_to_logical[pos2] = (
            pgs.position_to_logical[pos2],
            pgs.position_to_logical[pos1],
        )

        if l1 != -1:
            pgs.logical_to_position[l1] = pos2
        if l2 != -1:
            pgs.logical_to_position[l2] = pos1

        if decay is not None:
            decay[pos1] += self.decay_delta
            decay[pos2] += self.decay_delta

    def _get_distance(
        self,
        logical_qudits: Sequence[int],
        pgs: PositionGraphState,
        D: list[list[float]],
    ) -> float:
        """Calculate the expected number of swaps to connect logical qudits.

        This matches the CouplingGraph SABRE version exactly.
        """
        mapping = pgs.logical_to_position
        return self._get_distance_from_mapping(logical_qudits, mapping, D)
    
    def _get_distance_from_mapping(
        self,
        logical_qudits: Sequence[int],
        logical_to_position: Sequence[int],
        D: list[list[float]],
    ) -> float:
        """CG-equivalent distance heuristic from an explicit mapping."""
        min_term = float('inf')

        for q in logical_qudits:
            q_pos = int(logical_to_position[q])
            if q_pos < 0:
                return float('inf')

            term = 0.0
            for p in logical_qudits:
                if p == q:
                    continue
                p_pos = int(logical_to_position[p])
                if p_pos < 0:
                    return float('inf')
                term += D[q_pos][p_pos]

            min_term = min(term, min_term)

        return min_term

    def _uphill_swaps(
        self,
        logical_qudits: Sequence[int],
        pgs: PositionGraphState,
        D: list[list[float]],
    ) -> Iterator[tuple[int, int]]:
        """Yield the swaps necessary to bring some of the qudits together."""
        center_qudit = min(
            logical_qudits,
            key=lambda q: sum(
                D[int(pgs.logical_to_position[q])][int(pgs.logical_to_position[p])]
                for p in logical_qudits
                if p != q
            ),
        )

        center_pos = int(pgs.logical_to_position[center_qudit])
        spt = pgs.position_graph.get_shortest_path_tree(center_pos)

        for q in logical_qudits:
            if q == center_qudit:
                continue

            q_pos = int(pgs.logical_to_position[q])
            path = list(reversed(spt[q_pos]))

            _logger.debug(f'Moving {q} to {center_qudit} via {path}.')

            for p1, p2 in zip(path, path[1:]):
                # Match CG behavior: do not move the center qudit itself.
                if center_pos == p1 or center_pos == p2:
                    continue
                yield (p1, p2)

    def _apply_perm(self, perm: Sequence[int], mapping: list[int]) -> None:
        _logger.debug(f'applying permutation {perm}')

        mapping_c = {q: mapping[perm[i]] for i, q in enumerate(sorted(perm))}
        for q in perm:
            mapping[q] = mapping_c[q]
