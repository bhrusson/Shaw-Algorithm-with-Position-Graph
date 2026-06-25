"""This module implements the GeneralizedSabreAlgorithm class."""
from __future__ import annotations

import copy
from dataclasses import dataclass
import logging
from typing import Iterator
from typing import Sequence

import numpy as np

from bqskit.ir.circuit import Circuit
from bqskit.ir.gates.barrier import BarrierPlaceholder
from bqskit.ir.gates.circuitgate import CircuitGate
from bqskit.ir.gates.constant.swap import SwapGate
from bqskit.ir.operation import Operation
from bqskit.ir.point import CircuitPoint
from bqskit.qis.graph import CouplingGraph

_logger = logging.getLogger(__name__)


@dataclass
class HeuristicRegionCache:
    gate_qudits: dict[CircuitPoint, tuple[int, ...]]
    gate_scores: dict[CircuitPoint, float]
    points_by_logical: dict[int, set[CircuitPoint]]
    total_score: float
    num_points: int


@dataclass
class HeuristicScoreContext:
    front: HeuristicRegionCache
    extend: HeuristicRegionCache


class GeneralizedSabreAlgorithm():
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
        use_legacy_can_exe: bool = False,
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

        if not isinstance(use_legacy_can_exe, bool):
            raise TypeError(
                'Expected bool for use_legacy_can_exe'
                f', got {type(use_legacy_can_exe)}',
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
        self.use_legacy_can_exe = use_legacy_can_exe

    def _build_physical_to_logical(self, pi: Sequence[int]) -> list[int]:
        """Build the inverse physical-to-logical mapping for scoring."""
        physical_to_logical = [-1 for _ in range(len(pi))]
        for logical, physical in enumerate(pi):
            physical_to_logical[int(physical)] = int(logical)
        return physical_to_logical

    def _build_heuristic_region(
        self,
        circuit: Circuit,
        points: set[CircuitPoint],
        pi: Sequence[int],
        D: list[list[float]],
    ) -> HeuristicRegionCache:
        gate_qudits: dict[CircuitPoint, tuple[int, ...]] = {}
        gate_scores: dict[CircuitPoint, float] = {}
        points_by_logical: dict[int, set[CircuitPoint]] = {}
        total_score = 0.0

        for point in points:
            logical_qudits = tuple(int(q) for q in circuit[point].location)
            score = self._get_distance(logical_qudits, pi, D)
            gate_qudits[point] = logical_qudits
            gate_scores[point] = score
            total_score += score

            for logical in logical_qudits:
                points_by_logical.setdefault(logical, set()).add(point)

        return HeuristicRegionCache(
            gate_qudits=gate_qudits,
            gate_scores=gate_scores,
            points_by_logical=points_by_logical,
            total_score=total_score,
            num_points=len(points),
        )

    def _build_heuristic_context(
        self,
        circuit: Circuit,
        F: set[CircuitPoint],
        E: set[CircuitPoint],
        pi: Sequence[int],
        D: list[list[float]],
    ) -> HeuristicScoreContext:
        return HeuristicScoreContext(
            front=self._build_heuristic_region(circuit, F, pi, D),
            extend=self._build_heuristic_region(circuit, E, pi, D),
        )

    def _affected_region_points(
        self,
        region: HeuristicRegionCache,
        swapped_logicals: Sequence[int],
    ) -> set[CircuitPoint]:
        affected: set[CircuitPoint] = set()
        for logical in swapped_logicals:
            if logical == -1:
                continue
            affected.update(region.points_by_logical.get(logical, set()))
        return affected

    def forward_pass(
        self,
        circuit: Circuit,
        pi: list[int],
        cg: CouplingGraph,
        modify_circuit: bool = False,
    ) -> None:
        """
        Apply a forward pass of the Sabre algorithm to `pi`.

        Args:
            circuit (Circuit): The circuit to pass over.

            pi (list[int]): The input logical-to-physical mapping. This
                maps logical qudits to physical qudits. So, `pi[l] == p`
                implies logical qudit `l` is sitting on physical qudit `p`.

            cg (CouplingGraph): The connectivity of the hardware.

            modfiy_circuit (bool): Whether to modify the circuit as the
                pass is applied or not. (Default: False)
        """
        # Preprocessing
        D = cg.all_pairs_shortest_path()
        F = circuit.front
        decay = [1.0 for i in range(circuit.num_qudits)]
        iter_count = 0
        prev_executed_counts: dict[CircuitPoint, int] = {n: 0 for n in F}
        leading_swaps: list[tuple[int, int]] = []
        _logger.debug(f'Starting forward sabre pass with pi: {pi}.')

        if not all(r == circuit.radixes[0] for r in circuit.radixes):
            raise RuntimeError('Cannot currently map to hybrid-level systems.')
        radix = circuit.radixes[0]

        if modify_circuit:
            mapped_circuit = Circuit(circuit.num_qudits, circuit.radixes)

        # Main Loop
        while len(F) > 0:

            # Retrieve executable gates giving the current mapping `pi`
            execute_list = [n for n in F if self._can_exe(circuit[n], pi, cg)]

            # Execute the gates and update F
            if len(execute_list) > 0:
                leading_swaps = []

                for n in execute_list:
                    F.remove(n)
                    prev_executed_counts.pop(n)
                    _logger.debug(f'Executing gate at point {n}.')

                    if modify_circuit:
                        op = circuit[n]
                        physical_location = [pi[q] for q in op.location]
                        mapped_circuit.append_gate(
                            op.gate,
                            physical_location,
                            op.params,
                        )

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

            # If execute list is empty, check for local-minima
            elif len(leading_swaps) > 5 * cg.num_qudits:
                _logger.debug('Sabre stuck in local minima, backtracking...')

                # Backtrack by removing leading swaps
                for swap in reversed(leading_swaps):
                    self._apply_swap(swap, pi, decay)
                    if modify_circuit:
                        point = mapped_circuit._rear[swap[0]]
                        mapped_circuit.pop(point)
                leading_swaps = []

                # Override heuristic search to progress
                _logger.debug('Overriding sabre search...')
                all_logical_qudits = [circuit[n].location for n in F]
                qudits = min(
                    all_logical_qudits,
                    key=lambda qs: self._get_distance(qs, pi, D),
                )
                for swap in self._uphill_swaps(qudits, cg, pi, D):
                    self._apply_swap(swap, pi, decay)
                    if modify_circuit:
                        mapped_circuit.append_gate(SwapGate(radix), swap)
                _logger.debug('Stopping override.')
                continue

            # Pick and apply a swap
            E = self._calc_extended_set(circuit, F)
            best_swap = self._get_best_swap(circuit, F, E, D, cg, pi, decay)
            self._apply_swap(best_swap, pi, decay)
            leading_swaps.append(best_swap)

            if modify_circuit:
                mapped_circuit.append_gate(SwapGate(radix), best_swap)

            # Update loop counter and reset decay if necessary
            iter_count += 1
            if iter_count % self.decay_reset_interval == 0:
                for i in range(circuit.num_qudits):
                    decay[i] = 1.0

        if modify_circuit:
            circuit.become(mapped_circuit)

    def backward_pass(
        self,
        circuit: Circuit,
        pi: list[int],
        cg: CouplingGraph,
    ) -> None:
        """
        Apply a backward pass of the Sabre algorithm to `pi`.

        Args:
            circuit (Circuit): The circuit to pass over.

            pi (list[int]): The input logical-to-physical mapping. This
                maps logical qudits to physical qudits. So, `pi[l] == p`
                implies logical qudit `l` is sitting on physical qudit `p`.

            cg (CouplingGraph): The connectivity of the hardware.
        """
        # Preprocessing
        D = cg.all_pairs_shortest_path()
        F = circuit.rear
        decay = [1.0 for i in range(circuit.num_qudits)]
        iter_count = 0
        leading_swaps: list[tuple[int, int]] = []
        next_executed_counts: dict[CircuitPoint, int] = {n: 0 for n in F}
        _logger.debug(f'Starting backward sabre pass with pi: {pi}.')

        # Main Loop
        while len(F) > 0:

            # Retrieve executable gates giving the current mapping: pi
            execute_list = [n for n in F if self._can_exe(circuit[n], pi, cg)]

            # Execute the gates and update F
            if len(execute_list) > 0:
                leading_swaps = []

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

            # If execute list is empty, check for local-minima
            elif len(leading_swaps) > 5 * cg.num_qudits:
                _logger.debug('Sabre stuck in local minima, backtracking...')

                # Backtrack by removing leading swaps
                for swap in reversed(leading_swaps):
                    self._apply_swap(swap, pi, decay)
                leading_swaps = []

                # Override heuristic search to progress
                _logger.debug('Overriding sabre search...')
                all_logical_qudits = [circuit[n].location for n in F]
                qudits = min(
                    all_logical_qudits,
                    key=lambda qs: self._get_distance(qs, pi, D),
                )
                for swap in self._uphill_swaps(qudits, cg, pi, D):
                    self._apply_swap(swap, pi, decay)
                _logger.debug('Stopping override.')
                continue

            # Pick and apply a swap
            E = self._calc_extended_set(circuit, F)
            best_swap = self._get_best_swap(circuit, F, E, D, cg, pi, decay)
            self._apply_swap(best_swap, pi, decay)
            leading_swaps.append(best_swap)

            # Update loop counter and reset decay if necessary
            iter_count += 1
            if iter_count % self.decay_reset_interval == 0:
                for i in range(circuit.num_qudits):
                    decay[i] = 1.0

    def _can_exe(self, op: Operation, pi: list[int], cg: CouplingGraph) -> bool:
        """Return true if `op` is executable given the current mapping `pi`."""
        if isinstance(op.gate, BarrierPlaceholder):
            return True

        if isinstance(op.gate, CircuitGate):
            if all(g.num_qudits == 1 for g in op.gate._circuit.gate_set):
                return True

        if op.num_qudits == 1:
            return True

        physical_qudits = [pi[i] for i in op.location]
        if self.use_legacy_can_exe:
            return cg.get_subgraph(physical_qudits).is_fully_connected()

        return self._are_physical_qudits_connected(physical_qudits, cg)

    def _are_physical_qudits_connected(
        self,
        physical_qudits: list[int],
        cg: CouplingGraph,
    ) -> bool:
        """Return true if the induced physical subgraph is connected."""
        if len(physical_qudits) == 2:
            return physical_qudits[1] in cg._adj[physical_qudits[0]]

        physical_set = set(physical_qudits)
        frontier = {physical_qudits[0]}
        seen: set[int] = set()

        while frontier:
            qudit = frontier.pop()
            if qudit in seen:
                continue

            seen.add(qudit)
            frontier.update(cg._adj[qudit].intersection(physical_set) - seen)

        return len(seen) == len(physical_set)

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

    def _get_best_swap(
        self,
        circuit: Circuit,
        F: set[CircuitPoint],
        E: set[CircuitPoint],
        D: list[list[float]],
        cg: CouplingGraph,
        pi: list[int],
        decay: list[float],
    ) -> tuple[int, int]:
        """Return the best swap given the current algorithm state."""
        best_score = np.inf
        best_swap: tuple[int, int] | None = None
        swap_candidate_list = self._obtain_swaps(circuit, F, pi, cg)
        physical_to_logical = self._build_physical_to_logical(pi)
        heuristic_context = self._build_heuristic_context(circuit, F, E, pi, D)

        if _logger.isEnabledFor(logging.DEBUG):
            _logger.debug("Front layer F: %s", sorted(F))
            _logger.debug("Extended set E: %s", sorted(E) if E is not None else None)
            _logger.debug("Candidate swaps: %s", sorted(swap_candidate_list))

        for swap in swap_candidate_list:
            score = self._score_swap(
                pi,
                physical_to_logical,
                D,
                swap,
                decay,
                heuristic_context,
            )
            if score < best_score:
                best_score = score
                best_swap = swap
            _logger.debug("Swap candidate %s score=%f with pi=%s", swap, score, pi)

        if best_swap is None:
            raise RuntimeError('Unable to find best swap.')

        return best_swap

    def _obtain_swaps(
        self,
        circuit: Circuit,
        F: set[CircuitPoint],
        pi: list[int],
        cg: CouplingGraph,
    ) -> set[tuple[int, int]]:
        """Produce all physical swaps with at least one qudit in F."""
        all_qudits: set[int] = set()
        for n in F:
            all_qudits.update(circuit[n].location)
        physical_qudits = [pi[i] for i in all_qudits]

        swaps = set()
        for physical_qudit in physical_qudits:
            neighbors = cg.get_neighbors_of(physical_qudit)
            for neighbor in neighbors:
                a = min(neighbor, physical_qudit)
                b = max(neighbor, physical_qudit)
                swaps.add((a, b))

        return swaps

    def _score_swap(
        self,
        pi: list[int],
        physical_to_logical: list[int],
        D: list[list[float]],
        swap: tuple[int, int],
        decay: list[float],
        heuristic_context: HeuristicScoreContext,
    ) -> float:
        """Score a candidate swap by rescoring only affected gates."""
        pos1, pos2 = swap
        l1 = int(physical_to_logical[pos1])
        l2 = int(physical_to_logical[pos2])
        swapped_logicals = (l1, l2)

        pi[l1], pi[l2] = pi[l2], pi[l1]
        physical_to_logical[pos1], physical_to_logical[pos2] = (
            physical_to_logical[pos2],
            physical_to_logical[pos1],
        )

        try:
            front_total = heuristic_context.front.total_score
            affected_front = self._affected_region_points(
                heuristic_context.front,
                swapped_logicals,
            )

            for point in affected_front:
                logical_qudits = heuristic_context.front.gate_qudits[point]
                physical_qudits = [int(pi[i]) for i in logical_qudits]
                if pos1 in physical_qudits and pos2 in physical_qudits:
                    return np.inf

                front_total -= heuristic_context.front.gate_scores[point]
                front_total += self._get_distance(logical_qudits, pi, D)

            front = front_total / heuristic_context.front.num_points

            extend = 0.0
            if heuristic_context.extend.num_points > 0:
                extend_total = heuristic_context.extend.total_score
                affected_extend = self._affected_region_points(
                    heuristic_context.extend,
                    swapped_logicals,
                )
                for point in affected_extend:
                    extend_total -= heuristic_context.extend.gate_scores[point]
                    extend_total += self._get_distance(
                        heuristic_context.extend.gate_qudits[point],
                        pi,
                        D,
                    )
                extend = extend_total / heuristic_context.extend.num_points
                extend *= self.extended_set_weight

            decay_factor = max(decay[pos1], decay[pos2])

            return decay_factor * (front + extend)
        finally:
            pi[l1], pi[l2] = pi[l2], pi[l1]
            physical_to_logical[pos1], physical_to_logical[pos2] = (
                physical_to_logical[pos2],
                physical_to_logical[pos1],
            )


    def _apply_swap(
        self,
        swap: tuple[int, int],
        pi: list[int],
        decay: list[float],
    ) -> None:
        """Apply the swap to `pi` and update `decay`."""
        _logger.debug('applying swap %s' % str(swap))
        l1, l2 = pi.index(swap[0]), pi.index(swap[1])
        pi[l1], pi[l2] = pi[l2], pi[l1]

        decay[swap[0]] += self.decay_delta
        decay[swap[1]] += self.decay_delta

    def _get_distance(
        self,
        logical_qudits: Sequence[int],
        pi: list[int],
        D: list[list[float]],
    ) -> float:
        """Calculate the expected number of swaps to connect logical qudits."""
        min_term = np.inf
        for q in logical_qudits:
            term = 0.0
            for p in logical_qudits:
                if p == q:
                    continue
                term += D[pi[q]][pi[p]]
            min_term = min(term, min_term)
        return min_term

    def _uphill_swaps(
        self,
        logical_qudits: Sequence[int],
        cg: CouplingGraph,
        pi: list[int],
        D: list[list[float]],
    ) -> Iterator[tuple[int, int]]:
        """Yield the swaps necessary to bring some of the qudits together."""
        center_qudit = min(
            logical_qudits,
            key=lambda q: sum(
                D[pi[q]][pi[p]]
                for p in logical_qudits
                if p != q
            ),
        )

        for q in logical_qudits:
            if q == center_qudit:
                continue

            # TODO: Do not need to calculate entire tree
            spt = cg.get_shortest_path_tree(pi[center_qudit])
            path = list(reversed(spt[pi[q]]))

            _logger.debug(f'Moving {q} to {center_qudit} via {path}.')

            for p1, p2 in zip(path, path[1:]):
                if pi[center_qudit] == p1 or pi[center_qudit] == p2:
                    continue
                yield (p1, p2)

    def _apply_perm(self, perm: Sequence[int], pi: list[int]) -> None:
        """Apply the `perm` permutation to the current mapping `pi`."""
        _logger.debug('applying permutation %s' % str(perm))
        pi_c = {q: pi[perm[i]] for i, q in enumerate(sorted(perm))}
        for q in perm:
            pi[q] = pi_c[q]
