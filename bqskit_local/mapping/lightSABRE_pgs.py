"""LightSABRE-style PGS routing and layout helpers."""
from __future__ import annotations

from itertools import combinations
import logging
import random
from typing import Sequence

from bqskit.ir.circuit import Circuit
from bqskit.ir.gates.constant.swap import SwapGate
from bqskit.ir.point import CircuitPoint

from bqskit_local.mapping.sabre_pgs_behavioral_equivalence import (
    GeneralizedSabreAlgorithmPGS,
    HeuristicScoreContext,
    PositionState,
)
from bqskit_local.position.state import PositionGraphState

_logger = logging.getLogger(__name__)


class GeneralizedLightSABREAlgorithmPGS(GeneralizedSabreAlgorithmPGS):
    """
    LightSABRE-inspired PGS algorithm.

    This starts from the behavioral-equivalence SABRE implementation and
    layers on:
    - seeded stochastic tie-breaking,
    - multiple heuristic modes,
    - a paper-style release valve that backtracks to the last routed gate,
      then follows a shortest path from both ends until the qudits meet.
    """

    def __init__(
        self,
        decay_delta: float = 0.001,
        decay_reset_interval: int = 5,
        decay_reset_on_gate: bool = True,
        extended_set_size: int = 20,
        extended_set_weight: float = 0.5,
        cg_compatibility_mode: bool = False,
        heuristic: str = 'decay',
        seed: int | None = None,
        attempt_limit: int | None = None,
    ) -> None:
        super().__init__(
            decay_delta=decay_delta,
            decay_reset_interval=decay_reset_interval,
            decay_reset_on_gate=decay_reset_on_gate,
            extended_set_size=extended_set_size,
            extended_set_weight=extended_set_weight,
            cg_compatibility_mode=cg_compatibility_mode,
        )

        valid_heuristics = {
            'basic',
            'lookahead',
            'decay',
            'depth',
            'critical_path',
        }
        components = self._parse_heuristic_components(heuristic, valid_heuristics)
        if not components:
            raise ValueError(
                f'Expected heuristic components drawn from {sorted(valid_heuristics)}, got {heuristic}.',
            )

        if seed is not None and not isinstance(seed, int):
            raise TypeError(f'Expected int or None for seed, got {type(seed)}.')

        if attempt_limit is not None:
            if not isinstance(attempt_limit, int):
                raise TypeError(
                    f'Expected int or None for attempt_limit, got {type(attempt_limit)}.',
                )
            if attempt_limit < 1:
                raise ValueError('attempt_limit must be positive when specified.')

        self.heuristic = heuristic
        self.heuristic_components = components
        self.seed = seed
        self.attempt_limit = attempt_limit
        self._rng = random.Random(seed)

        # Paper-inspired heuristic constants.
        self.depth_weight = 1.0
        self.critical_alpha = 0.5

    def _parse_heuristic_components(
        self,
        heuristic: str,
        valid_heuristics: set[str],
    ) -> set[str]:
        """Parse a heuristic string like 'lookahead+decay+depth'."""
        if not isinstance(heuristic, str):
            raise TypeError(f'Expected str for heuristic, got {type(heuristic)}.')

        tokens = [
            token.strip()
            for token in heuristic.replace(',', '+').split('+')
            if token.strip()
        ]
        if not tokens:
            return set()

        unknown = [token for token in tokens if token not in valid_heuristics]
        if unknown:
            raise ValueError(
                f'Unknown heuristic component(s): {unknown}. '
                f'Expected components from {sorted(valid_heuristics)}.',
            )

        components = set(tokens)

        if 'basic' in components and 'lookahead' in components:
            raise ValueError('Heuristic cannot include both basic and lookahead.')

        # Preserve the old single-word semantics:
        # decay, depth, and critical_path all previously included lookahead.
        if 'basic' not in components and (
            'lookahead' in components
            or 'decay' in components
            or 'depth' in components
            or 'critical_path' in components
        ):
            components.add('lookahead')

        return components

    def _uses_lookahead(self) -> bool:
        return 'lookahead' in self.heuristic_components and 'basic' not in self.heuristic_components

    def _uses_decay(self) -> bool:
        return 'decay' in self.heuristic_components

    def _uses_depth(self) -> bool:
        return 'depth' in self.heuristic_components

    def _uses_critical_path(self) -> bool:
        return 'critical_path' in self.heuristic_components

    def begin_trial(self, trial_index: int = 0) -> None:
        """Reset trial-local randomness for reproducible tie-breaking."""
        if self.seed is None:
            self._rng = random.Random()
        else:
            self._rng = random.Random(self.seed + int(trial_index))

    def routed_swap_count(self, circuit: Circuit) -> int:
        """Count inserted swap operations in a routed circuit."""
        return sum(1 for op in circuit if isinstance(op.gate, SwapGate))

    def routed_trial_score(self, circuit: Circuit) -> tuple[int, int, int]:
        """Score a fully routed circuit with LightSABRE-style priority."""
        swap_count = self.routed_swap_count(circuit)
        depth = self._circuit_depth(circuit)
        return (swap_count, depth, circuit.num_operations)

    def _local_minimum_limit(self, num_qudits: int) -> int:
        if self.attempt_limit is not None:
            return self.attempt_limit
        return 10 * num_qudits

    def _circuit_depth(self, circuit: Circuit) -> int:
        """Estimate circuit depth by ASAP layering over qudit usage."""
        if circuit.num_qudits == 0:
            return 0

        qudit_depth = [0] * circuit.num_qudits
        max_depth = 0

        for op in circuit:
            if len(op.location) == 0:
                continue
            layer = 1 + max(qudit_depth[int(q)] for q in op.location)
            for qudit in op.location:
                qudit_depth[int(qudit)] = layer
            max_depth = max(max_depth, layer)

        return max_depth

    def _compute_gate_depths(self, circuit: Circuit) -> dict[CircuitPoint, int]:
        """Compute an ASAP depth index for every gate in the DAG."""
        frontier = set(circuit.front)
        prev_executed_counts: dict[CircuitPoint, int] = {
            point: 0
            for point in frontier
        }
        gate_depths: dict[CircuitPoint, int] = {}

        while frontier:
            point = min(frontier)
            frontier.remove(point)
            predecessors = circuit.prev(point)
            if len(predecessors) == 0:
                gate_depths[point] = 0
            else:
                gate_depths[point] = 1 + max(gate_depths[p] for p in predecessors)

            for successor in circuit.next(point):
                if successor not in prev_executed_counts:
                    prev_executed_counts[successor] = 1
                else:
                    prev_executed_counts[successor] += 1

                if prev_executed_counts[successor] == len(circuit.prev(successor)):
                    frontier.add(successor)

        return gate_depths

    def _compute_critical_path_ranks(self, circuit: Circuit) -> dict[CircuitPoint, int]:
        """
        Rank gates by remaining critical-path length.

        Rank 1 is the most critical gate.
        """
        gate_depths = self._compute_gate_depths(circuit)
        topo_order = sorted(gate_depths, key=lambda point: (gate_depths[point], point))
        remaining_depth: dict[CircuitPoint, int] = {}

        for point in reversed(topo_order):
            successors = circuit.next(point)
            if len(successors) == 0:
                remaining_depth[point] = 1
            else:
                remaining_depth[point] = 1 + max(
                    remaining_depth[successor]
                    for successor in successors
                )

        ranked_points = sorted(
            remaining_depth,
            key=lambda point: (-remaining_depth[point], point),
        )
        return {
            point: rank
            for rank, point in enumerate(ranked_points, start=1)
        }

    def _estimate_swap_depth_delta(
        self,
        swap: tuple[int, int],
        position_depths: Sequence[int] | None,
    ) -> float:
        """Estimate the additional depth incurred by appending this swap now."""
        if position_depths is None:
            return 0.0

        pos1, pos2 = swap
        current_max = max(position_depths, default=0)
        new_layer = 1 + max(position_depths[pos1], position_depths[pos2])
        return float(max(0, new_layer - current_max))

    def _critical_path_bonus(
        self,
        affected_front: Sequence[CircuitPoint],
        old_front_scores: dict[CircuitPoint, float],
        new_front_scores: dict[CircuitPoint, float],
        critical_ranks: dict[CircuitPoint, int] | None,
    ) -> float:
        """Reward swaps that improve the more critical front-layer gates."""
        if critical_ranks is None:
            return 0.0

        bonus = 0.0
        for point in affected_front:
            old_score = old_front_scores[point]
            new_score = new_front_scores[point]
            if new_score >= old_score:
                continue

            rank = critical_ranks.get(point, len(critical_ranks) + 1)
            weight = self.critical_alpha ** max(rank - 1, 0)
            bonus += weight * float(old_score - new_score)

        return bonus

    def _score_swap(
        self,
        pgs: PositionState,
        D: list[list[float]],
        swap: tuple[int, int],
        decay: list[float],
        heuristic_context: HeuristicScoreContext,
        position_depths: Sequence[int] | None = None,
        critical_ranks: dict[CircuitPoint, int] | None = None,
    ) -> float:
        """Score a candidate swap using the selected heuristic mode."""
        pos1, pos2 = swap
        l1 = int(pgs.position_to_logical[pos1])
        l2 = int(pgs.position_to_logical[pos2])
        swapped_logicals = (l1, l2)

        self._apply_swap_to_state(swap, pgs)
        try:
            mapping = pgs.logical_to_position
            front_total = heuristic_context.front.total_score
            affected_front = self._affected_region_points(
                heuristic_context.front,
                swapped_logicals,
            )
            use_critical_path = self._uses_critical_path()
            old_front_scores = (
                {
                    point: heuristic_context.front.gate_scores[point]
                    for point in affected_front
                }
                if use_critical_path else
                None
            )
            new_front_scores = {} if use_critical_path else None

            for point in affected_front:
                logical_qudits = heuristic_context.front.gate_qudits[point]
                physical_qudits = [int(mapping[i]) for i in logical_qudits]
                if pos1 in physical_qudits and pos2 in physical_qudits:
                    return float('inf')

                front_total -= heuristic_context.front.gate_scores[point]
                new_score = self._get_distance_from_mapping(
                    logical_qudits,
                    mapping,
                    D,
                )
                if new_front_scores is not None:
                    new_front_scores[point] = new_score
                front_total += new_score

            front = front_total / heuristic_context.front.num_points

            extend = 0.0
            if self._uses_lookahead() and heuristic_context.extend.num_points > 0:
                extend_total = heuristic_context.extend.total_score
                affected_extend = self._affected_region_points(
                    heuristic_context.extend,
                    swapped_logicals,
                )
                for point in affected_extend:
                    extend_total -= heuristic_context.extend.gate_scores[point]
                    extend_total += self._get_distance_from_mapping(
                        heuristic_context.extend.gate_qudits[point],
                        mapping,
                        D,
                    )
                extend = extend_total / heuristic_context.extend.num_points
                extend *= self.extended_set_weight

            base_score = front + extend

            if self._uses_decay():
                decay_factor = max(decay[pos1], decay[pos2])
                base_score = decay_factor * base_score

            if self._uses_depth():
                depth_delta = self._estimate_swap_depth_delta(swap, position_depths)
                base_score += self.depth_weight * depth_delta / 3.0

            if use_critical_path:
                critical_bonus = self._critical_path_bonus(
                    affected_front,
                    old_front_scores if old_front_scores is not None else {},
                    new_front_scores if new_front_scores is not None else {},
                    critical_ranks,
                )
                base_score -= critical_bonus

            return base_score
        finally:
            self._apply_swap_to_state(swap, pgs)

    def _get_best_swap(
        self,
        circuit: Circuit,
        F: set[CircuitPoint],
        E: set[CircuitPoint],
        D: list[list[float]],
        pgs: PositionGraphState,
        decay: list[float],
        position_depths: Sequence[int] | None = None,
        critical_ranks: dict[CircuitPoint, int] | None = None,
    ) -> tuple[int, int]:
        """Return the best swap with seeded random tie-breaking."""
        best_score = float('inf')
        best_swaps: list[tuple[int, int]] = []
        tolerance = 1e-12

        swap_candidate_list = self._obtain_swaps(circuit, F, pgs)
        scratch_pgs = self._scratch_pgs(pgs, 'score_swap')
        heuristic_context = self._build_heuristic_context(circuit, F, E, pgs, D)
        candidate_order = (
            list(swap_candidate_list)
            if self.cg_compatibility_mode
            else sorted(swap_candidate_list)
        )

        for swap in candidate_order:
            score = self._score_swap(
                scratch_pgs,
                D,
                swap,
                decay,
                heuristic_context,
                position_depths=position_depths,
                critical_ranks=critical_ranks,
            )
            if score + tolerance < best_score:
                best_score = score
                best_swaps = [swap]
            elif abs(score - best_score) <= tolerance:
                best_swaps.append(swap)

        if not best_swaps:
            raise RuntimeError('Unable to find best swap.')

        if self.cg_compatibility_mode or len(best_swaps) == 1:
            return best_swaps[0]

        return self._rng.choice(best_swaps)

    def _select_release_gate(
        self,
        circuit: Circuit,
        F: set[CircuitPoint],
        pgs: PositionState,
        D: list[list[float]],
    ) -> CircuitPoint:
        """Choose the front-layer gate for the release valve to route."""
        return min(
            F,
            key=lambda point: (
                self._get_distance(circuit[point].location, pgs, D),
                point,
            ),
        )

    def _select_release_pair(
        self,
        logical_qudits: Sequence[int],
        pgs: PositionState,
        D: list[list[float]],
    ) -> tuple[int, int] | None:
        """
        Choose the pair to route with the release valve.

        The paper is 2-qubit specific. For larger BQSKit gates, use the
        closest pair inside the selected gate.
        """
        logicals = [int(qudit) for qudit in logical_qudits]
        if len(logicals) < 2:
            return None
        if len(logicals) == 2:
            return (logicals[0], logicals[1])

        best_pair: tuple[int, int] | None = None
        best_distance = float('inf')
        for left, right in combinations(logicals, 2):
            left_pos = int(pgs.logical_to_position[left])
            right_pos = int(pgs.logical_to_position[right])
            distance = float(D[left_pos][right_pos])
            if distance < best_distance:
                best_distance = distance
                best_pair = (left, right)

        return best_pair

    def _meet_in_middle_release_swaps(
        self,
        path: Sequence[int],
    ) -> list[tuple[int, int]]:
        """Generate the paper's meet-in-the-middle release swap sequence."""
        swaps: list[tuple[int, int]] = []
        left = 0
        right = len(path) - 1

        while right - left > 1:
            swaps.append((int(path[left]), int(path[left + 1])))
            left += 1
            if right - left > 1:
                swaps.append((int(path[right - 1]), int(path[right])))
                right -= 1

        return swaps

    def _paper_release_swaps(
        self,
        circuit: Circuit,
        F: set[CircuitPoint],
        pgs: PositionState,
        D: list[list[float]],
    ) -> list[tuple[int, int]]:
        """Build the release-valve swaps described in the LightSABRE paper."""
        release_gate = self._select_release_gate(circuit, F, pgs, D)
        logical_pair = self._select_release_pair(circuit[release_gate].location, pgs, D)
        if logical_pair is None:
            return []

        left_logical, right_logical = logical_pair
        left_pos = int(pgs.logical_to_position[left_logical])
        right_pos = int(pgs.logical_to_position[right_logical])
        shortest_path = pgs.position_graph.shortest_path(left_pos, right_pos)
        if shortest_path is None:
            return []

        path, _ = shortest_path
        if len(path) < 2:
            return []

        return self._meet_in_middle_release_swaps(path)

    def _record_operation_depth(
        self,
        positions: Sequence[int],
        position_depths: list[int] | None,
    ) -> None:
        """Update the running depth estimate for a scheduled operation."""
        if position_depths is None or len(positions) == 0:
            return

        layer = 1 + max(position_depths[int(position)] for position in positions)
        for position in positions:
            position_depths[int(position)] = layer

    def _backtrack_mapped_swap(
        self,
        mapped_circuit: Circuit,
        swap: tuple[int, int],
    ) -> None:
        """Remove the most recently appended swap from the mapped circuit."""
        point = mapped_circuit._rear[swap[0]]
        mapped_circuit.pop(point)

    def forward_pass(
        self,
        circuit: Circuit,
        pgs,
        modify_circuit: bool = False,
    ) -> None:
        """Forward pass owned by the LightSABRE implementation."""
        D = pgs.position_graph.move_cost_matrix
        F = circuit.front
        decay = [1.0 for _ in range(pgs.num_qudits)]
        iter_count = 0
        prev_executed_counts: dict[CircuitPoint, int] = {n: 0 for n in F}
        leading_swaps: list[tuple[int, int]] = []
        position_depths = [0] * pgs.num_pos if self._uses_depth() else None
        depth_snapshot = (
            position_depths.copy()
            if position_depths is not None else
            None
        )
        critical_ranks = (
            self._compute_critical_path_ranks(circuit)
            if self._uses_critical_path() else
            None
        )

        _logger.debug('Starting LightSABRE forward pass with pgs: %s.', pgs)

        radix = circuit.radixes[0]
        mapped_circuit = Circuit(circuit.num_qudits, circuit.radixes) if modify_circuit else None

        while len(F) > 0:
            execute_list = [n for n in F if self._can_exe(circuit[n], pgs)]

            if len(execute_list) > 0:
                leading_swaps = []
                for n in execute_list:
                    F.remove(n)
                    prev_executed_counts.pop(n)
                    _logger.debug('Executing gate at point %s.', n)

                    op = circuit[n]
                    op_positions = [int(pgs.logical_to_position[q]) for q in op.location]
                    self._record_operation_depth(op_positions, position_depths)
                    if mapped_circuit is not None:
                        mapped_circuit.append_gate(op.gate, op_positions, op.params)

                    for successor in circuit.next(n):
                        if successor not in prev_executed_counts:
                            prev_executed_counts[successor] = 1
                        else:
                            prev_executed_counts[successor] += 1
                        num_prev_executed = prev_executed_counts[successor]
                        total_num_prev = len(circuit.prev(successor))
                        if num_prev_executed == total_num_prev:
                            F.add(successor)

                if position_depths is not None:
                    depth_snapshot = position_depths.copy()

                if self.decay_reset_on_gate:
                    iter_count = 0
                    for i in range(circuit.num_qudits):
                        decay[i] = 1.0
                continue

            if len(leading_swaps) >= self._local_minimum_limit(pgs.num_qudits):
                _logger.debug('LightSABRE release valve triggered, backtracking.')
                for swap in reversed(leading_swaps):
                    self._apply_swap(swap, pgs, decay)
                    if mapped_circuit is not None:
                        self._backtrack_mapped_swap(mapped_circuit, swap)

                if position_depths is not None and depth_snapshot is not None:
                    position_depths[:] = depth_snapshot

                leading_swaps = self._paper_release_swaps(circuit, F, pgs, D)
                _logger.debug('LightSABRE release swaps: %s', leading_swaps)
                for swap in leading_swaps:
                    self._apply_swap(
                        swap,
                        pgs,
                        decay,
                        position_depths=position_depths,
                    )
                    if mapped_circuit is not None:
                        mapped_circuit.append_gate(SwapGate(radix), swap)
                continue

            E = self._calc_extended_set(circuit, F)
            best_swap = self._get_best_swap(
                circuit,
                F,
                E,
                D,
                pgs,
                decay,
                position_depths=position_depths,
                critical_ranks=critical_ranks,
            )
            self._apply_swap(
                best_swap,
                pgs,
                decay,
                position_depths=position_depths,
            )
            leading_swaps.append(best_swap)

            if mapped_circuit is not None:
                mapped_circuit.append_gate(SwapGate(radix), best_swap)

            iter_count += 1
            if iter_count % self.decay_reset_interval == 0:
                for i in range(circuit.num_qudits):
                    decay[i] = 1.0

        if mapped_circuit is not None:
            circuit.become(mapped_circuit)

    def backward_pass(
        self,
        circuit: Circuit,
        pgs,
    ) -> None:
        """Backward pass owned by the LightSABRE implementation."""
        D = pgs.position_graph.move_cost_matrix
        F = circuit.rear
        decay = [1.0 for _ in range(pgs.num_qudits)]
        iter_count = 0
        leading_swaps: list[tuple[int, int]] = []
        next_executed_counts: dict[CircuitPoint, int] = {n: 0 for n in F}
        position_depths = [0] * pgs.num_pos if self._uses_depth() else None
        depth_snapshot = (
            position_depths.copy()
            if position_depths is not None else
            None
        )
        critical_ranks = (
            self._compute_critical_path_ranks(circuit)
            if self._uses_critical_path() else
            None
        )

        _logger.debug('Starting LightSABRE backward pass with pgs: %s.', pgs)

        while len(F) > 0:
            execute_list = [n for n in F if self._can_exe(circuit[n], pgs)]

            if len(execute_list) > 0:
                leading_swaps = []
                for n in execute_list:
                    F.remove(n)
                    next_executed_counts.pop(n)
                    _logger.debug('Executing backward gate at point %s.', n)

                    op = circuit[n]
                    op_positions = [int(pgs.logical_to_position[q]) for q in op.location]
                    self._record_operation_depth(op_positions, position_depths)

                    for predecessor in circuit.prev(n):
                        if predecessor not in next_executed_counts:
                            next_executed_counts[predecessor] = 1
                        else:
                            next_executed_counts[predecessor] += 1

                        num_next_executed = next_executed_counts[predecessor]
                        total_num_next = len(circuit.next(predecessor))
                        if num_next_executed == total_num_next:
                            F.add(predecessor)

                if position_depths is not None:
                    depth_snapshot = position_depths.copy()

                if self.decay_reset_on_gate:
                    iter_count = 0
                    for i in range(circuit.num_qudits):
                        decay[i] = 1.0
                continue

            if len(leading_swaps) >= self._local_minimum_limit(pgs.num_qudits):
                _logger.debug('LightSABRE backward release valve triggered, backtracking.')
                for swap in reversed(leading_swaps):
                    self._apply_swap(swap, pgs, decay)

                if position_depths is not None and depth_snapshot is not None:
                    position_depths[:] = depth_snapshot

                leading_swaps = self._paper_release_swaps(circuit, F, pgs, D)
                _logger.debug('LightSABRE backward release swaps: %s', leading_swaps)
                for swap in leading_swaps:
                    self._apply_swap(
                        swap,
                        pgs,
                        decay,
                        position_depths=position_depths,
                    )
                continue

            E = self._calc_extended_set(circuit, F)
            best_swap = self._get_best_swap(
                circuit,
                F,
                E,
                D,
                pgs,
                decay,
                position_depths=position_depths,
                critical_ranks=critical_ranks,
            )
            self._apply_swap(
                best_swap,
                pgs,
                decay,
                position_depths=position_depths,
            )
            leading_swaps.append(best_swap)

            iter_count += 1
            if iter_count % self.decay_reset_interval == 0:
                for i in range(circuit.num_qudits):
                    decay[i] = 1.0

    def _apply_swap(
        self,
        swap: tuple[int, int],
        pgs: PositionGraphState,
        decay: list[float] | None = None,
        position_depths: list[int] | None = None,
    ) -> None:
        """Apply the swap to `pgs` and update `decay`."""
        pos1, pos2 = swap
        l1 = int(pgs.position_to_logical[pos1])
        l2 = int(pgs.position_to_logical[pos2])

        _logger.debug('Applying swap physical%s logical(%d,%d)', swap, l1, l2)

        self._apply_swap_to_state(swap, pgs)

        if decay is not None:
            decay[pos1] += self.decay_delta
            decay[pos2] += self.decay_delta

        self._record_operation_depth((pos1, pos2), position_depths)
