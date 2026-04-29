"""Cached LightSABRE-style PGS routing and layout helpers."""
from __future__ import annotations

from typing import Sequence

from bqskit.ir.circuit import Circuit
from bqskit.ir.point import CircuitPoint

from bqskit_local.mapping.lightSABRE_pgs import DEFAULT_LIGHTSABRE_HEURISTIC
from bqskit_local.mapping.lightSABRE_pgs import GeneralizedLightSABREAlgorithmPGS
from bqskit_local.mapping.sabre_pgs_behavioral_equivalence import (
    HeuristicScoreContext,
    PositionState,
)
from bqskit_local.position.state import PositionGraphState


class GeneralizedCachedLightSABREAlgorithmPGS(GeneralizedLightSABREAlgorithmPGS):
    """LightSABRE variant that preserves heuristic-region caching."""

    def __init__(
        self,
        decay_delta: float = 0.001,
        decay_reset_interval: int = 5,
        decay_reset_on_gate: bool = True,
        extended_set_size: int = 20,
        extended_set_weight: float = 0.5,
        cg_compatibility_mode: bool = False,
        heuristic: str = DEFAULT_LIGHTSABRE_HEURISTIC,
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
            heuristic=heuristic,
            seed=seed,
            attempt_limit=attempt_limit,
        )

    def _score_swap(
        self,
        circuit: Circuit,
        F: set[CircuitPoint],
        E: set[CircuitPoint],
        pgs: PositionState,
        D: list[list[float]],
        swap: tuple[int, int],
        decay: list[float],
        heuristic_context: HeuristicScoreContext,
        position_depths: Sequence[int] | None = None,
        critical_ranks: dict[CircuitPoint, int] | None = None,
    ) -> float:
        """Score a candidate swap using cached frontier and extended-set data."""
        del circuit, F, E
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

            front = front_total

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
                circuit,
                F,
                E,
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
