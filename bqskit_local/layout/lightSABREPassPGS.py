from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Sequence

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit

from bqskit_local.mapping.lightSABRE_pgs import DEFAULT_LIGHTSABRE_HEURISTIC
from bqskit_local.mapping.lightSABRE_pgs import GeneralizedLightSABREAlgorithmPGS
from bqskit_local.position.graph import PositionGraph
from bqskit_local.position.state import PositionGraphState

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LabeledLayoutCandidate:
    label: str
    mapping: list[int]


class LightSABRELayoutCandidateHelper:
    def __init__(
        self,
        num_circuit_qudits: int,
        num_positions: int,
        layout_trials: int,
    ) -> None:
        self.num_circuit_qudits = num_circuit_qudits
        self.num_positions = num_positions
        self.layout_trials = layout_trials
        self._seen: set[tuple[int, ...]] = set()
        self._candidates: list[LabeledLayoutCandidate] = []

    def add(self, label: str, mapping: Sequence[int]) -> bool:
        if len(self._candidates) >= self.layout_trials:
            return False

        normalized = [int(position) for position in mapping[:self.num_circuit_qudits]]
        if len(normalized) != self.num_circuit_qudits:
            return False
        if len(set(normalized)) != len(normalized):
            return False
        if any(position < 0 or position >= self.num_positions for position in normalized):
            return False

        key = tuple(normalized)
        if key in self._seen:
            return False

        self._seen.add(key)
        self._candidates.append(LabeledLayoutCandidate(label=label, mapping=normalized))
        return True

    @property
    def candidates(self) -> list[LabeledLayoutCandidate]:
        return list(self._candidates)


class GeneralizedLightSABRELayoutPassPGS(BasePass, GeneralizedLightSABREAlgorithmPGS):
    """LightSABRE-style PGS layout search with multiple layout trials."""

    TRAFFIC_AWARE_SEED_NAME = 'traffic_aware_seed'

    def __init__(
        self,
        template_pgs: PositionGraphState,
        max_iterations: int = 3,
        swap_trials: int = 1,
        layout_trials: int = 5,
        decay_delta: float = 0.001,
        decay_reset_interval: int = 5,
        decay_reset_on_gate: bool = True,
        extended_set_size: int = 20,
        extended_set_weight: float = 0.5,
        cg_compatibility_mode: bool = False,
        heuristic: str = DEFAULT_LIGHTSABRE_HEURISTIC,
        seed: int | None = None,
        attempt_limit: int | None = None,
        starting_layouts: Sequence[Sequence[int]] | None = None,
        use_traffic_aware_seed: bool = True,
    ) -> None:
        if not isinstance(template_pgs, PositionGraphState):
            raise TypeError(
                f'Expected PositionGraphState, got {type(template_pgs)}.',
            )

        for name, value in (
            ('max_iterations', max_iterations),
            ('swap_trials', swap_trials),
            ('layout_trials', layout_trials),
        ):
            if not isinstance(value, int):
                raise TypeError(f'Expected int for {name}, got {type(value)}.')
            if value < 1:
                raise ValueError(f'{name} must be a positive integer.')

        self.template_pgs = template_pgs.copy()
        self.max_iterations = max_iterations
        self.swap_trials = swap_trials
        self.layout_trials = layout_trials
        self.use_traffic_aware_seed = bool(use_traffic_aware_seed)
        self.starting_layouts = (
            [
                [int(position) for position in layout]
                for layout in starting_layouts
            ]
            if starting_layouts is not None else
            []
        )

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

    def _build_pgs_from_mapping(
        self,
        mapping: Sequence[int],
        num_circuit_qudits: int,
    ) -> PositionGraphState:
        pgs = self.template_pgs.copy()
        pgs.clear_assignments()

        if len(mapping) != num_circuit_qudits:
            raise ValueError(
                f'Expected mapping of length {num_circuit_qudits}, got {len(mapping)}.',
            )

        if len(set(mapping)) != len(mapping):
            raise ValueError('Mapping must assign distinct positions.')

        for logical, pos in enumerate(mapping):
            pos = int(pos)
            if pos < 0 or pos >= pgs.num_pos:
                raise ValueError(f'Invalid position {pos} for logical {logical}.')
            pgs.set_qudit_position(logical, pos)

        return pgs

    def _rank_positions(self, reverse: bool = False) -> list[int]:
        base_pg = self.template_pgs.position_graph
        valid_positions = base_pg.get_valid_starting_positions()
        if len(valid_positions) < self.template_pgs.num_qudits:
            valid_positions = list(range(base_pg.graph.num_nodes()))

        ordered = sorted(
            valid_positions,
            key=lambda pos: (
                float(sum(base_pg.move_cost_matrix[pos][other] for other in valid_positions)),
                int(pos),
            ),
        )
        if reverse:
            ordered.reverse()
        return ordered

    def _circuit_metrics(
        self,
        circuit: Circuit,
    ) -> tuple[list[float], list[float], dict[tuple[int, int], float]]:
        num_qudits = circuit.num_qudits
        activity = [0.0] * num_qudits
        interaction_load = [0.0] * num_qudits
        pair_weight: dict[tuple[int, int], float] = {}
        total_ops = max(1, circuit.num_operations)

        for op_index, op in enumerate(circuit):
            weight = 2.0 - (op_index / total_ops)
            logicals = [int(qudit) for qudit in op.location]
            op_size = max(1, op.num_qudits)

            for logical in logicals:
                activity[logical] += weight * op_size

            if len(logicals) < 2:
                continue

            for left_index in range(len(logicals)):
                for right_index in range(left_index + 1, len(logicals)):
                    left = logicals[left_index]
                    right = logicals[right_index]
                    key = (left, right) if left < right else (right, left)
                    pair_weight[key] = pair_weight.get(key, 0.0) + weight
                    interaction_load[left] += weight
                    interaction_load[right] += weight

        return activity, interaction_load, pair_weight

    @staticmethod
    def _pair_score(
        pair_weight: dict[tuple[int, int], float],
        left: int,
        right: int,
    ) -> float:
        key = (left, right) if left < right else (right, left)
        return pair_weight.get(key, 0.0)

    def _interaction_seed_mapping(
        self,
        circuit: Circuit,
        reverse_positions: bool = False,
    ) -> list[int]:
        num_qudits = circuit.num_qudits
        base_pg = self.template_pgs.position_graph
        position_order = self._rank_positions(reverse=reverse_positions)
        position_rank = {int(pos): index for index, pos in enumerate(position_order)}
        activity, _, pair_weight = self._circuit_metrics(circuit)

        mapping = [-1] * num_qudits
        available_positions = position_order.copy()
        assigned_logicals: set[int] = set()

        while len(assigned_logicals) < num_qudits and available_positions:
            unassigned = [logical for logical in range(num_qudits) if logical not in assigned_logicals]
            logical = max(
                unassigned,
                key=lambda candidate: (
                    max((
                        self._pair_score(pair_weight, candidate, other)
                        for other in assigned_logicals
                    ), default=0.0),
                    activity[candidate],
                    -candidate,
                ),
            )

            if not assigned_logicals:
                chosen_position = available_positions.pop(0)
            else:
                best_partner = max(
                    assigned_logicals,
                    key=lambda other: (
                        self._pair_score(pair_weight, logical, other),
                        activity[other],
                        -other,
                    ),
                )
                partner_position = mapping[best_partner]
                if self._pair_score(pair_weight, logical, best_partner) > 0.0:
                    chosen_position = min(
                        available_positions,
                        key=lambda pos: (
                            float(base_pg.move_cost_matrix[pos][partner_position]),
                            position_rank[int(pos)],
                            int(pos),
                        ),
                    )
                    available_positions.remove(chosen_position)
                else:
                    chosen_position = available_positions.pop(0)

            mapping[logical] = int(chosen_position)
            assigned_logicals.add(logical)

        if any(position < 0 for position in mapping):
            raise RuntimeError('Interaction seed mapping failed to place every logical qudit.')

        return mapping

    def _activity_seed_mapping(
        self,
        circuit: Circuit,
        reverse_positions: bool = False,
    ) -> list[int]:
        num_qudits = circuit.num_qudits
        position_order = self._rank_positions(reverse=reverse_positions)
        activity, _, _ = self._circuit_metrics(circuit)

        logical_order = sorted(
            range(num_qudits),
            key=lambda logical: (activity[logical], -logical),
            reverse=True,
        )

        mapping = [-1] * num_qudits
        for logical, position in zip(logical_order, position_order):
            mapping[int(logical)] = int(position)

        if any(position < 0 for position in mapping):
            raise RuntimeError('Activity seed mapping failed to place every logical qudit.')

        return mapping

    def _traffic_aware_seed_mapping(
        self,
        circuit: Circuit,
        reverse_positions: bool = False,
        anchor_rotation: int = 0,
    ) -> list[int]:
        num_qudits = circuit.num_qudits
        base_pg = self.template_pgs.position_graph
        position_order = self._rank_positions(reverse=reverse_positions)
        position_rank = {int(pos): index for index, pos in enumerate(position_order)}
        activity, interaction_load, pair_weight = self._circuit_metrics(circuit)

        logical_order = sorted(
            range(num_qudits),
            key=lambda logical: (
                activity[logical],
                interaction_load[logical],
                -logical,
            ),
            reverse=True,
        )

        max_activity = max(activity, default=0.0)
        hub_threshold = max_activity * 0.75
        max_hubs = max(1, min(8, int(num_qudits ** 0.5)))
        hub_logicals = [
            logical
            for logical in logical_order
            if activity[logical] >= hub_threshold and interaction_load[logical] > 0.0
        ][:max_hubs]
        if not hub_logicals and logical_order:
            hub_logicals = [logical_order[0]]

        anchor_pool_size = min(
            len(position_order),
            max(len(hub_logicals) * 6, 12),
        )
        anchor_pool = position_order[:anchor_pool_size]
        chosen_anchors: list[int] = []

        for _ in hub_logicals:
            if not chosen_anchors:
                chosen_anchors.append(int(anchor_pool[0]))
                continue

            remaining_positions = [
                int(pos) for pos in anchor_pool
                if int(pos) not in chosen_anchors
            ]
            if not remaining_positions:
                break

            anchor = max(
                remaining_positions,
                key=lambda pos: (
                    min(
                        float(base_pg.move_cost_matrix[pos][other])
                        for other in chosen_anchors
                    ),
                    -position_rank[pos],
                    -pos,
                ),
            )
            chosen_anchors.append(anchor)

        if chosen_anchors:
            rotation = anchor_rotation % len(chosen_anchors)
            if rotation:
                chosen_anchors = chosen_anchors[rotation:] + chosen_anchors[:rotation]

        mapping = [-1] * num_qudits
        available_positions = [int(pos) for pos in position_order]
        assigned_logicals: set[int] = set()

        for logical, position in zip(hub_logicals, chosen_anchors):
            mapping[logical] = int(position)
            assigned_logicals.add(logical)
            available_positions.remove(int(position))

        while len(assigned_logicals) < num_qudits and available_positions:
            unassigned = [
                logical for logical in logical_order
                if logical not in assigned_logicals
            ]
            logical = max(
                unassigned,
                key=lambda candidate: (
                    sum(
                        self._pair_score(pair_weight, candidate, other)
                        for other in assigned_logicals
                    ),
                    activity[candidate],
                    interaction_load[candidate],
                    -candidate,
                ),
            )

            weighted_candidates = [
                (
                    other,
                    self._pair_score(pair_weight, logical, other),
                )
                for other in assigned_logicals
            ]
            weighted_candidates = [
                (other, weight)
                for other, weight in weighted_candidates
                if weight > 0.0
            ]

            if not weighted_candidates:
                chosen_position = available_positions.pop(0)
            else:
                chosen_position = min(
                    available_positions,
                    key=lambda pos: (
                        float(sum(
                            weight * base_pg.move_cost_matrix[int(pos)][mapping[other]]
                            for other, weight in weighted_candidates
                        )),
                        position_rank[int(pos)],
                        int(pos),
                    ),
                )
                available_positions.remove(int(chosen_position))

            mapping[logical] = int(chosen_position)
            assigned_logicals.add(logical)

        if any(position < 0 for position in mapping):
            raise RuntimeError(
                f'{self.TRAFFIC_AWARE_SEED_NAME} failed to place every logical qudit.',
            )

        return mapping

    def _build_eval_pgs(
        self,
        placement: Sequence[int],
        num_circuit_qudits: int,
    ) -> PositionGraphState:
        if not self.cg_compatibility_mode:
            return self._build_pgs_from_mapping(placement, num_circuit_qudits)

        base_pg = self.template_pgs.position_graph
        inverse_placement = {int(pos): i for i, pos in enumerate(placement)}
        local_pos_labels = [base_pg.position_labels[int(pos)] for pos in placement]
        local_edge_labels = {
            (inverse_placement[u], inverse_placement[v]): label
            for (u, v), label in base_pg.edge_labels.items()
            if u in inverse_placement and v in inverse_placement
        }

        local_pg = PositionGraph(local_pos_labels, local_edge_labels)
        pgs = PositionGraphState(
            local_pg,
            radices=list(self.template_pgs.radices[:num_circuit_qudits]),
            gateSet=self.template_pgs.gateSet,
        )
        for logical in range(num_circuit_qudits):
            pgs.set_qudit_position(logical, logical)
        return pgs

    def _candidate_start_mappings(
        self,
        circuit: Circuit,
        base_mapping: Sequence[int],
        num_circuit_qudits: int,
        data: PassData | None = None,
    ) -> list[LabeledLayoutCandidate]:
        rng = random.Random(self.seed)
        helper = LightSABRELayoutCandidateHelper(
            num_circuit_qudits=num_circuit_qudits,
            num_positions=self.template_pgs.num_pos,
            layout_trials=self.layout_trials,
        )

        def try_add(label: str, candidate: Sequence[int]) -> None:
            helper.add(label, candidate)

        try_add('base_mapping', base_mapping)

        for index, layout in enumerate(self.starting_layouts):
            try_add(f'starting_layout_{index}', layout)

        if data is not None:
            for key in ('lightsabre_starting_layouts', 'sabre_starting_layouts'):
                for index, layout in enumerate(data.get(key, [])):
                    try_add(f'{key}_{index}', layout)

        common_candidates: list[tuple[str, Sequence[int]]] = [
            ('rank_positions', self._rank_positions(reverse=False)[:num_circuit_qudits]),
            ('rank_positions_reverse', self._rank_positions(reverse=True)[:num_circuit_qudits]),
            ('activity_seed', self._activity_seed_mapping(circuit, reverse_positions=False)),
            ('activity_seed_reverse', self._activity_seed_mapping(circuit, reverse_positions=True)),
            ('interaction_seed', self._interaction_seed_mapping(circuit, reverse_positions=False)),
            ('interaction_seed_reverse', self._interaction_seed_mapping(circuit, reverse_positions=True)),
        ]
        if self.use_traffic_aware_seed:
            common_candidates.extend([
                (
                    self.TRAFFIC_AWARE_SEED_NAME,
                    self._traffic_aware_seed_mapping(
                        circuit,
                        reverse_positions=False,
                        anchor_rotation=0,
                    ),
                ),
                (
                    f'{self.TRAFFIC_AWARE_SEED_NAME}_reverse',
                    self._traffic_aware_seed_mapping(
                        circuit,
                        reverse_positions=True,
                        anchor_rotation=0,
                    ),
                ),
                (
                    f'{self.TRAFFIC_AWARE_SEED_NAME}_rotate1',
                    self._traffic_aware_seed_mapping(
                        circuit,
                        reverse_positions=False,
                        anchor_rotation=1,
                    ),
                ),
                (
                    f'{self.TRAFFIC_AWARE_SEED_NAME}_reverse_rotate1',
                    self._traffic_aware_seed_mapping(
                        circuit,
                        reverse_positions=True,
                        anchor_rotation=1,
                    ),
                ),
            ])

        for label, candidate in common_candidates:
            try_add(label, candidate)

        all_positions = list(range(self.template_pgs.num_pos))
        random_index = 0
        while len(helper.candidates) < self.layout_trials:
            if len(all_positions) == num_circuit_qudits:
                candidate = all_positions.copy()
                rng.shuffle(candidate)
            else:
                candidate = rng.sample(all_positions, num_circuit_qudits)

            try_add(f'random_{random_index}', candidate)
            random_index += 1

        return helper.candidates

    def _evaluate_layout(
        self,
        circuit: Circuit,
        layout: Sequence[int],
        layout_trial_index: int,
    ) -> tuple[int, tuple[int, int, int], Circuit, list[int]]:
        best_selection_score: int | None = None
        best_detail_score: tuple[int, int, int] | None = None
        best_circuit: Circuit | None = None
        best_final_mapping: list[int] | None = None

        for swap_trial in range(self.swap_trials):
            trial_circuit = circuit.copy()
            trial_pgs = self._build_eval_pgs(layout, circuit.num_qudits)
            self.begin_trial(layout_trial_index * self.swap_trials + swap_trial)
            self.forward_pass(trial_circuit, trial_pgs, modify_circuit=True)
            selection_score = self.routed_trial_selection_score(trial_circuit)
            detail_score = self.routed_trial_score(trial_circuit)

            if (
                best_selection_score is None
                or selection_score < best_selection_score
                or (
                    selection_score == best_selection_score
                    and best_detail_score is not None
                    and detail_score < best_detail_score
                )
            ):
                best_selection_score = selection_score
                best_detail_score = detail_score
                best_circuit = trial_circuit.copy()
                best_final_mapping = [
                    int(x) for x in trial_pgs.logical_to_position[:circuit.num_qudits]
                ]

        if (
            best_selection_score is None
            or best_detail_score is None
            or best_circuit is None
            or best_final_mapping is None
        ):
            raise RuntimeError('LightSABRE layout evaluation produced no trials.')

        return (
            best_selection_score,
            best_detail_score,
            best_circuit,
            best_final_mapping,
        )

    def layout_score(
        self,
        circuit: Circuit,
        layout: Sequence[int],
    ) -> tuple[float, float, float]:
        """
        Score a layout without routing using the LightSABRE front/lookahead terms.

        Returns:
            A tuple of:
            - total layout score
            - raw front-layer score
            - weighted extended-set score
        """
        pgs = self._build_pgs_from_mapping(layout, circuit.num_qudits)
        D = pgs.position_graph.move_cost_matrix
        mapping = pgs.logical_to_position
        F = set(circuit.front)
        E = self._calc_extended_set(circuit, F)

        front_total = 0.0
        for point in F:
            front_total += self._get_distance_from_mapping(
                tuple(int(q) for q in circuit[point].location),
                mapping,
                D,
            )

        extend = 0.0
        if self._uses_lookahead() and len(E) > 0:
            extend_total = 0.0
            for point in E:
                extend_total += self._get_distance_from_mapping(
                    tuple(int(q) for q in circuit[point].location),
                    mapping,
                    D,
                )
            extend = (extend_total / len(E)) * self.extended_set_weight

        return (front_total + extend, front_total, extend)

    async def run(self, circuit: Circuit, data: PassData) -> None:
        if getattr(data, 'placement', None) is not None:
            start_mapping = [int(x) for x in data.placement[:circuit.num_qudits]]
        else:
            start_mapping = list(range(circuit.num_qudits))
            data.placement = start_mapping.copy()

        best_layout: list[int] | None = None
        best_selection_score: int | None = None
        best_detail_score: tuple[int, int, int] | None = None
        ranked_layouts: list[tuple[int, tuple[int, int, int], str, list[int]]] = []
        routed_layout_candidates: list[dict[str, object]] = []

        for layout_trial_index, candidate in enumerate(
            self._candidate_start_mappings(circuit, start_mapping, circuit.num_qudits, data),
        ):
            mapping = candidate.mapping
            pgs = self._build_pgs_from_mapping(mapping, circuit.num_qudits)
            self.begin_trial(layout_trial_index)

            for _ in range(self.max_iterations):
                self.forward_pass(circuit, pgs, modify_circuit=False)
                self.backward_pass(circuit, pgs)

            perm = [int(x) for x in pgs.logical_to_position[:circuit.num_qudits]]
            (
                selection_score,
                detail_score,
                best_routed_circuit,
                best_final_mapping,
            ) = self._evaluate_layout(
                circuit,
                perm,
                layout_trial_index,
            )

            _logger.info(
                'LightSABRE layout trial %d label=%s produced selection_score=%s detail_score=%s layout=%s',
                layout_trial_index,
                candidate.label,
                selection_score,
                detail_score,
                perm,
            )

            ranked_layouts.append((selection_score, detail_score, candidate.label, perm))
            routed_layout_candidates.append({
                'label': candidate.label,
                'layout': perm.copy(),
                'selection_score': selection_score,
                'detail_score': tuple(detail_score),
                'final_mapping': list(best_final_mapping),
                'circuit': best_routed_circuit.copy(),
            })

            if (
                best_selection_score is None
                or selection_score < best_selection_score
                or (
                    selection_score == best_selection_score
                    and best_detail_score is not None
                    and detail_score < best_detail_score
                )
            ):
                best_selection_score = selection_score
                best_detail_score = detail_score
                best_layout = perm

        if best_layout is None:
            raise RuntimeError('LightSABRE layout was unable to produce a layout.')

        ranked_layouts.sort(key=lambda item: (item[0], item[1], item[2]))
        selected_layouts = [layout.copy() for _, _, _, layout in ranked_layouts]
        selected_labels = [label for _, _, label, _ in ranked_layouts]
        selected_scores = [score for score, _, _, _ in ranked_layouts]
        selected_detail_scores = [detail for _, detail, _, _ in ranked_layouts]
        data['lightsabre_layout_candidates'] = [layout.copy() for layout in selected_layouts]
        data['lightsabre_layout_candidate_labels'] = list(selected_labels)
        data['lightsabre_layout_candidate_scores'] = list(selected_scores)
        data['lightsabre_layout_candidate_detail_scores'] = [
            tuple(detail) for detail in selected_detail_scores
        ]
        routed_layout_candidates.sort(
            key=lambda item: (
                int(item['selection_score']),
                tuple(item['detail_score']),
            ),
        )
        data['lightsabre_routed_layout_candidates'] = routed_layout_candidates
        data['lightsabre_starting_layouts'] = [layout.copy() for layout in selected_layouts]
        self._apply_perm(best_layout, data.placement)
        _logger.info(
            'LightSABRE selected layout: %s, new placement: %s, best_selection_score=%s, best_detail_score=%s, ranked candidates=%d',
            best_layout,
            data.placement,
            best_selection_score,
            best_detail_score,
            len(selected_layouts),
        )

    # Backwards-compatible alias while the rename propagates.
    _circuit_aware_seed_mapping = _traffic_aware_seed_mapping
