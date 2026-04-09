from __future__ import annotations

import logging
import random
from typing import Sequence

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit

from bqskit_local.mapping.lightSABRE_pgs import GeneralizedLightSABREAlgorithmPGS
from bqskit_local.position.graph import PositionGraph
from bqskit_local.position.state import PositionGraphState

_logger = logging.getLogger(__name__)


class GeneralizedLightSABRELayoutPassPGS(BasePass, GeneralizedLightSABREAlgorithmPGS):
    """LightSABRE-style PGS layout search with multiple layout trials."""

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
        heuristic: str = 'decay',
        seed: int | None = None,
        attempt_limit: int | None = None,
        starting_layouts: Sequence[Sequence[int]] | None = None,
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

    def _interaction_seed_mapping(
        self,
        circuit: Circuit,
        reverse_positions: bool = False,
    ) -> list[int]:
        num_qudits = circuit.num_qudits
        base_pg = self.template_pgs.position_graph
        position_order = self._rank_positions(reverse=reverse_positions)
        activity = [0.0] * num_qudits
        pair_weight: dict[tuple[int, int], float] = {}

        total_ops = max(1, circuit.num_operations)
        for op_index, op in enumerate(circuit):
            if op.num_qudits < 2:
                continue

            weight = 2.0 - (op_index / total_ops)
            logicals = [int(qudit) for qudit in op.location]

            for logical in logicals:
                activity[logical] += weight * len(logicals)

            for left_index in range(len(logicals)):
                for right_index in range(left_index + 1, len(logicals)):
                    left = logicals[left_index]
                    right = logicals[right_index]
                    key = (left, right) if left < right else (right, left)
                    pair_weight[key] = pair_weight.get(key, 0.0) + weight

        mapping = [-1] * num_qudits
        available_positions = position_order.copy()
        assigned_logicals: set[int] = set()

        def pair_score(a: int, b: int) -> float:
            key = (a, b) if a < b else (b, a)
            return pair_weight.get(key, 0.0)

        while len(assigned_logicals) < num_qudits and available_positions:
            unassigned = [logical for logical in range(num_qudits) if logical not in assigned_logicals]
            logical = max(
                unassigned,
                key=lambda candidate: (
                    max((pair_score(candidate, other) for other in assigned_logicals), default=0.0),
                    activity[candidate],
                    -candidate,
                ),
            )

            if not assigned_logicals:
                chosen_position = available_positions.pop(0)
            else:
                best_partner = max(
                    assigned_logicals,
                    key=lambda other: (pair_score(logical, other), activity[other], -other),
                )
                partner_position = mapping[best_partner]
                if pair_score(logical, best_partner) > 0.0:
                    chosen_position = min(
                        available_positions,
                        key=lambda pos: (
                            float(base_pg.move_cost_matrix[pos][partner_position]),
                            position_order.index(pos),
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
        activity = [0.0] * num_qudits
        total_ops = max(1, circuit.num_operations)

        for op_index, op in enumerate(circuit):
            weight = 2.0 - (op_index / total_ops)
            for logical in op.location:
                activity[int(logical)] += weight * max(1, op.num_qudits)

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

    def _add_candidate_mapping(
        self,
        candidates: list[list[int]],
        seen: set[tuple[int, ...]],
        candidate: Sequence[int],
        num_circuit_qudits: int,
    ) -> None:
        normalized = [int(position) for position in candidate[:num_circuit_qudits]]
        if len(normalized) != num_circuit_qudits:
            return
        if len(set(normalized)) != len(normalized):
            return
        if any(position < 0 or position >= self.template_pgs.num_pos for position in normalized):
            return

        key = tuple(normalized)
        if key in seen:
            return

        seen.add(key)
        candidates.append(normalized)

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
    ) -> list[list[int]]:
        rng = random.Random(self.seed)
        candidates: list[list[int]] = []
        seen: set[tuple[int, ...]] = set()

        def try_add(candidate: Sequence[int]) -> None:
            if len(candidates) >= self.layout_trials:
                return
            self._add_candidate_mapping(candidates, seen, candidate, num_circuit_qudits)

        try_add(base_mapping)

        for layout in self.starting_layouts:
            try_add(layout)

        if data is not None:
            for key in ('lightsabre_starting_layouts', 'sabre_starting_layouts'):
                for layout in data.get(key, []):
                    try_add(layout)

        common_candidates = [
            self._rank_positions(reverse=False)[:num_circuit_qudits],
            self._rank_positions(reverse=True)[:num_circuit_qudits],
            self._activity_seed_mapping(circuit, reverse_positions=False),
            self._activity_seed_mapping(circuit, reverse_positions=True),
            self._interaction_seed_mapping(circuit, reverse_positions=False),
            self._interaction_seed_mapping(circuit, reverse_positions=True),
        ]

        for candidate in common_candidates:
            try_add(candidate)

        all_positions = list(range(self.template_pgs.num_pos))
        while len(candidates) < self.layout_trials:
            if len(all_positions) == num_circuit_qudits:
                candidate = all_positions.copy()
                rng.shuffle(candidate)
            else:
                candidate = rng.sample(all_positions, num_circuit_qudits)

            try_add(candidate)

        return candidates

    def _evaluate_layout(
        self,
        circuit: Circuit,
        layout: Sequence[int],
        layout_trial_index: int,
    ) -> tuple[int, int, int]:
        best_score: tuple[int, int, int] | None = None

        for swap_trial in range(self.swap_trials):
            trial_circuit = circuit.copy()
            trial_pgs = self._build_eval_pgs(layout, circuit.num_qudits)
            self.begin_trial(layout_trial_index * self.swap_trials + swap_trial)
            self.forward_pass(trial_circuit, trial_pgs, modify_circuit=True)
            score = self.routed_trial_score(trial_circuit)

            if best_score is None or score < best_score:
                best_score = score

        if best_score is None:
            raise RuntimeError('LightSABRE layout evaluation produced no trials.')

        return best_score

    async def run(self, circuit: Circuit, data: PassData) -> None:
        if getattr(data, 'placement', None) is not None:
            start_mapping = [int(x) for x in data.placement[:circuit.num_qudits]]
        else:
            start_mapping = list(range(circuit.num_qudits))
            data.placement = start_mapping.copy()

        best_layout: list[int] | None = None
        best_score: tuple[int, int, int] | None = None
        ranked_layouts: list[tuple[tuple[int, int, int], list[int]]] = []

        for layout_trial_index, mapping in enumerate(
            self._candidate_start_mappings(circuit, start_mapping, circuit.num_qudits, data),
        ):
            pgs = self._build_pgs_from_mapping(mapping, circuit.num_qudits)
            self.begin_trial(layout_trial_index)

            for _ in range(self.max_iterations):
                self.forward_pass(circuit, pgs, modify_circuit=False)
                self.backward_pass(circuit, pgs)

            perm = [int(x) for x in pgs.logical_to_position[:circuit.num_qudits]]
            score = self._evaluate_layout(circuit, perm, layout_trial_index)

            _logger.info(
                'LightSABRE layout trial %d produced score=%s layout=%s',
                layout_trial_index,
                score,
                perm,
            )

            ranked_layouts.append((score, perm))

            if best_score is None or score < best_score:
                best_score = score
                best_layout = perm

        if best_layout is None:
            raise RuntimeError('LightSABRE layout was unable to produce a layout.')

        ranked_layouts.sort(key=lambda item: item[0])
        selected_layouts = [best_layout]
        data['lightsabre_layout_candidates'] = [layout.copy() for layout in selected_layouts]
        data['lightsabre_layout_candidate_scores'] = [best_score]
        data['lightsabre_starting_layouts'] = [layout.copy() for layout in selected_layouts]
        self._apply_perm(best_layout, data.placement)
        _logger.info(
            'LightSABRE selected layout: %s, new placement: %s',
            best_layout,
            data.placement,
        )
