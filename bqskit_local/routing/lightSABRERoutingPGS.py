from __future__ import annotations

import logging
from typing import Sequence

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit

from bqskit_local.mapping.lightSABRE_pgs import DEFAULT_LIGHTSABRE_HEURISTIC
from bqskit_local.mapping.lightSABRE_pgs import GeneralizedLightSABREAlgorithmPGS
from bqskit_local.position.graph import PositionGraph
from bqskit_local.position.state import PositionGraphState

_logger = logging.getLogger(__name__)


class GeneralizedLightSABRERoutingPassPGS(BasePass, GeneralizedLightSABREAlgorithmPGS):
    """LightSABRE-style PGS routing with multiple seeded trials."""

    def __init__(
        self,
        template_pgs: PositionGraphState,
        decay_delta: float = 0.001,
        decay_reset_interval: int = 5,
        decay_reset_on_gate: bool = True,
        extended_set_size: int = 20,
        extended_set_weight: float = 0.5,
        cg_compatibility_mode: bool = False,
        heuristic: str = DEFAULT_LIGHTSABRE_HEURISTIC,
        seed: int | None = None,
        trials: int = 1,
        attempt_limit: int | None = None,
    ) -> None:
        if not isinstance(template_pgs, PositionGraphState):
            raise TypeError(
                f'Expected PositionGraphState, got {type(template_pgs)}.',
            )

        if not isinstance(trials, int):
            raise TypeError(f'Expected int for trials, got {type(trials)}.')

        if trials < 1:
            raise ValueError('trials must be a positive integer.')

        self.template_pgs = template_pgs.copy()
        self.trials = trials

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

    def _build_local_pgs(
        self,
        placement: Sequence[int],
        num_circuit_qudits: int,
    ) -> PositionGraphState:
        if len(placement) != num_circuit_qudits:
            raise ValueError(
                f'Expected placement of length {num_circuit_qudits}, got {len(placement)}.',
            )

        if len(set(placement)) != len(placement):
            raise ValueError('Placement must assign distinct positions.')

        base_pg = self.template_pgs.position_graph
        placement = [int(x) for x in placement]

        for pos in placement:
            if pos < 0 or pos >= base_pg.graph.num_nodes():
                raise ValueError(f'Invalid position {pos} in placement.')

        pgs = self.template_pgs.copy()
        pgs.clear_assignments()

        for logical, pos in enumerate(placement):
            pgs.set_qudit_position(logical, pos)

        return pgs

    def _build_compatibility_local_pgs(
        self,
        placement: Sequence[int],
        num_circuit_qudits: int,
    ) -> PositionGraphState:
        if len(placement) != num_circuit_qudits:
            raise ValueError(
                f'Expected placement of length {num_circuit_qudits}, got {len(placement)}.',
            )

        if len(set(placement)) != len(placement):
            raise ValueError('Placement must assign distinct positions.')

        base_pg = self.template_pgs.position_graph
        placement = [int(x) for x in placement]

        for pos in placement:
            if pos < 0 or pos >= base_pg.graph.num_nodes():
                raise ValueError(f'Invalid position {pos} in placement.')

        inverse_placement = {pos: i for i, pos in enumerate(placement)}
        local_pos_labels = [base_pg.position_labels[pos] for pos in placement]
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

    async def run(self, circuit: Circuit, data: PassData) -> None:
        precomputed_candidates = data.get('lightsabre_routed_layout_candidates', [])
        if len(precomputed_candidates) > 0:
            best_candidate = min(
                precomputed_candidates,
                key=lambda item: (
                    int(item['selection_score']),
                    tuple(item['detail_score']),
                ),
            )
            best_circuit = best_candidate['circuit']
            if not isinstance(best_circuit, Circuit):
                raise TypeError(
                    'Expected Circuit in lightsabre_routed_layout_candidates.',
                )

            circuit.become(best_circuit.copy())
            data.final_mapping = list(best_candidate['final_mapping'])
            data['lightsabre_selected_routing_placement'] = list(best_candidate['layout'])
            data['lightsabre_total_routing_trials'] = len(precomputed_candidates)
            data['lightsabre_selected_routing_score'] = int(best_candidate['selection_score'])
            data['lightsabre_selected_routing_detail_score'] = tuple(best_candidate['detail_score'])

            _logger.info(
                'LightSABRE selected precomputed routed candidate score=%s detail_score=%s final_mapping=%s candidates=%d',
                best_candidate['selection_score'],
                best_candidate['detail_score'],
                best_candidate['final_mapping'],
                len(precomputed_candidates),
            )
            return

        if getattr(data, 'placement', None) is not None:
            placement = [int(x) for x in data.placement[:circuit.num_qudits]]
        else:
            placement = list(range(circuit.num_qudits))

        candidate_placements: list[list[int]] = []
        seen: set[tuple[int, ...]] = set()

        def add_candidate(candidate: Sequence[int]) -> None:
            normalized = [int(x) for x in candidate[:circuit.num_qudits]]
            if len(normalized) != circuit.num_qudits:
                return
            key = tuple(normalized)
            if key in seen:
                return
            seen.add(key)
            candidate_placements.append(normalized)

        add_candidate(placement)
        for candidate in data.get('lightsabre_layout_candidates', []):
            add_candidate(candidate)

        best_selection_score: int | None = None
        best_detail_score: tuple[int, int, int] | None = None
        best_circuit: Circuit | None = None
        best_mapping: list[int] | None = None
        best_placement: list[int] | None = None

        global_trial_index = 0
        for placement_index, trial_placement in enumerate(candidate_placements):
            for local_trial_index in range(self.trials):
                if self.cg_compatibility_mode:
                    pgs = self._build_compatibility_local_pgs(
                        trial_placement,
                        circuit.num_qudits,
                    )
                else:
                    pgs = self._build_local_pgs(trial_placement, circuit.num_qudits)

                trial_circuit = circuit.copy()
                self.begin_trial(global_trial_index)
                self.forward_pass(trial_circuit, pgs, modify_circuit=True)

                selection_score = self.routed_trial_selection_score(trial_circuit)
                detail_score = self.routed_trial_score(trial_circuit)
                final_mapping = [
                    int(x) for x in pgs.logical_to_position[:circuit.num_qudits]
                ]

                _logger.info(
                    'LightSABRE routing placement %d trial %d produced selection_score=%s detail_score=%s final_mapping=%s',
                    placement_index,
                    local_trial_index,
                    selection_score,
                    detail_score,
                    final_mapping,
                )

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
                    best_circuit = trial_circuit
                    best_mapping = final_mapping
                    best_placement = trial_placement.copy()

                global_trial_index += 1

        if best_circuit is None or best_mapping is None or best_placement is None:
            raise RuntimeError('LightSABRE routing produced no successful trials.')

        circuit.become(best_circuit)
        data.final_mapping = best_mapping.copy()
        data['lightsabre_selected_routing_placement'] = best_placement.copy()
        data['lightsabre_total_routing_trials'] = global_trial_index
        data['lightsabre_selected_routing_score'] = best_selection_score
        data['lightsabre_selected_routing_detail_score'] = best_detail_score

        _logger.info(
            'LightSABRE selected routing score=%s detail_score=%s final_mapping=%s total_trials=%d',
            best_selection_score,
            best_detail_score,
            best_mapping,
            global_trial_index,
        )
