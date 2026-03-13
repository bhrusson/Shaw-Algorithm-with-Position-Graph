from __future__ import annotations

import copy
import logging

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit

from bqskit_local.mapping.sabre_pgs import GeneralizedSabreAlgorithmPGS
from bqskit_local.position.state import PositionGraphState

_logger = logging.getLogger(__name__)


class GeneralizedSabreLayoutPassPGS(BasePass, GeneralizedSabreAlgorithmPGS):
    """
    Uses the SABRE-PGS algorithm to choose an initial layout.
    """

    def __init__(
        self,
        total_passes: int = 1,
        decay_delta: float = 0.001,
        decay_reset_interval: int = 5,
        decay_reset_on_gate: bool = True,
        extended_set_size: int = 20,
        extended_set_weight: float = 0.5,
    ) -> None:
        if not isinstance(total_passes, int):
            raise TypeError(
                f'Expected int for total_passes, got {type(total_passes)}.',
            )

        if total_passes < 1:
            raise ValueError('Total passes must be a positive integer.')

        self.total_passes = total_passes

        super().__init__(
            decay_delta=decay_delta,
            decay_reset_interval=decay_reset_interval,
            decay_reset_on_gate=decay_reset_on_gate,
            extended_set_size=extended_set_size,
            extended_set_weight=extended_set_weight,
        )

    def _layout_score(self, circuit: Circuit, pgs: PositionGraphState) -> float:
        """
        Score a candidate layout using frontier and lookahead distances.
        Lower is better.
        """
        D = pgs.position_graph.move_cost_matrix
        F = set(circuit.front)
        E = self._calc_extended_set(circuit, F)

        total = 0.0

        for cp in F:
            loc = circuit[cp].location
            for i in range(len(loc)):
                p1 = int(pgs.logical_to_position[loc[i]])
                if p1 == -1:
                    return float("inf")
                for j in range(i + 1, len(loc)):
                    p2 = int(pgs.logical_to_position[loc[j]])
                    if p2 == -1:
                        return float("inf")
                    total += D[p1][p2]

        for cp in E:
            loc = circuit[cp].location
            for i in range(len(loc)):
                p1 = int(pgs.logical_to_position[loc[i]])
                if p1 == -1:
                    return float("inf")
                for j in range(i + 1, len(loc)):
                    p2 = int(pgs.logical_to_position[loc[j]])
                    if p2 == -1:
                        return float("inf")
                    total += self.extended_set_weight * D[p1][p2]

        return float(total)

    async def run(self, circuit: Circuit, data: PassData) -> None:
        if 'pgs' not in data:
            raise RuntimeError(
                'GeneralizedSabreLayoutPassPGS requires data["pgs"].',
            )

        pgs: PositionGraphState = data['pgs']

        placed = [p for p in pgs.logical_to_position[:circuit.num_qudits] if p != -1]
        if len(placed) != circuit.num_qudits:
            raise RuntimeError(
                'All circuit qudits must be assigned to positions before layout.',
            )

        best_pgs = None
        best_score = self._layout_score(circuit, pgs)

        for _ in range(self.total_passes):
            trial_pgs = copy.deepcopy(pgs)

            self.forward_pass(circuit, trial_pgs, modify_circuit=False)
            self.backward_pass(circuit, trial_pgs)

            score = self._layout_score(circuit, trial_pgs)
            if score < best_score:
                best_score = score
                best_pgs = trial_pgs

        if best_pgs is None:
            raise RuntimeError('Failed to compute a layout.')

        data['initial_mapping'] = best_pgs.logical_to_position.copy()

        pgs._logical_to_position[:] = best_pgs.logical_to_position
        pgs._position_to_logical[:] = best_pgs.position_to_logical

        _logger.info(
            f'Found layout: {data["initial_mapping"]}, score: {best_score}',
        )