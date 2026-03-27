from __future__ import annotations

import copy
import logging
from typing import Sequence

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit

from bqskit_local.mapping.sabre_pgs_behavioral_equivalence import GeneralizedSabreAlgorithmPGS
from bqskit_local.position.graph import PositionGraph
from bqskit_local.position.state import PositionGraphState

_logger = logging.getLogger(__name__)


class GeneralizedSabreLayoutPassPGS(BasePass, GeneralizedSabreAlgorithmPGS):
    """
    PGS layout pass redesigned to use only standard workflow-visible data.
    """

    def __init__(
        self,
        template_pgs: PositionGraphState,
        total_passes: int = 1,
        decay_delta: float = 0.001,
        decay_reset_interval: int = 5,
        decay_reset_on_gate: bool = True,
        extended_set_size: int = 20,
        extended_set_weight: float = 0.5,
    ) -> None:
        if not isinstance(template_pgs, PositionGraphState):
            raise TypeError(
                f'Expected PositionGraphState, got {type(template_pgs)}.',
            )

        if not isinstance(total_passes, int):
            raise TypeError(
                f'Expected int for total_passes, got {type(total_passes)}.',
            )

        if total_passes < 1:
            raise ValueError('Total passes must be a positive integer.')

        self.template_pgs = copy.deepcopy(template_pgs)
        self.total_passes = total_passes

        super().__init__(
            decay_delta=decay_delta,
            decay_reset_interval=decay_reset_interval,
            decay_reset_on_gate=decay_reset_on_gate,
            extended_set_size=extended_set_size,
            extended_set_weight=extended_set_weight,
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

        for logical, pos in enumerate(range(num_circuit_qudits)):
            pgs.set_qudit_position(logical, pos)

        return pgs

    async def run(self, circuit: Circuit, data: PassData) -> None:
        if getattr(data, 'placement', None) is not None:
            placement = [int(x) for x in data.placement[:circuit.num_qudits]]
        else:
            placement = list(range(circuit.num_qudits))

        pgs = self._build_local_pgs(placement, circuit.num_qudits)

        for _ in range(self.total_passes):
            self.forward_pass(circuit, pgs, modify_circuit=False)
            self.backward_pass(circuit, pgs)

        perm = [int(x) for x in pgs.logical_to_position[:circuit.num_qudits]]

        self._apply_perm(perm, data.placement)

        _logger.info(f'Found layout: {perm}, new placement: {data.placement}')
