from __future__ import annotations

import logging
from typing import Sequence

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit

from bqskit.superconducting.mapping.sabre_pgs import (
    GeneralizedSabreAlgorithmPGS,
)
from bqskit.superconducting.position.graph import PositionGraph
from bqskit.superconducting.position.state import PositionGraphState

_logger = logging.getLogger(__name__)


class GeneralizedSabreRoutingPassPGS(BasePass, GeneralizedSabreAlgorithmPGS):
    """Cached PGS routing pass using heuristic-region reuse."""

    def __init__(
        self,
        template_pgs: PositionGraphState,
        decay_delta: float = 0.001,
        decay_reset_interval: int = 5,
        decay_reset_on_gate: bool = True,
        extended_set_size: int = 20,
        extended_set_weight: float = 0.5,
        cg_compatibility_mode: bool = False,
    ) -> None:
        if not isinstance(template_pgs, PositionGraphState):
            raise TypeError(
                f'Expected PositionGraphState, got {type(template_pgs)}.',
            )

        self.template_pgs = template_pgs.copy()

        super().__init__(
            decay_delta=decay_delta,
            decay_reset_interval=decay_reset_interval,
            decay_reset_on_gate=decay_reset_on_gate,
            extended_set_size=extended_set_size,
            extended_set_weight=extended_set_weight,
            cg_compatibility_mode=cg_compatibility_mode,
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
        if len(placement) < self.template_pgs.num_pos:
            return self._build_local_pgs(placement, num_circuit_qudits)

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
        if getattr(data, 'placement', None) is not None:
            placement = [int(x) for x in data.placement[:circuit.num_qudits]]
        else:
            placement = list(range(circuit.num_qudits))

        if self.cg_compatibility_mode:
            pgs = self._build_compatibility_local_pgs(
                placement,
                circuit.num_qudits,
            )
        else:
            pgs = self._build_local_pgs(placement, circuit.num_qudits)

        _logger.info(f'SABRE routing start mapping: {pgs.logical_to_position}')
        _logger.info(f'SABRE routing start placement: {data.get("placement")}')

        self.forward_pass(circuit, pgs, modify_circuit=True)

        final_mapping = [int(x) for x in pgs.logical_to_position[:circuit.num_qudits]]
        data.final_mapping = final_mapping.copy()

        _logger.info(f'Finished SABRE routing with layout: {final_mapping}')
