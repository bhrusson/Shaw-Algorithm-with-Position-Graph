"""This module implements the PGS-native QCCD PAM layout pass."""
from __future__ import annotations

import copy
import logging

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.ir.point import CircuitPoint
from bqskit.passes.control.foreach import ForEachBlockPass
from bqskit.passes.mapping.pam import PAMBlockTAPermData
from bqskit.qis.graph import CouplingGraph
from bqskit_local.position.graph import EdgeCapability
from bqskit.shuttling.qccd.mapping.pam_PGS import (
    PermutationAwareQCCDMappingAlgorithmPGS,
)

_logger = logging.getLogger(__name__)


def _move_coupling_graph(model) -> CouplingGraph:
    edges = {
        tuple(sorted((u, v)))
        for (u, v), label in model.position_graph.edge_labels.items()
        if label.has_capability(EdgeCapability.MOVE)
    }
    return CouplingGraph(list(edges), model.num_positions)


class QCCDPAMLayoutPassPGS(PermutationAwareQCCDMappingAlgorithmPGS, BasePass):
    """Permutation-aware layout using PositionGraphState."""

    def __init__(
        self,
        total_passes: int = 1,
        gate_count_weight: float = .1,
        decay_delta: float = 0.001,
        decay_reset_interval: int = 5,
        decay_reset_on_gate: bool = True,
        extended_set_size: int = 10,
        extended_set_weight: float = 0.5,
        cogestion_segment_rate: float = 0.75,
    ) -> None:
        if not isinstance(total_passes, int):
            raise TypeError(f'Expected int for total_passes, got {type(total_passes)}')
        if total_passes < 1:
            raise ValueError('Total passes must be a positive integer.')
        self.total_passes = total_passes
        self.cogestion_segment_rate = cogestion_segment_rate
        super().__init__(
            gate_count_weight,
            decay_delta,
            decay_reset_interval,
            decay_reset_on_gate,
            extended_set_size,
            extended_set_weight,
            cogestion_rate=cogestion_segment_rate,
        )

    async def run(self, circuit: Circuit, data: PassData) -> None:
        self.qccd_machine = data.model
        cg = _move_coupling_graph(self.qccd_machine)
        perm_data: dict[CircuitPoint, PAMBlockTAPermData] = {}
        block_datas = data[ForEachBlockPass.key][-1]
        for block_data in block_datas:
            perm_data[block_data['point']] = block_data['permutation_data']

        if not cg.is_fully_connected():
            raise RuntimeError('Cannot route circuit on disconnected qudits.')

        ion_assignment = {
            int(logical): int(position)
            for logical, position in data['ion_assignment_qccd'].items()
        }
        pgs = self.qccd_machine.build_pgs_from_assignment(ion_assignment)
        for _ in range(self.total_passes):
            self.forward_pass(circuit, pgs, cg, perm_data, False)
            self.backward_pass(circuit, pgs)

        initial_placement = copy.copy(data.placement)
        final_assignment = self._assignment_from_pgs(pgs)
        data['ion_assignment_qccd'] = final_assignment
        data['ion_assignment_qccd_pgs'] = pgs
        data.model = self.qccd_machine
        _logger.info(
            'Finished PGS PAM layout. initial placement=%s, assignment=%s',
            initial_placement,
            final_assignment,
        )
