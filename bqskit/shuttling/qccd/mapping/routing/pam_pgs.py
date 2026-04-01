"""This module implements the PGS-native QCCD PAM routing pass."""
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


class QCCDPAMRoutingPassPGS(PermutationAwareQCCDMappingAlgorithmPGS, BasePass):
    """Permutation-aware routing using PositionGraphState."""

    out_data_key = '_pam_routing_block_out_data'

    def __init__(
        self,
        gate_count_weight: float = .1,
        decay_delta: float = 0.001,
        decay_reset_interval: int = 5,
        decay_reset_on_gate: bool = True,
        extended_set_size: int = 10,
        extended_set_weight: float = 0.5,
        cogestion_segment_rate: float = 0.75,
    ) -> None:
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
        if not cg.is_fully_connected():
            raise RuntimeError('Cannot route circuit on disconnected qudits.')
        logical_num_qudits = circuit.num_qudits

        perm_data: dict[CircuitPoint, PAMBlockTAPermData] = {}
        block_datas = data[ForEachBlockPass.key][-1]
        for block_data in block_datas:
            perm_data[block_data['point']] = block_data['permutation_data']

        ion_assignment = {
            int(logical): int(position)
            for logical, position in data['ion_assignment_qccd'].items()
        }
        data['initial_ion_assignment_qccd'] = copy.copy(ion_assignment)
        pgs = self.qccd_machine.build_pgs_from_assignment(ion_assignment)

        out_data, instruction_list, runtime = self.forward_pass(
            circuit,
            pgs,
            cg,
            perm_data,
            True,
        )
        final_assignment = self._assignment_from_pgs(pgs)
        data['ion_assignment_qccd'] = final_assignment
        data['ion_assignment_qccd_pgs'] = pgs
        data.final_mapping = list(range(logical_num_qudits))
        data[self.out_data_key] = out_data
        data['instruction_list'] = instruction_list
        data['moving_time'] = runtime
        _logger.info('Finished PGS PAM routing with assignment: %s', final_assignment)
