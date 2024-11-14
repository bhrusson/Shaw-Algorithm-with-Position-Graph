"""This module implements the PAMLayoutPass."""
from __future__ import annotations

import logging
import copy
from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.ir.point import CircuitPoint
from bqskit.passes.control.foreach import ForEachBlockPass
from bqskit.passes.mapping.pam import PAMBlockTAPermData
from bqskit.shuttling.qccd.mapping.pam import PermutationAwareQCCDMappingAlgorithm

_logger = logging.getLogger(__name__)


class QCCDPAMLayoutPass(PermutationAwareQCCDMappingAlgorithm, BasePass):
    """Layout algorithm using permutation-aware mapping."""

    def __init__(
        self,
        total_passes: int = 1,
        gate_count_weight: float = .1,
        decay_delta: float = 0.001,
        decay_reset_interval: int = 5,
        decay_reset_on_gate: bool = True,
        extended_set_size: int = 10,
        extended_set_weight: float = 0.5,
        cogestion_segment_rate: float = 0.75
    ) -> None:
        """
        Construct a PAMLayoutPass.

        Args:
            total_passes (int): The amount of forward and backward passes
                to apply before finalizing the layout.

            gate_count_weight (float): See
                :class:`PermutationAwareMappingAlgorithm` for info.
                (Default: 0.3)

            decay_delta (float): See :class:`GeneralizedSabreAlgorithm`
                for info. (Default: 0.001)

            decay_reset_interval (int): See :class:`GeneralizedSabreAlgorithm`
                for info. (Default: 5)

            decay_reset_on_gate (bool): See :class:`GeneralizedSabreAlgorithm`
                for info. (Default: True)

            extended_set_size (int): See :class:`GeneralizedSabreAlgorithm`
                for info. (Default: 20)

            extended_set_weight (float): See :class:`GeneralizedSabreAlgorithm`
                for info. (Default: 0.5)
        """
        if not isinstance(total_passes, int):
            m = f'Expected int for total_passes, got {type(total_passes)}'
            raise TypeError(m)

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
            extended_set_weight
        )

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        subgraph = data.model.coupling_graph #As this is QCCD, the subgraph should be fully connected
        data.connectivity
        # (only need to consider one type)
        self.qccd_machine = data.model
        perm_data: dict[CircuitPoint, PAMBlockTAPermData] = {}
        block_datas = data[ForEachBlockPass.key][-1]
        for block_data in block_datas:
            perm_data[block_data['point']] = block_data['permutation_data']

        if not subgraph.is_fully_connected():
            raise RuntimeError('Cannot route circuit on disconnected qudits.')

        pi = [i for i in range(circuit.num_qudits)]
        ion_assignment = data['ion_assignment_qccd']
        _logger.debug(f'Subgraph: {subgraph}')
        _logger.debug(f'Number of qudits in the circuit: {circuit.num_qudits}')
        for _ in range(self.total_passes):
            self.forward_pass(circuit, pi, ion_assignment, subgraph, perm_data)
            self.backward_pass(circuit, pi, ion_assignment)
        initial_placement = copy.copy(data.placement)
        _logger.debug(f'Data placement: {data.placement}')
        _logger.debug(f'Pi: {pi}')
        self._apply_perm(pi, data.placement, ion_assignment)
        _logger.info(f'Found layout: {pi}, new placement: {data.placement}')
        _logger.info(f'New coupling graph: {data.connectivity}')
        _logger.info(f'Start update machine model ....')
        _logger.info(f'Initial placement: {initial_placement}')
        data['ion_assignment_qccd'] = ion_assignment
        self.qccd_machine.update_wrt_perm(initial_placement=initial_placement,
                                          permutation=data.placement)
        data.model = self.qccd_machine
        _logger.info(f'Finished update machine model ...., new coupling graph: {self.qccd_machine.position_graph}')