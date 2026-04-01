"""This module implements the PGS-native QCCD layout pass."""
from __future__ import annotations

import logging

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.shuttling.qccd.QCCD_machine_PGS import QCCDMachineModel
from bqskit.shuttling.qccd.QCCD_mapping_PGS import QCCDMappingAlgorithm
from bqskit.shuttling.qccd.pgs_passes.common import build_pgs_from_passdata
from bqskit.shuttling.qccd.pgs_passes.common import export_pgs_views_to_passdata

_logger = logging.getLogger(__name__)


class QCCDLayoutPassPGS(QCCDMappingAlgorithm, BasePass):
    """Layout algorithm using the native PositionGraphState mapper."""

    def __init__(
        self,
        total_passes: int = 1,
        decay_delta: float = 0.001,
        decay_reset_interval: int = 5,
        decay_reset_on_gate: bool = True,
        extended_set_size: int = 5,
        extended_set_weight: float = 0.5,
        cogestion_rate: float = 0.75,
        force_bruteforce: bool = False,
    ) -> None:
        if not isinstance(total_passes, int):
            raise TypeError(
                f'Expected int for total_passes, got {type(total_passes)}.',
            )
        if total_passes < 1:
            raise ValueError('Total passes must be a positive integer.')

        self.total_passes = total_passes
        self.cogestion_rate = cogestion_rate
        super().__init__(
            decay_delta,
            decay_reset_interval,
            decay_reset_on_gate,
            extended_set_size,
            extended_set_weight,
            cogestion_rate=cogestion_rate,
            force_bruteforce=force_bruteforce,
        )

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        self.qccd_machine = data.model
        if not isinstance(self.qccd_machine, QCCDMachineModel):
            raise TypeError(
                f'Expected QCCDMachineModel, got {type(self.qccd_machine)}.',
            )

        pgs = build_pgs_from_passdata(self.qccd_machine, data)

        _logger.debug('Number of qudits in the circuit: %s', circuit.num_qudits)
        _logger.debug('Initial ion assignment: %s', data.get('program_ion_assignment_qccd', data.get('ion_assignment_qccd')))
        for _ in range(self.total_passes):
            self.forward_pass(circuit, pgs=pgs, modify_circuit=False)
            self.backward_pass(circuit, pgs=pgs)

        export_pgs_views_to_passdata(data, pgs)
        data.model = self.qccd_machine
