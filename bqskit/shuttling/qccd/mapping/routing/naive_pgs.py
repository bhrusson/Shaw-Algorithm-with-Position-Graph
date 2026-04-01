"""This module implements the PGS-native QCCD routing pass."""
from __future__ import annotations

import copy
import logging

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.shuttling.qccd.QCCD_machine_PGS import QCCDMachineModel
from bqskit.shuttling.qccd.QCCD_mapping_PGS import QCCDMappingAlgorithm
from bqskit.shuttling.qccd.pgs_passes.common import build_pgs_from_passdata
from bqskit.shuttling.qccd.pgs_passes.common import export_pgs_views_to_passdata
from bqskit.shuttling.qccd.pgs_passes.common import PROGRAM_ION_IDS_KEY

_logger = logging.getLogger(__name__)


class QCCDRoutingPassPGS(QCCDMappingAlgorithm, BasePass):
    """Routing pass using the native PositionGraphState mapper."""

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        self.qccd_machine = data.model
        if not isinstance(self.qccd_machine, QCCDMachineModel):
            raise TypeError(
                f'Expected QCCDMachineModel, got {type(self.qccd_machine)}.',
            )
        logical_num_qudits = circuit.num_qudits

        pgs = build_pgs_from_passdata(self.qccd_machine, data)
        program_ion_ids = list(data.get(PROGRAM_ION_IDS_KEY, []))
        initial_program_assignment = self._program_assignment_from_pgs(
            pgs,
            program_ion_ids,
        )
        data['initial_ion_assignment_qccd'] = copy.copy(initial_program_assignment)
        data['initial_program_ion_assignment_qccd'] = copy.copy(initial_program_assignment)
        data['initial_full_ion_assignment_qccd_pgs'] = copy.copy(
            self._full_assignment_from_pgs(pgs),
        )

        _logger.debug(
            'Ion assignment at the beginning of routing: %s',
            initial_program_assignment,
        )
        _logger.debug(
            'Number of qudits in the circuit: %s',
            circuit.num_qudits,
        )

        instruction_list = self.forward_pass(circuit, pgs=pgs, modify_circuit=True)
        _full_assignment, program_assignment, program_ion_ids = export_pgs_views_to_passdata(
            data,
            pgs,
        )
        # The QCCD routing pass already rewrites the circuit onto physical
        # positions, so downstream ApplyPlacement-style passes should see an
        # identity logical remapping just like the original QCCD routing pass.
        data.final_mapping = [
            int(pgs.logical_to_position[int(logical)])
            for logical in program_ion_ids[:logical_num_qudits]
        ]

        _logger.info('Finished routing with assignment: %s', program_assignment)
        data['instruction_list'] = instruction_list
