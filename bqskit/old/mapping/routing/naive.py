"""This module implements the PAMRoutingPass."""
from __future__ import annotations

import logging
import copy
from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.shuttling.qccd.QCCD_mapping import QCCDMappingAlgorithm

_logger = logging.getLogger(__name__)


class QCCDRoutingPass(QCCDMappingAlgorithm, BasePass):
    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        subgraph = data.model.coupling_graph
        self.qccd_machine = data.model
        self.qccd_machine.position_graph = subgraph
        if not subgraph.is_fully_connected():
            raise RuntimeError('Cannot route circuit on disconnected qudits.')

        pi = [i for i in range(circuit.num_qudits)]
        ion_assignment = data['ion_assignment_qccd']
        data['initial_ion_assignment_qccd'] = copy.copy(data['ion_assignment_qccd'])
        _logger.debug(f"Ion assignment at the beginning of routing: {ion_assignment}")
        _logger.debug(f'Subgraph: {subgraph}')
        _logger.debug(f'Number of qudits in the circuit: {circuit.num_qudits}')
        instruction_list = self.forward_pass(circuit, pi, ion_assignment, True)
        data.final_mapping = [pi[x] for x in data.final_mapping]

        _logger.info(f'Finished routing with layout: {str(pi)}')
        # data[self.out_data_key] = out_data
        data["instruction_list"] = instruction_list
        # data["moving_time"] = runtime
