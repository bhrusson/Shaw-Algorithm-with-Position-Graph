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


class QCCDRoutingPassPGS(BasePass):
    """
    Scaffold for a PGS-native QCCD routing pass.

    The intended end state is similar to the SABRE PGS routing wrapper: build
    workflow-visible state from ``PassData``, run the native PGS mapper, then
    write the resulting circuit/mapping data back into ``PassData``.
    """

    def __init__(
        self,
        gate_count_weight: float = 0.1,
        *,
        assignment_key: str = 'ion_assignment_qccd',
        cogestion_rate: float = 1.0,
        decay_delta: float = 0.001,
        decay_reset_interval: int = 5,
        decay_reset_on_gate: bool = True,
        extended_set_size: int = 5,
        extended_set_weight: float = 0.5,
        force_bruteforce: bool = False,
    ) -> None:
        self.gate_count_weight = float(gate_count_weight)
        self.assignment_key = assignment_key
        self.algo_kwargs = {
            'cogestion_rate': cogestion_rate,
            'decay_delta': decay_delta,
            'decay_reset_interval': decay_reset_interval,
            'decay_reset_on_gate': decay_reset_on_gate,
            'extended_set_size': extended_set_size,
            'extended_set_weight': extended_set_weight,
            'force_bruteforce': force_bruteforce,
        }

    async def run(self, circuit: Circuit, data: PassData) -> None:
        machine_model = data.model
        if not isinstance(machine_model, QCCDMachineModel):
            raise TypeError(
                f'Expected QCCDMachineModel in PassData.model, got {type(machine_model)}.',
            )
        logical_num_qudits = circuit.num_qudits

        pgs = build_pgs_from_passdata(
            machine_model,
            data,
            assignment_key=self.assignment_key,
        )
        algo = QCCDMappingAlgorithm(qccd_machine=machine_model, **self.algo_kwargs)
        program_ion_ids = list(data.get(PROGRAM_ION_IDS_KEY, []))

        initial_full_assignment = algo._full_assignment_from_pgs(pgs)
        initial_program_assignment = algo._program_assignment_from_pgs(
            pgs,
            program_ion_ids,
        )
        data['initial_full_ion_assignment_qccd_pgs'] = copy.copy(initial_full_assignment)
        data['initial_program_ion_assignment_qccd'] = copy.copy(initial_program_assignment)
        data['initial_ion_assignment_qccd'] = copy.copy(initial_program_assignment)
        instruction_list = algo.forward_pass(circuit, pgs=pgs, modify_circuit=True)

        full_assignment, program_assignment, program_ion_ids = export_pgs_views_to_passdata(
            data,
            pgs,
            assignment_key=self.assignment_key,
        )
        data['instruction_list'] = instruction_list
        data.final_mapping = [
            int(pgs.logical_to_position[int(logical)])
            for logical in program_ion_ids[:logical_num_qudits]
        ]
        data['qccd_routing_pgs_algorithm'] = algo
        data['qccd_routing_pgs_gate_count_weight'] = self.gate_count_weight

        _logger.info(
            'Finished QCCD PGS routing with final assignment: %s',
            program_assignment,
        )
