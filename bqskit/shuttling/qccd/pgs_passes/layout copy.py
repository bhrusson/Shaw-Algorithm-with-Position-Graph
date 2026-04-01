from __future__ import annotations

import copy
import logging
import os

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit

from bqskit.shuttling.qccd.QCCD_machine_PGS import QCCDMachineModel
from bqskit.shuttling.qccd.QCCD_mapping_PGS import QCCDMappingAlgorithm
from bqskit.shuttling.qccd.pgs_passes.common import build_pgs_from_passdata

_logger = logging.getLogger(__name__)


def _capture_layout_snapshots_enabled() -> bool:
    return os.getenv('BQSKIT_QCCD_CAPTURE_LAYOUT_WRAPPER', '').lower() in (
        '1', 'true', 'yes', 'on',
    )


class QCCDLayoutPassPGS(BasePass):
    """
    Scaffold for a PGS-native QCCD layout pass.

    This package mirrors the SABRE PGS structure: algorithm and state live in
    dedicated PGS modules, while pass wrappers live in a separate package and
    operate through workflow-visible ``PassData``.
    """

    def __init__(
        self,
        total_passes: int = 1,
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
        if not isinstance(total_passes, int):
            raise TypeError(
                f'Expected int for total_passes, got {type(total_passes)}.',
            )
        if total_passes < 1:
            raise ValueError('Total passes must be a positive integer.')

        self.total_passes = int(total_passes)
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

        pgs = build_pgs_from_passdata(
            machine_model,
            data,
            assignment_key=self.assignment_key,
        )
        algo = QCCDMappingAlgorithm(qccd_machine=machine_model, **self.algo_kwargs)
        snapshots: list[tuple[str, dict[int, int]]] = []
        forward_traces: list[tuple[str, list[dict[str, object]]]] = []
        if _capture_layout_snapshots_enabled():
            snapshots.append(('start', algo._assignment_from_pgs(pgs)))

        for layout_pass_index in range(self.total_passes):
            algo.forward_pass(circuit, pgs=pgs, modify_circuit=False)
            if _capture_layout_snapshots_enabled():
                forward_traces.append((
                    f'forward_{layout_pass_index + 1}',
                    copy.deepcopy(getattr(algo, 'last_forward_trace', [])),
                ))
            if _capture_layout_snapshots_enabled():
                snapshots.append((
                    f'forward_{layout_pass_index + 1}',
                    algo._assignment_from_pgs(pgs),
                ))
            algo.backward_pass(circuit, pgs=pgs)
            if _capture_layout_snapshots_enabled():
                snapshots.append((
                    f'backward_{layout_pass_index + 1}',
                    algo._assignment_from_pgs(pgs),
                ))

        final_assignment = algo._assignment_from_pgs(pgs)
        data[self.assignment_key] = final_assignment
        data['qccd_layout_pgs_algorithm'] = algo
        data['qccd_layout_pgs_total_passes'] = self.total_passes
        data[f'{self.assignment_key}_pgs'] = pgs
        if snapshots:
            data['qccd_layout_wrapper_snapshots'] = snapshots
            data['qccd_layout_wrapper_forward_traces'] = forward_traces
        data.model = machine_model

        _logger.info(
            'Finished QCCD PGS layout with assignment: %s',
            final_assignment,
        )
