from __future__ import annotations

import copy
import logging
import os
from pathlib import Path

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit

from bqskit.shuttling.qccd.QCCD_machine import QCCDMachineModel
from bqskit.shuttling.qccd.QCCD_mapping import QCCDMappingAlgorithm
from bqskit.shuttling.qccd.pgs_passes.common import build_pgs_from_passdata
from bqskit.shuttling.qccd.pgs_passes.common import export_pgs_views_to_passdata
from bqskit.shuttling.qccd.pgs_passes.common import PROGRAM_ION_IDS_KEY
from bqskit.shuttling.qccd.pgs_passes.common import profiled_call

_logger = logging.getLogger(__name__)


def _capture_layout_snapshots_enabled() -> bool:
    return os.getenv('BQSKIT_QCCD_CAPTURE_LAYOUT_WRAPPER', '').lower() in (
        '1', 'true', 'yes', 'on',
    )


def _capture_layout_pgs_arrays_enabled() -> bool:
    return os.getenv('BQSKIT_QCCD_CAPTURE_PGS_ARRAYS', '').lower() in (
        '1', 'true', 'yes', 'on',
    )


class QCCDLayoutPassPGS(QCCDMappingAlgorithm, BasePass):
    """Layout algorithm using PGS-native QCCD mapping."""

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
        assignment_key: str = 'ion_assignment_qccd',
        profile_dir: Path | None = None,
        profile_stem: str = 'qccd_pgs_layout',
        profile_sort: str = 'cumulative',
        trace_memory: bool = False,
        trace_memory_depth: int = 5,
        trace_memory_top: int = 20,
    ) -> None:
        """
        Construct a QCCDLayoutPassPGS.

        Args:
            total_passes (int): The amount of forward and backward passes
                to apply before finalizing the layout.

            decay_delta (float): See :class:`QCCDMappingAlgorithm`
                for info. (Default: 0.001)

            decay_reset_interval (int): See :class:`QCCDMappingAlgorithm`
                for info. (Default: 5)

            decay_reset_on_gate (bool): See :class:`QCCDMappingAlgorithm`
                for info. (Default: True)

            extended_set_size (int): See :class:`QCCDMappingAlgorithm`
                for info. (Default: 5)

            extended_set_weight (float): See :class:`QCCDMappingAlgorithm`
                for info. (Default: 0.5)

            cogestion_rate (float): See :class:`QCCDMappingAlgorithm`
                for info. (Default: 0.75)

            force_bruteforce (bool): See :class:`QCCDMappingAlgorithm`
                for info. (Default: False)

            assignment_key (str): PassData key used to store the ion
                assignment. (Default: 'ion_assignment_qccd')
        """
        if not isinstance(total_passes, int):
            m = f'Expected int for total_passes, got {type(total_passes)}'
            raise TypeError(m)

        if total_passes < 1:
            raise ValueError('Total passes must be a positive integer.')

        self.total_passes = total_passes
        self.assignment_key = assignment_key
        self.qccd_machine = None
        self.profile_dir = profile_dir
        self.profile_stem = profile_stem
        self.profile_sort = profile_sort
        self.trace_memory = trace_memory
        self.trace_memory_depth = trace_memory_depth
        self.trace_memory_top = trace_memory_top

        super().__init__(
            qccd_machine=None,
            decay_delta=decay_delta,
            decay_reset_interval=decay_reset_interval,
            decay_reset_on_gate=decay_reset_on_gate,
            extended_set_size=extended_set_size,
            extended_set_weight=extended_set_weight,
            cogestion_rate=cogestion_rate,
            force_bruteforce=force_bruteforce,
        )

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        machine_model = data.model
        if not isinstance(machine_model, QCCDMachineModel):
            raise TypeError(
                f'Expected QCCDMachineModel in PassData.model, got {type(machine_model)}.',
            )

        self.qccd_machine = machine_model

        pgs = build_pgs_from_passdata(
            machine_model,
            data,
            assignment_key=self.assignment_key,
        )

        snapshots: list[tuple[str, dict[int, int]]] = []
        forward_traces: list[tuple[str, list[dict[str, object]]]] = []
        backward_traces: list[tuple[str, list[dict[str, object]]]] = []
        pgs_array_snapshots: list[tuple[str, list[int]]] = []
        program_ion_ids = list(data.get(PROGRAM_ION_IDS_KEY, []))

        if _capture_layout_snapshots_enabled():
            snapshots.append((
                'start',
                copy.deepcopy(self._program_assignment_from_pgs(pgs, program_ion_ids)),
            ))
        if _capture_layout_pgs_arrays_enabled():
            pgs_array_snapshots.append((
                'start',
                [int(x) for x in pgs.logical_to_position.tolist()],
            ))

        _logger.debug(f'Machine model: {machine_model}')
        _logger.debug(f'Number of qudits in the circuit: {circuit.num_qudits}')

        try:
            for layout_pass_index in range(self.total_passes):
                profiled_call(
                    self.profile_dir,
                    f'{self.profile_stem}__forward_{layout_pass_index + 1}',
                    self.profile_sort,
                    self.forward_pass,
                    circuit,
                    pgs=pgs,
                    modify_circuit=False,
                    trace_memory=self.trace_memory,
                    trace_memory_depth=self.trace_memory_depth,
                    trace_memory_top=self.trace_memory_top,
                )
                if _capture_layout_snapshots_enabled():
                    forward_traces.append((
                        f'forward_{layout_pass_index + 1}',
                        copy.deepcopy(getattr(self, 'last_forward_trace', [])),
                    ))
                if _capture_layout_snapshots_enabled():
                    snapshots.append((
                        f'forward_{layout_pass_index + 1}',
                        copy.deepcopy(self._program_assignment_from_pgs(pgs, program_ion_ids)),
                    ))
                if _capture_layout_pgs_arrays_enabled():
                    pgs_array_snapshots.append((
                        f'forward_{layout_pass_index + 1}',
                        [int(x) for x in pgs.logical_to_position.tolist()],
                    ))

                profiled_call(
                    self.profile_dir,
                    f'{self.profile_stem}__backward_{layout_pass_index + 1}',
                    self.profile_sort,
                    self.backward_pass,
                    circuit,
                    pgs=pgs,
                    trace_memory=self.trace_memory,
                    trace_memory_depth=self.trace_memory_depth,
                    trace_memory_top=self.trace_memory_top,
                )
                if _capture_layout_snapshots_enabled():
                    backward_traces.append((
                        f'backward_{layout_pass_index + 1}',
                        copy.deepcopy(getattr(self, 'last_backward_trace', [])),
                    ))
                if _capture_layout_snapshots_enabled():
                    snapshots.append((
                        f'backward_{layout_pass_index + 1}',
                        copy.deepcopy(self._program_assignment_from_pgs(pgs, program_ion_ids)),
                    ))
                if _capture_layout_pgs_arrays_enabled():
                    pgs_array_snapshots.append((
                        f'backward_{layout_pass_index + 1}',
                        [int(x) for x in pgs.logical_to_position.tolist()],
                    ))
        finally:
            self._shutdown_routing_executor()

        export_pgs_views_to_passdata(
            data,
            pgs,
            assignment_key=self.assignment_key,
        )

        if snapshots:
            data['qccd_layout_wrapper_snapshots'] = snapshots
            data['qccd_layout_wrapper_forward_traces'] = forward_traces
            data['qccd_layout_wrapper_backward_traces'] = backward_traces
        if pgs_array_snapshots:
            data['qccd_layout_wrapper_pgs_arrays'] = pgs_array_snapshots
        data.model = self.qccd_machine
