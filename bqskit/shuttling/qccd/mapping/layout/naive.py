"""This module implements the PAMLayoutPass."""
from __future__ import annotations

import logging
import copy
import os
from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.shuttling.qccd.QCCD_mapping import QCCDMappingAlgorithm

_logger = logging.getLogger(__name__)


def _capture_layout_snapshots_enabled() -> bool:
    return os.getenv('BQSKIT_QCCD_CAPTURE_LAYOUT_WRAPPER', '').lower() in (
        '1', 'true', 'yes', 'on',
    )


class QCCDLayoutPass(QCCDMappingAlgorithm, BasePass):
    """Layout algorithm using permutation-aware mapping."""

    def __init__(
        self,
        total_passes: int = 1,
        decay_delta: float = 0.001,
        decay_reset_interval: int = 5,
        decay_reset_on_gate: bool = True,
        extended_set_size: int = 5,
        extended_set_weight: float = 0.5,
        cogestion_rate: float = 0.75,
        force_bruteforce: bool = True,
    ) -> None:
        """
        Construct a LayoutPass.

        Args:
            total_passes (int): The amount of forward and backward passes
                to apply before finalizing the layout.


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
        self.cogestion_rate = cogestion_rate
        super().__init__(
            decay_delta,
            decay_reset_interval,
            decay_reset_on_gate,
            extended_set_size,
            extended_set_weight,
            cogestion_rate,
            force_bruteforce=force_bruteforce,
        )

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        subgraph = data.model.coupling_graph #As this is QCCD, the subgraph should be fully connected
        self.qccd_machine = data.model
        if not subgraph.is_fully_connected():
            raise RuntimeError('Cannot route circuit on disconnected qudits.')
        pi = [i for i in range(circuit.num_qudits)]
        ion_assignment = data['ion_assignment_qccd']
        snapshots: list[tuple[str, dict[int, int]]] = []
        forward_traces: list[tuple[str, list[dict[str, object]]]] = []
        if _capture_layout_snapshots_enabled():
            snapshots.append(('start', copy.deepcopy(ion_assignment)))
        _logger.debug(f'Subgraph: {subgraph}')
        _logger.debug(f'Number of qudits in the circuit: {circuit.num_qudits}')
        for layout_pass_index in range(self.total_passes):
            self.forward_pass(circuit, pi, ion_assignment, False)
            if _capture_layout_snapshots_enabled():
                forward_traces.append((
                    f'forward_{layout_pass_index + 1}',
                    copy.deepcopy(getattr(self, 'last_forward_trace', [])),
                ))
            if _capture_layout_snapshots_enabled():
                snapshots.append((
                    f'forward_{layout_pass_index + 1}',
                    copy.deepcopy(ion_assignment),
                ))
            self.backward_pass(circuit, pi, ion_assignment)
            if _capture_layout_snapshots_enabled():
                snapshots.append((
                    f'backward_{layout_pass_index + 1}',
                    copy.deepcopy(ion_assignment),
                ))
        data['ion_assignment_qccd'] = ion_assignment
        if snapshots:
            data['qccd_layout_wrapper_snapshots'] = snapshots
            data['qccd_layout_wrapper_forward_traces'] = forward_traces
        data.model = self.qccd_machine
