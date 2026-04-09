from __future__ import annotations
import logging
from typing import Sequence

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit

from bqskit_local.mapping.sabre_pgs_behavioral_equivalence import GeneralizedSabreAlgorithmPGS
from bqskit_local.position.state import PositionGraphState

_logger = logging.getLogger(__name__)


class GeneralizedSabreLayoutPassPGS(BasePass, GeneralizedSabreAlgorithmPGS):
    """
    PGS layout pass redesigned to use only standard workflow-visible data.
    """

    def __init__(
        self,
        template_pgs: PositionGraphState,
        total_passes: int = 1,
        decay_delta: float = 0.001,
        decay_reset_interval: int = 5,
        decay_reset_on_gate: bool = True,
        extended_set_size: int = 20,
        extended_set_weight: float = 0.5,
        cg_compatibility_mode: bool = False,
    ) -> None:
        if not isinstance(template_pgs, PositionGraphState):
            raise TypeError(
                f'Expected PositionGraphState, got {type(template_pgs)}.',
            )

        if not isinstance(total_passes, int):
            raise TypeError(
                f'Expected int for total_passes, got {type(total_passes)}.',
            )

        if total_passes < 1:
            raise ValueError('Total passes must be a positive integer.')

        self.template_pgs = template_pgs.copy()
        self.total_passes = total_passes

        super().__init__(
            decay_delta=decay_delta,
            decay_reset_interval=decay_reset_interval,
            decay_reset_on_gate=decay_reset_on_gate,
            extended_set_size=extended_set_size,
            extended_set_weight=extended_set_weight,
            cg_compatibility_mode=cg_compatibility_mode,
        )

    def _build_pgs_from_mapping(
        self,
        mapping: Sequence[int],
        num_circuit_qudits: int,
    ) -> PositionGraphState:
        pgs = self.template_pgs.copy()
        pgs.clear_assignments()

        if len(mapping) != num_circuit_qudits:
            raise ValueError(
                f'Expected mapping of length {num_circuit_qudits}, got {len(mapping)}.',
            )

        if len(set(mapping)) != len(mapping):
            raise ValueError('Mapping must assign distinct positions.')

        for logical, pos in enumerate(mapping):
            pos = int(pos)
            if pos < 0 or pos >= pgs.num_pos:
                raise ValueError(f'Invalid position {pos} for logical {logical}.')
            pgs.set_qudit_position(logical, pos)

        return pgs

    async def run(self, circuit: Circuit, data: PassData) -> None:
        if getattr(data, 'placement', None) is not None:
            start_mapping = [int(x) for x in data.placement[:circuit.num_qudits]]
        else:
            start_mapping = list(range(circuit.num_qudits))
            data.placement = start_mapping.copy()

        pgs = self._build_pgs_from_mapping(start_mapping, circuit.num_qudits)

        for _ in range(self.total_passes):
            self.forward_pass(circuit, pgs, modify_circuit=False)
            self.backward_pass(circuit, pgs)

        perm = [int(x) for x in pgs.logical_to_position[:circuit.num_qudits]]
        self._apply_perm(perm, data.placement)

        _logger.info(f'Found layout: {perm}, new placement: {data.placement}')
