from __future__ import annotations
from typing import Sequence

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit

from bqskit.superconducting.position.state import PositionGraphState


class SetPGSPass(BasePass):
    """
    Initialize standard mapping state for PGS passes.

    This pass avoids storing custom objects in PassData, since those do not
    appear to persist back out of the compiler workflow.
    """

    def __init__(
        self,
        pgs: PositionGraphState,
        placement: Sequence[int] | None = None,
    ) -> None:
        if not isinstance(pgs, PositionGraphState):
            raise TypeError(
                f'Expected PositionGraphState, got {type(pgs)}.',
            )

        self.template_pgs = pgs.copy()
        self.placement = None if placement is None else list(placement)

    async def run(self, circuit: Circuit, data: PassData) -> None:
        if self.template_pgs.num_qudits < circuit.num_qudits:
            raise RuntimeError('PositionGraphState too small for circuit.')

        if self.template_pgs.num_pos < circuit.num_qudits:
            raise RuntimeError('Not enough positions for circuit.')

        if self.placement is None:
            placement = list(range(self.template_pgs.num_qudits))
        else:
            placement = [int(x) for x in self.placement]

        if len(placement) < circuit.num_qudits:
            raise ValueError(
                'Placement length must be at least circuit.num_qudits.',
            )

        if len(set(placement)) != len(placement):
            raise ValueError('Placement must assign distinct positions.')

        data.placement = placement.copy()
        data.initial_mapping = placement.copy()
        data.final_mapping = placement.copy()
