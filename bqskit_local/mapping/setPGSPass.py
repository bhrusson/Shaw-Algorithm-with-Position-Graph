from __future__ import annotations

from typing import Sequence

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit

from bqskit_local.position.state import PositionGraphState


class SetPGSPass(BasePass):
    """Initialize PositionGraphState for PGS passes."""

    def __init__(
        self,
        pgs: PositionGraphState,
        placement: Sequence[int] | None = None,
    ) -> None:
        if not isinstance(pgs, PositionGraphState):
            raise TypeError(
                f'Expected PositionGraphState, got {type(pgs)}.',
            )

        self.pgs = pgs
        self.placement = None if placement is None else list(placement)

    async def run(self, circuit: Circuit, data: PassData) -> None:
        if self.pgs.num_qudits < circuit.num_qudits:
            raise RuntimeError('PositionGraphState too small for circuit.')

        if self.placement is not None:
            if len(self.placement) != circuit.num_qudits:
                raise ValueError(
                    'Placement length must equal circuit.num_qudits.',
                )

            for logical, pos in enumerate(self.placement):
                self.pgs.set_qudit_position(logical, int(pos))

        mapping = [
            int(x) for x in self.pgs.logical_to_position[:circuit.num_qudits]
        ]

        data["pgs"] = self.pgs
        data["position_graph"] = self.pgs.position_graph
        data["initial_mapping"] = mapping.copy()
        data["final_mapping"] = mapping.copy()