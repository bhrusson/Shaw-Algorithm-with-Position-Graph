"""This module implements the BarrierPlaceholder class."""
from __future__ import annotations

from typing import Sequence

from bqskit.ir.gate import Gate
from bqskit.qis.unitary.unitary import RealVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class ShuttlingShiftGate(Gate):
    """ShuttlingShiftGate. """

    def __init__(self, num_qudits: int, radixes: Sequence[int] = []) -> None:
        """Construct a ShuttlingShiftGate."""
        self._name = 'ShuttlingShiftGate'
        self._qasm_name = 'ShuttlingShiftGate'
        self._num_qudits = num_qudits
        self._radixes = tuple(radixes) if radixes else tuple([2] * num_qudits)
        self._num_params = 0

    def get_unitary(self, params: RealVector = []) -> UnitaryMatrix:
        return UnitaryMatrix.identity(self.dim, self.radixes)

    def __eq__(self, other: object) -> bool:
        return (
                isinstance(other, ShuttlingShiftGate)
                and other.num_qudits == self.num_qudits
                and other.radixes == self.radixes
        )

    def __hash__(self) -> int:
        return hash(('ShuttlingShiftGate', self.num_qudits, self.radixes))
