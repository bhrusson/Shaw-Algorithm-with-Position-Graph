"""This module implements the UnfoldPass class."""
from __future__ import annotations

import logging

from bqskit_local.compiler.basePassPGS import BasePassPGS
from bqskit_local.compiler.passDataPGS import PassDataPGS
from bqskit.ir.circuit import Circuit


_logger = logging.getLogger(__name__)


class UnfoldPassPGS(BasePassPGS):
    """
    The UnfoldPass class.

    The UnfoldPass unfolds all CircuitGate blocks into the circuit.
    """

    async def run(self, circuit: Circuit, data: PassDataPGS) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        _logger.debug('Unfolding the circuit.')
        circuit.unfold_all()