"""Tracked wrapper around the standard CouplingGraph SABRE routing pass."""
from __future__ import annotations

import logging

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit

from bqskit_local.mapping.heuristic_stats import ensure_heuristic_stats
from bqskit_local.mapping.heuristic_stats import reset_heuristic_stats
from bqskit_local.mapping.heuristic_stats import summarize_heuristic_stats
from bqskit_local.mapping.sabre import GeneralizedSabreAlgorithm


_logger = logging.getLogger(__name__)


class GeneralizedSabreRoutingPass(BasePass, GeneralizedSabreAlgorithm):
    """
    Uses the Sabre algorithm to route the circuit and records heuristic stats.

    See :class:`GeneralizedSabreAlgorithm` for more info.
    """

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        reset_heuristic_stats(self)

        subgraph = data.connectivity
        if not subgraph.is_fully_connected():
            raise RuntimeError('Cannot route circuit on disconnected qudits.')

        pi = [i for i in range(circuit.num_qudits)]
        self.forward_pass(circuit, pi, subgraph, modify_circuit=True)
        data.final_mapping = [pi[x] for x in data.final_mapping]
        data['sabre_routing_heuristic_stats'] = (
            summarize_heuristic_stats(ensure_heuristic_stats(self))
            if self.collect_heuristic_stats
            else None
        )

        _logger.info(f'Finished routing with layout: {str(pi)}')
