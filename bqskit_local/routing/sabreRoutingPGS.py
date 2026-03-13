from __future__ import annotations

import logging

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit

from bqskit_local.mapping.sabre_pgs import GeneralizedSabreAlgorithmPGS


_logger = logging.getLogger(__name__)


class GeneralizedSabreRoutingPassPGS(BasePass, GeneralizedSabreAlgorithmPGS):

    async def run(self, circuit: Circuit, data: PassData) -> None:

        if "pgs" not in data:
            raise RuntimeError(
                'GeneralizedSabreRoutingPassPGS requires data["pgs"].'
            )

        pgs = data["pgs"]

        self.forward_pass(circuit, pgs, modify_circuit=True)

        data["pgs"] = pgs

        data["final_mapping"] = [
            int(x) for x in pgs.logical_to_position[:circuit.num_qudits]
        ]

        _logger.info(
            f'Finished routing with layout: {data["final_mapping"]}'
        )