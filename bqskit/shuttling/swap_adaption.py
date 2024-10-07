from __future__ import annotations

from bqskit import Circuit
from bqskit.compiler import PassData
from bqskit.compiler.basepass import BasePass
from bqskit.ir.gates import SwapGate

SWAP_circuit = Circuit.from_file("experiments/results/experiment_circuits/"
                                 "test_circuits/swap.qasm")


class SwapAdaption(BasePass):
    def __init__(self) -> None:
        """
        Nothing
        """

    async def run(self, circuit: Circuit, data: PassData) -> None:
        num_swap = circuit.count(SwapGate())
        while num_swap > 0:
            point = circuit.point(SwapGate())
            """
                Double check
            """
            op = circuit.get_operation(point)
            if op.gate == SwapGate():
                circuit.replace_with_circuit(point, SWAP_circuit)
            num_swap = circuit.count(SwapGate())
        circuit.unfold_all()
        assert circuit.count(SwapGate()) == 0
        return None
