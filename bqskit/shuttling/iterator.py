from __future__ import annotations

from bqskit.ir.iterator import CircuitIterator
from bqskit.ir import Circuit
from bqskit.ir.gates import *
from bqskit.ir import Operation


class CircuitOddEvenIterator(CircuitIterator):
    """Fast and simple iteration through circuit."""

    def __init__(self, circuit: Circuit):
        """Set a CircuitDagIterator to iterate through `circuit`."""
        self.circuit = circuit
        self.frontier = circuit.front
        self.prev_layer_parity = bool(0)
        self.layer_cnt = 0
        self.gates_per_layer = self.return_reorder_gate_by_layers()

    def return_reorder_gate_by_layers(self):
        layer = self.circuit[self.layer_cnt]
        even_rzz = []
        odd_rzz = []
        reordered_layer = []
        for op in layer:
            if op.gate == RZZGate():
                if op.location[0] > op.location[1]:
                    odd_rzz.append(op)
                else:
                    even_rzz.append(op)
        if even_rzz == [] and odd_rzz != []:
            self.prev_layer_parity = bool(1)
            reordered_layer = layer
        elif odd_rzz == [] and even_rzz != []:
            self.prev_layer_parity = bool(0)
            reordered_layer = layer
        elif odd_rzz != [] and even_rzz != []:
            if self.prev_layer_parity:
                reordered_layer.extend(odd_rzz)
                reordered_layer.extend(even_rzz)
            else:
                reordered_layer.extend(even_rzz)
                reordered_layer.extend(odd_rzz)
        else:
            reordered_layer = layer
        return reordered_layer

    def __next__(self) -> Operation:
        if len(self.gates_per_layer) == 0:
            self.layer_cnt += 1
            if self.layer_cnt == self.circuit.num_cycles:
                raise StopIteration
            else:
                self.gates_per_layer = self.return_reorder_gate_by_layers()
                op = self.gates_per_layer.pop(0)
        else:
            op = self.gates_per_layer.pop(0)
        return op

    def __iter__(self) -> CircuitIterator:
        return self
