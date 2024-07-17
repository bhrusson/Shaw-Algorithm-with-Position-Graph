from __future__ import annotations

import itertools as it
from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.passes.control.foreach import ForEachBlockPass
from bqskit.ir.circuit import Circuit
from bqskit.utils.typing import is_integer


def all_possible_gate_zones_of_size(size: int, num_qubits: int, ) -> list[tuple]:
    """Calculates all valid gate zone with specific size of `n` qudits circuit."""
    return list(it.combinations(range(num_qubits), size))


def filter_invalid_gate_zones(combs: list[tuple], size: int) -> list[set]:
    """Filter out invalid gate zone from all combinations"""
    valid_gate_zones = []
    for i in combs:
        invalid_flag = False
        if len(i) != size:
            raise ValueError('The combinations has invalid size, expected {}, got {}'.format(size, len(i)))
        for idx in range(size - 1):
            if i[idx] + 1 == i[idx + 1]:
                invalid_flag = True
                continue
        if not invalid_flag:
            valid_gate_zones.append(set(i))
    return valid_gate_zones


class GateZoneSelectionPass(BasePass):
    """Pass that selects possible gate zones from the model."""

    key = ForEachBlockPass.pass_down_key_prefix + 'gate_zones'

    def __init__(self, block_size: int) -> None:
        """
        Construct a GateZoneSelectionPass.

        Args:
            num_qubits (int): The number of qubits to select possible gate zones for.

        Raises:
            ValueError: If block_size is <= 1.
        """
        if not is_integer(block_size):
            raise TypeError(f'Expected integer, got {type(block_size)}.')

        if block_size <= 1:
            raise ValueError(f'Expected integer > 1, got {block_size}.')

        self.block_size = block_size

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        # print("Block size: ", self.block_size)
        possible_gate_zones = {}
        for i in range(1, ((self.block_size + 1) // 2) + 1):
            all_possible_gate_zones = all_possible_gate_zones_of_size(i, self.block_size)
            possible_gate_zones[i] = filter_invalid_gate_zones(all_possible_gate_zones, i)
        data[self.key] = possible_gate_zones
