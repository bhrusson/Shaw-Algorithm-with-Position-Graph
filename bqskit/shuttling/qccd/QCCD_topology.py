from bqskit.ir.circuit import Circuit
from bqskit.qis.graph import CouplingGraph
from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.passes.control.foreach import ForEachBlockPass
from bqskit.utils.typing import is_integer


class QCCDSubtopologySelectionPass(BasePass):
    """Pass that selects necessary subtopologies from the model."""

    key = ForEachBlockPass.pass_down_key_prefix + 'sub_topologies'

    def __init__(self, block_size: int) -> None:
        """
        Construct a SubtopologySelectionPass.

        Args:
            block_size (int): The max block size to select subtopologies for.

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
        tops = {}
        for i in range(2, self.block_size + 1):
            tops[i] = [CouplingGraph.linear(i)]

        data[self.key] = tops
