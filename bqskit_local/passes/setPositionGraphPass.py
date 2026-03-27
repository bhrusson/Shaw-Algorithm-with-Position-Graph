from bqskit_local.position.state import PositionGraphState
from bqskit.compiler.basepass import BasePass
from bqskit.ir.circuit import Circuit


class SetPositionGraphPass(BasePass):
    def __init__(self, position_graph_state):
        self.pgs = position_graph_state

    async def run(self, circuit, data):
        data.position_graph_state = self.pgs
        data.pgs = PositionGraphState.from_graph(self.pg)