from bqskit.compiler.machine import MachineModel
from bqskit.qis.graph import CouplingGraph
from typing import Sequence
from bqskit_local.position.state import PositionGraphState


class MachineModelPGS(MachineModel):

    def __init__(self, pgs: PositionGraphState):
        self.pgs = pgs

        super().__init__(
            pgs.num_qudits,
            self._build_coupling_graph(),
            pgs.gateSet,
            pgs.radices,
        )

    @property
    def coupling_graph(self):
        return self._build_coupling_graph()

    def _build_coupling_graph(self) -> CouplingGraph:
        edges = []

        for q1 in range(self.pgs.num_qudits):
            p1 = self.pgs.logical_to_position[q1]
            if p1 == -1:
                continue

            for q2 in range(self.pgs.num_qudits):
                if q1 == q2:
                    continue

                p2 = self.pgs.logical_to_position[q2]
                if p2 == -1:
                    continue

                if self.pgs.position_graph.gate_is_executable(p1, p2):
                    edges.append((q1, q2))

        return CouplingGraph(edges)
    
    def refresh(self) -> None:
        """
        Rebuild the coupling graph after qudit movement.
        """
        self.coupling_graph = self._build_coupling_graph()