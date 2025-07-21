from bqskit.position.graph import PositionGraph

class PositionGraphState:
    def __init__(
        self,
        pg: PositionGraph,
        logical_to_position_map: list[int]
    ):
        self.pg = pg
        self.logical_to_position_map = logical_to_position_map 

    def get_position(self, qudit_index: int) -> Tuple:
        """Return the position tuple for the given qudit index."""
        return tuple(self.logical_to_position_map[qudit_index])

    def can_execute_qudit(self, qudit_index) -> bool:
        pass

    def get_position_state(self, position_index):
        pass

