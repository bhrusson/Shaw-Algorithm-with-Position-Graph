from .graph import PositionGraph
import numpy as np
from typing import Tuple, List, Sequence, Mapping, Dict, Callable, Optional


class PositionGraphState:
    def __init__(
        self,
        pg: PositionGraph,
        radices: Sequence[int]
    ):
        self._pg = pg
        self.radices = radices
        self.num_qudits = len(self.radices)
        self.num_pos = len(pg.position_labels)
        self._logical_to_physical = np.full(self.num_qudits, -1, dtype=int)  # -1 = unassigned
        self._physical_to_logical = np.full(len(pg.position_labels), -1, dtype=int)
        self.history: List[dict] = []
 

    @property
    def position_graph(self) -> PositionGraph:
        return self._pg
    
    @property
    def logical_to_physical(self) -> np.ndarray:
        return self._logical_to_physical
    
    @property
    def physical_to_logical(self) -> np.ndarray:
        return self._physical_to_logical
    
    def set_qudit_position(self, qudit_id: int, pos_id: int) -> None:
        if qudit_id < 0 or qudit_id >= self.num_qudits:
            raise IndexError("Invalid qudit index")
        if pos_id < 0 or pos_id >= self.num_pos:
            raise IndexError("Invalid position index")
        old_position = self.logical_to_physical[qudit_id]
        if old_position != -1:
            self.physical_to_logical[old_position] = -1
        self.logical_to_physical[qudit_id] = pos_id
        self.physical_to_logical[pos_id] = qudit_id

    
    def move_qudit(self, qudit_id: int, target_pos: int) -> None:
        self.set_qudit_position(qudit_id, target_pos)
    
    def get_position_of_qudit(self, qudit_id: int) -> Optional[int]:
        return self.logical_to_physical[qudit_id]
    
    def get_logical_qudit_at_position(self, pos_id: int) -> Optional[int]:
        return self.physical_to_logical[pos_id]
    
    def swap_positions(self, qudit_1: int, qudit_2: int) -> None:
        pos_1 = self.logical_to_physical[qudit_1]
        pos_2 = self.logical_to_physical[qudit_2]
                
        self.logical_to_physical[qudit_1], self.logical_to_physical[qudit_2] = pos_2, pos_1
        self.physical_to_logical[pos_1], self.physical_to_logical[pos_2] = qudit_1, qudit_2


    
    def record_state(self) -> None:
        snapshot = {
            "positions": self.logical_to_physical.copy(),
        }
        self.history.append(snapshot)

    def in_cluster(self, qudits: List[int]) -> bool:
        positions = [self.logical_to_physical[q] for q in qudits]
        return self._pg.in_cluster(positions)


    
    def get_shortest_path_tree(self, start_pos: int) -> List[Tuple[int, ...]]:
        # 1. get lengths
        lengths = self._pg.shortest_path_lengths
        # 2. get paths
        paths = self._pg.shortest_paths
        # Now combine into a list indexed by position
        tree = []
        for node in range(len(self._pg.position_labels)):
            if node not in paths:
                tree.append(tuple())  # unreachable
            else:
                tree.append(tuple(paths[node]))
        return tree
