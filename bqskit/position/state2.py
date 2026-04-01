from .graph import PositionGraph
import numpy as np
from typing import Tuple, List, Sequence, Mapping, Dict, Callable, Optional

"""
This maps the qudits to their positions on the position graph.
"""
class PositionGraphState:
    def __init__(
        self,
        pg: PositionGraph,
        radices: Sequence[int], #Length of this is the number of qudits
        
    ):
        self.pg = pg
        self.radices = radices
        self.num_qudits = len(radices)        
        self.num_pos = len(pg.position_labels)
        self.positions = np.full(self.num_pos, -1, dtype=int)  # -1 = unassigned
        self.qudit_values = np.zeros(self.num_qudits, dtype=int)
        self.history = []


    def set_qudit_position(self, qudit_id: int, pos_id: int) -> None:
        if qudit_id < 0 or qudit_id >= self.num_qudits:
            raise IndexError("Invalid qudit index")
        if pos_id < 0 or pos_id >= self.num_pos:
            raise IndexError("Invalid position index")
        self.positions[qudit_id] = pos_id
    
    def move_qudit(self, qudit_id: int, target_pos: int) -> None:
        self.set_qudit_position(qudit_id, target_pos)

    def get_position(self, qudit_id: int) -> Optional[int]:
        return (self.positions[qudit_id])
    
    def get_qudit_at_position(self, pos_id: int) -> Optional[int]:
        index = np.flatnonzero(self.positions == pos_id)[0][0] if np.any(self.positions == pos_id) else -1
        return index
        
    def record_state(self) -> None:
        snapshot = {
            "positions": self.positions.copy(),
            "values": self.qudit_values.copy()
        }
        self.history.append(snapshot)

    def apply_operation(self, qudit_ids: List[int], values: List[int]) -> None:
        for qid, val in zip(qudit_ids, values):
            self.set_qudit_value(qid, val)

