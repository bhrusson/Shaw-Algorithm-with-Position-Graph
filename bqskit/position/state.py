from .graph import PositionGraph
import numpy as np
from typing import Tuple, List, Sequence, Mapping, Dict, Callable, Optional


class PositionGraphState:
    def __init__(
        self,
        pg: PositionGraph,
    ):
        self.pg = pg
        self.num_qudits = pg.num_qudits
        self.num_pos = pg.len(pg.position_labels)
        self.positions = np.full(self.num_qudits, -1, dtype=int)  # -1 = unassigned
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
    
    def set_qudit_value(self, qudit_id: int, value: int) -> None:
        radix = self.pg.radices[qudit_id]
        if value < 0 or value >= radix:
            raise ValueError(f"Qudit value must be in 0..{radix-1}")
        self.qudit_values[qudit_id] = value

    def get_position(self, qudit_id: int) -> Optional[int]:
        return tuple(self.positions[qudit_id])
    
    def get_qudits_at_position(self, pos_id: int) -> List[int]:
        return list(np.where(self.positions == pos_id)[0])
    
    
    def record_state(self) -> None:
        snapshot = {
            "positions": self.positions.copy(),
            "values": self.qudit_values.copy()
        }
        self.history.append(snapshot)

    def apply_operation(self, qudit_ids: List[int], values: List[int]) -> None:
        for qid, val in zip(qudit_ids, values):
            self.set_qudit_value(qid, val)

