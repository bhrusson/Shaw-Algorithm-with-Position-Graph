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
        self.positions = np.full(self.num_qudits, -1, dtype=int)  # -1 = unassigned
        self.qudit_values = np.zeros(self.num_qudits, dtype=int)
        self.history = []


    @property
    def position_graph(self) -> PositionGraph:
        return self._pg
    
    def set_qudit_position(self, qudit_id: int, pos_id: int) -> None:
        if qudit_id < 0 or qudit_id >= self.num_qudits:
            raise IndexError("Invalid qudit index")
        if pos_id < 0 or pos_id >= self.num_pos:
            raise IndexError("Invalid position index")
        self.positions[qudit_id] = pos_id
    
    def is_valid_qudit_id(self,qudit_id: int):
        if qudit_id < 0 or qudit_id >= self.num_qudits:
            raise IndexError("Invalid qudit index")
        
    def move_qudit(self, qudit_id: int, target_pos: int) -> None:
        self.set_qudit_position(qudit_id, target_pos)
    
    def set_qudit_value(self, qudit_id: int, value: int) -> None:
        radix = self.radices[qudit_id]
        if value < 0 or value >= radix:
            raise ValueError(f"Qudit value must be in 0..{radix-1}")
        self.qudit_values[qudit_id] = value

    def get_position(self, qudit_id: int) -> Optional[int]:
        return tuple(self.positions[qudit_id])
    
    def get_qudit_at_position(self, pos_id: int) -> Optional[int]:
        indices = np.flatnonzero(self.positions == pos_id)
        index = indices[0] if indices.size > 0 else -1
        return index
    
    def swap_positions(self, qudit_1: int, qudit_2: int) -> None:
        pos_1 = self.positions[qudit_1]
        pos_2 = self.positions[qudit_2]
        self.positions[qudit_1], self.positions[qudit_2] = pos_2, pos_1

    
    def record_state(self) -> None:
        snapshot = {
            "positions": self.positions.copy(),
            "values": self.qudit_values.copy()
        }
        self.history.append(snapshot)

    def apply_operation(self, qudit_ids: List[int], values: List[int]) -> None:
        for qid, val in zip(qudit_ids, values):
            self.set_qudit_value(qid, val)


