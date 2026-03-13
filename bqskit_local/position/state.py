from .graph import PositionGraph
import numpy as np
from typing import Tuple, List, Sequence, Mapping, Dict, Callable, Optional
from bqskit.compiler.gateset import GateSet
from bqskit.qis.graph import CouplingGraph



class PositionGraphState:
    def __init__(
        self,
        pg: PositionGraph,
        radices: Sequence[int],
        gateSet: GateSet | None = None,

    
    ):
        self._pg = pg
        self.radices = radices
        self.num_qudits = len(self.radices)
        self.num_pos = len(pg.position_labels)
        self._logical_to_position = np.full(self.num_qudits, -1, dtype=int) #the index represents the logical qubit value, the value represents the position_index and -1 = unassigned
        self._position_to_logical = np.full(len(pg.position_labels), -1, dtype=int) #index represents the position_index and value is which logical qubit is present.  -1 = unassigned
        self.history: List[dict] = []
 
        if gateSet is None:            
            gateSet = GateSet.default_gate_set(radices)
        else:
            gateSet = GateSet(gateSet)

        self._gateSet = gateSet

    @property
    def position_graph(self) -> PositionGraph:
        return self._pg
    
    @property
    def gateSet(self) -> GateSet:
        return self._gateSet
    
    @property
    def logical_to_position(self) -> np.ndarray:
        return self._logical_to_position
    
    @property
    def position_to_logical(self) -> np.ndarray:
        return self._position_to_logical
    
    
    def set_qudit_position(self, qudit_id: int, pos_id: int) -> None:
        if qudit_id < 0 or qudit_id >= self.num_qudits:
            raise IndexError("Invalid qudit index")
        if pos_id < 0 or pos_id >= self.num_pos:
            raise IndexError("Invalid position index")

        current_owner = self.position_to_logical[pos_id]
        if current_owner != -1 and current_owner != qudit_id:
            raise RuntimeError(
                f"Position {pos_id} already occupied by qudit {current_owner}"
            )

        old_position = self.logical_to_position[qudit_id]
        if old_position != -1:
            self.position_to_logical[old_position] = -1

        self.logical_to_position[qudit_id] = pos_id
        self.position_to_logical[pos_id] = qudit_id


    def is_fully_connected(self) -> bool:
        return True #temp value

    def assert_consistent(self) -> None:
        for q, p in enumerate(self.logical_to_position):
            if p == -1:
                continue
            if self.position_to_logical[p] != q:
                raise RuntimeError(
                    f"Inconsistent mapping: logical {q} → physical {p}, "
                    f"but physical {p} → logical {self.position_to_logical[p]}"
                )

        if len([p for p in self.logical_to_position if p != -1]) != \
        len(set(p for p in self.logical_to_position if p != -1)):
            raise RuntimeError("Duplicate physical occupancy detected")
    
    def get_position_of_qudit(self, qudit_id: int) -> int:
        return self.logical_to_position[qudit_id]
    
    def get_logical_qudit_at_position(self, pos_id: int) -> int:
        return self.position_to_logical[pos_id]
    
    def to_coupling_graph(self) -> CouplingGraph:
        edges = []

        for q1 in range(self.num_qudits):
            p1 = self.logical_to_position[q1]
            if p1 == -1:
                continue

            for q2 in range(self.num_qudits):
                if q1 == q2:
                    continue

                p2 = self.logical_to_position[q2]
                if p2 == -1:
                    continue

                if self._pg.gate_is_executable(p1, p2):
                    edges.append((q1, q2))

        return CouplingGraph(edges)
    
    def swap_logical_qudits(self, qudit_1: int, qudit_2: int) -> None:
        pos_1 = self.logical_to_position[qudit_1]
        pos_2 = self.logical_to_position[qudit_2]
                
        if pos_1 == -1 or pos_2 == -1:
            raise RuntimeError("Cannot swap unplaced qudits")

        if qudit_1 == qudit_2:
            raise RuntimeError("Attempted to swap qudit with itself")

        # swap logical → physical
        self.logical_to_position[qudit_1], self.logical_to_position[qudit_2] = pos_2, pos_1

        # swap physical → logical 
        self.position_to_logical[pos_1], self.position_to_logical[pos_2] = qudit_2, qudit_1

    def __str__(self):
        return (
            f"logical→pos {self._logical_to_position}\n"
            f"pos→logical {self._position_to_logical}"
        )
    
    def record_state(self) -> None:
        snapshot = {
            "positions": self.logical_to_position.copy(),
        }
        self.history.append(snapshot)

    def in_cluster(self, qudits: List[int]) -> bool:
        positions = [self.logical_to_position[q] for q in qudits if self.logical_to_position[q] != -1]
        return self._pg.in_cluster(positions)
    
    def apply_perm(self, perm: Sequence[int]) -> None:
        """
        Apply a permutation of physical positions.

        perm[p_old] = p_new
        """
        if len(perm) != self.num_pos:
            raise ValueError("Permutation length must equal number of positions.")

        old_pos_to_log = self._position_to_logical.copy()

        # Reset mappings
        self._position_to_logical[:] = -1
        self._logical_to_position[:] = -1

        # Apply permutation
        for old_pos, new_pos in enumerate(perm):
            logical = old_pos_to_log[old_pos]
            if logical != -1:
                self._position_to_logical[new_pos] = logical
                self._logical_to_position[logical] = new_pos

        self.assert_consistent()


