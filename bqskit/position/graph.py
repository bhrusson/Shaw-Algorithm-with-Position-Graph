import numpy as np
import rustworkx as rx
from enum import Enum
from typing import Tuple, List, Sequence, Mapping
from dataclass import dataclass

#   Information to be represented:
#   number of qudits - 
#   radices of quidits (list of integers where the index of an integer represents the basis or radix of a qudit.  radices[4] == 3 implies qudit 4 is a qutrit
#   How the qudits are connected - What types of different connections are there? i.e. swap, move, - defines what types of operations could be possible
#   abilities of each qubits, (be measured, executed, storage,)
#   adjacenecy graph, adjacency 

#   rustworkx

class PositionCapability(Enum):
    NONE = 0
    EXECUTE = 0b001
    MEASURE = 0b010
    STARTING = 0b100

@dataclass
class VertexLabel:
    capability: int

    def has_capability(self, capability: PositionCapability) -> bool:
        return self.capability & capability != 0


# TODO (BRENT): Read dataclass documentation, update the other labels accordingly
# Combine EdgeCapabilities into one (Execute doesn't need to be separate from moovement (in implmentation))

class EdgeMovementLabel(Enum):
    NONE = 0
    MOVE = 1
    SWAP = 2
    MOVE_SWAP = 3

class EdgeExecutionLabel(Enum):
    NONE = 0
    EXECUTE = 1

EdgeLabel = Tuple[EdgeMovementLabel, EdgeExecutionLabel]


class PositionGraph:
    def __init__(
            self,
            radices: Sequence[int],
            node_labels: Sequence[VertexLabel], #Length of this is the number of Positions availble for qudits
            connection_labels: Mapping[Tuple[int, int], EdgeLabel] #Key value type of information, use as a dictionary
        ) -> None:
        self.radices = list(radices)
        self.node_labels = list(node_labels)
        self.connection_labels = connection_labels

        self.graph = rx.PyDiGraph()
        self.graph.add_nodes_from(node_labels)
        self.graph.add_edges_from(list(connection_labels.items()))

    def check_node_index(self, index: int) -> None:
        if (index < 0 or index >= len(self.node_labels)):
            raise ValueError(f"Invalid index: {index} \nValid range: 0 to {len(self.node_labels)-1}")

    @property
    def get_num_qudits(self) -> int:
        return len(self.radices)
   
    @property
    def radices(self) -> list[int]:
         return self.radices

    @property
    def position_labels(self) -> list[VertexLabel]:
        return self.node_labels

    def position_label(self, node_index: int) -> VertexLabel:
        self.check_node_index(node_index)
        return self.node_labels[node_index]

    # TODO: Brent Calling code should use the position vocabulary not the node vocab...
    def position_has_capability(self, node_index: int, capability: PositionCapability) -> bool:
        pass

    # todo like above
    def get_all_edge_labels(self) -> List[EdgeLabel]:
            return [label for _, _, label in self.connection_labels]

    # todo like above
    def get_edge_label(self, edge_index1: int, edge_index2: int) -> EdgeLabel:
            # Find the edge in connection_labels
            for u, v, label in self.connection_labels:
                if u == edge_index1 and v == edge_index2:
                    return label
            raise KeyError(f"Edge ({edge_index1} -> {edge_index2}) not found.")

    # get nodes with capabilities
    def get_nodes_with_label(self, label: VertexLabel) -> List[int]:
            return [i for i, node_label in enumerate(self.node_labels) if node_label == label]

    # this is the projected graph thingy
    def get_edges_with_label(self, label: EdgeLabel) -> List[Tuple[int, int]]:
            return [(u, v) for u, v, edge_label in self.connection_labels if edge_label == label]
   
    # position
    def get_neighbors(self, node_index: int) -> List[int]:
         return self.graph.neighbors(node_index)
    
    def get_successor(self, node_index: int) -> List[int]:
         return self.graph.successors(node_index)
    
    def get_predecessor_indices(self, node_index: int) -> List[int]:
         return self.graph.predecessor_indices(node_index)
    
    def get_predecessors(self, node_index: int) -> List[int]:
         return self.graph.predecessors(node_index)
    
    def get_ancestors(self, node_index: int) -> List[int]:
         return rx.ancestors(self.graph,node_index)
    
    def get_descendents(self, node_index: int) -> List[int]:
         return self.graph.predecessors(node_index)
    
    def find_successors_by_edge(self, node_index: int, edge_filter_function: function) -> List[int]:
         return self.graph.find_successors_by_edge(node_index,edge_filter_function)
    
    def can_execute_node(self, node_index: int) -> bool:
         return (self.get_node_label(node_index).value == 1 or self.get_node_label(node_index).value == 3)
    
    def can_measure_node(self, node_index: int) -> bool:
         return (self.get_node_label(node_index).value == 2 or self.get_node_label(node_index).value == 3)
    
    def can_execute_edge(self, edge_index1: int, edge_index2: int) -> bool:
        return self.get_edge_label(edge_index1,edge_index2)[1].value == 1
       
    def can_move_edge(self, edge_index1: int, edge_index2: int) -> bool:
        return (self.get_edge_label(edge_index1,edge_index2)[0].value == 2 or self.get_edge_label(edge_index1,edge_index2)[0].value == 3)
    
    #not finished
    def can_move(self, src: int, dst: int) -> bool:
        self.check_node_index(src)
        self.check_node_index(dst)
        label = self.connection_labels.get((src, dst))
        return label is not None and label[0] in [EdgeMovementLabel.MOVE, EdgeMovementLabel.MOVE_SWAP]

    #not finished
    def can_swap(self, src: int, dst: int) -> bool:
        self.check_node_index(src)
        self.check_node_index(dst)
        label = self.connection_labels.get((src, dst))
        return label is not None and label[0] in [EdgeMovementLabel.SWAP, EdgeMovementLabel.MOVE_SWAP]
    
    #not finished
    def path_to(self, src: int, dst: int) -> List[int]:
        self.check_node_index(src)
        self.check_node_index(dst)
        path = rx.digraph_dijkstra_shortest_paths(self.graph, src)[0].get(dst)
        if path is None:
            raise ValueError(f"No path from {src} to {dst}")
        return path

    def get_projected_graph(self, edge_capability: TODO) -> list[Edge]:
        """TODO: BRENT"""
        pass
    
    def get_valid_starting_position(self) -> list[int]:
        """TODO: not all postions are valid starting positions for physical qubits. this should return the valid ones."""
        pass

    
    #Can functions, Can move from 1 position to another, How to move from 1 pos to another, 
    # We need to be able to "reason about" clusterts of executable gates. Succcinct set of function calls that allows us to work with this concept
    # i.e. the concept of a trap in ions is a cluster of nodes with the executable label. 
    # We want to find those clusters, find the nearest, etc. findnearestfrom(index) vs findNearestEmpty()(state)
    # Potentially have the postiongraphState to have an instance of the positionGraph. 

    # 
    #



# For this PositionGraphState I want to show the current state of the mapping of qudits to their available positions.
# I want to return the specific position of any specific qudit
# I want to to return the state of any specific position

#key value, logicial qubits to physical positions
#
#
     
