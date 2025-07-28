import numpy as np
import rustworkx as rx
from enum import Enum
from typing import Tuple, List, Sequence, Mapping, Dict, Callable
from dataclasses import dataclass

class PositionCapability(Enum):
    NONE = 0
    EXECUTE = 0b001
    MEASURE = 0b010
    STARTING = 0b100

@dataclass
class PositionLabel:
    capability: int
    weights: Dict[int, float]


    def has_capability(self, capability: PositionCapability) -> bool:
        return self.capability & capability != 0
    
    def get_weight(self, capability: PositionCapability) -> float:
        if not self.has_capability(capability):
            raise ValueError(f"Capability {capability.name} not present in this edge")
        return self.weights.get(capability, float('inf')) 

# TODO (BRENT): Read dataclass documentation, update the other labels accordingly
# Combine EdgeCapabilities into one (Execute doesn't need to be separate from moovement (in implmentation))
#
# Brent - I have attempted to do this.

class EdgeCapability(Enum):
    NONE = 0
    MOVE = 0b001
    SWAP = 0b010
    EXECUTE = 0b100

@dataclass
class EdgeLabel:
    capability: int
    weights: Dict[int, float]

    def has_capability(self, capability: EdgeCapability) -> bool:
          return self.capability & capability != 0
     
    def get_weight(self, capability: EdgeCapability) -> float:
        if not self.has_capability(capability):
            raise ValueError(f"Capability {capability.name} not present in this edge")
        return self.weights.get(capability, float('inf'))

class PositionGraph:
    def __init__(
            self,
            radices: Sequence[int], #Length of this is the number of qudits
            pos_labels: Sequence[PositionLabel], #Length of this is the number of Positions availble for qudits
            edge_labels: Mapping[Tuple[int, int], EdgeLabel] #Key value type of information, use as a dictionary
        ) -> None:
        self._radices = list(radices)
        self._pos_labels = list(pos_labels)
        self._edge_labels = edge_labels

        self._graph = rx.PyDiGraph()
        self._graph.add_nodes_from(pos_labels)
        self._graph.add_edges_from(list(edge_labels.items()))

    def check_node_index(self, index: int) -> None:
        if (index < 0 or index >= len(self.pos_labels)):
            raise ValueError(f"Invalid index: {index} \nValid range: 0 to {len(self.pos_labels)-1}")

    @property
    def get_num_qudits(self) -> int:
        return len(self._radices)
    
    @property
    def graph(self) -> rx.PyDiGraph:
        return self._graph
   
    @property
    def radices(self) -> list[int]:
         return self._radices

    @property
    def position_labels(self) -> list[PositionLabel]:
        return self._pos_labels
    
    @property
    def edge_labels(self) ->  Dict[Tuple[int, int], EdgeLabel]:
        return self._edge_labels

    def position_label(self, pos_index: int) -> PositionLabel:
        self.check_node_index(pos_index)
        return self.position_labels[pos_index]

    # TODO: Brent Calling code should use the position vocabulary not the node vocab...
    def position_has_capability(self, pos_index: int, capability: PositionCapability) -> bool:
        self.check_node_index(pos_index)
        return self.position_labels[pos_index].has_capability(capability)


    # todo like above
    def get_all_edge_labels(self) -> List[EdgeLabel]:
            return [label for _, _, label in self.edge_labels]

    # todo like above
    def get_edge_label(self, edge_index1: int, edge_index2: int) -> EdgeLabel:
            try:
                 return self.edge_labels[(edge_index1, edge_index2)]
            except KeyError:
                 raise KeyError(f"Edge ({edge_index1} -> {edge_index2}) not found.")

    # get nodes with capabilities
    def get_positions_with_label(self, label: PositionLabel) -> List[int]:
            return [i for i, node_label in enumerate(self.position_labels) if node_label == label]

    # this is the projected graph thingy
    def get_edges_with_label(self, label: EdgeLabel) -> List[Tuple[int, int]]:
            return [(u, v) for u, v, edge_label in self.edge_labels if edge_label == label]
   
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
    
    def get_descendants(self, node_index: int) -> List[int]:
         return  rx.descendants(self.graph, node_index)
    
    def find_successors_by_edge(self, node_index: int, edge_filter: Callable) -> List[int]:
         return self.graph.find_successors_by_edge(node_index,edge_filter)
    
    def can_execute_position(self, node_index: int) -> bool:
         return self.position_label(node_index).has_capability(PositionCapability.EXECUTE)
    
    def can_measure_position(self, node_index: int) -> bool:
         return self.position_label(node_index).has_capability(PositionCapability.MEASURE)
    
    def can_be_starting_position(self, node_index: int) -> bool:
         return self.position_label(node_index).has_capability(PositionCapability.STARTING)
    
    def can_execute_edge(self, edge_index1: int, edge_index2: int) -> bool:
        return self.get_edge_label(edge_index1,edge_index2).has_capability(EdgeCapability.EXECUTE)
       
    def can_move_edge(self, edge_index1: int, edge_index2: int) -> bool:
        return self.get_edge_label(edge_index1,edge_index2).has_capability(EdgeCapability.MOVE)
    
    def can_swap_edge(self, edge_index1: int, edge_index2: int) -> bool:
        return self.get_edge_label(edge_index1,edge_index2).has_capability(EdgeCapability.SWAP)
    

    def get_subgraph_by_position_capibility(self, position_capability: PositionCapability) -> rx.PyDiGraph:
        valid_nodes = [i for i, label in enumerate(self.position_labels) if label.has_capability(position_capability)]
        return self._graph.subgraph(valid_nodes)
        
 
    def get_projected_graph(self, edge_capability: EdgeCapability, weight_filter: Callable[[EdgeLabel],bool] = None) -> rx.PyDiGraph:
        # Filter edges based on capability and optional weight filter
        edges_to_include = []
        for (u, v), label in self._edge_labels.items():
            if label.has_capability(edge_capability):
                if weight_filter is None or weight_filter(label):
                    edges_to_include.append((u, v))
        
        # Get all nodes connected by these edges (to avoid isolated nodes)
        nodes_to_include = set()
        for u, v in edges_to_include:
            nodes_to_include.add(u)
            nodes_to_include.add(v)
        
        # Use rustworkx's subgraph method: it takes a list of node indices
        subgraph = self._graph.subgraph(list(nodes_to_include))
        
        # Now we need to filter edges within subgraph to only those edges we want,
        # because subgraph keeps *all* edges between those nodes.
        # So let's remove edges that don't match capability and weight_filter:
        edges_to_remove = []
        for edge in subgraph.edge_list():
            u, v, _ = edge
            label = self._edge_labels.get((u, v))
            if label is None or not label.has_capability(edge_capability) or (weight_filter and not weight_filter(label)):
                edges_to_remove.append((u, v))
        
        for u, v in edges_to_remove:
            subgraph.remove_edge(subgraph.find_edge(u, v))
        
        return subgraph
    
    def get_valid_starting_position(self) -> list[int]:
        return[
            index
            for index, label in enumerate(self.position_labels)
            if label.has_capability(PositionCapability.STARTING)
        ]

    
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
     
