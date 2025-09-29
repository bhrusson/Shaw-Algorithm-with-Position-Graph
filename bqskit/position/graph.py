import numpy as np
import rustworkx as rx
from enum import Enum
from typing import Tuple, List, Sequence, Mapping, Dict, Callable, Optional
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

    def check_pos_index(self, index: int) -> None:
        if (index < 0 or index >= len(self.position_labels)):
            raise ValueError(f"Invalid index: {index} \nValid range: 0 to {len(self.position_labels)-1}")

    @property
    def num_qudits(self) -> int:
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
        self.check_pos_index(pos_index)
        return self.position_labels[pos_index]

    # TODO: Brent Calling code should use the position vocabulary not the node vocab...
    # Brent - I have updated this vocabulary
    def position_has_capability(self, pos_index: int, capability: PositionCapability) -> bool:
        self.check_pos_index(pos_index)
        return self.position_labels[pos_index].has_capability(capability)

    @property
    def all_edge_labels(self) -> List[EdgeLabel]:
            return [label for _, _, label in self.edge_labels]

   
    def edge_label(self, edge_index1: int, edge_index2: int) -> EdgeLabel:
            try:
                 return self.edge_labels[(edge_index1, edge_index2)]
            except KeyError:
                 raise KeyError(f"Edge ({edge_index1} -> {edge_index2}) not found.")

    
    def positions_with_label(self, label: PositionLabel) -> List[int]:
            return [i for i, node_label in enumerate(self.position_labels) if node_label == label]

    
    def edges_with_label(self, label: EdgeLabel) -> List[Tuple[int, int]]:
            return [(u, v) for u, v, edge_label in self.edge_labels if edge_label == label]
   
    
    def subgraph_by_position_capibility(self, position_capability: PositionCapability) -> rx.PyDiGraph:
        valid_nodes = [i for i, label in enumerate(self.position_labels) if label.has_capability(position_capability)]
        return self._graph.subgraph(valid_nodes)
        
 
    #I made this a lot simpler, instead of building a subgraph with new indices and removing things isntead I am
    #keeping all positions, but only preserving relevant edges
    def get_projected_graph(self, edge_capability: EdgeCapability, weight_filter: Callable[[EdgeLabel],bool] = None) -> rx.PyDiGraph:
        projected = rx.PyDiGraph()
        projected.add_nodes_from(self._graph.nodes())

        # Filter edges based on capability and optional weight filter
        for (u, v), label in self._edge_labels.items():
            if label.has_capability(edge_capability):
                if weight_filter is None or weight_filter(label):
                    projected.add_edge(u,v,label)
        
        return projected
        
    
    def get_valid_starting_position(self) -> list[int]:
        return[
            index
            for index, label in enumerate(self.position_labels)
            if label.has_capability(PositionCapability.STARTING)
        ]

    #Trying to define an executable cluster exactly still. MOVE vs SWAP (needs both, either, directional?), fully connected, vs Strongly Connected, vs Weakly Connected
    #Edges in the cluster must be connected looking at the MOVE projected graph 
    #Positions also must have the EXECUTE label
    def executable_clusters(self, move_weight_filter: Callable[[EdgeLabel],bool] = None, execute_weight_filter: Callable[[EdgeLabel],bool] = None) -> list[list[int]]:
        move_graph = self.get_projected_graph(EdgeCapability.MOVE,move_weight_filter)
        execute_graph = self.get_projected_graph(EdgeCapability.EXECUTE,execute_weight_filter)
        execute_positions = [
            i for i, label in enumerate(self.position_labels)
            if label.has_capability(PositionCapability.EXECUTE)
        ]
        clusters = []
        for component in rx.strongly_connected_components(move_graph):    
            pos = [p for p in component if p in execute_positions]
            if not pos:
                continue

            subgraph = execute_graph.subgraph(pos)
            n = len(pos)
            # Fully connected directed subgraph: n*(n-1) edges
            if subgraph.num_edges() == n * (n - 1):
                clusters.append(pos)
        
        return clusters
        
    #This is less strict on being fully connected, the direction of execute edges are ignored.
    #I believe this will show if a position has a path to an executable cluster
    def connected_to_executable_clusters(self, move_weight_filter: Callable[[EdgeLabel],bool] = None, execute_weight_filter: Callable[[EdgeLabel],bool] = None) -> list[list[int]]:
        move_graph = self.get_projected_graph(EdgeCapability.MOVE,move_weight_filter)
        execute_graph = self.get_projected_graph(EdgeCapability.EXECUTE,execute_weight_filter)
        execute_positions = [
            i for i, label in enumerate(self.position_labels)
            if label.has_capability(PositionCapability.EXECUTE)
        ]
        clusters = []

        for component in rx.strongly_connected_components(move_graph):
            pos = [n for n in component if n in execute_positions]
            if not pos:
                continue 

            execute_subgraph = execute_graph.subgraph(pos)
            if len(list(rx.weakly_connected_components(execute_subgraph))) == 1:
                clusters.append(pos)

        return clusters
    
    def nearest_cluster(self, pos_id: int, clusters: Sequence[Sequence[int]]) -> Optional[List[int]]:
        move_graph = self.get_projected_graph(EdgeCapability.MOVE)

        min_distance = float('inf')
        nearest = None

        for cluster in clusters:
            # Compute distance to the nearest node in this cluster
            for target in cluster:
                try:
                    distance = rx.digraph_dijkstra_shortest_path_lengths(move_graph, pos_id)[target] #Do I need to add an optional edge cost fn?
                    if distance < min_distance:
                        min_distance = distance
                        nearest = cluster
                except KeyError:
                    # No path to this target node
                    continue

        return nearest
    
    #Can functions, Can move from 1 position to another, How to move from 1 pos to another, 
    # We need to be able to "reason about" clusterts of executable gates. Succcinct set of function calls that allows us to work with this concept
    # i.e. the concept of a trap in ions is a cluster of nodes with the executable label. 
    # We want to find those clusters, find the nearest, etc. findnearestfrom(index) vs findNearestEmpty()(state)
    # Potentially have the postiongraphState to have an instance of the positionGraph. 

    # A set of nodes, all connected, Also fully connected in the execution projected_grpah/subgraph. 
    # Movement projected grpah looks at only edges that allow move/swap. The sets of nodes must be connected in that graph

    #If I look at only the nodes with execute edges, this set of nodes also needs to be conencted. 

    #The node also needs to have the executable position label



# For this PositionGraphState I want to show the current state of the mapping of qudits to their available positions.
# I want to return the specific position of any specific qudit
# I want to to return the state of any specific position

#key value, logicial qubits to physical positions
#
#
     
