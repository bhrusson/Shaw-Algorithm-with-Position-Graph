import numpy as np
import rustworkx as rx
from rustworkx import AllPairsPathMapping
from enum import IntFlag
from typing import Tuple, List, Sequence, Mapping, Dict, Callable, Optional
from dataclasses import dataclass

#First I am defining classes for the labels

class PositionCapability(IntFlag):
    NONE = 0
    EXECUTE = 0b001
    MEASURE = 0b010
    STARTING = 0b100

@dataclass (frozen=True)
class PositionLabel:
    capability: int
    weights: Dict[PositionCapability, float]
    
    def has_capability(self, capability: PositionCapability) -> bool:
        if capability == PositionCapability.NONE:
            return self.capability == PositionCapability.NONE.value
        return bool(self.capability & capability)
         
    def get_weight(self, capability: PositionCapability) -> float:
        if not self.has_capability(capability):
            raise ValueError(f"Capability {capability.name} not present in this position")
        return self.weights.get(capability, float('inf'))

class EdgeCapability(IntFlag):
    NONE = 0
    MOVE = 0b001
    SWAP = 0b010
    EXECUTE = 0b100

@dataclass (frozen=True)
class EdgeLabel:
    capability: int
    weights: Dict[EdgeCapability, float]    #These should match the EdgeCapability values, so 1:0.2 , 
                                            #2:0.5 , 4:0.7 would be a weight of 0.2 on MOVE, 0.5
                                            #on SWAP etc. 
    
    #bitwise and
    def has_capability(self, capability: EdgeCapability) -> bool:
        if capability == EdgeCapability.NONE:
            return self.capability == EdgeCapability.NONE.value
        return bool(self.capability & capability)

     
    def get_weight(self, capability: EdgeCapability) -> float:
        if not self.has_capability(capability):
            raise ValueError(f"Capability {capability.name} not present in this edge")
        return self.weights.get(capability, float('inf'))


#This is the main class that uses the others

class PositionGraph:
    def __init__(
            self,
            pos_labels: Sequence[PositionLabel], #Length of this is the number of Positions availble for qudits
            edge_labels: Mapping[Tuple[int, int], EdgeLabel] #Key value type of information, use as a dictionary
        ) -> None:
        self._pos_labels = list(pos_labels)
        self._edge_labels = dict(edge_labels)

        self._graph = rx.PyDiGraph()
        self._graph.add_nodes_from(self._pos_labels)
        self._graph.add_edges_from([(u, v, lbl) for (u, v), lbl in self._edge_labels.items()])

        self._executable_clusters = self.executable_clusters2()
        self._dijkstra_shortest_path_lengths = self.digraph_all_pairs_dijkstra_path_lengths(edge_capability=EdgeCapability.MOVE)
        self._dijkstra_shortest_paths = self.all_pairs_dijkstra_shortest_paths(edge_capability=EdgeCapability.MOVE)
        #move_graph = self.get_projected_graph(EdgeCapability.MOVE)
        #execute_graph = self.get_projected_graph(EdgeCapability.EXECUTE)

    def __str__(self) -> str:

        output = "_pos_labels" + str(self._pos_labels)  + "\n" + "_edge_labels" + str(self._edge_labels) + "\n" 
        return output


    def check_pos_index(self, index: int) -> None:
        if (index < 0 or index >= len(self.position_labels)):
            raise ValueError(f"Invalid index: {index} \nValid range: 0 to {len(self.position_labels)-1}")
    
    @property
    def graph(self) -> rx.PyDiGraph:
        return self._graph
   
    @property
    def position_labels(self) -> list[PositionLabel]:
        return self._pos_labels
    
    @property
    def edge_labels(self) ->  Dict[Tuple[int, int], EdgeLabel]:
        return self._edge_labels
    
    @property
    def shortest_path_lengths(self) -> AllPairsPathMapping:
        return self._dijkstra_shortest_path_lengths
    
    @property
    def shortest_paths(self) -> AllPairsPathMapping:
        return self._dijkstra_shortest_paths
    
    @property
    def clusters(self) -> AllPairsPathMapping:
        return self._executable_clusters

    def position_label(self, pos_index: int) -> PositionLabel:
        self.check_pos_index(pos_index)
        return self.position_labels[pos_index]

    def position_has_capability(self, pos_index: int, capability: PositionCapability) -> bool:
        self.check_pos_index(pos_index)
        return self.position_labels[pos_index].has_capability(capability)

    @property
    def all_edge_labels(self) -> List[EdgeLabel]:
            return list(self._edge_labels.values())
    
    def edge_label(self, edge_index1: int, edge_index2: int) -> EdgeLabel:
            try:
                 return self.edge_labels[(edge_index1, edge_index2)]
            except KeyError:
                 raise KeyError(f"Edge ({edge_index1} -> {edge_index2}) not found.")           

    
    def positions_with_label(self, label: PositionLabel) -> List[int]:
            return [i for i, node_label in enumerate(self.position_labels) if node_label == label]

    
    def edges_with_label(self, label: EdgeLabel) -> List[Tuple[int, int]]:
            return [(u, v) for (u, v), edge_label in self.edge_labels.items() if edge_label == label]
   
    
    def subgraph_by_position_capability(self, position_capability: PositionCapability) -> rx.PyDiGraph:
        valid_nodes = [i for i, label in enumerate(self.position_labels) if label.has_capability(position_capability)]
        return self._graph.subgraph(valid_nodes)
        
 
    #I made this a lot simpler, instead of building a subgraph with new indices and removing things isntead I am
    #keeping all positions, but only preserving relevant edges
    def get_projected_graph(self, edge_capability: EdgeCapability, weight_filter: Callable[[EdgeLabel],bool] = None) -> rx.PyDiGraph:
        projected = rx.PyDiGraph()
        projected.add_nodes_from(self._graph.nodes())

        # Filter edges based on capability and optional weight filter
        for (u, v), label in self._edge_labels.items():
            if (label.has_capability(edge_capability) and (weight_filter is None or weight_filter(label))):
                projected.add_edge(u,v,label)
        
        return projected
        
    
    def get_valid_starting_positions(self) -> list[int]:
        return[
            index
            for index, label in enumerate(self.position_labels)
            if label.has_capability(PositionCapability.STARTING)
        ]
    
    def in_cluster(self, pos: Sequence[int]) -> bool:
        if not pos:
            return False

        cluster_idx = None

        for p in pos:
            found = None
            for i, cluster in enumerate(self._executable_clusters):
                if p in cluster:
                    found = i
                    break

            if found is None:
                return False  # position not in any cluster

            if cluster_idx is None:
                cluster_idx = found
            elif found != cluster_idx:
                return False  # positions span multiple clusters

        return True
    
    def executable_clusters3(self) -> Sequence[Sequence[int]]:
        clusters = []
        visited = set()

        for start_node in range(len(self.position_labels)):
            if start_node in visited:
                continue

            # BFS/DFS cluster build
            cluster = []
            stack = [start_node]
            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                cluster.append(node)

                for neighbor in self.graph.neighbors_undirected(node):
                    edge_label = self.edge_labels.get((node, neighbor))
                    reverse_edge_label = self.edge_labels.get((neighbor, node))

                    # must exist and satisfy EXECUTE + (MOVE or SWAP) + bidirectional EXECUTE
                    if not edge_label or not reverse_edge_label:
                        continue

                    if (
                        edge_label.has_capability(EdgeCapability.EXECUTE)
                        and reverse_edge_label.has_capability(EdgeCapability.EXECUTE)
                        and (
                            edge_label.has_capability(EdgeCapability.MOVE)
                            or edge_label.has_capability(EdgeCapability.SWAP)
                        )
                    ):
                        stack.append(neighbor)

            # check all position labels in cluster have EXECUTE
            if all(
                self.position_labels[node].has_capability(EdgeCapability.EXECUTE)
                for node in cluster
            ):
                clusters.append(cluster)

        return clusters
    


    def executable_clusters(
        self,
        move_weight_filter: Callable[[EdgeLabel], bool] = None,
        execute_weight_filter: Callable[[EdgeLabel], bool] = None
    ) -> Sequence[Sequence[int]]:
    
        move_graph = self.get_projected_graph(EdgeCapability.MOVE, move_weight_filter)
        exec_graph = self.get_projected_graph(EdgeCapability.EXECUTE, execute_weight_filter)
        execute_nodes = [i for i, label in enumerate(self.position_labels) 
                     if label.has_capability(PositionCapability.EXECUTE)]
        for g in [move_graph, exec_graph]:
            for i, label in enumerate(self.position_labels):
                if i not in execute_nodes:
                    try:
                        g.remove_node(i)
                    except IndexError:
                        continue

        clusters = []
        for component in rx.weakly_connected_components(move_graph):
            # Only keep those where the EXECUTE graph is fully connected (all nodes have EXECUTE edges to each other)
            subgraph_exec_edges = [(u, v) for u, v in exec_graph.edge_list() if u in component and v in component]

            if len(subgraph_exec_edges) > 0:  # at least one EXECUTE edge between them
                clusters.append(list(component))
        return clusters
    
    
    #Every position in a cluster can execute a gate with every/any other position.
    # 
    def executable_clusters2(
    self,
    move_weight_filter: Callable[[EdgeLabel], bool] = None,
    execute_weight_filter: Callable[[EdgeLabel], bool] = None
    ) -> Sequence[Sequence[int]]:

        move_graph = self.get_projected_graph(EdgeCapability.MOVE, move_weight_filter)
        exec_graph = self.get_projected_graph(EdgeCapability.EXECUTE, execute_weight_filter)

        execute_nodes = [i for i, label in enumerate(self.position_labels)
                        if label.has_capability(PositionCapability.EXECUTE)]

        filtered_move_edges = [(u, v) for u, v in move_graph.edge_list()
                            if u in execute_nodes and v in execute_nodes]

        # Build adjacency map
        adj = {i: set() for i in execute_nodes}
        for u, v in filtered_move_edges:
            adj[u].add(v)
            adj[v].add(u)  # for weak connectivity

        visited = set()
        clusters = []

        for node in execute_nodes:
            if node not in visited:
                stack = [node]
                component = set()
                while stack:
                    n = stack.pop()
                    if n not in visited:
                        visited.add(n)
                        component.add(n)
                        stack.extend(adj[n] - visited)

                # EXECUTE edges fully connect?
                subgraph_exec_edges = [(u, v) for u, v in exec_graph.edge_list()
                                    if u in component and v in component]
                n = len(component)
                if len(subgraph_exec_edges) == n * (n - 1):
                    clusters.append(sorted(component))

        return clusters

    # you can treat the return as a read-only mapping/dict.
    # of the form: {0: {1: [0, 2, 3, 1], 2: [0, 2]}}
    def all_pairs_dijkstra_shortest_paths(
        self,
        edge_capability: EdgeCapability = EdgeCapability.MOVE
    ) -> AllPairsPathMapping:
        """
        Compute all-pairs shortest paths using rustworkx, using the specified
        edge capability's weights as the cost metric.

        edge_capability: which EdgeCapability to query on each EdgeLabel
        """
        def edge_cost_fn(edge_label: EdgeLabel):
            # rustworkx will pass the edge data (we stored EdgeLabel as the edge data).
            # If an edge lacks the capability, return infinite cost so it's effectively ignored.
            try:
                if edge_label is None:
                    return float('inf')
                if not edge_label.has_capability(edge_capability):
                    return float('inf')
                return edge_label.get_weight(edge_capability)
            except Exception:
                return float('inf')

        return rx.all_pairs_dijkstra_shortest_paths(self._graph, edge_cost_fn)
    
    def digraph_all_pairs_dijkstra_path_lengths(
        self,
        edge_capability: EdgeCapability = EdgeCapability.MOVE
    ) -> AllPairsPathMapping:
        """
        Compute all-pairs shortest paths using rustworkx, using the specified
        edge capability's weights as the cost metric.

        edge_capability: which EdgeCapability to query on each EdgeLabel
        """
        def edge_cost_fn(edge_label: EdgeLabel):
            # rustworkx will pass the edge data (we stored EdgeLabel as the edge data).
            # If an edge lacks the capability, return infinite cost so it's effectively ignored.
            try:
                if edge_label is None:
                    return float('inf')
                if not edge_label.has_capability(edge_capability):
                    return float('inf')
                return edge_label.get_weight(edge_capability)
            except Exception:
                return float('inf')

        return rx.digraph_all_pairs_dijkstra_path_lengths(self._graph, edge_cost_fn)
    
    #this is direcitonal, two qudit gates only right now
    # see also the method: in_cluster()
    def gate_is_executable(self,edge_index_1,edge_index2) -> bool:
        return self.edge_label(edge_index_1,edge_index2).has_capability(EdgeCapability.EXECUTE)
    


    #This will return a mapping from each executable cluster to the list of positions that can reach it via MOVE edges
    def connected_to_executable_clusters(
        self,
        move_weight_filter: Callable[[EdgeLabel],bool] = None,
        execute_weight_filter: Callable[[EdgeLabel],bool] = None
    ) -> Mapping[int,Sequence[int]]:
        move_graph = self.get_projected_graph(EdgeCapability.MOVE,move_weight_filter)
        execute_graph = self.get_projected_graph(EdgeCapability.EXECUTE,execute_weight_filter)
        execute_positions = [
            i for i, label in enumerate(self.position_labels)
            if label.has_capability(PositionCapability.EXECUTE)
        ]
        move_components = list(rx.weakly_connected_components(move_graph))
        cluster_map: dict[int, list[int]] = {}
        cluster_idx = 0
        for move_component in move_components:
            exec_nodes = [n for n in move_component if n in execute_positions]
            if not exec_nodes:
                continue
            exec_subgraph = execute_graph.subgraph(exec_nodes)
            exec_clusters = list(rx.weakly_connected_components(exec_subgraph))

            for exec_cluster in exec_clusters:
                connected_nodes = set()
                undirected_move = move_graph.to_undirected()

                for node in move_component:
                    for target in exec_cluster:
                        if rx.has_path(undirected_move, node, target):
                            connected_nodes.add(node)
                            break  # Stop after first reachable target

                cluster_map[cluster_idx] = sorted(list(connected_nodes))
                cluster_idx += 1
        return cluster_map
    


    def nearest_cluster(self, pos_id: int, clusters: Sequence[Sequence[int]]) -> Optional[Tuple[List[int], float]]:
        move_graph = self.get_projected_graph(EdgeCapability.MOVE)
        self.check_pos_index(pos_id)

        min_distance = float("inf")
        nearest = None

        # Compute all distances once
        distances = rx.digraph_dijkstra_shortest_path_lengths(
            move_graph,
            pos_id,
            edge_cost_fn=lambda edge: edge.get_weight(EdgeCapability.MOVE),
        )

        for cluster in clusters:
            # Find the closest reachable node in this cluster
            cluster_dists = [distances[t] for t in cluster if t in distances]
            if cluster_dists:
                dist = min(cluster_dists)
                if dist < min_distance:
                    min_distance = dist
                    nearest = cluster

        if nearest is None:
            return None  # no reachable cluster
        return nearest, min_distance


    def shortest_path(self,start: int,target: int,edge_capability: EdgeCapability = EdgeCapability.MOVE) -> Optional[Tuple[List[int], float]]:
        self.check_pos_index(start)
        self.check_pos_index(target)

        graph = self.get_projected_graph(edge_capability)

        try:
            # Compute Dijkstra shortest path lengths and predecessors
            lengths, predecessors = rx.digraph_dijkstra_shortest_path_lengths(
                graph,
                start,
                edge_cost_fn=lambda edge: edge.get_weight(edge_capability),
                return_predecessors=True
            )
        except Exception as e:
            raise RuntimeError(f"Failed to compute shortest path: {e}")

        if target not in lengths:
            # No path exists
            return None

        # Reconstruct the path from predecessors
        path = []
        current = target
        while current != start:
            path.append(current)
            current = predecessors.get(current)
        path.append(start)
        path.reverse()

        return path, lengths[target]
    
    from typing import Dict, Tuple, List, Optional

    def get_shortest_path_tree(
        self,
        qudit_pos: int,
        edge_capability: EdgeCapability = EdgeCapability.MOVE
    ) -> List[Tuple[int, ...]]:
        """
        Compute the shortest-path tree from a given qudit position.

        Returns:
            A list of tuples, where each tuple is the path from qudit_pos
            to a reachable position in the graph. The index in the list
            corresponds to the position index.
            - If a position is unreachable, the tuple is empty.
        """
        self.check_pos_index(qudit_pos)
        graph = self.get_projected_graph(edge_capability)

        # Dijkstra shortest paths with predecessors
        lengths, predecessors = rx.digraph_dijkstra_shortest_path_lengths(
            graph,
            qudit_pos,
            edge_cost_fn=lambda edge: edge.get_weight(edge_capability),
            return_predecessors=True
        )

        paths: List[Tuple[int, ...]] = []

        for node_index in range(len(self.position_labels)):
            if node_index not in lengths:
                # Unreachable
                paths.append(())
                continue

            # Reconstruct path from predecessors
            path = []
            current = node_index
            while current != qudit_pos:
                path.append(current)
                current = predecessors[current]
            path.append(qudit_pos)
            path.reverse()
            paths.append(tuple(path))

        return paths


        
    
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
     
