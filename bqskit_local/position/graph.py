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
    capability: PositionCapability
    weights: Dict[PositionCapability, float]
    
    def has_capability(self, capability: PositionCapability) -> bool:
        assert isinstance(capability, PositionCapability)
        if capability == PositionCapability.NONE:
            return self.capability == PositionCapability.NONE
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
    capability: EdgeCapability
    weights: Dict[EdgeCapability, float]    #These should match the EdgeCapability values, so 1:0.2 , 
                                            #2:0.5 , 4:0.7 would be a weight of 0.2 on MOVE, 0.5
                                            #on SWAP etc. 
    
    #bitwise and
    def has_capability(self, capability: EdgeCapability) -> bool:
        assert isinstance(capability, EdgeCapability)
        if capability == EdgeCapability.NONE:
            return self.capability == EdgeCapability.NONE
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
            edge_labels: Mapping[Tuple[int, int], EdgeLabel], #Key value type of information, use as a dictionary
        ) -> None:

        self._pos_labels = list(pos_labels)
        self._edge_labels = dict(edge_labels)

        self._graph = rx.PyDiGraph()
        self._graph.add_nodes_from(self._pos_labels)
        self._graph.add_edges_from([(u, v, lbl) for (u, v), lbl in self._edge_labels.items()])


        
        self._move_graph = self.get_projected_graph(EdgeCapability.MOVE)
        self._execute_graph = self.get_projected_graph(EdgeCapability.EXECUTE)
        self._executable_clusters = self.fully_executable_clusters()
        self._dijkstra_shortest_path_lengths = self.digraph_all_pairs_dijkstra_path_lengths(edge_capability=EdgeCapability.MOVE)
        self._dijkstra_shortest_paths = self.all_pairs_dijkstra_shortest_paths(edge_capability=EdgeCapability.MOVE)
        self._shortest_path_hops_tree = self.all_pairs_hop_shortest_path_trees()
        self._move_cost_matrix = self.build_move_cost_matrix()
        self._cluster_distance_maps = self.build_cluster_distance_map()
        self._move_gradient = self.build_move_gradient()
        self._swap_neighbors = self._build_swap_neighbors()


    def __str__(self) -> str:

        output = "_pos_labels" + str(self._pos_labels)  + "\n" + "_edge_labels" + str(self._edge_labels) + "\n" 
        return output


    def check_pos_index(self, index: int) -> None:
        if (index < 0 or index >= len(self._pos_labels)):
            raise ValueError(f"Invalid index: {index} \nValid range: 0 to {len(self._pos_labels)-1}")
    
    @property
    def move_cost_matrix(self) -> np.ndarray:
        return self._move_cost_matrix
    
    @property
    def execute_graph(self) -> rx.PyDiGraph:
        return self._execute_graph
    
    @property
    def swap_neighbors(self) -> dict[int, tuple[int, ...]]:
        return self._swap_neighbors
    
    @property
    def move_graph(self) -> rx.PyDiGraph:
        return self._move_graph

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
    def shortest_path_hops_tree(self) -> Dict[int, Dict[int, Tuple[int, ...]]]:
        return self._shortest_path_hops_tree
    
    @property
    def clusters(self) -> Sequence[Sequence[int]]:
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
    
    def edge_label(self, u: int, v: int) -> EdgeLabel:
        if (u, v) in self.edge_labels:
            return self.edge_labels[(u, v)]
        raise KeyError(f"Edge ({u}->{v}) not found.")
    
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
    
    def get_swap_neighbors(self, pos: int) -> tuple[int, ...]:
        return self._swap_neighbors[pos]
    
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
    
    def locally_executable_regions(self) -> Sequence[Sequence[int]]:
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
                self.position_labels[node].has_capability(PositionCapability.EXECUTE)
                for node in cluster
            ):
                clusters.append(cluster)

        return clusters
    


    def move_connected_exec_components(
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
    def fully_executable_clusters(
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

                fully_connected = True
                for u in component:
                    for v in component:
                        if u != v and not exec_graph.has_edge(u, v):
                            fully_connected = False
                            break
                    if not fully_connected:
                        break

                if fully_connected:
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

    def all_pairs_hop_shortest_path_trees(self) -> Dict[int, Dict[int, Tuple[int, ...]]]:
        """
        Compute all-pairs shortest paths by hop-count using CG-style tie-breaks.

        This intentionally mirrors the legacy CouplingGraph shortest-path tree
        logic: source-by-source traversal, unit-cost relaxations, and no path
        replacement on equal-length ties. The weighted Dijkstra caches remain
        unchanged for experiments that use MOVE weights directly.
        """
        all_paths: Dict[int, Dict[int, Tuple[int, ...]]] = {}
        for source in range(len(self.position_labels)):
            source_paths = self.get_hop_shortest_path_tree(source)
            all_paths[source] = {
                target: path
                for target, path in enumerate(source_paths)
                if len(path) > 0
            }

        return all_paths

    def get_hop_shortest_path_tree(self, source: int) -> List[Tuple[int, ...]]:
        """
        Return hop-shortest paths from `source` using legacy CG tie-breaking.

        This mirrors CouplingGraph.get_shortest_path_tree by:
        - tracking unvisited nodes,
        - selecting the next node by shortest hop distance,
        - relaxing neighbors with unit cost,
        - and preserving the first path found on equal-hop ties.
        """
        self.check_pos_index(source)

        unvisited_positions = set(range(len(self.position_labels)))
        distances = {i: np.inf for i in range(len(self.position_labels))}
        paths: List[Tuple[int, ...]] = [tuple() for _ in range(len(self.position_labels))]
        distances[source] = 0
        paths[source] = (source,)

        while len(unvisited_positions) > 0:
            unvisited_distances = [
                (node, dist)
                for node, dist in distances.items()
                if node in unvisited_positions
            ]
            unvisited_distances.sort(key=lambda item: item[1])
            current = unvisited_distances[0][0]

            if distances[current] == np.inf:
                break

            neighbors = set(self.move_graph.neighbors(current))
            unvisited_neighbors = unvisited_positions.intersection(neighbors)

            for neighbor in unvisited_neighbors:
                if distances[current] + 1 < distances[neighbor]:
                    distances[neighbor] = distances[current] + 1
                    paths[neighbor] = paths[current] + (neighbor,)

            unvisited_positions.remove(current)

        return paths
    
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
        move_graph = self.move_graph
        self.check_pos_index(pos_id)

        min_distance = float("inf")
        nearest = None

        # Compute all distances once
        distances = rx.digraph_dijkstra_shortest_path_lengths(
            move_graph,
            pos_id,
            edge_cost_fn=lambda edge: (
            edge.get_weight(EdgeCapability.MOVE)
            if edge and edge.has_capability(EdgeCapability.MOVE)
            else float("inf")
            )    
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
    

    def is_adjacent(self, a: int, b: int) -> bool:
        lbl = self.edge_labels.get((a, b))
        return lbl is not None and lbl.has_capability(EdgeCapability.MOVE)
    
    def distance(self, a: int, b: int) -> float:
        self.check_pos_index(a)
        self.check_pos_index(b)
        return float(self._move_cost_matrix[a, b])

    def shortest_path(self,start: int,target: int,edge_capability: EdgeCapability = EdgeCapability.MOVE) -> Optional[Tuple[List[int], float]]:
        self.check_pos_index(start)
        self.check_pos_index(target)

        if edge_capability == EdgeCapability.MOVE:
            graph = self._move_graph
        elif edge_capability == EdgeCapability.EXECUTE:
            graph = self._execute_graph
        else:
            graph = self.get_projected_graph(edge_capability)

        try:
            # Compute Dijkstra shortest path lengths and predecessors
            lengths, predecessors = rx.digraph_dijkstra_shortest_path_lengths(
                graph,
                start,
                edge_cost_fn=lambda edge: (
                    edge.get_weight(edge_capability)
                    if edge and edge.has_capability(edge_capability)
                    else float("inf")
                ),
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
            pred = predecessors[current]
            if isinstance(pred, list):
                pred = pred[0]
            current = pred
        path.append(start)
        path.reverse()

        return path, lengths[target]
    
    def build_move_cost_matrix(self) -> np.ndarray:
        """
        Build a dense matrix of shortest MOVE costs between all positions.
        """
        n = len(self.position_labels)
        D = np.full((n, n), np.inf)

        for src, targets in self.shortest_path_lengths.items():
            for dst, cost in targets.items():
                D[src, dst] = cost

        np.fill_diagonal(D, 0.0)

        return D
    def build_cluster_distance_map(self) -> Dict[int, np.ndarray]:
        """
        Precompute distance from every position to each executable cluster.
        Returns:
            {cluster_id : distance_vector}
        """
        n = len(self.position_labels)
        D = self.move_cost_matrix

        cluster_maps = {}

        for cid, cluster in enumerate(self.clusters):
            dist = np.full(n, np.inf)

            for p in range(n):
                dist[p] = min(D[p, c] for c in cluster)

            cluster_maps[cid] = dist

        return cluster_maps
    
    def build_move_gradient(self) -> Dict[int, Dict[int, List[int]]]:
        """
        For each cluster and position, give best neighbor moves.
        
        Returns:
            cluster_id → position → list of improving neighbors
        """
        gradients = {}

        for cid, dist in self._cluster_distance_maps.items():

            cluster_grad = {}

            for p in range(len(self.position_labels)):

                neighbors = list(self.graph.neighbors(p))
                best = []

                for n in neighbors:
                    if dist[n] < dist[p]:
                        best.append(n)

                cluster_grad[p] = best

            gradients[cid] = cluster_grad

        return gradients

    def _build_swap_neighbors(self) -> dict[int, tuple[int, ...]]:
        """
        Build a cache of neighbors reachable by MOVE or SWAP edges.
        """
        neighbors: dict[int, set[int]] = {i: set() for i in range(len(self._pos_labels))}

        for (u, v), label in self._edge_labels.items():
            if label.has_capability(EdgeCapability.MOVE) or \
            label.has_capability(EdgeCapability.SWAP):
                neighbors[u].add(v)
                neighbors[v].add(u)  # treat as undirected for swap candidate generation

        return {k: tuple(sorted(vs)) for k, vs in neighbors.items()}

    def get_shortest_path_tree(
        self,
        qudit_pos: int,
        edge_capability: EdgeCapability = EdgeCapability.MOVE,
    ) -> List[Tuple[int, ...]]:
        """
        Compute shortest paths from qudit_pos to every reachable position.

        Returns:
            A list of tuples, where index i contains the shortest path
            from qudit_pos to i. If i is unreachable, the tuple is empty.
        """
        self.check_pos_index(qudit_pos)

        if edge_capability == EdgeCapability.MOVE:
            graph = self.move_graph
        elif edge_capability == EdgeCapability.EXECUTE:
            graph = self.execute_graph
        else:
            graph = self.get_projected_graph(edge_capability)

        paths_dict = rx.digraph_dijkstra_shortest_paths(
            graph,
            qudit_pos,
            weight_fn=lambda edge: (
                edge.get_weight(edge_capability)
                if edge and edge.has_capability(edge_capability)
                else float("inf")
            ),
        )

        paths: List[Tuple[int, ...]] = []
        for node_index in range(len(self.position_labels)):
            if node_index in paths_dict:
                paths.append(tuple(paths_dict[node_index]))
            else:
                paths.append(())

        return paths


        

     


