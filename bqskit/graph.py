"""
    We need the ability to determine the executability of a gate.
        Gate are executable when "all participating qubits in executable zone",
        Executable zones are connected subgraph such that
            (1) all nodes have executable flag
            (2) there exists a spanning subgraph such that all edges are executable
              (... let think about this ... 0-1-2 group: 0-2 is also executable...)
"""
class PositionGraph:
    def __init__(self, ...):
        pass
    """
        Choice between: 
            * Edge list
            * Adjacency list
            * Adjacency matrix
        What are methods that depending on these?
        * Edge list, Adjacency list: good for enumeration, 
        * Adjacency matrix: good for finding shortest path
    """


    """
        _get_node_neighbors:
            Args: 
                 (Node of a position graph)
                 
            Return the neighbors of the nodes based on position graph.
    """
    def _get_node_neighbors(self, node):
        pass

    """
        _get_edge_info:
            Args: 
                node_1 (Node of a position graph)
                node_2 (Node of a position graph)

            Return the info of the edges (this can have different set of characteristic based on hardware)
            * For example in QCCD architecture, based on edge info we can derive the node info
    """
    def _get_edge_info(self, node_1, node_2):
       pass

    """
        _check_executable:
            Args:
                nodes (List of nodes of a position graph)
                
            Return True if the block form by the list of nodes is executable, False otherwise.
    """
    def _check_executable(self, nodes):
        pass

    """
        _get_shortest_path:
            Args:
                node_1 (Node of a position graph)
                node_2 (Node of a position graph)
            Return the shortest path between node_1 and node_2 using Edge list representation.
    """
    def _get_shortest_path(self, node_1, node_2):
        pass

    """
        1) How to find a way to express the state of position graph over time? 
        Currently, we are using ion_assignment variable. Is there a better way to express this?
        2) Is it needed to fully identified the labelling of edges and nodes? How to abstract this from 
        the hardware architecture?
        3) _get_connected_zone? For example in both neutral atom and QCCD, there are connected zone, each represent 
        specific jobs (open question)
        4) How to incorporate tools for optimization of shortest path formulation as the state of position graph? 
        (least priority)
    """