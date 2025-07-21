from abc import ABC, abstractmethod
from typing import List, Tuple, Dict
from bqskit.graph import PositionGraph
from bqskit.shuttling.qccd import Node, Segment
from bqskit.compiler.gateset import GateSet, GateSetLike
from bqskit.ir.gates.parameterized import RZGate, RZZGate, U1qPi2Gate, U1qPiGate

from typing import NamedTuple


class Node(NamedTuple):
    id: str
    max_num: int
    executable: bool = True
    measureable: bool = False


class Segment(NamedTuple):
    id: str
    left: Node | Segment
    right: Node | Segment

class MachineModel(ABC):
    @abstractmethod
    def to_position_graph(self) -> PositionGraph:
        pass

    @abstractmethod
    def get_segment(self,
                    segment_id: str) -> Segment | None:
        pass

    @abstractmethod
    def get_neighbor(self,
                     element_id: str) -> Node | None:
        pass

    @abstractmethod
    def get_node(self,
                 node_id: str) -> Node | None:
        pass


class QCCDMachineModel(MachineModel):
    def __init__(self,
                 num_nodes: int,
                 max_size_per_node: list[int],
                 nodes_id: list[str] | None = None,
                 executable_nodes: list[bool] | None = None,
                 measurable_nodes: list[bool] | None = None,
                 initial_ions: list[int] | None = None,
                 ) -> None:
        """
            Construct QCCD physical machine.
            Args:
                num_nodes (int): Number of nodes in the physical machine.
                max_size_per_node (list[int]): Number of max size nodes in the physical machine.
                nodes_id (list[str] | None): List of node IDs.
                initial_ions (list[int]): Initial number of ions per trap.
                executable_nodes (list[bool] | None): List of executable nodes.
                measurable_nodes (list[bool] | None): List of measurable nodes.
                initial_ions (list[int] | None): Initial number of ions per trap.
        """
        self.num_nodes = num_nodes
        assert num_nodes == len(max_size_per_node), "Len of list of max size is not equal to the number of nodes."
        self.max_size_per_node = max_size_per_node

        if nodes_id is None:
            self.nodes_id = range(self.num_nodes)
        else:
            assert num_nodes == len(nodes_id), "Len of list of nodes id is not equal to the number of nodes."
            self.nodes_id = nodes_id

        if executable_nodes is not None:
            assert nodes_id == len(executable_nodes), ("Len of list of executable nodes id "
                                                       "is not equal to the number of nodes.")
        else:
            self.executable_nodes = [True] * num_nodes

        if measurable_nodes is not None:
            assert num_nodes == len(measurable_nodes), ("Len of list of measurable nodes id "
                                                        "is not equal to the number of nodes.")
        else:
            self.measurable_nodes = [True] * num_nodes

        if initial_ions is not None:
            assert num_nodes == len(initial_ions), ("Len of list of initial amount of ions per trap "
                                                    "is not equal to the number of traps")
        else:
            initial_ions = [0] * num_nodes

        self.node_list = [Node(id=nodes_id[idx],
                               max_num_ions=max_size_per_node[idx],
                               executable=executable_nodes[idx],
                               measureable=measurable_nodes[idx],
                               initial_ions=initial_ions[idx])
                          for idx in range(self.num_nodes)]
        self.executable_node_list = [node for node in self.node_list if node.executable]
        self.segment_list = []

    def get_node(self,
                 node_id: str) -> Node | None:
        """
            Return node based on node id.
        """
        for node in self.node_list:
            if node.id == node_id:
                return node
        return None

    def get_segment(self,
                    segment_id: str) -> Segment | None:
        """
            Return segment based on segment id.
        """
        for segment in self.segment_list:
            if segment.id.split('_')[-1] == segment_id:
                return segment
        return None

    def get_neighbor(self,
                     element_id: str) -> Node | None:
        """
            Return neighbor based on the id.
        """
        node = self.get_node(element_id)
        neighbor_lst = []
        for segment in self.segment_list:
            if segment.left == node:
                neighbor_lst.append(segment.right.id)
            elif segment.right == node:
                neighbor_lst.append(segment.left.id)
        return neighbor_lst if len(neighbor_lst) > 0 else None

    def add_segment(self,
                    left: Node | Segment,
                    right: Node | Segment,
                    segment_id: str = None) -> None:
        """
            Add a segment to the QCCD physical machine.
            Args:
                left(Node | Segment): The left part of the segment
                right(Node | Segment): The right part of the segment
                segment_id (str, optional): Segment ID.
        """
        if segment_id is None:
            segment_id = 'segment_' + str(len(self.segment_list))
        self.segment_list.append(Segment(id=segment_id,
                                         left=left,
                                         right=right))

    def to_position_graph(self):
        pg = PositionGraph()
        return pg


class SCMachineModel(MachineModel):
    def __init__(self,
                 num_nodes: int,
                 nodes_id: list[str] | None = None,
                 executable_nodes: list[bool] | None = None,
                 measurable_nodes: list[bool] | None = None,
                 ) -> None:
        """
            Construct SC physical machine.
            Args:
                num_nodes (int): Number of nodes in the physical machine.
                max_size_per_node (list[int]): Number of max size nodes in the physical machine.
                nodes_id (list[str] | None): List of node IDs.
                executable_nodes (list[bool] | None): List of executable nodes.
                measurable_nodes (list[bool] | None): List of measurable nodes.
        """
        self.num_nodes = num_nodes
        self.max_size_per_node = [1] * num_nodes
        if nodes_id is None:
            self.nodes_id = range(self.num_nodes)
        else:
            assert num_nodes == len(nodes_id), "Len of list of nodes id is not equal to the number of nodes."
            self.nodes_id = nodes_id

        if executable_nodes is not None:
            assert nodes_id == len(executable_nodes), ("Len of list of executable nodes id "
                                                       "is not equal to the number of nodes.")
        else:
            self.executable_nodes = [True] * num_nodes

        if measurable_nodes is not None:
            assert num_nodes == len(measurable_nodes), ("Len of list of measurable nodes id "
                                                        "is not equal to the number of nodes.")
        else:
            self.measurable_nodes = [True] * num_nodes

        self.node_list = [Node(id=self.nodes_id[idx],
                               max_num_ions=self.max_size_per_node[idx],
                               executable=self.executable_nodes[idx],
                               measureable=self.measurable_nodes[idx])
                          for idx in range(self.num_nodes)]
        self.executable_node_list = [node for node in self.node_list if node.executable]
        self.segment_list = []

    def get_node(self,
                 node_id: str) -> Node | None:
        """
            Return node based on node id.
        """
        for node in self.node_list:
            if node.id == node_id:
                return node
        return None

    def get_segment(self,
                    segment_id: str) -> Segment | None:
        """
            Return segment based on segment id.
        """
        for segment in self.segment_list:
            if segment.id.split('_')[-1] == segment_id:
                return segment
        return None

    def get_neighbor(self,
                     element_id: str) -> Node | None:
        """
            Return neighbor based on the id.
        """
        node = self.get_node(element_id)
        neighbor_lst = []
        for segment in self.segment_list:
            if segment.left == node:
                neighbor_lst.append(segment.right.id)
            elif segment.right == node:
                neighbor_lst.append(segment.left.id)
        return neighbor_lst if len(neighbor_lst) > 0 else None

    def add_segment(self,
                    left: Node | Segment,
                    right: Node | Segment,
                    segment_id: str = None) -> None:
        """
            Add a segment to the QCCD physical machine.
            Args:
                left(Node | Segment): The left part of the segment
                right(Node | Segment): The right part of the segment
                segment_id (str, optional): Segment ID.
        """
        if segment_id is None:
            segment_id = 'segment_' + str(len(self.segment_list))
        self.segment_list.append(Segment(id=segment_id,
                                         left=left,
                                         right=right))

    def to_position_graph(self):
        pg = PositionGraph()
        return pg
