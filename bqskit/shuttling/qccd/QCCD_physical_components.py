from __future__ import annotations
import numpy as np
from typing import NamedTuple


class Trap(NamedTuple):
    id: str
    max_num_ions: int
    initial_num_ions: int
    executable: bool = True
    measureable: bool = False


class Junction(NamedTuple):
    id: str


class Segment(NamedTuple):
    id: str
    left: Trap | Junction | Segment
    right: Trap | Junction | Segment


#physical_graph = {
#     'traps': [Trap(trap_id='0', max_num_ions=3, initial_num_ions=3),
#               Trap(trap_id='1', max_num_ions=3, initial_num_ions=3),
#               Trap(trap_id='2', max_num_ions=3, initial_num_ions=3),
#               Trap(trap_id='3', max_num_ions=3, initial_num_ions=3)],
#     'num_junctions': [Junction(junction_id='0'),
#                       Junction(junction_id='1')],
#     'segments': [Segment(segment_id='trap_junction_0', left=)]
#     ]
#
# }

# class Segment(NamedTuple):
#     left: Tuple[int, bool]
#     right: Tuple[int, bool]
#
#     @staticmethod
#     def between_trap_and_junction(trap_id: int,
#                                   junction_id: int) -> Segment:
#         return Segment(left=(trap_id, False), right=(junction_id, True))
#
#     @staticmethod
#     def between_junction_and_junction(junction_id_1: int,
#                                       junction_id_2: int) -> Segment:
#         return Segment(left=(junction_id_1, True), right=(junction_id_2, True))
#
#     @staticmethod
#     def between_trap_and_trap(trap_id_1: int,
#                               trap_id_2: int) -> Segment:
#         return Segment(left=(trap_id_1, False), right=(trap_id_2, False))

class QCCD_physical_machine:
    """
    Class to create QCCD physical machine.
    """

    def __init__(self,
                 num_traps: int,
                 num_junctions: int,
                 max_traps_size: list[int],
                 initial_ions: list[int] | None = None,
                 executable_traps: list[bool] | None = None,
                 measurable_traps: list[bool] | None = None,
                 traps_id: list[str] | None = None,
                 junctions_id: list[str] | None = None) -> None:
        """
            Construct a QCCD physical machine.
            Args:
                num_traps (int): Number of traps.
                num_junctions (int): Number of junctions.
                max_traps_size (list[int]): Size of traps.
                initial_ions (list[int]): Initial number of ions per trap.
                executable_traps (list[bool]): True if the traps are executable.
                measurable_traps (list[bool]): True if the traps are measurable.
                traps_id (list[str]): List of trap IDs.
                junctions_id (list[str]): List of junction IDs.
        """
        self.num_traps = num_traps
        self.num_junctions = num_junctions
        assert num_traps == len(max_traps_size), "Len of list of traps size is not equal to the number of traps"

        if initial_ions is not None:
            assert num_traps == len(initial_ions), ("Len of list of initial amount of ions per trap "
                                                    "is not equal to the number of traps")
        else:
            initial_ions = [0] * num_traps

        if executable_traps is not None:
            assert num_traps == len(executable_traps), ("Len of list of executable trap "
                                                        "is not equal to the number of traps")
        else:
            executable_traps = [True] * num_traps

        if measurable_traps is not None:
            assert num_traps == len(measurable_traps), ("Len of list of measurable trap "
                                                        "is not equal to the number of traps")
        else:
            measurable_traps = [True] * num_traps

        if traps_id is not None:
            assert num_traps == len(traps_id), "Len of list of traps id is not equal to the number of traps"
        else:
            traps_id = ['trap_' + str(i) for i in range(num_traps)]

        self.trap_list = [Trap(id=traps_id[idx],
                               max_num_ions=max_traps_size[idx],
                               initial_num_ions=initial_ions[idx],
                               executable=executable_traps[idx],
                               measureable=measurable_traps[idx])
                          for idx in range(num_traps)]

        if junctions_id is not None:
            assert num_junctions == len(junctions_id), ("Len of list of junction id is not equal to the number of "
                                                        "junctions")
        else:
            junctions_id = ['junction_' + str(i) for i in range(num_junctions)]

        self.junction_list = [Junction(id=junctions_id[idx]) for idx in range(num_junctions)]
        self.segment_list = []

    def get_trap(self,
                 trap_id: str) -> Trap | None:
        """
            Return trap based on trap id.
        """
        for trap in self.trap_list:
            if trap.id == trap_id:
                return trap
        return None

    def get_junction(self,
                     junction_id: str) -> Junction | None:
        """
            Return junction based on junction id.
        """
        for junction in self.junction_list:
            if junction.id.split('_')[-1] == junction_id:
                return junction
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
                     element_id: str) -> list[Junction | Trap] | None:
        """
        Return neighbor based on the id.
        """
        trap = self.get_trap(element_id)
        neighbor_lst = []
        for segment in self.segment_list:
            if segment.left == trap:
                neighbor_lst.append(segment.right.id)
            elif segment.right == trap:
                neighbor_lst.append(segment.left.id)
        return neighbor_lst if len(neighbor_lst) > 0 else None

    def add_segment(self,
                    left: Trap | Junction | Segment,
                    right: Trap | Junction | Segment,
                    segment_id: str = None) -> None:
        """
            Add a segment to the QCCD physical machine.
            Args:
                left(Trap | Junction | Segment): The left part of the segment
                right(Trap | Junction | Segment): The right part of the segment
                segment_id (str, optional): Segment ID.
        """
        if segment_id is None:
            segment_id = 'segment_' + str(len(self.segment_list))
        self.segment_list.append(Segment(id=segment_id,
                                         left=left,
                                         right=right))

    def print_physical_machine(self) -> None:
        """
            Print QCCD physical machine.
        """
        print("QCCD physical machine:")
        print("### Traps ### ")
        for trap in self.trap_list:
            print(trap)
        print("### Junctions ### ")
        for junction in self.junction_list:
            print(junction)
        print("### Segments ### ")
        for segment in self.segment_list:
            print(segment)


if __name__ == "__main__":
    from bqskit.shuttling.qccd.QCCD_util import create_testing_physical_machine
    physical_model = create_testing_physical_machine()
