"""This module implements the QCCDMachineModel class."""
from __future__ import annotations
import copy
import numpy as np
from typing import Sequence
from typing import List
from typing import cast
from typing import TYPE_CHECKING
from itertools import combinations
from bqskit.ir import Operation
from bqskit.ir.gates.parameterized.rz import RZGate
from bqskit.ir.gates.parameterized.u1q import U1qPi2Gate, U1qPiGate
from bqskit.ir.gates.parameterized.rzz import RZZGate
from bqskit.compiler import MachineModel
from bqskit.compiler.gateset import GateSet
from bqskit.compiler.gateset import GateSetLike
from bqskit.qis.graph import CouplingGraph
from bqskit.shuttling.qccd.QCCD_physical_components import QCCD_physical_machine

if TYPE_CHECKING:
    from bqskit.ir.circuit import Circuit


class QCCDMachineModel(MachineModel):
    """A QCCD model of a quantum processing unit."""

    def __init__(self,
                 physical_graph: QCCD_physical_machine,
                 timing_data: dict,
                 multi_qudit_gate_type: str,
                 gate_set: GateSetLike | None = None
                 ) -> None:
        """
        MachineModel Constructor.

        Args:
            physical_graph (QCCD_physical_machine): The physical
                graph of the QCCD architecture, representing the assignment of
                traps, junction with respect to the space graph of the QCCD
                architecture.

            timing_data (dict[str, tuple[int, int]] | None): The timing data
                of the physical operations in the QCCD architecture which consists
                of shuttling time (move, split and merge) and execution time.

            gate_set (GateSetLike | None): The native gate set available
                on the machine. If left as None, the default gate set
                will be used. See :func:`~GateSet.default_gate_set`.

            radixes (Sequence[int]): A sequence with its length equal
                to `num_qudits`. Each element specifies the base of a
                qudit. Defaults to qubits.

        Raises:
            ValueError: If `num_qudits` is nonpositive.

        Note:
            Pre-built models for many active QPUs exist in the
            :obj:`~bqskit.ext` package.
        """

        if gate_set is None:
            gate_set = GateSet({U1qPi2Gate, U1qPiGate, RZZGate(), RZGate()})
        else:
            gate_set = GateSet(gate_set)

        if not isinstance(gate_set, GateSet):
            raise TypeError(f'Expected GateSet, got {type(gate_set)}.')

        self.gate_set = gate_set
        self.physical_graph = physical_graph
        self.timing_data = timing_data
        self.gate_type = multi_qudit_gate_type
        (self.position_graph,
         self.physical_to_position,
         self.position_to_physical,
         self.segment_assignment,
         self.trap_end_points,
         self.total_num_positions) = self.generate_position_graph()
        self.coupling_graph = self.position_graph
        self.num_qudits = self.position_graph.num_qudits
        self.radixes = tuple([2] * self.num_qudits)
        self.timing_mat = []
        self.calculate_timing_matrix()

    def calculate_timing_matrix(self) -> None:
        self.timing_mat = [
            [np.inf for _ in range(self.position_graph.num_qudits)]
            for _ in range(self.position_graph.num_qudits)
        ]
        for q1, q2 in self.segment_assignment:
            if (self.position_to_physical[q1] == "trap"
                    and self.position_to_physical[q2] == "trap"):
                self.timing_mat[q1][q2] = self.timing_data["inner_swap"]
                self.timing_mat[q2][q1] = self.timing_data["inner_swap"]
            elif (self.position_to_physical[q1] == "trap" or
                  self.position_to_physical[q2] == "trap"):
                self.timing_mat[q1][q2] = self.timing_data["merge"] + self.timing_data["segment"]
                self.timing_mat[q2][q1] = self.timing_data["merge"] + self.timing_data["segment"]  # assumption merge
                # and split having the same time
            else:
                if self.segment_assignment[(q1, q2)] == "segment":
                    self.timing_mat[q1][q2] = self.timing_data["segment"]
                    self.timing_mat[q2][q1] = self.timing_data["segment"]
                elif self.segment_assignment[(q1, q2)] == "junction_X":
                    self.timing_mat[q1][q2] = self.timing_data["junction_X"] + self.timing_data["segment"]
                    self.timing_mat[q2][q1] = self.timing_data["junction_X"] + self.timing_data["segment"]
                elif self.segment_assignment[(q1, q2)] == "junction_Y":
                    self.timing_mat[q1][q2] = self.timing_data["junction_Y"] + self.timing_data["segment"]
                    self.timing_mat[q2][q1] = self.timing_data["junction_Y"] + self.timing_data["segment"]

    def two_qudit_gate_time(self,
                            p1: int,
                            p2: int,
                            ion_distance: int = None,
                            ) -> float:
        if p1 == p2:
            raise ValueError("P1 and P2 of the two-qudit gate can not be the same.")
        if self.get_trap_id(p1) != self.get_trap_id(p2):
            raise ValueError(f"P1 and P2 can not be on different trap. P1 = {p1} is in {self.get_trap_id(p1)} "
                             f"while P2 = {p2} is in {self.get_trap_id(p2)}.")
        if ion_distance is None:
            ion_distance = abs(p1 - p2)
        if self.gate_type == "Duan":
            return (-22 + 100 * ion_distance) * 1e-6
        elif self.gate_type == "Trout":
            return (10 + 38 * ion_distance) * 1e-6
        elif self.gate_type == "PM":
            return (160 + 5 * ion_distance) * 1e-6
        elif self.gate_type == "FM":
            trap_capacity = self.physical_graph.get_trap(self.get_trap_id(p1)).max_num_ions
            return max(100, 13.33 * trap_capacity - 54) * 1e-6

    def generate_position_graph(self) -> (CouplingGraph,
                                          dict,
                                          dict,
                                          dict,
                                          dict,
                                          int):
        """
            Constructing the position graph (representation of physical machine)
        """
        total_amount_positions = 0
        current_position_idx = 0
        coupling_graph = []
        position_assignment = {}
        physical_assignment = {}
        coupling_assignment = {}
        trap_end_point = {}
        # Accounting amount of traps
        for trap in self.physical_graph.trap_list:
            total_amount_positions += trap.max_num_ions
            position_assignment[trap.id] = range(current_position_idx,
                                                 current_position_idx + trap.max_num_ions)
            for position in range(current_position_idx,
                                  current_position_idx + trap.max_num_ions):
                physical_assignment[position] = 'trap'
            for _ in range(trap.max_num_ions - 1):
                coupling_graph.append((current_position_idx, current_position_idx + 1))
                coupling_assignment[(current_position_idx, current_position_idx + 1)] = 'trap'
                current_position_idx += 1
            current_position_idx += 1
        # Initialize list of junctions:
        junction_neighbor = {}
        for junction in self.physical_graph.junction_list:
            junction_neighbor[junction.id] = []
        # Accounting all the segments
        segment_space = []
        for segment in self.physical_graph.segment_list:
            total_amount_positions += 1
            position_assignment[segment.id] = [current_position_idx]
            physical_assignment[current_position_idx] = 'segment'
            # Junction to junction
            if segment.left in self.physical_graph.junction_list and segment.right in self.physical_graph.junction_list:
                junction_neighbor[segment.left.id].append(current_position_idx)
                junction_neighbor[segment.right.id].append(current_position_idx)
            # Trap to Junction
            elif segment.left in self.physical_graph.junction_list:
                right = min(position_assignment[segment.right.id])
                junction_neighbor[segment.left.id].append(current_position_idx)
                coupling_graph.append((current_position_idx, right))
                if segment.right.id in trap_end_point.keys():
                    trap_end_point[segment.right.id].append(right)
                else:
                    trap_end_point[segment.right.id] = [right]
                coupling_assignment[(current_position_idx, right)] = 'segment'
            # Trap to Junction
            elif segment.right in self.physical_graph.junction_list:
                left = max(position_assignment[segment.left.id])
                junction_neighbor[segment.right.id].append(current_position_idx)
                coupling_graph.append((left, current_position_idx))
                coupling_assignment[(left, current_position_idx)] = 'segment'
                if segment.left.id in trap_end_point.keys():
                    trap_end_point[segment.left.id].append(left)
                else:
                    trap_end_point[segment.left.id] = [left]
            # Trap connect to trap
            else:
                left = max(position_assignment[segment.left.id])
                right = min(position_assignment[segment.right.id])
                if segment.right.id in trap_end_point.keys():
                    trap_end_point[segment.right.id].append(right)
                else:
                    trap_end_point[segment.right.id] = [right]
                if segment.left.id in trap_end_point.keys():
                    trap_end_point[segment.left.id].append(left)
                else:
                    trap_end_point[segment.left.id] = [left]
                coupling_graph.append((left, current_position_idx))
                coupling_graph.append((current_position_idx, right))
                coupling_assignment[(left, current_position_idx)] = 'segment'
                coupling_assignment[(current_position_idx, right)] = 'segment'
            segment_space.append(current_position_idx)
            current_position_idx += 1
        position_assignment['segment_space'] = segment_space
        # Form new coupling in place of junctions
        for junction in self.physical_graph.junction_list:
            if len(junction_neighbor[junction.id]) != 3 and len(junction_neighbor[junction.id]) != 4:
                raise ValueError(
                    f"The number of degree at junction {junction.id} is {len(junction_neighbor[junction.id])} "
                    f"which is not an appropriate number")
            possible_combinations = combinations(junction_neighbor[junction.id], 2)
            for possible_combination in list(possible_combinations):
                coupling_graph.append(possible_combination)
                if len(junction_neighbor[junction.id]) == 3:
                    coupling_assignment[possible_combination] = 'junction_Y'
                elif len(junction_neighbor[junction.id]) == 4:
                    coupling_assignment[possible_combination] = 'junction_X'
        return (CouplingGraph(coupling_graph), position_assignment, physical_assignment,
                coupling_assignment, trap_end_point, total_amount_positions)

    def update_wrt_perm(self,
                        initial_placement: list[int],
                        permutation: list[int]) -> None:
        placement = initial_placement
        pi_c = {q: placement[permutation[i]] for i, q in enumerate(sorted(permutation))}
        for q in permutation:
            placement[q] = pi_c[q]
        print("Placement: ", placement)
        # Update position graph
        new_position_graph_edges = []
        for coupling in self.position_graph:
            print("Coupling:", coupling)
            new_position_graph_edges.append((placement.index(coupling[0]),
                                             placement.index(coupling[1])))
        self.position_graph = CouplingGraph(new_position_graph_edges, self.position_graph.num_qudits)
        self.coupling_graph = self.position_graph

        # Update physical_to_position,
        new_physical_to_position = {}
        for key in self.physical_to_position.keys():
            key_value = self.physical_to_position[key]
            new_key_value = [placement.index(value) for value in key_value]
            new_physical_to_position[key] = new_key_value
        self.physical_to_position = new_physical_to_position

        # Update position_to_physical
        new_position_to_physical = {}
        for key in self.position_to_physical.keys():
            new_key = placement.index(key)
            new_position_to_physical[new_key] = self.position_to_physical[key]
        self.position_to_physical = new_position_to_physical

        # Update segment assginment
        new_segment_assignment = {}
        for key in self.segment_assignment.keys():
            key = list(key)
            new_key = [placement.index(key_val) for key_val in key]
            new_segment_assignment[tuple(new_key)] = self.segment_assignment[tuple(key)]
        self.segment_assignment = new_segment_assignment

        # Update trap_end_points
        trap_end_point = {}
        for key in self.trap_end_points.keys():
            new_value = [placement.index(value) for value in self.trap_end_points[key]]
            trap_end_point[key] = new_value
        self.trap_end_points = trap_end_point

        # Recalculate timing mat
        self.calculate_timing_matrix()

    def get_trap_id(self,
                    position: int) -> str | None:
        """
            If the position is in the trap
                Return the trap id
            Else
                Return None
        """
        for trap in self.physical_graph.trap_list:
            if position in self.physical_to_position[trap.id]:
                return trap.id
        return None

    def check_valid_assignment(self,
                               ion_assignment: dict) -> bool:
        """
            Check if the ion assignment is valid
                (1) There is no repetition in the assignment.
                (2) All the ions are in trap.
        """
        if (len(set(ion_assignment.keys())) != len(ion_assignment.keys()) or
                len(set(ion_assignment.items())) != len(ion_assignment.items())):
            return False
        # print("Pass repetition test!")
        for position in ion_assignment.values():
            if self.position_to_physical[position] != 'trap':
                # print("Position", position, "is not trap")
                # print("Position", position, "is", self.position_to_physical[position])
                return False
        return True

    def trap_is_fully_occupied(self,
                               trap_id: str,
                               ion_assignment: dict) -> (bool, list):
        """
            Check if the given trap is fully occupied w.r.t the ion assignment
                If yes, return True and empty list
                If no, return False and list of available space within the trap.
        """
        trap = self.physical_graph.get_trap(trap_id)
        trap_count = 0
        unoccupied_space = list(self.physical_to_position[trap.id])
        for qubit in ion_assignment.keys():
            space = ion_assignment[qubit]
            if space in self.physical_to_position[trap.id]:
                trap_count += 1
                unoccupied_space.remove(space)
        if trap_count == trap.max_num_ions:
            return True, []
        else:
            return False, unoccupied_space

    def gate_cost(self, current_gate: Operation):
        """
            Return the execution time of specific gate
        """
        if current_gate.num_qudits == 1:
            return self.timing_data['sq_timings']
        elif current_gate.num_qudits == 2:
            return self.timing_data['tq_timings']
        else:
            raise ValueError('The gate is not either single-qubit or two-qubit gate.')

    def gate_is_executable(self,
                           current_gate: Operation,
                           pi: list,
                           ion_assignment: dict) -> bool:
        """
            Check if the gate is executable given current ion assignment and physical machine
            pi: logical to physical
            ion_assignment: physical to position
        """
        trap_id_lst = [self.get_trap_id(ion_assignment[pi[qudit]]) for qudit in current_gate.location]
        # print("Trap id list: ", trap_id_lst)
        if None in trap_id_lst:
            return False
        trap_is_executable = [self.physical_graph.get_trap(trap_id).executable for trap_id in trap_id_lst]
        # print("Trap is_executable: ", trap_is_executable)
        if any(tid is None for tid in trap_id_lst) or any(excutable is False for excutable in trap_is_executable):
            return False
        return all(tid == trap_id_lst[0] for tid in trap_id_lst)

    def path_is_blocked(self,
                        position1: int,
                        position2: int,
                        ion_assignment: dict) -> (bool, list[int]):
        """
            Check if the shortest path between the two positions is blocked by an ion or not
            If yes, return the penalty of its
        """
        shortest_path_pos1 = self.position_graph.get_shortest_path_tree(position1)
        path = shortest_path_pos1[position2]
        blocked_ions = []
        blocked_flag = False
        for idx_point in range(1, len(path) - 1):
            if path[idx_point] in ion_assignment.values():
                blocked_ions.append(path[idx_point])
                blocked_flag = True
        return blocked_flag, blocked_ions

    def all_pair_travelling_time(self) -> list[list[float]]:
        """
        Calculate all pairs matrix using Floyd-Warshall.

        Returns:
            D (list[list[int]]): D[i][j] is the length of the shortest
                path from i to j.
        """
        D = copy.deepcopy(self.timing_mat)
        for k in range(self.position_graph.num_qudits):
            for i in range(self.position_graph.num_qudits):
                for j in range(self.position_graph.num_qudits):
                    D[i][j] = min(D[i][j], D[i][k] + D[k][j])
        for id in range(self.position_graph.num_qudits):
            D[id][id] = 0.0
        return cast(List[List[float]], D)

    def travelling_time_from_point(self,
                                   position1: int,
                                   position2: int) -> float:
        """
            Time cost to move an ion from position 1 to position 2
        """
        shortest_path_pos1 = self.position_graph.get_shortest_path_tree(position1)
        path = shortest_path_pos1[position2]
        runtime = 0.0
        ion_status = self.position_to_physical[position1]
        for idx_point in range(len(path) - 1):
            coupling = tuple(sorted((path[idx_point], path[idx_point + 1])))
            if self.segment_assignment[coupling] == 'segment':
                if ion_status == 'trap':
                    runtime += self.timing_data['split']
                    runtime += self.timing_data['merge']
                    runtime += self.timing_data['segment']
                    ion_status = 'segment'
            elif self.segment_assignment[coupling] == 'trap':
                if ion_status == 'segment':
                    runtime += self.timing_data['inner_swap']
                    ion_status = 'trap'
            elif self.segment_assignment[coupling] == 'junction_Y':
                runtime += self.timing_data['junction_Y']
            elif self.segment_assignment[coupling] == 'junction_X':
                runtime += self.timing_data['junction_X']
            else:
                raise ValueError(
                    f"The segment has invalid assignment: "
                    f"{self.segment_assignment[self.segment_assignment[sorted((path[idx_point], path[idx_point + 1]))]]}")
        return runtime


if __name__ == '__main__':
    from bqskit.shuttling.qccd.QCCD_util import create_testing_physical_machine
    type = 'linear'
    trap_capacity = 2
    num_traps = 3
    physical_model = create_testing_physical_machine(type=type,
                                                     trap_capacity=trap_capacity,
                                                     num_traps=num_traps)
    timing_data = {'sq_timings': 30e-6,
                   'tq_timings': 40e-6,
                   'segment': 5e-6,
                   'inner_swap': 42e-6,
                   'split': 80e-6,
                   'merge': 80e-6,
                   'junction_Y': 100e-6,
                   'junction_X': 120e-6}
    machine_model = QCCDMachineModel(physical_graph=physical_model,
                                     multi_qudit_gate_type='FM',
                                     timing_data=timing_data)
    print("Position graph...")
    print(machine_model.position_graph)
    print("Physical to position mapping...")
    print(machine_model.physical_to_position)
    print("Position to physical mapping...")
    print(machine_model.position_to_physical)
    print("Coupling assignment mapping...")
    print(machine_model.segment_assignment)
    print("All trap end points...")
    print(machine_model.trap_end_points)
    print("Total number of positions...")
    print(machine_model.total_num_positions)
    # new_placement = [3, 4, 0, 2, 1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    # print("There is a new placement :", new_placement)
    # machine_model.update_wrt_perm(initial_placement=list(range(machine_model.total_num_positions)),
    #                               permutation=new_placement)
    # print("After updating the placement ....")
    # print("Position graph...")
    # print(machine_model.position_graph)
    # print("Physical to position mapping...")
    # print(machine_model.physical_to_position)
    # print("Position to physical mapping...")
    # print(machine_model.position_to_physical)
    # print("Coupling assignment mapping...")
    # print(machine_model.segment_assignment)
    # print("All trap end points...")
    # print(machine_model.trap_end_points)
    # print("Total number of positions...")
    # print(machine_model.total_num_positions)
    # print("Timing matrix...")
    # print(machine_model.timing_mat)
    # print("All pair travelling time...")
    # print(machine_model.all_pair_travelling_time())
