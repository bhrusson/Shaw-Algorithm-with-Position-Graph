"""This module implements the QCCDMachineModel class."""
from __future__ import annotations

from typing import Sequence
from typing import TYPE_CHECKING
from itertools import combinations
from bqskit.ir import Operation
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
                 gate_set: GateSetLike | None = None) -> None:
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
            gate_set = GateSet({U1qPi2Gate, U1qPiGate, RZZGate()})
        else:
            gate_set = GateSet(gate_set)

        if not isinstance(gate_set, GateSet):
            raise TypeError(f'Expected GateSet, got {type(gate_set)}.')

        self.gate_set = gate_set
        self.physical_graph = physical_graph
        self.timing_data = timing_data
        (self.position_graph, self.physical_to_position, self.segment_assignment,
         self.total_num_positions) = self.generate_position_graph()

    def generate_position_graph(self) -> (CouplingGraph, dict, int):
        total_amount_positions = 0
        current_position_idx = 0
        coupling_graph = []
        position_assignment = {}
        coupling_assignment = {}
        # Accounting amount of traps
        for trap in self.physical_graph.trap_list:
            total_amount_positions += trap.max_num_ions
            position_assignment[trap.id] = range(current_position_idx,
                                                 current_position_idx + trap.max_num_ions)
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
            if segment.left in self.physical_graph.junction_list and segment.right in self.physical_graph.junction_list:
                junction_neighbor[segment.left.id].append(current_position_idx)
                junction_neighbor[segment.right.id].append(current_position_idx)
            elif segment.left in self.physical_graph.junction_list:
                right = min(position_assignment[segment.right.id])
                junction_neighbor[segment.left.id].append(current_position_idx)
                coupling_graph.append((current_position_idx, right))
                coupling_assignment[(current_position_idx, right)] = 'segment'
            elif segment.right in self.physical_graph.junction_list:
                left = max(position_assignment[segment.left.id])
                junction_neighbor[segment.right.id].append(current_position_idx)
                coupling_graph.append((left, current_position_idx))
                coupling_assignment[(left, current_position_idx)] = 'segment'
            else:
                left = max(position_assignment[segment.left.id])
                right = min(position_assignment[segment.right.id])
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
            for possible_combination in possible_combinations:
                coupling_graph.append(possible_combination)
                if len(junction_neighbor[junction.id]) == 3:
                    coupling_assignment[possible_combination] = 'junction_Y'
                elif len(junction_neighbor[junction.id]) == 4:
                    coupling_assignment[possible_combination] = 'junction_X'
        return CouplingGraph(coupling_graph), position_assignment, coupling_assignment, total_amount_positions

    def get_trap_id(self,
                    position: int) -> str | None:
        for trap in self.physical_graph.trap_list:
            if position in self.physical_to_position[trap.id]:
                return trap.id
        return None

    def trap_is_fully_occupied(self,
                               trap_id: str,
                               ion_assignment: dict) -> (bool, list):
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
        if current_gate.num_qudits == 1:
            return self.timing_data['sq_timings']
        elif current_gate.num_qudits == 2:
            return self.timing_data['tq_timings']
        else:
            raise ValueError('The gate is not either single-qubit or two-qubit gate.')

    def gate_is_executable(self,
                           current_gate: Operation,
                           ion_assignment: dict) -> bool:
        trap_id_lst = [self.get_trap_id(ion_assignment[qudit]) for qudit in current_gate.location]
        trap_is_executable = [self.physical_graph.get_trap(trap_id).executable for trap_id in trap_id_lst]
        if any(tid is None for tid in trap_id_lst) or any(excutable is False for excutable in trap_is_executable):
            return False
        return all(tid == trap_id_lst[0] for tid in trap_id_lst)

    def travelling_time_from_point(self,
                                   position1: int,
                                   position2: int) -> float:
        shortest_path_pos1 = self.position_graph.get_shortest_path_tree(position1)
        path = shortest_path_pos1[position2]
        runtime = 0.0
        ion_status = 'trap'
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
    physical_model = create_testing_physical_machine()
    timing_data = {'sq_timings': 1e-3,
                   'tq_timings': 2e-3}
    machine_model = QCCDMachineModel(physical_graph=physical_model,
                                     timing_data=timing_data)
    print("Position graph...")
    print(machine_model.position_graph)
    print("Physical to position mapping...")
    print(machine_model.physical_to_position)
    print("Coupling assignment mapping...")
    print(machine_model.segment_assignment)
    print("Total number of positions...")
    print(machine_model.total_num_positions)
    print(machine_model.position_graph.get_shortest_path_tree(10))
