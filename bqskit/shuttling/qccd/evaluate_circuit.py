from __future__ import annotations

import copy

import numpy as np
from bqskit.ir import Operation
from bqskit.ir.circuit import Circuit
from bqskit.ir.circuit import CircuitPoint
from bqskit import enable_logging

import logging

_logger = logging.getLogger(__name__)


def process_if_executable(location: CircuitPoint,
                          circuit: Circuit,
                          to_be_processed: dict,
                          next_layers_gate: set):
    if len(circuit.next(location)) != 0:
        for new_location in circuit.next(location):
            if new_location in to_be_processed:
                to_be_processed[new_location] += 1
            else:
                to_be_processed[new_location] = 1
            if to_be_processed[new_location] == len(circuit.prev(new_location)):
                next_layers_gate.add(new_location)
    return next_layers_gate, to_be_processed


def heuristic_assignment(current_gate: Operation,
                         QCCD_model: QCCDMachineModel,
                         starting_ion_assignment: dict) -> (dict, float):
    """
    The heuristic assignment to move the ions in such way that it allow the designated gate to be executable.

    Args:
        current_gate (Operation): The gate to be executed.

        QCCD_model (QCCDMachineModel): Data about the QCCD machine model.

        starting_ion_assignment (dict): The ion assignment of the QCCD trap circuit.
    """
    location = current_gate.location
    ion_assignment = starting_ion_assignment
    qubit_i_position = ion_assignment[location[0]]
    qubit_j_position = ion_assignment[location[1]]
    # print("Position of i:", qubit_i_position)
    # print("Position of j:", qubit_j_position)
    qubit_i_trap_id = QCCD_model.get_trap_id(qubit_i_position)
    qubit_j_trap_id = QCCD_model.get_trap_id(qubit_j_position)
    # print("Trap position of i:", qubit_i_trap_id)
    # print("Trap position of j:", qubit_j_trap_id)
    # Check for unoccupied space
    _, unoccupied_space_i = QCCD_model.trap_is_fully_occupied(qubit_i_trap_id, ion_assignment)
    _, unoccupied_space_j = QCCD_model.trap_is_fully_occupied(qubit_j_trap_id, ion_assignment)
    # print("Unoccupied space of i is:", unoccupied_space_i)
    # print("Unoccupied space of j is:", unoccupied_space_j)
    # Finding the shortest path and get the cost
    if unoccupied_space_i != [] or unoccupied_space_j != []:
        best_move = []
        cost = np.inf
        for space in unoccupied_space_i:
            #print(f"Trying moving ion from space {qubit_j_position} to space {space}.")
            movement_cost = QCCD_model.travelling_time_from_point(qubit_j_position, space)
            if movement_cost < cost:
                best_move = [location[1], space]
                cost = movement_cost
        for space in unoccupied_space_j:
            #print(f"Trying moving ion from space {qubit_i_position} to space {space}.")
            movement_cost = QCCD_model.travelling_time_from_point(qubit_i_position, space)
            if movement_cost < cost:
                best_move = [location[0], space]
                cost = movement_cost
        #print(f"Best move: Moving ion from space {best_move[0]} to space {best_move[1]}")
        #print("Cost:", cost)
        ion_assignment[best_move[0]] = best_move[1]
        #print("New assignment when the trap has space:", ion_assignment)
        return ion_assignment, cost
    else:
        # There is no availability within the trap
        nearset_point_i = QCCD_model.position_graph.get_neighbors_of(qubit_i_position)
        nearset_point_j = QCCD_model.position_graph.get_neighbors_of(qubit_j_position)
        best_move = []
        cost = np.inf
        for space in nearset_point_i:
            #print(f"Trying moving ion from space {qubit_j_position} to space {space}.")
            movement_cost = QCCD_model.travelling_time_from_point(qubit_j_position, space)
            if movement_cost < cost + 3 * QCCD_model.timing_data['junction_Y']:
                best_move = [location[1], space]
                cost = movement_cost + 3 * QCCD_model.timing_data['junction_Y']
        for space in nearset_point_j:
            #print(f"Trying moving ion from space {qubit_i_position} to space {space}.")
            movement_cost = QCCD_model.travelling_time_from_point(qubit_i_position, space)
            if movement_cost < cost + 3 * QCCD_model.timing_data['junction_Y']:
                best_move = [location[0], space]
                cost = movement_cost + 3 * QCCD_model.timing_data['junction_Y']
        #print(f"Best move: Moving ion from space {best_move[0]} to space {best_move[1]}")
        #print("Cost:", cost)
        ion_assignment[best_move[0]] = best_move[1]
        #print("New assignment when the trap does not have spare space:", ion_assignment)
        return ion_assignment, cost


def evaluate_small_circuit(circuit: Circuit,
                           QCCD_model: QCCDMachineModel,
                           starting_ion_assignment: dict) -> float:
    """
    Evaluate a small circuit using QCCD machine model.
    Assuming that all ion are initially stayed in a trap (If the circuit size is 3 qubits). There is only one case
    """
    runtime = 0.0
    frontier = circuit.front
    ion_assignment_state = copy.deepcopy(starting_ion_assignment)
    to_be_processed = dict()
    while len(frontier) != 0:
        #print(f"Current ion assignment state: {ion_assignment_state}")
        processed_points = []
        next_layers_gate = set()
        for location in frontier:
            op = circuit.get_operation(location)
            if QCCD_model.gate_is_executable(op, ion_assignment_state):
                processed_points.append(location)
                runtime += QCCD_model.gate_cost(op)
                next_layers_gate, to_be_processed = process_if_executable(location=location,
                                                                          circuit=circuit,
                                                                          to_be_processed=to_be_processed,
                                                                          next_layers_gate=next_layers_gate)
                #print(f"{op} has been processed.")

        # Try processing new point (depth first search)
        while next_layers_gate:
            location = next_layers_gate.pop()
            op = circuit.get_operation(location)
            if QCCD_model.gate_is_executable(op, ion_assignment_state):
                runtime += QCCD_model.gate_cost(op)
                next_layers_gate, to_be_processed = process_if_executable(location=location,
                                                                          circuit=circuit,
                                                                          to_be_processed=to_be_processed,
                                                                          next_layers_gate=next_layers_gate)
                #print(f"{op} has been processed.")
            else:
                frontier.add(location)
                #print(f"Operation at location {location} has been added to the frontier.")

        # Modifying the frontier
        for location in processed_points:
            frontier.remove(location)
            #print(f"Operation at location {location} has been removed.")

        # Modifying the ion assignment state such that at next iteration the first gate in frontier is executable
        # (As this function is for 3-qubit circuit, we only consider the state of 2-1)
        if len(frontier) != 0:
            focused_gate = circuit.get_operation(list(frontier)[0])
            ion_assignment_state, assignment_cost = heuristic_assignment(focused_gate,
                                                                         QCCD_model,
                                                                         ion_assignment_state)
            runtime += assignment_cost
    return runtime


# def evaluate_big_circuit(circuit: Circuit,
#                          QCCD_model: QCCDMachineModel,
#                          starting_ion_assignment: dict) -> float:
#     """
#     """
#     """
#     Evaluating the circuit with respect to the ion assignment and given trap_topology.
#
#     Args:
#         circuit (Circuit): Input circuit to be evaluated.
#
#         data (PassData): Data about the QCCD machine model.
#
#         ion_assignment (dict): The ion assignment of the QCCD trap circuit.
#     """
#     ion_assignment_state = starting_ion_assignment
#     frontier = circuit.front
#     to_be_processed = dict()
#     total_cost = 0.0
#     while len(frontier) != 0:
#         print(f"Current ion assignment state: {ion_assignment_state}")
#         investigating_further = True
#         while investigating_further:
#             processed_points = []
#             investigating_further = False
#             next_layers_gate = set()
#             for location in frontier:
#                 processed = False
#                 op = circuit.get_operation(location)
#                 if QCCD_model.gate_is_executable(op, ion_assignment_state):
#                     print(f"{op} has been processed.")
#                     total_cost += QCCD_model.gate_cost(op)
#                     processed_points.append(location)
#                     processed = True
#                 if len(circuit.next(location)) != 0 and processed is True:
#                     for new_location in circuit.next(location):
#                         if new_location in to_be_processed:
#                             to_be_processed[new_location] += 1
#                         else:
#                             to_be_processed[new_location] = 1
#                         if to_be_processed[new_location] == len(circuit.prev(new_location)):
#                             investigating_further = True
#                             next_layers_gate.add(new_location)
#             # Modifying the frontier
#             for location in processed_points:
#                 frontier.remove(location)
#                 print(f"Operation at location {location} has been removed.")
#             for new_location in next_layers_gate:
#                 frontier.add(new_location)
#                 print(f"Operation at location {new_location} has been added to the frontier.")
#
#         # Modifying the ion assignment state such that the next set of gate is executable
#         """
#             This can be further improved...
#         """
#         next_circuit_point = list(frontier)[0]
#         next_operation = circuit.get_operation(next_circuit_point)
#         ion_assignment_state, assignment_cost = heuristic_assignment(next_operation, QCCD_model,
#                                                                      starting_ion_assignment)
#         total_cost += assignment_cost
#     return total_cost

if __name__ == "__main__":
    from bqskit.shuttling.qccd.QCCD_machine import QCCDMachineModel
    from bqskit.shuttling.qccd.QCCD_util import (create_testing_physical_machine, create_simple_circuit_1,
                                                 create_simple_circuit_2)
    enable_logging(True)
    physical_model = create_testing_physical_machine()
    # Source (https://arxiv.org/pdf/2004.04706, https://journals.aps.org/pra/pdf/10.1103/PhysRevA.95.052319,
    # https://github.com/CQCL/pytket-phir/blob/main/pytket/phir/qtm_machine.py)
    timing_data = {'sq_timings': 30e-6,
                   'tq_timings': 40e-6,
                   'segment': 5e-6,
                   'inner_swap': 42e-6,
                   'split': 80e-6,
                   'merge': 80e-6,
                   'junction_Y': 100e-6,
                   'junction_X': 120e-6}
    machine_model = QCCDMachineModel(physical_graph=physical_model,
                                     timing_data=timing_data)
    # ToDo: Find a proper way to get this ion assginment...
    ion_assignment = {0: 0, 1: 2, 2: 6}
    test_circuit = create_simple_circuit_1(3)
    print("##### Checking simple circuit 1 ...")
    cost = evaluate_small_circuit(circuit=test_circuit,
                                  QCCD_model=machine_model,
                                  starting_ion_assignment=ion_assignment)
    print("Cost of the first circuit: ", cost)

    ion_assignment = {0: 0, 1: 2, 2: 6}
    test_circuit = create_simple_circuit_2(3)
    print("##### Checking simple circuit 2 ...")
    cost = evaluate_small_circuit(circuit=test_circuit,
                                  QCCD_model=machine_model,
                                  starting_ion_assignment=ion_assignment)
    print("Cost of the second circuit: ", cost)
