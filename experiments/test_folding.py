from __future__ import annotations
from enum import Enum

import numpy as np
from bqskit import Circuit
from bqskit.ir.gates import *
from bqskit.ir.point import CircuitPoint
from bqskit.ir.location import CircuitLocation
from bqskit.ir import Operation


class MachineSchedulingState(Enum):
    """Shift State of a Shuttling Machine."""
    EVEN = 0
    ODD = 1

    def flip(self) -> MachineSchedulingState:
        """Flip the state."""
        if self is MachineSchedulingState.EVEN:
            return MachineSchedulingState.ODD
        else:
            return MachineSchedulingState.EVEN


def matches_state(op: Operation, state: MachineSchedulingState) -> bool:
    """Check if the operation matches the state."""
    if op.num_qudits == 1:
        return True

    min_operation = np.min(op.location)
    return (
            state == MachineSchedulingState.EVEN and min_operation % 2 == 0
            or
            state == MachineSchedulingState.ODD and min_operation % 2 == 1
    )


# Zone is overloaded with GateZone, come up with another name (TODO)
ShiftZone = list[CircuitPoint]


def has_processed_dependency(location: CircuitPoint, processed_gates: set[CircuitPoint], circuit: Circuit) -> bool:
    """Check if the current operation is executable at the moment"""

    return all(gate in processed_gates for gate in circuit.prev(location))


def zone_circuit(circuit: Circuit) -> (list[ShiftZone], list[int], list[MachineSchedulingState]):
    """
    Separate the circuit into zones delimited by shifts.

    The machine will always start in the even state.
    """
    machine_state = MachineSchedulingState.EVEN
    zones = []
    machine_states = []
    zones_weight = []
    frontier = circuit.front
    to_be_processed = dict()
    while len(frontier) != 0:
        print("Current machine state: ", machine_state)
        print("Current frontier: ", frontier)
        zone = []
        weight = 0
        investigating_further = True
        # Select the executable points wrt the current machine states
        while investigating_further:
            processed_points = []
            investigating_further = False
            next_layers_gate = set()

            for location in frontier:
                processed = False
                op = circuit.get_operation(location)
                if matches_state(op, machine_state):
                    print("Checking the current: ", op)
                    print("Location: ", location)
                    print("Number of previous gates: ", len(circuit.prev(location)))
                    zone.append(location)
                    if op.num_qudits == 2:  # can restrict to certain gate types
                        weight += 1
                    processed_points.append(location)
                    processed = True

                if len(circuit.next(location)) != 0 and processed is True:
                    for new_location in circuit.next(location):
                        if new_location in to_be_processed:
                            to_be_processed[new_location] += 1
                        else:
                            to_be_processed[new_location] = 1
                        if to_be_processed[new_location] == len(circuit.prev(new_location)):
                            investigating_further = True
                            print(f"Gate {circuit.get_operation(new_location)} is added inside iteration")
                            next_layers_gate.add(new_location)
            # Modifying the frontier
            for location in processed_points:
                frontier.remove(location)
            for new_location in next_layers_gate:
                frontier.add(new_location)
            print("Remains Frontier: ", frontier)
        print("Frontier at the next zone: ", frontier)
        zones.append(zone)
        zones_weight.append(weight)
        machine_states.append(machine_state)
        machine_state = machine_state.flip()
    return zones, zones_weight, machine_states


def test_zone_circuit():
    # You should test everything all the time!
    # leave them in the code well labeled.
    # !
    circuit = Circuit(5)
    circuit.append_gate(HGate(), 0)
    circuit.append_gate(HGate(), 1)
    circuit.append_gate(RZZGate(), [0, 1])
    circuit.append_gate(RZZGate(), [2, 3])
    circuit.append_gate(HGate(), 2)
    circuit.append_gate(RZZGate(), [3, 4])
    circuit.append_gate(HGate(), 3)
    circuit.append_gate(HGate(), 4)
    circuit.append_gate(HGate(), 0)
    circuit.append_gate(HGate(), 1)
    circuit.append_gate(RZZGate(), [1, 2])
    print(zone_circuit(circuit))
    # assert zone_circuit(circuit) == [[(0, 1), (0, 2), (0, 0), (1, 2), (1, 0), (2, 1), (2, 0)], [(3, 1), (1, 3), (2, 3), (2, 4)]]


from bqskit import enable_logging

enable_logging(True)


def test_surround_filter():
    circuit = Circuit(6)
    # whole wall of even
    circuit.append_gate(RZZGate(), [0, 1])
    circuit.append_gate(RZZGate(), [2, 3])
    circuit.append_gate(RZZGate(), [4, 5])

    # one odd gate; problematic point in test
    circuit.append_gate(RZZGate(), [3, 4])

    # whole wall of even
    circuit.append_gate(RZZGate(), [0, 1])
    circuit.append_gate(RZZGate(), [2, 3])
    circuit.append_gate(RZZGate(), [4, 5])

    region = circuit.surround((1, 3), 4, None, False,
                              lambda node: (node[2][0] > 1 and node[2][-1] < 6) or (node[2][0] % 2 == 0))
    print(region.location)
    assert region.location == CircuitLocation([2, 3, 4, 5])


def test_surround_filter_hard():
    circuit = Circuit(7)
    # whole wall of even
    circuit.append_gate(RZZGate(), [0, 1])
    circuit.append_gate(RZZGate(), [2, 3])
    circuit.append_gate(RZZGate(), [4, 5])

    # one odd gate; problematic point in test
    circuit.append_gate(RZZGate(), [3, 4])

    # whole wall of even
    circuit.append_gate(RZZGate(), [0, 1])
    circuit.append_gate(RZZGate(), [2, 3])
    circuit.append_gate(RZZGate(), [4, 5])

    # more odd gates to really test filter
    circuit.append_gate(RZZGate(), [5, 6])
    circuit.append_gate(RZZGate(), [5, 6])
    circuit.append_gate(RZZGate(), [5, 6])
    circuit.append_gate(RZZGate(), [5, 6])
    circuit.append_gate(RZZGate(), [5, 6])
    # 0, 1, 2
    region = circuit.surround((1, 3), 4, None, False,
                              None)  #lambda node: (node[2][0] > 1 and node[2][-1] < 6) or (node[2][0] % 2 == 0))
    print(region.location)
    assert region.location == CircuitLocation([2, 3, 4, 5])


# def return_reorder_gate_by_layers(
#         circuit: Circuit,
#         layer_idx: int,
#         prev_layer_parity: MachineSchedulingState) -> list[Operation]:
#     """Choose gates for iteration that match the state."""
#     layer = circuit[layer_idx]
#     even_rzz = []
#     odd_rzz = []
#     reordered_layer = []
#     for op in layer:
#         if op.gate == RZZGate():
#             if op.location[0] > op.location[1]:
#                 if int(op.location[1] % 2) == 1:
#                     odd_rzz.append(op)
#                 else:
#                     even_rzz.append(op)
#             elif op.location[0] < op.location[1]:
#                 if int(op.location[0] % 2) == 1:
#                     odd_rzz.append(op)
#                 else:
#                     even_rzz.append(op)
#             else:
#                 raise ValueError(f"Weird operation location {op.location}")
#         else:
#             reordered_layer.append(op)
#     if even_rzz == [] and odd_rzz != []:
#         # prev_layer_parity = bool(1)
#         reordered_layer = layer
#     elif odd_rzz == [] and even_rzz != []:
#         # prev_layer_parity = bool(0)
#         reordered_layer = layer
#     elif odd_rzz != [] and even_rzz != []:
#         # print("Current machine state: ", prev_layer_parity)
#         if prev_layer_parity:
#             reordered_layer.extend(odd_rzz)
#             reordered_layer.extend(even_rzz)
#         else:
#             reordered_layer.extend(even_rzz)
#             reordered_layer.extend(odd_rzz)
#     else:
#         reordered_layer = layer
#     return reordered_layer


def dry_scheduling_ver2(circuit: Circuit):
    """
    Assign weights to the circuit's zones delimited by shifts.

    Used to determine the problematic points.
    """
    zones, zones_weight, machine_states = zone_circuit(circuit)
    potential_shift_locations = []
    for zone_idx in range(1, len(zones)):
        if zones_weight == 1:  # possible problematic points (Experiments needed)
            for point in zones[zone_idx]:
                if circuit[point].gate == RZZGate():
                    potential_shift_locations.append(point)
    return zones, zones_weight, machine_states, potential_shift_locations


# def dry_scheduling(circuit: Circuit):
#     """
#     Assign weights to the circuit's zones delimited by shifts.
#
#     Used to determine the problematic points.
#     """
#     # Setup
#     machine_state = MachineSchedulingState.EVEN
#     shift_counts = 0
#     shifts_weight = []
#     shifts_location = []
#     state_before_shifts = []
#     circ_depth = circuit.num_cycles
#     # Dry-scheduling
#     for idx in range(circ_depth):
#         reordered_layer = return_reorder_gate_by_layers(circuit, idx, machine_state)
#         for op in reordered_layer:  # TODO: re-order the gate such that it requires the least amount of shift gates
#             if op.gate == RZZGate() or op.gate == SwapGate():
#                 # Adding all possible Rzz gates
#                 if op.location[0] > op.location[1]:
#                     # Adding weighted shift
#                     if op.location[1] % 2 == 0:
#                         if machine_state is True:
#                             shift_counts += 1
#                             state_before_shifts.append(machine_state)
#                             machine_state = not machine_state
#                             shifts_weight.append(1)
#                             shifts_location.append((idx, op.location[1]))
#                         elif machine_state is False:
#                             if not shifts_weight:
#                                 continue
#                             else:
#                                 shifts_weight[-1] = shifts_weight[-1] + 1
#                     elif op.location[1] % 2 == 1:
#                         if machine_state is False:
#                             shift_counts += 1
#                             state_before_shifts.append(machine_state)
#                             machine_state = not machine_state
#                             shifts_weight.append(1)
#                             shifts_location.append((idx, op.location[1]))
#                         elif machine_state is True:
#                             if not shifts_weight:
#                                 continue
#                             else:
#                                 shifts_weight[-1] = shifts_weight[-1] + 1
#                     else:
#                         raise ValueError(f"Invalid operation location {op.location[1]} on circuit.")
#                 else:
#                     if op.location[0] % 2 == 0:
#                         if machine_state is True:
#                             shift_counts += 1
#                             state_before_shifts.append(machine_state)
#                             machine_state = not machine_state
#                             shifts_weight.append(1)
#                             shifts_location.append((idx, op.location[0]))
#                         elif machine_state is False:
#                             if not shifts_weight:
#                                 continue
#                             else:
#                                 shifts_weight[-1] = shifts_weight[-1] + 1
#                     elif op.location[0] % 2 == 1:
#                         if machine_state is False:
#                             shift_counts += 1
#                             state_before_shifts.append(machine_state)
#                             machine_state = not machine_state
#                             shifts_weight.append(1)
#                             shifts_location.append((idx, op.location[0]))
#                         elif machine_state is True:
#                             if not shifts_weight:
#                                 continue
#                             else:
#                                 shifts_weight[-1] = shifts_weight[-1] + 1
#                     else:
#                         raise ValueError(f"Invalid operation location {op.location[0]} on circuit.")
#     return shift_counts, shifts_weight, shifts_location, state_before_shifts


# def find_problematic_points_ver2(circuit: Circuit, lookahead: int):
#     """ Find problematic points using lookahead window where instantiation is needed based on dry scheduling"""
#     odd_even_points = []
#     problematic_points = []
#     circ_depth = circuit.num_cycles
#     for idx in range(circ_depth):
#         layer = circuit[idx]
#         for op in layer:
#             if op.gate == RZZGate() or op.gate == SwapGate():
#                 if op.location[0] > op.location[1]:
#                     odd_even_points.append((idx, op.location[1]))
#                 else:
#                     odd_even_points.append((idx, op.location[0]))
#     # Window-sliding
#     for point_idx in range(len(odd_even_points)):
#         if point_idx + lookahead < len(odd_even_points):
#             odd_count = len([odd_even_points[idx][1] for idx in range(point_idx, point_idx + lookahead)
#                              if odd_even_points[idx][1] % 2 == 1])
#             even_count = lookahead - odd_count
#         else:
#             odd_count = len([odd_even_points[idx][1] for idx in range(point_idx, len(odd_even_points))
#                              if odd_even_points[idx][1] % 2 == 1])
#             even_count = len(odd_even_points) - point_idx - odd_count
#         if even_count > odd_count and odd_even_points[point_idx][1] % 2 == 1:
#             problematic_points.append((odd_even_points[point_idx], 1))
#         elif odd_count > even_count and odd_even_points[point_idx][1] % 2 == 0:
#             problematic_points.append((odd_even_points[point_idx], 0))
#         else:
#             continue
#     return problematic_points


def alternate_circuit_structure(input_circuit: Circuit, state: MachineSchedulingState) -> Circuit:
    """ Alternate the given circuit to create a temple circuit without the need of shift gate"""
    tmp_circuit = Circuit(num_qudits=input_circuit.num_qudits)
    circ_depth = input_circuit.num_cycles
    for layer_idx in range(circ_depth):
        layer = input_circuit[layer_idx]
        for operation in layer:
            if operation.num_qudits == 1:
                tmp_circuit.append_gate(gate=U3Gate(), location=operation.location[0])
            elif operation.num_qudits == 2:
                if ((operation.location == (1, 2) or operation.location == (2, 1))
                        and state is MachineSchedulingState.EVEN):
                    for i in range(input_circuit.num_qudits):
                        tmp_circuit.append_gate(gate=U3Gate(), location=i)
                    new_op = Operation(gate=operation.gate, location=(0, 1))
                    tmp_circuit.append(new_op)
                    if tmp_circuit.num_qudits == 4:
                        new_op = Operation(gate=operation.gate, location=(2, 3))
                        tmp_circuit.append(new_op)
                    for i in range(input_circuit.num_qudits):
                        tmp_circuit.append_gate(gate=U3Gate(), location=i)
                elif ((operation.location == (0, 1) or operation.location == (1, 0)
                       or operation.location == (2, 3) or operation.location == (3, 2))
                      and state is MachineSchedulingState.ODD):
                    for i in range(input_circuit.num_qudits):
                        tmp_circuit.append_gate(gate=U3Gate(), location=i)
                    new_op = Operation(gate=operation.gate, location=(1, 2))
                    tmp_circuit.append(new_op)
                    for i in range(input_circuit.num_qudits):
                        tmp_circuit.append_gate(gate=U3Gate(), location=i)
                else:
                    for i in range(input_circuit.num_qudits):
                        tmp_circuit.append_gate(gate=U3Gate(), location=i)
                    tmp_circuit.append_gate(gate=RZZGate(), location=operation.location)
                    for i in range(input_circuit.num_qudits):
                        tmp_circuit.append_gate(gate=U3Gate(), location=i)
            else:
                raise ValueError("Unsupported circuit gate with big qudits")
    return tmp_circuit


def folding_ver2(circuit: Circuit):
    """ Automatic identify and re-instantiate trouble points """
    zones, zones_weights, machine_states = zone_circuit(circuit)
    potential_shift_locations = []
    potential_shift_state = []
    for zone_idx in range(1, len(zones)):
        if zones_weights[zone_idx] == 1:  # possible problematic points (Experiments needed)
            for point in zones[zone_idx]:
                if circuit.get_operation(point).gate == RZZGate():
                    potential_shift_locations.append(point)
                    potential_shift_state.append(machine_states[zone_idx])
    reversed_problem_points = potential_shift_locations[::-1]
    reversed_states = potential_shift_state[::-1]
    print("Problematic points: ", reversed_problem_points)
    for p, q in zip(reversed_problem_points, reversed_states):
        print("Point: ", p)
        print("Machine State:", q)
        min_point = min(p)
        max_point = max(p)
        denominator = 0 if q is MachineSchedulingState.EVEN else 1
        circuit_region = circuit.surround(point=p, num_qudits=4,
                                          filter=lambda region: (region.max_qudit > max_point and region.min_qudit < min_point)
                                                              or (region.min_qudit % 2 == denominator))
        print("Circuit region: ", circuit_region)
        folded_point = circuit.fold(circuit_region)
        op = circuit.get_operation(folded_point)
        target_unitary = op.get_unitary()
        old_block_circuit = op.gate._circuit  # type: ignore
        ### Instantiation
        # tmp_circuit = alternate_circuit_structure(old_block_circuit, q)
        # instantiated_circuit = tmp_circuit.instantiate(target=target_unitary, multistarts=5)
        ### Qsearch
        # print("Running Qsearch......")
        # qsearch_shift_pass = QSearchSynthesisPass(
        #     layer_generator=ShuttlingShiftGenerator(q),
        #     max_layer=6,
        #     heuristic_function=HeuristicSearch(
        #         heuristic_factor=2,
        #         qtm_machine=qtm_machine.H1
        #     ),
        # )
        # sub_workflow = [qsearch_shift_pass]
        # with Compiler() as compiler:
        #     search_circuit = compiler.compile(old_block_circuit, sub_workflow)
        distance = instantiated_circuit.get_unitary().get_distance_from(target_unitary, 2)
        print("Distance between instantiation and target unitary", distance)
        if distance < 1e-8:
            circuit.replace_with_circuit(folded_point, instantiated_circuit)
            print("Successfully replace the problem point with instantiation")
        circuit.unfold_all()
    return circuit


def main():
    # Configuration Setup
    # enable_logging(True)
    # qtm_machine = QtmMachine.H1

    # Load circuit
    circuit_name = "adder9"
    print(f"Scheduling {circuit_name}.")
    circuit = Circuit.from_file(
        "experiments/results/experiment_circuits/output_circuits/"
        f"{circuit_name}_without_scheduling.qasm"
    )
    folding_ver2(circuit)
    #     #
    #     # # Find shift zones
    #     # num_shifts, shift_weights, shift_locations, states_before_shift = dry_scheduling(circuit)
    #     # print("Required shifts:", num_shifts)
    #     # print("Shift weights:", shift_weights)
    #     # print("Shift locations:", shift_locations)
    #     # print("State before shifts:", states_before_shift)


if __name__ == "__main__":
    main()

    # test_surround_filter()
    # test_surround_filter_hard()
### Folding then scheduling:

# problem_points = find_problematic_points_ver2(initial_circuit, lookahead=5)
# instantiated_circ = folding_ver2(target_circuit)
# print(instantiated_circ.to('qasm'))
# required_shifts, shift_w_lst, shift_location_lst, state_shift = dry_scheduling(initial_circuit)
# print("Required shifts:", required_shifts)
# print("Shift weights:", shift_w_lst)
# print("State before shifts:", state_shift)
# print(len(state_shift))
# print(len(shift_w_lst))
# print(f"Total of {len(problem_points)} problematic points")
# print("Problematic points: ", problem_points)
# reversed_problem_points = problem_points[::-1]


# test_zone_circuit()
# test_....
# test_....
