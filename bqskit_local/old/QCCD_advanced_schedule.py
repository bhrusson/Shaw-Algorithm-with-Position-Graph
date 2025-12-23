import copy
import ast

from bqskit import Circuit
from bqskit.shuttling.qccd import QCCDMachineModel

class MoveEvent:
    def __init__(self,
                 move: tuple[int, int],
                 duration: float):
        self.move = move
        self.duration = duration


class ExecuteEvent:
    def __init__(self,
                 circuit: Circuit,
                 start_cycle: int,
                 executed_ions: list):
        self.circuit = circuit
        self.start_cycle = start_cycle
        self.executed_ions = executed_ions


class TimeSchedule:
    def __init__(self,
                 initial_ion_assignment: dict,
                 initial_mapping: list,
                 qccd_machine: QCCDMachineModel,
                 permutation_after_executed: dict):
        self.qccd_machine = qccd_machine
        self.list_of_events = []
        self.event_timings = []
        self.initial_mapping = initial_mapping
        self.latest_ion_assignment = initial_ion_assignment
        self.permutation_after_executed = permutation_after_executed
        self.ion_assignment_wrt_timing = {
            0.0: initial_ion_assignment
        }
        self.hold_ion_wrt_timing = {
            0.0: []
        }

    def return_final_runtime(self):
        return list(self.ion_assignment_wrt_timing)[-1]

    def add_execute_event(self, event: ExecuteEvent):
        self.list_of_events.append(event)
        starting_cycle = event.start_cycle
        block_duration = 0.0
        next_gate = event.circuit[starting_cycle][0]
        print("Next gate:", next_gate)
        print(f"Executing block {event.executed_ions} at starting cycle {starting_cycle}")
        while next_gate.gate.name != 'barrier':
            for op in event.circuit[starting_cycle]:
                if op.num_qudits == 2:
                    block_duration += self.qccd_machine.two_qudit_gate_time(
                        p1=self.latest_ion_assignment[self.initial_mapping.index(op.location[0])],
                        p2=self.latest_ion_assignment[self.initial_mapping.index(op.location[1])], )
                else:
                    continue
            starting_cycle += 1
            if starting_cycle >= event.circuit.num_cycles:
                break
            else:
                next_gate = event.circuit[starting_cycle][0]
        print("Final block duration: ", block_duration)
        timestamp = list(self.ion_assignment_wrt_timing.keys())[
            list(self.ion_assignment_wrt_timing.values()).index(self.latest_ion_assignment)]
        update_timestamp = block_duration / 1e-6 + timestamp
        update_assignment = self.permutation_after_executed[tuple(event.executed_ions)]
        self.ion_assignment_wrt_timing[update_timestamp] = update_assignment
        self.hold_ion_wrt_timing[update_timestamp] = [update_assignment[self.initial_mapping.index(ion)] for ion in event.executed_ions]
        self.latest_ion_assignment = update_assignment

    def add_move_event(self, event: MoveEvent):
        self.list_of_events.append(event)
        satisfied_timestamp = []
        assignment_idx = 0
        for assignment in self.ion_assignment_wrt_timing.values():
            if self.move_event_is_valid(ion_assignment=assignment,
                                        move=event.move,):
                satisfied_timestamp.append(list(self.ion_assignment_wrt_timing.keys())
                                           [assignment_idx])
            assignment_idx += 1
        """
            Mechanism to choose which assignment to parallel
        """
        sorted_timestamp = sorted(satisfied_timestamp)[::-1]
        print("Sorted timestamp: ", sorted_timestamp)
        selected_timestamp = sorted_timestamp[0]
        for timestamp in sorted_timestamp:
            if len(self.hold_ion_wrt_timing[timestamp]) == 3:
                if (event.move[0] not in self.hold_ion_wrt_timing[timestamp] and
                        event.move[1] not in self.hold_ion_wrt_timing[timestamp]):
                    assignment = self.ion_assignment_wrt_timing[timestamp]
                    selected_timestamp = list(self.ion_assignment_wrt_timing.keys())[list(
                        self.ion_assignment_wrt_timing.keys()).index(timestamp) - 1]
                else:
                    assignment = self.ion_assignment_wrt_timing[timestamp]
                    selected_timestamp = timestamp
                break
            else:
                if (event.move[0] not in self.hold_ion_wrt_timing[timestamp]) and (event.move[1] not in self.hold_ion_wrt_timing[timestamp]):
                    continue
                else:
                    assignment = self.ion_assignment_wrt_timing[timestamp]
                    selected_timestamp = timestamp
                    break
        print("Assignment: ", assignment)
        """
            Add move to schedule
        """
        # timestamp = list(self.ion_assignment_wrt_timing.keys())[
        #     list(self.ion_assignment_wrt_timing.values()).index(assignment)]
        self.event_timings.append(selected_timestamp)
        update_assignment = self.perform_event(ion_assignment=assignment,
                                               move=event.move, )
        update_timestamp = int(selected_timestamp + event.duration / 1e-6)
        self.ion_assignment_wrt_timing[update_timestamp] = update_assignment
        self.hold_ion_wrt_timing[update_timestamp] = list(event.move)
        if assignment == self.latest_ion_assignment:
            self.latest_ion_assignment = update_assignment
        # Todo: how to deal with validation of parallel move in this step leads to next step...

    def move_event_is_valid(self,
                            ion_assignment: dict,
                            move: tuple[int, int]):
        assignment = ion_assignment
        l1 = list(assignment.keys())[list(assignment.values()).index(move[0])] \
            if move[0] in list(assignment.values()) else None
        l2 = list(assignment.keys())[list(assignment.values()).index(move[1])] \
            if move[1] in list(assignment.values()) else None
        if l1 is None and l2 is None:
            return False
        trap_id_ions = [self.qccd_machine.get_trap_id(space) for space in move]
        if l1 is not None and l2 is not None:
            if trap_id_ions[0] is None or trap_id_ions[1] is None:
                return False
        return True

    def perform_event(self,
                      ion_assignment: dict,
                      move: tuple[int, int]):
        update_assignment = copy.copy(ion_assignment)
        l1 = list(ion_assignment.keys())[list(ion_assignment.values()).index(move[0])] \
            if move[0] in list(ion_assignment.values()) else None
        l2 = list(ion_assignment.keys())[list(ion_assignment.values()).index(move[1])] \
            if move[1] in list(ion_assignment.values()) else None
        if l1 is None and l2 is None:
            raise RuntimeError(f'The move {move} is not a valid move as there is no ion in these assignment.')
        if l1 is None:
            update_assignment[l2] = move[0]  # Move ion to the adjacent available space
        elif l2 is None:
            update_assignment[l1] = move[1]  # Move ion to the adjacent available space
        else:
            update_assignment[l1], update_assignment[l2] = move[1], move[0]  # Inner trap swap
        return update_assignment


class Schedule_QCCD:
    def __init__(self, instructions_list: list,
                 circuit: Circuit,
                 initial_mapping: list,
                 initial_ion_assignment: dict,
                 qccd_machine: QCCDMachineModel):
        self.instructions_list = instructions_list
        self.circuit = circuit
        self.initial_mapping = initial_mapping
        self.initial_ion_assignment = initial_ion_assignment
        self.qccd_machine = qccd_machine
        self.list_of_move_events = []
        self.list_of_execute_events = []
        self.permutated_after_executed = {}
        move_cycle = 0
        for instruction in instructions_list:
            print("Instruction: ", instruction)
            print("Circuit cycle: ", self.circuit[move_cycle])
            instruction_parts = instruction[0].split(' ')
            if instruction_parts[0] == 'Move':
                cost_part = instruction[2].split(' ')
                cost = float(cost_part[1])
                move_1 = int(instruction_parts[1][1:-1])
                move_2 = int(instruction_parts[2][:-1])
                self.list_of_move_events.append([(move_1, move_2), cost])
                move_cycle = move_cycle + 1
            elif instruction_parts[0] == 'Execute':
                executed_ions = []
                ion_idx = 0
                for ion in instruction_parts[2:]:
                    if ion_idx == 0:
                        executed_ions.append(int(ion[1:-1]))
                    else:
                        executed_ions.append(int(ion[:-1]))
                    ion_idx += 1
                block_of_gates = [executed_ions, move_cycle]
                next_gate = None
                while next_gate != 'barrier':
                    move_cycle = move_cycle + 1
                    if move_cycle + 1 >= circuit.num_cycles:
                        next_gate = 'barrier'
                    else:
                        next_gate = self.circuit[move_cycle][0].gate.name
                self.permutated_after_executed[tuple(executed_ions)] = ast.literal_eval(instruction[-1])
                self.list_of_execute_events.append(block_of_gates)
        print("List of move events: ", self.list_of_move_events)
        print("List of execute events: ", self.list_of_execute_events)
        self.schedule = TimeSchedule(qccd_machine=qccd_machine,
                                     initial_mapping=self.initial_mapping,
                                     initial_ion_assignment=self.initial_ion_assignment,
                                     permutation_after_executed=self.permutated_after_executed, )
        print("Physical to pos: ", self.qccd_machine.physical_to_position)

    def check_if_block_exectuable(self,
                                  ion_assignment: dict,
                                  block_of_gates: list):
        executed_ions = block_of_gates[0]
        mapped_ions = [self.initial_mapping.index(ion) for ion in executed_ions]
        print("Executed ions: ", mapped_ions)
        trap_per_ion = [self.qccd_machine.get_trap_id(ion_assignment[ion]) for ion in mapped_ions]
        print("Trap per ions: ", trap_per_ion)
        trap_per_ion = set(trap_per_ion)
        if len(trap_per_ion) == 1 and trap_per_ion.pop() is not None:
            return True
        else:
            return False

    def scheduling(self):
        latest_ion_assignment = self.initial_ion_assignment
        print("############################################")
        print("Start scheduling .......")
        print("Initial ion assignment: ", self.initial_ion_assignment)
        while self.list_of_execute_events:
            # print(self.schedule.ion_assignment_wrt_timing)
            print(latest_ion_assignment)
            if self.check_if_block_exectuable(
                    ion_assignment=latest_ion_assignment,
                    block_of_gates=self.list_of_execute_events[0]):
                print(f"Executing block {self.list_of_execute_events[0]}")
                block_of_gate = self.list_of_execute_events.pop(0)
                event = ExecuteEvent(circuit=self.circuit,
                                     start_cycle=block_of_gate[1],
                                     executed_ions=block_of_gate[0])
                self.schedule.add_execute_event(event)
                latest_ion_assignment = self.schedule.latest_ion_assignment
                print("Ion assignment get update to ", latest_ion_assignment)
            else:
                print(f"Performing move {self.list_of_move_events[0]}")
                move_event, cost = self.list_of_move_events.pop(0)
                event = MoveEvent(move=move_event,
                                  duration=cost)
                self.schedule.add_move_event(event)
                latest_ion_assignment = self.schedule.latest_ion_assignment
                print("Ion assignment get update to ", latest_ion_assignment)

    def return_runtime(self):
        return self.schedule.return_final_runtime()

