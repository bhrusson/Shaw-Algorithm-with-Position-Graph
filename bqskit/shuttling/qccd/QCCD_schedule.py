import copy
import ast
import numpy as np
from bqskit import Circuit
from bqskit.shuttling.qccd import QCCDMachineModel


def schedule_QCCD(
        instructions_list: list,
        circuit: Circuit,
        initial_mapping: list,
        initial_ion_assignment: dict,
        qccd_machine: QCCDMachineModel,
        parallel: bool = True
):
    runtime = 0.0
    num_cycles = 0
    ion_assignment = copy.copy(initial_ion_assignment)
    ion_assignment_to_be_update = []
    parallization_ops = []
    parallization_moves = []
    executing_blocks = []
    drop_out_duration = 0.0
    executing_duration = 0.0
    next_gate = None
    just_executed = False
    #print(f"Start scheduling QCCD.... Parallization: {parallel}")
    parallel_moves_w_gate_execution = []
    for instruction in instructions_list:
        #print("Instruction {}".format(instruction))
        instruction_parts = instruction[0].split(' ')
        # print("Ion assignment {}".format(ion_assignment))
        # print("instruction_parts {}".format(instruction_parts))
        if instruction_parts[0] == 'Execute':
            """
                If the instruction is 'Execute', look at the circuit and try to execute it until meet a barrier.
            """
            #print("Current num cycle {}".format(num_cycles))
            if parallization_moves:
                #print("Perform parallel moves ....")
                parallelable_moves = []
                serial_moves = []
                for move in parallization_moves:
                    #print("move {}".format(move))
                    if not serial_moves:
                        serial_moves.append(move)
                    else:
                        if move[0][0] in serial_moves[-1][0] or move[0][1] in serial_moves[-1][0]:
                            serial_moves.append(move)
                        else:
                            parallelable_moves.append(serial_moves)
                            serial_moves = []
                            added_flg = False
                            for serial_move_idx in range(len(parallelable_moves)):
                                if (move[0][0] in parallelable_moves[serial_move_idx][-1][0] or
                                        move[0][1] in parallelable_moves[serial_move_idx][-1][0]):
                                    parallelable_moves[serial_move_idx].append(move)
                                    added_flg = True
                                    break
                            if not added_flg:
                                serial_moves.append(move)
                parallelable_moves.append(serial_moves)
                #print("parallelable_moves dict: ", parallelable_moves)
                serial_moves_runtime = []
                for serial_move in parallelable_moves:
                    tmp_runtime = 0.0
                    for move in serial_move:
                        tmp_runtime += move[1]
                    serial_moves_runtime.append(tmp_runtime)
                # ion_assignment = copy.copy(ion_assignment_to_be_update[-1])
                # print("Ion assignment is updated: {}".format(ion_assignment))
                runtime += max(serial_moves_runtime)
                runtime -= 5e-6
                parallization_moves = []
                #print("Time stamp after parallel moves: {}".format(runtime))
            if num_cycles != 0:
                if not just_executed:
                    ion_assignment = copy.copy(ion_assignment_to_be_update[-1])
                    #print("Ion assignment is updated: {}".format(ion_assignment))
                    ion_assignment_to_be_update = []
                    executing_blocks = []
                    drop_out_duration = 0.0
            gate_block = []
            while next_gate != 'barrier':
                for op in circuit[num_cycles]:
                    #print("Operation: ", op)
                    if op.num_qudits == 2 and not parallel:
                        runtime += qccd_machine.two_qudit_gate_time(
                            p1=ion_assignment[initial_mapping.index(op.location[0])],
                            p2=ion_assignment[initial_mapping.index(op.location[1])], )
                    elif op.num_qudits == 2 and parallel:
                        gate_block.append(op)
                    else:
                        continue
                if num_cycles + 1 >= circuit.num_cycles:
                    next_gate = 'barrier'
                else:
                    next_gate = circuit[num_cycles + 1][0].gate.name
                num_cycles += 1
            parallization_ops.append(gate_block)
            num_cycles += 1
            next_gate = None
            # Update ion assignment after selected the best permutation
            if not parallel:
                ion_assignment = ast.literal_eval(instruction[-1])
            else:
                ion_assignment_to_be_update.append(ast.literal_eval(instruction[-1]))
                #print("Ion assignment tobe is updated: {}".format(ion_assignment_to_be_update[-1]))
            just_executed = True
        elif instruction_parts[0] == 'Move':
            """
                If the instruction is 'Move', move the according ion and update the ion assignment.
            """
            just_executed = False
            if parallization_ops:
                # print("Perform parallel ops ....")
                # print("Ion assignment: {}".format(ion_assignment))
                # print("Initial mapping: {}".format(initial_mapping))
                executable_dict = {qccd_machine.physical_graph.trap_list[i].id: 0 for i in
                                   range(len(qccd_machine.physical_graph.trap_list))}
                ops_space = set()
                block_runtime = []
                for block in parallization_ops:
                    for op in block:
                        # print("Operation: ", op)
                        # print("P1: ", ion_assignment[initial_mapping.index(op.location[0])])
                        # print("P2: ", ion_assignment[initial_mapping.index(op.location[1])])
                        trap_id_op = [qccd_machine.get_trap_id(ion_assignment[initial_mapping.index(i)]) for i in
                                      op.location]
                        for i in op.location:
                            ops_space.add(ion_assignment[initial_mapping.index(i)])
                        #print("Trap id: ", trap_id_op)
                        trap_id_op = set(trap_id_op)
                        if len(trap_id_op) > 1 or list(trap_id_op)[0] is None:
                            #print(f"More than one trap id....{trap_id_op}")
                            trap = trap_id_op.pop()
                            if trap is not None:
                                executable_dict[trap] += 100e-6
                            else:
                                executable_dict[trap_id_op.pop()] += 100e-6
                            #raise ValueError(f"More than one trap id....{trap_id_op}")
                        else:
                            executable_dict[trap_id_op.pop()] += qccd_machine.two_qudit_gate_time(
                                p1=ion_assignment[initial_mapping.index(op.location[0])],
                                p2=ion_assignment[initial_mapping.index(op.location[1])], )
                    block_runtime.append(max(executable_dict.values()))
                runtime += max(block_runtime)
                executing_duration = max(block_runtime)
                ion_assignment = copy.copy(ion_assignment_to_be_update[-1])
                #print("Updated ion assignment: {}".format(ion_assignment))
                parallization_ops = []
                ion_assignment_to_be_update = []
                executing_blocks = list(ops_space)
            cost_part = instruction[2].split(' ')
            cost = float(cost_part[1])
            if parallel:
                if instruction_parts[1][1:-1].isdigit():
                    move_1 = int(instruction_parts[1][1:-1])
                else:
                    move_1 = eval(instruction_parts[1][1:-1])
                if instruction_parts[2][:-1].isdigit():
                    move_2 = int(instruction_parts[2][:-1])
                else:
                    move_2 = eval(instruction_parts[2][:-1])
                drop_out_flag = True
                if move_1 not in executing_blocks and move_2 not in executing_blocks:
                    for move in parallization_moves:
                        if move_1 in move[0] or move_2 in move[0]:
                            drop_out_flag = False
                else:
                    drop_out_flag = False
                if drop_out_duration > executing_duration:
                    drop_out_flag = False
                if not executing_blocks:
                    drop_out_flag = False
                if executing_blocks == [] and executing_duration == 0.0:
                    drop_out_flag = True
                # print("Drop out flag: ", drop_out_flag)
                # print("executing blocks: ", executing_blocks)
                # print("Current dropout duration: ", drop_out_duration)
                # print("Executing duration: ", executing_duration)
                if not drop_out_flag:
                    parallization_moves.append(((move_1, move_2), cost))
                if drop_out_flag:
                    drop_out_duration += cost
                ion_assignment_to_be_update.append(ast.literal_eval(instruction[1]))
                # print("Ion assignment tobe is updated: {}".format(ion_assignment_to_be_update[-1]))
            else:
                ion_assignment = ast.literal_eval(instruction[1])
                runtime += cost
            num_cycles += 1
            # print("Current runtime {}".format(runtime))
        else:
            raise ValueError(f"Instruction must be 'Execute' or 'Move'. But return {instruction_parts[0]}")
    return runtime


if __name__ == "__main__":
    import pickle

    circuit_lst = [
        "QAOA_16_compiled",
        "QuantumVolume_16",
        "QFT_16_compiled",
        "TFIM_n16_s100_compiled",
        "TFXY_n16_s100_compiled",
        "QAOA_20_compiled",
        "QuantumVolume_20",
        "QFT_20_compiled"
    ]

    architecture_lst = [
        "H",
        "G2x3"
    ]

    parameter_set = {
        "H": [["4", "5"], ["5", "6"]],
        "G2x3": [["3", "4"], ["4", "5"]],
    }


    class DummyClass:
        def __init__(self, *args, **kwargs):
            pass


    class IgnoringUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            try:
                return super().find_class(module, name)
            except Exception:
                # Return a dummy function or class to skip the problematic variable
                #print(f"Skipping problematic reference: {module}.{name}")
                return DummyClass  # Return a dummy class

    circuit_lst = ["QFT_20_compiled"]
    architecture_lst = ["Enchilada"]
    parameter_set = {"Enchilada": ["6"]}
    num_layout = 2
    for circuit_idx in range(len(circuit_lst)):
        for architecture in architecture_lst:
            parameter = parameter_set[architecture][0] if circuit_idx < 5 else parameter_set[architecture][1]
            for param_idx in range(len(parameter)):
                param = parameter[param_idx]
                # if param_idx == 0:
                #     continue
                file_name = f"SHAPER_{circuit_lst[circuit_idx]}_{architecture}_{param}_{num_layout}"
                qasm_name = f"{circuit_lst[circuit_idx]}_{architecture}_{param}_{num_layout}"
                qasm_result_filename = f"bqskit/shuttling/qccd/new_result/{file_name}.pkl"
                with open(
                        qasm_result_filename,
                        "rb") as input_file:
                    stored_data = data = IgnoringUnpickler(input_file).load()
                (runtime, compile_time, instruction_lst, output_circuit, gate_counts,
                 initial_ion_assignment, initial_mapping, final_mapping, machine_model) = stored_data
                output_circuit = Circuit.from_file(f"bqskit/shuttling/qccd/new_result/{qasm_name}.qasm")
                runtime = schedule_QCCD(instructions_list=instruction_lst,
                                        circuit=output_circuit,
                                        initial_mapping=initial_mapping,
                                        initial_ion_assignment=initial_ion_assignment,
                                        qccd_machine=machine_model)
                # print(f"File {file_name} ...")
                # print("Final runtime: ", runtime / 1e-6)
                print(round(runtime/1e-6))

    # qasm_result_filename = f"bqskit/shuttling/qccd/result/{file_name}.pkl"
    # (runtime, compile_time, instruction_lst, output_circuit, gate_counts,
    #  initial_ion_assignment, initial_mapping, final_mapping, machine_model) = stored_data
    # print("Number of instructions: {}".format(len(instruction_lst)))
    # for instruction in instruction_lst:
    #     print(instruction)
    # runtime = schedule_QCCD(instructions_list=instruction_lst,
    #                         circuit=output_circuit,
    #                         initial_mapping=initial_mapping,
    #                         initial_ion_assignment=initial_ion_assignment,
    #                         qccd_machine=machine_model)
    # print("Final runtime: ", runtime / 1e-6)
