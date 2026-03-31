import copy
import ast
import math
import numpy as np
from collections import defaultdict
from bqskit import Circuit
from bqskit.shuttling.qccd import QCCDMachineModel


def _get_chain_sizes(ion_assignment, qccd_machine):
    """Count how many ions currently sit in each trap."""
    chain_sizes = defaultdict(int)
    for pos in ion_assignment:
        trap_id = qccd_machine.get_trap_id(pos)
        if trap_id is not None:
            chain_sizes[trap_id] += 1
    return dict(chain_sizes)


def _fit_A(chain_size):
    """
    Empirical A-fit used in the older QCCDSim analyzer code.
    The paper only states A ∝ N / ln(N); this keeps the repo-style fit.
    """
    if chain_size <= 1:
        return 1e-4
    return max(1e-4, 1e-4 * chain_size / math.log(chain_size) - 5.3e-4)


def _gate_fidelity_for_op(
    op,
    ion_assignment,
    initial_mapping,
    qccd_machine,
    trap_heating,
    background_heating_rate=1.0,
):
    """
    Compute one two-qubit gate fidelity:
        F = 1 - Gamma * tau - A * (2 * nbar + 1)
    using the current trap heating as nbar.
    """
    p1 = ion_assignment[initial_mapping.index(op.location[0])]
    p2 = ion_assignment[initial_mapping.index(op.location[1])]
    trap_id_1 = qccd_machine.get_trap_id(p1)
    trap_id_2 = qccd_machine.get_trap_id(p2)

    # If they are not co-located, this scheduler still assigns a fallback gate time.
    # Fidelity is charged to whichever trap is available.
    if trap_id_1 is None and trap_id_2 is None:
        trap_id = None
        chain_size = 2
    else:
        trap_id = trap_id_1 if trap_id_1 is not None else trap_id_2
        chain_sizes = _get_chain_sizes(ion_assignment, qccd_machine)
        chain_size = max(2, chain_sizes.get(trap_id, 2))

    tau = qccd_machine.two_qudit_gate_time(p1=p1, p2=p2)  # seconds
    # In the older analyzer, x1 = gate_time_est / 1e6 with radial_heating_rate = 1
    # We keep the same scaling convention here.
    x1 = float(background_heating_rate * tau / 1e6)

    nbar = 0.0 if trap_id is None else trap_heating.get(trap_id, 0.0)
    A = _fit_A(chain_size)
    x2 = float(A * (2.0 * nbar + 1.0))

    fidelity = max(1e-4, 1.0 - x1 - x2)
    return fidelity, x1, x2


def _find_single_moved_ion(before_assignment, after_assignment):
    """Return the unique moved ion index if exactly one ion moved; else None."""
    diff_idx = [
        i for i, (b, a) in enumerate(zip(before_assignment, after_assignment))
        if b != a
    ]
    if len(diff_idx) != 1:
        return None
    return diff_idx[0]


def _estimate_num_segments(move_1, move_2):
    """
    Best-effort segment count from the instruction text.
    If the move endpoints are distinct, assume at least one segment.
    """
    if move_1 == move_2:
        return 0
    return 1


def _apply_move_heating(
    before_assignment,
    after_assignment,
    initial_mapping,
    qccd_machine,
    trap_heating,
    ion_carry_heating,
    move_1,
    move_2,
    k1=0.1,
    k2=0.01,
):
    """
    Approximate paper-style heating at the scheduler level.

    Because this IR only exposes `Move`, not explicit Split/Move/Merge,
    we treat an inter-trap move as:
        split from source trap  -> +k1 to source chain
        shuttle over segments   -> +k2 * (#segments) to moving ion
        merge into dest trap    -> +(carried heating + k1) to destination chain
    """
    moved_idx = _find_single_moved_ion(before_assignment, after_assignment)
    if moved_idx is None:
        return

    logical_ion = initial_mapping[moved_idx]
    old_pos = before_assignment[moved_idx]
    new_pos = after_assignment[moved_idx]

    src_trap = qccd_machine.get_trap_id(old_pos)
    dst_trap = qccd_machine.get_trap_id(new_pos)
    num_segments = _estimate_num_segments(move_1, move_2)

    # Shuttle contribution to the moved ion.
    ion_carry_heating[logical_ion] += k2 * num_segments

    # Only charge split/merge if the ion actually changes trap.
    if src_trap is not None and dst_trap is not None and src_trap != dst_trap:
        # Split source chain.
        trap_heating[src_trap] += k1

        # Merge into destination chain.
        trap_heating[dst_trap] += ion_carry_heating[logical_ion] + k1
        ion_carry_heating[logical_ion] = 0.0


def schedule_QCCD_w_fid(
    instructions_list: list,
    circuit: Circuit,
    initial_mapping: list,
    initial_ion_assignment: dict,
    qccd_machine: QCCDMachineModel,
    parallel: bool = True,
    k1: float = 0.1,
    k2: float = 0.01,
    background_heating_rate: float = 1.0,
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
    shuttling_time = 0.0
    execution_time = 0.0
    next_gate = None
    just_executed = False
    first_time_executed = True

    # --- NEW: motional heating + fidelity bookkeeping ---
    trap_heating = {
        trap.id: 0.0 for trap in qccd_machine.physical_graph.trap_list
    }
    ion_carry_heating = {logical: 0.0 for logical in initial_mapping}
    log_fidelity = 0.0
    gate_fidelities = []
    f_background_term = []
    f_mode_term = []

    print("Total number of cycles: ", circuit.num_cycles)

    def flush_parallel_moves():
        nonlocal runtime, shuttling_time, parallization_moves
        if not parallization_moves:
            return

        parallelable_moves = []
        serial_moves = []
        for move in parallization_moves:
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
                        if (
                            move[0][0] in parallelable_moves[serial_move_idx][-1][0]
                            or move[0][1] in parallelable_moves[serial_move_idx][-1][0]
                        ):
                            parallelable_moves[serial_move_idx].append(move)
                            added_flg = True
                            break
                    if not added_flg:
                        serial_moves.append(move)
        parallelable_moves.append(serial_moves)

        serial_moves_runtime = []
        for serial_move in parallelable_moves:
            tmp_runtime = 0.0
            for move in serial_move:
                tmp_runtime += move[1]
            serial_moves_runtime.append(tmp_runtime)

        runtime += max(serial_moves_runtime)
        shuttling_time += max(serial_moves_runtime)

        # Existing scheduler behavior:
        runtime -= 40e-6
        parallization_moves = []

    def flush_parallel_ops():
        nonlocal runtime, execution_time, executing_duration
        nonlocal ion_assignment, parallization_ops, ion_assignment_to_be_update
        nonlocal executing_blocks, log_fidelity, gate_fidelities
        nonlocal f_background_term, f_mode_term

        if not parallization_ops:
            return

        executable_dict = {
            qccd_machine.physical_graph.trap_list[i].id: 0
            for i in range(len(qccd_machine.physical_graph.trap_list))
        }
        ops_space = set()
        block_runtime = []

        for block in parallization_ops:
            for op in block:
                trap_id_op = [
                    qccd_machine.get_trap_id(
                        ion_assignment[initial_mapping.index(i)]
                    )
                    for i in op.location
                ]
                for i in op.location:
                    ops_space.add(ion_assignment[initial_mapping.index(i)])

                trap_id_op = set(trap_id_op)
                if len(trap_id_op) > 1 or list(trap_id_op)[0] is None:
                    trap = trap_id_op.pop()
                    if trap is not None:
                        executable_dict[trap] += 120e-6
                    # elif list(trap_id_op) == []:
                    #     pass
                    else:
                        executable_dict[trap_id_op.pop()] += 120e-6
                else:
                    executable_dict[trap_id_op.pop()] += qccd_machine.two_qudit_gate_time(
                        p1=ion_assignment[initial_mapping.index(op.location[0])],
                        p2=ion_assignment[initial_mapping.index(op.location[1])],
                    )
                    

                # --- NEW: charge gate fidelity here ---
                fidelity, x1, x2 = _gate_fidelity_for_op(
                    op=op,
                    ion_assignment=ion_assignment,
                    initial_mapping=initial_mapping,
                    qccd_machine=qccd_machine,
                    trap_heating=trap_heating,
                    background_heating_rate=background_heating_rate,
                )
                log_fidelity += math.log(fidelity)
                gate_fidelities.append(fidelity)
                f_background_term.append(x1)
                f_mode_term.append(x2)

            block_runtime.append(max(executable_dict.values()))

        runtime += max(block_runtime)
        execution_time += max(block_runtime)
        executing_duration = max(block_runtime)

        if ion_assignment_to_be_update:
            ion_assignment = copy.copy(ion_assignment_to_be_update[-1])

        parallization_ops = []
        ion_assignment_to_be_update = []
        executing_blocks = list(ops_space)

    for instruction in instructions_list:
        instruction_parts = instruction[0].split(' ')

        if instruction_parts[0] == 'Execute':
            if first_time_executed:
                first_time_executed = False

            # --- existing move flush, now preserved in helper ---
            flush_parallel_moves()

            if num_cycles != 0:
                if not just_executed:
                    if ion_assignment_to_be_update:
                        ion_assignment = copy.copy(ion_assignment_to_be_update[-1])
                    ion_assignment_to_be_update = []
                    executing_blocks = []
                    drop_out_duration = 0.0

            gate_block = []
            while next_gate != 'barrier':
                for op in circuit[num_cycles]:
                    if op.num_qudits == 2 and not parallel:
                        gate_time = qccd_machine.two_qudit_gate_time(
                            p1=ion_assignment[initial_mapping.index(op.location[0])],
                            p2=ion_assignment[initial_mapping.index(op.location[1])],
                        )
                        runtime += gate_time
                        execution_time += gate_time

                        # --- NEW: serial fidelity accounting ---
                        fidelity, x1, x2 = _gate_fidelity_for_op(
                            op=op,
                            ion_assignment=ion_assignment,
                            initial_mapping=initial_mapping,
                            qccd_machine=qccd_machine,
                            trap_heating=trap_heating,
                            background_heating_rate=background_heating_rate,
                        )
                        log_fidelity += math.log(fidelity)
                        gate_fidelities.append(fidelity)
                        f_background_term.append(x1)
                        f_mode_term.append(x2)

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

            if not parallel:
                ion_assignment = ast.literal_eval(instruction[-1])
            else:
                ion_assignment_to_be_update.append(ast.literal_eval(instruction[-1]))

            just_executed = True

        elif instruction_parts[0] == 'Move':
            just_executed = False

            # --- existing parallel-op flush, now preserved in helper ---
            flush_parallel_ops()

            cost_part = instruction[2].split(' ')
            cost = float(cost_part[1])
            new_assignment = ast.literal_eval(instruction[1])

            if parallel:
                if instruction_parts[1][1:-1].isdigit():
                    move_1 = int(instruction_parts[1][1:-1])
                else:
                    move_1 = eval(instruction_parts[1][1:-1])

                if instruction_parts[2][:-1].isdigit():
                    move_2 = int(instruction_parts[2][:-1])
                else:
                    move_2 = eval(instruction_parts[2][:-1])

                # --- NEW: apply heating BEFORE storing the new assignment ---
                base_assignment = (
                    copy.copy(ion_assignment_to_be_update[-1])
                    if ion_assignment_to_be_update
                    else copy.copy(ion_assignment)
                )
                _apply_move_heating(
                    before_assignment=base_assignment,
                    after_assignment=new_assignment,
                    initial_mapping=initial_mapping,
                    qccd_machine=qccd_machine,
                    trap_heating=trap_heating,
                    ion_carry_heating=ion_carry_heating,
                    move_1=move_1,
                    move_2=move_2,
                    k1=k1,
                    k2=k2,
                )

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
                if executing_blocks == [] and executing_duration == 0.0 and first_time_executed:
                    drop_out_flag = True

                if not drop_out_flag:
                    parallization_moves.append(((move_1, move_2), cost))
                else:
                    drop_out_duration += cost

                ion_assignment_to_be_update.append(new_assignment)

            else:
                # serial move
                move_1 = move_2 = 0
                _apply_move_heating(
                    before_assignment=copy.copy(ion_assignment),
                    after_assignment=new_assignment,
                    initial_mapping=initial_mapping,
                    qccd_machine=qccd_machine,
                    trap_heating=trap_heating,
                    ion_carry_heating=ion_carry_heating,
                    move_1=move_1,
                    move_2=move_2,
                    k1=k1,
                    k2=k2,
                )
                ion_assignment = new_assignment
                runtime += cost
                shuttling_time += cost
        else:
            raise ValueError(
                f"Instruction must be 'Execute' or 'Move'. But return {instruction_parts[0]}"
            )

    # --- NEW: flush tail blocks so the last execute/move contributes ---
    flush_parallel_ops()
    flush_parallel_moves()

    if ion_assignment_to_be_update:
        ion_assignment = copy.copy(ion_assignment_to_be_update[-1])

    app_fidelity = math.exp(log_fidelity) if gate_fidelities else 1.0
    shuttling_profile = (
        shuttling_time / (shuttling_time + execution_time)
        if (shuttling_time + execution_time) > 0
        else 0.0
    )

    return {
        "runtime": runtime,
        "shuttling_profile": shuttling_profile,
        "application_fidelity": app_fidelity,
        "trap_heating": trap_heating,
        "gate_fidelities": gate_fidelities,
        "f_background_term": f_background_term,
        "f_mode_term": f_mode_term,
        "final_ion_assignment": ion_assignment,
    }

if __name__ == "__main__":
    import pickle

    circuit_lst = [
        "QAOA_wsq_8_compiled",
        #"QuantumVolume_16",
        "QFT_wsq_8_compiled",
        "TFIM_wsq_n8_s100_compiled",
        "TFXY_wsq_n8_s100_compiled",
        #"QAOA_20_compiled",
        #"QuantumVolume_20",
        #"QFT_20_compiled"
    ]

    architecture_lst = [
        "H",
        "G2x3"
    ]

    parameter_set = {
        "H": [["2", "3"]],
        "G2x3": [["2", "3"]],
    }
    all_variance = []
    num_layout = 2
    all_runtime = []
    for circuit_idx in range(len(circuit_lst)):
        for architecture in architecture_lst:
            parameter = parameter_set[architecture][0] if circuit_idx < 5 else parameter_set[architecture][1]
            for param_idx in range(len(parameter)):
                param = parameter[param_idx]
                print(f"SHAPER_{circuit_lst[circuit_idx]}_{architecture}_{param}_{num_layout}")
                shuttling_time = []
                fidelity = [] 
                for idx in range(1, 6):
                    file_name = f"SHAPER_{circuit_lst[circuit_idx]}_idx{idx}_{architecture}_{param}_{num_layout}"
                    qasm_name = f"{circuit_lst[circuit_idx]}_idx{idx}_{architecture}_{param}_{num_layout}"
                    qasm_result_filename = f"/work/acslab/users/baobach/bqskit-shuttling/bqskit/shuttling/qccd/paper_result/{file_name}.pkl"
                    with open(
                            qasm_result_filename,
                            "rb") as input_file:
                        stored_data = data = pickle.load(input_file)
                    (runtime, compile_time, instruction_lst, gate_counts,
                     initial_ion_assignment, initial_mapping, final_mapping, machine_model) = stored_data
                    output_circuit = Circuit.from_file(f"/work/acslab/users/baobach/bqskit-shuttling/bqskit/shuttling/qccd/paper_result/{qasm_name}.qasm")
                    result = schedule_QCCD(instructions_list=instruction_lst,
                                            circuit=output_circuit,
                                            initial_mapping=initial_mapping,
                                            initial_ion_assignment=initial_ion_assignment,
                                            qccd_machine=machine_model)
                    shuttling_time.append(result["runtime"]/1e-6)
                    fidelity.append(result["application_fidelity"])
                all_runtime.append(runtime_lst[int(np.argmin(shuttling_time))])
                all_fidelity.append(fidelity[int(np.argmin(shuttling_time))])
                print(f"Min shuttling time:  {np.min(shuttling_time)}")
                print(f"Fidelity:  {fidelity[int(np.argmin(shuttling_time))]}")
    print(np.average(all_runtime))
    print(np.average(all_fidelity))
