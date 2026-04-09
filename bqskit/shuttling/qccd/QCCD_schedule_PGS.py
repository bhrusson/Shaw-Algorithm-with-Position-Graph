from __future__ import annotations

# Legacy scheduler kept for reference only.
# Active compare and grid flows now use bqskit.shuttling.QCCD_schedule_new.

import ast
import re
from functools import lru_cache

from bqskit import Circuit
from bqskit.shuttling.qccd.QCCD_machine_PGS import QCCDMachineModel


def _normalize_assignment(
    assignment: dict[int, int],
) -> dict[int, int]:
    return {
        int(logical): int(position)
        for logical, position in assignment.items()
    }


@lru_cache(maxsize=4096)
def _literal_eval_scheduler_payload(payload: str):
    try:
        return ast.literal_eval(payload)
    except (SyntaxError, ValueError):
        sanitized = re.sub(
            r'\b(?:np|numpy)\.int(?:8|16|32|64)?\(\s*(-?\d+)\s*\)',
            r'\1',
            payload,
        )
        sanitized = re.sub(
            r'\bint\(\s*(-?\d+)\s*\)',
            r'\1',
            sanitized,
        )
        return ast.literal_eval(sanitized)


@lru_cache(maxsize=4096)
def _parse_assignment_items(
    assignment_text: str,
) -> tuple[tuple[int, int], ...]:
    assignment = _literal_eval_scheduler_payload(assignment_text)
    if not isinstance(assignment, dict):
        raise ValueError(f'Invalid assignment payload: {assignment_text}.')
    return tuple(sorted(
        (int(logical), int(position))
        for logical, position in assignment.items()
    ))


def _parse_assignment_text(assignment_text: str) -> dict[int, int]:
    return dict(_parse_assignment_items(assignment_text))


def _parse_move_positions(instruction_text: str) -> tuple[int, int]:
    move = _literal_eval_scheduler_payload(
        instruction_text.removeprefix('Move ').strip(),
    )
    if not isinstance(move, tuple) or len(move) != 2:
        raise ValueError(f'Invalid move instruction: {instruction_text}.')
    return int(move[0]), int(move[1])


def schedule_QCCD_PGS(
    instructions_list: list,
    circuit: Circuit,
    initial_mapping: list,
    initial_ion_assignment: dict,
    qccd_machine: QCCDMachineModel,
    parallel: bool = True,
):
    """
    Schedule a QCCD circuit emitted by the PGS passes.

    Unlike the legacy scheduler, this version treats ``op.location`` as
    already-physical positions in the QCCD position graph and keeps the live
    placement state in ``PositionGraphState`` snapshots.
    """
    del initial_mapping

    runtime = 0.0
    num_cycles = 0
    current_assignment = _normalize_assignment(initial_ion_assignment)
    pending_assignment_updates: list[dict[int, int]] = []
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

    for instruction in instructions_list:
        instruction_parts = instruction[0].split(' ')
        if instruction_parts[0] == 'Execute':
            if first_time_executed:
                first_time_executed = False
            if parallization_moves:
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
                                if (move[0][0] in parallelable_moves[serial_move_idx][-1][0] or
                                        move[0][1] in parallelable_moves[serial_move_idx][-1][0]):
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
                runtime -= 5e-6
                parallization_moves = []
            if num_cycles != 0 and not just_executed and pending_assignment_updates:
                current_assignment = pending_assignment_updates[-1]
                pending_assignment_updates = []
                executing_blocks = []
                drop_out_duration = 0.0

            gate_block = []
            while next_gate != 'barrier':
                for op in circuit[num_cycles]:
                    if op.num_qudits == 2 and not parallel:
                        gate_time = qccd_machine.two_qudit_gate_time(
                            p1=op.location[0],
                            p2=op.location[1],
                        )
                        runtime += gate_time
                        execution_time += gate_time
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
                current_assignment = _parse_assignment_text(instruction[-1])
            else:
                pending_assignment_updates.append(
                    _parse_assignment_text(instruction[-1]),
                )
            just_executed = True

        elif instruction_parts[0] == 'Move':
            just_executed = False
            if parallization_ops:
                executable_dict = {qccd_machine.physical_graph.trap_list[i].id: 0 for i in
                                   range(len(qccd_machine.physical_graph.trap_list))}
                ops_space = set()
                block_runtime = []
                for block in parallization_ops:
                    for op in block:
                        trap_id_op = [qccd_machine.get_trap_id(pos) for pos in op.location]
                        for pos in op.location:
                            ops_space.add(pos)
                        trap_id_op = set(trap_id_op)
                        if len(trap_id_op) > 1 or list(trap_id_op)[0] is None:
                            trap = trap_id_op.pop()
                            if trap is not None:
                                executable_dict[trap] += 100e-6
                            else:
                                executable_dict[trap_id_op.pop()] += 100e-6
                        else:
                            executable_dict[trap_id_op.pop()] += qccd_machine.two_qudit_gate_time(
                                p1=op.location[0],
                                p2=op.location[1],
                            )
                    block_runtime.append(max(executable_dict.values()))

                runtime += max(block_runtime)
                execution_time += max(block_runtime)
                executing_duration = max(block_runtime)
                current_assignment = pending_assignment_updates[-1]
                parallization_ops = []
                pending_assignment_updates = []
                executing_blocks = list(ops_space)

            cost_part = instruction[2].split(' ')
            cost = float(cost_part[1])
            if parallel:
                move_1, move_2 = _parse_move_positions(instruction[0])
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
                if drop_out_flag:
                    drop_out_duration += cost
                pending_assignment_updates.append(
                    _parse_assignment_text(instruction[1]),
                )
            else:
                current_assignment = _parse_assignment_text(instruction[1])
                runtime += cost
                shuttling_time += cost
            num_cycles += 1
        else:
            raise ValueError(
                f"Instruction must be 'Execute' or 'Move'. But return {instruction_parts[0]}",
            )

    _ = current_assignment
    total_active_time = shuttling_time + execution_time
    shuttling_share = 0.0 if total_active_time == 0.0 else shuttling_time / total_active_time
    return runtime, shuttling_share
