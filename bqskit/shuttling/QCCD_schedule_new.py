import pickle

import ast
import copy
import math
import re
from collections import defaultdict
from typing import Any, Dict, List, Tuple

from bqskit.ir.gates.barrier import BarrierPlaceholder


def schedule_qccd_from_instructions_v3(
    instruction_lst: List[list],
    initial_ion_assignment: Dict[int, int],
    machine_model,
    full_initial_ion_assignment: Dict[int, int] | None = None,
    circuit: Any | None = None,
    parallel: bool = True,
    background_heating_rate: float = 1.0,
    base_gate_fidelity: float = 0.992,
    min_gate_fidelity: float = 1e-4,
    validate_instruction_cost: bool = True,
    honeywell_mode: bool = True,
    intra_trap_swap_mode: str = "gate",   # "gate", "ion", "none"
    execute_location_mode: str = "auto",
):
    """
    Schedule directly from instruction_lst using the position graph, and
    also infer QCCDSim-style events + replay a QCCDSim-like fidelity model.

    Event inference:
      Execute at [q1, q2]   -> Gate
      trap -> segment       -> Split
      segment -> segment    -> Move
      segment -> trap       -> Merge
      trap -> trap (same trap) -> folded into the next Split as swap metadata
                                 via intra_trap_swap_mode

    Returns:
      runtime / shuttling profiles from the scheduler
      application_fidelity from inferred-event replay
      inferred_events for inspection
    """

    # ------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------
    def parse_assignment(x):
        return ast.literal_eval(x) if isinstance(x, str) else copy.deepcopy(x)

    def parse_execute_payload(text: str) -> List[int]:
        m = re.search(r"Execute\s+at\s+\[(.*)\]", text)
        if m is None:
            raise ValueError(f"Cannot parse execute instruction: {text}")
        payload = ast.literal_eval("[" + m.group(1).strip() + "]")
        return [int(value) for value in payload]

    def parse_move(text: str) -> tuple[int, int]:
        """
        Handles:
          Move (4, 5)
          Move (np.int64(5), 11)
          Move (np.int32(7), np.int64(12))
        """
        if not text.strip().startswith("Move"):
            raise ValueError(f"Cannot parse move instruction: {text}")

        payload = text.strip()[len("Move"):].strip()
        payload = re.sub(r"np\.int\d+\(\s*(-?\d+)\s*\)", r"\1", payload)

        try:
            u, v = ast.literal_eval(payload)
            return int(u), int(v)
        except Exception as e:
            raise ValueError(f"Cannot parse move instruction: {text}") from e

    def parse_cost(text: str) -> float:
        m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
        if m is None:
            raise ValueError(f"Cannot parse cost from: {text}")
        return float(m.group(0))

    def resolve_execute_qudits(
        payload: List[int],
        assignment: Dict[int, int],
    ) -> List[int]:
        if execute_location_mode not in {"auto", "logical", "physical"}:
            raise ValueError(
                "execute_location_mode must be one of {'auto', 'logical', 'physical'}",
            )

        inverse_assignment = {position: logical for logical, position in assignment.items()}
        assignment_keys = set(assignment.keys())
        assignment_values = set(assignment.values())

        if execute_location_mode == "logical":
            return [int(value) for value in payload]

        if execute_location_mode == "physical":
            return [int(inverse_assignment.get(value, value)) for value in payload]

        if all(value in assignment_values for value in payload):
            return [int(inverse_assignment[value]) for value in payload]
        if all(value in assignment_keys for value in payload):
            return [int(value) for value in payload]

        resolved: List[int] = []
        for value in payload:
            resolved.append(int(inverse_assignment.get(value, value)))
        return resolved

    def build_instruction_operation_blocks(circuit_obj: Any | None) -> List[List[Tuple[int, Any]]]:
        if circuit_obj is None or not hasattr(circuit_obj, "operations_with_cycles"):
            return []

        blocks: List[List[Tuple[int, Any]]] = []
        current_block: List[Tuple[int, Any]] = []

        for cycle, op in circuit_obj.operations_with_cycles():
            if isinstance(op.gate, BarrierPlaceholder):
                blocks.append(current_block)
                current_block = []
            else:
                current_block.append((cycle, op))

        if current_block:
            blocks.append(current_block)

        return blocks

    # ------------------------------------------------------------
    # Position-graph metadata
    # ------------------------------------------------------------
    def build_position_metadata():
        """
        Build:
          - position_kind[p] in {"trap", "segment"}
          - trap_owner[p] = trap_id for trap positions, else None
          - physical_owner[p] = name like "trap_0", "segment_1"
          - trap_slot_index[p] = local slot index within the trap
        """
        position_kind = {}
        trap_owner = {}
        physical_owner = {}
        trap_slot_index = {}

        physical_to_position = machine_model.physical_to_position

        for phys, pos_list in physical_to_position.items():
            if phys == "segment_space":
                continue

            positions = list(pos_list)

            if phys.startswith("trap_"):
                trap_id = int(phys.split("_")[1])
                positions_sorted = sorted(positions)
                for k, p in enumerate(positions_sorted):
                    position_kind[p] = "trap"
                    trap_owner[p] = trap_id
                    physical_owner[p] = phys
                    trap_slot_index[p] = k

            elif phys.startswith("segment_"):
                for p in positions:
                    position_kind[p] = "segment"
                    trap_owner[p] = None
                    physical_owner[p] = phys
                    trap_slot_index[p] = None

        return position_kind, trap_owner, physical_owner, trap_slot_index

    position_kind, trap_owner, physical_owner, trap_slot_index = build_position_metadata()

    # Use the machine's coupling graph exactly as given.
    raw_coupling_graph = getattr(machine_model, "coupling_graph", None)
    if raw_coupling_graph is None:
        position_graph = getattr(machine_model, "position_graph", None)
        move_graph = getattr(position_graph, "move_graph", None)
        if move_graph is None:
            raise AttributeError("Machine model is missing both coupling_graph and position_graph.move_graph.")
        raw_coupling_graph = move_graph.edge_list()

    coupling_edges = set()
    for e in raw_coupling_graph:
        u, v = tuple(e)
        coupling_edges.add((u, v))
        coupling_edges.add((v, u))

    # ------------------------------------------------------------
    # travel_time_map is a list[list] edge-cost matrix
    # ------------------------------------------------------------
    travel_time_map = machine_model.all_pair_travelling_time()

    def edge_cost(u: int, v: int, fallback: float = None) -> float:
        """
        travel_time_map[u][v] is the cost of the edge (u, v) in the position graph.
        """
        try:
            val = float(travel_time_map[u][v])
        except Exception:
            if fallback is not None:
                return fallback
            raise ValueError(f"Cannot read travel_time_map[{u}][{v}]")

        if val <= 0:
            if fallback is not None:
                return fallback
            raise ValueError(
                f"travel_time_map[{u}][{v}]={val}, which looks invalid for an active move edge."
            )

        return val

    def classify_move(u: int, v: int) -> str:
        ku, kv = position_kind[u], position_kind[v]

        if ku == "trap" and kv == "segment":
            return "split"
        if ku == "segment" and kv == "trap":
            return "merge"
        if ku == "segment" and kv == "segment":
            return "junction"
        if ku == "trap" and kv == "trap":
            if trap_owner[u] == trap_owner[v]:
                return "intra_trap"
            return "trap_to_trap_invalid"

        raise ValueError(f"Unknown move type for ({u}, {v}).")

    def pos_to_logical(assignment: Dict[int, int]) -> Dict[int, int]:
        return {pos: q for q, pos in assignment.items()}

    def ion_at_position(assignment: Dict[int, int], pos: int):
        return pos_to_logical(assignment).get(pos, None)

    # ------------------------------------------------------------
    # Gate helpers
    # ------------------------------------------------------------
    def get_chain_sizes(assignment: Dict[int, int]) -> Dict[int, int]:
        counts = defaultdict(int)
        for _, pos in assignment.items():
            if position_kind[pos] == "trap":
                counts[trap_owner[pos]] += 1
        return dict(counts)

    def fit_A(chain_size: int) -> float:
        # Same practical fit as the older analyzer code.
        if chain_size <= 1:
            return 1e-4
        return max(1e-4, 1e-4 * chain_size / math.log(chain_size) - 5.3e-4)

    def gate_info(q1: int, q2: int, assignment: Dict[int, int]):
        p1 = assignment[q1]
        p2 = assignment[q2]

        if position_kind[p1] != "trap" or position_kind[p2] != "trap":
            raise ValueError(
                f"Gate ({q1}, {q2}) is not executable: positions ({p1}, {p2}) are not both in traps."
            )

        t1 = trap_owner[p1]
        t2 = trap_owner[p2]
        if t1 != t2:
            raise ValueError(
                f"Gate ({q1}, {q2}) is not executable: positions ({p1}, {p2}) are in different traps."
            )

        duration = float(machine_model.two_qudit_gate_time(p1=p1, p2=p2))
        return {
            "trap_id": t1,
            "p1": p1,
            "p2": p2,
            "duration": duration,
            "tokens": {
                ("trap", t1),
                ("pos", p1),
                ("pos", p2),
            },
        }

    # ------------------------------------------------------------
    # Greedy parallel batching
    # ------------------------------------------------------------
    def batch_disjoint(tasks: List[dict]) -> List[List[dict]]:
        rounds = []
        used = []

        for task in tasks:
            placed = False
            for i in range(len(rounds)):
                if task["tokens"].isdisjoint(used[i]):
                    rounds[i].append(task)
                    used[i].update(task["tokens"])
                    placed = True
                    break
            if not placed:
                rounds.append([task])
                used.append(set(task["tokens"]))

        return rounds

    # ------------------------------------------------------------
    # Pending intra-trap reorder metadata
    # ------------------------------------------------------------
    def fresh_swap_meta():
        return {
            "swap_cnt": 0,
            "swap_hops": 0,
            "ion_hops": 0,
            "swap_pair": None,   # tuple(logical1, logical2)
        }

    pending_swap_meta = defaultdict(fresh_swap_meta)

    # ------------------------------------------------------------
    # Runtime state
    # ------------------------------------------------------------
    del full_initial_ion_assignment

    current_assignment = parse_assignment(initial_ion_assignment)
    instruction_operation_blocks = build_instruction_operation_blocks(circuit)
    if instruction_operation_blocks and len(instruction_operation_blocks) != len(instruction_lst):
        instruction_operation_blocks = []

    runtime = 0.0
    execution_time = 0.0
    shuttling_time_critical = 0.0
    shuttling_time_physical = 0.0

    move_type_counts = defaultdict(int)

    pending_execute = []
    pending_exposed_moves = []

    # overlap window created by the previous execute block
    overlap_budget = 0.0
    overlap_used = 0.0
    overlap_tokens = set()

    execute_rounds = []
    move_rounds = []

    # ------------------------------------------------------------
    # Inferred event list
    # ------------------------------------------------------------
    inferred_events = []
    next_event_id = 0

    def append_event(event_type: str, info: dict, before_assignment: Dict[int, int], after_assignment: Dict[int, int]):
        nonlocal next_event_id
        inferred_events.append({
            "id": next_event_id,
            "type": event_type,                  # Gate / Split / Move / Merge
            "info": copy.deepcopy(info),
            "before_assignment": copy.deepcopy(before_assignment),
            "after_assignment": copy.deepcopy(after_assignment),
        })
        next_event_id += 1

    # ------------------------------------------------------------
    # Flush helpers
    # ------------------------------------------------------------
    def flush_execute_block():
        nonlocal runtime, execution_time
        nonlocal overlap_budget, overlap_used, overlap_tokens
        nonlocal pending_execute

        if not pending_execute:
            overlap_budget = 0.0
            overlap_used = 0.0
            overlap_tokens = set()
            return

        tasks = []
        for item in pending_execute:
            tasks.append({
                "duration": item["duration"],
                "tokens": item["tokens"],
                "round_entry": item["round_entry"],
            })

        rounds = batch_disjoint(tasks) if parallel else [[t] for t in tasks]

        block_runtime = 0.0
        block_tokens = set()

        for rnd in rounds:
            round_dt = max(t["duration"] for t in rnd)
            block_runtime += round_dt

            round_log = []
            for t in rnd:
                round_log.append(t["round_entry"])
                block_tokens.update(t["tokens"])
            execute_rounds.append(round_log)

        # Emit inferred Gate events in instruction order.
        for item in pending_execute:
            for gate_event in item["gate_events"]:
                append_event(
                    "Gate",
                    gate_event,
                    item["before_assignment"],
                    item["after_assignment"],
                )

        runtime += block_runtime
        execution_time += block_runtime

        overlap_budget = block_runtime
        overlap_used = 0.0
        overlap_tokens = block_tokens

        final_assignment = copy.deepcopy(pending_execute[-1]["after_assignment"])
        current_assignment.clear()
        current_assignment.update(final_assignment)
        pending_execute = []

    def flush_exposed_moves():
        nonlocal runtime, shuttling_time_critical
        nonlocal pending_exposed_moves

        if not pending_exposed_moves:
            return

        rounds = batch_disjoint(pending_exposed_moves) if parallel else [[t] for t in pending_exposed_moves]

        for rnd in rounds:
            round_dt = max(t["cost"] for t in rnd)
            runtime += round_dt
            shuttling_time_critical += round_dt
            move_rounds.append([(t["move"], t["move_type"], t["cost"]) for t in rnd])

        pending_exposed_moves = []

    # ------------------------------------------------------------
    # Main scan: runtime + event inference
    # ------------------------------------------------------------
    def build_execute_info(
        payload: List[int],
        before_assignment: Dict[int, int],
        block_ops: List[Tuple[int, Any]],
    ) -> Dict[str, Any]:
        def resolve_block_locations(op_location: Any) -> Tuple[List[int], List[int]]:
            raw_values = [int(qudit) for qudit in op_location]

            if execute_location_mode == "physical":
                physical_positions = raw_values
                logical_qudits = [
                    ion_at_position(before_assignment, position)
                    for position in physical_positions
                ]
                if any(qudit is None for qudit in logical_qudits):
                    raise ValueError(
                        f"Cannot resolve physical execute locations {physical_positions} "
                        "to logical qudits in the current assignment.",
                    )
                return [int(qudit) for qudit in logical_qudits], physical_positions

            if execute_location_mode == "logical":
                logical_qudits = raw_values
                return logical_qudits, [before_assignment[q] for q in logical_qudits]

            if all(value in before_assignment.values() for value in raw_values):
                physical_positions = raw_values
                logical_qudits = [
                    ion_at_position(before_assignment, position)
                    for position in physical_positions
                ]
                if all(qudit is not None for qudit in logical_qudits):
                    return [int(qudit) for qudit in logical_qudits], physical_positions

            logical_qudits = raw_values
            return logical_qudits, [before_assignment[q] for q in logical_qudits]

        if not block_ops:
            logical_qudits = resolve_execute_qudits(payload, before_assignment)
            if len(logical_qudits) != 2:
                touched_positions = [
                    before_assignment[q]
                    for q in logical_qudits
                    if q in before_assignment
                ]
                return {
                    "duration": 0.0,
                    "tokens": {("pos", pos) for pos in touched_positions},
                    "round_entry": tuple(logical_qudits),
                    "gate_events": [],
                }

            q1, q2 = logical_qudits
            info = gate_info(q1, q2, before_assignment)
            return {
                "duration": info["duration"],
                "tokens": info["tokens"],
                "round_entry": (q1, q2, info["duration"]),
                "gate_events": [{
                    "ions": [q1, q2],
                    "trap": info["trap_id"],
                    "duration": info["duration"],
                }],
            }

        cycle_durations: Dict[int, float] = defaultdict(float)
        cycle_entries: Dict[int, List[Any]] = defaultdict(list)
        tokens = set()
        gate_events = []

        for cycle, op in block_ops:
            logical_qudits, physical_positions = resolve_block_locations(op.location)
            tokens.update({("pos", pos) for pos in physical_positions})

            for pos in physical_positions:
                if position_kind[pos] == "trap" and trap_owner[pos] is not None:
                    tokens.add(("trap", trap_owner[pos]))

            if len(logical_qudits) == 1:
                duration = float(machine_model.timing_data.get("sq_timings", 0.0))
                cycle_durations[cycle] = max(cycle_durations[cycle], duration)
                cycle_entries[cycle].append((logical_qudits[0], duration))
                continue

            if len(logical_qudits) != 2:
                raise ValueError(
                    f"Unsupported execute block arity {len(logical_qudits)} in {logical_qudits}.",
                )

            q1, q2 = logical_qudits
            info = gate_info(q1, q2, before_assignment)
            cycle_durations[cycle] = max(cycle_durations[cycle], info["duration"])
            cycle_entries[cycle].append((q1, q2, info["duration"]))
            gate_events.append({
                "ions": [q1, q2],
                "trap": info["trap_id"],
                "duration": info["duration"],
            })

        ordered_cycles = sorted(cycle_entries.keys())
        return {
            "duration": sum(cycle_durations[cycle] for cycle in ordered_cycles),
            "tokens": tokens,
            "round_entry": [cycle_entries[cycle] for cycle in ordered_cycles],
            "gate_events": gate_events,
        }

    for inst_index, inst in enumerate(instruction_lst):
        head = inst[0].strip()
        block_ops = (
            instruction_operation_blocks[inst_index]
            if inst_index < len(instruction_operation_blocks)
            else []
        )
        #print("Instruction: ", inst)
        #print("Head: ", head)
        if head.startswith("Execute"):
            flush_exposed_moves()
            before_assignment = copy.deepcopy(current_assignment)
            after_assignment = parse_assignment(inst[2] if len(inst) >= 3 else inst[1])
            execute_info = build_execute_info(
                parse_execute_payload(head),
                before_assignment,
                block_ops,
            )
            pending_execute.append({
                **execute_info,
                "before_assignment": before_assignment,
                "after_assignment": after_assignment,
            })
            current_assignment.clear()
            current_assignment.update(after_assignment)

        elif head.startswith("Move"):
            flush_execute_block()

            before_assignment = copy.deepcopy(current_assignment)

            u, v = parse_move(head)
            if (u, v) not in coupling_edges:
                raise ValueError(f"Move ({u}, {v}) is not an edge in machine_model.coupling_graph.")
            if ion_at_position(before_assignment, u) is None:
                u, v = v, u
            if ion_at_position(before_assignment, v) and ion_at_position(before_assignment, u)is None:
                continue
            #print(f"Move {(u, v)}")
            move_type = classify_move(u, v)
            if move_type == "trap_to_trap_invalid":
                raise ValueError(f"Direct trap-to-trap move ({u}, {v}) is invalid on this machine.")
            #print(f"Move {(u, v)} with type {move_type}")
            provided_cost = parse_cost(inst[2]) if len(inst) >= 3 else None
            model_cost = edge_cost(u, v, fallback=provided_cost)
            cost = model_cost if provided_cost is None else provided_cost

            if validate_instruction_cost and provided_cost is not None:
                if abs(provided_cost - model_cost) > 1e-12:
                    raise ValueError(
                        f"Cost mismatch on move ({u}, {v}): instruction={provided_cost}, model={model_cost}"
                    )

            shuttling_time_physical += cost
            move_type_counts[move_type] += 1

            after_assignment = parse_assignment(inst[1])

            # Infer analyzer-style events.
            if move_type == "intra_trap":
                trap = trap_owner[u]
                hop = abs(trap_slot_index[u] - trap_slot_index[v])
                ion_u = ion_at_position(before_assignment, u)
                ion_v = ion_at_position(before_assignment, v)

                if intra_trap_swap_mode == "gate":
                    pending_swap_meta[trap]["swap_cnt"] += 1
                    pending_swap_meta[trap]["swap_hops"] += hop
                    if ion_u is not None and ion_v is not None:
                        pending_swap_meta[trap]["swap_pair"] = (ion_u, ion_v)
                elif intra_trap_swap_mode == "ion":
                    pending_swap_meta[trap]["ion_hops"] += hop
                    if ion_u is not None and ion_v is not None:
                        pending_swap_meta[trap]["swap_pair"] = (ion_u, ion_v)
                elif intra_trap_swap_mode == "none":
                    pass
                else:
                    raise ValueError(
                        "intra_trap_swap_mode must be one of {'gate', 'ion', 'none'}"
                    )

            elif move_type == "split":
                trap = trap_owner[u]
                moved_ion = ion_at_position(before_assignment, u)
                if moved_ion is None:
                    #print(f"No ion at trap port {u} before split ({u}, {v}).")
                    current_assignment.clear()
                    current_assignment.update(before_assignment)
                    continue
                    #raise ValueError(f"No ion at trap port {u} before split ({u}, {v}).")

                meta = pending_swap_meta[trap]
                i1 = moved_ion
                i2 = None

                if meta["swap_hops"] > 0:
                    local_ions = [
                        q for q, pos in before_assignment.items()
                        if position_kind[pos] == "trap" and trap_owner[pos] == trap and q != moved_ion
                    ]
                    if local_ions:
                        # simplest valid choice: any other ion currently in the same trap
                        i2 = local_ions[0]

                append_event(
                    "Split",
                    {
                        "ions": [moved_ion],
                        "trap": trap,
                        "seg": v,
                        "swap_cnt": meta["swap_cnt"],
                        "ion_hops": meta["ion_hops"],
                        "swap_hops": meta["swap_hops"],
                        "i1": i1,
                        "i2": i2,
                        "from_pos": u,
                        "to_pos": v,
                        "cost": cost,
                    },
                    before_assignment,
                    after_assignment,
                )

                pending_swap_meta[trap] = fresh_swap_meta()

            elif move_type == "junction":
                moved_ion = ion_at_position(before_assignment, u)
                if moved_ion is None:
                    #raise ValueError(f"No ion at segment {u} before move ({u}, {v}).")
                    #print(f"No ion at segment {u} before move ({u}, {v}).")
                    current_assignment.clear()
                    current_assignment.update(before_assignment)
                    continue
                append_event(
                    "Move",
                    {
                        "ions": [moved_ion],
                        "source_seg": u,
                        "dest_seg": v,
                        "from_pos": u,
                        "to_pos": v,
                        "cost": cost,
                    },
                    before_assignment,
                    after_assignment,
                )

            elif move_type == "merge":
                moved_ion = ion_at_position(before_assignment, u)
                if moved_ion is None:
                    #raise ValueError(f"No ion at segment {u} before merge ({u}, {v}).")
                    #print(f"No ion at segment {u} before merge ({u}, {v}).")
                    current_assignment.clear()
                    current_assignment.update(before_assignment)
                    continue
                append_event(
                    "Merge",
                    {
                        "ions": [moved_ion],
                        "trap": trap_owner[v],
                        "seg": u,
                        "from_pos": u,
                        "to_pos": v,
                        "cost": cost,
                    },
                    before_assignment,
                    after_assignment,
                )

            # Update layout to the new post-move assignment.
            current_assignment.clear()
            current_assignment.update(after_assignment)
            #print("Current assignment: ", current_assignment)
            move_tokens = {
                ("edge", min(u, v), max(u, v)),
                ("pos", u),
                ("pos", v),
            }

            can_hide = (
                parallel
                and overlap_budget > 0.0
                and overlap_used + cost <= overlap_budget
                and move_tokens.isdisjoint(overlap_tokens)
            )

            if can_hide:
                overlap_used += cost
                overlap_tokens.update(move_tokens)
            else:
                pending_exposed_moves.append({
                    "move": (u, v),
                    "move_type": move_type,
                    "cost": cost,
                    "tokens": move_tokens,
                })

        else:
            raise ValueError(f"Unknown instruction: {head}")

    flush_execute_block()
    flush_exposed_moves()

    # ------------------------------------------------------------
    # QCCDSim-like replay on inferred events
    # ------------------------------------------------------------
    def chain_size_in_trap(assignment: Dict[int, int], trap_id: int) -> int:
        n = 0
        for _, pos in assignment.items():
            if position_kind[pos] == "trap" and trap_owner[pos] == trap_id:
                n += 1
        return n

    def qccd_gate_fidelity_by_ions(assignment: Dict[int, int], chain_heating: Dict[int, float], ion1: int, ion2: int):
        p1 = assignment[ion1]
        p2 = assignment[ion2]

        if position_kind[p1] != "trap" or position_kind[p2] != "trap":
            raise ValueError(
                f"Gate fidelity asked for ions ({ion1}, {ion2}) not both in traps: ({p1}, {p2})."
            )

        trap_id_1 = trap_owner[p1]
        trap_id_2 = trap_owner[p2]
        if trap_id_1 != trap_id_2:
            raise ValueError(
                f"Gate fidelity asked for ions ({ion1}, {ion2}) in different traps: ({p1}, {p2})."
            )

        trap_id = trap_id_1
        chain_size = max(2, chain_size_in_trap(assignment, trap_id))
        A = fit_A(chain_size)

        gate_time_est = float(machine_model.two_qudit_gate_time(p1=p1, p2=p2))

        # two_qudit_gate_time is already in seconds in your scheduler world.
        x1 = float(background_heating_rate * gate_time_est)
        x2 = float(A * (2.0 * chain_heating[trap_id] + 1.0))
        fidelity = max(min_gate_fidelity, base_gate_fidelity - x1 - x2)
        return fidelity, x1, x2

    replay_chain_heating = {trap_id: 0.0 for trap_id in set(trap_owner.values()) if trap_id is not None}
    replay_qubit_heating_quantas = {q: 0.0 for q in initial_ion_assignment.keys()}

    replay_log_fidelity = 0.0
    gate_fidelity_list = []
    f_background_list = []
    f_mode_list = []
    replay_op_count = defaultdict(int)

    for ev in inferred_events:
        etype = ev["type"]
        info = ev["info"]
        assignment_before = ev["before_assignment"]
        #print("Info: ", info)
        #print("etype: ", etype)
        #print("Assignment before: ", assignment_before)
        if etype == "Gate":
            ion1, ion2 = info["ions"]
            trap = info["trap"]
            f, x1, x2 = qccd_gate_fidelity_by_ions(assignment_before, replay_chain_heating, ion1, ion2)
            replay_log_fidelity += math.log(f)
            gate_fidelity_list.append(f)
            f_background_list.append(x1)
            f_mode_list.append(x2)
            replay_op_count["Gate"] += 1

        elif etype == "Split":
            moved_ion = info["ions"][0]
            trap = info["trap"]
            chain_size = max(1, chain_size_in_trap(assignment_before, trap))
            quanta = float(replay_chain_heating[trap]) / chain_size

            ion_swap_hops = info["ion_hops"]
            gate_swap_hops = info["swap_hops"]
            i1 = info["i1"]
            i2 = info["i2"]
            #print("Ion swaps: ", ion_swap_hops)
            #print("Gate swaps: ", gate_swap_hops)
            if ion_swap_hops != 0:
                replay_chain_heating[trap] += 0.1 * ion_swap_hops + 0.1 * (ion_swap_hops - 1)
                replay_chain_heating[trap] += 0.01 * ion_swap_hops
                replay_qubit_heating_quantas[moved_ion] = quanta + 0.1

            if gate_swap_hops != 0:
                if i1 is not None and i2 is not None and i1 != i2:
                    f_swap, x1_swap, x2_swap = qccd_gate_fidelity_by_ions(
                        assignment_before, replay_chain_heating, i1, i2
                    )
                    replay_log_fidelity += math.log(f_swap)
                    gate_fidelity_list.extend([f_swap, f_swap, f_swap])
                    f_background_list.extend([x1_swap, x1_swap, x1_swap])
                    f_mode_list.extend([x2_swap, x2_swap, x2_swap])

                val = 2.0 if honeywell_mode else 0.1
                replay_chain_heating[trap] = replay_chain_heating[trap] - quanta + val
                replay_qubit_heating_quantas[moved_ion] = quanta + val

            replay_op_count["Split"] += 1

        elif etype == "Move":
            moved_ion = info["ions"][0]
            val = 2.0 if honeywell_mode else 0.01
            replay_qubit_heating_quantas[moved_ion] += val
            replay_op_count["Move"] += 1

        elif etype == "Merge":
            moved_ion = info["ions"][0]
            trap = info["trap"]
            val = 2.0 if honeywell_mode else 0.1
            replay_chain_heating[trap] += replay_qubit_heating_quantas[moved_ion] + val
            replay_qubit_heating_quantas[moved_ion] = 0.0
            replay_op_count["Merge"] += 1

        else:
            raise ValueError(f"Unknown inferred event type: {etype}")

    application_fidelity = math.exp(replay_log_fidelity) if inferred_events else 1.0

    shuttling_profile_critical = (
        shuttling_time_critical / (shuttling_time_critical + execution_time)
        if (shuttling_time_critical + execution_time) > 0
        else 0.0
    )

    shuttling_profile_physical = (
        shuttling_time_physical / (shuttling_time_physical + execution_time)
        if (shuttling_time_physical + execution_time) > 0
        else 0.0
    )

    return {
        "runtime": runtime,
        "execution_time": execution_time,
        "shuttling_time_critical": shuttling_time_critical,
        "shuttling_time_physical": shuttling_time_physical,
        "shuttling_profile_critical": shuttling_profile_critical,
        "shuttling_profile_physical": shuttling_profile_physical,
        "application_fidelity": application_fidelity,
        "gate_fidelities": gate_fidelity_list,
        "f_background_term": f_background_list,
        "f_mode_term": f_mode_list,
        "move_type_counts": dict(move_type_counts),
        "final_ion_assignment": copy.deepcopy(current_assignment),
        "execute_rounds": execute_rounds,
        "move_rounds": move_rounds,
        "inferred_events": inferred_events,
        "replay_chain_heating": replay_chain_heating,
        "replay_qubit_heating_quantas": replay_qubit_heating_quantas,
        "replay_op_count": dict(replay_op_count),
        "pending_swap_meta_leftover": {
            trap: meta for trap, meta in pending_swap_meta.items()
            if meta["swap_cnt"] != 0 or meta["swap_hops"] != 0 or meta["ion_hops"] != 0
        },
    }


def print_event_trace(schedule_result: Dict[str, Any]) -> None:
    inferred_events = schedule_result.get("inferred_events", [])
    if not inferred_events:
        print("No inferred events recorded.")
        return

    for event in inferred_events:
        event_id = event.get("id", "?")
        event_type = event.get("type", "Unknown")
        info = event.get("info", {})
        print(f"[{event_id}] {event_type}: {info}")


def _run_benchmark_demo() -> None:
    import numpy as np

    circuit_lst = [
        "QFT_wsq_16_compiled",
        "TFIM_n16_s100_compiled",
        "TFXY_n16_s100_compiled",
        "QFT_20_compiled",
    ]
    architecture_lst = [
        "H",
        "G2x3",
    ]
    parameter_set = {
        "H": ["6"],
        "G2x3": ["5"],
    }

    num_layout = 2
    circuit_idx = 3
    architecture = architecture_lst[0]
    param = parameter_set[architecture][0]

    algo = "SHAW"
    print(
        f"Algo: {algo} - Circuit: {circuit_lst[circuit_idx]} "
        f"with archictecture {architecture} with trap capacity {param}",
    )
    runtimes = []
    fidelitys = []
    for idx in range(1, 11):
        print(f"==================== idx: {idx} ==================== ")
        if idx == 7:
            continue
        file_name = (
            f"{algo}_{circuit_lst[circuit_idx]}_idx{idx}_"
            f"{architecture}_{param}_{num_layout}"
        )
        qasm_result_filename = f"paper_result_16/{file_name}.pkl"
        with open(qasm_result_filename, "rb") as input_file:
            stored_data = pickle.load(input_file)
        (
            runtime,
            compile_time,
            instruction_lst,
            gate_counts,
            initial_ion_assignment,
            initial_mapping,
            final_mapping,
            machine_model,
        ) = stored_data

        result = schedule_qccd_from_instructions_v3(
            instruction_lst=instruction_lst,
            initial_ion_assignment=initial_ion_assignment,
            machine_model=machine_model,
            parallel=True,
        )
        print("Runtime (us):", result["runtime"] / 1e-6)
        print("Application fidelity:", result["application_fidelity"])
        runtimes.append(result["runtime"] / 1e-6)
        fidelitys.append(result["application_fidelity"])
    print("Final runtime:", np.min(runtimes))
    print("Final fidelity: ", fidelitys[np.argmin(runtimes)])
    # print("Execution time (us):", result["execution_time"] / 1e-6)
    # print("Critical shuttling time (us):", result["shuttling_time_critical"] / 1e-6)
    # print("Physical shuttling time (us):", result["shuttling_time_physical"] / 1e-6)
    # print("Critical shuttling profile:", result["shuttling_profile_critical"])
    # print("Physical shuttling profile:", result["shuttling_profile_physical"])

    # print("Trap heating:", result["trap_heating"])
    # print("Gate count:", result["gate_count"])
    # print("Final assignment:", result["final_ion_assignment"])
    # print("Execute rounds:", result["execute_rounds"])
    # print("Move rounds:", result["move_rounds"])


if __name__ == "__main__":
    _run_benchmark_demo()
