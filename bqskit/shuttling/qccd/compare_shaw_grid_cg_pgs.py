from __future__ import annotations

import argparse
import ast
import copy
import os
import re
from pathlib import Path
from timeit import default_timer as timer
from typing import Any

from bqskit.compiler import Compiler
from bqskit.compiler.gateset import GateSet
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CXGate
from bqskit.ir.gates.parameterized import U3Gate
from bqskit.passes import ApplyPlacement
from bqskit.passes import QuickPartitioner
from bqskit.passes import SetModelPass
from bqskit.passes import UnfoldPass
from bqskit.passes import UpdateDataPass

from bqskit.shuttling.qccd import create_grid_physical_machine
from bqskit.shuttling.qccd.QCCD_mapping import QCCDMappingAlgorithm as CGMappingAlgorithm
from bqskit.shuttling.qccd.QCCD_mapping_PGS import QCCDMappingAlgorithm as PGSMappingAlgorithm
from bqskit.shuttling.qccd.mapping import QCCDLayoutPass
from bqskit.shuttling.qccd.mapping import QCCDRoutingPass
from bqskit.shuttling.qccd.pgs_passes import QCCDLayoutPassPGS
from bqskit.shuttling.qccd.pgs_passes import QCCDRoutingPassPGS
from bqskit.shuttling.qccd.QCCD_machine import QCCDMachineModel as CGMachineModel
from bqskit.shuttling.qccd.QCCD_machine_PGS import (
    QCCDMachineModel as PGSMachineModel,
)
from bqskit.shuttling.QCCD_schedule_new import schedule_qccd_from_instructions_v3
from bqskit.shuttling.qccd.run_grid_pgs_shaw import build_assignment


TIMING_DATA = {
    'sq_timings': 30e-6,
    'tq_timings': 40e-6,
    'segment': 5e-6,
    'inner_swap': 42e-6,
    'split': 80e-6,
    'merge': 80e-6,
    'junction_Y': 100e-6,
    'junction_X': 120e-6,
}


def congestion_rate(machine_model: Any, num_qudits: int) -> float:
    executable_spaces = 0
    for trap in machine_model.physical_graph.executable_trap_list:
        executable_spaces += trap.max_num_ions
    return 1.0 if num_qudits == executable_spaces else 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Compare SHAW CG vs PGS instruction streams on a grid.',
    )
    parser.add_argument('input_filename')
    parser.add_argument('--trap-capacity', type=int, default=3)
    parser.add_argument('--num-layout-passes', type=int, default=2)
    parser.add_argument('--gate-type', default='FM')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--grid-cols', type=int, default=1)
    parser.add_argument('--grid-rows', type=int, default=1)
    parser.add_argument(
        '--routing-mode',
        choices=['heuristic', 'bruteforce'],
        default='bruteforce',
    )
    parser.add_argument(
        '--stage',
        choices=['layout', 'layout-trace', 'layout-wrapper-trace', 'forward-pass', 'full', 'full-matched-layout'],
        default='full',
    )
    parser.add_argument(
        '--cg-congestion-rate-override',
        type=float,
        default=None,
    )
    parser.add_argument(
        '--pgs-congestion-rate-override',
        type=float,
        default=None,
    )
    parser.add_argument('--verbose-status', action='store_true')
    parser.add_argument('--print-paths', action='store_true')
    parser.add_argument(
        '--print-position-graph',
        action='store_true',
        help='Print raw position-graph positions and edges after each machine model is built.',
    )
    parser.add_argument('--window', type=int, default=8)
    parser.add_argument(
        '--pgs-move-path-modes',
        nargs='+',
        choices=['hops', 'weighted'],
        default=['hops'],
        help='Run one or more PGS variants using hop-based or weighted move-path caches.',
    )
    return parser.parse_args()


def parse_execute_payload(text: str) -> list[int]:
    match = re.search(r'Execute\s+at\s+\[(.*)\]', text)
    if match is None:
        raise ValueError(f'Cannot parse execute payload from: {text}')
    payload = '[' + match.group(1).strip() + ']'
    values = ast.literal_eval(payload)
    return [int(v) for v in values]


def parse_assignment(text: str) -> dict[int, int]:
    return {int(k): int(v) for k, v in ast.literal_eval(text).items()}


def normalize_execute(
    inst: list[Any],
    assignment: dict[int, int],
    execute_location_mode: str,
) -> tuple[str, list[int], dict[int, int]]:
    payload = parse_execute_payload(inst[0])
    inverse_assignment = {position: logical for logical, position in assignment.items()}
    if execute_location_mode not in {'logical', 'physical'}:
        raise ValueError(
            "execute_location_mode must be one of {'logical', 'physical'}",
        )

    if execute_location_mode == 'logical':
        normalized_qudits = [int(value) for value in payload]
    else:
        normalized_qudits = [
            int(inverse_assignment.get(value, value))
            for value in payload
        ]

    return ('Execute', sorted(normalized_qudits), parse_assignment(inst[1]))


def normalize_move(inst: list[Any]) -> tuple[str, tuple[int, int], dict[int, int]]:
    move_text = inst[0].strip()
    payload = move_text[len('Move'):].strip()
    payload = re.sub(r'np\.int\d+\(\s*(-?\d+)\s*\)', r'\1', payload)
    left, right = ast.literal_eval(payload)
    return ('Move', (int(left), int(right)), parse_assignment(inst[1]))


def normalize_instruction(
    inst: list[Any],
    assignment: dict[int, int],
    execute_location_mode: str,
) -> tuple[str, Any, dict[int, int]]:
    if inst[0].startswith('Execute'):
        return normalize_execute(inst, assignment, execute_location_mode)
    if inst[0].startswith('Move'):
        return normalize_move(inst)
    return (inst[0], None, parse_assignment(inst[1]))


def run_compare_scheduler(
    backend: str,
    instruction_list: list[Any],
    circuit: Circuit,
    initial_mapping: list[int],
    initial_ion_assignment: dict[int, int],
    full_initial_ion_assignment: dict[int, int] | None,
    machine_model: Any,
) -> dict[str, Any]:
    del initial_mapping
    if backend == 'PGS':
        analytics_result = schedule_qccd_from_instructions_v3(
            instruction_lst=instruction_list,
            initial_ion_assignment=initial_ion_assignment,
            full_initial_ion_assignment=full_initial_ion_assignment,
            machine_model=machine_model,
            circuit=circuit,
            parallel=True,
            execute_location_mode='physical',
        )
        return {
            'runtime': float(analytics_result['runtime']),
            'shuttling_share': float(analytics_result['shuttling_profile_critical']),
            'application_fidelity': analytics_result['application_fidelity'],
            'execute_rounds': len(analytics_result['execute_rounds']),
            'move_rounds': len(analytics_result['move_rounds']),
        }

    schedule_result = schedule_qccd_from_instructions_v3(
        instruction_lst=instruction_list,
        initial_ion_assignment=initial_ion_assignment,
        full_initial_ion_assignment=full_initial_ion_assignment,
        machine_model=machine_model,
        circuit=circuit,
        parallel=True,
        execute_location_mode='logical',
    )
    return {
        'runtime': float(schedule_result['runtime']),
        'shuttling_share': None,
        'application_fidelity': schedule_result['application_fidelity'],
        'execute_rounds': len(schedule_result['execute_rounds']),
        'move_rounds': len(schedule_result['move_rounds']),
    }


def print_path_comparison(
    cg_model: CGMachineModel,
    pgs_model: PGSMachineModel,
    move_pairs: list[tuple[int, int]],
) -> None:
    seen: set[tuple[int, int]] = set()
    print('Path comparison:')
    for left, right in move_pairs:
        pair = tuple(sorted((int(left), int(right))))
        if pair in seen:
            continue
        seen.add(pair)
        cg_tree = cg_model.position_graph.get_shortest_path_tree(pair[0])
        cg_path = list(cg_tree[pair[1]])
        pgs_path = list(pgs_model.get_move_path(pair[0], pair[1]))
        print(f'  {pair}: CG={cg_path} PGS={pgs_path} same={cg_path == pgs_path}')


def print_position_graph_details(label: str, machine_model: Any) -> None:
    position_graph = machine_model.position_graph
    print(f'{label} position graph details:')
    print(f'  physical_to_position: {machine_model.physical_to_position}')

    if hasattr(position_graph, 'position_labels') and hasattr(position_graph, 'edge_labels'):
        num_positions = len(position_graph.position_labels)
        num_edges = len(position_graph.edge_labels)
        print('  positions:')
        for index, position_label in enumerate(position_graph.position_labels):
            print(
                f'    {index}: capability={position_label.capability} '
                f'weights={position_label.weights}',
            )
        print('  edges:')
        for (u, v), edge_label in sorted(position_graph.edge_labels.items()):
            print(
                f'    ({int(u)}, {int(v)}): capability={edge_label.capability} '
                f'weights={edge_label.weights}',
            )
        print(f'  num_positions: {num_positions}')
        print(f'  num_edges: {num_edges}')
        return

    positions = list(range(int(position_graph.num_qudits)))
    edges = sorted(
        (int(edge[0]), int(edge[1]))
        for edge in position_graph
    )
    print(f'  positions: {positions}')
    print(f'  edges: {edges}')
    print(f'  num_positions: {len(positions)}')
    print(f'  num_edges: {len(edges)}')


class _TemporaryEnv:
    def __init__(self, key: str, value: str | None) -> None:
        self.key = key
        self.value = value
        self.previous = os.environ.get(key)

    def __enter__(self) -> None:
        if self.value is None:
            os.environ.pop(self.key, None)
        else:
            os.environ[self.key] = self.value

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self.previous is None:
            os.environ.pop(self.key, None)
        else:
            os.environ[self.key] = self.previous


def compile_case(
    backend: str,
    circuit: Circuit,
    assignment: dict[int, int],
    trap_capacity: int,
    num_layout_passes: int,
    gate_type: str,
    grid_cols: int,
    grid_rows: int,
    routing_mode: str,
    stage: str,
    congestion_override: float | None = None,
    pgs_move_path_mode: str | None = None,
    print_position_graph: bool = False,
) -> dict[str, Any]:
    env_value = pgs_move_path_mode if backend == 'PGS' else None
    with _TemporaryEnv('BQSKIT_PGS_MOVE_PATH_MODE', env_value):
        physical_model = create_grid_physical_machine(
            num_cols=grid_cols,
            num_rows=grid_rows,
            trap_capacity=trap_capacity,
        )
        gate_set = GateSet({U3Gate(), CXGate()})
        model_cls = PGSMachineModel if backend == 'PGS' else CGMachineModel
        machine_model = model_cls(
            gate_set=gate_set,
            physical_graph=physical_model,
            multi_qudit_gate_type=gate_type,
            timing_data=TIMING_DATA,
        )
        if print_position_graph:
            graph_label = backend
            if backend == 'PGS' and pgs_move_path_mode is not None:
                graph_label = f'{backend}[{pgs_move_path_mode}]'
            print_position_graph_details(graph_label, machine_model)
        congestion = congestion_rate(machine_model, circuit.num_qudits)
        if congestion_override is not None:
            congestion = float(congestion_override)
        force_bruteforce = routing_mode == 'bruteforce'

        if backend == 'PGS':
            workflow = [
                UnfoldPass(),
                SetModelPass(machine_model),
                UpdateDataPass(key='ion_assignment_qccd', val=assignment),
                QuickPartitioner(3),
                ApplyPlacement(),
                QCCDLayoutPassPGS(
                    total_passes=num_layout_passes,
                    cogestion_rate=congestion,
                    force_bruteforce=force_bruteforce,
                ),
            ]
            if stage == 'full':
                workflow.extend([
                    QCCDRoutingPassPGS(
                        0.1,
                        cogestion_rate=congestion,
                        force_bruteforce=force_bruteforce,
                    ),
                    ApplyPlacement(),
                    UnfoldPass(),
                ])
        else:
            workflow = [
                UnfoldPass(),
                SetModelPass(machine_model),
                UpdateDataPass(key='ion_assignment_qccd', val=assignment),
                QuickPartitioner(3),
                ApplyPlacement(),
                QCCDLayoutPass(
                    total_passes=num_layout_passes,
                    cogestion_rate=congestion,
                    force_bruteforce=force_bruteforce,
                ),
            ]
            if stage == 'full':
                workflow.extend([
                    QCCDRoutingPass(
                        cogestion_rate=congestion,
                        force_bruteforce=force_bruteforce,
                    ),
                    ApplyPlacement(),
                    UnfoldPass(),
                ])

        with Compiler() as compiler:
            start = timer()
            output_circuit, data = compiler.compile(circuit, workflow, request_data=True)
            compile_time_s = timer() - start
        schedule_result = None
        if stage == 'full':
            schedule_result = run_compare_scheduler(
                backend=backend,
                instruction_list=data['instruction_list'],
                circuit=output_circuit,
                initial_mapping=list(getattr(data, 'initial_mapping', list(range(circuit.num_qudits)))),
                initial_ion_assignment=data['initial_ion_assignment_qccd'],
                full_initial_ion_assignment=data.get('initial_full_ion_assignment_qccd_pgs'),
                machine_model=data.model,
            )

        return {
            'backend': backend,
            'pgs_move_path_mode': pgs_move_path_mode,
            'data': data,
            'output_circuit': output_circuit,
            'instruction_list': data.get('instruction_list', []),
            'compile_time_s': compile_time_s,
            'runtime_us': None if schedule_result is None else schedule_result['runtime'] / 1e-6,
            'application_fidelity': None if schedule_result is None else schedule_result['application_fidelity'],
            'execute_rounds': None if schedule_result is None else schedule_result['execute_rounds'],
            'move_rounds': None if schedule_result is None else schedule_result['move_rounds'],
            'shuttling_share': None if schedule_result is None else schedule_result['shuttling_share'],
        }


def run_pre_layout_workflow(
    backend: str,
    circuit: Circuit,
    assignment: dict[int, int],
    trap_capacity: int,
    gate_type: str,
    grid_cols: int,
    grid_rows: int,
    print_position_graph: bool = False,
) -> tuple[Circuit, Any]:
    physical_model = create_grid_physical_machine(
        num_cols=grid_cols,
        num_rows=grid_rows,
        trap_capacity=trap_capacity,
    )
    gate_set = GateSet({U3Gate(), CXGate()})
    model_cls = PGSMachineModel if backend == 'PGS' else CGMachineModel
    machine_model = model_cls(
        gate_set=gate_set,
        physical_graph=physical_model,
        multi_qudit_gate_type=gate_type,
        timing_data=TIMING_DATA,
    )
    if print_position_graph:
        print_position_graph_details(f'{backend} pre-layout', machine_model)
    workflow = [
        UnfoldPass(),
        SetModelPass(machine_model),
        UpdateDataPass(key='ion_assignment_qccd', val=assignment),
        QuickPartitioner(3),
    ]
    if backend in {'CG', 'PGS'}:
        workflow.append(ApplyPlacement())

    with Compiler() as compiler:
        output_circuit, data = compiler.compile(circuit, workflow, request_data=True)

    return output_circuit, data


def trace_layout_case(
    backend: str,
    circuit: Circuit,
    assignment: dict[int, int],
    trap_capacity: int,
    num_layout_passes: int,
    gate_type: str,
    grid_cols: int,
    grid_rows: int,
    routing_mode: str,
    congestion_override: float | None = None,
    print_position_graph: bool = False,
) -> dict[str, Any]:
    pre_circuit, pre_data = run_pre_layout_workflow(
        backend,
        circuit,
        assignment,
        trap_capacity,
        gate_type,
        grid_cols,
        grid_rows,
        print_position_graph,
    )
    force_bruteforce = routing_mode == 'bruteforce'
    congestion = congestion_rate(pre_data.model, pre_circuit.num_qudits)
    if congestion_override is not None:
        congestion = float(congestion_override)
    snapshots: list[tuple[str, dict[int, int]]] = []

    if backend == 'CG':
        algo = CGMappingAlgorithm(
            qccd_machine=pre_data.model,
            cogestion_rate=congestion,
            force_bruteforce=force_bruteforce,
        )
        pi = [i for i in range(pre_circuit.num_qudits)]
        ion_assignment = copy.deepcopy(pre_data['ion_assignment_qccd'])
        snapshots.append(('start', copy.deepcopy(ion_assignment)))
        for layout_pass_index in range(num_layout_passes):
            algo.forward_pass(pre_circuit, pi, ion_assignment, False)
            snapshots.append((f'forward_{layout_pass_index + 1}', copy.deepcopy(ion_assignment)))
            algo.backward_pass(pre_circuit, pi, ion_assignment)
            snapshots.append((f'backward_{layout_pass_index + 1}', copy.deepcopy(ion_assignment)))
        final_assignment = ion_assignment
    else:
        algo = PGSMappingAlgorithm(
            qccd_machine=pre_data.model,
            cogestion_rate=congestion,
            force_bruteforce=force_bruteforce,
        )
        pgs = algo._make_pgs(copy.deepcopy(pre_data['ion_assignment_qccd']))
        snapshots.append(('start', algo._assignment_from_pgs(pgs)))
        for layout_pass_index in range(num_layout_passes):
            algo.forward_pass(pre_circuit, pgs=pgs, modify_circuit=False)
            snapshots.append((f'forward_{layout_pass_index + 1}', algo._assignment_from_pgs(pgs)))
            algo.backward_pass(pre_circuit, pgs=pgs)
            snapshots.append((f'backward_{layout_pass_index + 1}', algo._assignment_from_pgs(pgs)))
        final_assignment = algo._assignment_from_pgs(pgs)

    front_points = sorted(pre_circuit.front)
    rear_points = sorted(pre_circuit.rear)
    front_locations = [tuple(int(q) for q in pre_circuit[n].location) for n in front_points[:5]]
    rear_locations = [tuple(int(q) for q in pre_circuit[n].location) for n in rear_points[:5]]

    return {
        'backend': backend,
        'initial_assignment': copy.deepcopy(pre_data['ion_assignment_qccd']),
        'final_assignment': final_assignment,
        'snapshots': snapshots,
        'front_locations': front_locations,
        'rear_locations': rear_locations,
    }


def trace_layout_wrapper_case(
    backend: str,
    circuit: Circuit,
    assignment: dict[int, int],
    trap_capacity: int,
    num_layout_passes: int,
    gate_type: str,
    grid_cols: int,
    grid_rows: int,
    routing_mode: str,
    congestion_override: float | None = None,
    print_position_graph: bool = False,
) -> dict[str, Any]:
    previous_wrapper = os.environ.get('BQSKIT_QCCD_CAPTURE_LAYOUT_WRAPPER')
    previous_forward = os.environ.get('BQSKIT_QCCD_CAPTURE_TRACE')
    os.environ['BQSKIT_QCCD_CAPTURE_LAYOUT_WRAPPER'] = '1'
    os.environ['BQSKIT_QCCD_CAPTURE_TRACE'] = '1'
    try:
        result = compile_case(
            backend,
            circuit,
            assignment,
            trap_capacity,
            num_layout_passes,
            gate_type,
            grid_cols,
            grid_rows,
            routing_mode,
            'layout',
            congestion_override=congestion_override,
            print_position_graph=print_position_graph,
        )
    finally:
        if previous_wrapper is None:
            os.environ.pop('BQSKIT_QCCD_CAPTURE_LAYOUT_WRAPPER', None)
        else:
            os.environ['BQSKIT_QCCD_CAPTURE_LAYOUT_WRAPPER'] = previous_wrapper
        if previous_forward is None:
            os.environ.pop('BQSKIT_QCCD_CAPTURE_TRACE', None)
        else:
            os.environ['BQSKIT_QCCD_CAPTURE_TRACE'] = previous_forward

    data = result['data']
    snapshots = copy.deepcopy(data.get('qccd_layout_wrapper_snapshots', []))
    front_points = sorted(result['output_circuit'].front)
    rear_points = sorted(result['output_circuit'].rear)
    front_locations = [
        tuple(int(q) for q in result['output_circuit'][n].location)
        for n in front_points[:5]
    ]
    rear_locations = [
        tuple(int(q) for q in result['output_circuit'][n].location)
        for n in rear_points[:5]
    ]
    return {
        'backend': backend,
        'initial_assignment': copy.deepcopy(snapshots[0][1]) if snapshots else copy.deepcopy(data['ion_assignment_qccd']),
        'final_assignment': copy.deepcopy(data['ion_assignment_qccd']),
        'snapshots': snapshots,
        'forward_traces': copy.deepcopy(data.get('qccd_layout_wrapper_forward_traces', [])),
        'front_locations': front_locations,
        'rear_locations': rear_locations,
    }


def trace_forward_pass_case(
    backend: str,
    circuit: Circuit,
    assignment: dict[int, int],
    trap_capacity: int,
    gate_type: str,
    grid_cols: int,
    grid_rows: int,
    routing_mode: str,
    congestion_override: float | None = None,
    print_position_graph: bool = False,
) -> dict[str, Any]:
    pre_circuit, pre_data = run_pre_layout_workflow(
        backend,
        circuit,
        assignment,
        trap_capacity,
        gate_type,
        grid_cols,
        grid_rows,
        print_position_graph,
    )
    force_bruteforce = routing_mode == 'bruteforce'
    congestion = congestion_rate(pre_data.model, pre_circuit.num_qudits)
    if congestion_override is not None:
        congestion = float(congestion_override)

    if backend == 'CG':
        algo = CGMappingAlgorithm(
            qccd_machine=pre_data.model,
            cogestion_rate=congestion,
            force_bruteforce=force_bruteforce,
        )
        pi = [i for i in range(pre_circuit.num_qudits)]
        ion_assignment = copy.deepcopy(pre_data['ion_assignment_qccd'])
        algo.forward_pass(pre_circuit, pi, ion_assignment, False)
        final_assignment = copy.deepcopy(ion_assignment)
    else:
        algo = PGSMappingAlgorithm(
            qccd_machine=pre_data.model,
            cogestion_rate=congestion,
            force_bruteforce=force_bruteforce,
        )
        pgs = algo._make_pgs(copy.deepcopy(pre_data['ion_assignment_qccd']))
        algo.forward_pass(pre_circuit, pgs=pgs, modify_circuit=False)
        final_assignment = algo._assignment_from_pgs(pgs)

    return {
        'backend': backend,
        'initial_assignment': copy.deepcopy(pre_data['ion_assignment_qccd']),
        'final_assignment': final_assignment,
        'front_locations': [
            tuple(int(q) for q in pre_circuit[n].location)
            for n in sorted(pre_circuit.front)[:5]
        ],
        'trace': copy.deepcopy(getattr(algo, 'last_forward_trace', [])),
    }


def compile_from_matched_layout(
    backend: str,
    circuit: Circuit,
    assignment: dict[int, int],
    trap_capacity: int,
    num_layout_passes: int,
    gate_type: str,
    grid_cols: int,
    grid_rows: int,
    routing_mode: str,
    congestion_override: float | None = None,
    pgs_move_path_mode: str | None = None,
    print_position_graph: bool = False,
) -> dict[str, Any]:
    env_value = pgs_move_path_mode if backend == 'PGS' else None
    with _TemporaryEnv('BQSKIT_PGS_MOVE_PATH_MODE', env_value):
        pre_circuit, pre_data = run_pre_layout_workflow(
            backend,
            circuit,
            assignment,
            trap_capacity,
            gate_type,
            grid_cols,
            grid_rows,
            print_position_graph,
        )
        force_bruteforce = routing_mode == 'bruteforce'
        congestion = congestion_rate(pre_data.model, pre_circuit.num_qudits)
        if congestion_override is not None:
            congestion = float(congestion_override)

        if backend == 'CG':
            algo = CGMappingAlgorithm(
                qccd_machine=pre_data.model,
                cogestion_rate=congestion,
                force_bruteforce=force_bruteforce,
            )
            pi = [i for i in range(pre_circuit.num_qudits)]
            ion_assignment = copy.deepcopy(pre_data['ion_assignment_qccd'])
            for _ in range(num_layout_passes):
                algo.forward_pass(pre_circuit, pi, ion_assignment, False)
                algo.backward_pass(pre_circuit, pi, ion_assignment)
            routed_circuit = pre_circuit.copy()
            instruction_list = algo.forward_pass(routed_circuit, pi, ion_assignment, True)
            output_circuit = routed_circuit
            final_assignment = copy.deepcopy(ion_assignment)
        else:
            algo = PGSMappingAlgorithm(
                qccd_machine=pre_data.model,
                cogestion_rate=congestion,
                force_bruteforce=force_bruteforce,
            )
            pgs = algo._make_pgs(copy.deepcopy(pre_data['ion_assignment_qccd']))
            for _ in range(num_layout_passes):
                algo.forward_pass(pre_circuit, pgs=pgs, modify_circuit=False)
                algo.backward_pass(pre_circuit, pgs=pgs)
            routed_circuit = pre_circuit.copy()
            instruction_list = algo.forward_pass(routed_circuit, pgs=pgs, modify_circuit=True)
            output_circuit = routed_circuit
            final_assignment = algo._assignment_from_pgs(pgs)

        schedule_result = run_compare_scheduler(
            backend=backend,
            instruction_list=instruction_list,
            circuit=output_circuit,
            initial_mapping=list(getattr(pre_data, 'initial_mapping', list(range(circuit.num_qudits)))),
            initial_ion_assignment=copy.deepcopy(assignment),
            full_initial_ion_assignment=copy.deepcopy(
                assignment if backend == 'CG' else pre_data.get('full_ion_assignment_qccd_pgs', assignment),
            ),
            machine_model=pre_data.model,
        )

        return {
            'backend': backend,
            'pgs_move_path_mode': pgs_move_path_mode,
            'data': {'ion_assignment_qccd': final_assignment},
            'output_circuit': output_circuit,
            'instruction_list': instruction_list,
            'compile_time_s': None,
            'runtime_us': schedule_result['runtime'] / 1e-6,
            'application_fidelity': schedule_result['application_fidelity'],
            'execute_rounds': schedule_result['execute_rounds'],
            'move_rounds': schedule_result['move_rounds'],
            'shuttling_share': schedule_result['shuttling_share'],
        }


def print_forward_trace_entry(prefix: str, entry: dict[str, Any]) -> None:
    print(f"  {prefix} action          : {entry['action']}")
    print(f"  {prefix} front_locations : {entry['front_locations']}")
    print(f"  {prefix} front_executable: {entry['front_executable']}")
    print(f"  {prefix} execute_locations: {entry['execute_locations']}")
    print(f"  {prefix} best_move       : {entry['best_move']}")
    print(f"  {prefix} brute_force_gate: {entry['brute_force_gate']}")
    print(f"  {prefix} brute_force_trace: {entry['brute_force_trace']}")
    print(f"  {prefix} pre_assignment  : {entry['pre_assignment']}")
    print(f"  {prefix} post_assignment : {entry['post_assignment']}")


def print_compact_forward_trace(prefix: str, trace: list[dict[str, Any]], window: int) -> None:
    print(f'  {prefix}trace steps: {len(trace)}')
    for index, entry in enumerate(trace[:window]):
        print(
            f"  {prefix}{index}: action={entry['action']} "
            f"front={entry['front_locations']} exec={entry['execute_locations']} "
            f"best_move={entry['best_move']} brute_gate={entry['brute_force_gate']}"
        )


def print_resolve_trace(prefix: str, trace: dict[str, Any] | None) -> None:
    if trace is None:
        return
    resolve_entries = trace.get('resolve_trace', [])
    if not resolve_entries:
        return
    print(f"  {prefix} resolve_trace:")
    for entry in resolve_entries[:8]:
        print(
            f"    call={entry['num_call']} target={entry['target']} "
            f"blockage={entry['blockage']} branch={entry.get('branch')}"
        )
        print(f"    path={entry['path']}")
        print(f"    initial_neighbors={entry['initial_blockage_neighbors']}")
        print(f"    filtered_neighbors={entry['filtered_blockage_neighbors']}")
        print(f"    potential_blockage={entry['potential_blockage']}")
        if 'congestion_rates' in entry:
            print(f"    congestion_rates={entry['congestion_rates']}")
            print(f"    congestion_scores={entry['congestion_scores']}")
            print(f"    chosen_neighbor={entry['chosen_neighbor']}")


def format_optional_float(value: float | None, digits: int = 6) -> str:
    if value is None:
        return 'n/a'
    return f'{value:.{digits}f}'


def result_label(result: dict[str, Any], *, multi_pgs: bool) -> str:
    if result['backend'] != 'PGS' or not multi_pgs:
        return result['backend']
    return f"PGS[{result['pgs_move_path_mode']}]"


def print_result_summary(result: dict[str, Any], *, multi_pgs: bool) -> None:
    label = result_label(result, multi_pgs=multi_pgs)
    print(
        f"  {label:<13} compile_time_s={format_optional_float(result['compile_time_s'])} "
        f"runtime_us={format_optional_float(result['runtime_us'], 3)} "
        f"fidelity={format_optional_float(result['application_fidelity'], 12)} "
        f"instructions={len(result['instruction_list'])} "
        f"execute_rounds={result['execute_rounds']} "
        f"move_rounds={result['move_rounds']}",
    )


def main() -> None:
    args = parse_args()
    pgs_move_path_modes = list(dict.fromkeys(args.pgs_move_path_modes))
    if len(pgs_move_path_modes) > 1 and args.stage not in (
        'layout',
        'full',
        'full-matched-layout',
    ):
        raise ValueError(
            'Multiple PGS move-path modes are only supported for '
            "{'layout', 'full', 'full-matched-layout'} stages.",
        )
    os.environ['BQSKIT_QCCD_VERBOSE'] = '1' if args.verbose_status else '0'
    os.environ['BQSKIT_QCCD_CAPTURE_TRACE'] = (
        '1' if args.stage == 'forward-pass' else '0'
    )
    circuit_path = (
        Path('bqskit/shuttling/qccd/benchmark_circuits')
        / f'{args.input_filename}.qasm'
    )
    circuit = Circuit.from_file(str(circuit_path))

    base_model = CGMachineModel(
        gate_set=GateSet({U3Gate(), CXGate()}),
        physical_graph=create_grid_physical_machine(
            num_cols=args.grid_cols,
            num_rows=args.grid_rows,
            trap_capacity=args.trap_capacity,
        ),
        multi_qudit_gate_type=args.gate_type,
        timing_data=TIMING_DATA,
    )
    pgs_base_model = PGSMachineModel(
        gate_set=GateSet({U3Gate(), CXGate()}),
        physical_graph=create_grid_physical_machine(
            num_cols=args.grid_cols,
            num_rows=args.grid_rows,
            trap_capacity=args.trap_capacity,
        ),
        multi_qudit_gate_type=args.gate_type,
        timing_data=TIMING_DATA,
    )
    assignment = build_assignment(base_model, circuit.num_qudits, args.seed)

    print(f'Initial ion assignment: {assignment}')
    print(f'Routing mode: {args.routing_mode}')
    print(f'Stage: {args.stage}')
    if len(pgs_move_path_modes) > 1:
        print(f'PGS move-path modes: {pgs_move_path_modes}')
    if args.stage in ('layout', 'full', 'full-matched-layout'):
        if args.stage == 'full-matched-layout':
            cg = compile_from_matched_layout(
                'CG',
                circuit,
                assignment,
                args.trap_capacity,
                args.num_layout_passes,
                args.gate_type,
                args.grid_cols,
                args.grid_rows,
                args.routing_mode,
                congestion_override=args.cg_congestion_rate_override,
                print_position_graph=args.print_position_graph,
            )
            pgs_results = [
                compile_from_matched_layout(
                    'PGS',
                    circuit,
                    assignment,
                    args.trap_capacity,
                    args.num_layout_passes,
                    args.gate_type,
                    args.grid_cols,
                    args.grid_rows,
                    args.routing_mode,
                    congestion_override=args.pgs_congestion_rate_override,
                    pgs_move_path_mode=mode,
                    print_position_graph=args.print_position_graph,
                )
                for mode in pgs_move_path_modes
            ]
        else:
            cg = compile_case(
                'CG',
                circuit,
                assignment,
                args.trap_capacity,
                args.num_layout_passes,
                args.gate_type,
                args.grid_cols,
                args.grid_rows,
                args.routing_mode,
                args.stage,
                congestion_override=args.cg_congestion_rate_override,
                print_position_graph=args.print_position_graph,
            )
            pgs_results = [
                compile_case(
                    'PGS',
                    circuit,
                    assignment,
                    args.trap_capacity,
                    args.num_layout_passes,
                    args.gate_type,
                    args.grid_cols,
                    args.grid_rows,
                    args.routing_mode,
                    args.stage,
                    congestion_override=args.pgs_congestion_rate_override,
                    pgs_move_path_mode=mode,
                    print_position_graph=args.print_position_graph,
                )
                for mode in pgs_move_path_modes
            ]
        pgs = pgs_results[0]
    if args.stage == 'layout':
        print('Layout summary:')
        print(f"  CG  compile_time_s={cg['compile_time_s']:.6f}")
        print(f"  CG layout assignment : {cg['data']['ion_assignment_qccd']}")
        for pgs_result in pgs_results:
            label = result_label(pgs_result, multi_pgs=len(pgs_results) > 1)
            print(f"  {label} compile_time_s={pgs_result['compile_time_s']:.6f}")
            print(f"  {label} layout assignment: {pgs_result['data']['ion_assignment_qccd']}")
            print(
                f"  Same layout assignment vs CG ({label}): "
                f"{cg['data']['ion_assignment_qccd'] == pgs_result['data']['ion_assignment_qccd']}",
            )
        return
    if args.stage == 'layout-trace':
        cg_trace = trace_layout_case(
            'CG',
            circuit,
            assignment,
            args.trap_capacity,
            args.num_layout_passes,
            args.gate_type,
            args.grid_cols,
            args.grid_rows,
            args.routing_mode,
            congestion_override=args.cg_congestion_rate_override,
            print_position_graph=args.print_position_graph,
        )
        pgs_trace = trace_layout_case(
            'PGS',
            circuit,
            assignment,
            args.trap_capacity,
            args.num_layout_passes,
            args.gate_type,
            args.grid_cols,
            args.grid_rows,
            args.routing_mode,
            congestion_override=args.pgs_congestion_rate_override,
            print_position_graph=args.print_position_graph,
        )
        print('Layout trace:')
        print(f"  CG pre-layout front locations : {cg_trace['front_locations']}")
        print(f"  PGS pre-layout front locations: {pgs_trace['front_locations']}")
        print(f"  CG pre-layout rear locations  : {cg_trace['rear_locations']}")
        print(f"  PGS pre-layout rear locations : {pgs_trace['rear_locations']}")
        print(f"  CG start assignment  : {cg_trace['initial_assignment']}")
        print(f"  PGS start assignment : {pgs_trace['initial_assignment']}")
        print(
            '  Same pre-layout assignment: '
            f"{cg_trace['initial_assignment'] == pgs_trace['initial_assignment']}",
        )
        first_layout_divergence = None
        first_layout_entries: tuple[tuple[str, dict[int, int]], tuple[str, dict[int, int]]] | None = None
        for (cg_label, cg_assignment), (pgs_label, pgs_assignment) in zip(
            cg_trace['snapshots'],
            pgs_trace['snapshots'],
        ):
            same = cg_assignment == pgs_assignment
            print(f'  {cg_label}: same={same}')
            if first_layout_divergence is None and not same:
                first_layout_divergence = cg_label
                first_layout_entries = (
                    (cg_label, cg_assignment),
                    (pgs_label, pgs_assignment),
                )
        if first_layout_divergence is None:
            print('  First layout divergence: none')
        else:
            print(f'  First layout divergence: {first_layout_divergence}')
            if first_layout_entries is not None:
                print(f'  CG divergent assignment : {first_layout_entries[0][1]}')
                print(f'  PGS divergent assignment: {first_layout_entries[1][1]}')
        return
    if args.stage == 'layout-wrapper-trace':
        cg_trace = trace_layout_wrapper_case(
            'CG',
            circuit,
            assignment,
            args.trap_capacity,
            args.num_layout_passes,
            args.gate_type,
            args.grid_cols,
            args.grid_rows,
            args.routing_mode,
            congestion_override=args.cg_congestion_rate_override,
            print_position_graph=args.print_position_graph,
        )
        pgs_trace = trace_layout_wrapper_case(
            'PGS',
            circuit,
            assignment,
            args.trap_capacity,
            args.num_layout_passes,
            args.gate_type,
            args.grid_cols,
            args.grid_rows,
            args.routing_mode,
            congestion_override=args.pgs_congestion_rate_override,
            print_position_graph=args.print_position_graph,
        )
        print('Layout wrapper trace:')
        print(f"  CG pre-layout front locations : {cg_trace['front_locations']}")
        print(f"  PGS pre-layout front locations: {pgs_trace['front_locations']}")
        print(f"  CG pre-layout rear locations  : {cg_trace['rear_locations']}")
        print(f"  PGS pre-layout rear locations : {pgs_trace['rear_locations']}")
        print(f"  CG start assignment  : {cg_trace['initial_assignment']}")
        print(f"  PGS start assignment : {pgs_trace['initial_assignment']}")
        print(
            '  Same pre-layout assignment: '
            f"{cg_trace['initial_assignment'] == pgs_trace['initial_assignment']}",
        )
        first_layout_divergence = None
        first_layout_entries: tuple[tuple[str, dict[int, int]], tuple[str, dict[int, int]]] | None = None
        for (cg_label, cg_assignment), (pgs_label, pgs_assignment) in zip(
            cg_trace['snapshots'],
            pgs_trace['snapshots'],
        ):
            same = cg_assignment == pgs_assignment
            print(f'  {cg_label}: same={same}')
            if first_layout_divergence is None and not same:
                first_layout_divergence = cg_label
                first_layout_entries = (
                    (cg_label, cg_assignment),
                    (pgs_label, pgs_assignment),
                )
        if first_layout_divergence is None:
            print('  First layout divergence: none')
        else:
            print(f'  First layout divergence: {first_layout_divergence}')
            if first_layout_entries is not None:
                print(f'  CG divergent assignment : {first_layout_entries[0][1]}')
                print(f'  PGS divergent assignment: {first_layout_entries[1][1]}')
            if first_layout_divergence.startswith('forward_'):
                cg_forward_trace = next(
                    (trace for label, trace in cg_trace['forward_traces']
                     if label == first_layout_divergence),
                    [],
                )
                pgs_forward_trace = next(
                    (trace for label, trace in pgs_trace['forward_traces']
                     if label == first_layout_divergence),
                    [],
                )
                print(f'  {first_layout_divergence} compact trace:')
                print_compact_forward_trace('CG ', cg_forward_trace, args.window)
                print_compact_forward_trace('PGS', pgs_forward_trace, args.window)
        print(f"  CG final assignment  : {cg_trace['final_assignment']}")
        print(f"  PGS final assignment : {pgs_trace['final_assignment']}")
        return
    if args.stage == 'forward-pass':
        cg_trace = trace_forward_pass_case(
            'CG',
            circuit,
            assignment,
            args.trap_capacity,
            args.gate_type,
            args.grid_cols,
            args.grid_rows,
            args.routing_mode,
            congestion_override=args.cg_congestion_rate_override,
            print_position_graph=args.print_position_graph,
        )
        pgs_trace = trace_forward_pass_case(
            'PGS',
            circuit,
            assignment,
            args.trap_capacity,
            args.gate_type,
            args.grid_cols,
            args.grid_rows,
            args.routing_mode,
            congestion_override=args.pgs_congestion_rate_override,
            print_position_graph=args.print_position_graph,
        )
        print('Forward pass trace:')
        print(f"  CG initial assignment : {cg_trace['initial_assignment']}")
        print(f"  PGS initial assignment: {pgs_trace['initial_assignment']}")
        print(
            '  Same initial assignment: '
            f"{cg_trace['initial_assignment'] == pgs_trace['initial_assignment']}",
        )
        print(f"  CG first-front locations : {cg_trace['front_locations']}")
        print(f"  PGS first-front locations: {pgs_trace['front_locations']}")
        print(f"  CG trace steps : {len(cg_trace['trace'])}")
        print(f"  PGS trace steps: {len(pgs_trace['trace'])}")

        first_divergence = None
        for index, (cg_entry, pgs_entry) in enumerate(
            zip(cg_trace['trace'], pgs_trace['trace']),
        ):
            if cg_entry != pgs_entry:
                first_divergence = index
                break
        if first_divergence is None and len(cg_trace['trace']) != len(pgs_trace['trace']):
            first_divergence = min(len(cg_trace['trace']), len(pgs_trace['trace']))

        if first_divergence is None:
            print('  No forward-pass divergence found.')
            print(f"  CG final forward assignment : {cg_trace['final_assignment']}")
            print(f"  PGS final forward assignment: {pgs_trace['final_assignment']}")
            return

        print(f'  First forward-pass divergence index: {first_divergence}')
        start = max(0, first_divergence - 1)
        end = min(
            max(len(cg_trace['trace']), len(pgs_trace['trace'])),
            first_divergence + args.window,
        )
        for index in range(start, end):
            print(f'\nStep {index}')
            if index < len(cg_trace['trace']):
                print_forward_trace_entry('CG ', cg_trace['trace'][index])
                print_resolve_trace('CG ', cg_trace['trace'][index].get('brute_force_trace'))
            else:
                print('  CG  <no entry>')
            if index < len(pgs_trace['trace']):
                print_forward_trace_entry('PGS', pgs_trace['trace'][index])
                print_resolve_trace('PGS', pgs_trace['trace'][index].get('brute_force_trace'))
            else:
                print('  PGS <no entry>')
        print(f"\n  CG final forward assignment : {cg_trace['final_assignment']}")
        print(f"  PGS final forward assignment: {pgs_trace['final_assignment']}")
        return
    print('Summary:')
    print_result_summary(cg, multi_pgs=len(pgs_results) > 1)
    for pgs_result in pgs_results:
        print_result_summary(pgs_result, multi_pgs=len(pgs_results) > 1)

    if args.print_paths:
        for pgs_result in pgs_results:
            move_pairs: list[tuple[int, int]] = []
            for inst in cg['instruction_list'] + pgs_result['instruction_list']:
                if inst[0].startswith('Move'):
                    _, move, _ = normalize_move(inst)
                    move_pairs.append(move)
            label = result_label(pgs_result, multi_pgs=len(pgs_results) > 1)
            print(f'\nPath comparison for {label}:')
            with _TemporaryEnv('BQSKIT_PGS_MOVE_PATH_MODE', pgs_result.get('pgs_move_path_mode')):
                print_path_comparison(base_model, pgs_base_model, move_pairs)

    if len(pgs_results) != 1:
        print('\nDetailed normalized divergence view is only shown for a single PGS mode.')
        return

    current_cg_assignment = copy.deepcopy(
        cg['data'].get('initial_ion_assignment_qccd', assignment),
    )
    current_pgs_assignment = copy.deepcopy(
        pgs['data'].get('initial_ion_assignment_qccd', assignment),
    )
    first_divergence = None
    first_state_divergence = None

    for i, (cg_inst, pgs_inst) in enumerate(
        zip(cg['instruction_list'], pgs['instruction_list']),
    ):
        cg_norm = normalize_instruction(
            cg_inst,
            current_cg_assignment,
            'logical',
        )
        pgs_norm = normalize_instruction(
            pgs_inst,
            current_pgs_assignment,
            'physical',
        )

        current_cg_assignment = cg_norm[2]
        current_pgs_assignment = pgs_norm[2]

        if first_state_divergence is None and current_cg_assignment != current_pgs_assignment:
            first_state_divergence = i
        if cg_norm != pgs_norm:
            first_divergence = i
            break

    if first_divergence is None:
        if len(cg['instruction_list']) != len(pgs['instruction_list']):
            first_divergence = min(
                len(cg['instruction_list']),
                len(pgs['instruction_list']),
            )
        else:
            print('No divergence found after normalization.')
            return

    print(f'First divergence index: {first_divergence}')
    if first_state_divergence is None:
        print('First state divergence index: none')
    else:
        print(f'First state divergence index: {first_state_divergence}')

    start = max(0, first_divergence - 3)
    end = first_divergence + args.window

    cg_assignment = copy.deepcopy(
        cg['data'].get('initial_ion_assignment_qccd', assignment),
    )
    pgs_assignment = copy.deepcopy(
        pgs['data'].get('initial_ion_assignment_qccd', assignment),
    )
    for i in range(min(end, len(cg['instruction_list']), len(pgs['instruction_list']))):
        cg_inst = cg['instruction_list'][i]
        pgs_inst = pgs['instruction_list'][i]
        cg_norm = normalize_instruction(
            cg_inst,
            cg_assignment,
            'logical',
        )
        pgs_norm = normalize_instruction(
            pgs_inst,
            pgs_assignment,
            'physical',
        )
        cg_assignment = cg_norm[2]
        pgs_assignment = pgs_norm[2]

        if i < start:
            continue

        print(f'\nIndex {i}')
        print(f'  CG raw : {cg_inst}')
        print(f'  CG norm: {cg_norm[:2]}')
        print(f'  PGS raw : {pgs_inst}')
        print(f'  PGS norm: {pgs_norm[:2]}')
        print(f'  CG next assignment : {cg_assignment}')
        print(f'  PGS next assignment: {pgs_assignment}')

    if len(cg['instruction_list']) != len(pgs['instruction_list']):
        print('\nRemaining tail:')
        print(f"  CG remaining instructions : {len(cg['instruction_list']) - min(len(cg['instruction_list']), len(pgs['instruction_list']))}")
        print(f"  PGS remaining instructions: {len(pgs['instruction_list']) - min(len(cg['instruction_list']), len(pgs['instruction_list']))}")


if __name__ == '__main__':
    main()
