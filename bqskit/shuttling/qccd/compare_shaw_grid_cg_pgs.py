from __future__ import annotations

import argparse
import ast
import copy
import io
import os
import re
from contextlib import nullcontext
from contextlib import redirect_stdout
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
from bqskit.shuttling.qccd.QCCD_cached_mapping import (
    QCCDMappingAlgorithm as CGCachedMappingAlgorithm,
)
from bqskit.shuttling.qccd.QCCD_mapping_PGS import QCCDMappingAlgorithm as PGSMappingAlgorithm
from bqskit.shuttling.qccd.mapping import QCCDLayoutPass
from bqskit.shuttling.qccd.mapping import QCCDCachedLayoutPass
from bqskit.shuttling.qccd.mapping import QCCDRoutingPass
from bqskit.shuttling.qccd.mapping import QCCDCachedRoutingPass
from bqskit.shuttling.qccd.pgs_passes import QCCDLayoutPassPGS
from bqskit.shuttling.qccd.pgs_passes import QCCDRoutingPassPGS
from bqskit.shuttling.qccd.QCCD_machine import QCCDMachineModel as CGMachineModel
from bqskit.shuttling.qccd.QCCD_cached_machine import (
    QCCDMachineModel as CGCachedMachineModel,
)
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

COMPARE_LAYOUT_SCORING = {
    'decay_delta': 0.001,
    'decay_reset_interval': 5,
    'decay_reset_on_gate': True,
    'extended_set_size': 5,
    'extended_set_weight': 0.5,
}

COMPARE_PGS_ROUTING_SCORING = {
    'gate_count_weight': 0.1,
    **COMPARE_LAYOUT_SCORING,
}

BACKEND_CHOICES = ('CG', 'CG-CACHED', 'PGS')


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
        '--backends',
        nargs='+',
        choices=BACKEND_CHOICES,
        default=['CG', 'PGS'],
        help='Backends to run in compare mode.',
    )
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
        '--congestion-rate-override',
        type=float,
        default=None,
        help='Override the computed congestion rate for both CG and PGS unless a backend-specific override is provided.',
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
    parser.add_argument(
        '--print-machine',
        action='store_true',
        help='Print machine creation and other setup stdout.',
    )
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


def backend_execute_location_mode(backend: str) -> str:
    return 'physical' if backend == 'PGS' else 'logical'


def backend_components(backend: str) -> dict[str, Any]:
    if backend == 'CG':
        return {
            'model_cls': CGMachineModel,
            'mapping_cls': CGMappingAlgorithm,
            'layout_cls': QCCDLayoutPass,
            'routing_cls': QCCDRoutingPass,
        }
    if backend == 'CG-CACHED':
        return {
            'model_cls': CGCachedMachineModel,
            'mapping_cls': CGCachedMappingAlgorithm,
            'layout_cls': QCCDCachedLayoutPass,
            'routing_cls': QCCDCachedRoutingPass,
        }
    if backend == 'PGS':
        return {
            'model_cls': PGSMachineModel,
            'mapping_cls': PGSMappingAlgorithm,
            'layout_cls': QCCDLayoutPassPGS,
            'routing_cls': QCCDRoutingPassPGS,
        }
    raise ValueError(f'Unknown backend: {backend}')


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
    print_machine: bool = False,
) -> dict[str, Any]:
    components = backend_components(backend)
    env_value = pgs_move_path_mode if backend == 'PGS' else None
    with _TemporaryEnv('BQSKIT_PGS_MOVE_PATH_MODE', env_value):
        machine_stdout = (
            nullcontext()
            if (print_position_graph or print_machine)
            else redirect_stdout(io.StringIO())
        )
        with machine_stdout:
            physical_model = create_grid_physical_machine(
                num_cols=grid_cols,
                num_rows=grid_rows,
                trap_capacity=trap_capacity,
            )
            gate_set = GateSet({U3Gate(), CXGate()})
            model_cls = components['model_cls']
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
                components['layout_cls'](
                    total_passes=num_layout_passes,
                    cogestion_rate=congestion,
                    force_bruteforce=force_bruteforce,
                ),
            ]
            if stage == 'full':
                workflow.extend([
                    components['routing_cls'](
                        cogestion_rate=congestion,
                        force_bruteforce=force_bruteforce,
                    ),
                    ApplyPlacement(),
                    UnfoldPass(),
                ])

        compile_stdout = nullcontext() if print_machine else redirect_stdout(io.StringIO())
        with compile_stdout:
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
            'machine_model': data.model,
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
    print_machine: bool = False,
) -> tuple[Circuit, Any]:
    components = backend_components(backend)
    machine_stdout = (
        nullcontext()
        if (print_position_graph or print_machine)
        else redirect_stdout(io.StringIO())
    )
    with machine_stdout:
        physical_model = create_grid_physical_machine(
            num_cols=grid_cols,
            num_rows=grid_rows,
            trap_capacity=trap_capacity,
        )
        gate_set = GateSet({U3Gate(), CXGate()})
        model_cls = components['model_cls']
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

    compile_stdout = nullcontext() if print_machine else redirect_stdout(io.StringIO())
    with compile_stdout:
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
    print_machine: bool = False,
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
        print_machine,
    )
    force_bruteforce = routing_mode == 'bruteforce'
    congestion = congestion_rate(pre_data.model, pre_circuit.num_qudits)
    if congestion_override is not None:
        congestion = float(congestion_override)
    snapshots: list[tuple[str, dict[int, int]]] = []
    components = backend_components(backend)

    if backend != 'PGS':
        algo = components['mapping_cls'](
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
    print_machine: bool = False,
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
            print_machine=print_machine,
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
        'backward_traces': copy.deepcopy(data.get('qccd_layout_wrapper_backward_traces', [])),
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
    print_machine: bool = False,
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
        print_machine,
    )
    force_bruteforce = routing_mode == 'bruteforce'
    congestion = congestion_rate(pre_data.model, pre_circuit.num_qudits)
    if congestion_override is not None:
        congestion = float(congestion_override)

    components = backend_components(backend)
    if backend != 'PGS':
        algo = components['mapping_cls'](
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
        components = backend_components(backend)
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

        if backend != 'PGS':
            algo = components['mapping_cls'](
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
            'machine_model': pre_data.model,
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


def print_compact_backward_trace(prefix: str, trace: list[dict[str, Any]], window: int) -> None:
    print(f'  {prefix}trace steps: {len(trace)}')
    for index, entry in enumerate(trace[:window]):
        move_search = entry.get('move_search') or {}
        print(
            f"  {prefix}{index}: action={entry.get('action')} "
            f"front={entry.get('front_locations')} exec={entry.get('execute_locations')} "
            f"best_move={entry.get('best_move')} "
            f"best_score={move_search.get('best_score')} "
            f"candidates={move_search.get('candidate_moves')}"
        )


def first_trace_divergence_index(
    lhs_trace: list[dict[str, Any]],
    rhs_trace: list[dict[str, Any]],
) -> int | None:
    max_len = max(len(lhs_trace), len(rhs_trace))
    for index in range(max_len):
        lhs_entry = lhs_trace[index] if index < len(lhs_trace) else None
        rhs_entry = rhs_trace[index] if index < len(rhs_trace) else None
        if lhs_entry != rhs_entry:
            return index
    return None


def format_compact_backward_entry(
    prefix: str,
    index: int,
    entry: dict[str, Any] | None,
) -> str:
    if entry is None:
        return f'  {prefix}{index}: <missing>'
    move_search = entry.get('move_search') or {}
    return (
        f"  {prefix}{index}: action={entry.get('action')} "
        f"front={entry.get('front_locations')} exec={entry.get('execute_locations')} "
        f"best_move={entry.get('best_move')} "
        f"best_score={move_search.get('best_score')} "
        f"candidates={move_search.get('candidate_moves')} "
        f"post={entry.get('post_assignment')}"
    )


def print_compact_backward_trace_pair(
    cg_trace: list[dict[str, Any]],
    pgs_trace: list[dict[str, Any]],
    window: int,
    lhs_label: str = 'CG',
    rhs_label: str = 'PGS',
) -> None:
    print(f'  {lhs_label} trace steps : {len(cg_trace)}')
    print(f'  {rhs_label} trace steps: {len(pgs_trace)}')
    divergence_index = first_trace_divergence_index(cg_trace, pgs_trace)
    if divergence_index is None:
        print('  No backward-trace entry divergence found.')
        divergence_index = 0
    else:
        print(f'  First differing backward-trace entry: {divergence_index}')

    max_len = max(len(cg_trace), len(pgs_trace))
    half_window = max(1, window // 2)
    start = max(0, divergence_index - half_window)
    end = min(max_len, start + window)
    start = max(0, end - window)
    print(f'  Showing backward-trace entries [{start}, {end}):')
    for index in range(start, end):
        cg_entry = cg_trace[index] if index < len(cg_trace) else None
        pgs_entry = pgs_trace[index] if index < len(pgs_trace) else None
        print(format_compact_backward_entry(f'{lhs_label} ', index, cg_entry))
        print(format_compact_backward_entry(f'{rhs_label}', index, pgs_entry))


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


def _format_setting(value: Any) -> str:
    if isinstance(value, float):
        return f'{value:.12g}'
    return str(value)


def print_compare_run_config(
    args: argparse.Namespace,
    *,
    circuit: Circuit,
    base_models: dict[str, Any],
    backends: list[str],
    pgs_move_path_modes: list[str],
) -> None:
    force_bruteforce = args.routing_mode == 'bruteforce'
    backend_overrides = {
        'CG': (
            args.cg_congestion_rate_override
            if args.cg_congestion_rate_override is not None
            else args.congestion_rate_override
        ),
        'CG-CACHED': (
            args.cg_congestion_rate_override
            if args.cg_congestion_rate_override is not None
            else args.congestion_rate_override
        ),
        'PGS': (
            args.pgs_congestion_rate_override
            if args.pgs_congestion_rate_override is not None
            else args.congestion_rate_override
        ),
    }
    cg_override = (
        args.cg_congestion_rate_override
        if args.cg_congestion_rate_override is not None
        else args.congestion_rate_override
    )
    pgs_override = (
        args.pgs_congestion_rate_override
        if args.pgs_congestion_rate_override is not None
        else args.congestion_rate_override
    )
    computed_congestion = {
        backend: congestion_rate(base_models[backend], circuit.num_qudits)
        for backend in backends
    }
    effective_congestion = {
        backend: (
            computed_congestion[backend]
            if backend_overrides[backend] is None
            else float(backend_overrides[backend])
        )
        for backend in backends
    }
    congestion_source = {
        backend: (
            'computed'
            if backend_overrides[backend] is None
            else (
                'shared override'
                if (
                    (backend in {'CG', 'CG-CACHED'} and args.cg_congestion_rate_override is None)
                    or (backend == 'PGS' and args.pgs_congestion_rate_override is None)
                )
                else 'backend override'
            )
        )
        for backend in backends
    }

    print('Compare config:')
    settings: list[tuple[str, Any]] = [
        ('input_filename', args.input_filename),
        ('backends', backends),
        ('circuit_num_qudits', circuit.num_qudits),
        ('stage', args.stage),
        ('routing_mode', args.routing_mode),
        ('force_bruteforce', force_bruteforce),
        ('gate_type', args.gate_type),
        ('seed', args.seed),
        ('grid_cols', args.grid_cols),
        ('grid_rows', args.grid_rows),
        ('trap_capacity', args.trap_capacity),
        ('num_layout_passes', args.num_layout_passes),
        ('pgs_move_path_modes', pgs_move_path_modes),
        ('timing_data', TIMING_DATA),
        ('layout_scoring', COMPARE_LAYOUT_SCORING),
    ]
    if 'CG' in backends:
        settings.extend([
            ('cg_congestion_rate', f"{effective_congestion['CG']:.12g} ({congestion_source['CG']})"),
            ('cg_computed_congestion_rate', computed_congestion['CG']),
            (
                'cg_layout_kwargs',
                {
                    **COMPARE_LAYOUT_SCORING,
                    'congestion_rate': effective_congestion['CG'],
                    'force_bruteforce': force_bruteforce,
                },
            ),
            (
                'cg_routing_kwargs',
                {
                    'congestion_rate': effective_congestion['CG'],
                    'force_bruteforce': force_bruteforce,
                },
            ),
        ])
    if 'CG-CACHED' in backends:
        settings.extend([
            ('cg_cached_congestion_rate', f"{effective_congestion['CG-CACHED']:.12g} ({congestion_source['CG-CACHED']})"),
            ('cg_cached_computed_congestion_rate', computed_congestion['CG-CACHED']),
            (
                'cg_cached_layout_kwargs',
                {
                    **COMPARE_LAYOUT_SCORING,
                    'congestion_rate': effective_congestion['CG-CACHED'],
                    'force_bruteforce': force_bruteforce,
                },
            ),
            (
                'cg_cached_routing_kwargs',
                {
                    'congestion_rate': effective_congestion['CG-CACHED'],
                    'force_bruteforce': force_bruteforce,
                },
            ),
        ])
    if 'PGS' in backends:
        settings.extend([
            ('pgs_congestion_rate', f"{effective_congestion['PGS']:.12g} ({congestion_source['PGS']})"),
            ('pgs_computed_congestion_rate', computed_congestion['PGS']),
            (
                'pgs_layout_kwargs',
                {
                    **COMPARE_LAYOUT_SCORING,
                    'congestion_rate': effective_congestion['PGS'],
                    'force_bruteforce': force_bruteforce,
                },
            ),
            (
                'pgs_routing_kwargs',
                {
                    **COMPARE_PGS_ROUTING_SCORING,
                    'congestion_rate': effective_congestion['PGS'],
                    'force_bruteforce': force_bruteforce,
                },
            ),
        ])
    for key, value in settings:
        print(f"  {key:<26} {_format_setting(value)}")


def main() -> None:
    args = parse_args()
    backends = list(dict.fromkeys(args.backends))
    cg_congestion_override = (
        args.cg_congestion_rate_override
        if args.cg_congestion_rate_override is not None
        else args.congestion_rate_override
    )
    pgs_congestion_override = (
        args.pgs_congestion_rate_override
        if args.pgs_congestion_rate_override is not None
        else args.congestion_rate_override
    )
    backend_overrides = {
        'CG': cg_congestion_override,
        'CG-CACHED': cg_congestion_override,
        'PGS': pgs_congestion_override,
    }
    pgs_move_path_modes = list(dict.fromkeys(args.pgs_move_path_modes))
    if 'PGS' not in backends:
        pgs_move_path_modes = ['hops']
    if len(pgs_move_path_modes) > 1 and args.stage not in (
        'layout',
        'full',
        'full-matched-layout',
    ):
        raise ValueError(
            'Multiple PGS move-path modes are only supported for '
            "{'layout', 'full', 'full-matched-layout'} stages.",
        )
    if args.stage in ('layout-trace', 'layout-wrapper-trace', 'forward-pass') and len(backends) != 2:
        raise ValueError(
            f'{args.stage} currently requires exactly two backends.',
        )
    os.environ['BQSKIT_QCCD_VERBOSE'] = '1' if args.verbose_status else '0'
    os.environ['BQSKIT_QCCD_PRINT_MACHINE'] = '1' if args.print_machine else '0'
    os.environ['BQSKIT_QCCD_CAPTURE_TRACE'] = (
        '1' if args.stage == 'forward-pass' else '0'
    )
    circuit_path = (
        Path('bqskit/shuttling/qccd/benchmark_circuits')
        / f'{args.input_filename}.qasm'
    )
    circuit = Circuit.from_file(str(circuit_path))

    base_models: dict[str, Any] = {}
    for backend in backends:
        model_cls = backend_components(backend)['model_cls']
        base_models[backend] = model_cls(
            gate_set=GateSet({U3Gate(), CXGate()}),
            physical_graph=create_grid_physical_machine(
                num_cols=args.grid_cols,
                num_rows=args.grid_rows,
                trap_capacity=args.trap_capacity,
            ),
            multi_qudit_gate_type=args.gate_type,
            timing_data=TIMING_DATA,
        )
    assignment_model = (
        base_models['CG']
        if 'CG' in base_models
        else next(iter(base_models.values()))
    )
    assignment = build_assignment(assignment_model, circuit.num_qudits, args.seed)

    print_compare_run_config(
        args,
        circuit=circuit,
        base_models=base_models,
        backends=backends,
        pgs_move_path_modes=pgs_move_path_modes,
    )
    print(f'Initial ion assignment: {assignment}')
    print(f'Routing mode: {args.routing_mode}')
    print(f'Stage: {args.stage}')
    if 'PGS' in backends and len(pgs_move_path_modes) > 1:
        print(f'PGS move-path modes: {pgs_move_path_modes}')
    if args.stage in ('layout', 'full', 'full-matched-layout'):
        results: list[dict[str, Any]] = []
        for backend in backends:
            if backend == 'PGS':
                modes = pgs_move_path_modes
            else:
                modes = [None]
            for mode in modes:
                if args.stage == 'full-matched-layout':
                    result = compile_from_matched_layout(
                        backend,
                        circuit,
                        assignment,
                        args.trap_capacity,
                        args.num_layout_passes,
                        args.gate_type,
                        args.grid_cols,
                        args.grid_rows,
                        args.routing_mode,
                        congestion_override=backend_overrides[backend],
                        pgs_move_path_mode=mode,
                        print_position_graph=args.print_position_graph,
                    )
                else:
                    result = compile_case(
                        backend,
                        circuit,
                        assignment,
                        args.trap_capacity,
                        args.num_layout_passes,
                        args.gate_type,
                        args.grid_cols,
                        args.grid_rows,
                        args.routing_mode,
                        args.stage,
                        congestion_override=backend_overrides[backend],
                        pgs_move_path_mode=mode,
                        print_position_graph=args.print_position_graph,
                    )
                results.append(result)
    if args.stage == 'layout':
        print('Layout summary:')
        reference = results[0]
        print(
            f"  {result_label(reference, multi_pgs=('PGS' in backends and len(pgs_move_path_modes) > 1))} "
            f"compile_time_s={reference['compile_time_s']:.6f}",
        )
        print(f"  {result_label(reference, multi_pgs=('PGS' in backends and len(pgs_move_path_modes) > 1))} layout assignment : {reference['data']['ion_assignment_qccd']}")
        for result in results[1:]:
            label = result_label(result, multi_pgs=('PGS' in backends and len(pgs_move_path_modes) > 1))
            print(f"  {label} compile_time_s={result['compile_time_s']:.6f}")
            print(f"  {label} layout assignment: {result['data']['ion_assignment_qccd']}")
            print(
                f"  Same layout assignment vs {result_label(reference, multi_pgs=('PGS' in backends and len(pgs_move_path_modes) > 1))} ({label}): "
                f"{reference['data']['ion_assignment_qccd'] == result['data']['ion_assignment_qccd']}",
            )
        return
    if args.stage == 'layout-trace':
        lhs_backend, rhs_backend = backends
        cg_trace = trace_layout_case(
            lhs_backend,
            circuit,
            assignment,
            args.trap_capacity,
            args.num_layout_passes,
            args.gate_type,
            args.grid_cols,
            args.grid_rows,
            args.routing_mode,
            congestion_override=backend_overrides[lhs_backend],
            print_position_graph=args.print_position_graph,
            print_machine=args.print_machine,
        )
        pgs_trace = trace_layout_case(
            rhs_backend,
            circuit,
            assignment,
            args.trap_capacity,
            args.num_layout_passes,
            args.gate_type,
            args.grid_cols,
            args.grid_rows,
            args.routing_mode,
            congestion_override=backend_overrides[rhs_backend],
            print_position_graph=args.print_position_graph,
            print_machine=args.print_machine,
        )
        print('Layout trace:')
        print(f"  {lhs_backend} pre-layout front locations : {cg_trace['front_locations']}")
        print(f"  {rhs_backend} pre-layout front locations: {pgs_trace['front_locations']}")
        print(f"  {lhs_backend} pre-layout rear locations  : {cg_trace['rear_locations']}")
        print(f"  {rhs_backend} pre-layout rear locations : {pgs_trace['rear_locations']}")
        print(f"  {lhs_backend} start assignment  : {cg_trace['initial_assignment']}")
        print(f"  {rhs_backend} start assignment : {pgs_trace['initial_assignment']}")
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
                print(f'  {lhs_backend} divergent assignment : {first_layout_entries[0][1]}')
                print(f'  {rhs_backend} divergent assignment: {first_layout_entries[1][1]}')
        return
    if args.stage == 'layout-wrapper-trace':
        lhs_backend, rhs_backend = backends
        cg_trace = trace_layout_wrapper_case(
            lhs_backend,
            circuit,
            assignment,
            args.trap_capacity,
            args.num_layout_passes,
            args.gate_type,
            args.grid_cols,
            args.grid_rows,
            args.routing_mode,
            congestion_override=backend_overrides[lhs_backend],
            print_position_graph=args.print_position_graph,
            print_machine=args.print_machine,
        )
        pgs_trace = trace_layout_wrapper_case(
            rhs_backend,
            circuit,
            assignment,
            args.trap_capacity,
            args.num_layout_passes,
            args.gate_type,
            args.grid_cols,
            args.grid_rows,
            args.routing_mode,
            congestion_override=backend_overrides[rhs_backend],
            print_position_graph=args.print_position_graph,
            print_machine=args.print_machine,
        )
        print('Layout wrapper trace:')
        print(f"  {lhs_backend} pre-layout front locations : {cg_trace['front_locations']}")
        print(f"  {rhs_backend} pre-layout front locations: {pgs_trace['front_locations']}")
        print(f"  {lhs_backend} pre-layout rear locations  : {cg_trace['rear_locations']}")
        print(f"  {rhs_backend} pre-layout rear locations : {pgs_trace['rear_locations']}")
        print(f"  {lhs_backend} start assignment  : {cg_trace['initial_assignment']}")
        print(f"  {rhs_backend} start assignment : {pgs_trace['initial_assignment']}")
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
                print(f'  {lhs_backend} divergent assignment : {first_layout_entries[0][1]}')
                print(f'  {rhs_backend} divergent assignment: {first_layout_entries[1][1]}')
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
                print_compact_forward_trace(f'{lhs_backend} ', cg_forward_trace, args.window)
                print_compact_forward_trace(f'{rhs_backend}', pgs_forward_trace, args.window)
            if first_layout_divergence.startswith('backward_'):
                cg_backward_trace = next(
                    (trace for label, trace in cg_trace['backward_traces']
                     if label == first_layout_divergence),
                    [],
                )
                pgs_backward_trace = next(
                    (trace for label, trace in pgs_trace['backward_traces']
                     if label == first_layout_divergence),
                    [],
                )
                print(f'  {first_layout_divergence} compact trace:')
                print_compact_backward_trace_pair(
                    cg_backward_trace,
                    pgs_backward_trace,
                    args.window,
                    lhs_label=lhs_backend,
                    rhs_label=rhs_backend,
                )
        print(f"  {lhs_backend} final assignment  : {cg_trace['final_assignment']}")
        print(f"  {rhs_backend} final assignment : {pgs_trace['final_assignment']}")
        return
    if args.stage == 'forward-pass':
        lhs_backend, rhs_backend = backends
        cg_trace = trace_forward_pass_case(
            lhs_backend,
            circuit,
            assignment,
            args.trap_capacity,
            args.gate_type,
            args.grid_cols,
            args.grid_rows,
            args.routing_mode,
            congestion_override=backend_overrides[lhs_backend],
            print_position_graph=args.print_position_graph,
            print_machine=args.print_machine,
        )
        pgs_trace = trace_forward_pass_case(
            rhs_backend,
            circuit,
            assignment,
            args.trap_capacity,
            args.gate_type,
            args.grid_cols,
            args.grid_rows,
            args.routing_mode,
            congestion_override=backend_overrides[rhs_backend],
            print_position_graph=args.print_position_graph,
            print_machine=args.print_machine,
        )
        print('Forward pass trace:')
        print(f"  {lhs_backend} initial assignment : {cg_trace['initial_assignment']}")
        print(f"  {rhs_backend} initial assignment: {pgs_trace['initial_assignment']}")
        print(
            '  Same initial assignment: '
            f"{cg_trace['initial_assignment'] == pgs_trace['initial_assignment']}",
        )
        print(f"  {lhs_backend} first-front locations : {cg_trace['front_locations']}")
        print(f"  {rhs_backend} first-front locations: {pgs_trace['front_locations']}")
        print(f"  {lhs_backend} trace steps : {len(cg_trace['trace'])}")
        print(f"  {rhs_backend} trace steps: {len(pgs_trace['trace'])}")

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
            print(f"  {lhs_backend} final forward assignment : {cg_trace['final_assignment']}")
            print(f"  {rhs_backend} final forward assignment: {pgs_trace['final_assignment']}")
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
                print_forward_trace_entry(f'{lhs_backend} ', cg_trace['trace'][index])
                print_resolve_trace(f'{lhs_backend} ', cg_trace['trace'][index].get('brute_force_trace'))
            else:
                print(f'  {lhs_backend}  <no entry>')
            if index < len(pgs_trace['trace']):
                print_forward_trace_entry(f'{rhs_backend}', pgs_trace['trace'][index])
                print_resolve_trace(f'{rhs_backend}', pgs_trace['trace'][index].get('brute_force_trace'))
            else:
                print(f'  {rhs_backend} <no entry>')
        print(f"\n  {lhs_backend} final forward assignment : {cg_trace['final_assignment']}")
        print(f"  {rhs_backend} final forward assignment: {pgs_trace['final_assignment']}")
        return
    print('Summary:')
    multi_pgs = ('PGS' in backends and len(pgs_move_path_modes) > 1)
    for result in results:
        print_result_summary(result, multi_pgs=multi_pgs)

    if args.print_paths:
        if len(results) != 2:
            print('\nPath comparison is only shown for two-result runs.')
        else:
            lhs_result, rhs_result = results
            move_pairs: list[tuple[int, int]] = []
            for inst in lhs_result['instruction_list'] + rhs_result['instruction_list']:
                if inst[0].startswith('Move'):
                    _, move, _ = normalize_move(inst)
                    move_pairs.append(move)
            lhs_model = lhs_result['machine_model']
            rhs_model = rhs_result['machine_model']
            if lhs_result['backend'] == 'PGS' or rhs_result['backend'] == 'PGS':
                print('\nPath comparison:')
                if lhs_result['backend'] == 'PGS':
                    print_path_comparison(rhs_model, lhs_model, move_pairs)
                else:
                    print_path_comparison(lhs_model, rhs_model, move_pairs)
            else:
                print('\nPath comparison currently expects one PGS backend.')

    if len(results) != 2:
        print('\nDetailed normalized divergence view is only shown for two-result runs.')
        return
    lhs_result, rhs_result = results

    current_lhs_assignment = copy.deepcopy(
        lhs_result['data'].get('initial_ion_assignment_qccd', assignment),
    )
    current_rhs_assignment = copy.deepcopy(
        rhs_result['data'].get('initial_ion_assignment_qccd', assignment),
    )
    first_divergence = None
    first_state_divergence = None

    for i, (lhs_inst, rhs_inst) in enumerate(
        zip(lhs_result['instruction_list'], rhs_result['instruction_list']),
    ):
        lhs_norm = normalize_instruction(
            lhs_inst,
            current_lhs_assignment,
            backend_execute_location_mode(lhs_result['backend']),
        )
        rhs_norm = normalize_instruction(
            rhs_inst,
            current_rhs_assignment,
            backend_execute_location_mode(rhs_result['backend']),
        )

        current_lhs_assignment = lhs_norm[2]
        current_rhs_assignment = rhs_norm[2]

        if first_state_divergence is None and current_lhs_assignment != current_rhs_assignment:
            first_state_divergence = i
        if lhs_norm != rhs_norm:
            first_divergence = i
            break

    if first_divergence is None:
        if len(lhs_result['instruction_list']) != len(rhs_result['instruction_list']):
            first_divergence = min(
                len(lhs_result['instruction_list']),
                len(rhs_result['instruction_list']),
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

    lhs_assignment = copy.deepcopy(
        lhs_result['data'].get('initial_ion_assignment_qccd', assignment),
    )
    rhs_assignment = copy.deepcopy(
        rhs_result['data'].get('initial_ion_assignment_qccd', assignment),
    )
    for i in range(min(end, len(lhs_result['instruction_list']), len(rhs_result['instruction_list']))):
        lhs_inst = lhs_result['instruction_list'][i]
        rhs_inst = rhs_result['instruction_list'][i]
        lhs_norm = normalize_instruction(
            lhs_inst,
            lhs_assignment,
            backend_execute_location_mode(lhs_result['backend']),
        )
        rhs_norm = normalize_instruction(
            rhs_inst,
            rhs_assignment,
            backend_execute_location_mode(rhs_result['backend']),
        )
        lhs_assignment = lhs_norm[2]
        rhs_assignment = rhs_norm[2]

        if i < start:
            continue

        print(f'\nIndex {i}')
        print(f"  {lhs_result['backend']} raw : {lhs_inst}")
        print(f"  {lhs_result['backend']} norm: {lhs_norm[:2]}")
        print(f"  {rhs_result['backend']} raw : {rhs_inst}")
        print(f"  {rhs_result['backend']} norm: {rhs_norm[:2]}")
        print(f"  {lhs_result['backend']} next assignment : {lhs_assignment}")
        print(f"  {rhs_result['backend']} next assignment: {rhs_assignment}")

    if len(lhs_result['instruction_list']) != len(rhs_result['instruction_list']):
        print('\nRemaining tail:')
        common_len = min(len(lhs_result['instruction_list']), len(rhs_result['instruction_list']))
        print(f"  {lhs_result['backend']} remaining instructions : {len(lhs_result['instruction_list']) - common_len}")
        print(f"  {rhs_result['backend']} remaining instructions: {len(rhs_result['instruction_list']) - common_len}")


if __name__ == '__main__':
    main()
