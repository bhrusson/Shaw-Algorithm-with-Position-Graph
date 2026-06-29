from __future__ import annotations

import argparse
import ast
import copy
import io
import os
import pickle
import random
import re
import tracemalloc
from contextlib import nullcontext
from contextlib import redirect_stdout
from pathlib import Path
from timeit import default_timer as timer
from typing import Any

from bqskit.compiler import Compiler
from bqskit.compiler.basepass import BasePass
from bqskit.compiler.gateset import GateSet
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CXGate
from bqskit.ir.gates.parameterized import U3Gate
from bqskit.passes import ApplyPlacement
from bqskit.passes import QuickPartitioner
from bqskit.passes import SetModelPass
from bqskit.passes import UnfoldPass
from bqskit.passes import UpdateDataPass

from bqskit.shuttling.qccd import create_grid_physical_machine
from bqskit.shuttling.qccd import create_testing_physical_machine
from bqskit.shuttling.qccd.QCCD_machine import QCCDMachineModel
from bqskit.shuttling.qccd.QCCD_schedule import print_event_trace
from bqskit.shuttling.qccd.QCCD_schedule import schedule_qccd_from_instructions_v3
from bqskit.shuttling.qccd.pgs_passes import QCCDLayoutPassPGS
from bqskit.shuttling.qccd.pgs_passes import QCCDRoutingPassPGS


REPO_ROOT = Path(__file__).resolve().parent
BENCHMARK_DIR_ENV_VAR = 'BQSKIT_SHUTTLING_BENCHMARK_DIR'

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


class TimedPass(BasePass):
    """Wrap a compiler pass and record its wall time in PassData."""

    def __init__(self, label: str, pass_obj: BasePass) -> None:
        self.label = label
        self.pass_obj = pass_obj

    @property
    def name(self) -> str:
        return f'Timed({self.label})'

    async def run(self, circuit: Circuit, data: PassData) -> None:
        start = timer()
        await self.pass_obj.run(circuit, data)
        elapsed = timer() - start
        timings = data.get('qccd_pass_timings', [])
        timings.append({
            'label': self.label,
            'pass': self.pass_obj.name,
            'seconds': elapsed,
            'operations': int(circuit.num_operations),
            'qudits': int(circuit.num_qudits),
        })
        data['qccd_pass_timings'] = timings


class TemporaryEnv:
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


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {'1', 'true', 't', 'yes', 'y', 'on'}:
        return True
    if normalized in {'0', 'false', 'f', 'no', 'n', 'off'}:
        return False
    raise argparse.ArgumentTypeError(f'Expected a boolean value, got {value!r}.')


def format_optional_float(value: float | None, digits: int = 6) -> str:
    if value is None:
        return 'n/a'
    return f'{value:.{digits}f}'


def print_sweep_style_summary(
    *,
    label: str,
    compile_time_s: float | None,
    runtime_us: float | None,
    fidelity: float | None,
    instructions: int,
    execute_rounds: int | None,
    move_rounds: int | None,
) -> None:
    execute_text = 'n/a' if execute_rounds is None else str(execute_rounds)
    move_text = 'n/a' if move_rounds is None else str(move_rounds)
    print(
        f'  {label:<13} '
        f'compile_time_s={format_optional_float(compile_time_s)} '
        f'runtime_us={format_optional_float(runtime_us, 3)} '
        f'fidelity={format_optional_float(fidelity, 12)} '
        f'instructions={instructions} '
        f'execute_rounds={execute_text} '
        f'move_rounds={move_text}',
    )


def benchmark_filename(input_filename: str) -> str:
    return input_filename if input_filename.endswith('.qasm') else f'{input_filename}.qasm'


def benchmark_search_dirs(args: argparse.Namespace) -> list[Path]:
    dirs: list[Path] = []
    if args.benchmark_dir is not None:
        dirs.append(args.benchmark_dir)
    env_dir = os.environ.get(BENCHMARK_DIR_ENV_VAR)
    if env_dir:
        dirs.append(Path(env_dir))
    dirs.extend([
        REPO_ROOT / 'benchmark_circuits',
        REPO_ROOT / 'bqskit' / 'shuttling' / 'qccd' / 'benchmark_circuits',
    ])
    return dirs


def resolve_benchmark_circuit_path(input_filename: str, args: argparse.Namespace) -> Path:
    candidate = Path(input_filename)
    candidates = [candidate]
    if candidate.suffix != '.qasm':
        candidates.append(candidate.with_suffix('.qasm'))

    for path in candidates:
        expanded = path.expanduser()
        if expanded.exists():
            return expanded.resolve()

    filename = benchmark_filename(input_filename)
    checked: list[Path] = []
    for root in benchmark_search_dirs(args):
        path = root.expanduser() / filename
        checked.append(path)
        if path.exists():
            return path.resolve()

    checked_text = '\n  '.join(str(path) for path in checked)
    raise FileNotFoundError(
        f'Could not find benchmark circuit {filename}. Checked:\n  {checked_text}',
    )


def build_assignment(
    machine_model: QCCDMachineModel,
    num_qudits: int,
    seed: int,
) -> dict[int, int]:
    rng = random.Random(seed)
    available = []
    for trap in machine_model.physical_graph.trap_list:
        available += list(machine_model.physical_to_position[trap.id])
    if num_qudits > len(available):
        raise ValueError(
            f'Circuit requires {num_qudits} qudits but the machine only has '
            f'{len(available)} available trap positions.',
        )
    chosen = rng.sample(available, num_qudits)
    return {i: int(chosen[i]) for i in range(num_qudits)}


def congestion_rate(machine_model: QCCDMachineModel, num_qudits: int) -> float:
    executable_spaces = 0
    for trap in machine_model.physical_graph.executable_trap_list:
        executable_spaces += trap.max_num_ions
    return 1.0 if num_qudits == executable_spaces else 0.5


def create_physical_model(args: argparse.Namespace) -> object:
    if args.trap_type == 'grid':
        return create_grid_physical_machine(
            num_cols=args.grid_cols,
            num_rows=args.grid_rows,
            trap_capacity=args.trap_capacity,
        )

    return create_testing_physical_machine(
        type=args.trap_type,
        trap_capacity=args.trap_capacity,
        num_traps=args.num_traps,
    )


def parse_assignment(text: str) -> dict[int, int]:
    return {int(k): int(v) for k, v in ast.literal_eval(text).items()}


def normalize_move(inst: list[Any]) -> tuple[str, tuple[int, int], dict[int, int]]:
    move_text = inst[0].strip()
    payload = move_text[len('Move'):].strip()
    payload = re.sub(r'np\.int\d+\(\s*(-?\d+)\s*\)', r'\1', payload)
    left, right = ast.literal_eval(payload)
    return ('Move', (int(left), int(right)), parse_assignment(inst[1]))


def coupling_edge_set(machine_model: Any) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()

    if hasattr(machine_model, 'coupling_graph'):
        graph = machine_model.coupling_graph
        raw_edges = graph.edge_labels.keys() if hasattr(graph, 'edge_labels') else graph
        for edge in raw_edges:
            u = int(edge[0])
            v = int(edge[1])
            edges.add((u, v))
            edges.add((v, u))
        return edges

    if hasattr(machine_model, 'get_move_neighbors'):
        for position in range(int(getattr(machine_model, 'num_positions', 0))):
            for neighbor in machine_model.get_move_neighbors(position):
                u = int(position)
                v = int(neighbor)
                edges.add((u, v))
                edges.add((v, u))
        return edges

    graph = getattr(machine_model, 'position_graph', None)
    if graph is not None and hasattr(graph, 'edge_labels'):
        for edge in graph.edge_labels.keys():
            u = int(edge[0])
            v = int(edge[1])
            edges.add((u, v))
            edges.add((v, u))

    return edges


def logical_at_position(assignment: dict[int, int], position: int) -> int | None:
    for logical, physical in assignment.items():
        if int(physical) == int(position):
            return int(logical)
    return None


def safe_trap_id(machine_model: Any, position: int) -> Any:
    if not hasattr(machine_model, 'get_trap_id'):
        return None
    try:
        return machine_model.get_trap_id(int(position))
    except Exception:
        return None


def safe_move_neighbors(machine_model: Any, position: int) -> list[int]:
    if hasattr(machine_model, 'get_move_neighbors'):
        try:
            return [int(x) for x in machine_model.get_move_neighbors(int(position))]
        except Exception:
            return []
    graph = getattr(machine_model, 'position_graph', None)
    move_graph = getattr(graph, 'move_graph', None)
    if move_graph is not None and hasattr(move_graph, 'neighbors_undirected'):
        try:
            return [int(x) for x in move_graph.neighbors_undirected(int(position))]
        except Exception:
            return []
    return []


def safe_move_path(machine_model: Any, source: int, target: int) -> list[int] | None:
    if hasattr(machine_model, 'get_move_path'):
        try:
            return [int(x) for x in machine_model.get_move_path(int(source), int(target))]
        except Exception:
            return None
    graph = getattr(machine_model, 'position_graph', None)
    if graph is None or not hasattr(graph, 'get_shortest_path_tree'):
        return None
    try:
        tree = graph.get_shortest_path_tree(int(source))
        return [int(x) for x in tree[int(target)]]
    except Exception:
        return None


def format_instruction_window(
    instruction_list: list[Any],
    center_index: int,
    window: int,
) -> str:
    radius = max(0, int(window))
    start = max(0, center_index - radius)
    stop = min(len(instruction_list), center_index + radius + 1)
    lines = [f'instruction window [{start}, {stop}):']
    for idx in range(start, stop):
        marker = '>>' if idx == center_index else '  '
        lines.append(f'  {marker} {idx}: {instruction_list[idx]}')
    return '\n'.join(lines)


def format_invalid_move_diagnostics(
    *,
    move: tuple[int, int],
    inst_index: int,
    instruction_list: list[Any],
    machine_model: Any,
    current_assignment: dict[int, int],
    after_assignment: dict[int, int],
    window: int,
) -> str:
    u, v = int(move[0]), int(move[1])
    before_u = logical_at_position(current_assignment, u)
    before_v = logical_at_position(current_assignment, v)
    after_u = logical_at_position(after_assignment, u)
    after_v = logical_at_position(after_assignment, v)

    endpoint_report = {
        'u': {
            'position': u,
            'trap_id': safe_trap_id(machine_model, u),
            'logical_before': before_u,
            'logical_after': after_u,
            'move_neighbors': safe_move_neighbors(machine_model, u),
        },
        'v': {
            'position': v,
            'trap_id': safe_trap_id(machine_model, v),
            'logical_before': before_v,
            'logical_after': after_v,
            'move_neighbors': safe_move_neighbors(machine_model, v),
        },
        'positions_same_trap': safe_trap_id(machine_model, u) == safe_trap_id(machine_model, v),
        'suggested_move_path': safe_move_path(machine_model, u, v),
    }

    return (
        f'[PGS] Invalid emitted move {move}: not an edge in the machine model.\n'
        f'instruction_index: {inst_index}\n'
        f'raw_instruction: {instruction_list[inst_index]}\n'
        f'endpoint_report: {endpoint_report}\n'
        f'{format_instruction_window(instruction_list, inst_index, window)}'
    )


def validate_instruction_moves_against_machine(
    instruction_list: list[Any],
    machine_model: Any,
    *,
    initial_assignment: dict[int, int],
    window: int,
) -> None:
    edges = coupling_edge_set(machine_model)
    current_assignment = copy.deepcopy(initial_assignment)
    for inst_index, inst in enumerate(instruction_list):
        head = inst[0].strip()
        if head.startswith('Execute'):
            next_assignment = parse_assignment(inst[2] if len(inst) >= 3 else inst[1])
            current_assignment = next_assignment
            continue
        if not head.startswith('Move'):
            continue
        move, after_assignment = normalize_move(inst)[1:]
        if move in edges:
            current_assignment = after_assignment
            continue
        raise ValueError(
            format_invalid_move_diagnostics(
                move=move,
                inst_index=inst_index,
                instruction_list=instruction_list,
                machine_model=machine_model,
                current_assignment=current_assignment,
                after_assignment=after_assignment,
                window=window,
            ),
        )


def build_workflow(
    args: argparse.Namespace,
    machine_model: QCCDMachineModel,
    ion_assignment: dict[int, int],
    congestion: float,
    stem: str,
) -> list[BasePass]:
    gate_count_weight = 0.1
    force_bruteforce = args.routing_mode == 'bruteforce'

    workflow: list[tuple[str, BasePass]] = [
        ('initial_unfold', UnfoldPass()),
        ('set_model', SetModelPass(machine_model)),
        ('set_initial_assignment', UpdateDataPass(key='ion_assignment_qccd', val=ion_assignment)),
        ('quick_partitioner', QuickPartitioner(args.block_size)),
    ]

    workflow.extend([
        ('pre_layout_apply_placement', ApplyPlacement()),
        ('qccd_layout_pgs', QCCDLayoutPassPGS(
            total_passes=args.num_layout_passes,
            cogestion_rate=congestion,
            force_bruteforce=force_bruteforce,
            profile_dir=args.profile_dir,
            profile_stem=f'{stem}__layout',
            profile_sort=args.profile_sort,
            trace_memory=args.trace_memory,
            trace_memory_depth=args.trace_memory_depth,
            trace_memory_top=args.trace_memory_top,
        )),
        ('qccd_routing_pgs', QCCDRoutingPassPGS(
            gate_count_weight,
            cogestion_rate=congestion,
            force_bruteforce=force_bruteforce,
            profile_dir=args.profile_dir,
            profile_stem=f'{stem}__routing',
            profile_sort=args.profile_sort,
            append_barriers=args.with_barriers,
            trace_memory=args.trace_memory,
            trace_memory_depth=args.trace_memory_depth,
            trace_memory_top=args.trace_memory_top,
        )),
        ('post_routing_apply_placement', ApplyPlacement()),
        ('final_unfold', UnfoldPass()),
    ])

    if args.pass_timings:
        return [TimedPass(label, pass_obj) for label, pass_obj in workflow]
    return [pass_obj for _, pass_obj in workflow]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run QCCD SHAW mapping with the PGS backend.',
    )
    parser.add_argument('input_filename', help='Benchmark stem, .qasm filename, or QASM path.')
    parser.add_argument(
        '--algorithm',
        choices=['shaw'],
        default='shaw',
        help='Mapping workflow to run. Default: shaw.',
    )
    parser.add_argument(
        '--trap-type',
        default='grid',
        help='Physical architecture: grid, H, H2, Helios, Enchilada, one_trap, linear, or G2x3.',
    )
    parser.add_argument('--trap-capacity', type=int, default=3)
    parser.add_argument('--num-layout-passes', type=int, default=2)
    parser.add_argument('--gate-type', default='FM')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--run-index', default='0')
    parser.add_argument('--grid-cols', type=int, default=1)
    parser.add_argument('--grid-rows', type=int, default=1)
    parser.add_argument('--num-traps', type=int, default=None)
    parser.add_argument('--block-size', type=int, default=3)
    parser.add_argument(
        '--routing-mode',
        choices=['heuristic', 'bruteforce'],
        default='bruteforce',
        help='Use heuristic move selection or force brute-force fallback.',
    )
    parser.add_argument(
        '--congestion-rate-override',
        type=float,
        default=None,
        help='Override the computed congestion rate used by layout/routing.',
    )
    parser.add_argument('--benchmark-dir', type=Path, default=None)
    parser.add_argument('--print-events', action='store_true')
    parser.add_argument('--save-pkl', action='store_true')
    parser.add_argument('--save-qasm', action='store_true')
    parser.add_argument(
        '--save-results',
        action='store_true',
        help='Save both the compiled QASM and pickle result.',
    )
    parser.add_argument(
        '--result-dir',
        type=Path,
        default=Path('outputs/qccd'),
        help='Directory for optional output files.',
    )
    parser.add_argument(
        '--summary-only',
        action='store_true',
        help='Suppress verbose internal output and print only the final summary.',
    )
    parser.add_argument(
        '--window',
        type=int,
        default=6,
        help='Number of surrounding instructions to show when move validation fails.',
    )
    parser.add_argument(
        '--profile-dir',
        type=Path,
        default=None,
        help='Optional directory for cProfile output from layout/routing.',
    )
    parser.add_argument('--profile-sort', default='cumulative')
    parser.add_argument('--pass-timings', action='store_true')
    parser.add_argument('--with-barriers', action='store_true')
    parser.add_argument(
        '--with-scheduler',
        nargs='?',
        const=True,
        default=True,
        type=parse_bool,
        help='Run post-compile instruction validation and scheduling. Default: true.',
    )
    parser.add_argument(
        '--scheduler-execute-location',
        choices=['physical', 'logical'],
        default='physical',
    )
    parser.add_argument('--trace-memory', action='store_true')
    parser.add_argument('--trace-memory-depth', type=int, default=5)
    parser.add_argument('--trace-memory-top', type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.save_results:
        args.save_qasm = True
        args.save_pkl = True

    machine_env_value = '0' if args.summary_only else None
    verbose_env_value = '0' if args.summary_only else None
    machine_context = TemporaryEnv('BQSKIT_QCCD_PRINT_MACHINE', machine_env_value)
    verbose_context = TemporaryEnv('BQSKIT_QCCD_VERBOSE', verbose_env_value)

    if not args.summary_only:
        print(f'Input filename: {args.input_filename}')
        print(f'Algorithm: {args.algorithm.upper()}')
        print(f'Trap type: {args.trap_type}')
        print(f'Trap capacity: {args.trap_capacity}')
        if args.trap_type == 'grid':
            print(f'Grid: {args.grid_cols}x{args.grid_rows}')
        print(f'Num layout passes: {args.num_layout_passes}')
        print('Mapper backend: PGS')
        print(f'Schedule backend: {"new" if args.with_scheduler else "disabled"}')
        print(f'Routing mode: {args.routing_mode}')
        print(f'Seed: {args.seed}')

    machine_output_sink = io.StringIO()
    machine_stdout_context = redirect_stdout(machine_output_sink) if args.summary_only else nullcontext()
    with machine_context, machine_stdout_context:
        physical_model = create_physical_model(args)

    gate_set = GateSet({U3Gate(), CXGate()})
    machine_model = QCCDMachineModel(
        gate_set=gate_set,
        physical_graph=physical_model,
        multi_qudit_gate_type=args.gate_type,
        timing_data=TIMING_DATA,
    )

    circuit_path = resolve_benchmark_circuit_path(args.input_filename, args)
    circuit = Circuit.from_file(str(circuit_path))
    ion_assignment = build_assignment(machine_model, circuit.num_qudits, args.seed)
    congestion = congestion_rate(machine_model, circuit.num_qudits)
    if args.congestion_rate_override is not None:
        congestion = float(args.congestion_rate_override)

    architecture_stem = args.trap_type
    if args.trap_type == 'grid':
        architecture_stem = f'grid_{args.grid_cols}x{args.grid_rows}'
    stem = (
        f'{args.algorithm.upper()}_{Path(args.input_filename).stem}_idx{args.run_index}_'
        f'{architecture_stem}_trap_{args.trap_capacity}_passes_{args.num_layout_passes}'
    )
    workflow = build_workflow(args, machine_model, ion_assignment, congestion, stem)

    output_sink = io.StringIO()
    suppress_context = redirect_stdout(output_sink) if args.summary_only else nullcontext()

    memory_current_mb = None
    memory_peak_mb = None
    memory_top_stats: list[str] = []
    with verbose_context:
        with Compiler() as compiler:
            if args.trace_memory:
                tracemalloc.start(max(1, int(args.trace_memory_depth)))
            start = timer()
            with suppress_context:
                output_circuit, data = compiler.compile(circuit, workflow, request_data=True)
            compile_time = timer() - start
            if args.trace_memory:
                current, peak = tracemalloc.get_traced_memory()
                memory_current_mb = current / 1024 / 1024
                memory_peak_mb = peak / 1024 / 1024
                snapshot = tracemalloc.take_snapshot()
                memory_top_stats = [
                    str(stat)
                    for stat in snapshot.statistics('lineno')[
                        :max(0, int(args.trace_memory_top))
                    ]
                ]
                tracemalloc.stop()

        schedule_result = None
        if args.with_scheduler:
            schedule_context = redirect_stdout(output_sink) if args.summary_only else nullcontext()
            with schedule_context:
                validate_instruction_moves_against_machine(
                    data['instruction_list'],
                    data.model,
                    initial_assignment=data['initial_ion_assignment_qccd'],
                    window=args.window,
                )
                schedule_result = schedule_qccd_from_instructions_v3(
                    instruction_lst=data['instruction_list'],
                    initial_ion_assignment=data['initial_ion_assignment_qccd'],
                    full_initial_ion_assignment=data.get('initial_full_ion_assignment_qccd_pgs'),
                    machine_model=data.model,
                    circuit=output_circuit,
                    parallel=True,
                    execute_location_mode=args.scheduler_execute_location,
                )

    runtime_us = None
    fidelity = None
    execute_rounds = None
    move_rounds = None
    if schedule_result is not None:
        runtime_us = float(schedule_result['runtime']) / 1e-6
        fidelity = schedule_result['application_fidelity']
        execute_rounds = len(schedule_result['execute_rounds'])
        move_rounds = len(schedule_result['move_rounds'])

    instruction_count = len(data['instruction_list'])

    print(f'Input filename: {args.input_filename}')
    print(f'Circuit path: {circuit_path}')
    print(f'Algorithm: {args.algorithm.upper()}')
    print(f'Trap type: {args.trap_type}')
    print(f'Trap capacity: {args.trap_capacity}')
    if args.trap_type == 'grid':
        print(f'Grid: {args.grid_cols}x{args.grid_rows}')
    print(f'Num layout passes: {args.num_layout_passes}')
    print(f'Routing mode: {args.routing_mode}')
    print(f'Seed: {args.seed}')
    print(f'Original operation count: {circuit.num_operations}')
    print(f'Compiled operation count: {output_circuit.num_operations}')
    print(f"Initial ion assignment: {data.get('initial_ion_assignment_qccd', ion_assignment)}")
    print(f"Final ion assignment: {data.get('ion_assignment_qccd')}")
    print(f"Initial mapping: {data.get('initial_mapping')}")
    print(f"Final mapping: {data.get('final_mapping')}")
    print(f'Compile time (s): {compile_time}')
    print(f'Instruction count: {instruction_count}')
    print(f'Runtime (us): {runtime_us}')
    print(f'Application fidelity: {fidelity}')
    print(f'Execute rounds: {execute_rounds}')
    print(f'Move rounds: {move_rounds}')

    if args.pass_timings:
        print('Pass timings:')
        total_pass_time = 0.0
        for entry in data.get('qccd_pass_timings', []):
            total_pass_time += float(entry['seconds'])
            print(
                f"  {entry['label']:<28} "
                f"{float(entry['seconds']):10.6f}s "
                f"ops={entry['operations']} "
                f"qudits={entry['qudits']}",
            )
        print(f'  {"total_profiled_pass_time":<28} {total_pass_time:10.6f}s')

    if args.profile_dir is not None:
        print(f'Profile directory: {args.profile_dir}')
    if args.trace_memory and memory_current_mb is not None and memory_peak_mb is not None:
        print(f'Tracemalloc current MB: {memory_current_mb:.3f}')
        print(f'Tracemalloc peak MB: {memory_peak_mb:.3f}')
        print('Tracemalloc top allocations:')
        for stat in memory_top_stats:
            print(f'  {stat}')

    print('Summary:')
    print_sweep_style_summary(
        label=args.algorithm.upper(),
        compile_time_s=compile_time,
        runtime_us=runtime_us,
        fidelity=fidelity,
        instructions=instruction_count,
        execute_rounds=execute_rounds,
        move_rounds=move_rounds,
    )

    if args.print_events and schedule_result is not None:
        print_event_trace(schedule_result)

    result_dir = args.result_dir
    saved_paths: list[Path] = []
    if args.save_qasm:
        result_dir.mkdir(parents=True, exist_ok=True)
        qasm_path = result_dir / f'{stem}.qasm'
        output_circuit.save(str(qasm_path))
        saved_paths.append(qasm_path)

    if args.save_pkl:
        result_dir.mkdir(parents=True, exist_ok=True)
        pkl_path = result_dir / f'{stem}.pkl'
        result = [
            None if schedule_result is None else schedule_result['runtime'],
            compile_time,
            data['instruction_list'],
            output_circuit.gate_counts,
            data['initial_ion_assignment_qccd'],
            data['initial_mapping'],
            data['final_mapping'],
            data.model,
        ]
        with pkl_path.open('wb') as handle:
            pickle.dump(result, handle, pickle.HIGHEST_PROTOCOL)
        saved_paths.append(pkl_path)

    for path in saved_paths:
        print(f'Saved: {path}')


if __name__ == '__main__':
    main()
