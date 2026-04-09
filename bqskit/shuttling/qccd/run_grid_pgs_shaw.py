from __future__ import annotations

import argparse
import io
import os
import pickle
import random
from contextlib import nullcontext
from contextlib import redirect_stdout
from pathlib import Path
from timeit import default_timer as timer

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

from bqskit.shuttling.QCCD_schedule_new import print_event_trace
from bqskit.shuttling.QCCD_schedule_new import schedule_qccd_from_instructions_v3
from bqskit.shuttling.qccd import create_grid_physical_machine
from bqskit.shuttling.qccd.QCCD_machine_PGS import QCCDMachineModel
from bqskit.shuttling.qccd.pgs_passes import QCCDLayoutPassPGS
from bqskit.shuttling.qccd.pgs_passes import QCCDRoutingPassPGS


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


def build_assignment(machine_model: QCCDMachineModel, num_qudits: int, seed: int) -> dict[int, int]:
    rng = random.Random(seed)
    available = []
    for trap in machine_model.physical_graph.trap_list:
        available += list(machine_model.physical_to_position[trap.id])
    chosen = rng.sample(available, num_qudits)
    return {i: int(chosen[i]) for i in range(num_qudits)}


def congestion_rate(machine_model: QCCDMachineModel, num_qudits: int) -> float:
    executable_spaces = 0
    for trap in machine_model.physical_graph.executable_trap_list:
        executable_spaces += trap.max_num_ions
    return 1.0 if num_qudits == executable_spaces else 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run SHAW with the PGS backend on a grid architecture and interpret instruction_list.',
    )
    parser.add_argument('input_filename', help='Benchmark circuit filename without .qasm.')
    parser.add_argument('--trap-capacity', type=int, default=3)
    parser.add_argument('--num-layout-passes', type=int, default=2)
    parser.add_argument('--gate-type', default='FM')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--run-index', default='0')
    parser.add_argument('--grid-cols', type=int, default=1)
    parser.add_argument('--grid-rows', type=int, default=1)
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
    parser.add_argument('--print-events', action='store_true')
    parser.add_argument('--save-pkl', action='store_true')
    parser.add_argument('--save-qasm', action='store_true')
    parser.add_argument(
        '--summary-only',
        action='store_true',
        help='Suppress verbose internal output and print only the final summary.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.summary_only:
        print(f'Input filename: {args.input_filename}')
        print('Algorithm: SHAW')
        print('Trap type: grid')
        print(f'Trap capacity: {args.trap_capacity}')
        print(f'Grid: {args.grid_cols}x{args.grid_rows}')
        print(f'Num layout passes: {args.num_layout_passes}')
        print('Mapper backend: PGS')
        print('Schedule backend: new')
        print(f'Routing mode: {args.routing_mode}')
        print(f'Seed: {args.seed}')

    if args.summary_only:
        output_sink = io.StringIO()
        machine_context = redirect_stdout(output_sink)
    else:
        machine_context = nullcontext()

    with machine_context:
        physical_model = create_grid_physical_machine(
            num_cols=args.grid_cols,
            num_rows=args.grid_rows,
            trap_capacity=args.trap_capacity,
        )
    gate_set = GateSet({U3Gate(), CXGate()})
    machine_model = QCCDMachineModel(
        gate_set=gate_set,
        physical_graph=physical_model,
        multi_qudit_gate_type=args.gate_type,
        timing_data=TIMING_DATA,
    )

    circuit_path = Path('bqskit/shuttling/qccd/benchmark_circuits') / f'{args.input_filename}.qasm'
    circuit = Circuit.from_file(str(circuit_path))
    ion_assignment = build_assignment(machine_model, circuit.num_qudits, args.seed)
    congestion = congestion_rate(machine_model, circuit.num_qudits)
    if args.congestion_rate_override is not None:
        congestion = float(args.congestion_rate_override)
    gate_count_weight = 0.1
    force_bruteforce = args.routing_mode == 'bruteforce'

    workflow = [
        UnfoldPass(),
        SetModelPass(machine_model),
        UpdateDataPass(key='ion_assignment_qccd', val=ion_assignment),
        QuickPartitioner(3),
        ApplyPlacement(),
        QCCDLayoutPassPGS(
            total_passes=args.num_layout_passes,
            cogestion_rate=congestion,
            force_bruteforce=force_bruteforce,
        ),
        QCCDRoutingPassPGS(
            gate_count_weight,
            cogestion_rate=congestion,
            force_bruteforce=force_bruteforce,
        ),
        ApplyPlacement(),
        UnfoldPass(),
    ]

    if args.summary_only:
        output_sink = io.StringIO()
        compile_context = redirect_stdout(output_sink)
        schedule_context = redirect_stdout(output_sink)
        verbose_context = _TemporaryEnv('BQSKIT_QCCD_VERBOSE', '0')
    else:
        compile_context = nullcontext()
        schedule_context = nullcontext()
        verbose_context = nullcontext()

    with verbose_context:
        with Compiler() as compiler:
            start = timer()
            with compile_context:
                output_circuit, data = compiler.compile(circuit, workflow, request_data=True)
            compile_time = timer() - start

        with schedule_context:
            schedule_result = schedule_qccd_from_instructions_v3(
                instruction_lst=data['instruction_list'],
                initial_ion_assignment=data['initial_ion_assignment_qccd'],
                full_initial_ion_assignment=data.get('initial_full_ion_assignment_qccd_pgs'),
                machine_model=data.model,
                circuit=output_circuit,
                parallel=True,
                execute_location_mode='physical',
            )

    runtime_us = float(schedule_result['runtime']) / 1e-6
    fidelity = schedule_result['application_fidelity']
    instruction_count = len(data['instruction_list'])
    execute_rounds = len(schedule_result['execute_rounds'])
    move_rounds = len(schedule_result['move_rounds'])

    print(f'Input filename: {args.input_filename}')
    print(f'Trap capacity: {args.trap_capacity}')
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
    print(f'Runtime (us): {runtime_us}')
    print(f'Application fidelity: {fidelity}')
    print(f'Instruction count: {instruction_count}')
    print(f'Execute rounds: {execute_rounds}')
    print(f'Move rounds: {move_rounds}')
    print('Summary:')
    print_sweep_style_summary(
        label='PGS',
        compile_time_s=compile_time,
        runtime_us=runtime_us,
        fidelity=fidelity,
        instructions=instruction_count,
        execute_rounds=execute_rounds,
        move_rounds=move_rounds,
    )

    if args.print_events:
        print_event_trace(schedule_result)

    result_dir = Path('bqskit/shuttling/qccd/paper_result_grid')
    result_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        f'SHAW_{args.input_filename}_idx{args.run_index}_'
        f'grid_{args.trap_capacity}_{args.num_layout_passes}'
    )

    if args.save_qasm:
        output_circuit.save(str(result_dir / f'{stem}.qasm'))

    if args.save_pkl:
        result = [
            schedule_result['runtime'],
            compile_time,
            data['instruction_list'],
            output_circuit.gate_counts,
            data['initial_ion_assignment_qccd'],
            data['initial_mapping'],
            data['final_mapping'],
            data.model,
        ]
        with open(result_dir / f'{stem}.pkl', 'wb') as f:
            pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
