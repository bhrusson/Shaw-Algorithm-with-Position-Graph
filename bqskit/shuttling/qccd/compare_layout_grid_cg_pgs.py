from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
from typing import Any

from bqskit.compiler.gateset import GateSet
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CXGate
from bqskit.ir.gates.parameterized import U3Gate

from bqskit.shuttling.qccd import create_grid_physical_machine
from bqskit.shuttling.qccd.compare_shaw_grid_cg_pgs import TIMING_DATA
from bqskit.shuttling.qccd.compare_shaw_grid_cg_pgs import compile_case
from bqskit.shuttling.qccd.compare_shaw_grid_cg_pgs import print_position_graph_details
from bqskit.shuttling.qccd.compare_shaw_grid_cg_pgs import result_label
from bqskit.shuttling.qccd.QCCD_machine import QCCDMachineModel as CGMachineModel
from bqskit.shuttling.qccd.QCCD_machine_PGS import (
    QCCDMachineModel as PGSMachineModel,
)
from bqskit.shuttling.qccd.run_grid_pgs_shaw import build_assignment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Compare CG vs PGS final layouts on a grid architecture.',
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
        help='Used only to match the layout pass configuration from the full compare flow.',
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
    parser.add_argument(
        '--print-position-graph',
        action='store_true',
    )
    parser.add_argument(
        '--pgs-move-path-modes',
        nargs='+',
        choices=['hops', 'weighted'],
        default=['hops'],
    )
    parser.add_argument('--verbose-status', action='store_true')
    return parser.parse_args()


def build_base_machine(machine_cls: type[Any], args: argparse.Namespace) -> Any:
    return machine_cls(
        gate_set=GateSet({U3Gate(), CXGate()}),
        physical_graph=create_grid_physical_machine(
            num_cols=args.grid_cols,
            num_rows=args.grid_rows,
            trap_capacity=args.trap_capacity,
        ),
        multi_qudit_gate_type=args.gate_type,
        timing_data=TIMING_DATA,
    )


def trap_layout_view(
    machine_model: Any,
    assignment: dict[int, int],
) -> list[str]:
    position_to_logical = {int(pos): int(logical) for logical, pos in assignment.items()}
    lines: list[str] = []
    for trap in machine_model.physical_graph.trap_list:
        positions = list(machine_model.physical_to_position[trap.id])
        occupants = [position_to_logical.get(int(pos), None) for pos in positions]
        formatted = ', '.join(
            f'{int(pos)}:{("-" if logical is None else logical)}'
            for pos, logical in zip(positions, occupants)
        )
        lines.append(f'  {trap.id}: [{formatted}]')
    return lines


def print_layout_result(
    result: dict[str, Any],
    machine_model: Any,
    *,
    multi_pgs: bool,
) -> None:
    label = result_label(result, multi_pgs=multi_pgs)
    assignment = copy.deepcopy(result['data']['ion_assignment_qccd'])
    print(f'{label} layout:')
    print(f'  compile_time_s={result["compile_time_s"]:.6f}')
    print(f'  assignment={assignment}')
    print('  trap occupancy:')
    for line in trap_layout_view(machine_model, assignment):
        print(line)


def main() -> None:
    args = parse_args()
    pgs_move_path_modes = list(dict.fromkeys(args.pgs_move_path_modes))
    os.environ['BQSKIT_QCCD_VERBOSE'] = '1' if args.verbose_status else '0'

    circuit_path = (
        Path('bqskit/shuttling/qccd/benchmark_circuits')
        / f'{args.input_filename}.qasm'
    )
    circuit = Circuit.from_file(str(circuit_path))

    base_model = build_base_machine(CGMachineModel, args)
    assignment = build_assignment(base_model, circuit.num_qudits, args.seed)

    print(f'Initial ion assignment: {assignment}')
    print(f'Grid: {args.grid_cols}x{args.grid_rows}')
    print(f'Trap capacity: {args.trap_capacity}')
    print(f'Layout passes: {args.num_layout_passes}')
    print(f'Routing mode hint: {args.routing_mode}')
    if len(pgs_move_path_modes) > 1:
        print(f'PGS move-path modes: {pgs_move_path_modes}')

    cg_result = compile_case(
        'CG',
        circuit,
        assignment,
        args.trap_capacity,
        args.num_layout_passes,
        args.gate_type,
        args.grid_cols,
        args.grid_rows,
        args.routing_mode,
        'layout',
        congestion_override=args.cg_congestion_rate_override,
        print_position_graph=args.print_position_graph,
    )
    cg_machine = build_base_machine(CGMachineModel, args)

    pgs_results: list[tuple[dict[str, Any], Any]] = []
    for mode in pgs_move_path_modes:
        result = compile_case(
            'PGS',
            circuit,
            assignment,
            args.trap_capacity,
            args.num_layout_passes,
            args.gate_type,
            args.grid_cols,
            args.grid_rows,
            args.routing_mode,
            'layout',
            congestion_override=args.pgs_congestion_rate_override,
            pgs_move_path_mode=mode,
            print_position_graph=args.print_position_graph,
        )
        pgs_results.append((result, build_base_machine(PGSMachineModel, args)))

    if args.print_position_graph:
        print_position_graph_details('CG final-view machine', cg_machine)
        for result, machine in pgs_results:
            print_position_graph_details(
                f'{result_label(result, multi_pgs=len(pgs_results) > 1)} final-view machine',
                machine,
            )

    print('\nFinal layouts:')
    print_layout_result(cg_result, cg_machine, multi_pgs=len(pgs_results) > 1)
    for result, machine in pgs_results:
        print()
        print_layout_result(result, machine, multi_pgs=len(pgs_results) > 1)
        print(
            '  same assignment as CG='
            f"{result['data']['ion_assignment_qccd'] == cg_result['data']['ion_assignment_qccd']}",
        )


if __name__ == '__main__':
    main()
