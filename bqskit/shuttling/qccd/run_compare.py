from __future__ import annotations

import argparse
import copy
import logging
import random
import statistics
from dataclasses import dataclass
from timeit import default_timer as timer

from bqskit.compiler import Compiler
from bqskit.compiler.gateset import GateSet
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CCXGate
from bqskit.ir.gates import CNOTGate
from bqskit.ir.gates import CXGate
from bqskit.ir.gates import HGate
from bqskit.ir.gates.parameterized import U3Gate
from bqskit.passes import (
    ApplyPlacement,
    EmbedAllPermutationsPass,
    ForEachBlockPass,
    QuickPartitioner,
    QSearchSynthesisPass,
    SetModelPass,
    UnfoldPass,
    UpdateDataPass,
)

from bqskit.shuttling.qccd import QCCDSubtopologySelectionPass
from bqskit.shuttling.qccd.QCCD_machine import QCCDMachineModel as QCCDMachineModelCG
from bqskit.shuttling.qccd.QCCD_machine_PGS import QCCDMachineModel as QCCDMachineModelPGS
from bqskit.shuttling.qccd.QCCD_schedule import schedule_QCCD
from bqskit.shuttling.qccd.QCCD_schedule_PGS import schedule_QCCD_PGS
from bqskit.shuttling.qccd.QCCD_util import create_testing_physical_machine
from bqskit.shuttling.qccd.mapping import (
    QCCDPAMLayoutPass,
    QCCDPAMLayoutPassPGS,
    QCCDPAMRoutingPass,
    QCCDPAMRoutingPassPGS,
    QCCDLayoutPass,
    QCCDLayoutPassPGS,
    QCCDRoutingPass,
    QCCDRoutingPassPGS,
)

_logger = logging.getLogger(__name__)

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


@dataclass
class CompareResult:
    architecture: str
    algorithm: str
    backend: str
    compile_time_s: float
    schedule_time_s: float
    instruction_count: int
    final_mapping_head: list[int]
    final_assignment_head: list[tuple[int, int]]


def build_machine_model(
    architecture: str,
    trap_capacity: int,
    gate_type: str,
    backend: str,
):
    physical_model = create_testing_physical_machine(
        type=architecture,
        trap_capacity=trap_capacity,
    )
    gate_set = GateSet({U3Gate(), CXGate()})
    machine_cls = QCCDMachineModelPGS if backend == 'PGS' else QCCDMachineModelCG
    return machine_cls(
        gate_set=gate_set,
        physical_graph=physical_model,
        multi_qudit_gate_type=gate_type,
        timing_data=TIMING_DATA,
    )


def build_assignment(machine_model, num_qudits: int, seed: int) -> dict[int, int]:
    rng = random.Random(seed)
    available = []
    for trap in machine_model.physical_graph.trap_list:
        available += list(machine_model.physical_to_position[trap.id])
    chosen = rng.sample(available, num_qudits)
    return {i: int(chosen[i]) for i in range(num_qudits)}


def build_test_circuit(num_qudits: int, kind: str) -> Circuit:
    circ = Circuit(num_qudits)
    if kind == 'ghz':
        for q in range(num_qudits):
            circ.append_gate(HGate(), [q])
        for q in range(1, num_qudits):
            circ.append_gate(CNOTGate(), [0, q])
        return circ

    if kind == 'ladder':
        for q in range(num_qudits):
            circ.append_gate(HGate(), [q])
        for q in range(num_qudits - 1):
            circ.append_gate(CNOTGate(), [q, q + 1])
        for q in range(num_qudits - 2):
            circ.append_gate(CNOTGate(), [q, q + 2])
        return circ

    if kind == 'shaper':
        for q in range(min(num_qudits, 6)):
            circ.append_gate(HGate(), [q])
        for q in range(0, num_qudits - 2, 3):
            circ.append_gate(CCXGate(), [q, q + 1, q + 2])
        for q in range(0, num_qudits - 1, 2):
            circ.append_gate(CNOTGate(), [q, q + 1])
        return circ

    raise ValueError(f'Unknown circuit kind: {kind}')


def congestion_rate(machine_model, num_qudits: int) -> float:
    executable_spaces = 0
    for trap in machine_model.physical_graph.executable_trap_list:
        executable_spaces += trap.max_num_ions
    return 0.5 if num_qudits == executable_spaces else 1.0


def build_workflow(machine_model, algorithm: str, backend: str, assignment, num_layout_passes: int):
    qsearch_pass = QSearchSynthesisPass()
    block_size = 2 if algorithm == 'SHAW' else 3
    cong = congestion_rate(machine_model, len(assignment))
    gate_count_weight = 0.1

    if algorithm == 'SHAW':
        layout = QCCDLayoutPassPGS(total_passes=num_layout_passes, cogestion_rate=cong) if backend == 'PGS' \
            else QCCDLayoutPass(total_passes=num_layout_passes, cogestion_rate=cong)
        routing = QCCDRoutingPassPGS(gate_count_weight, cogestion_rate=cong) if backend == 'PGS' \
            else QCCDRoutingPass(gate_count_weight, cogestion_rate=cong)
        workflow = [
            UnfoldPass(),
            SetModelPass(machine_model),
            UpdateDataPass(key='ion_assignment_qccd', val=assignment),
            layout,
            routing,
        ]
        if backend != 'PGS':
            workflow.append(ApplyPlacement())
        workflow.append(UnfoldPass())
        return workflow

    if algorithm == 'SHAPER':
        layout = QCCDPAMLayoutPassPGS(total_passes=num_layout_passes, cogestion_segment_rate=cong) if backend == 'PGS' \
            else QCCDPAMLayoutPass(total_passes=num_layout_passes, cogestion_segment_rate=cong)
        routing = QCCDPAMRoutingPassPGS(gate_count_weight, cogestion_segment_rate=cong) if backend == 'PGS' \
            else QCCDPAMRoutingPass(gate_count_weight, cogestion_segment_rate=cong)
        workflow = [
            UnfoldPass(),
            SetModelPass(machine_model),
            UpdateDataPass(key='ion_assignment_qccd', val=assignment),
            QCCDSubtopologySelectionPass(block_size),
            QuickPartitioner(block_size),
            ForEachBlockPass(
                EmbedAllPermutationsPass(
                    inner_synthesis=qsearch_pass,
                    input_perm=True,
                    vary_topology=True,
                ),
            ),
        ]
        if backend != 'PGS':
            workflow.append(ApplyPlacement())
        workflow.extend([layout, routing])
        if backend != 'PGS':
            workflow.append(ApplyPlacement())
        workflow.append(UnfoldPass())
        return workflow

    raise ValueError(f'Unknown algorithm: {algorithm}')


def run_case(
    architecture: str,
    trap_capacity: int,
    gate_type: str,
    algorithm: str,
    backend: str,
    circuit_kind: str,
    num_qudits: int,
    num_layout_passes: int,
    seed: int,
) -> CompareResult:
    machine_model = build_machine_model(architecture, trap_capacity, gate_type, backend)
    circuit = build_test_circuit(num_qudits, circuit_kind)
    assignment = build_assignment(machine_model, circuit.num_qudits, seed)
    workflow = build_workflow(
        machine_model,
        algorithm,
        backend,
        copy.deepcopy(assignment),
        num_layout_passes,
    )

    with Compiler() as compiler:
        start = timer()
        output_circuit, data = compiler.compile(circuit, workflow, request_data=True)
        compile_time = timer() - start

    schedule_start = timer()
    scheduler = schedule_QCCD_PGS if backend == 'PGS' else schedule_QCCD
    schedule_runtime = scheduler(
        data['instruction_list'],
        output_circuit,
        data.initial_mapping,
        data['initial_ion_assignment_qccd'],
        data.model,
        parallel=True,
    )
    schedule_time = timer() - schedule_start
    if isinstance(schedule_runtime, tuple):
        schedule_runtime = schedule_runtime[0]

    final_assignment = data.get('ion_assignment_qccd', {})
    return CompareResult(
        architecture=architecture,
        algorithm=algorithm,
        backend=backend,
        compile_time_s=compile_time,
        schedule_time_s=float(schedule_runtime),
        instruction_count=len(data.get('instruction_list', [])),
        final_mapping_head=list(data.get('final_mapping', []))[:12],
        final_assignment_head=sorted(
            (int(k), int(v)) for k, v in final_assignment.items()
        )[:12],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Compare legacy and PGS QCCD workflows.')
    parser.add_argument('--architectures', nargs='+', default=['H', 'G2x3'])
    parser.add_argument('--algorithms', nargs='+', default=['SHAW', 'SHAPER'])
    parser.add_argument('--backends', nargs='+', default=['CG', 'PGS'])
    parser.add_argument('--circuit-kind', default='shaper', choices=['ghz', 'ladder', 'shaper'])
    parser.add_argument('--trap-capacity', type=int, default=4)
    parser.add_argument('--gate-type', default='FM')
    parser.add_argument('--num-qudits', type=int, default=6)
    parser.add_argument('--num-layout-passes', type=int, default=2)
    parser.add_argument('--seed', type=int, default=1234)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    args = parse_args()

    results: list[CompareResult] = []
    for architecture in args.architectures:
        for algorithm in args.algorithms:
            for backend in args.backends:
                result = run_case(
                    architecture=architecture,
                    trap_capacity=args.trap_capacity,
                    gate_type=args.gate_type,
                    algorithm=algorithm,
                    backend=backend,
                    circuit_kind=args.circuit_kind,
                    num_qudits=args.num_qudits,
                    num_layout_passes=args.num_layout_passes,
                    seed=args.seed,
                )
                results.append(result)
                print(f'{architecture} {algorithm} {backend}')
                print(f'  compile_time_s: {result.compile_time_s:.6f}')
                print(f'  schedule_time_s: {result.schedule_time_s:.6f}')
                print(f'  instruction_count: {result.instruction_count}')
                print(f'  final_mapping_head: {result.final_mapping_head}')
                print(f'  final_assignment_head: {result.final_assignment_head}')

    print('Summary')
    for architecture in args.architectures:
        for algorithm in args.algorithms:
            subset = [
                r for r in results
                if r.architecture == architecture and r.algorithm == algorithm
            ]
            if len(subset) < 2:
                continue
            subset.sort(key=lambda r: r.backend)
            compile_times = {r.backend: r.compile_time_s for r in subset}
            if 'CG' in compile_times and 'PGS' in compile_times:
                speedup = compile_times['CG'] / compile_times['PGS']
                print(f'  {architecture} {algorithm} pgs_speedup_x: {speedup:.3f}')
                mapping_match = (
                    next(r for r in subset if r.backend == 'CG').final_mapping_head
                    == next(r for r in subset if r.backend == 'PGS').final_mapping_head
                )
                print(f'  {architecture} {algorithm} same_mapping_head: {mapping_match}')


if __name__ == '__main__':
    main()
