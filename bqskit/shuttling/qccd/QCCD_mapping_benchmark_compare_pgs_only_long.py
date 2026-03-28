from __future__ import annotations

import copy
import io
import logging
import statistics
from collections import deque
from contextlib import redirect_stdout
from time import perf_counter

from bqskit.ir import Operation
from bqskit.ir.gates import CCXGate
from bqskit.ir.gates import CXGate

from bqskit.shuttling.qccd.QCCD_machine_PGS import QCCDMachineModel
from bqskit.shuttling.qccd.QCCD_mapping_PGS import (
    QCCDMappingAlgorithm as QCCDMappingAlgorithmPGS,
)
from bqskit.shuttling.qccd.QCCD_mapping_PGS_simple import (
    QCCDMappingAlgorithm as QCCDMappingAlgorithmPGSSimple,
)
from bqskit.shuttling.qccd.QCCD_util import create_testing_physical_machine


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

# This benchmark is intentionally much heavier than the quick/large variants.
# The defaults are tuned to be plausibly "many minutes" on PGS_simple while
# still being straightforward to scale locally if needed.
MACHINE_TYPE = 'linear'
TRAP_CAPACITY = 6
NUM_TRAPS = 32
NUM_IONS = 96
NUM_ROUNDS = 18
NUM_REPEATS = 1


def build_machine_model() -> QCCDMachineModel:
    physical_model = create_testing_physical_machine(
        type=MACHINE_TYPE,
        trap_capacity=TRAP_CAPACITY,
        num_traps=NUM_TRAPS,
    )
    return QCCDMachineModel(
        physical_graph=physical_model,
        multi_qudit_gate_type='FM',
        timing_data=TIMING_DATA,
    )


def build_base_assignment(machine_model: QCCDMachineModel) -> dict[int, int]:
    trap_positions = sorted(
        pos for pos, kind in machine_model.position_to_physical.items()
        if kind == 'trap'
    )
    if NUM_IONS > len(trap_positions):
        raise ValueError(
            f'NUM_IONS={NUM_IONS} exceeds available trap positions '
            f'{len(trap_positions)} for this machine.',
        )

    spread_positions: list[int] = []
    dq: deque[int] = deque(trap_positions)
    take_from_left = True
    while dq and len(spread_positions) < NUM_IONS:
        spread_positions.append(dq.popleft() if take_from_left else dq.pop())
        take_from_left = not take_from_left

    return {logical: int(position) for logical, position in enumerate(spread_positions)}


def build_test_gates(num_ions: int) -> list[Operation]:
    if num_ions < 12:
        raise ValueError('Need at least 12 ions for the long benchmark.')

    gates: list[Operation] = []
    stride = max(3, num_ions // 12)
    third = num_ions // 3
    two_third = (2 * num_ions) // 3

    for round_idx in range(NUM_ROUNDS):
        base_shift = (round_idx * 5) % num_ions
        for offset in range(0, num_ions, stride):
            a = (offset + base_shift) % num_ions
            b = (offset + third + 2 * round_idx) % num_ions
            c = (offset + two_third + 3 * round_idx) % num_ions

            if len({a, b, c}) == 3:
                gates.append(Operation(CCXGate(), (a, b, c)))

            d = (offset + num_ions // 2 + round_idx) % num_ions
            if a != d:
                gates.append(Operation(CXGate(), (a, d)))

    return gates


def benchmark(mapping_cls: type, name: str) -> dict[str, object]:
    machine_model = build_machine_model()
    base_assignment = build_base_assignment(machine_model)
    test_gates = build_test_gates(len(base_assignment))
    mapping_algo = mapping_cls(
        qccd_machine=machine_model,
        cogestion_rate=1.0,
        decay_delta=0.0,
        extended_set_size=5,
        extended_set_weight=0.5,
    )
    D = machine_model.all_pair_travelling_time()
    pi = list(range(machine_model.num_qudits))

    times: list[float] = []
    total_moves = 0
    final_changed: dict[int, int] = {}
    final_gate_summaries: list[tuple[str, int]] = []

    for _ in range(NUM_REPEATS):
        ion_assignment = copy.deepcopy(base_assignment)
        start = copy.deepcopy(ion_assignment)
        run_gate_summaries: list[tuple[str, int]] = []
        pgs = machine_model.build_pgs_from_assignment(ion_assignment) if name == 'PGS' else None
        t0 = perf_counter()
        total_moves = 0
        for gate in test_gates:
            if name == 'PGS':
                moves = mapping_algo._brute_force_congestion(
                    gate=gate,
                    D=D,
                    pgs=pgs,
                )
                ion_assignment = mapping_algo._assignment_from_pgs(pgs)
            else:
                moves = mapping_algo._brute_force_congestion(
                    gate=gate,
                    D=D,
                    pi=pi,
                    ion_assignment=ion_assignment,
                )
            typed_moves = [(int(a), int(b)) for a, b in moves]
            total_moves += len(typed_moves)
            run_gate_summaries.append((str(gate), len(typed_moves)))
        times.append(perf_counter() - t0)
        final_changed = {
            int(k): int(ion_assignment[k])
            for k, v in start.items()
            if ion_assignment[k] != v
        }
        final_gate_summaries = run_gate_summaries

    return {
        'name': name,
        'graph_type': type(machine_model.position_graph).__name__,
        'times': times,
        'avg': statistics.mean(times),
        'total_moves': total_moves,
        'num_gates': len(test_gates),
        'changed': final_changed,
        'gate_summaries_head': final_gate_summaries[:20],
    }


def summarize(result: dict[str, object]) -> None:
    print(result['name'])
    print(f"  graph_type: {result['graph_type']}")
    print(f"  times_s: {[round(t, 6) for t in result['times']]}")
    print(f"  avg_s: {result['avg']:.6f}")
    print(f"  num_gates: {result['num_gates']}")
    print(f"  total_moves: {result['total_moves']}")
    print(f"  changed_count: {len(result['changed'])}")
    print(f"  changed_sample: {sorted(result['changed'].items())[:20]}")
    print(f"  per_gate_move_counts_head: {result['gate_summaries_head']}")


def main() -> None:
    logging.disable(logging.CRITICAL)
    buf = io.StringIO()
    with redirect_stdout(buf):
        simple_result = benchmark(QCCDMappingAlgorithmPGSSimple, 'PGS_simple')
    with redirect_stdout(buf):
        pgs_result = benchmark(QCCDMappingAlgorithmPGS, 'PGS')

    print('Configuration')
    print(f'  machine_type: {MACHINE_TYPE}')
    print(f'  trap_capacity: {TRAP_CAPACITY}')
    print(f'  num_traps: {NUM_TRAPS}')
    print(f'  num_ions: {NUM_IONS}')
    print(f'  num_rounds: {NUM_ROUNDS}')
    print(f'  repeats: {NUM_REPEATS}')
    summarize(simple_result)
    summarize(pgs_result)
    print('Comparison')
    print(f"  native_speedup_x_vs_simple: {simple_result['avg'] / pgs_result['avg']:.3f}")
    print(f"  same_total_moves: {simple_result['total_moves'] == pgs_result['total_moves']}")
    print(f"  same_changed_sample: {sorted(simple_result['changed'].items())[:20] == sorted(pgs_result['changed'].items())[:20]}")


if __name__ == '__main__':
    main()
