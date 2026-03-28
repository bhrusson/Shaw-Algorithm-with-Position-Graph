from __future__ import annotations

import io
import logging
import statistics
import copy
from contextlib import redirect_stdout
from time import perf_counter

from bqskit.ir import Operation
from bqskit.ir.gates import CCXGate, CXGate

from bqskit.shuttling.qccd.QCCD_machine import QCCDMachineModel as QCCDMachineModelCG
from bqskit.shuttling.qccd.QCCD_machine_PGS import QCCDMachineModel as QCCDMachineModelPGS
from bqskit.shuttling.qccd.QCCD_mapping import QCCDMappingAlgorithm as QCCDMappingAlgorithmCG
from bqskit.shuttling.qccd.QCCD_mapping_PGS_simple import QCCDMappingAlgorithm as QCCDMappingAlgorithmPGSSimple
from bqskit.shuttling.qccd.QCCD_mapping_PGS import QCCDMappingAlgorithm as QCCDMappingAlgorithmPGS
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

BASE_ASSIGNMENT = {
    0: 24, 1: 2, 2: 62, 3: 52, 4: 64, 5: 19, 6: 39, 7: 28, 8: 65,
    9: 40, 10: 51, 11: 26, 12: 53, 13: 66, 14: 4, 15: 14, 16: 7,
    17: 55, 18: 3, 19: 61, 20: 16, 21: 38, 22: 15, 23: 29, 24: 27,
    25: 9, 26: 8, 27: 56, 28: 18, 29: 46, 30: 10, 31: 30, 32: 54,
    33: 33, 34: 63, 35: 11, 36: 13, 37: 36, 38: 43, 39: 20, 40: 5,
    41: 21, 42: 45, 43: 58, 44: 50,
}

TEST_GATES = [
    Operation(CCXGate(), (0, 13, 16)),
    Operation(CCXGate(), (4, 11, 21)),
    Operation(CXGate(), (2, 29)),
    Operation(CCXGate(), (7, 20, 31)),
    Operation(CXGate(), (1, 35)),
    Operation(CCXGate(), (6, 18, 32)),
    Operation(CXGate(), (3, 27)),
    Operation(CCXGate(), (8, 22, 41)),
    Operation(CXGate(), (10, 43)),
    Operation(CCXGate(), (5, 17, 30)),
    Operation(CXGate(), (12, 40)),
    Operation(CCXGate(), (9, 24, 34)),
    Operation(CXGate(), (14, 38)),
    Operation(CCXGate(), (11, 23, 44)),
    Operation(CXGate(), (15, 42)),
    Operation(CCXGate(), (2, 19, 33)),
    Operation(CXGate(), (6, 37)),
    Operation(CCXGate(), (4, 20, 36)),
    Operation(CXGate(), (13, 39)),
    Operation(CCXGate(), (1, 25, 43)),
    Operation(CXGate(), (7, 34)),
    Operation(CCXGate(), (3, 18, 41)),
    Operation(CXGate(), (8, 30)),
    Operation(CCXGate(), (10, 21, 32)),
]

# Keep this at 1. The original version may take many minutes.
NUM_REPEATS = 1


def benchmark(machine_cls: type, mapping_cls: type, name: str) -> dict[str, object]:
    physical_model = create_testing_physical_machine(trap_capacity=6, type='H2')
    machine_model = machine_cls(
        physical_graph=physical_model,
        multi_qudit_gate_type='FM',
        timing_data=TIMING_DATA,
    )
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
        ion_assignment = copy.deepcopy(BASE_ASSIGNMENT)
        start = copy.deepcopy(ion_assignment)
        run_gate_summaries: list[tuple[str, int]] = []
        pgs = machine_model.build_pgs_from_assignment(ion_assignment) if name == 'PGS' else None
        t0 = perf_counter()
        total_moves = 0
        for gate in TEST_GATES:
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
        'gate_summaries': final_gate_summaries,
        'changed': final_changed,
    }


def summarize(result: dict[str, object]) -> None:
    print(result['name'])
    print(f"  graph_type: {result['graph_type']}")
    print(f"  times_s: {[round(t, 6) for t in result['times']]}")
    print(f"  avg_s: {result['avg']:.6f}")
    print(f"  total_moves: {result['total_moves']}")
    print(f"  changed_count: {len(result['changed'])}")
    print(f"  changed_sample: {sorted(result['changed'].items())[:20]}")
    print('  per_gate_move_counts:')
    for gate, move_count in result['gate_summaries']:
        print(f'    {gate}: {move_count}')


def main() -> None:
    logging.disable(logging.CRITICAL)
    buf = io.StringIO()
    with redirect_stdout(buf):
        cg_result = benchmark(QCCDMachineModelCG, QCCDMappingAlgorithmCG, 'Original')
    with redirect_stdout(buf):
        simple_result = benchmark(QCCDMachineModelPGS, QCCDMappingAlgorithmPGSSimple, 'PGS_simple')
    with redirect_stdout(buf):
        pgs_result = benchmark(QCCDMachineModelPGS, QCCDMappingAlgorithmPGS, 'PGS')

    summarize(cg_result)
    summarize(simple_result)
    summarize(pgs_result)
    print('Comparison')
    print(f"  repeats: {NUM_REPEATS}")
    print(f"  simple_speedup_x_vs_original: {cg_result['avg'] / simple_result['avg']:.3f}")
    print(f"  native_speedup_x_vs_original: {cg_result['avg'] / pgs_result['avg']:.3f}")
    print(f"  native_speedup_x_vs_simple: {simple_result['avg'] / pgs_result['avg']:.3f}")
    print(f"  same_total_moves_simple: {cg_result['total_moves'] == simple_result['total_moves']}")
    print(f"  same_total_moves_native: {cg_result['total_moves'] == pgs_result['total_moves']}")


if __name__ == '__main__':
    main()
