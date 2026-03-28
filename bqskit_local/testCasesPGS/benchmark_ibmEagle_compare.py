from __future__ import annotations

import io
import logging
import statistics
from contextlib import redirect_stdout
from time import perf_counter

from bqskit.compiler import CompilationTask, Compiler, MachineModel
from bqskit.ir.gates import CNOTGate, HGate
from bqskit.passes import (
    GeneralizedSabreLayoutPass,
    GeneralizedSabreRoutingPass,
    SetModelPass,
)
from bqskit.qis.graph import CouplingGraph

from bqskit_local.layout.sabrePassPGS import GeneralizedSabreLayoutPassPGS
from bqskit_local.mapping.setPGSPass import SetPGSPass
from bqskit_local.position.state import PositionGraphState
from bqskit_local.routing.sabreRoutingPGS import GeneralizedSabreRoutingPassPGS
from bqskit_local.testCasesPGS.ibmEagleCommon import (
    IBM_EAGLE_NUM_QUDITS,
    IBM_EAGLE_UNDIRECTED_COUPLING_MAP,
    build_eagle_position_graph,
    build_eagle_test_circuit,
)


def run_cg() -> dict[str, object]:
    circ = build_eagle_test_circuit()
    cg = CouplingGraph(IBM_EAGLE_UNDIRECTED_COUPLING_MAP, IBM_EAGLE_NUM_QUDITS)
    model = MachineModel(
        num_qudits=IBM_EAGLE_NUM_QUDITS,
        coupling_graph=cg,
        gate_set={CNOTGate(), HGate()},
    )
    passes = [
        SetModelPass(model),
        GeneralizedSabreLayoutPass(total_passes=3),
        GeneralizedSabreRoutingPass(decay_delta=0.5),
    ]
    task = CompilationTask(circ, passes)
    data = task.data
    with Compiler(num_workers=1) as compiler:
        t0 = perf_counter()
        compiled = compiler.compile(circ, passes, data=data)
        elapsed = perf_counter() - t0
    return {
        'time_s': elapsed,
        'ops': compiled.num_operations,
        'cycles': compiled.num_cycles,
        'final_mapping': list(data.get('final_mapping', [])),
        'head_ops': [str(op) for i, op in enumerate(compiled) if i < 20],
    }


def run_pgs() -> dict[str, object]:
    circ = build_eagle_test_circuit()
    pg = build_eagle_position_graph()
    template_pgs = PositionGraphState(pg, radices=[2] * IBM_EAGLE_NUM_QUDITS)
    passes = [
        SetPGSPass(template_pgs, placement=list(range(IBM_EAGLE_NUM_QUDITS))),
        GeneralizedSabreLayoutPassPGS(template_pgs, total_passes=3),
        GeneralizedSabreRoutingPassPGS(template_pgs, decay_delta=0.5),
    ]
    task = CompilationTask(circ, passes)
    data = task.data
    with Compiler(num_workers=1) as compiler:
        t0 = perf_counter()
        compiled = compiler.compile(circ, passes, data=data)
        elapsed = perf_counter() - t0
    return {
        'time_s': elapsed,
        'ops': compiled.num_operations,
        'cycles': compiled.num_cycles,
        'final_mapping': list(data.get('final_mapping', [])),
        'head_ops': [str(op) for i, op in enumerate(compiled) if i < 20],
    }


def summarize(name: str, result: dict[str, object]) -> None:
    print(name)
    print(f"  time_s: {result['time_s']:.6f}")
    print(f"  ops: {result['ops']}")
    print(f"  cycles: {result['cycles']}")
    print(f"  final_mapping_head: {result['final_mapping'][:20]}")
    print('  first_ops:')
    for op in result['head_ops']:
        print(f'    {op}')


def main() -> None:
    logging.disable(logging.CRITICAL)
    buf = io.StringIO()
    with redirect_stdout(buf):
        cg_result = run_cg()
    with redirect_stdout(buf):
        pgs_result = run_pgs()

    summarize('CG', cg_result)
    summarize('PGS', pgs_result)
    print('Comparison')
    print(f"  delta_time_s: {pgs_result['time_s'] - cg_result['time_s']:.6f}")
    print(f"  same_ops: {cg_result['ops'] == pgs_result['ops']}")
    print(f"  same_cycles: {cg_result['cycles'] == pgs_result['cycles']}")
    print(f"  same_first_ops: {cg_result['head_ops'] == pgs_result['head_ops']}")
    print(
        '  same_final_mapping_head: '
        f"{cg_result['final_mapping'][:20] == pgs_result['final_mapping'][:20]}"
    )


if __name__ == '__main__':
    main()
