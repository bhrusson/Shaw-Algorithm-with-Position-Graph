from __future__ import annotations

import argparse
import logging
from time import perf_counter

from bqskit.compiler import CompilationTask, Compiler, MachineModel
from bqskit.ir.gates import CNOTGate, HGate
from bqskit.passes import GeneralizedSabreLayoutPass, GeneralizedSabreRoutingPass, SetModelPass
from bqskit.qis.graph import CouplingGraph

from bqskit_local.testCasesPGS.ibmEagleCommon import (
    IBM_EAGLE_COUPLING_MAP,
    IBM_EAGLE_NUM_QUDITS,
    IBM_EAGLE_UNDIRECTED_COUPLING_MAP,
    build_named_eagle_circuit,
)

_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run the IBM Eagle CouplingGraph SABRE workflow.',
    )
    parser.add_argument(
        '--workload',
        choices=['test', 'stress', 'challenge'],
        default='challenge',
        help='Which Eagle workload to compile.',
    )
    parser.add_argument(
        '--sabre-layout-passes',
        type=int,
        default=3,
        help='Number of layout forward/backward passes to use for SABRE.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    circ = build_named_eagle_circuit(args.workload)
    cg = CouplingGraph(IBM_EAGLE_UNDIRECTED_COUPLING_MAP, IBM_EAGLE_NUM_QUDITS)
    model = MachineModel(
        num_qudits=IBM_EAGLE_NUM_QUDITS,
        coupling_graph=cg,
        gate_set={CNOTGate(), HGate()},
    )

    print("Architecture: IBM Eagle / Washington")
    print("Workload:", args.workload)
    print("Number of qudits:", circ.num_qudits)
    print("Number of operations:", circ.num_operations)
    print("Number of undirected couplings:", len(IBM_EAGLE_UNDIRECTED_COUPLING_MAP))

    passes = [
        SetModelPass(model),
        GeneralizedSabreLayoutPass(total_passes=args.sabre_layout_passes),
        GeneralizedSabreRoutingPass(decay_delta=0.5),
    ]
    print("passes", str(passes))

    compiler = Compiler()
    task = CompilationTask(circ, passes)
    data = task.data

    _logger.info("Driver data before compile: initial_mapping=%s", data.get("initial_mapping"))
    _logger.info("Driver data before compile: final_mapping=%s", data.get("final_mapping"))
    _logger.info("Driver data before compile: placement=%s", data.get("placement"))

    start_time = perf_counter()
    compiled = compiler.compile(circ, passes, data=data)
    elapsed_time = perf_counter() - start_time

    _logger.info("Driver data after compile: initial_mapping=%s", data.get("initial_mapping"))
    _logger.info("Driver data after compile: final_mapping=%s", data.get("final_mapping"))
    _logger.info("Driver data after compile: placement=%s", data.get("placement"))

    print("Compilation runtime (s):", f"{elapsed_time:.3f}")
    print("Original operation count:", circ.num_operations)
    print("Compiled operation count:", compiled.num_operations)


if __name__ == '__main__':
    main()
