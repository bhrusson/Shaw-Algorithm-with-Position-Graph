from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
from time import perf_counter

from bqskit.compiler import CompilationTask
from bqskit.compiler import Compiler
from bqskit.compiler import MachineModel
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CNOTGate
from bqskit.ir.gates import HGate
from bqskit.passes import GeneralizedSabreLayoutPass
from bqskit.passes import GeneralizedSabreRoutingPass
from bqskit.passes import SetModelPass
from bqskit.qis.graph import CouplingGraph

from bqskit_local.testCasesPGS.grid32Common import GRID32_NUM_QUDITS
from bqskit_local.testCasesPGS.grid32Common import build_32x32_grid_edges
from bqskit_local.testCasesPGS.ibmEagleCommon import IBM_EAGLE_NUM_QUDITS
from bqskit_local.testCasesPGS.ibmEagleCommon import IBM_EAGLE_UNDIRECTED_COUPLING_MAP


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run native BQSKit CouplingGraph SABRE on benchmark QASM circuits.',
    )
    parser.add_argument('input_filename', help='Benchmark circuit filename without .qasm.')
    parser.add_argument(
        '--architecture',
        choices=['ibm-eagle', 'grid-32x32'],
        default='ibm-eagle',
        help='Which target architecture to compile onto.',
    )
    parser.add_argument(
        '--sabre-layout-passes',
        type=int,
        default=3,
        help='Number of layout forward/backward passes to use for SABRE.',
    )
    return parser.parse_args()


def load_circuit(input_filename: str) -> Circuit:
    circuit_path = (
        Path('bqskit/shuttling/qccd/benchmark_circuits')
        / f'{input_filename}.qasm'
    )
    return Circuit.from_file(str(circuit_path))


def select_connected_subset(
    edges: list[tuple[int, int]],
    subset_size: int,
    num_nodes: int,
) -> list[int]:
    neighbors: dict[int, set[int]] = {node: set() for node in range(num_nodes)}
    for u, v in edges:
        neighbors[int(u)].add(int(v))
        neighbors[int(v)].add(int(u))

    root = min(
        range(num_nodes),
        key=lambda node: (-len(neighbors[node]), node),
    )
    order: list[int] = []
    seen = {root}
    queue = deque([root])

    while queue and len(order) < subset_size:
        node = queue.popleft()
        order.append(node)
        for neighbor in sorted(neighbors[node]):
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)

    if len(order) != subset_size:
        raise ValueError(
            f'Unable to find connected subset of size {subset_size}.',
        )

    return order


def build_model(
    architecture: str,
    num_circuit_qudits: int,
) -> tuple[str, MachineModel, list[int]]:
    if architecture == 'ibm-eagle':
        selected_nodes = select_connected_subset(
            IBM_EAGLE_UNDIRECTED_COUPLING_MAP,
            num_circuit_qudits,
            IBM_EAGLE_NUM_QUDITS,
        )
        index_of = {node: i for i, node in enumerate(selected_nodes)}
        compact_edges = [
            (index_of[u], index_of[v])
            for u, v in IBM_EAGLE_UNDIRECTED_COUPLING_MAP
            if u in index_of and v in index_of
        ]
        coupling_graph = CouplingGraph(compact_edges, num_circuit_qudits)
        model = MachineModel(
            num_qudits=num_circuit_qudits,
            coupling_graph=coupling_graph,
            gate_set={CNOTGate(), HGate()},
        )
        return 'IBM Eagle / Washington subset', model, selected_nodes

    if architecture == 'grid-32x32':
        grid_edges = build_32x32_grid_edges()
        selected_nodes = select_connected_subset(
            grid_edges,
            num_circuit_qudits,
            GRID32_NUM_QUDITS,
        )
        index_of = {node: i for i, node in enumerate(selected_nodes)}
        compact_edges = [
            (index_of[u], index_of[v])
            for u, v in grid_edges
            if u in index_of and v in index_of
        ]
        coupling_graph = CouplingGraph(compact_edges, num_circuit_qudits)
        model = MachineModel(
            num_qudits=num_circuit_qudits,
            coupling_graph=coupling_graph,
            gate_set={CNOTGate(), HGate()},
        )
        return '32x32 grid subset', model, selected_nodes

    raise ValueError(f'Unsupported architecture: {architecture}.')


def main() -> None:
    args = parse_args()

    circuit = load_circuit(args.input_filename)
    architecture_name, model, selected_nodes = build_model(
        args.architecture,
        circuit.num_qudits,
    )

    print(f'Input filename: {args.input_filename}')
    print('Algorithm: SABRE-CG')
    print(f'Architecture: {architecture_name}')
    print(f'Architecture nodes: {selected_nodes}')
    print(f'SABRE layout passes: {args.sabre_layout_passes}')
    print(f'Number of qudits: {circuit.num_qudits}')
    print(f'Number of operations: {circuit.num_operations}')
    print(f'Architecture qudits: {model.num_qudits}')
    print(f'Number of undirected couplings: {len(model.coupling_graph)}')

    passes = [
        SetModelPass(model),
        GeneralizedSabreLayoutPass(total_passes=args.sabre_layout_passes),
        GeneralizedSabreRoutingPass(decay_delta=0.5),
    ]
    print('passes', str(passes))

    compiler = Compiler()
    task = CompilationTask(circuit, passes)
    data = task.data

    start_time = perf_counter()
    compiled = compiler.compile(circuit, passes, data=data)
    elapsed_time = perf_counter() - start_time

    print(f'Initial mapping: {data.get("initial_mapping")}')
    print(f'Final mapping: {data.get("final_mapping")}')
    print(f'Placement: {data.get("placement")}')
    print(f'Compilation runtime (s): {elapsed_time:.3f}')
    print(f'Original operation count: {circuit.num_operations}')
    print(f'Compiled operation count: {compiled.num_operations}')


if __name__ == '__main__':
    main()
