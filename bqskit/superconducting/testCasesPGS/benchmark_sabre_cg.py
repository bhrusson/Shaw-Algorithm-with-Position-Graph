from __future__ import annotations

import argparse
import asyncio
from collections import deque
import cProfile
from pathlib import Path
import pstats
from time import perf_counter

from bqskit.compiler import CompilationTask
from bqskit.compiler import Compiler
from bqskit.compiler import MachineModel
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CNOTGate
from bqskit.ir.gates import HGate
from bqskit.passes.mapping.layout.sabre import GeneralizedSabreLayoutPass
from bqskit.passes.mapping.routing.sabre import GeneralizedSabreRoutingPass
from bqskit.passes import SetModelPass
from bqskit.qis.graph import CouplingGraph

from bqskit.superconducting.benchmark_circuits import resolve_benchmark_circuit_path
from bqskit.superconducting.testCasesPGS.grid32Common import GRID32_NUM_QUDITS
from bqskit.superconducting.testCasesPGS.grid32Common import build_32x32_grid_edges
from bqskit.superconducting.testCasesPGS.ibmEagleCommon import IBM_EAGLE_NUM_QUDITS
from bqskit.superconducting.testCasesPGS.ibmEagleCommon import IBM_EAGLE_UNDIRECTED_COUPLING_MAP
from bqskit.superconducting.testCasesPGS.square_grid_common import (
    build_square_grid_edges,
    format_square_grid_architecture,
    parse_square_grid_architecture,
    square_grid_num_qudits,
)
from bqskit.superconducting.testCasesPGS.synthetic_multiqudit import (
    load_synthetic_multi35_circuit,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run native BQSKit CouplingGraph SABRE on benchmark QASM circuits.',
    )
    parser.add_argument('input_filename', help='Benchmark circuit filename without .qasm.')
    parser.add_argument(
        '--architecture',
        default='ibm-eagle',
        help='Target architecture, e.g. ibm-eagle, grid-8x8, or grid-12x12.',
    )
    parser.add_argument(
        '--sabre-layout-passes',
        type=int,
        default=3,
        help='Number of layout forward/backward passes to use for SABRE.',
    )
    parser.add_argument(
        '--profile-output',
        default=None,
        help='Optional path for a cProfile .prof dump of the compile section.',
    )
    return parser.parse_args()


def compile_with_optional_profile(
    compiler: Compiler,
    circuit: Circuit,
    passes: list[object],
    data: object,
    task: CompilationTask,
    profile_output: str | None,
) -> Circuit:
    """Run compilation, optionally saving cProfile output."""
    if profile_output is None:
        return compiler.compile(circuit, passes, data=data)

    profile_path = Path(profile_output)
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile = cProfile.Profile()
    try:
        profile.enable()
        return asyncio.run(task.run())
    finally:
        profile.disable()
        profile.dump_stats(str(profile_path))
        txt_path = profile_path.with_suffix(profile_path.suffix + '.txt')
        with txt_path.open('w', encoding='utf-8') as handle:
            stats = pstats.Stats(profile, stream=handle)
            stats.strip_dirs().sort_stats('cumulative').print_stats(80)


def load_circuit(input_filename: str) -> Circuit:
    synthetic = load_synthetic_multi35_circuit(input_filename)
    if synthetic is not None:
        return synthetic

    circuit_path = resolve_benchmark_circuit_path(input_filename)
    return Circuit.from_file(str(circuit_path))


def select_connected_subset(
    edges: list[tuple[int, int]],
    subset_size: int,
    num_nodes: int,
) -> list[int]:
    if subset_size > num_nodes:
        raise ValueError(
            f'Architecture has {num_nodes} qudits, but circuit requires '
            f'{subset_size}.',
        )

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

    square_grid_dims = parse_square_grid_architecture(architecture)
    if square_grid_dims is not None:
        rows, cols = square_grid_dims
        grid_edges = build_square_grid_edges(rows, cols)
        selected_nodes = select_connected_subset(
            grid_edges,
            num_circuit_qudits,
            square_grid_num_qudits(rows, cols),
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
        return format_square_grid_architecture(rows, cols), model, selected_nodes

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
        GeneralizedSabreLayoutPass(
            total_passes=args.sabre_layout_passes,
        ),
        GeneralizedSabreRoutingPass(
            decay_delta=0.5,
        ),
    ]
    print('passes', str(passes))

    task = CompilationTask(circuit, passes)
    data = task.data

    start_time = perf_counter()
    with Compiler() as compiler:
        compiled = compile_with_optional_profile(
            compiler,
            circuit,
            passes,
            data,
            task,
            args.profile_output,
        )
    elapsed_time = perf_counter() - start_time

    print(f'Initial mapping: {data.get("initial_mapping")}')
    print(f'Final mapping: {data.get("final_mapping")}')
    print(f'Placement: {data.get("placement")}')
    print(f'Compilation runtime (s): {elapsed_time:.3f}')
    if args.profile_output is not None:
        print('Profile mode: inline workflow')
        print(f'Profile output: {args.profile_output}')
        print(f'Profile summary: {args.profile_output}.txt')
    print(f'Original operation count: {circuit.num_operations}')
    print(f'Compiled operation count: {compiled.num_operations}')

if __name__ == '__main__':
    main()
