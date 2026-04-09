from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
from time import perf_counter

from bqskit.compiler import CompilationTask
from bqskit.compiler import Compiler
from bqskit.ir.circuit import Circuit

from bqskit_local.layout.sabrePassPGS import GeneralizedSabreLayoutPassPGS
from bqskit_local.mapping.setPGSPass import SetPGSPass
from bqskit_local.position.graph import EdgeLabel
from bqskit_local.position.graph import PositionGraph
from bqskit_local.position.state import PositionGraphState
from bqskit_local.routing.sabreRoutingPGS import GeneralizedSabreRoutingPassPGS
from bqskit_local.testCasesPGS.grid16Common import GRID16_NUM_QUDITS
from bqskit_local.testCasesPGS.grid16Common import build_16x16_grid_edges
from bqskit_local.testCasesPGS.grid16Common import build_16x16_position_graph
from bqskit_local.testCasesPGS.grid32Common import GRID32_NUM_QUDITS
from bqskit_local.testCasesPGS.grid32Common import build_32x32_grid_edges
from bqskit_local.testCasesPGS.grid32Common import build_32x32_position_graph
from bqskit_local.testCasesPGS.grid8Common import GRID8_NUM_QUDITS
from bqskit_local.testCasesPGS.grid8Common import build_8x8_grid_edges
from bqskit_local.testCasesPGS.grid8Common import build_8x8_position_graph
from bqskit_local.testCasesPGS.ibmEagleCommon import IBM_EAGLE_NUM_QUDITS
from bqskit_local.testCasesPGS.ibmEagleCommon import IBM_EAGLE_UNDIRECTED_COUPLING_MAP
from bqskit_local.testCasesPGS.ibmEagleCommon import build_eagle_position_graph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run local PositionGraph SABRE on benchmark QASM circuits.',
    )
    parser.add_argument('input_filename', help='Benchmark circuit filename without .qasm.')
    parser.add_argument(
        '--architecture',
        choices=['ibm-eagle', 'grid-32x32', 'grid-16x16', 'grid8x8'],
        default='ibm-eagle',
        help='Which target architecture to compile onto.',
    )
    parser.add_argument(
        '--cg-compat',
        action='store_true',
        help='Match CouplingGraph SABRE tie-breaking behavior.',
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


def build_template_pgs(
    architecture: str,
    num_circuit_qudits: int,
) -> tuple[str, PositionGraphState, list[int]]:
    def build_subset_template(
        base_graph: PositionGraph,
        edges: list[tuple[int, int]],
        num_nodes: int,
        architecture_name: str,
    ) -> tuple[str, PositionGraphState, list[int]]:
        selected_nodes = select_connected_subset(
            edges,
            num_circuit_qudits,
            num_nodes,
        )
        index_of = {node: i for i, node in enumerate(selected_nodes)}
        compact_pos_labels = [
            base_graph.position_labels[node]
            for node in selected_nodes
        ]
        compact_edge_labels: dict[tuple[int, int], EdgeLabel] = {
            (index_of[u], index_of[v]): label
            for (u, v), label in base_graph.edge_labels.items()
            if u in index_of and v in index_of
        }
        graph = PositionGraph(compact_pos_labels, compact_edge_labels)
        pgs = PositionGraphState(graph, radices=[2] * num_circuit_qudits)
        return architecture_name, pgs, selected_nodes

    if architecture == 'ibm-eagle':
        return build_subset_template(
            build_eagle_position_graph(),
            IBM_EAGLE_UNDIRECTED_COUPLING_MAP,
            IBM_EAGLE_NUM_QUDITS,
            'IBM Eagle / Washington PositionGraph subset',
        )

    if architecture == 'grid-32x32':
        return build_subset_template(
            build_32x32_position_graph(),
            build_32x32_grid_edges(),
            GRID32_NUM_QUDITS,
            '32x32 grid PositionGraph subset',
        )

    if architecture == 'grid-16x16':
        return build_subset_template(
            build_16x16_position_graph(),
            build_16x16_grid_edges(),
            GRID16_NUM_QUDITS,
            '16x16 grid PositionGraph subset',
        )

    if architecture == 'grid8x8':
        return build_subset_template(
            build_8x8_position_graph(),
            build_8x8_grid_edges(),
            GRID8_NUM_QUDITS,
            '8x8 grid PositionGraph subset',
        )

    raise ValueError(f'Unsupported architecture: {architecture}.')


def main() -> None:
    args = parse_args()

    circuit = load_circuit(args.input_filename)
    architecture_name, template_pgs, selected_nodes = build_template_pgs(
        args.architecture,
        circuit.num_qudits,
    )

    print(f'Input filename: {args.input_filename}')
    print('Algorithm: SABRE-PGS')
    print(f'Architecture: {architecture_name}')
    print(f'Architecture nodes: {selected_nodes}')
    print(f'CG compatibility mode: {args.cg_compat}')
    print(f'SABRE layout passes: {args.sabre_layout_passes}')
    print(f'Number of qudits: {circuit.num_qudits}')
    print(f'Number of operations: {circuit.num_operations}')
    print(f'Number of positions: {template_pgs.num_pos}')
    print(f'Number of directed edges: {len(template_pgs.position_graph.edge_labels)}')

    passes = [
        SetPGSPass(template_pgs, placement=list(range(circuit.num_qudits))),
        GeneralizedSabreLayoutPassPGS(
            template_pgs,
            total_passes=args.sabre_layout_passes,
            cg_compatibility_mode=args.cg_compat,
        ),
        GeneralizedSabreRoutingPassPGS(
            template_pgs,
            decay_delta=0.5,
            cg_compatibility_mode=args.cg_compat,
        ),
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
