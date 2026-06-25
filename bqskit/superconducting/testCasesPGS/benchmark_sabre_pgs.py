from __future__ import annotations

import argparse
import asyncio
from collections import deque
import cProfile
import json
from pathlib import Path
import pstats
from time import perf_counter

from bqskit.compiler import CompilationTask
from bqskit.compiler import Compiler
from bqskit.ir.circuit import Circuit

from bqskit_local.benchmark_circuits import resolve_benchmark_circuit_path
from bqskit_local.layout.cached_lightSABREPassPGS import (
    GeneralizedCachedLightSABRELayoutPassPGS,
)
from bqskit_local.layout.cached_sabrePassPGS import (
    GeneralizedCachedSabreLayoutPassPGS,
)
from bqskit_local.layout.lightSABREPassPGS import GeneralizedLightSABRELayoutPassPGS
from bqskit_local.layout.sabrePassPGS import GeneralizedSabreLayoutPassPGS
from bqskit_local.mapping.lightSABRE_pgs import DEFAULT_LIGHTSABRE_HEURISTIC
from bqskit_local.mapping.heuristic_stats import combine_heuristic_stats
from bqskit_local.mapping.heuristic_stats import summarize_heuristic_stats
from bqskit_local.mapping.setPGSPass import SetPGSPass
from bqskit_local.position.graph import EdgeLabel
from bqskit_local.position.graph import PositionGraph
from bqskit_local.position.state import PositionGraphState
from bqskit_local.routing.cached_lightSABRERoutingPGS import (
    GeneralizedCachedLightSABRERoutingPassPGS,
)
from bqskit_local.routing.cached_sabreRoutingPGS import (
    GeneralizedCachedSabreRoutingPassPGS,
)
from bqskit_local.routing.lightSABRERoutingPGS import GeneralizedLightSABRERoutingPassPGS
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
from bqskit_local.testCasesPGS.square_grid_common import (
    build_square_grid_edges,
    build_square_grid_position_graph,
    format_square_grid_architecture,
    parse_square_grid_architecture,
    square_grid_num_qudits,
)
from bqskit_local.testCasesPGS.synthetic_multiqudit import (
    load_synthetic_multi35_circuit,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run local PositionGraph SABRE or LightSABRE on benchmark QASM circuits.',
    )
    parser.add_argument('input_filename', help='Benchmark circuit filename without .qasm.')
    parser.add_argument(
        '--architecture',
        default='ibm-eagle',
        help='Target architecture, e.g. ibm-eagle, grid-8x8, or grid-12x12.',
    )
    parser.add_argument(
        '--cg-compat',
        action='store_true',
        help='Match CouplingGraph SABRE tie-breaking behavior.',
    )
    parser.add_argument(
        '--algorithm',
        choices=['sabre', 'sabre-cached', 'lightsabre', 'lightsabre-cached'],
        default='lightsabre',
        help='Which local PGS mapping algorithm to run.',
    )
    parser.add_argument(
        '--heuristic',
        default=DEFAULT_LIGHTSABRE_HEURISTIC,
        help='LightSABRE heuristic components, e.g. lookahead+decay.',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Seed for LightSABRE trial tie-breaking.',
    )
    parser.add_argument(
        '--layout-trials',
        type=int,
        default=5,
        help='LightSABRE layout trial count.',
    )
    parser.add_argument(
        '--routing-trials',
        type=int,
        default=4,
        help='LightSABRE routing trial count.',
    )
    parser.add_argument(
        '--max-iterations',
        type=int,
        default=3,
        help='LightSABRE bidirectional layout iterations.',
    )
    parser.add_argument(
        '--attempt-limit',
        type=int,
        default=None,
        help='Optional local-minimum threshold override for LightSABRE.',
    )
    parser.add_argument(
        '--disable-traffic-aware-seed',
        '--disable-circuit-aware-seed',
        dest='disable_traffic_aware_seed',
        action='store_true',
        help='Disable the additional traffic-aware LightSABRE layout seed variants.',
    )
    parser.add_argument(
        '--sabre-layout-passes',
        type=int,
        default=3,
        help='Number of layout forward/backward passes to use for SABRE.',
    )
    parser.add_argument(
        '--layout-only',
        action='store_true',
        help='Only generate and score layout candidates; skip routing.',
    )
    parser.add_argument(
        '--track-heuristic-stats',
        action='store_true',
        help='Print SABRE frontier/extended-set heuristic statistics as JSON.',
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

    square_grid_dims = parse_square_grid_architecture(architecture)
    if square_grid_dims is not None:
        rows, cols = square_grid_dims
        return build_subset_template(
            build_square_grid_position_graph(rows, cols),
            build_square_grid_edges(rows, cols),
            square_grid_num_qudits(rows, cols),
            format_square_grid_architecture(rows, cols),
        )

    raise ValueError(f'Unsupported architecture: {architecture}.')


def main() -> None:
    args = parse_args()
    effective_cg_compat = args.cg_compat

    circuit = load_circuit(args.input_filename)
    architecture_name, template_pgs, selected_nodes = build_template_pgs(
        args.architecture,
        circuit.num_qudits,
    )

    print(f'Input filename: {args.input_filename}')
    print(f'Algorithm: {args.algorithm}')
    print(f'Architecture: {architecture_name}')
    print(f'Architecture nodes: {selected_nodes}')
    print(f'CG compatibility mode: {effective_cg_compat}')
    if args.algorithm in ('lightsabre', 'lightsabre-cached'):
        print(f'LightSABRE heuristic: {args.heuristic}')
        print(f'LightSABRE seed: {args.seed}')
        print(f'LightSABRE layout trials: {args.layout_trials}')
        print(f'LightSABRE routing trials: {args.routing_trials}')
        print(f'LightSABRE max iterations: {args.max_iterations}')
        print(f'LightSABRE attempt limit: {args.attempt_limit}')
        print(
            'LightSABRE traffic-aware seed: '
            f'{not args.disable_traffic_aware_seed}',
        )
    elif args.algorithm == 'sabre':
        print(f'SABRE layout passes: {args.sabre_layout_passes}')
    else:
        print(f'Cached SABRE layout passes: {args.sabre_layout_passes}')
    print(f'Number of qudits: {circuit.num_qudits}')
    print(f'Number of operations: {circuit.num_operations}')
    print(f'Number of positions: {template_pgs.num_pos}')
    print(f'Number of directed edges: {len(template_pgs.position_graph.edge_labels)}')

    if args.algorithm in ('lightsabre', 'lightsabre-cached'):
        layout_pass_cls = (
            GeneralizedCachedLightSABRELayoutPassPGS
            if args.algorithm == 'lightsabre-cached'
            else GeneralizedLightSABRELayoutPassPGS
        )
        routing_pass_cls = (
            GeneralizedCachedLightSABRERoutingPassPGS
            if args.algorithm == 'lightsabre-cached'
            else GeneralizedLightSABRERoutingPassPGS
        )
        layout_pass = layout_pass_cls(
            template_pgs,
            max_iterations=args.max_iterations,
            layout_trials=args.layout_trials,
            swap_trials=args.routing_trials,
            heuristic=args.heuristic,
            seed=args.seed,
            attempt_limit=args.attempt_limit,
            use_traffic_aware_seed=not args.disable_traffic_aware_seed,
            cg_compatibility_mode=effective_cg_compat,
        )
        routing_pass = routing_pass_cls(
            template_pgs,
            heuristic=args.heuristic,
            seed=args.seed,
            trials=args.routing_trials,
            attempt_limit=args.attempt_limit,
            cg_compatibility_mode=effective_cg_compat,
        )
        passes = [
            SetPGSPass(template_pgs, placement=list(range(circuit.num_qudits))),
            layout_pass,
            routing_pass,
        ]
    elif args.algorithm == 'sabre':
        passes = [
            SetPGSPass(template_pgs, placement=list(range(circuit.num_qudits))),
            GeneralizedSabreLayoutPassPGS(
                template_pgs,
                total_passes=args.sabre_layout_passes,
                cg_compatibility_mode=effective_cg_compat,
                collect_heuristic_stats=args.track_heuristic_stats,
            ),
            GeneralizedSabreRoutingPassPGS(
                template_pgs,
                decay_delta=0.5,
                cg_compatibility_mode=effective_cg_compat,
                collect_heuristic_stats=args.track_heuristic_stats,
            ),
        ]
    else:
        passes = [
            SetPGSPass(template_pgs, placement=list(range(circuit.num_qudits))),
            GeneralizedCachedSabreLayoutPassPGS(
                template_pgs,
                total_passes=args.sabre_layout_passes,
                cg_compatibility_mode=effective_cg_compat,
            ),
            GeneralizedCachedSabreRoutingPassPGS(
                template_pgs,
                decay_delta=0.5,
                cg_compatibility_mode=effective_cg_compat,
            ),
        ]
    for sabre_pass in passes:
        if hasattr(sabre_pass, 'collect_heuristic_stats'):
            sabre_pass.collect_heuristic_stats = args.track_heuristic_stats
    print('passes', str(passes))

    if args.layout_only:
        if args.algorithm not in ('lightsabre', 'lightsabre-cached'):
            raise ValueError('--layout-only is only supported for LightSABRE variants.')

        start_mapping = list(range(circuit.num_qudits))
        candidate_results: list[tuple[float, float, float, str, list[int]]] = []
        for layout_trial_index, candidate in enumerate(
            layout_pass._candidate_start_mappings(
                circuit,
                start_mapping,
                circuit.num_qudits,
                None,
            ),
        ):
            pgs = layout_pass._build_pgs_from_mapping(candidate.mapping, circuit.num_qudits)
            layout_pass.begin_trial(layout_trial_index)
            for _ in range(layout_pass.max_iterations):
                layout_pass.forward_pass(circuit, pgs, modify_circuit=False)
                layout_pass.backward_pass(circuit, pgs)

            perm = [int(x) for x in pgs.logical_to_position[:circuit.num_qudits]]
            score, front_score, extend_score = layout_pass.layout_score(circuit, perm)
            candidate_results.append(
                (score, front_score, extend_score, candidate.label, perm),
            )

        candidate_results.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
        print('LightSABRE layout-only candidate scores:')
        for score, front_score, extend_score, label, _ in candidate_results:
            print(
                f'  {label}: layout_score={score:.6f}, '
                f'front={front_score:.6f}, extend={extend_score:.6f}',
            )
        if candidate_results:
            best_score, best_front, best_extend, best_label, best_layout = candidate_results[0]
            print(f'Best layout label: {best_label}')
            print(f'Best layout score: {best_score:.6f}')
            print(f'Best layout front score: {best_front:.6f}')
            print(f'Best layout extend score: {best_extend:.6f}')
            print(f'Best layout mapping: {best_layout}')
        return

    compiler = Compiler()
    task = CompilationTask(circuit, passes)
    data = task.data

    start_time = perf_counter()
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
    if args.algorithm in ('lightsabre', 'lightsabre-cached'):
        candidate_labels = data.get('lightsabre_layout_candidate_labels', [])
        candidate_scores = data.get('lightsabre_layout_candidate_scores', [])
        candidate_detail_scores = data.get('lightsabre_layout_candidate_detail_scores', [])
        if candidate_labels and candidate_scores and candidate_detail_scores:
            print('LightSABRE layout candidate scores:')
            for label, score, detail in zip(
                candidate_labels,
                candidate_scores,
                candidate_detail_scores,
            ):
                print(f'  {label}: selection_score={score}, detail_score={tuple(detail)}')
    print(f'Compilation runtime (s): {elapsed_time:.3f}')
    if args.profile_output is not None:
        print('Profile mode: inline workflow')
        print(f'Profile output: {args.profile_output}')
        print(f'Profile summary: {args.profile_output}.txt')
    print(f'Original operation count: {circuit.num_operations}')
    print(f'Compiled operation count: {compiled.num_operations}')
    if args.track_heuristic_stats:
        layout_stats = summarize_heuristic_stats(
            data.get('sabre_layout_heuristic_stats'),
        )
        routing_stats = summarize_heuristic_stats(
            data.get('sabre_routing_heuristic_stats'),
        )
        total_stats = combine_heuristic_stats(layout_stats, routing_stats)
        print(
            'Heuristic stats JSON: '
            + json.dumps(
                {
                    'layout': layout_stats,
                    'routing': routing_stats,
                    'total': total_stats,
                },
                sort_keys=True,
            ),
        )


if __name__ == '__main__':
    main()
