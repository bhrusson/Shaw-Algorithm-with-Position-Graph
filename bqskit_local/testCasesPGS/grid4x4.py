from __future__ import annotations

import argparse
import logging
from time import perf_counter

from bqskit.compiler import CompilationTask, Compiler
from bqskit.ir.gates.constant.swap import SwapGate

from bqskit_local.layout.cached_lightSABREPassPGS import (
    GeneralizedCachedLightSABRELayoutPassPGS,
)
from bqskit_local.layout.cached_sabrePassPGS import (
    GeneralizedCachedSabreLayoutPassPGS,
)
from bqskit_local.layout.lightSABREPassPGS import GeneralizedLightSABRELayoutPassPGS
from bqskit_local.layout.sabrePassPGS import GeneralizedSabreLayoutPassPGS
from bqskit_local.mapping.lightSABRE_pgs import DEFAULT_LIGHTSABRE_HEURISTIC
from bqskit_local.mapping.setPGSPass import SetPGSPass
from bqskit_local.position.state import PositionGraphState
from bqskit_local.routing.cached_lightSABRERoutingPGS import (
    GeneralizedCachedLightSABRERoutingPassPGS,
)
from bqskit_local.routing.cached_sabreRoutingPGS import (
    GeneralizedCachedSabreRoutingPassPGS,
)
from bqskit_local.routing.lightSABRERoutingPGS import GeneralizedLightSABRERoutingPassPGS
from bqskit_local.routing.sabreRoutingPGS import GeneralizedSabreRoutingPassPGS
from bqskit_local.testCasesPGS.grid4Common import (
    GRID4_NUM_QUDITS,
    build_4x4_challenge_circuit,
    build_4x4_position_graph,
)

_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def count_swaps(circuit) -> int:
    return sum(1 for op in circuit if isinstance(op.gate, SwapGate))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run the 4x4 grid PositionGraph SABRE workflow.',
    )
    parser.add_argument(
        '--rounds',
        type=int,
        default=2,
        help='Number of challenge-circuit rounds.',
    )
    parser.add_argument(
        '--cg-compat',
        action='store_true',
        help='Match CouplingGraph SABRE decisions exactly.',
    )
    parser.add_argument(
        '--algorithm',
        choices=['sabre', 'sabre-cached', 'lightsabre', 'lightsabre-cached'],
        default='lightsabre',
        help='Which PGS mapping algorithm to run.',
    )
    parser.add_argument(
        '--heuristic',
        default=DEFAULT_LIGHTSABRE_HEURISTIC,
        help='LightSABRE heuristic components, e.g. decay or lookahead+decay+depth.',
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
        '--sabre-layout-passes',
        type=int,
        default=3,
        help='Number of layout forward/backward passes to use for SABRE.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    effective_cg_compat = args.cg_compat

    circ = build_4x4_challenge_circuit(rounds=args.rounds)
    pg = build_4x4_position_graph()
    template_pgs = PositionGraphState(pg, radices=[2] * GRID4_NUM_QUDITS)

    print('Architecture: 4x4 grid PositionGraph')
    print('Challenge rounds:', args.rounds)
    print('Algorithm:', args.algorithm)
    print('CG compatibility mode:', effective_cg_compat)
    print('Number of qudits:', circ.num_qudits)
    print('Number of operations:', circ.num_operations)
    print('Number of positions:', len(pg.position_labels))
    print('Number of directed edges:', len(pg.edge_labels))
    print('Move neighbors of 0:', pg.get_swap_neighbors(0))
    print('Distance 0 -> 15:', pg.distance(0, GRID4_NUM_QUDITS - 1))

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
        passes = [
            SetPGSPass(template_pgs, placement=list(range(GRID4_NUM_QUDITS))),
            layout_pass_cls(
                template_pgs,
                max_iterations=args.max_iterations,
                layout_trials=args.layout_trials,
                swap_trials=args.routing_trials,
                heuristic=args.heuristic,
                seed=args.seed,
                cg_compatibility_mode=effective_cg_compat,
            ),
            routing_pass_cls(
                template_pgs,
                heuristic=args.heuristic,
                seed=args.seed,
                trials=args.routing_trials,
                cg_compatibility_mode=effective_cg_compat,
            ),
        ]
    elif args.algorithm == 'sabre':
        passes = [
            SetPGSPass(template_pgs, placement=list(range(GRID4_NUM_QUDITS))),
            GeneralizedSabreLayoutPassPGS(
                template_pgs,
                total_passes=args.sabre_layout_passes,
                cg_compatibility_mode=effective_cg_compat,
            ),
            GeneralizedSabreRoutingPassPGS(
                template_pgs,
                decay_delta=0.5,
                cg_compatibility_mode=effective_cg_compat,
            ),
        ]
    else:
        passes = [
            SetPGSPass(template_pgs, placement=list(range(GRID4_NUM_QUDITS))),
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

    compiler = Compiler()
    task = CompilationTask(circ, passes)
    data = task.data

    start_time = perf_counter()
    compiled = compiler.compile(circ, passes, data=data)
    elapsed_time = perf_counter() - start_time

    print('Compilation runtime (s):', f'{elapsed_time:.3f}')
    print('Original operation count:', circ.num_operations)
    print('Compiled operation count:', compiled.num_operations)
    print('Inserted swap count:', count_swaps(compiled))
    print('Circuit depth:', compiled.num_cycles)


if __name__ == '__main__':
    main()
