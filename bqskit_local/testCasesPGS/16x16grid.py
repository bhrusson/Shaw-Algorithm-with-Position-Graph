from __future__ import annotations

import argparse
import logging
from time import perf_counter

from bqskit.compiler import CompilationTask, Compiler

from bqskit_local.layout.lightSABREPassPGS import GeneralizedLightSABRELayoutPassPGS
from bqskit_local.layout.sabrePassPGS import GeneralizedSabreLayoutPassPGS
from bqskit_local.mapping.setPGSPass import SetPGSPass
from bqskit_local.position.state import PositionGraphState
from bqskit_local.routing.lightSABRERoutingPGS import GeneralizedLightSABRERoutingPassPGS
from bqskit_local.routing.sabreRoutingPGS import GeneralizedSabreRoutingPassPGS
from bqskit_local.testCasesPGS.grid16Common import (
    GRID16_NUM_QUDITS,
    build_16x16_challenge_circuit,
    build_16x16_position_graph,
)

_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run the 16x16 grid PositionGraph SABRE workflow.',
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
        choices=['sabre', 'lightsabre'],
        default='sabre',
        help='Which PGS mapping algorithm to run.',
    )
    parser.add_argument(
        '--heuristic',
        default='decay',
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
    effective_cg_compat = args.cg_compat or args.algorithm == 'sabre'

    circ = build_16x16_challenge_circuit(rounds=args.rounds)
    pg = build_16x16_position_graph()
    template_pgs = PositionGraphState(pg, radices=[2] * GRID16_NUM_QUDITS)

    print("Architecture: 16x16 grid PositionGraph")
    print("Challenge rounds:", args.rounds)
    print("Algorithm:", args.algorithm)
    print("CG compatibility mode:", effective_cg_compat)
    print("Number of qudits:", circ.num_qudits)
    print("Number of operations:", circ.num_operations)
    print("Number of positions:", len(pg.position_labels))
    print("Number of directed edges:", len(pg.edge_labels))
    print("Move neighbors of 0:", pg.get_swap_neighbors(0))
    print("Distance 0 -> 255:", pg.distance(0, GRID16_NUM_QUDITS - 1))

    if args.algorithm == 'lightsabre':
        passes = [
            SetPGSPass(template_pgs, placement=list(range(GRID16_NUM_QUDITS))),
            GeneralizedLightSABRELayoutPassPGS(
                template_pgs,
                max_iterations=args.max_iterations,
                layout_trials=args.layout_trials,
                swap_trials=args.routing_trials,
                heuristic=args.heuristic,
                seed=args.seed,
                cg_compatibility_mode=effective_cg_compat,
            ),
            GeneralizedLightSABRERoutingPassPGS(
                template_pgs,
                heuristic=args.heuristic,
                seed=args.seed,
                trials=args.routing_trials,
                decay_delta=0.5,
                cg_compatibility_mode=effective_cg_compat,
            ),
        ]
    else:
        passes = [
            SetPGSPass(template_pgs, placement=list(range(GRID16_NUM_QUDITS))),
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
