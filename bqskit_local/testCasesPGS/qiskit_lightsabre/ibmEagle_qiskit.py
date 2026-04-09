from __future__ import annotations

import argparse

from bqskit_local.testCasesPGS.ibmEagleCommon import (
    IBM_EAGLE_NUM_QUDITS,
    IBM_EAGLE_UNDIRECTED_COUPLING_MAP,
    build_named_eagle_circuit,
)
from bqskit_local.testCasesPGS.qiskit_lightsabre.common import print_result
from bqskit_local.testCasesPGS.qiskit_lightsabre.common import run_qiskit_sabre


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run Qiskit SABRE/LightSABRE on the Eagle workloads.',
    )
    parser.add_argument(
        '--workload',
        choices=['test', 'stress', 'challenge'],
        default='challenge',
        help='Which Eagle workload to compile.',
    )
    parser.add_argument(
        '--mode',
        choices=['sabre', 'lightsabre', 'both'],
        default='both',
        help='Which Qiskit mode to run.',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Seed passed into Qiskit SabreLayout.',
    )
    parser.add_argument(
        '--max-iterations',
        type=int,
        default=3,
        help='SabreLayout max_iterations value.',
    )
    parser.add_argument(
        '--layout-trials',
        type=int,
        default=None,
        help='Override Qiskit layout_trials.',
    )
    parser.add_argument(
        '--swap-trials',
        type=int,
        default=None,
        help='Override Qiskit swap_trials.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    circuit = build_named_eagle_circuit(args.workload)

    print('Architecture: IBM Eagle / Washington (Qiskit)')
    print('Workload:', args.workload)
    print('Number of qudits:', circuit.num_qudits)
    print('Number of operations:', circuit.num_operations)
    print('Number of undirected couplings:', len(IBM_EAGLE_UNDIRECTED_COUPLING_MAP))
    print('Seed:', args.seed)
    print('Max iterations:', args.max_iterations)

    modes = ['sabre', 'lightsabre'] if args.mode == 'both' else [args.mode]
    for mode in modes:
        result = run_qiskit_sabre(
            circuit,
            IBM_EAGLE_UNDIRECTED_COUPLING_MAP,
            mode=mode,
            seed=args.seed,
            max_iterations=args.max_iterations,
            layout_trials=args.layout_trials,
            swap_trials=args.swap_trials,
        )
        print_result(result)


if __name__ == '__main__':
    main()
