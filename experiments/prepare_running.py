from bqskit.compiler import Compiler, MachineModel
from bqskit.ir import Circuit
from bqskit.ir.opt import ScipyMinimizer, HilbertSchmidtCostGenerator
from bqskit.passes import *
from bqskit.qis import UnitaryMatrix, CouplingGraph
from pytket.phir.qtm_machine import QtmMachine, QTM_MACHINES_MAP
from bqskit.ir.gates import RZZGate, RZGate, U1qPi2Gate, U1qPiGate, PermutationGate
from bqskit.shuttling import ShuttlingLayerGenerator, HeuristicSearch, ShuttlingEmbedAllPermutationsPass, \
    GateZoneSelectionPass
from bqskit.shuttling.util import get_duration_from_circ, check_executable_circuit
from bqskit.compiler.passdata import PassData

from bqskit import enable_logging
from experiments.circuit_generator import circuit_generate

enable_logging(True)

qtm_machine = QtmMachine.H1_3
machine = QTM_MACHINES_MAP.get(qtm_machine)
machine_model = MachineModel(machine.size, CouplingGraph.linear(machine.size),
                             {RZGate(), U1qPi2Gate, U1qPiGate, RZZGate()})

tofolli_gate = UnitaryMatrix([
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 1, 0],
])

fredkin_gate = UnitaryMatrix([
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1],
])

target = circuit_generate("QuantumVolume", 3, 3, True)
# target = tofolli_gate
# print("Quantum Volume unitary: ", target)
circuit = Circuit.from_unitary(target)

sq_synthesis = QSearchSynthesisPass(
    layer_generator=SingleQuditLayerGenerator(None, allow_repeats=True),
    heuristic_function=DijkstraHeuristic(),
    instantiate_options={
        'method': 'minimization',
        'minimizer': ScipyMinimizer(),
        'cost_fn_gen': HilbertSchmidtCostGenerator(),
    },
)


def estimated_runtime(circ: Circuit) -> float:
    """Return estimated runtime of the circuit with the given machine."""
    return get_duration_from_circ(circ, qtm_machine)


# qsearch_pass = LEAPSynthesisPass(layer_generator=ShuttlingLayerGenerator(),
#                                  heuristic_function=HeuristicSearch(heuristic_factor=1, qtm_machine=qtm_machine))
workflow = [
    #SetModelPass(machine_model),
    QSearchSynthesisPass(),
    # PermutationAwareSynthesisPass(inner_synthesis=qsearch_pass, scoring_fn=estimated_runtime),
    # GroupSingleQuditGatePass(),
    # ForEachBlockPass(
    #     sq_synthesis
    # ),
    # UnfoldPass(),
    #ScanningGateRemovalPass()
]

# block_size = 3
# workflow = [
#     SetModelPass(machine_model),
#     SubtopologySelectionPass(block_size),
#     GateZoneSelectionPass(block_size),
#     QuickPartitioner(block_size),
#     ForEachBlockPass(
#         ShuttlingEmbedAllPermutationsPass(inner_synthesis=qsearch_pass)
#     )
# ]
