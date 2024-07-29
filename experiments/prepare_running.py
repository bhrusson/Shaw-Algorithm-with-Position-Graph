from bqskit.compiler import Compiler, MachineModel
from bqskit.ir import Circuit
from bqskit.ir.opt import ScipyMinimizer, HilbertSchmidtCostGenerator
from bqskit.passes import *
from bqskit.qis import CouplingGraph
from pytket.phir.qtm_machine import QtmMachine, QTM_MACHINES_MAP
from bqskit.ir.gates import RZZGate, RZGate, U1qPi2Gate, U1qPiGate
from bqskit.shuttling import ShuttlingLayerGenerator, HeuristicSearch, ShuttlingEmbedAllPermutationsPass, \
    GateZoneSelectionPass, OddEvenSchedulingPass, ReplacementPass, ZoneSchedulerPass
from bqskit.shuttling.mapping.layout.pam import PAMLayoutPass
from bqskit.shuttling.mapping.routing.pam import PAMRoutingPass
from bqskit.shuttling.util import check_executable_circuit, get_duration_from_circ
from bqskit import enable_logging

enable_logging(True)

qtm_machine = QtmMachine.H1
machine = QTM_MACHINES_MAP.get(qtm_machine)
machine_model = MachineModel(machine.size, CouplingGraph.linear(machine.size),
                             {RZGate(),
                              U1qPi2Gate, U1qPiGate, RZZGate()})

num_qudits = 8
circuit_type = "PhaseEstimator"
cir = Circuit.from_file(f"experiments/results/experiment_circuits"
                        f"/input_circuits/{circuit_type}_{num_qudits}.qasm")

target_unitary = cir
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


qsearch_pass = QSearchSynthesisPass(layer_generator=ShuttlingLayerGenerator(),
                                    heuristic_function=HeuristicSearch(heuristic_factor=10, qtm_machine=qtm_machine))
block_size = 3
num_layout_passes = 3
workflow = [
    UnfoldPass(),
    SetModelPass(machine_model),
    SubtopologySelectionPass(block_size),
    GateZoneSelectionPass(block_size),
    QuickPartitioner(block_size),
    ForEachBlockPass(
        ShuttlingEmbedAllPermutationsPass(inner_synthesis=qsearch_pass,
                                          qtm_machine=QtmMachine.H1)
    ),
    ApplyPlacement(),
    PAMLayoutPass(num_layout_passes),
    PAMRoutingPass(0.1),
    ApplyPlacement(),
    UnfoldPass(),
    ZoneSchedulerPass(),
    ReplacementPass(),
    GroupSingleQuditGatePass(),
    ForEachBlockPass(
        sq_synthesis
    ),
    UnfoldPass()
    #OddEvenSchedulingPass()
]
