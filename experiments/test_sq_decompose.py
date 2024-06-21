# import bqskit
# import numpy as np
# from bqskit.ir.opt import ScipyMinimizer, HilbertSchmidtCostGenerator
# from bqskit.compiler import Compiler, MachineModel
# from bqskit.qis import CouplingGraph
# from bqskit.ir import Circuit
# from bqskit.ir.gates import *
# from bqskit.passes import *
# from bqskit import enable_logging
# from pytket.phir.qtm_machine import QtmMachine, QTM_MACHINES_MAP
# from bqskit.shuttling import ShuttlingLayerGenerator, HeuristicSearch, \
#     GateZoneSelectionPass, OddEvenSchedulingPass
#
# enable_logging(True)
# qtm_machine = QtmMachine.H1_1
# machine = QTM_MACHINES_MAP.get(qtm_machine)
# machine_model = MachineModel(2, CouplingGraph.linear(2),
#                              {RZGate(),
#                               U1qPi2Gate, U1qPiGate, RZZGate()})
#
# sq_synthesis = QSearchSynthesisPass(
#     layer_generator=SingleQuditLayerGenerator(None, allow_repeats=True),
#     heuristic_function=DijkstraHeuristic(),
#     instantiate_options={
#         'method': 'minimization',
#         'minimizer': ScipyMinimizer(),
#         'cost_fn_gen': HilbertSchmidtCostGenerator(),
#     }
# )
#
# circuit = Circuit(4).from_unitary(SwapGate().get_unitary())
#
# qsearch_pass = QSearchSynthesisPass(layer_generator=ShuttlingLayerGenerator(),
#                                     heuristic_function=HeuristicSearch(heuristic_factor=1, qtm_machine=qtm_machine))
#
# workflow = [SetModelPass(machine_model),
#             qsearch_pass,
#             GroupSingleQuditGatePass(),
#             ForEachBlockPass(
#                 sq_synthesis
#             ),
#             UnfoldPass(),]
#
# with Compiler() as compiler:
#     output_circuit, data = compiler.compile(circuit, workflow, request_data=True)
#
# print(output_circuit.to('qasm'))
# output_circuit.save(f"experiments/results/experiment_circuits/test_circuits/swap.qasm")
# print(f"Distance {output_circuit.get_unitary().get_distance_from(SwapGate().get_unitary())}")

# from bqskit.ir.gates import ControlledGate, U1Gate
# g = ControlledGate(U1Gate())
# from bqskit import Circuit
#
# c = Circuit(2)
# c.append_gate(g, (0, 1))
# print(c.to('qasm'))
a = {1, 2, 3}
for i in a.copy():
    if i == 2:
        a.add(4)
    print(i)
