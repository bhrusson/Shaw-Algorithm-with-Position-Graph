import pickle

from bqskit.passes import *
from bqskit.ir import Circuit
from bqskit.qis.graph import CouplingGraph
from bqskit.compiler import Compiler
from bqskit.shuttling.qccd.QCCD_machine import (QCCDMachineModel)
from bqskit.shuttling.qccd.QCCD_util import create_testing_physical_machine
from bqskit.shuttling.qccd.QCCD_layergen import QCCDLayerGenerator
from bqskit.shuttling.qccd.QCCD_heuristic_search import QCCDHeuristicFunction
from bqskit import enable_logging

enable_logging(True)
# physical_model = create_testing_physical_machine()
# timing_data = {'sq_timings': 30e-6,
#                'tq_timings': 40e-6,
#                'segment': 5e-6,
#                'inner_swap': 42e-6,
#                'split': 80e-6,
#                'merge': 80e-6,
#                'junction_Y': 100e-6,
#                'junction_X': 120e-6}
# machine_model = QCCDMachineModel(physical_graph=physical_model,
#                                  timing_data=timing_data,
#                                  multi_qudit_gate_type="FM")
# ion_assignment = {0: 0, 1: 2, 2: 6}
#
# qsearch_pass = QSearchSynthesisPass(layer_generator=QCCDLayerGenerator(),
#                                     heuristic_function=AStarHeuristic(heuristic_factor=10.0))
# block_size = 3
# workflow = [
#     UnfoldPass(),
#     qsearch_pass
# ]
# test_circuit = 'toffoli.qasm'
# input_filename = "experiments/results/experiment_circuits/input_circuits/" + test_circuit
# cir = Circuit.from_file(input_filename)
# with Compiler() as compiler:
#     output_circuit, data = compiler.compile(cir, workflow, request_data=True)
#
# """
# Save qasm file
# """
# qasm_result_filename = "experiments/results/experiment_circuits/QCCD_output_circuits/" + test_circuit
# output_circuit.save(qasm_result_filename)
# cg = CouplingGraph({(9, 10), (13, 14), (10, 11), (0, 4), (11, 16), (12, 13), (14, 16), (5, 13), (2, 3), (6, 7), (12, 14), (14, 15), (15, 16), (2, 12), (0, 5), (8, 15), (1, 3), (7, 8)})
# physical_location = [1, 2, 0]
# local_graph = cg.get_subgraph(physical_location)
# print(local_graph)
result_filename = "bqskit/shuttling/qccd/new_result/QCCDSim_QAOA_16_compiled_G2x3_4_Greedy.pkl"
with open(result_filename, 'rb') as f:
    data = pickle.load(f)
print(data)
